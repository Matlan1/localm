# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The per-namespace memory store: persistence + retrieval + forgetting.

One store == one ``(principal, agent, scope_key)`` namespace, backed by a single
JSONL file (one record per line) under ``<home>/memory/<agent>/<ns>.jsonl`` with
an OPTIONAL aligned vector sidecar ``<ns>.vec.json`` (``{"dim", "vectors": {id:
vec}}``). Keying vectors by record id (not position) keeps them correct across
edits and deletes.

Design mirrors ``localm/rag/store.py`` (home-scale JSON, atomic tmp+replace, BM25
always-available, embeddings OPTIONAL). It deliberately stays SMALL: consolidation
+ decay + the ``N_MAX`` cap keep a namespace to a few hundred distilled records,
never a transcript, so whole-file rewrites are cheap.

Retrieval blends the Generative-Agents signals (Park et al. 2023): relevance
(lexical BM25, optionally 50/50 with embedding cosine when an embedder is present),
recency (exponential decay since last use), and importance (write-time salience).
This store imports no session/audit state; the privacy gate lives with the caller
(see ``gating.writes_allowed`` and the ``reinforce`` flag on ``recall``).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

from localm.rag.bm25 import BM25
from localm.rag.store import _cosine

from .record import MemoryRecord

EmbedFn = Callable[[list[str]], list[list[float]]]

# ---- bounds (DoS / poisoning / bloat) ------------------------------------- #
MAX_TEXT_LEN = 500          # per-record text cap (truncate on store)
N_MAX = 256                 # records per namespace; prune evicts the weakest
K_CAP = 32                  # hard ceiling on a recall k

# ---- retrieval scoring ---------------------------------------------------- #
W_REL, W_REC, W_IMP = 0.5, 0.3, 0.2     # blend weights (sum 1.0)
TAU_DAYS = 30.0             # recency e-folding time: 30d -> 0.37, 90d -> 0.05
FLOOR = 0.05               # a recalled memory must beat this normalized score
TINY_CORPUS = 8            # below this, BM25 idf is noisy -> rank by rec+imp only
VEC_COVERAGE = 0.8         # blend cosine only when >= this fraction have vectors

# ---- forgetting ----------------------------------------------------------- #
PRUNE_FLOOR = 0.02         # decayed(importance*recency) below this is forgettable
_DAY = 86400.0

_AGENTS = ("chat", "coder")


def _memory_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    from localm.config import home_dir
    return home_dir() / "memory"


def namespace_hash(principal: str, agent: str, scope_key: str) -> str:
    """Stable 16-hex namespace id. UTF-8 encoded before hashing so a crafted
    unicode principal cannot alias another; agent + scope_key are joined with a
    delimiter that record ids never contain."""
    raw = "|".join((principal or "owner", agent, scope_key or "")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def namespace_file(principal: str, agent: str, scope_key: str,
                   root: Optional[Path] = None) -> Path:
    if agent not in _AGENTS:
        raise ValueError(f"unknown memory agent {agent!r}; expected one of {_AGENTS}")
    base = _memory_root(root).resolve()
    path = (base / agent / f"{namespace_hash(principal, agent, scope_key)}.jsonl")
    # Path-safety: the resolved file must stay under <home>/memory. agent is
    # allow-listed and the ns is a hex hash, so this is belt-and-suspenders
    # against any future caller that lets a component flow in unchecked.
    rp = path.resolve()
    if not (rp == base or _within(rp, base)):
        raise ValueError("refusing a memory path outside the memory directory")
    return path


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _maxnorm(scores: list[float]) -> list[float]:
    top = max(scores) if scores else 0.0
    return [s / top for s in scores] if top > 0 else [0.0 for _ in scores]


class MemoryStore:
    """JSONL-backed store for one ``(principal, agent, scope_key)`` namespace."""

    def __init__(self, principal: str, agent: str, scope_key: str = "", *,
                 root: Optional[Path] = None) -> None:
        self.principal = principal or "owner"
        self.agent = agent
        self.scope_key = scope_key or ""
        self._file = namespace_file(self.principal, agent, self.scope_key, root=root)
        self._records: list[MemoryRecord] = []
        self._vectors: dict = {}        # id -> vector (present only when embedded)
        self._dim: Optional[int] = None
        self._bm25: Optional[BM25] = None
        self._load()

    # ----------------------------------------------------------------- IO -- #
    @property
    def path(self) -> Path:
        return self._file

    def _vec_file(self) -> Path:
        return self._file.with_suffix(".vec.json")

    def _load(self) -> None:
        self._records = []
        if self._file.is_file():
            for line in self._file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(MemoryRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue      # a partial/corrupt line must not break recall
        self._vectors = {}
        self._dim = None
        vf = self._vec_file()
        if vf.is_file():
            try:
                data = json.loads(vf.read_text(encoding="utf-8"))
                vecs = data.get("vectors", {})
                ids = {r.id for r in self._records}
                if isinstance(vecs, dict):
                    self._vectors = {k: v for k, v in vecs.items() if k in ids and v}
                    self._dim = data.get("dim") or _first_dim(self._vectors)
            except (json.JSONDecodeError, OSError, ValueError):
                self._vectors = {}
                self._dim = None
        self._bm25 = None

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False)
                         for r in self._records)
        self._atomic_write(self._file, body + ("\n" if body else ""))
        vf = self._vec_file()
        if self._vectors:
            self._atomic_write(
                vf, json.dumps({"dim": self._dim, "vectors": self._vectors}))
        else:
            vf.unlink(missing_ok=True)
        self._bm25 = None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------- basics -- #
    def all(self) -> list[MemoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, mem_id: str) -> Optional[MemoryRecord]:
        return next((r for r in self._records if r.id == mem_id), None)

    def _embed_one(self, text: str, embed_fn: Optional[EmbedFn]) -> Optional[list]:
        """Embed *text*, honouring the single-dimensionality invariant. A vector
        of a different dim than the store's (a switched embedding model) is
        dropped, not stored, so cosine never mixes dims (best-effort, never
        raises - memory writes must not crash on an embedder hiccup)."""
        if embed_fn is None:
            return None
        try:
            vec = embed_fn([text])[0]
        except Exception:
            return None
        if not vec:
            return None
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            return None
        return list(vec)

    def add(self, record: MemoryRecord, *, embed_fn: Optional[EmbedFn] = None,
            save: bool = True) -> MemoryRecord:
        record.text = record.text[:MAX_TEXT_LEN]
        self._records.append(record)
        vec = self._embed_one(record.text, embed_fn)
        if vec is not None:
            self._vectors[record.id] = vec
        if save:
            self._save()
        return record

    def update(self, mem_id: str, *, embed_fn: Optional[EmbedFn] = None,
               save: bool = True, **fields) -> Optional[MemoryRecord]:
        rec = self.get(mem_id)
        if rec is None:
            return None
        text_changed = False
        for key, val in fields.items():
            if key == "text" and isinstance(val, str):
                val = val.strip()[:MAX_TEXT_LEN]
                text_changed = val != rec.text
                rec.text = val
            elif key == "importance":
                from .record import _clamp01
                rec.importance = _clamp01(val)
            elif key in ("last_used", "created", "uses", "kind", "source", "meta"):
                setattr(rec, key, val)
        rec.updated = time.time()
        if text_changed and embed_fn is not None:
            vec = self._embed_one(rec.text, embed_fn)
            if vec is not None:
                self._vectors[rec.id] = vec
            else:
                self._vectors.pop(rec.id, None)
        if save:
            self._save()
        return rec

    def delete(self, mem_id: str, *, save: bool = True) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r.id != mem_id]
        self._vectors.pop(mem_id, None)
        removed = len(self._records) != before
        if removed and save:
            self._save()
        return removed

    def clear(self) -> None:
        self._records = []
        self._vectors = {}
        self._dim = None
        self._save()

    def replace(self, records: list[MemoryRecord], *,
                embed_fn: Optional[EmbedFn] = None) -> None:
        """Overwrite the whole namespace in ONE atomic save (used by the
        consolidation batch and prune, so a crash leaves the pre-change store
        intact - never a half-consolidated state)."""
        keep_ids = {r.id for r in records}
        self._vectors = {k: v for k, v in self._vectors.items() if k in keep_ids}
        self._records = []
        for r in records:
            r.text = (r.text or "").strip()[:MAX_TEXT_LEN]
            self._records.append(r)
            if embed_fn is not None and r.id not in self._vectors:
                vec = self._embed_one(r.text, embed_fn)
                if vec is not None:
                    self._vectors[r.id] = vec
        self._save()

    # --------------------------------------------------------- retrieval -- #
    def _relevance(self, query: str,
                   embed_fn: Optional[EmbedFn]) -> list[float]:
        n = len(self._records)
        if n < TINY_CORPUS:
            # BM25 idf is unstable on a handful of records; skip the relevance
            # signal and let recency+importance rank. (Transparent to callers.)
            return [0.0] * n
        if self._bm25 is None:
            self._bm25 = BM25([r.text for r in self._records])
        rel = _maxnorm(self._bm25.scores(query))
        vec_rel = self._vector_relevance(query, embed_fn)
        if vec_rel is not None:
            rel = [0.5 * a + 0.5 * b for a, b in zip(rel, vec_rel)]
        return rel

    def _vector_relevance(self, query: str,
                          embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        if embed_fn is None or not self._vectors:
            return None
        n = len(self._records)
        if n == 0 or len(self._vectors) / n < VEC_COVERAGE:
            return None
        dims = {len(v) for v in self._vectors.values()}
        if len(dims) != 1:
            return None
        stored_dim = next(iter(dims))
        try:
            qvec = embed_fn([query])[0]
        except Exception:
            return None
        if not qvec or len(qvec) != stored_dim:
            return None
        out = [
            _cosine(qvec, self._vectors[r.id]) if r.id in self._vectors else 0.0
            for r in self._records
        ]
        return _maxnorm(out)

    def recall(self, query: str, *, k: int = 6, embed_fn: Optional[EmbedFn] = None,
               reinforce: bool = False, now: Optional[float] = None) -> list[MemoryRecord]:
        """Top-*k* records for *query* by relevance+recency+importance.

        ``reinforce=True`` bumps last_used/uses on the returned records (a WRITE):
        the caller passes ``reinforce=gating.writes_allowed(surface)`` so privacy
        mode recalls WITHOUT any side effect. Deterministic (stable tie-break)."""
        if not (query or "").strip() or not self._records:
            return []
        k = max(1, min(int(k), K_CAP))
        now = time.time() if now is None else now
        rel = self._relevance(query, embed_fn)
        # Recency is RAW exponential decay (already in [0,1]), NOT max-normalised:
        # max-normalising would make the newest record always score 1.0, so recall
        # could never fall silent on a store of only stale/irrelevant memories.
        # Raw decay (Generative-Agents style) lets an old, unimportant, off-topic
        # memory score below the floor and be dropped.
        rec = [
            math.exp(-((now - r.last_used) / _DAY) / TAU_DAYS)
            for r in self._records
        ]
        scored = []
        for i, r in enumerate(self._records):
            score = W_REL * rel[i] + W_REC * rec[i] + W_IMP * r.importance
            scored.append((score, i, r))
        # Sort by score desc; ties keep insertion order (index asc) for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits = [(s, r) for s, i, r in scored if s > FLOOR][:k]
        results = [r for _s, r in hits]
        if reinforce and results:
            for r in results:
                r.last_used = now
                r.uses += 1
            self._save()
        return results

    # --------------------------------------------------------- forgetting - #
    def _decayed(self, rec: MemoryRecord, now: float) -> float:
        recency = math.exp(-((now - rec.last_used) / _DAY) / TAU_DAYS)
        # Reinforcement lifts effective importance a little (frequently-used
        # memories resist decay), capped at 1.0.
        eff_imp = min(1.0, rec.importance + 0.05 * rec.uses)
        return eff_imp * recency

    def prune(self, *, now: Optional[float] = None, n_max: int = N_MAX) -> int:
        """Forget decayed, low-value memories and enforce the size cap. User- and
        import-sourced records are never auto-dropped by decay (only the size cap
        may evict them, weakest first); synth memories below ``PRUNE_FLOOR`` are
        forgotten. Returns the number removed."""
        now = time.time() if now is None else now
        kept = [
            r for r in self._records
            if r.source in ("user", "import") or self._decayed(r, now) >= PRUNE_FLOOR
        ]
        if len(kept) > n_max:
            kept.sort(key=lambda r: self._decayed(r, now), reverse=True)
            kept = kept[:n_max]
        removed = len(self._records) - len(kept)
        if removed:
            self.replace(kept)
        return removed


def _first_dim(vectors: dict) -> Optional[int]:
    for v in vectors.values():
        if v:
            return len(v)
    return None
