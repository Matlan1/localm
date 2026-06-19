# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Persistent document collections ("knowledge bases").

Layout - one directory per collection under ``<data dir>/rag/``:

    rag/<name>/meta.json      {"name", "created", "docs": {path: {mtime, size, chunks}}}
    rag/<name>/chunks.jsonl   one chunk per line: {"source", "pos", "text"}
    rag/<name>/vectors.json   optional: {"dim", "vectors": [[...]|null, ...]}
                              aligned with chunks.jsonl line order

Collections are explicit user data (like generated images): indexing writes
to disk in every session mode. Rewrites are whole-file + atomic rename -
corpora here are home-scale (thousands of chunks, not millions).

Retrieval is hybrid: BM25 always; when vectors exist for (almost) all chunks
and the caller can embed the query, scores become an equal blend of
max-normalised BM25 and cosine similarity.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Optional

from .bm25 import BM25
from .chunk import chunk_text
from .extract import EXTRACTABLE_SUFFIXES, ExtractError, extract_text

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Windows reserved device names: they match _NAME_RE but mkdir raises on them.
_RESERVED_NAMES = {"con", "prn", "aux", "nul",
                   *(f"com{i}" for i in range(1, 10)),
                   *(f"lpt{i}" for i in range(1, 10))}


def _first_dim(vectors: list) -> Optional[int]:
    """Dimensionality of the first non-empty vector, or None."""
    for v in vectors:
        if v:
            return len(v)
    return None

# Directories never worth indexing when a folder is added
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
              ".pytest_cache", ".mypy_cache", "dist", "build", ".idea",
              ".vscode"}

EmbedFn = Callable[[list[str]], list[list[float]]]
ProgressFn = Callable[[str], None]


def rag_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "rag"


# Well-known credential / secret folders under the user's home that must never
# be indexed even though they sit inside an allowed root. The localm data dir
# (home_dir(), holding the API keystore + registry) is denied separately.
_SENSITIVE_HOME_SUBDIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".localm",
)
# Lower-cased for matching a path component ANYWHERE in a resolved path, so a
# nested ~/proj/.ssh is caught too, and case-insensitively so a ".SSH"
# component cannot slip past on a case-insensitive filesystem.
_SENSITIVE_NAMES = frozenset(s.lower() for s in _SENSITIVE_HOME_SUBDIRS)


def _path_within(child: Path, parent: Path) -> bool:
    """True when *child* is *parent* or lives underneath it (both resolved)."""
    try:
        child, parent = child.resolve(), parent.resolve()
    except (OSError, ValueError):
        return False
    if child == parent:
        return True
    if hasattr(child, "is_relative_to"):
        return child.is_relative_to(parent)
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def indexing_roots(extra: Optional[list] = None) -> list[Path]:
    """Default-allowed roots for API-driven indexing.

    The user's home folder and the working directory cover legitimate document
    libraries while excluding system paths (``C:/Windows``, ``/etc``). Power
    users can widen this with the ``rag_indexing_roots`` config key. The localm
    data dir and credential folders are still denied by ``confine_index_path``
    even though they sit under home.
    """
    roots: list[Path] = [Path.home(), Path.cwd()]
    try:
        from localm.config import load_config
        for r in load_config().get("rag_indexing_roots", []) or []:
            roots.append(Path(r).expanduser())
    except Exception:
        pass
    for r in (extra or []):
        roots.append(Path(r).expanduser())
    out: list[Path] = []
    for r in roots:
        try:
            out.append(r.resolve())
        except (OSError, ValueError):
            continue
    return out


def confine_index_path(p, allowed_roots: Optional[list] = None) -> Path:
    """Resolve *p* and verify it is safe for API-driven indexing.

    Always rejects the localm data directory (it holds the API key, registry and
    config) and well-known credential folders. When *allowed_roots* is given,
    also require the path to live under one of them, so an API caller - a browser
    page on the loopback port, or a remote client - cannot read arbitrary files
    like ``C:/Windows/win.ini`` or ``/etc/passwd`` and serve them back (C2).
    Raises ``ValueError`` on an out-of-bounds path. CLI callers pass no roots and
    are not confined (a local user can already read their own files).
    """
    from localm.config import home_dir
    try:
        rp = Path(p).expanduser().resolve()
    except (OSError, ValueError):
        raise ValueError(f"Invalid path: {p}")

    if _path_within(rp, home_dir()):
        raise ValueError(
            f"Refusing to index the localm data directory "
            f"(it holds the API key and registry): {p}")
    # Credential folders are denied wherever they appear in the resolved path,
    # not only at the home root: ~/proj/.ssh and <cwd>/sub/.aws are as sensitive
    # as ~/.ssh. rp is already resolved, so this also catches a symlink that
    # points into a credential dir. Tradeoff: a folder literally named one of
    # these (a real ./.docker you wanted to index) is refused too - acceptable
    # for a credential denylist. The real confinement is allowed_roots (plus the
    # same-origin guard on the route); this is defense in depth.
    if any(part.lower() in _SENSITIVE_NAMES for part in rp.parts):
        raise ValueError(f"Refusing to index a credential directory: {p}")

    if allowed_roots is not None and not any(
            _path_within(rp, r) for r in allowed_roots):
        raise ValueError(
            f"Path is outside the allowed indexing roots "
            f"(your home folder / the working directory): {p}")
    return rp


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            "Collection names must be 1-64 letters, digits, '-' or '_'")
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(f"'{name}' is a reserved device name and cannot be used")
    return name


def collection_names(base: Optional[Path] = None) -> list[str]:
    base = base or rag_dir()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / "meta.json").is_file())


def delete_collection(name: str, base: Optional[Path] = None) -> bool:
    import shutil
    base = base or rag_dir()
    path = base / _check_name(name)
    if not (path / "meta.json").is_file():
        return False
    shutil.rmtree(path)
    return True


class Collection:
    def __init__(self, name: str, base: Optional[Path] = None) -> None:
        self.name = _check_name(name)
        self.dir = (base or rag_dir()) / self.name
        self._meta: dict = {}
        self._chunks: list[dict] = []
        self._vectors: Optional[list] = None     # aligned with _chunks, or None
        self._vec_dim: Optional[int] = None       # dimensionality of stored vectors
        self._bm25: Optional[BM25] = None
        self.corrupt: bool = False
        if self.exists():
            self._load()

    # ------------------------------------------------------------- #
    #  Lifecycle / IO                                                #
    # ------------------------------------------------------------- #

    def exists(self) -> bool:
        return (self.dir / "meta.json").is_file()

    def create(self) -> "Collection":
        if self.exists():
            return self
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta = {"name": self.name, "created": time.time(), "docs": {}}
        self._save()
        return self

    def _load(self) -> None:
        # A corrupt meta.json must not crash construction (and thus the whole
        # collections listing) - flag it and present an empty collection.
        try:
            meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("meta.json is not an object")
            self._meta = meta
        except (json.JSONDecodeError, ValueError, OSError):
            self.corrupt = True
            self._meta = {"name": self.name, "docs": {}}
            self._chunks = []
            self._vectors = None
            self._vec_dim = None
            self._bm25 = None
            return
        self._chunks = []
        chunks_file = self.dir / "chunks.jsonl"
        if chunks_file.is_file():
            for line in chunks_file.read_text(encoding="utf-8").splitlines():
                try:
                    self._chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._vectors = None
        self._vec_dim = None
        vec_file = self.dir / "vectors.json"
        if vec_file.is_file():
            try:
                data = json.loads(vec_file.read_text(encoding="utf-8"))
                vectors = data.get("vectors", [])
                if len(vectors) == len(self._chunks):
                    self._vectors = vectors
                    self._vec_dim = data.get("dim") or _first_dim(vectors)
            except (json.JSONDecodeError, OSError):
                pass
        self._bm25 = None

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))
        self._atomic_write("chunks.jsonl", "\n".join(
            json.dumps(c, ensure_ascii=False) for c in self._chunks))
        if self._vectors is not None and any(v for v in self._vectors):
            self._vec_dim = _first_dim(self._vectors)
            self._atomic_write("vectors.json", json.dumps(
                {"dim": self._vec_dim, "vectors": self._vectors}))
        else:
            (self.dir / "vectors.json").unlink(missing_ok=True)
            self._vec_dim = None
        self._bm25 = None

    def _atomic_write(self, filename: str, content: str) -> None:
        tmp = self.dir / (filename + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.dir / filename)

    # ------------------------------------------------------------- #
    #  Indexing                                                      #
    # ------------------------------------------------------------- #

    @staticmethod
    def _expand(paths: list,
                allowed_roots: Optional[list] = None) -> list[Path]:
        """Resolve files + recursive folder contents to indexable files.

        When *allowed_roots* is given, files resolving outside the confinement
        (system paths, credential dirs, or symlinks escaping an allowed folder)
        are silently dropped - add_paths already validated the top-level inputs,
        so this only catches sneaky nested escapes."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_file():
                out.append(p.resolve())
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if (f.is_file()
                            and f.suffix.lower() in EXTRACTABLE_SUFFIXES
                            and not any(part in _SKIP_DIRS for part in f.parts)):
                        out.append(f.resolve())
        # de-dup, keep order
        seen: set = set()
        deduped = [p for p in out if not (p in seen or seen.add(p))]
        if allowed_roots is None:
            return deduped
        kept: list[Path] = []
        for p in deduped:
            try:
                confine_index_path(p, allowed_roots)
            except ValueError:
                continue   # nested escape (symlink / credential dir) -> skip
            kept.append(p)
        return kept

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  on_progress: Optional[ProgressFn] = None,
                  allowed_roots: Optional[list] = None) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size) are skipped;
        changed ones are re-indexed in place. Returns counters plus per-file
        failures. embed_fn failures degrade to lexical-only, never abort.

        When *allowed_roots* is given (the HTTP API passes the user's home + the
        working dir), an out-of-bounds top-level path raises ``ValueError`` and
        nested escapes are dropped (C2). CLI callers omit it and stay unconfined.
        Indexing with an embedding model whose dimensionality differs from the
        collection's also raises ``ValueError`` rather than corrupting the
        vectors with mixed dimensions (C3).
        """
        say = on_progress or (lambda _t: None)
        if allowed_roots is not None:
            for p in paths:
                confine_index_path(p, allowed_roots)   # raises ValueError
        files = self._expand(paths, allowed_roots)
        if not files:
            return {"added": 0, "updated": 0, "skipped": 0, "failed": [],
                    "chunks": len(self._chunks)}

        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None

        for f in files:
            key = str(f)
            try:
                stat = f.stat()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            known = self._meta["docs"].get(key)
            if known and known.get("mtime") == stat.st_mtime \
                    and known.get("size") == stat.st_size:
                skipped += 1
                continue
            try:
                text = extract_text(f)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {f.name}: {e}")
                continue
            new_chunks = chunk_text(text)
            for c in new_chunks:
                c["source"] = key

            vectors: list = [None] * len(new_chunks)
            if not embed_broken and new_chunks:
                try:
                    vecs = embed_fn([c["text"] for c in new_chunks])
                except Exception as e:
                    embed_broken = True
                    say(f"embeddings unavailable ({e}) - indexing lexical-only")
                else:
                    if len(vecs) == len(new_chunks):
                        new_dim = _first_dim(vecs)
                        # A different embedding dimensionality means a different
                        # model: refuse rather than store mixed-dim vectors that
                        # would silently mis-score every query (C3).
                        if (self._vec_dim is not None and new_dim is not None
                                and new_dim != self._vec_dim):
                            raise ValueError(
                                f"Embedding dimension changed "
                                f"({self._vec_dim} -> {new_dim}): this "
                                f"collection was built with a different "
                                f"embedding model. Rebuild it (delete and "
                                f"re-add) or index with the original model.")
                        vectors = vecs
                        if self._vec_dim is None and new_dim is not None:
                            self._vec_dim = new_dim

            # Replace any previous chunks (and vectors) for this document
            if known:
                keep = [i for i, c in enumerate(self._chunks)
                        if c.get("source") != key]
                self._chunks = [self._chunks[i] for i in keep]
                if self._vectors is not None:
                    self._vectors = [self._vectors[i] for i in keep]
                updated += 1
            else:
                added += 1
            if self._vectors is None:
                self._vectors = [None] * len(self._chunks)
            self._chunks.extend(new_chunks)
            self._vectors.extend(vectors)
            self._meta["docs"][key] = {
                "mtime": stat.st_mtime, "size": stat.st_size,
                "chunks": len(new_chunks),
            }
            say(f"indexed {f.name} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def remove_doc(self, source: str) -> bool:
        if source not in self._meta.get("docs", {}):
            return False
        keep = [i for i, c in enumerate(self._chunks)
                if c.get("source") != source]
        self._chunks = [self._chunks[i] for i in keep]
        if self._vectors is not None:
            self._vectors = [self._vectors[i] for i in keep]
        del self._meta["docs"][source]
        self._save()
        return True

    # ------------------------------------------------------------- #
    #  Retrieval                                                     #
    # ------------------------------------------------------------- #

    def query(self, text: str, k: int = 4,
              embed_fn: Optional[EmbedFn] = None) -> list[dict]:
        """Top-*k* chunks for *text*: max-normalised BM25, blended 50/50 with
        cosine similarity when vectors cover the corpus and the query can be
        embedded."""
        if not text.strip() or not self._chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25([c["text"] for c in self._chunks])
        scores = self._bm25.scores(text)
        top = max(scores) if scores else 0.0
        if top > 0:
            scores = [s / top for s in scores]

        vec_scores = self._vector_scores(text, embed_fn)
        if vec_scores is not None:
            scores = [0.5 * lex + 0.5 * vec
                      for lex, vec in zip(scores, vec_scores)]

        order = sorted(range(len(scores)), key=lambda i: scores[i],
                       reverse=True)[:max(1, k)]
        return [
            {**self._chunks[i], "score": round(scores[i], 4)}
            for i in order if scores[i] > 0
        ]

    def _vector_scores(self, text: str,
                       embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        if embed_fn is None or self._vectors is None:
            return None
        present = [v for v in self._vectors if v]
        if not self._chunks or len(present) / len(self._chunks) < 0.8:
            return None
        # Stored vectors must share one dimensionality. A legacy collection with
        # mixed-dim vectors (built before the C3 add-time guard) is ambiguous -
        # skip vector scoring and answer lexically rather than mis-score with
        # zeros for the odd-dim chunks.
        dims = {len(v) for v in present}
        if len(dims) > 1:
            return None
        stored_dim = next(iter(dims))
        try:
            qvec = embed_fn([text])[0]
        except Exception:
            return None
        # A switched embedding model yields query vectors of a different
        # dimensionality than the stored ones: fall back to lexical-only rather
        # than crash or return wrong scores.
        if len(qvec) != stored_dim:
            return None
        out = []
        for v in self._vectors:
            out.append(_cosine(qvec, v) if v else 0.0)
        # normalise to [0, 1] like the lexical side
        top = max(out, default=0.0)
        return [s / top for s in out] if top > 0 else out

    # ------------------------------------------------------------- #
    #  Introspection                                                 #
    # ------------------------------------------------------------- #

    def stats(self) -> dict:
        vectors = self._vectors or []
        return {
            "name": self.name,
            "created": self._meta.get("created"),
            "n_docs": len(self._meta.get("docs", {})),
            "n_chunks": len(self._chunks),
            "has_vectors": bool(vectors) and all(v for v in vectors),
            "corrupt": self.corrupt,
        }

    def docs(self) -> list[dict]:
        return [
            {"path": path, **info}
            for path, info in sorted(self._meta.get("docs", {}).items())
        ]


def _cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        # Mismatched dims must never be scored as a real (zero) similarity - that
        # silently mis-ranks. Callers (_vector_scores) guarantee equal lengths;
        # reaching here is a bug or unrepaired mixed-dim data (C3).
        raise ValueError(
            f"cosine similarity needs equal-length vectors "
            f"(got {len(a)} and {len(b)})")
    try:
        import numpy as np
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(va @ vb) / denom if denom else 0.0
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
