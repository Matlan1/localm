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
    def _expand(paths: list) -> list[Path]:
        """Resolve files + recursive folder contents to indexable files."""
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
        return [p for p in out if not (p in seen or seen.add(p))]

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  on_progress: Optional[ProgressFn] = None) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size) are skipped;
        changed ones are re-indexed in place. Returns counters plus per-file
        failures. embed_fn failures degrade to lexical-only, never abort.
        """
        say = on_progress or (lambda _t: None)
        files = self._expand(paths)
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
                    if len(vecs) == len(new_chunks):
                        vectors = vecs
                except Exception as e:
                    embed_broken = True
                    say(f"embeddings unavailable ({e}) - indexing lexical-only")

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
        covered = sum(1 for v in self._vectors if v)
        if not self._chunks or covered / len(self._chunks) < 0.8:
            return None
        try:
            qvec = embed_fn([text])[0]
        except Exception:
            return None
        # A switched embedding model yields query vectors of a different
        # dimensionality than the stored ones: numpy (va @ vb) would raise and
        # the no-numpy path would silently truncate. Skip vector scoring and
        # fall back to lexical-only rather than crash or return wrong scores.
        dim = self._vec_dim or _first_dim(self._vectors)
        if dim is not None and len(qvec) != dim:
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
    if len(a) != len(b):       # mismatched dims (switched embed model) -> no signal
        return 0.0
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
