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
import hashlib
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from localm.debuglog import logger as _log
from .bm25 import BM25
from .chunk import chunk_text
from .extract import (BLACKLISTED_SUFFIXES, ExtractError, classify_format,
                      extract_bytes, extract_text, is_secret_index_name)

ClassifyFn = Callable[[str], Optional[str]]
DescribeImageFn = Callable[[bytes, str], Optional[str]]

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


def _well_formed_vectors(vectors) -> bool:
    """Cheap (O(n)) structural check that *vectors* is what ``_save`` writes: a
    list whose entries are each a null placeholder (a missing embedding) or a
    list/tuple. A hand-corrupted or truncated vectors.json can hold scalars or
    strings that pass JSON parsing but crash cosine scoring at QUERY time with an
    opaque error (``object of type 'int' has no len()``); catch that at load and
    degrade to BM25 with a reason instead of shipping a delayed crash."""
    return isinstance(vectors, list) and all(
        (not v) or isinstance(v, (list, tuple)) for v in vectors)

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


class ConfinementError(ValueError):
    """A path may not be indexed. ``reason`` tells the caller WHY, so the API can
    offer 'add and continue' for a fixable whitelist miss (``outside_allowed``)
    but hard-refuse the rest (``data_dir`` / ``credential`` / ``denied`` /
    ``invalid``). Subclasses ``ValueError`` so existing ``except ValueError``
    sites keep catching it."""

    def __init__(self, message: str, *, path: Path, reason: str):
        super().__init__(message)
        self.path = path
        self.reason = reason


_INDEX_MODES = ("whitelist", "blacklist")


def indexing_policy(cfg: Optional[dict] = None) -> dict:
    """The current RAG indexing confinement policy, read from config.

    ``mode`` is ``whitelist`` (index only your home folder, the working directory,
    and the ``rag_allowed_roots`` you added) or ``blacklist`` (index anywhere
    EXCEPT the ``rag_denied_roots`` you listed). In BOTH modes the localm data dir
    and credential folders are still refused - a hard floor that
    ``confine_index_path`` enforces separately and no mode can turn off. Returns
    resolved ``Path`` lists so callers compare like-for-like.
    """
    if cfg is None:
        try:
            from localm.config import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
    mode = cfg.get("rag_indexing_mode", "whitelist")
    if mode not in _INDEX_MODES:
        mode = "whitelist"

    def _resolve(key: str) -> list[Path]:
        out: list[Path] = []
        for r in cfg.get(key, []) or []:
            try:
                out.append(Path(r).expanduser().resolve())
            except (OSError, ValueError):
                continue
        return out

    return {"mode": mode,
            "allowed": _resolve("rag_allowed_roots"),
            "denied": _resolve("rag_denied_roots")}


def confine_index_path(p, policy: Optional[dict] = None) -> Path:
    """Resolve *p* and verify it may be indexed, raising ``ConfinementError`` (a
    ``ValueError``) otherwise.

    The HARD FLOOR is enforced ALWAYS, even when *policy* is None: the localm data
    directory (it holds the API key, registry and config) and well-known
    credential folders (``.ssh``, ``.aws``, ...) are never indexable - wherever
    they appear in the resolved path, so a nested ``~/proj/.ssh`` or a symlink
    into one is caught too.

    With a *policy* (the HTTP API passes ``indexing_policy()``):
      - ``whitelist``: *p* must be within your home folder, the working directory,
        or a ``rag_allowed_roots`` entry, else ``reason='outside_allowed'`` (the
        route may offer the owner to add it and continue);
      - ``blacklist``: *p* is allowed unless it is within a ``rag_denied_roots``
        entry, then ``reason='denied'``.

    ``policy=None`` means hard-floor only: the local CLI operator, who can already
    read their own files, is otherwise unconfined.
    """
    from localm.config import home_dir
    try:
        rp = Path(p).expanduser().resolve()
    except (OSError, ValueError):
        raise ConfinementError(f"Invalid path: {p}",
                               path=Path(str(p)), reason="invalid")

    # --- hard floor: always, in every mode and for the CLI ---
    if _path_within(rp, home_dir()):
        raise ConfinementError(
            f"Refusing to index the localm data directory "
            f"(it holds the API key and registry): {p}",
            path=rp, reason="data_dir")
    # Credential folders are denied wherever they appear in the resolved path,
    # not only at the home root: ~/proj/.ssh and <cwd>/sub/.aws are as sensitive
    # as ~/.ssh. rp is already resolved, so this also catches a symlink that
    # points into a credential dir. Tradeoff: a folder literally named one of
    # these (a real ./.docker you wanted to index) is refused too - acceptable
    # for a credential denylist.
    if any(part.lower() in _SENSITIVE_NAMES for part in rp.parts):
        raise ConfinementError(f"Refusing to index a credential directory: {p}",
                               path=rp, reason="credential")

    if policy is None:
        return rp

    if policy.get("mode") == "blacklist":
        # Allow anything not explicitly denied (the hard floor above still holds).
        # Path(d) coerces in case a caller hand-built the policy with str entries
        # (indexing_policy() already returns Paths; this just never trusts that).
        for d in policy.get("denied", []):
            if _path_within(rp, Path(d)):
                raise ConfinementError(
                    f"This folder is on your denied list, so it is not indexed: {p}",
                    path=rp, reason="denied")
        return rp

    # whitelist: your home folder + the working dir are always allowed, plus the
    # roots you added. Anything else is a fixable miss (the owner can widen).
    roots: list[Path] = []
    for r in [Path.home(), Path.cwd(), *policy.get("allowed", [])]:
        try:
            roots.append(Path(r).resolve())   # coerce str entries, then resolve
        except (OSError, ValueError):
            continue
    if any(_path_within(rp, r) for r in roots):
        return rp
    raise ConfinementError(
        f"This folder is outside the folders localm may index. Add it to your "
        f"allowed folders in Settings to index it: {p}",
        path=rp, reason="outside_allowed")


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


# Per-collection-NAME locks. The mutation race (CHK-RAG-LOCK) is across Collection
# INSTANCES: two requests each construct Collection(name), each _load()s the same
# on-disk state, each add a different doc, and each _save()s - last writer wins and
# one update is silently lost. A per-instance lock cannot help (different objects);
# the lock must be keyed by the collection name so writes to one collection serialise
# process-wide. Keyed by name, so the map is bounded by the number of collections
# (small, stable), not per-event. RLock so a locked method may call another safely.
_COLLECTION_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _collection_lock(name: str):
    with _LOCKS_GUARD:
        lock = _COLLECTION_LOCKS.get(name)
        if lock is None:
            lock = threading.RLock()
            _COLLECTION_LOCKS[name] = lock
        return lock


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
        # Why semantic (vector) scoring is unavailable when it should be present.
        # None = vectors are used, or legitimately absent (no embeddings indexed).
        # A non-None string means a corrupt/stale/mismatched vectors index was
        # DETECTED and we fell back to BM25 lexical - surfaced, not silently
        # swallowed (AGENTS rule 5). Exposed via stats() and logged once.
        self.vector_degrade_reason: Optional[str] = None
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
        self.vector_degrade_reason = None
        vec_file = self.dir / "vectors.json"
        if vec_file.is_file():
            # vectors.json PRESENT but unusable is the unexpected case: do NOT
            # collapse it with "simply absent" into one silent path (AGENTS rule 5).
            try:
                data = json.loads(vec_file.read_text(encoding="utf-8"))
                vectors = data.get("vectors", [])
            except (json.JSONDecodeError, OSError) as e:
                data, vectors = None, None
                self._note_vector_degrade(
                    f"vectors.json is unreadable ({type(e).__name__}); "
                    f"using BM25 lexical retrieval only", warn=True)
            if vectors is not None:
                if not _well_formed_vectors(vectors):
                    # Valid JSON but the entries are not vectors (scalars/strings
                    # from a hand-edit or truncation): would crash cosine at query
                    # time, so treat it as corrupt here rather than later.
                    self._note_vector_degrade(
                        "vectors.json is malformed (entries are not vectors); "
                        "using BM25 lexical retrieval only", warn=True)
                elif len(vectors) == len(self._chunks):
                    self._vectors = vectors
                    self._vec_dim = data.get("dim") or _first_dim(vectors)
                elif vectors:
                    # A non-empty vectors list that does not line up with the
                    # chunks is a stale/partial index, not "no embeddings yet".
                    self._note_vector_degrade(
                        f"vectors.json has {len(vectors)} vectors for "
                        f"{len(self._chunks)} chunks (stale or partial index); "
                        f"using BM25 lexical retrieval only", warn=True)
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
        # Windows MoveFileEx can transiently deny the rename if another process
        # (AV real-time scan, Search Indexer) briefly has a handle open; retry
        # rather than fail a good write (mirrors episodes.EpisodeStore.add).
        for attempt in range(5):
            try:
                tmp.replace(self.dir / filename)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)

    # ------------------------------------------------------------- #
    #  Indexing                                                      #
    # ------------------------------------------------------------- #

    @staticmethod
    def _expand(paths: list,
                policy: Optional[dict] = None) -> list[Path]:
        """Resolve files + recursive folder contents to indexable files.

        When *policy* is given, files resolving outside the confinement (system
        paths, credential dirs, denied roots, or symlinks escaping an allowed
        folder) are silently dropped - add_paths already validated the top-level
        inputs, so this only catches sneaky nested escapes."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_file():
                out.append(p.resolve())
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if (f.is_file()
                            and f.suffix.lower() not in BLACKLISTED_SUFFIXES
                            and not is_secret_index_name(f.name)
                            and not any(part in _SKIP_DIRS for part in f.parts)):
                        out.append(f.resolve())
        # de-dup, keep order
        seen: set = set()
        deduped = [p for p in out if not (p in seen or seen.add(p))]
        if policy is None:
            return deduped
        kept: list[Path] = []
        for p in deduped:
            try:
                confine_index_path(p, policy)
            except ValueError:
                continue   # nested escape (symlink / credential / denied) -> skip
            kept.append(p)
        return kept

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  classify_fn: Optional[ClassifyFn] = None,
                  describe_image_fn: Optional[DescribeImageFn] = None,
                  on_progress: Optional[ProgressFn] = None,
                  policy: Optional[dict] = None,
                  force: bool = False) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size+content hash) are
        skipped; changed ones are re-indexed in place. Pass ``force=True`` to
        re-index every file regardless (``localm rag add --force`` / repair).
        Returns counters plus per-file failures. embed_fn failures degrade to
        lexical-only, never abort.

        When *policy* is given (the HTTP API passes ``indexing_policy()``), an
        out-of-bounds top-level path raises ``ValueError`` and nested escapes are
        dropped (C2). CLI callers omit it and stay unconfined. Indexing with an
        embedding model whose dimensionality differs from the collection's also
        raises ``ValueError`` rather than corrupting the vectors with mixed
        dimensions (C3).
        """
        # Serialise the whole read-modify-write per collection AND re-sync with the
        # latest committed state under the lock, so a concurrent add_paths() that
        # finished first is not read-stale-then-overwritten (CHK-RAG-LOCK).
        with _collection_lock(self.name):
            self._load()
            return self._add_paths_locked(
                paths, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, policy=policy, force=force)

    def _add_paths_locked(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                          classify_fn: Optional[ClassifyFn] = None,
                          describe_image_fn: Optional[DescribeImageFn] = None,
                          on_progress: Optional[ProgressFn] = None,
                          policy: Optional[dict] = None,
                          force: bool = False) -> dict:
        """The add_paths read-modify-write body. MUST run under
        _collection_lock(self.name) after a fresh _load() (see add_paths)."""
        say = on_progress or (lambda _t: None)
        if policy is not None:
            for p in paths:
                confine_index_path(p, policy)   # raises ValueError
        files = self._expand(paths, policy)
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
            # Content hash, not just (mtime, size): a same-size edit whose mtime
            # is unchanged (coarse-mtime FS, cp -p / rsync --times, git restore,
            # restore-from-backup, mtime-preserving editors) would otherwise be
            # silently skipped and the index left stale. Legacy entries lacking
            # "hash" compare unequal and self-heal on the next add.
            try:
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            if not force and known \
                    and known.get("mtime") == stat.st_mtime \
                    and known.get("size") == stat.st_size \
                    and known.get("hash") == digest:
                skipped += 1
                continue
            try:
                text = extract_text(f, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {f.name}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Label the document's format heuristic-first (free); the LLM tie-break
            # is only consulted for an unknown extension whose structure is unclear
            # AND a chat model is loaded (classify_fn short-circuits otherwise).
            fmt = classify_format(text, f.name, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

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
                "hash": digest,
                "chunks": len(new_chunks),
            }
            say(f"indexed {f.name} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def add_uploads(self, uploads: list, *, embed_fn: Optional[EmbedFn] = None,
                    classify_fn: Optional[ClassifyFn] = None,
                    describe_image_fn: Optional[DescribeImageFn] = None,
                    on_progress: Optional[ProgressFn] = None,
                    force: bool = False) -> dict:
        """Index documents UPLOADED from the caller's own device (the per-device
        path for a client that cannot browse the server disk).

        Each item is ``{"filename": str, "data": bytes}``. Extraction runs in
        memory (``extract_bytes`` - same zip-bomb / encoding guards as chat
        attachments); the resulting chunks and optional vectors ARE persisted,
        because a knowledge base is explicit user data like ``add_paths``. There is
        deliberately NO filesystem confinement here: nothing on the server disk is
        read, so the whitelist/blacklist policy does not apply - the bytes are the
        caller's own device content. Docs are keyed ``upload:<filename>`` and
        deduped by content hash, so re-uploading an unchanged file is skipped.
        Returns the same counters as ``add_paths``.

        The uploaded BYTES are not retained (only the extracted chunks/vectors), so
        an ``upload:<name>`` doc cannot be re-read from disk: ``localm rag repair``
        simply skips these keys (Path('upload:x') is not a file) - their chunks
        persist and are never lost, they just cannot be re-embedded from source.
        """
        with _collection_lock(self.name):
            self._load()
            return self._add_uploads_locked(
                uploads, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, force=force)

    def _add_uploads_locked(self, uploads: list, *,
                            embed_fn: Optional[EmbedFn] = None,
                            classify_fn: Optional[ClassifyFn] = None,
                            describe_image_fn: Optional[DescribeImageFn] = None,
                            on_progress: Optional[ProgressFn] = None,
                            force: bool = False) -> dict:
        """The add_uploads body. MUST run under _collection_lock after _load().

        Mirrors the per-document body of _add_paths_locked (chunk -> embed ->
        replace-prior-chunks -> record meta), but sourced from in-memory bytes with
        a hash-only dedup (no fs mtime/size). Kept as a separate loop rather than
        refactoring the battle-tested path loop, so the server-disk path is
        untouched."""
        say = on_progress or (lambda _t: None)
        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None

        for up in uploads:
            filename = (str(up.get("filename") or "").strip() or "upload")
            data = up.get("data") or b""
            key = f"upload:{filename}"
            digest = hashlib.sha256(data).hexdigest()
            known = self._meta["docs"].get(key)
            if not force and known and known.get("hash") == digest:
                skipped += 1
                continue
            try:
                text = extract_bytes(data, filename, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {filename}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Same heuristic-first labeling as add_paths (see there).
            fmt = classify_format(text, filename, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

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
                        if (self._vec_dim is not None and new_dim is not None
                                and new_dim != self._vec_dim):
                            raise ValueError(
                                f"Embedding dimension changed "
                                f"({self._vec_dim} -> {new_dim}): this collection "
                                f"was built with a different embedding model. "
                                f"Rebuild it (delete and re-add) or index with the "
                                f"original model.")
                        vectors = vecs
                        if self._vec_dim is None and new_dim is not None:
                            self._vec_dim = new_dim

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
                "size": len(data), "hash": digest,
                "chunks": len(new_chunks), "uploaded": True,
            }
            say(f"indexed {filename} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def documents(self) -> list:
        """The source paths currently indexed in this collection (for repair)."""
        return list(self._meta.get("docs", {}).keys())

    def remove_doc(self, source: str) -> bool:
        # Same per-collection lock + re-sync as add_paths so a concurrent add and
        # remove on one collection cannot lose each other's write (CHK-RAG-LOCK).
        with _collection_lock(self.name):
            self._load()
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

    def _note_vector_degrade(self, reason: str, *, warn: bool) -> None:
        """Record WHY semantic (vector) scoring is unavailable and surface it once.

        We do not hide problems (AGENTS rule 5): a corrupt, stale, or dimensionally
        mismatched vectors index must not silently vanish into BM25-only. Recording
        the reason (exposed via ``stats()``) and logging genuine corruption once -
        idempotent, so a per-query path does not spam the log - is the right
        altitude: the lexical fallback still works, but the fault is discoverable."""
        if self.vector_degrade_reason == reason:
            return
        self.vector_degrade_reason = reason
        if warn:
            _log.warning("RAG collection %r: %s", self.name, reason)

    def _vector_scores(self, text: str,
                       embed_fn: Optional[EmbedFn]) -> Optional[list[float]]:
        if embed_fn is None or self._vectors is None:
            return None
        present = [v for v in self._vectors if v]
        if not self._chunks or len(present) / len(self._chunks) < 0.8:
            # Partial coverage is expected while a collection is still embedding,
            # so record it (visible in stats) but do not warn - not a corruption.
            self._note_vector_degrade(
                "vector coverage below 80% (index not fully embedded); "
                "using BM25 lexical retrieval only", warn=False)
            return None
        # Stored vectors must share one dimensionality. A legacy collection with
        # mixed-dim vectors (built before the C3 add-time guard) is ambiguous -
        # skip vector scoring and answer lexically rather than mis-score with
        # zeros for the odd-dim chunks.
        dims = {len(v) for v in present}
        if len(dims) > 1:
            self._note_vector_degrade(
                "stored vectors have mixed dimensionality (legacy index); "
                "using BM25 lexical retrieval only", warn=True)
            return None
        stored_dim = next(iter(dims))
        try:
            qvec = embed_fn([text])[0]
        except Exception as e:
            # The embedder raised: a real failure (backend down, model unloaded),
            # not an expected transient like partial coverage - surface it. The
            # note is idempotent, so this warns once per distinct error, not per query.
            self._note_vector_degrade(
                f"query embedding failed ({type(e).__name__}); "
                f"using BM25 lexical retrieval only", warn=True)
            return None
        # A switched embedding model yields query vectors of a different
        # dimensionality than the stored ones: fall back to lexical-only rather
        # than crash or return wrong scores.
        if len(qvec) != stored_dim:
            self._note_vector_degrade(
                f"embedding model changed (query dim {len(qvec)} != stored "
                f"{stored_dim}); using BM25 lexical retrieval only - rebuild the "
                f"collection to restore semantic search", warn=True)
            return None
        # Vectors are usable: clear any stale query-time degrade note.
        self.vector_degrade_reason = None
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
        present = [v for v in (self._vectors or []) if v]
        return {
            "name": self.name,
            "created": self._meta.get("created"),
            "n_docs": len(self._meta.get("docs", {})),
            "n_chunks": len(self._chunks),
            # "has vectors" = whether query() will actually blend embeddings: the
            # same >=80% coverage threshold _vector_scores uses, NOT "every chunk
            # embedded". A partially-embedded collection (80-99%) still does hybrid
            # retrieval, so the retrieval-mode label must not under-report it (rule 5).
            "has_vectors": bool(present) and len(present) >= 0.8 * len(self._chunks),
            "corrupt": self.corrupt,
            # Why semantic search fell back to BM25 (None when vectors are used or
            # legitimately absent); surfaced instead of silently swallowed.
            "vector_degrade_reason": self.vector_degrade_reason,
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
