# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Persistent document collections ("knowledge bases").

Layout - one directory per collection under ``<data dir>/rag/``:

    rag/<name>/meta.json      {"name", "created",
                               "docs": {path: {mtime, size, chunks}},
                               "roots": {folder: {"added": ts}}}
    rag/<name>/chunks.jsonl   one chunk per line: {"source", "pos", "text"}
    rag/<name>/vectors.json   optional: {"dim", "vectors": [[...]|null, ...]}
                              aligned with chunks.jsonl line order

``roots`` records the FOLDERS that were indexed, alongside the per-file ``docs``
entries the folder walk produced. ``resync()`` re-walks these roots through the
ordinary incremental path.

Collections are explicit user data (like generated images): indexing writes
to disk in every session mode. Rewrites are whole-file + atomic rename.

Retrieval is hybrid: BM25 always; when vectors exist for (almost) all chunks
and the caller can embed the query, scores become an equal blend of
max-normalised BM25 and cosine similarity. Those scores are relative to the
best chunk; ``query(relevant_only=True)`` additionally drops hits below an
absolute relevance floor, so an unrelated query returns nothing.
"""

from __future__ import annotations

import json
import os  # noqa: F401
import re
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

from localm.jsonl import split_jsonl  # noqa: F401
from localm.storekit import NamespaceLockRegistry, atomic_write as _storekit_atomic_write
from .bm25 import BM25
from .collection_lock import (CollectionLockedError, collection_write_lock,
                              lock_path_for, wait_budget)
from .extract import ExtractError, extract_bytes  # noqa: F401
from ._store.cache import (  # noqa: F401
    _COLLECTION_CACHE, _COLLECTION_CACHE_MAX_BYTES, _FLOAT_BYTES, _INT_BYTES,
    _JSON_CONTAINERS, _MAX_CACHED_COLLECTIONS, _POSTING_BYTES, _PROXY_BYTES,
    _Snapshot, _SnapshotCache, _bm25_nbytes, _cache_key,
    _collection_cache_fingerprint, _copy_json, _files_changing, _freeze_rows,
    _freeze_snapshot, _get_cached_collection_data, _invalidate_collection_cache,
    _json_nbytes)
from ._store.confine import (  # noqa: F401
    _INDEX_MODES, _SENSITIVE_HOME_SUBDIRS, _SENSITIVE_NAMES, ConfinementError,
    _network_drives_allowed_fresh, _path_within, confine_index_path,
    indexing_policy)
from ._store.embedding import _CollectionEmbedding
from ._store.files import (  # noqa: F401
    _STATS_CACHE_KEY, _WARNED_DEGRADES, _CollectionFiles, _plain_chunks,
    _plain_rows)
from ._store.indexing import (  # noqa: F401
    _MAX_WALK_DEPTH, _SKIP_DIRS, _CollectionIndexing, _walk_files)
from ._store.introspect import _CollectionIntrospection
from ._store.provenance import (  # noqa: F401
    _provenance_excludable, collection_provenance_note,
    collection_provenance_report)
from ._store.resync import _CollectionResync
from ._store.search import (  # noqa: F401
    _CONVERSATION_REF_RE, LEXICAL_RELEVANCE_COVERAGE, RELEVANCE_FLOORS,
    _CollectionSearch, refers_to_conversation)
from ._store.sidecar import (  # noqa: F401
    _MAX_REJECTED_KEPT, _REJECTED_VECTORS, _CollectionSidecar)
from ._store.types import ClassifyFn, DescribeImageFn, EmbedFn, ProgressFn  # noqa: F401
from ._store.vectors import (  # noqa: F401
    _NUMPY_DEGRADE_LOGGED, _NUMPY_IS_STUB, _cosine, _first_dim, _maxnorm, _numpy,
    _vectors_finite, _warn_numpy_degrade, _well_formed_vectors)


# Printable, 1-64 characters, no control chars, no path separators, no
# Windows-reserved punctuation, and no ".". See
# test_dot_rejected_would_collide_with_lock_sibling.
_NAME_RE = re.compile(r'\A[^\x00-\x1f\x7f./\\:*?"<>|]{1,64}\Z')


# Windows reserved device names: they match _NAME_RE but mkdir raises on them.
_RESERVED_NAMES = {"con", "prn", "aux", "nul",
                   *(f"com{i}" for i in range(1, 10)),
                   *(f"lpt{i}" for i in range(1, 10))}


def rag_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "rag"


def check_collection_name(name: str) -> str:
    """Validate a collection name, returning it, or raise ``ValueError``."""
    name = name or ""
    if not _NAME_RE.match(name):
        raise ValueError(
            'Collection names must be 1-64 characters and cannot contain '
            '. / \\ : * ? " < > | or control characters')
    if name != name.strip():
        raise ValueError("Collection names cannot start or end with whitespace")
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(f"'{name}' is a reserved device name and cannot be used")
    return name


# Internal alias for the in-module call sites.
_check_name = check_collection_name


def collection_names(base: Optional[Path] = None) -> list[str]:
    base = base or rag_dir()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / "meta.json").is_file())


def delete_collection(name: str, base: Optional[Path] = None,
                      on_wait: Optional[Callable[[str], None]] = None) -> bool:
    """Delete a collection, waiting for any in-flight write to finish first.

    Deleting takes the same locks a write does. Raises ``CollectionLockedError``
    if another process's run does not finish in time, rather than deleting
    underneath it."""
    import shutil
    base = base or rag_dir()
    path = base / _check_name(name)
    if not (path / "meta.json").is_file():
        return False
    # Bounds the in-process half too: refuses after the same budget instead of
    # queueing behind a re-sync.
    budget = wait_budget()
    local = _collection_lock(name)
    if not local.acquire(timeout=budget):
        raise CollectionLockedError(name, None, budget, same_process=True)
    try:
        with collection_write_lock(lock_path_for(path), collection=name,
                                   op="a delete", on_wait=on_wait):
            if not (path / "meta.json").is_file():
                return False      # someone else deleted it while we waited
            with _files_changing(path):
                shutil.rmtree(path)
    finally:
        local.release()
    return True


def relabel_embedding_model(old_name: str, new_name: str,
                            base: Optional[Path] = None) -> "tuple[list, list]":
    """Rewrite the recorded embedding-model label *old_name* to *new_name* in
    every collection that carries it.

    Only meta.json is read and rewritten, under the same two locks every
    writer takes, each waited on for at most ``wait_budget()``. A collection
    whose meta.json cannot be parsed has no readable label and is left alone.
    Returns ``(relabelled, busy)``: the collection names rewritten, and the ones
    skipped because another write held them past the wait budget."""
    base = base or rag_dir()
    relabelled: list = []
    busy: list = []
    for name in collection_names(base):
        found = Collection._peek_meta(name, base)
        if found is None or found[2].get("embedding_model") != old_name:
            continue
        coll_dir = found[1]
        local = _collection_lock(name)
        if not local.acquire(timeout=wait_budget()):
            busy.append(name)
            continue
        try:
            with collection_write_lock(lock_path_for(coll_dir), collection=name,
                                       op="a model rename"):
                found = Collection._peek_meta(name, base)
                if found is None or found[2].get("embedding_model") != old_name:
                    continue
                meta = found[2]
                meta["embedding_model"] = new_name
                with _files_changing(coll_dir):
                    _storekit_atomic_write(coll_dir / "meta.json",
                                           json.dumps(meta, indent=2))
                relabelled.append(name)
        except CollectionLockedError:
            busy.append(name)
        finally:
            local.release()
    return relabelled, busy


# Per-collection-NAME locks: concurrent writes to one collection serialise
# process-wide across separate Collection instances. Keyed by name, so the map
# is bounded by the number of collections. RLock, so a locked method may call
# another.
#
# PER PROCESS: it does not reach a CLI invocation. That half is covered by
# collection_lock.collection_write_lock, a lock FILE beside the collection
# directory, which is always held INSIDE this lock.
_COLLECTION_LOCKS = NamespaceLockRegistry()


def _collection_lock(name: str):
    # Keyed case-INSENSITIVELY: Collection("Docs") and Collection("docs") are
    # two names but the same directory and the same lock file on Windows and
    # macOS, so folding them means two threads meet here rather than at the
    # lock file.
    return _COLLECTION_LOCKS.get(name.casefold())


# The constructor and the entry points that take a collection's locks; the rest
# of Collection's behaviour is in the mixins under localm/rag/_store/.
class Collection(_CollectionFiles, _CollectionSidecar, _CollectionIndexing,
                 _CollectionResync, _CollectionSearch, _CollectionEmbedding,
                 _CollectionIntrospection):
    def __init__(self, name: str, base: Optional[Path] = None, *,
                 cache: bool = True) -> None:
        """*cache* False loads straight from disk and never stores the result
        in the process-wide collection cache."""
        self.name = _check_name(name)
        self.dir = (base or rag_dir()) / self.name
        self._use_cache = cache
        self._meta: dict = {}
        self._chunks: list[dict] = []
        self._vectors: Optional[list] = None     # aligned with _chunks, or None
        self._vec_dim: Optional[int] = None       # dimensionality of stored vectors
        self._bm25: Optional[BM25] = None
        # The chunks list self._bm25 was built from, and its length then.
        self._bm25_source: Optional[list] = None
        self._bm25_size = 0
        self._norm_matrix: Any = None
        # The vectors list self._norm_matrix was built from.
        self._norm_source: Optional[list] = None
        # Weak reference to the cached snapshot this instance was served from.
        self._snapshot_ref: Optional[weakref.ref] = None
        # True when _load() could not read or parse meta.json.
        self._meta_unreadable: bool = False
        self.corrupt: bool = False
        # How many lines of chunks.jsonl _load() had to skip as unparseable or
        # wrong-shape; 0 whenever the file is clean or absent. Exposed via
        # stats().
        self.chunks_bad_lines: int = 0
        # Why semantic (vector) scoring is unavailable when it should be present.
        # None = vectors are used, or legitimately absent (no embeddings indexed).
        # A non-None string means a corrupt/stale/mismatched vectors index was
        # detected and scoring fell back to BM25 lexical. Exposed via stats() and
        # logged once.
        self.vector_degrade_reason: Optional[str] = None
        # True when _load() found a vectors.json on disk and REFUSED to use it;
        # _save() then sets that file aside instead of deleting it. Distinct from
        # vector_degrade_reason, which is also set by query-time degrades (a
        # failed query embedding, partial coverage).
        self._vectors_file_rejected: bool = False
        if self.exists():
            self._load()

    def exists(self) -> bool:
        return (self.dir / "meta.json").is_file()

    def _write_lock(self, op: str, on_progress: Optional[ProgressFn] = None):
        """The CROSS-PROCESS write lock for this collection.

        Every read-modify-write entry point takes it INSIDE the per-process
        ``_collection_lock``, never the other way round, so at most one thread
        of this process is ever at the lock file.

        The two halves have DIFFERENT waiting rules. Writers inside one process
        QUEUE for as long as it takes. A writer in ANOTHER process is bounded
        and ends in a refusal. ``delete_collection`` is the one caller that
        bounds both (see its docstring).

        The wait is reported through the caller's existing progress channel.
        """
        return collection_write_lock(
            lock_path_for(self.dir), collection=self.name, op=op,
            on_wait=on_progress)

    def create(self) -> "Collection":
        """Create the collection if it does not exist yet.

        Takes the write lock and re-checks existence inside it. The fast path
        (already exists) takes no lock at all."""
        if self.exists():
            return self
        with _collection_lock(self.name), self._write_lock("a create"):
            if self.exists():
                return self       # somebody else created it while we waited
            self.dir.mkdir(parents=True, exist_ok=True)
            self._meta = {"name": self.name, "created": time.time(), "docs": {}}
            self._save()
        return self

    def add_paths(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                  classify_fn: Optional[ClassifyFn] = None,
                  describe_image_fn: Optional[DescribeImageFn] = None,
                  on_progress: Optional[ProgressFn] = None,
                  policy: Optional[dict] = None,
                  force: bool = False,
                  model_name: Optional[str] = None) -> dict:
        """
        Index files/folders. Unchanged files (same mtime+size+content hash) are
        skipped; changed ones are re-indexed in place. Pass ``force=True`` to
        re-index every file regardless (``localm rag add --force`` / repair).
        Returns counters plus per-file failures. embed_fn failures degrade to
        lexical-only, never abort.

        When *policy* is given (the HTTP API passes ``indexing_policy()``), an
        out-of-bounds top-level path raises ``ValueError`` and nested escapes are
        dropped. CLI callers omit it and stay unconfined. Indexing with an
        embedding model whose dimensionality differs from the collection's also
        raises ``ValueError``.

        *model_name*, like ``reembed()``'s, is the EMBEDDING model's name, only
        recorded (as ``embedding_model()``) the first time this collection is
        actually embedded - passing it when *embed_fn* is None is harmless, it is
        simply never reached.
        """
        # Serialise the whole read-modify-write per collection, re-reading the
        # latest committed state under the lock. The _load() must happen INSIDE
        # both locks.
        with _collection_lock(self.name), self._write_lock("an index", on_progress):
            self._load(use_cache=False)
            return self._add_paths_locked(
                paths, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, policy=policy, force=force,
                model_name=model_name)

    def resync(self, *, embed_fn: Optional[EmbedFn] = None,
               classify_fn: Optional[ClassifyFn] = None,
               describe_image_fn: Optional[DescribeImageFn] = None,
               on_progress: Optional[ProgressFn] = None,
               policy: Optional[dict] = None,
               force: bool = False,
               prune_missing: bool = False,
               model_name: Optional[str] = None) -> dict:
        """Bring the index back in line with the folders it was built from.

        Re-walks every persisted root through the ORDINARY incremental path
        (``add_paths``): a file ADDED to an indexed folder since the last run is
        picked up, a CHANGED file is re-indexed, an unchanged file is skipped by
        content hash. Individually indexed files are re-checked too. This is what
        a scheduled ``rag`` job calls; it is also ``localm rag resync``.

        DELETION SEMANTICS. A document whose file has VANISHED is FLAGGED
        (``missing: True`` + ``missing_since``), not dropped: its chunks stay
        indexed and stay searchable. The flag CLEARS by itself when the file
        comes back. Actual deletion happens only when the caller passes
        ``prune_missing=True``.

        A root that is not currently an available directory (deleted, unmounted,
        unreadable, or replaced by a file) is REPORTED and skipped whole, and
        every document underneath it is left completely untouched - not indexed,
        not flagged, not pruned. The same holds for a root the current
        ``policy`` refuses.

        *policy* is applied exactly as in ``add_paths``, including to a root that
        was legal when it was added but is outside the owner's allowed folders
        now. Callers that run unattended (the jobs runner) always pass one.

        *model_name*: see ``add_paths()`` - forwarded to the same first-embed
        recording.

        Returns the ``add_paths`` counters plus ``missing`` (newly flagged),
        ``missing_total``, ``restored``, ``pruned``, ``roots``,
        ``unavailable_roots`` and ``blocked_roots`` (each ``{root, reason}``), and
        ``vector_degrade_reason`` (why semantic search is degraded after this run,
        None when it is fine).
        """
        with _collection_lock(self.name), self._write_lock("a re-sync", on_progress):
            self._load(use_cache=False)
            return self._resync_locked(
                embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force, prune_missing=prune_missing,
                model_name=model_name)

    def add_uploads(self, uploads: list, *, embed_fn: Optional[EmbedFn] = None,
                    classify_fn: Optional[ClassifyFn] = None,
                    describe_image_fn: Optional[DescribeImageFn] = None,
                    on_progress: Optional[ProgressFn] = None,
                    force: bool = False,
                    model_name: Optional[str] = None) -> dict:
        """Index documents UPLOADED from the caller's own device (the per-device
        path for a client that cannot browse the server disk).

        Each item is ``{"filename": str, "data": bytes}``. Extraction runs in
        memory (``extract_bytes`` - same zip-bomb / encoding guards as chat
        attachments); the resulting chunks and optional vectors ARE persisted.
        There is NO filesystem confinement here: nothing on the server disk is
        read, so the whitelist/blacklist policy does not apply. Docs are keyed
        ``upload:<filename>`` and deduped by content hash, so re-uploading an
        unchanged file is skipped. Returns the same counters as ``add_paths``.

        The uploaded BYTES are not retained (only the extracted chunks/vectors), so
        an ``upload:<name>`` doc cannot be re-read from disk: ``localm rag repair``
        simply skips these keys (Path('upload:x') is not a file) - their chunks
        persist, they just cannot be re-embedded from source.

        *model_name*: see ``add_paths()`` - the embedding model's name, recorded
        the first time this collection is actually embedded.
        """
        with _collection_lock(self.name), self._write_lock("an upload", on_progress):
            self._load(use_cache=False)
            return self._add_uploads_locked(
                uploads, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn,
                on_progress=on_progress, force=force, model_name=model_name)

    def reembed(self, *, embed_fn: EmbedFn, model_name: Optional[str] = None,
                on_progress: Optional[ProgressFn] = None,
                batch: int = 32) -> dict:
        """Recompute EVERY vector from the stored chunk text, with a new model.

        The chunk text is already on disk in chunks.jsonl, so nothing needs
        re-reading, re-chunking, or even to still exist: a collection whose
        sources moved, were deleted, or arrived as uploads re-embeds exactly the
        same as one whose files are all present. ``rag repair --embed`` differs
        in that it re-indexes FROM THE ORIGINAL SOURCE FILES.

        Every vector is computed into a LOCAL list first, and ``self._vectors``
        is only replaced once the whole set is in hand and validated, so a
        failing embedder leaves the previous index exactly as it was. One full
        vector set is held in memory during the run.
        """
        with _collection_lock(self.name), self._write_lock("reembed", on_progress):
            self._load(use_cache=False)
            return self._reembed_locked(
                embed_fn=embed_fn, model_name=model_name,
                on_progress=on_progress, batch=batch)

    def remove_doc(self, source: str) -> bool:
        # Same per-collection lock and re-load as add_paths, and the same
        # cross-process lock.
        with _collection_lock(self.name), self._write_lock("a document removal"):
            self._load(use_cache=False)
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
