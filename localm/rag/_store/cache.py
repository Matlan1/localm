# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-wide cache of loaded collection snapshots."""

from __future__ import annotations

import array
import contextlib
import os
import sys
import threading
import types
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from localm.rag import store as _st

from ..bm25 import BM25, ENGLISH_STOP_WORDS

if TYPE_CHECKING:
    from localm.rag.store import Collection


#: Ceiling, in estimated bytes, on everything the collection cache holds.
_COLLECTION_CACHE_MAX_BYTES = 256 * 1024 * 1024


#: Ceiling on how many collections the cache holds at once.
_MAX_CACHED_COLLECTIONS = 8


_FLOAT_BYTES = sys.getsizeof(0.5)


_INT_BYTES = sys.getsizeof(1000)


_POSTING_BYTES = sys.getsizeof((1, 2))


_PROXY_BYTES = sys.getsizeof(types.MappingProxyType({}))


_JSON_CONTAINERS = (dict, list)


def _collection_cache_fingerprint(coll_dir: Path) -> dict:
    """(file id, mtime_ns, size) for meta.json, chunks.jsonl and vectors.json,
    None for a file that does not exist. Every atomic replace gives the file a
    new id."""
    def _stat(name: str) -> tuple[int, int, int] | None:
        try:
            st = os.stat(coll_dir / name)
        except OSError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)
    return {
        "meta": _stat("meta.json"),
        "chunks": _stat("chunks.jsonl"),
        "vectors": _stat("vectors.json"),
    }


def _cache_key(coll_dir: Path) -> str:
    try:
        path = coll_dir.resolve()
    except (OSError, RuntimeError, ValueError):
        path = coll_dir
    return os.path.normcase(str(path))


def _copy_json(value):
    """A copy of a JSON-shaped value with every dict and list copied at every
    level; scalars are shared. Iterative, so nesting depth is unbounded."""
    if type(value) not in _JSON_CONTAINERS:
        return value
    root = dict(value) if type(value) is dict else list(value)
    pending = [root]
    while pending:
        node = pending.pop()
        entries = node.items() if type(node) is dict else enumerate(node)
        for k, v in entries:
            if type(v) in _JSON_CONTAINERS:
                child = dict(v) if type(v) is dict else list(v)
                node[k] = child
                pending.append(child)
    return root


def _json_nbytes(value) -> int:
    """Estimated memory held by a JSON-shaped value, containers and contents.
    Iterative, so nesting depth is unbounded."""
    size = 0
    pending = [value]
    while pending:
        node = pending.pop()
        size += sys.getsizeof(node)
        kind = type(node)
        if kind is dict:
            for k, v in node.items():
                size += sys.getsizeof(k)
                pending.append(v)
        elif kind is list:
            pending.extend(node)
    return size


def _bm25_nbytes(index: BM25) -> int:
    """Estimated memory held by a BM25 index: its postings and their term
    strings, the idf table, and per chunk its length and its posting index."""
    size = sys.getsizeof(index) + sys.getsizeof(vars(index))
    postings = getattr(index, "_postings", {})
    size += sys.getsizeof(postings)
    for term, plist in postings.items():
        size += (sys.getsizeof(term) + sys.getsizeof(plist)
                 + len(plist) * _POSTING_BYTES)
    idf = getattr(index, "_idf", {})
    size += sys.getsizeof(idf) + len(idf) * _FLOAT_BYTES
    lengths = getattr(index, "_lengths", ())
    size += sys.getsizeof(lengths) + len(lengths) * 2 * _INT_BYTES
    return size


def _freeze_rows(vectors: Optional[list], compact: bool):
    """(read-only rows, estimated bytes) for a vectors list. *compact* stores
    each row as a read-only float64 buffer; otherwise each row is a tuple."""
    if vectors is None:
        return None, 0
    rows: list = []
    nbytes = 0
    for v in vectors:
        if v is None:
            rows.append(None)
        elif compact:
            try:
                buf = array.array("d", v)
            except (TypeError, ValueError, OverflowError):
                return _freeze_rows(vectors, compact=False)
            view = memoryview(buf).toreadonly()
            rows.append(view)
            nbytes += sys.getsizeof(buf) + sys.getsizeof(view)
        else:
            row = tuple(v)
            rows.append(row)
            nbytes += sys.getsizeof(row) + len(row) * _FLOAT_BYTES
    frozen = tuple(rows)
    return frozen, nbytes + sys.getsizeof(frozen)


class _Snapshot:
    """One collection's loaded state, frozen: read-only chunks, vector rows and
    norm matrix, plus a private copy of meta.json, all from one read of the
    files ``fingerprint`` describes. ``bm25`` is built from ``chunks`` alone."""

    __slots__ = ("key", "fingerprint", "meta", "chunks", "vectors", "vec_dim",
                 "norm_matrix", "degrade", "bad_lines", "corrupt",
                 "meta_unreadable", "rejected", "bm25", "bm25_lock", "nbytes",
                 "__weakref__")


def _freeze_snapshot(coll: Collection, key: str, fingerprint: dict) -> _Snapshot:
    """A snapshot of *coll*'s just-loaded state. Wraps *coll*'s chunk dicts and
    marks its norm matrix read-only, so *coll* must be re-served from the
    snapshot (``_serve_snapshot``) once it is cached."""
    snap = _Snapshot()
    snap.key = key
    snap.fingerprint = fingerprint
    snap.meta = _copy_json(coll._meta)
    nbytes = _json_nbytes(snap.meta)
    chunks = []
    for c in coll._chunks:
        chunks.append(types.MappingProxyType(c))
        nbytes += _json_nbytes(c) + _PROXY_BYTES
    snap.chunks = tuple(chunks)
    nbytes += sys.getsizeof(snap.chunks)
    matrix = coll._norm_matrix
    snap.vectors, rows_nbytes = _freeze_rows(coll._vectors, compact=matrix is not None)
    nbytes += rows_nbytes
    if matrix is not None:
        matrix.setflags(write=False)
        nbytes += max(sys.getsizeof(matrix), matrix.nbytes)
    snap.norm_matrix = matrix
    snap.vec_dim = coll._vec_dim
    snap.degrade = coll.vector_degrade_reason
    snap.bad_lines = coll.chunks_bad_lines
    snap.corrupt = coll.corrupt
    snap.meta_unreadable = coll._meta_unreadable
    snap.rejected = coll._vectors_file_rejected
    snap.bm25 = None
    snap.bm25_lock = threading.Lock()
    snap.nbytes = nbytes
    return snap


class _SnapshotCache:
    """Process-wide LRU of collection snapshots, bounded by estimated bytes
    (``_COLLECTION_CACHE_MAX_BYTES``) and by entry count
    (``_MAX_CACHED_COLLECTIONS``).

    Every write to a collection's files runs inside ``begin_write`` /
    ``end_write``; both drop that collection's entry and advance an epoch. A
    reader takes ``read_token`` before reading and ``put`` stores its snapshot
    only when no write was in progress then and the epoch has not moved since.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Snapshot] = OrderedDict()
        self._bytes = 0
        self._epoch = 0
        self._writing = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def total_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def get(self, key: str, coll_dir: Path) -> Optional[_Snapshot]:
        """The cached snapshot for *key*, or None. An entry whose fingerprint
        no longer matches the files in *coll_dir* is dropped."""
        with self._lock:
            snap = self._entries.get(key)
        if snap is None:
            return None
        current = _st._collection_cache_fingerprint(coll_dir)
        with self._lock:
            if self._entries.get(key) is not snap:
                return None
            if snap.fingerprint != current:
                self._discard_locked(key)
                return None
            self._entries.move_to_end(key)
            return snap

    def read_token(self) -> Optional[int]:
        """The current epoch, or None while any write is in progress."""
        with self._lock:
            return None if self._writing else self._epoch

    def token_current(self, token: Optional[int]) -> bool:
        with self._lock:
            return token is not None and not self._writing and token == self._epoch

    def put(self, snap: _Snapshot, token: Optional[int]) -> Optional[_Snapshot]:
        """Cache *snap* and return the snapshot now cached for its key, which is
        an existing one with the same fingerprint when there is one. None when
        nothing was stored: a write started or finished since *token* was
        taken, or *snap* alone exceeds the byte budget."""
        with self._lock:
            # Stores only when no write started or finished since the token was
            # taken. See test_in_process_write_blocks_the_store_even_when_stats_cannot_show_it.
            if token is None or self._writing or token != self._epoch:
                return None
            if snap.nbytes > _st._COLLECTION_CACHE_MAX_BYTES:
                return None
            current = self._entries.get(snap.key)
            if current is not None and current.fingerprint == snap.fingerprint:
                self._entries.move_to_end(snap.key)
                return current
            self._discard_locked(snap.key)
            self._entries[snap.key] = snap
            self._bytes += snap.nbytes
            self._evict_locked()
            return snap if self._entries.get(snap.key) is snap else None

    def lexical_index(self, snap: _Snapshot) -> BM25:
        """*snap*'s BM25 index, built once from *snap*'s own chunks. It is kept
        on *snap*, and counted against the budget, only while *snap* is the
        cached entry for its key and the total still fits."""
        with snap.bm25_lock:
            index = snap.bm25
            if index is not None:
                return index
            index = BM25([c["text"] for c in snap.chunks],
                         stop_words=ENGLISH_STOP_WORDS)
            extra = _bm25_nbytes(index)
            with self._lock:
                if (self._entries.get(snap.key) is snap
                        and snap.nbytes + extra <= _st._COLLECTION_CACHE_MAX_BYTES):
                    snap.bm25 = index
                    snap.nbytes += extra
                    self._bytes += extra
                    self._entries.move_to_end(snap.key)
                    self._evict_locked()
            return index

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._discard_locked(key)

    def begin_write(self, key: str) -> None:
        with self._lock:
            self._writing += 1
            self._epoch += 1
            self._discard_locked(key)

    def end_write(self, key: str) -> None:
        with self._lock:
            self._writing -= 1
            self._epoch += 1
            self._discard_locked(key)

    def _discard_locked(self, key: str) -> None:
        snap = self._entries.pop(key, None)
        if snap is not None:
            self._bytes -= snap.nbytes

    def _evict_locked(self) -> None:
        """Drop least recently used entries until both ceilings hold."""
        while self._entries and (self._bytes > _st._COLLECTION_CACHE_MAX_BYTES
                                 or len(self._entries) > _MAX_CACHED_COLLECTIONS):
            _key, snap = self._entries.popitem(last=False)
            self._bytes -= snap.nbytes


_COLLECTION_CACHE = _SnapshotCache()


def _get_cached_collection_data(coll_dir: Path) -> Optional[_Snapshot]:
    return _COLLECTION_CACHE.get(_cache_key(coll_dir), coll_dir)


def _invalidate_collection_cache(coll_dir: Path) -> None:
    _COLLECTION_CACHE.invalidate(_cache_key(coll_dir))


@contextlib.contextmanager
def _files_changing(coll_dir: Path):
    """Wrap a change to a collection's files: its cache entry is dropped on
    entry and on exit, and no reader stores a snapshot while it runs."""
    key = _cache_key(coll_dir)
    _COLLECTION_CACHE.begin_write(key)
    try:
        yield
    finally:
        _COLLECTION_CACHE.end_write(key)
