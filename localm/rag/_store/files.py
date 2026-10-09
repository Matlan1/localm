# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading and writing a collection's files: meta.json, chunks.jsonl and
vectors.json, and the ``_stats_cache`` block in meta.json."""

from __future__ import annotations

import json
import weakref
from typing import Optional

from localm.debuglog import logger as _log
from localm.jsonl import dumps_lines
from localm.rag import store as _st

from .cache import (_COLLECTION_CACHE, _Snapshot, _cache_key, _copy_json,
                    _files_changing, _freeze_snapshot)
from .sidecar import _REJECTED_VECTORS
from .vectors import _first_dim, _vectors_finite, _well_formed_vectors


#: Warn-once keys already logged in THIS process: vector degrades
#: (_note_vector_degrade) and the chunks.jsonl malformed-line warning in _load().
#: Never consulted for state, only for whether to LOG. Every key starts with the
#: collection dir plus a distinguishing tag (a literal string, or the degrade
#: text).
_WARNED_DEGRADES: set = set()


#: meta.json key for the derived-stats cache _save() writes and peek_stats() /
#: peek_detail() read. Internal/derived, never user data; a meta.json without it
#: is simply "no cache yet".
_STATS_CACHE_KEY = "_stats_cache"


def _plain_chunks(chunks: list) -> list:
    """*chunks* with every chunk as a plain dict, for serialising."""
    return [c if type(c) is dict else dict(c) for c in chunks]


def _plain_rows(vectors: list) -> list:
    """*vectors* with every row as a list or tuple, for serialising."""
    return [v if v is None or type(v) in (list, tuple) else list(v)
            for v in vectors]


class _CollectionFiles:
    """Mixin of ``Collection``: loading a collection from disk or the snapshot
    cache, and writing it back."""

    def _serve_snapshot(self, snap: _Snapshot) -> None:
        """Adopt *snap*'s state: fresh containers over its read-only chunks and
        rows, and a private copy of its meta."""
        self.corrupt = snap.corrupt
        self.chunks_bad_lines = snap.bad_lines
        self._meta_unreadable = snap.meta_unreadable
        self._meta = _copy_json(snap.meta)
        self._chunks = list(snap.chunks)
        self._vectors = None if snap.vectors is None else list(snap.vectors)
        self._vec_dim = snap.vec_dim
        self.vector_degrade_reason = snap.degrade
        self._vectors_file_rejected = snap.rejected
        self._norm_matrix = snap.norm_matrix
        self._norm_source = self._vectors
        self._bm25 = None
        self._bm25_source = None
        self._snapshot_ref = weakref.ref(snap)

    def _load(self, *, use_cache: Optional[bool] = None) -> None:
        """Read this collection from disk, or adopt its cached snapshot.

        *use_cache* False reads the files and neither consults nor fills the
        cache; None follows the instance's ``cache`` setting. A read is stored
        only when the file fingerprint taken before it matches the one taken
        after it and no write ran in between (see ``_SnapshotCache``)."""
        use_cache = self._use_cache if use_cache is None else use_cache
        self._snapshot_ref = None
        cache_key = token = before = None
        if use_cache:
            cache_key = _cache_key(self.dir)
            cached = _COLLECTION_CACHE.get(cache_key, self.dir)
            if cached is not None:
                self._serve_snapshot(cached)
                return
            token = _COLLECTION_CACHE.read_token()
            if token is not None:
                before = _st._collection_cache_fingerprint(self.dir)
        # A corrupt meta.json is flagged, not fatal, and does not discard the
        # INDEPENDENT chunks.jsonl / vectors.json files. Execution falls through
        # to load the chunks and then rebuild a minimal docs map from their
        # sources.
        self.corrupt = False
        self.chunks_bad_lines = 0
        meta_corrupt = self._read_meta()
        self._read_chunks()
        self._read_vectors()
        self._bm25 = None
        # If meta.json was corrupt but chunks survived, rebuild a minimal docs
        # map from the chunk sources. The rebuilt entries lack mtime/size/hash,
        # so a later add/repair re-reads the file. Gated on META corruption: a
        # valid meta whose chunks.jsonl merely had a bad line keeps its real
        # docs map.
        if meta_corrupt and self._chunks:
            self._rebuild_docs_from_chunks()

        self._build_norm_matrix()
        self._norm_source = self._vectors
        self._bm25_source = None
        self._meta_unreadable = meta_corrupt

        if (before is not None and _COLLECTION_CACHE.token_current(token)
                and self._min_cached_nbytes() <= _st._COLLECTION_CACHE_MAX_BYTES
                and _st._collection_cache_fingerprint(self.dir) == before):
            cached = _COLLECTION_CACHE.put(
                _freeze_snapshot(self, cache_key, before), token)
            if cached is not None:
                self._serve_snapshot(cached)

    def _read_meta(self) -> bool:
        """Read meta.json into ``self._meta``. Returns True, with ``self.corrupt``
        set and a minimal meta in place, when it cannot be read or parsed or is
        not a JSON object."""
        try:
            meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("meta.json is not an object")
            self._meta = meta
        except (json.JSONDecodeError, ValueError, OSError):
            self.corrupt = True
            self._meta = {"name": self.name, "docs": {}}
            return True
        return False

    def _read_chunks(self) -> None:
        """Read chunks.jsonl into ``self._chunks``, skipping and counting lines
        that are not a chunk."""
        self._chunks = []
        chunks_file = self.dir / "chunks.jsonl"
        if chunks_file.is_file():
            bad_lines = 0
            # split_jsonl, NOT str.splitlines(): JSONL is delimited by LINE FEED
            # and nothing else.
            for line in _st.split_jsonl(chunks_file.read_text(encoding="utf-8")):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                # A chunk MUST be a dict carrying a str "text". A
                # valid-JSON-but-wrong-shape line (a scalar or array, or a dict
                # missing "text") is skipped and counted as corruption instead.
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    bad_lines += 1
                    continue
                self._chunks.append(obj)
            self.chunks_bad_lines = bad_lines
            if bad_lines:
                self.corrupt = True
                # Warn-once, using the same process-scoped set as
                # _note_vector_degrade below; self.corrupt is still set
                # unconditionally above. Keyed on the bad_lines COUNT as well as
                # the dir, so a fault that changes shape warns again.
                key = ("chunks_malformed", str(self.dir), bad_lines)
                if key not in _WARNED_DEGRADES:
                    _WARNED_DEGRADES.add(key)
                    _log.warning("RAG collection %r: skipped %d malformed line(s) in "
                                 "chunks.jsonl; run 'localm rag repair'",
                                 self.name, bad_lines)

    def _read_vectors(self) -> None:
        """Read vectors.json into ``self._vectors`` when it is usable and lines up
        with ``self._chunks``; otherwise record why in
        ``self.vector_degrade_reason``."""
        self._vectors = None
        self._vec_dim = None
        self.vector_degrade_reason = None
        self._vectors_file_rejected = False
        vec_file = self.dir / "vectors.json"
        if vec_file.is_file():
            # vectors.json PRESENT but unusable is handled on its own path, never
            # collapsed into "simply absent".
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
                    # Valid JSON but the entries are not vectors (scalars or
                    # strings from a hand-edit or truncation): treated as corrupt.
                    self._note_vector_degrade(
                        "vectors.json is malformed (entries are not vectors); "
                        "using BM25 lexical retrieval only", warn=True)
                elif len(vectors) == len(self._chunks):
                    if not _vectors_finite(vectors):
                        # Structurally a vector list, but a component is NaN/inf
                        # or non-numeric.
                        self._note_vector_degrade(
                            "vectors.json has non-finite (NaN/inf) or non-numeric "
                            "values; using BM25 lexical retrieval only", warn=True)
                    else:
                        self._vectors = vectors
                        self._vec_dim = data.get("dim") or _first_dim(vectors)
                elif vectors:
                    # A non-empty vectors list that does not line up with the
                    # chunks. FEWER vectors than chunks is a partial embed; MORE
                    # means orphaned entries from a prior, larger chunk set.
                    kind = ("a partial embed" if len(vectors) < len(self._chunks)
                            else "orphaned entries from a prior, larger index")
                    self._note_vector_degrade(
                        f"vectors.json has {len(vectors)} vectors for "
                        f"{len(self._chunks)} chunks ({kind}); "
                        f"using BM25 lexical retrieval only", warn=True)
            # Any reason recorded in this block means the file IS there and was
            # refused. Remembered for _save().
            self._vectors_file_rejected = self.vector_degrade_reason is not None
        # A sidecar an earlier write set aside (_quarantine_rejected_vectors)
        # keeps semantic search degraded until the index is REBUILT, so the
        # reason is restated here. Checked independently of vectors.json, and
        # gated on the current index being COMPLETE rather than on that file
        # merely existing.
        if (self.vector_degrade_reason is None       # keep a more specific reason
                and not self._vector_index_complete()
                and self._rejected_vector_files()):
            self._note_vector_degrade(
                f"an earlier vector index was unusable and was set aside as "
                f"{_REJECTED_VECTORS} (nothing was deleted), and the current one "
                f"does not cover every chunk; using BM25 lexical retrieval only - "
                f"rebuild it with 'localm rag repair <name> --embed'",
                warn=True)

    def _rebuild_docs_from_chunks(self) -> None:
        """Replace the docs map with one rebuilt from the chunk sources."""
        rebuilt: dict = {}
        for c in self._chunks:
            src = c.get("source")
            if not src:
                continue
            entry = rebuilt.setdefault(src, {"chunks": 0})
            entry["chunks"] += 1
            if str(src).startswith("upload:"):
                entry["uploaded"] = True
        self._meta["docs"] = rebuilt

    def _build_norm_matrix(self) -> None:
        """Set ``self._norm_matrix`` to the row-normalised float32 matrix of the
        stored vectors, or None when numpy is unavailable, a stored vector does
        not have ``self._vec_dim`` components, or building the matrix fails."""
        self._norm_matrix = None
        if self._vectors and self._vec_dim and _st._numpy is not None:
            try:
                np = _st._numpy
                if all((not v) or len(v) == self._vec_dim for v in self._vectors):
                    mat = np.zeros((len(self._vectors), self._vec_dim), dtype="float32")
                    for idx, vec in enumerate(self._vectors):
                        if vec:
                            mat[idx] = vec
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    normalized = mat / norms
                    self._norm_matrix = np.where(np.isfinite(normalized), normalized, 0.0)
            except Exception:
                self._norm_matrix = None

    def _min_cached_nbytes(self) -> int:
        """A lower bound on this instance's cached size: chunk text plus 8
        bytes per stored vector component."""
        size = sum(len(c.get("text") or "") for c in self._chunks)
        if self._vectors is not None and self._vec_dim:
            size += len(self._vectors) * self._vec_dim * 8
        return size

    def _save(self) -> None:
        """Write meta.json, chunks.jsonl and vectors.json from this instance's
        state, as one change to the collection's files (``_files_changing``)."""
        with _files_changing(self.dir):
            self._write_files()
        self._snapshot_ref = None

    def _write_files(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))
        # dumps_lines escapes the line-break-alikes json.dumps(ensure_ascii=False)
        # would otherwise emit raw (U+0085/U+2028/U+2029), so a record cannot be
        # split in half by a line-oriented reader.
        self._atomic_write("chunks.jsonl", dumps_lines(_plain_chunks(self._chunks)))
        # meta.json and chunks.jsonl were just rewritten from this instance's own
        # in-memory state, which _load() only ever fills with well-formed
        # records, so both corruption flags are cleared here as well as in
        # _load().
        self.corrupt = False
        self.chunks_bad_lines = 0
        self._meta_unreadable = False
        # The fate of a REJECTED vectors.json is decided FIRST, before anything
        # below writes or unlinks that filename.
        if self._vectors_file_rejected:
            if self._chunks:
                self._quarantine_rejected_vectors()
            self._vectors_file_rejected = False
        # "Complete" means every chunk has a usable vector. Partial coverage
        # does not clear a set-aside sidecar's degrade.
        complete = self._vector_index_complete()
        if self._vectors is not None and any(v for v in self._vectors):
            self._vec_dim = _first_dim(self._vectors)
            self._atomic_write("vectors.json", json.dumps(
                {"dim": self._vec_dim, "vectors": _plain_rows(self._vectors)}))
        else:
            # Nothing usable to write. A REJECTED file was already moved out of
            # the way above.
            (self.dir / "vectors.json").unlink(missing_ok=True)
            self._vec_dim = None
        if not self._chunks:
            # Every document is gone. Stored vectors are positional against
            # chunks, so nothing is left to realign a set-aside sidecar to.
            self._discard_rejected_vectors("the collection no longer has any "
                                           "documents to realign them to")
            self.vector_degrade_reason = None
        elif complete:
            # Every chunk has a vector, so the degrade clears. The sidecar file
            # itself is KEPT.
            self.vector_degrade_reason = None
        elif self._rejected_vector_files():
            self.vector_degrade_reason = (
                f"an earlier vector index was unusable and was set aside as "
                f"{_REJECTED_VECTORS} (nothing was deleted), and the current one "
                f"does not cover every chunk; using BM25 lexical retrieval only - "
                f"rebuild it with 'localm rag repair <name> --embed'")
        else:
            self.vector_degrade_reason = None
        self._bm25 = None
        self._bm25_source = None
        # Cache the LISTING-relevant fields that are NOT otherwise persisted -
        # vector_degrade_reason and the vector-coverage math above - so a listing
        # can answer from meta.json alone, without reconstructing this Collection
        # at all (see peek_stats() / peek_detail() below).
        #
        # The cache carries a cheap (mtime_ns, size) fingerprint of chunks.jsonl
        # and vectors.json, taken AFTER they were written: peek_stats() /
        # peek_detail() stat (never read) both files again and refuse the cache
        # the instant either no longer matches, and fall back to a full load
        # whenever the cache is missing entirely.
        #
        # A second, small atomic write: the meta.json write at the top of this
        # method happens BEFORE vector_degrade_reason is finalised above.
        self._meta[_STATS_CACHE_KEY] = self._stats_cache_block()
        self._atomic_write("meta.json", json.dumps(self._meta, indent=2))

    def _stats_cache_block(self) -> dict:
        """The ``_stats_cache`` block for meta.json, computed from THIS
        instance's current in-memory state and a FRESH fingerprint of
        chunks.jsonl/vectors.json taken right now (see ``_file_fingerprint``).

        Caller must hold this collection's write lock; without that, the
        fingerprint could describe files a concurrent writer is mid-way
        through replacing."""
        return {
            "n_chunks": len(self._chunks),
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "vector_degrade_reason": self.vector_degrade_reason,
            "corrupt": self.corrupt,
            "chunks_bad_lines": self.chunks_bad_lines,
            "fingerprint": self._file_fingerprint(),
            # Lets a listing-time caller compare against the currently active
            # embedding model WITHOUT loading anything.
            "vector_dim": self._vec_dim,
        }

    def _vector_index_complete(self) -> bool:
        """True when every chunk currently has a usable vector.

        Stricter than "a vectors.json exists": partial coverage is not complete,
        and neither is an empty chunk list."""
        return bool(self._chunks) and (
            self._vectors is not None
            and len(self._vectors) == len(self._chunks)
            and all(v for v in self._vectors))

    def _save_meta(self) -> None:
        """Persist meta.json ONLY, leaving chunks.jsonl / vectors.json alone.

        For a change that touches nothing but metadata (a newly recorded root, a
        missing/restored flag). Chunks are untouched, so the cached BM25 index
        stays valid too."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with _files_changing(self.dir):
            self._atomic_write("meta.json", json.dumps(self._meta, indent=2))
        self._snapshot_ref = None

    def _atomic_write(self, filename: str, content: str) -> None:
        # storekit.atomic_write: unique temp name plus a Windows PermissionError
        # retry.
        _st._storekit_atomic_write(self.dir / filename, content)

    def _note_vector_degrade(self, reason: str, *, warn: bool) -> None:
        """Record WHY semantic (vector) scoring is unavailable and surface it once.

        A corrupt, stale, or dimensionally mismatched vectors index does not
        silently vanish into BM25-only: it is recorded (exposed via
        ``stats()``) and genuine corruption is logged.

        "Once" means once per PROCESS, not once per instance: ``_load`` runs
        from ``__init__`` and every request builds a FRESH Collection, so an
        instance-scoped guard would log the same sentence on every /api/rag
        call. The instance field is still set unconditionally, so ``stats()``
        and the GUI's "needs repair" state are unaffected; only the duplicate
        LOG LINE is suppressed. A genuinely NEW reason for the same collection
        still warns."""
        self.vector_degrade_reason = reason
        if not warn:
            return
        key = (str(self.dir), reason)
        if key in _WARNED_DEGRADES:
            return
        _WARNED_DEGRADES.add(key)
        _log.warning("RAG collection %r: %s", self.name, reason)

    def _file_fingerprint(self) -> dict:
        """(mtime_ns, size) for chunks.jsonl and vectors.json (None for
        either that does not exist) - a cheap (stat only, no content read)
        signature of the two files the ``_stats_cache`` block is derived
        from. Written by _save() right after both files were rewritten, and
        checked by ``_fingerprint_matches`` before the cache is ever trusted, so
        a file that changed WITHOUT going through this class is detected and the
        cache refused."""
        def _stat(name: str) -> "list[int] | None":
            try:
                st = (self.dir / name).stat()
            except OSError:
                return None
            return [st.st_mtime_ns, st.st_size]
        return {"chunks": _stat("chunks.jsonl"), "vectors": _stat("vectors.json")}
