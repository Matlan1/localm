# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading a collection's state: stats, documents and roots, the
meta.json-only path that answers without a full load, and confinement
checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from localm.rag import store as _st

from ..collection_lock import (CollectionLockedError, collection_write_lock,
                               lock_path_for)
from .cache import _files_changing
from .confine import _path_within
from .files import _STATS_CACHE_KEY

if TYPE_CHECKING:
    from localm.rag.store import Collection


class _CollectionIntrospection:
    """Mixin of ``Collection``: stats, documents, roots, the meta.json-only path and
    confinement checks."""

    def roots(self) -> list:
        """The folders indexed into this collection, resolved and sorted.

        These are what ``resync`` re-walks. Empty for a collection built only
        from individually named files or uploads, and for one whose meta.json was
        corrupt; re-add the folder to restore them."""
        return self._roots_from_meta(self._meta)

    @staticmethod
    def _roots_from_meta(meta: dict) -> list:
        roots = meta.get("roots")
        return sorted(roots) if isinstance(roots, dict) else []

    def documents(self) -> list:
        """The source paths currently indexed in this collection (for repair)."""
        return list(self._meta.get("docs", {}).keys())

    @staticmethod
    def _has_vectors(chunks: list, vectors: Optional[list]) -> bool:
        """"Has vectors" = whether query() will actually blend embeddings: the
        same >=80% coverage threshold _vector_scores uses, NOT "every chunk
        embedded". A partially-embedded collection (80-99%) still does hybrid
        retrieval and is reported as having vectors.

        Shared with _save(), which caches the same value into meta.json."""
        present = [v for v in (vectors or []) if v]
        return bool(present) and len(present) >= 0.8 * len(chunks)

    def stats(self) -> dict:
        docs = self._meta.get("docs", {})
        return {
            "name": self.name,
            "created": self._meta.get("created"),
            "n_docs": len(docs),
            # Indexed documents whose source file was gone at the last resync.
            # Still counted in n_docs and still searchable: the flag says the
            # index is ahead of the disk, it does not remove anything.
            "n_missing": sum(1 for e in docs.values()
                             if isinstance(e, dict) and e.get("missing")),
            "n_roots": len(self.roots()),
            "n_chunks": len(self._chunks),
            "has_vectors": self._has_vectors(self._chunks, self._vectors),
            "corrupt": self.corrupt,
            # Count of chunks.jsonl lines _load() had to skip (0 if none), so a
            # caller can name a count instead of a generic "index damaged".
            "chunks_bad_lines": self.chunks_bad_lines,
            # Why semantic search fell back to BM25 (None when vectors are used or
            # legitimately absent).
            "vector_degrade_reason": self.vector_degrade_reason,
            "vector_dim": self._vec_dim,
        }

    def vector_dim(self) -> Optional[int]:
        """The dimensionality of THIS collection's currently stored vectors, or
        None when it cannot be established: no usable vectors are stored at all
        (see ``stats()["has_vectors"]``), or ``_load()`` found the file present
        but unusable for a reason ``vector_degrade_reason`` names.

        Reads the same ``_vec_dim`` the add-time consistency guard trusts.
        ``_load()`` computes it as ``data.get("dim") or _first_dim(vectors)``,
        so a legacy vectors.json without a "dim" field still resolves it from
        the first stored vector; only an unusable/empty index gives None."""
        return self._vec_dim

    def docs(self) -> list[dict]:
        return self._docs_from_meta(self._meta)

    @staticmethod
    def _docs_from_meta(meta: dict) -> list[dict]:
        return [
            {"path": path, **info}
            for path, info in sorted(meta.get("docs", {}).items())
        ]

    @staticmethod
    def _fingerprint_matches(coll_dir: Path, recorded) -> bool:
        """True when *recorded* (the cache's "fingerprint" value) still
        matches chunks.jsonl and vectors.json on disk RIGHT NOW. Duplicates
        the same two os.stat() calls as _file_fingerprint(); this one runs from
        peek_stats() BEFORE any Collection exists."""
        if not isinstance(recorded, dict):
            return False
        def _stat(name: str) -> list[int] | None:
            try:
                st = (coll_dir / name).stat()
            except OSError:
                return None
            return [st.st_mtime_ns, st.st_size]
        return (_stat("chunks.jsonl") == recorded.get("chunks")
                and _stat("vectors.json") == recorded.get("vectors"))

    @classmethod
    def _peek_meta(cls, name: str, base: Optional[Path] = None
                    ) -> tuple[str, Path, dict] | None:
        """(checked name, collection dir, parsed meta.json), or None when
        there is nothing here the lazy path can trust enough to skip the full
        load: an invalid name, no meta.json, or one that fails to parse. On None
        the caller falls back to the real ``Collection(name)``, which carries
        the corrupt-meta.json recovery."""
        try:
            checked_name = _st._check_name(name)
        except ValueError:
            return None
        coll_dir = (base or _st.rag_dir()) / checked_name
        meta_path = coll_dir / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, RecursionError):
            return None
        if not isinstance(meta, dict):
            return None
        return checked_name, coll_dir, meta

    @classmethod
    def _stats_from_meta(cls, checked_name: str, coll_dir: Path,
                         meta: dict) -> Optional[dict]:
        """stats()-shaped dict from an already-parsed meta.json, using ONLY the
        cache _save() writes - never a re-derivation of
        has_vectors/vector_degrade_reason from a partial read of chunks.jsonl /
        vectors.json. None when the cache is absent, its recorded file
        fingerprint no longer matches chunks.jsonl / vectors.json on disk (see
        ``_fingerprint_matches``), or it does not look like this class wrote it;
        the caller must then fall back to the full, authoritative
        Collection(name).stats()."""
        cache = meta.get(_STATS_CACHE_KEY)
        if not isinstance(cache, dict) or not isinstance(cache.get("n_chunks"), int):
            return None
        if not cls._fingerprint_matches(coll_dir, cache.get("fingerprint")):
            return None
        docs = meta.get("docs", {})
        if not isinstance(docs, dict):
            return None
        return {
            "name": checked_name,
            "created": meta.get("created"),
            "n_docs": len(docs),
            "n_missing": sum(1 for e in docs.values()
                             if isinstance(e, dict) and e.get("missing")),
            "n_roots": len(cls._roots_from_meta(meta)),
            "n_chunks": cache["n_chunks"],
            "has_vectors": bool(cache.get("has_vectors")),
            "corrupt": bool(cache.get("corrupt")),
            # 0 when the cache does not carry this field; the caller then
            # falls back to generic "index damaged" wording instead of a count.
            "chunks_bad_lines": cache.get("chunks_bad_lines", 0),
            "vector_degrade_reason": cache.get("vector_degrade_reason"),
            # Absent when the cache does not carry this field; the collection
            # then falls back to the cold load-and-backfill path below once.
            "vector_dim": cache.get("vector_dim"),
        }

    @classmethod
    def peek_stats(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``stats()`` without constructing a full ``Collection`` - reads
        meta.json alone and trusts its cached derived fields, never
        chunks.jsonl or vectors.json. None means "cannot answer cheaply and
        correctly" (see ``_stats_from_meta``); the caller MUST fall back to
        the real ``Collection(name).stats()`` in that case."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        return cls._stats_from_meta(checked_name, coll_dir, meta)

    @classmethod
    def peek_detail(cls, name: str, base: Optional[Path] = None) -> Optional[dict]:
        """``peek_stats()`` plus the docs list, for the collection-detail route -
        both read meta.json exactly once. Same None contract as peek_stats()."""
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        checked_name, coll_dir, meta = found
        stats = cls._stats_from_meta(checked_name, coll_dir, meta)
        if stats is None:
            return None
        return {**stats, "docs": cls._docs_from_meta(meta)}

    @staticmethod
    def _doc_is_host_path(doc_key: str) -> bool:
        """False for a doc key ``add_uploads`` records (``upload:<filename>``,
        no host filesystem path behind it); True for a doc key ``add_paths``
        records (an absolute, resolved host path)."""
        return not doc_key.startswith("upload:")

    @classmethod
    def _docs_within_roots(cls, doc_keys, key_roots: list) -> bool:
        """True when every host-filesystem doc key in *doc_keys* resolves
        under one of *key_roots* (``_path_within``, both sides resolved).
        Upload-recorded keys (see ``_doc_is_host_path``) are skipped. An
        empty *key_roots*, or one whose entries all fail to resolve to a
        real path, returns True and False respectively."""
        if not key_roots:
            return True
        roots: list[Path] = []
        for r in key_roots:
            try:
                roots.append(Path(r).expanduser().resolve())
            except (OSError, ValueError):
                continue
        if not roots:
            return False
        for key in doc_keys:
            if not cls._doc_is_host_path(key):
                continue
            if not any(_path_within(Path(key), r) for r in roots):
                return False
        return True

    @classmethod
    def confined_to(cls, name: str, key_roots: list, base: Optional[Path] = None
                    ) -> Optional[bool]:
        """Whether every host-filesystem document indexed into collection
        *name* resolves under one of *key_roots*. Reads meta.json only, the
        same cheap path ``peek_stats``/``peek_detail`` use.

        An empty *key_roots* always returns True. Otherwise: True/False from
        ``_docs_within_roots`` over the collection's recorded doc keys, or
        None when meta.json cannot be read or parsed (missing, invalid JSON,
        or a "docs" field that is not an object). A caller enforcing
        confinement treats None the same as False."""
        if not key_roots:
            return True
        found = cls._peek_meta(name, base)
        if found is None:
            return None
        _checked_name, _coll_dir, meta = found
        docs = meta.get("docs", {})
        if not isinstance(docs, dict):
            return None
        return cls._docs_within_roots(docs.keys(), key_roots)

    def is_confined_to(self, key_roots: list) -> bool:
        """Whether every host-filesystem document THIS loaded instance can
        serve resolves under one of *key_roots*: every key of its docs map and
        the source of every chunk it holds, so the answer describes exactly
        what ``query()`` returns from this instance.

        An empty *key_roots* returns True. False when this instance's meta.json
        could not be read or its "docs" field is not an object, the cases
        ``confined_to`` answers None."""
        if not key_roots:
            return True
        docs = self._meta.get("docs", {})
        if self._meta_unreadable or not isinstance(docs, dict):
            return False
        sources = set(docs)
        sources.update(str(c["source"]) for c in self._chunks if c.get("source"))
        return self._docs_within_roots(sources, key_roots)

    @classmethod
    def load_and_maybe_backfill(cls, name: str, base: Optional[Path] = None
                                ) -> Collection:
        """The COLD-fallback path for ``peek_stats()``/``peek_detail()``: a
        full, authoritative load of *name* straight from disk (``Collection(
        name, base, cache=False)``, so it never fills the collection cache),
        with an opportunistic attempt to backfill its ``_stats_cache`` so
        future listings of this SAME collection stop paying the full-load cost.

        ORDERING: the write lock is acquired FIRST and the load happens INSIDE
        it - never load-then-lock. Nothing else can write to this collection
        while the lock is held (every real writer takes the SAME file lock
        before touching disk), so what ``_load()`` reads is current and the
        fingerprint taken from that same held state describes exactly what was
        read.

        Takes ONLY the cross-process file lock, not the in-process
        ``_collection_lock``: this method never mutates anything, it only
        re-derives a cache from bytes it read under the file lock.

        ``timeout=0``: an opportunistic path serving a READ. On a busy lock this
        returns a fully loaded, fully correct ``Collection`` without writing a
        cache."""
        base = base or _st.rag_dir()
        checked_name = _st._check_name(name)
        coll_dir = base / checked_name
        try:
            with collection_write_lock(
                    lock_path_for(coll_dir), collection=checked_name,
                    op="a stats-cache backfill", timeout=0):
                # _load() runs INSIDE the lock, from disk.
                coll = cls(checked_name, base, cache=False)
                if coll.exists():
                    coll._meta[_STATS_CACHE_KEY] = coll._stats_cache_block()
                    with _files_changing(coll.dir):
                        coll._atomic_write("meta.json", json.dumps(coll._meta, indent=2))
                return coll
        except CollectionLockedError:
            # Busy: full load from disk, no stats-cache write.
            return cls(checked_name, base, cache=False)
