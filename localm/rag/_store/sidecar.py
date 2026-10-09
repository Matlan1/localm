# SPDX-License-Identifier: AGPL-3.0-or-later
"""Set-aside vector indexes: a vectors.json that could not be used is kept
as ``vectors.json.rejected[.N]``."""

from __future__ import annotations

import os

from localm.debuglog import logger as _log


# Where a vectors.json that _load() REFUSED is set aside when the chunks it was
# (mis)aligned with get rewritten. Preserved, never deleted.
_REJECTED_VECTORS = "vectors.json.rejected"


#: How many set-aside vector indexes to KEEP per collection. Older ones are
#: deleted with a warning.
_MAX_REJECTED_KEPT = 3


class _CollectionSidecar:
    """Mixin of ``Collection``: setting aside, pruning and discarding rejected
    vectors.json files."""

    def _rejected_vector_files(self) -> list:
        """Every set-aside vectors sidecar, oldest name first."""
        try:
            return sorted(p for p in self.dir.glob(_REJECTED_VECTORS + "*")
                          if p.is_file())
        except OSError:
            return []

    def _discard_rejected_vectors(self, why: str) -> None:
        """Delete set-aside sidecars, saying why.

        The ONLY place they are ever removed. Announced at warning level."""
        for p in self._rejected_vector_files():
            try:
                p.unlink()
            except OSError as e:
                _log.warning("RAG collection %r: could not remove %s (%s); it is "
                             "left in place", self.name, p.name, e)
                continue
            _log.warning("RAG collection %r: removed the set-aside vector index "
                         "%s - %s.", self.name, p.name, why)

    def _quarantine_rejected_vectors(self) -> None:
        """Set a rejected vectors.json aside as ``vectors.json.rejected``.

        The bytes stay on disk for recovery, and no loader will ever pair them
        with chunks again. ``_load`` reports the set-aside file as a degrade for
        as long as it exists."""
        src = self.dir / "vectors.json"
        if not src.is_file():
            return
        dest = self._free_rejected_name()
        if dest is None:
            _log.warning(
                "RAG collection %r: an unusable vectors.json could not be set "
                "aside because %s and its numbered siblings all exist; it is "
                "left in place so nothing is overwritten. Rebuild the index "
                "('localm rag repair %s --embed') or clear the old .rejected "
                "files by hand.", self.name, _REJECTED_VECTORS, self.name)
            return
        try:
            os.replace(src, dest)
        except OSError as e:
            # Best-effort; the failure is logged and the file is left in place.
            _log.warning("RAG collection %r: could not set the unusable "
                         "vectors.json aside as %s (%s); it is left in place",
                         self.name, dest.name, e)
            return
        _log.warning("RAG collection %r: the unusable vectors.json was set aside "
                     "as %s (%s). Nothing was deleted; re-embed with "
                     "'localm rag reembed %s' (no source files needed) or rebuild "
                     "from source with 'localm rag repair %s --embed'.",
                     self.name, dest.name,
                     self.vector_degrade_reason or "unusable",
                     self.name, self.name)
        self._prune_rejected_vectors()

    def _prune_rejected_vectors(self) -> None:
        """Keep only the newest ``_MAX_REJECTED_KEPT`` set-aside indexes.

        Each set-aside file is a full copy of the vector index.

        Ordered by MTIME, not by name: ``_rejected_vector_files`` sorts
        lexicographically, which puts ``.rejected.20`` before ``.rejected.3``.
        Deletion is announced at WARNING level."""
        files = self._rejected_vector_files()
        if len(files) <= _MAX_REJECTED_KEPT:
            return
        try:
            by_age = sorted(files, key=lambda p: p.stat().st_mtime)
        except OSError as e:
            _log.warning("RAG collection %r: could not order the set-aside vector "
                         "indexes to prune them (%s); all are left in place",
                         self.name, e)
            return
        for p in by_age[:-_MAX_REJECTED_KEPT]:
            try:
                size = p.stat().st_size
                p.unlink()
            except OSError as e:
                _log.warning("RAG collection %r: could not prune %s (%s); it is "
                             "left in place", self.name, p.name, e)
                continue
            _log.warning("RAG collection %r: pruned the oldest set-aside vector "
                         "index %s (%.1f MB) - keeping the newest %d. The current "
                         "index is unaffected.",
                         self.name, p.name, size / 1048576.0, _MAX_REJECTED_KEPT)

    def _free_rejected_name(self):
        """An unused ``vectors.json.rejected[.N]`` path, or None if there is none.

        ``os.replace`` overwrites its destination, so the name is numbered and
        every preserved copy is kept. Past the cap (20) None is returned and the
        caller leaves the file where it is."""
        first = self.dir / _REJECTED_VECTORS
        if not first.exists():
            return first
        for n in range(2, 21):
            candidate = self.dir / f"{_REJECTED_VECTORS}.{n}"
            if not candidate.exists():
                return candidate
        return None
