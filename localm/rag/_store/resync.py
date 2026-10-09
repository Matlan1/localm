# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-syncing a collection with the folders it was indexed from."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from localm.rag import store as _st

from .confine import _path_within
from .types import ProgressFn


class _CollectionResync:
    """Mixin of ``Collection``: re-walking the indexed folders and reconciling
    vanished documents."""

    def _resync_locked(self, *, embed_fn, classify_fn, describe_image_fn,
                       on_progress, policy, force, prune_missing,
                       model_name=None) -> dict:
        """The resync body. MUST run under _collection_lock after _load()."""
        say = on_progress or (lambda _t: None)
        available, unavailable, blocked = self._partition_roots(policy, say)
        # Roots we could not judge: nothing under them is indexed, flagged, or
        # pruned this run.
        skipped_roots = [Path(r["root"]) for r in (unavailable + blocked)]

        targets: list = list(available)
        targets.extend(self._resyncable_files(skipped_roots, policy, say))

        # Snapshot the flagged-missing set BEFORE indexing: re-indexing a document
        # REPLACES its docs entry wholesale (_add_paths_locked), dropping the
        # flag.
        docs_before = self._meta.get("docs", {})
        was_missing = {k for k, e in docs_before.items()
                       if isinstance(e, dict) and e.get("missing")}

        if targets:
            result = self._add_paths_locked(
                targets, embed_fn=embed_fn, classify_fn=classify_fn,
                describe_image_fn=describe_image_fn, on_progress=on_progress,
                policy=policy, force=force, model_name=model_name)
        else:
            result = {"added": 0, "updated": 0, "skipped": 0, "failed": [],
                      "chunks": len(self._chunks)}

        missing, restored, pruned = self._reconcile_missing(
            skipped_roots, was_missing=was_missing,
            prune_missing=prune_missing, say=say)
        if pruned:
            self._save()            # chunks and vectors changed
        elif missing or restored:
            self._save_meta()       # only flags changed - see _save_meta

        docs = self._meta.get("docs", {})
        result.update({
            # Re-read AFTER the reconcile: pruning drops chunks, so the count
            # _add_paths_locked returned is stale by then.
            "chunks": len(self._chunks),
            "roots": self.roots(),
            "unavailable_roots": unavailable,
            "blocked_roots": blocked,
            "missing": missing,
            "restored": restored,
            "pruned": pruned,
            "missing_total": sum(
                1 for e in docs.values()
                if isinstance(e, dict) and e.get("missing")),
            # Why semantic search is degraded, AFTER this run (None when it is
            # fine).
            "vector_degrade_reason": self.vector_degrade_reason,
        })
        return result

    def _partition_roots(self, policy: Optional[dict], say: ProgressFn):
        """Split the persisted roots into (available, unavailable, blocked).

        Availability is checked FIRST and reported, never assumed. ``is_dir()``
        answers most of that; see ``_unmounted_reason`` for the case it cannot
        see."""
        available: list = []
        unavailable: list = []
        blocked: list = []
        for raw in self.roots():
            root = Path(raw)
            if not root.is_dir():
                # is_dir() is False for gone, unreadable, AND replaced-by-a-file.
                # The response is the same for all three (skip whole, touch
                # nothing); this branches only to report the right reason.
                reason = ("the indexed folder is now a file, not a directory"
                          if root.exists() else
                          "the indexed folder is not available (deleted, "
                          "unmounted, or unreadable)")
                unavailable.append({"root": raw, "reason": reason})
                say(f"skipping {raw}: {reason} - nothing under it was changed")
                continue
            reason = self._unmounted_reason(root)
            if reason:
                unavailable.append({"root": raw, "reason": reason})
                say(f"skipping {raw}: {reason} - nothing under it was changed")
                continue
            if policy is not None:
                try:
                    _st.confine_index_path(root, policy)
                except ValueError as e:
                    blocked.append({"root": raw, "reason": str(e)})
                    say(f"skipping {raw}: {e}")
                    continue
            available.append(root)
        return available, unavailable, blocked

    def _unmounted_reason(self, root: Path) -> Optional[str]:
        """Why *root* looks like an UNMOUNTED mount point, or None if it is fine.

        ``is_dir()`` cannot see this on POSIX: unmounting leaves the mount point
        behind as an ordinary, existing, EMPTY directory.

        All three conditions are required and none is sufficient alone: *root*
        is a mount point, it is empty, and this collection holds documents
        indexed under it.
        """
        try:
            if not os.path.ismount(root):
                return None
            if next(root.iterdir(), None) is not None:
                return None
        except OSError:
            # Could not look inside at all: reported as unavailable, so nothing
            # under this root is touched.
            return ("the indexed folder could not be read (disconnected, or "
                    "permission denied)")
        if not self._has_docs_under(root):
            return None
        return ("the indexed folder is an empty mount point, so its drive or "
                "share appears to be unmounted")

    def _has_docs_under(self, root: Path) -> bool:
        """True when at least one indexed document's source lives under *root*.

        Uploads are excluded: an ``upload:`` key is not a filesystem path, so it
        can neither be under a root nor be evidence that one lost its contents."""
        return any(
            not str(key).startswith("upload:") and _path_within(Path(key), root)
            for key in self._meta.get("docs", {})
        )

    def _resyncable_files(self, skipped_roots: list, policy: Optional[dict],
                          say: ProgressFn) -> list:
        """Existing document source files that are safe to re-index this run.

        Uploads have no source file (``upload:`` keys) and are skipped. So is
        anything under a skipped root, and anything the current policy refuses;
        the last of those is filtered HERE rather than handed to
        ``_add_paths_locked``, whose top-level confinement check raises."""
        out: list = []
        for key in sorted(self._meta.get("docs", {})):
            if str(key).startswith("upload:"):
                continue
            p = Path(key)
            if any(_path_within(p, r) for r in skipped_roots):
                continue
            try:
                if not p.is_file():
                    continue        # gone: the missing pass decides what to do
            except OSError:
                continue
            if policy is not None:
                try:
                    _st.confine_index_path(p, policy)
                except ValueError as e:
                    say(f"skipping {key}: {e}")
                    continue
            out.append(p)
        return out

    def _reconcile_missing(self, skipped_roots: list, *, was_missing: set,
                           prune_missing: bool, say: ProgressFn):
        """Flag documents whose file has vanished, clear the flag on ones that
        came back, and prune only when explicitly asked. Returns
        (newly_missing, restored, pruned) as lists of doc keys.

        *was_missing* is the flagged set as it stood BEFORE this run indexed
        anything; a document that came back CHANGED has already been re-indexed,
        which rewrites its entry and drops the flag.

        Mutates ``self._meta`` / ``self._chunks`` in place; the caller saves."""
        docs = self._meta.get("docs", {})
        newly_missing: list = []
        restored: list = []
        pruned: list = []
        for key in sorted(docs):
            entry = docs.get(key)
            if not isinstance(entry, dict) or str(key).startswith("upload:"):
                continue
            p = Path(key)
            if any(_path_within(p, r) for r in skipped_roots):
                continue        # unreachable root: no verdict, no change
            try:
                present = p.exists()
            except (OSError, ValueError):
                # Could not ask at all: treated as present, so an unanswerable
                # question is never resolved in the destructive direction.
                present = True
            if present:
                entry.pop("missing", None)
                entry.pop("missing_since", None)
                if key in was_missing:
                    restored.append(key)
                    say(f"back: {key}")
                continue
            if prune_missing:
                pruned.append(key)
            elif not entry.get("missing"):
                entry["missing"] = True
                entry["missing_since"] = time.time()
                newly_missing.append(key)
                say(f"missing: {key} (kept in the index, flagged)")
        if pruned:
            drop = set(pruned)
            keep = [i for i, c in enumerate(self._chunks)
                    if c.get("source") not in drop]
            self._chunks = [self._chunks[i] for i in keep]
            if self._vectors is not None:
                self._vectors = [self._vectors[i] for i in keep]
            for key in pruned:
                docs.pop(key, None)
                say(f"pruned: {key} (file is gone)")
        return newly_missing, restored, pruned
