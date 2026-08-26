# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fill in missing memory vectors across every namespace, to completion.

Walks every namespace under the memory root and drives the bounded
``MemoryStore.backfill_vectors`` (64 records per call) until nothing is left to
embed, so a caller that needs completion - `setup-embeddings` - gets it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

# A namespace can be large, and each pass re-saves the whole store. This caps the
# total work one invocation does; the remainder is picked up by the next run. It
# is a stop, not a budget.
_MAX_PASSES_PER_NS = 400


def _namespaces(root: Path):
    """Every namespace store file under *root*.

    Layout is ``<root>/<agent>/<ns_hash>.jsonl`` (see store.namespace_file). The
    sidecars that live beside a store - the episodic watermark, the forgotten
    archive, the pending corrections - are JSON, not JSONL, so matching the
    store extension excludes them by construction.
    """
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.jsonl"))


def backfill_all(root: Path, embed_fn: Optional[Callable], *,
                 on_progress: Optional[Callable] = None) -> dict:
    """Embed every vectorless record in every namespace under *root*.

    Returns ``{"namespaces": n, "embedded": n, "remaining": n, "unreadable": n}``.
    ``remaining`` is non-zero only when a namespace hit ``_MAX_PASSES_PER_NS`` or
    an embed call kept failing, so a caller can report a shortfall instead of
    claiming completion. ``unreadable`` counts namespaces that could not even be
    opened (corrupt JSONL, a locked file); it is separate from ``remaining``
    because such a namespace's true vectorless count is unknown, not zero.
    """
    if embed_fn is None:
        return {"namespaces": 0, "embedded": 0, "remaining": 0, "unreadable": 0}

    from localm.memory.store import MemoryStore

    embedded = 0
    remaining = 0
    unreadable = 0
    seen = 0
    for path in _namespaces(root):
        try:
            store = MemoryStore.open_file(path)
        except Exception as e:
            unreadable += 1
            from localm.debuglog import logger as _dbg
            _dbg.warning("memory backfill: could not open namespace %s (%s)",
                         path, e)
            continue
        seen += 1
        for _ in range(_MAX_PASSES_PER_NS):
            try:
                filled = store.backfill_vectors(embed_fn)
            except Exception:
                # One namespace failing must not abandon the rest; the shortfall
                # shows up in "remaining" rather than as a silent success.
                break
            if not filled:
                break
            embedded += filled
            if on_progress:
                on_progress(embedded)
        remaining += store.vectorless_count()
    return {"namespaces": seen, "embedded": embedded, "remaining": remaining,
            "unreadable": unreadable}


def vectorless_scan(root: Path) -> tuple[int, int]:
    """Count vectorless records across every namespace under *root*, and count
    namespaces that could not even be opened (corrupt JSONL, a locked or
    unreadable file). Returns ``(total, unreadable)``.

    An unreadable namespace contributes 0 to ``total``, exactly as a fully
    embedded namespace does, so the two counts are returned separately and a
    caller can tell them apart.
    """
    from localm.memory.store import MemoryStore

    total = 0
    unreadable = 0
    for path in _namespaces(root):
        try:
            total += MemoryStore.open_file(path).vectorless_count()
        except Exception as e:
            unreadable += 1
            from localm.debuglog import logger as _dbg
            _dbg.warning("memory backfill: could not open namespace %s (%s)",
                         path, e)
    return total, unreadable


def vectorless_total(root: Path) -> int:
    """How many records across every namespace still lack a vector. Used to
    decide whether a backfill is worth announcing, and to report honestly."""
    total, _unreadable = vectorless_scan(root)
    return total
