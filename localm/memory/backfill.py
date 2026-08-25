# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fill in missing memory vectors across every namespace, to completion."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

# A namespace can be large, and each pass re-saves the whole store. This caps the
# total work one invocation will do so a pathological store cannot hang a CLI
# command forever; the remainder is picked up by the next run. Deliberately far
# above any realistic memory store - it is a stop, not a budget.
_MAX_PASSES_PER_NS = 400


def _namespaces(root: Path):
    """Every namespace store file under *root*."""
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.jsonl"))


def backfill_all(root: Path, embed_fn: Optional[Callable], *,
                 on_progress: Optional[Callable] = None) -> dict:
    """Embed every vectorless record in every namespace under *root*."""
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
    """Count vectorless records across every namespace under *root*, and count namespaces that could not even be opened (corrupt JSONL, a locked or unreadable file)."""
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
    """How many records across every namespace still lack a vector."""
    total, _unreadable = vectorless_scan(root)
    return total
