# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared JSONL-store mechanics: atomic tmp+replace writes and a lazy per-namespace lock registry."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def atomic_write(path: Path, data: str) -> None:
    """Write *data* to *path* atomically: write a unique per-writer temp file, then replace *path* with it."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    try:
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        # On success the replace consumed tmp, so this is a no-op. On a give-up
        # (or any other error) it stops ONE ORPHAN ACCUMULATING PER FAILURE next
        # to the target (REG-631) - the give-up itself still raises, because a
        # write that did not happen must never look like one that did.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            # Best-effort by design: the write's outcome is already decided, and
            # raising here would turn a stray temp file into a broken write.
            # Logged, not silenced (rule 5), so a persistent leak is discoverable.
            from localm.debuglog import logger
            logger.debug("storekit: could not remove temp file %s (%s)", tmp, e)


class NamespaceLockRegistry:
    """Lazily-created ``threading.RLock`` per key, behind one shared creation guard."""

    def __init__(self) -> None:
        self._locks: dict = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock
