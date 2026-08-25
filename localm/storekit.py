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
        # Removes the temp file when the replace gave up; a no-op on success.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            # Best-effort: the write's outcome is already decided.
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
