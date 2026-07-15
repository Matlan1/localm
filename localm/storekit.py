# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared JSONL-store mechanics: atomic tmp+replace writes and a lazy
per-namespace lock registry.

``localm/rag/store.py`` and ``localm/memory/store.py`` each independently
hand-wrote this same low-level "atomic write + per-namespace RLock registry"
scaffolding (CF-9/CF-10) - deliberately, per both modules' own docstrings
("mirrors localm/rag/store.py") - but the two copies had already drifted:
rag's atomic write retried on ``PermissionError`` (a documented Windows
AV-lock workaround), memory's did not. RAG and Agent Memory stay correctly
separate FEATURES (different data, different lifecycles); only this layer
beneath both is shared. New, small, kernel-level - sibling to
``selfclient.py``/``bindhost.py``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def atomic_write(path: Path, data: str) -> None:
    """Write *data* to *path* atomically: write a unique per-writer temp
    file, then replace *path* with it.

    The temp name is unique per (pid, thread) so two independent writers
    never collide on the same temp file even outside a lock (e.g. a sidecar
    file write) - on top of, not instead of, each store's own per-namespace
    lock serializing writers to a given target. Retries the replace on
    ``PermissionError``: on Windows, an AV real-time scanner or the Search
    Indexer can transiently hold a handle to *path*, which would otherwise
    fail a good write (see PR #566 / the Windows os.replace-under-concurrent-
    open history this generalizes - previously only rag/store.py had it).

    The temp file is always cleaned up, so a permanently-failing replace does not
    leave one orphan behind per attempt (REG-631).
    """
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
    """Lazily-created ``threading.RLock`` per key, behind one shared creation
    guard.

    Both RAG (keyed by collection name) and Agent Memory (keyed by namespace
    hash) independently re-implemented this exact pattern to serialize
    concurrent writers to the SAME on-disk store: two request handlers each
    construct their own ``Collection``/``MemoryStore`` instance, ``_load()``
    the same on-disk state, mutate their own in-memory copy, and ``_save()``
    - last writer wins and the other update is silently lost unless writes to
    the same key are serialized. A per-instance lock cannot help (different
    objects); the lock must be keyed by name/namespace so it serializes
    process-wide. RLock so a locked method may call another locked method on
    the same key without deadlocking.
    """

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
