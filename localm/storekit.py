# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared JSONL-store mechanics: atomic tmp+replace writes and a lazy
per-namespace lock registry.

Used by ``localm/rag/store.py`` and ``localm/memory/store.py``. RAG and Agent
Memory stay separate FEATURES (different data, different lifecycles); only this
layer beneath both is shared.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional


def atomic_write(path: Path, data: str | Iterable[str]) -> None:
    """Write *data* to *path* atomically: write a unique per-writer temp
    file, then replace *path* with it. *data* is a string, or an iterable of
    strings written one after another.

    The temp name is unique per (pid, thread), so two independent writers
    never collide on the same temp file even outside a lock (e.g. a sidecar
    file write) - on top of, not instead of, each store's own per-namespace
    lock serializing writers to a given target. Retries the replace on
    ``PermissionError``: on Windows, an AV real-time scanner or the Search
    Indexer can transiently hold a handle to *path*.

    The temp file is always cleaned up, so a permanently-failing replace leaves
    no orphan behind.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        if isinstance(data, str):
            tmp.write_text(data, encoding="utf-8")
        else:
            with open(tmp, "w", encoding="utf-8") as fh:
                for piece in data:
                    fh.write(piece)
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        # Removes the temp file when the write or the replace failed; a no-op
        # on success.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            # Best-effort: the write's outcome is already decided.
            from localm.debuglog import logger
            logger.debug("storekit: could not remove temp file %s (%s)", tmp, e)


def read_json_checked(path: Path, default):
    """``(value, read_ok)`` for the JSON store at *path*. ``read_ok`` is True
    only when *path* itself parsed or no file exists; it is False when a file is
    present and *path* could not be read, whether the ``.bak`` then parsed (the
    value is the previous generation) or not (the value is *default*)."""
    from localm import config
    return config._read_json_checked(path, default)


def read_for_write(path: Path, default, shape: type, refusal: str):
    """The store at *path* for a read-modify-write: the parsed value, which has
    type *shape*. Raises ``config.ConfigUnreadable`` with the message *refusal*
    when a file is present but *path* could not be read, or it parses to
    another type, because writing back would replace the store with an older
    generation or with *default*. The caller holds :func:`store_lock` for the
    whole read-modify-write."""
    from localm import config
    data, read_ok = config._read_json_checked(path, default)
    if not read_ok or not isinstance(data, shape):
        raise config.ConfigUnreadable(refusal)
    return data


def store_lock(path: Path, *, timeout: Optional[float] = None):
    """Context manager holding the cross-process lock on the store at *path*.
    A holder proven running keeps the lock however long it holds it; one proven
    gone is reclaimed at once. Raises ``config.ConfigLockTimeout`` when another
    process holds it for the whole wait (*timeout* seconds; None for the default
    budget), writing nothing. Yields a handle with ``assert_held()``."""
    from localm import config
    return config._cross_process_lock(path, timeout=timeout)


def write_private(path: Path, text: str, *, retrying: bool = True) -> bool:
    """Write *text* to the store at *path* atomically and owner-restricted, via
    a temp file unique to the writer. Re-checks this process's
    :func:`store_lock` on *path* immediately before the replace, raising
    ``config.ConfigLockLost`` (nothing written) when it was taken over. Returns
    whether the pre-rename restriction succeeded. *retrying* rides out a
    transient Windows sharing violation on the replace."""
    from localm import config
    config.assert_store_lock_held(path)
    return config.atomic_write_private(path, text, retrying=retrying)


class NamespaceLockRegistry:
    """Lazily-created ``threading.RLock`` per key, behind one shared creation
    guard.

    Serializes concurrent writers to the SAME on-disk store: two request
    handlers each construct their own ``Collection``/``MemoryStore`` instance,
    ``_load()`` the same state, mutate their own copy and ``_save()``. The lock
    is keyed by name/namespace, not per instance, so it serializes
    process-wide. RLock, so a locked method may call another locked method on
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
