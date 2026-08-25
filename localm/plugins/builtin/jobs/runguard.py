# SPDX-License-Identifier: AGPL-3.0-or-later
"""A process-global guard that serialises job RUNS so two never execute at once."""

from __future__ import annotations

import threading
from contextlib import contextmanager

# A single non-reentrant lock shared by every job-run entry point in this process.
_RUN_LOCK = threading.Lock()


def is_running() -> bool:
    """True when a job run currently holds the guard."""
    if _RUN_LOCK.acquire(blocking=False):
        _RUN_LOCK.release()
        return False
    return True


@contextmanager
def run_slot():
    """Yield True when this caller acquired the run slot (and must run), or False when a run is already in flight (the caller should skip, not stack a second run)."""
    acquired = _RUN_LOCK.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            _RUN_LOCK.release()
