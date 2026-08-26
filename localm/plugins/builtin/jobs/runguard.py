# SPDX-License-Identifier: AGPL-3.0-or-later
"""A process-global guard that serialises job RUNS so two never execute at once.

A scheduled tick and a GUI "run now" both load a model into VRAM, so overlapping
runs stack a second model load on the first and the GPU OOMs. This non-reentrant
lock lets each entry point ask "is a run already in flight?" and SKIP rather than
stack:

  - the scheduler tick skips its due jobs (they are due again next tick);
  - the "run now" route returns a clear busy response instead of loading a model.

It is PROCESS-LOCAL: it coordinates the server's scheduler loop and its route
handlers. It does not coordinate a separate ``localm job run`` CLI process
against a running server.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

# A single non-reentrant lock shared by every job-run entry point in this process.
_RUN_LOCK = threading.Lock()


def is_running() -> bool:
    """True when a job run currently holds the guard. Best-effort snapshot (for
    status/diagnostics); the real serialisation is :func:`run_slot`'s atomic acquire."""
    if _RUN_LOCK.acquire(blocking=False):
        _RUN_LOCK.release()
        return False
    return True


@contextmanager
def run_slot():
    """Yield True when this caller acquired the run slot (and must run), or False
    when a run is already in flight (the caller should skip, not stack a second run).
    Releases the slot on exit only when it was acquired here."""
    acquired = _RUN_LOCK.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            _RUN_LOCK.release()
