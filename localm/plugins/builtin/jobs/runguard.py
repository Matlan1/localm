# SPDX-License-Identifier: AGPL-3.0-or-later
"""A process-global guard that serialises job RUNS so two never execute at once.

A scheduled tick and a GUI "run now" both load a model into VRAM. When a job's
interval is shorter than its run time (60s interval, a 15-58s run) the runs overlap,
the second model load stacks on the first, and the GPU OOMs - the repeated
cudaMalloc / ROCm failures the maintainer hit (U-4). This non-reentrant lock lets
each entry point ask "is a run already in flight?" and SKIP rather than stack:

  - the scheduler tick skips its due jobs (they are due again next tick);
  - the "run now" route returns a clear busy response instead of loading a model.

It is PROCESS-LOCAL: it coordinates the server's scheduler loop and its route
handlers (the reported case, all in the one server process). It does not coordinate a
separate ``localm job run`` CLI process against a running server - that cross-process
case is out of scope here.
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
