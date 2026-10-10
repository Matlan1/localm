# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one sd.cpp worker of this process, shared by every caller.

``lock`` serialises use of ``runner``. The worker exits after
``IDLE_UNLOAD_SECONDS`` without use (``arm_idle_timer``), on ``free()``, or when
this process exits.
"""

from __future__ import annotations

import threading
from typing import Optional

from .runner import SdRunner

IDLE_UNLOAD_SECONDS = 600.0

runner = SdRunner()
lock = threading.Lock()
_idle_timer: Optional[threading.Timer] = None
_timer_lock = threading.Lock()


def cancel_idle_timer() -> None:
    global _idle_timer
    with _timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None


def arm_idle_timer(seconds: Optional[float] = None) -> None:
    """(Re)start the countdown after which an unused worker is stopped."""
    global _idle_timer
    with _timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        t = threading.Timer(IDLE_UNLOAD_SECONDS if seconds is None else seconds, _idle_unload)
        t.daemon = True
        t.name = "localm-sdcpp-idle"
        _idle_timer = t
        t.start()


def _idle_unload() -> None:
    if lock.acquire(blocking=False):
        try:
            runner.shutdown()
        finally:
            lock.release()


def free() -> bool:
    """Stop the worker, waiting for a running job to finish first. True once no
    worker is running."""
    with lock:
        cancel_idle_timer()
        runner.shutdown()
        return not runner.is_alive()


def worker_pid() -> Optional[int]:
    """The live worker's process id, or None."""
    pid = runner.pid
    return pid if pid is not None and runner.is_alive() else None
