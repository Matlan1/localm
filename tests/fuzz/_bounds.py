# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time and allocation bounds for fuzz properties."""
from __future__ import annotations

import threading
import time
import tracemalloc

DEFAULT_WALL_SECONDS = 120.0
DEFAULT_CPU_SECONDS = 30.0
DEFAULT_PEAK_BYTES = 48 * 1024 * 1024


def returns_within(fn, *args, seconds: float = DEFAULT_WALL_SECONDS,
                   cpu_seconds: float = DEFAULT_CPU_SECONDS, **kwargs):
    """Call ``fn(*args, **kwargs)`` and return its value; re-raise whatever it
    raises.

    Two bounds, because the box this runs on is shared and a wall-clock limit
    alone fails on load rather than on a slow parser:

    * *cpu_seconds*: CPU time the call itself spent on its thread. A spin or a
      quadratic blow-up exceeds it however idle or busy the machine is.
    * *seconds*: wall time before the call is declared hung. The call runs on a
      daemon thread so a hang fails the property instead of freezing the run
      (the stuck thread is abandoned, not killed)."""
    box: dict = {}

    def run():
        start = time.thread_time()
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            box["exc"] = exc
        box["cpu"] = time.thread_time() - start

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    name = getattr(fn, "__name__", fn)
    if t.is_alive():
        raise AssertionError(f"{name} still running after {seconds}s")
    if box["cpu"] > cpu_seconds:
        raise AssertionError(f"{name} used {box['cpu']:.1f}s of CPU (limit {cpu_seconds}s)")
    if "exc" in box:
        raise box["exc"]
    return box["value"]


def peak_allocation(fn, *args, **kwargs):
    """Return ``(value_or_exception, peak_bytes)`` for one call, measured with
    tracemalloc. The exception is returned, not raised, so the caller can check
    the peak even when the call fails."""
    tracemalloc.start()
    try:
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            result = exc
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak
