# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-pool saturation detectability (_executor_health.py) - the "make
exhaustion detectable, not silent" half of the thread-pool-exhaustion fix
(dev-notes/decisions-2026-07-30-release-gate.md, Q2).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from localm.inference._executor_health import (
    pool_health,
    start_executor_saturation_watch,
)


def test_pool_health_none_executor_reports_nothing_to_watch():
    # The asyncio default executor does not exist until the first
    # run_in_executor(None, ...) call lazily creates it - an idle server
    # legitimately has None here, and that must read as "not saturated",
    # never as an error.
    assert pool_health(None) == {
        "max_workers": None, "threads_spawned": 0, "queued": 0, "saturated": False}


def test_pool_health_idle_pool_is_not_saturated():
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        h = pool_health(pool)
        assert h["max_workers"] == 2
        assert h["saturated"] is False
    finally:
        pool.shutdown(wait=False)


def test_pool_health_reports_saturated_and_recovers():
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        ev = threading.Event()
        pool.submit(ev.wait)            # occupies the only worker
        fut2 = pool.submit(lambda: 1)   # queues behind it - nothing is free
        deadline = time.monotonic() + 5.0
        h = pool_health(pool)
        while not h["saturated"] and time.monotonic() < deadline:
            time.sleep(0.05)
            h = pool_health(pool)
        assert h["threads_spawned"] == 1
        assert h["queued"] == 1
        assert h["saturated"] is True

        ev.set()
        fut2.result(timeout=5)
        # Recovered: the queued item ran, nothing is waiting any more.
        h2 = pool_health(pool)
        assert h2["saturated"] is False, h2
    finally:
        pool.shutdown(wait=False)


def test_pool_health_never_raises_on_a_malformed_executor():
    class _NotARealExecutor:
        pass

    h = pool_health(_NotARealExecutor())
    assert h["saturated"] is False
    assert h["max_workers"] is None


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def test_saturation_watch_warns_once_then_throttles_to_debug():
    """A pool saturated past the threshold for a SUSTAINED window logs
    exactly ONE warning, then drops to DEBUG for as long as it stays
    saturated - the log-once-then-throttle contract
    (.claude/rules/hard-won-rules.md's VRAM-probe log-flood lesson), proven
    here rather than assumed."""
    from localm.debuglog import logger as _dbg

    handler = _CaptureHandler()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)

    async def scenario():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        ev = threading.Event()
        fut1 = loop.run_in_executor(None, ev.wait)     # occupies the worker
        fut2 = loop.run_in_executor(None, lambda: 1)   # queues behind it
        await asyncio.sleep(0.3)

        # Generous margin over threshold (not a tight race): this box runs
        # many concurrent test sessions, and a timing assertion with little
        # headroom is exactly the flakiness class this repo's own test-slot
        # policy has been burned by before (widened a 2s budget to 15s for
        # the identical reason). 0.5s threshold vs a 3s window is 6x margin.
        stop, thread = start_executor_saturation_watch(loop, threshold=0.5, poll=0.2)
        try:
            await asyncio.sleep(3.0)
        finally:
            stop.set()
            thread.join(timeout=2)
            ev.set()
            await fut1
            await fut2
            pool.shutdown(wait=False)

    try:
        asyncio.run(scenario())
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    warnings = [r for r in handler.records
               if r[0] == "WARNING" and "thread pool has been fully saturated" in r[1]]
    debugs = [r for r in handler.records
             if r[0] == "DEBUG" and "still saturated" in r[1]]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING (log-once), got {len(warnings)}: {warnings}")
    assert len(debugs) >= 1, "expected at least one throttled DEBUG line after the warning"


def test_saturation_watch_resets_streak_on_recovery():
    """A pool that recovers (goes idle) and then re-saturates warns AGAIN -
    the streak must not stay 'already warned' forever after the first
    incident, or a second, unrelated exhaustion goes unreported."""
    from localm.debuglog import logger as _dbg

    handler = _CaptureHandler()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)

    async def scenario():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)

        # Generous margins throughout - see the identical note in
        # test_saturation_watch_warns_once_then_throttles_to_debug.
        stop, thread = start_executor_saturation_watch(loop, threshold=0.4, poll=0.2)
        try:
            # First saturation window.
            ev1 = threading.Event()
            f1a = loop.run_in_executor(None, ev1.wait)
            f1b = loop.run_in_executor(None, lambda: 1)
            await asyncio.sleep(2.0)   # well past the threshold - first warning
            ev1.set()
            await f1a
            await f1b
            await asyncio.sleep(1.0)   # recover - streak resets

            # Second, independent saturation window.
            ev2 = threading.Event()
            f2a = loop.run_in_executor(None, ev2.wait)
            f2b = loop.run_in_executor(None, lambda: 1)
            await asyncio.sleep(2.0)   # well past the threshold again
            ev2.set()
            await f2a
            await f2b
        finally:
            stop.set()
            thread.join(timeout=2)
            pool.shutdown(wait=False)

    try:
        asyncio.run(scenario())
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    warnings = [r for r in handler.records
               if r[0] == "WARNING" and "thread pool has been fully saturated" in r[1]]
    assert len(warnings) == 2, (
        f"expected two separate warnings (one per streak), got "
        f"{len(warnings)}: {warnings}")
