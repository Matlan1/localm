# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-pool saturation detectability (_executor_health.py): pool exhaustion
must be detectable, never silent.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from localm.inference._executor_health import (
    anyio_pool_health,
    executors_snapshot,
    pool_health,
    start_executor_saturation_watch,
)


def test_pool_health_none_executor_reports_nothing_to_watch():
    # The asyncio default executor does not exist until the first
    # run_in_executor(None, ...) call lazily creates it, so an idle server has
    # None here and that reads as "not saturated", never as an error.
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
    """A pool saturated past the threshold for a SUSTAINED window logs exactly
    ONE warning, then drops to DEBUG for as long as it stays saturated."""
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

        # Generous margin over the threshold rather than a tight race: a 0.5s
        # threshold against a 3s window.
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

        # Generous margins throughout.
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


# --- anyio's default thread pool - the THIRD pool, the one behind ---
# --- fastapi.concurrency.run_in_threadpool ---

def test_anyio_pool_health_none_limiter_reports_nothing_to_watch():
    # No captured reference at all (startup capture never ran, or failed). Unlike
    # pool_health(None)'s "not created yet" this means "cannot observe from here",
    # and the reported shape is identical: never saturated, no counts.
    assert anyio_pool_health(None) == {
        "max_workers": None, "threads_spawned": None, "queued": None,
        "saturated": False}


def test_anyio_pool_health_idle_limiter_is_not_saturated():
    async def scenario():
        import anyio.to_thread
        limiter = anyio.to_thread.current_default_thread_limiter()
        h = anyio_pool_health(limiter)
        assert h["max_workers"] == limiter.total_tokens
        assert h["threads_spawned"] == 0
        assert h["saturated"] is False
    asyncio.run(scenario())


def test_anyio_pool_health_reports_saturated_and_recovers():
    """The ThreadPoolExecutor case above, for anyio's CapacityLimiter: the
    limiter is shrunk to a single token, so one blocked run_sync call plus one
    queued behind it produces real saturated/recovered transitions."""
    async def scenario():
        import anyio.to_thread
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 1
        ev = threading.Event()

        t1 = asyncio.create_task(anyio.to_thread.run_sync(ev.wait))
        deadline = time.monotonic() + 5.0
        h = anyio_pool_health(limiter)
        while h["threads_spawned"] != 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            h = anyio_pool_health(limiter)
        assert h["threads_spawned"] == 1, "the first call never acquired the token"

        t2 = asyncio.create_task(anyio.to_thread.run_sync(lambda: 1))
        deadline = time.monotonic() + 5.0
        h = anyio_pool_health(limiter)
        while not h["saturated"] and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            h = anyio_pool_health(limiter)
        assert h["threads_spawned"] == 1
        assert h["queued"] == 1
        assert h["saturated"] is True

        ev.set()
        await t1
        await t2
        h2 = anyio_pool_health(limiter)
        assert h2["saturated"] is False, h2
    asyncio.run(scenario())


def test_anyio_pool_health_never_raises_on_a_malformed_limiter():
    class _NotARealLimiter:
        pass

    h = anyio_pool_health(_NotARealLimiter())
    assert h["saturated"] is False
    assert h["max_workers"] is None


def test_executors_snapshot_includes_anyio_key():
    """executors_snapshot - what both the saturation watch and GET
    /debug/stacks consume - surfaces anyio's pool under the "anyio" key, with a
    real captured limiter."""
    async def scenario():
        import anyio.to_thread
        limiter = anyio.to_thread.current_default_thread_limiter()
        loop = asyncio.get_running_loop()
        snapshot = executors_snapshot(loop, anyio_limiter=limiter)
        assert "anyio" in snapshot
        assert snapshot["anyio"]["max_workers"] == limiter.total_tokens
        assert snapshot["anyio"]["saturated"] is False
    asyncio.run(scenario())


def test_executors_snapshot_anyio_key_present_but_unobservable_without_capture():
    """The default (no anyio_limiter passed) must not omit the key: a caller
    reading the snapshot sees "anyio" present and marked unobservable, never
    absent."""
    async def scenario():
        loop = asyncio.get_running_loop()
        snapshot = executors_snapshot(loop)
        assert snapshot["anyio"] == {
            "max_workers": None, "threads_spawned": None, "queued": None,
            "saturated": False}
    asyncio.run(scenario())


def test_saturation_watch_detects_anyio_saturation():
    """The background watch (a plain daemon thread, NOT async) warns on anyio
    saturation using a limiter reference captured from async code beforehand -
    the "cannot fetch it itself, must be handed a live reference" contract
    _executor_health.py's module docstring describes."""
    from localm.debuglog import logger as _dbg

    handler = _CaptureHandler()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)

    async def scenario():
        import anyio.to_thread
        loop = asyncio.get_running_loop()
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 1
        ev = threading.Event()
        t1 = asyncio.create_task(anyio.to_thread.run_sync(ev.wait))
        t2 = asyncio.create_task(anyio.to_thread.run_sync(lambda: 1))
        await asyncio.sleep(0.3)   # let t1 actually acquire the token

        stop, thread = start_executor_saturation_watch(
            loop, threshold=0.5, poll=0.2, anyio_limiter=limiter)
        try:
            await asyncio.sleep(3.0)
        finally:
            stop.set()
            thread.join(timeout=2)
            ev.set()
            await t1
            await t2

    try:
        asyncio.run(scenario())
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    warnings = [r for r in handler.records
               if r[0] == "WARNING" and "thread pool has been fully saturated" in r[1]
               and r[1].startswith("anyio")]
    assert len(warnings) == 1, (
        f"expected exactly one anyio WARNING, got: {handler.records}")


def test_debug_stacks_reports_anyio_pool_over_real_http(monkeypatch):
    """End-to-end through a REAL request: GET /debug/stacks is an async handler
    with its own running loop, so it fetches anyio's default thread limiter live
    on every call (see http_server.py) instead of needing a captured reference
    the way the background saturation watch does."""
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    from localm.inference.http_server import create_app
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    app = create_app(e)
    with TestClient(app) as c:
        r = c.get("/debug/stacks",
                 headers={"Authorization": f"Bearer {app.state.shell_token}"})
        assert r.status_code == 200, r.text
        executors = r.json()["executors"]
        assert "anyio" in executors
        assert executors["anyio"]["max_workers"] is not None
        assert executors["anyio"]["saturated"] is False
