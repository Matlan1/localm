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
    anyio_pool_health,
    executors_snapshot,
    pool_health,
    start_executor_saturation_watch,
)


def test_pool_health_none_executor_reports_nothing_to_watch():
    # The asyncio default executor does not exist until the first
    # run_in_executor(None, ...) call lazily creates it - an idle server
    # legitimately has None here, and that must read as "not saturated",
    # never as an error.
    # "shutdown" is False, not None: a pool that does not exist yet has not been
    # shut down, and that is a measured distinction rather than an unknown.
    assert pool_health(None) == {
        "max_workers": None, "threads_spawned": 0, "queued": 0,
        "saturated": False, "shutdown": False}


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


# --- anyio's default thread pool - the THIRD pool (fastapi.concurrency. ---
# --- run_in_threadpool's pool; see _executor_health.py's module docstring) --

def test_anyio_pool_health_none_limiter_reports_nothing_to_watch():
    # No captured reference at all (startup capture never ran, or failed) -
    # unlike pool_health(None)'s "not created yet", this means "cannot
    # observe from here", but the REPORTED shape is deliberately identical:
    # never saturated, no counts.
    assert anyio_pool_health(None) == {
        "max_workers": None, "threads_spawned": None, "queued": None,
        "saturated": False, "shutdown": None}


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
    """Mirrors test_pool_health_reports_saturated_and_recovers exactly, but
    for anyio's CapacityLimiter instead of a ThreadPoolExecutor - shrinks the
    limiter to a single token so one blocked run_sync call plus one queued
    behind it is enough to prove real saturated/recovered transitions, not
    just the None/idle shapes above."""
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
    """Proves the wiring, not just the standalone function: executors_snapshot
    (what both the saturation watch and GET /debug/stacks consume) surfaces
    anyio's pool under the "anyio" key at all, with a real captured limiter."""
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
    """The default (no anyio_limiter passed) must not silently omit the key -
    a caller reading the snapshot should see "anyio" present and honestly
    unobservable, not absent (which would look like a caller-side bug rather
    than an intentional 'nothing captured here' state)."""
    async def scenario():
        loop = asyncio.get_running_loop()
        snapshot = executors_snapshot(loop)
        assert snapshot["anyio"] == {
            "max_workers": None, "threads_spawned": None, "queued": None,
            "saturated": False, "shutdown": None}
    asyncio.run(scenario())


def test_saturation_watch_detects_anyio_saturation():
    """End-to-end proof the background watch (a plain daemon thread, NOT
    async) actually warns on anyio saturation using a limiter reference
    captured from async code beforehand - the exact "cannot fetch it itself,
    must be handed a live reference" contract _executor_health.py's module
    docstring describes, exercised for real rather than asserted."""
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
    """End-to-end through a REAL request, not just the unit-level function
    calls above: GET /debug/stacks is an async handler with its own running
    loop, so it fetches anyio's default thread limiter live on every call
    (see http_server.py) rather than needing a captured reference the way
    the background saturation watch does. Proves the wiring actually landed
    in the route, not just in _executor_health.py in isolation."""
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


# --------------------------------------------------------------------------- #
#  A SHUT-DOWN pool is reported as such, not as a quiet one                    #
# --------------------------------------------------------------------------- #

def test_a_shut_down_pool_is_otherwise_indistinguishable_from_light_load():
    """The measurement that justifies reporting `shutdown` at all.

    After .shutdown() the other three fields still read plausibly, so the
    pre-existing `saturated` proxy computes False and the whole shape looks
    like a pool under light load - while every caller of it is getting
    RuntimeError. This test pins that collapse, so if a future change ever
    makes `saturated` able to express the dead state on its own, this fails
    and tells the next reader the extra key has become redundant.
    """
    pool = ThreadPoolExecutor(max_workers=2)
    pool.submit(lambda: 1).result(timeout=10)
    pool.shutdown(wait=True)

    h = pool_health(pool)
    assert h["saturated"] is False, (
        "a dead pool now reports saturated - the collapse this key works "
        "around may have changed shape")
    assert h["max_workers"] == 2
    # queued is the wake-up sentinel shutdown() itself enqueues, which is
    # exactly why the numbers look ordinary.
    assert h["queued"] is not None


def test_pool_health_reports_a_shut_down_pool_distinctly():
    live = ThreadPoolExecutor(max_workers=2)
    dead = ThreadPoolExecutor(max_workers=2)
    try:
        dead.shutdown(wait=True)
        assert pool_health(live)["shutdown"] is False
        assert pool_health(dead)["shutdown"] is True
    finally:
        live.shutdown(wait=False)


def test_pool_health_never_claims_a_shutdown_state_it_could_not_read():
    # A pool it cannot introspect must report None ("cannot say"), never False
    # ("measured healthy") - the same never-trust-in-silence rule the rest of
    # this module follows.
    class _NotARealExecutor:
        pass

    assert pool_health(_NotARealExecutor())["shutdown"] is None


def test_anyio_pool_never_claims_a_shutdown_state():
    # A CapacityLimiter has no shut-down concept, so False would assert a fact
    # about that pool nobody measured.
    async def scenario():
        import anyio.to_thread
        limiter = anyio.to_thread.current_default_thread_limiter()
        assert anyio_pool_health(limiter)["shutdown"] is None
    asyncio.run(scenario())


def test_the_watch_warns_once_for_a_dead_pool_and_never_for_an_unknown_one(monkeypatch):
    """A dead pool gets exactly ONE line however long it stays dead, and a pool
    whose state is unknown (None) gets none at all.

    Deterministic by construction rather than by timing: the faked snapshot
    counts its own calls and signals once it has been polled five times, so
    "exactly one warning across five ticks" is a fact about the throttle, not a
    race against the clock.
    """
    import localm.inference._executor_health as eh
    from localm.debuglog import logger as _dbg

    DEAD = {"max_workers": 4, "threads_spawned": 1, "queued": 1,
            "saturated": False, "shutdown": True}
    IDLE = {"max_workers": 4, "threads_spawned": 0, "queued": 0,
            "saturated": False, "shutdown": False}
    UNKNOWN = {"max_workers": None, "threads_spawned": None, "queued": None,
               "saturated": False, "shutdown": None}

    polled_enough = threading.Event()
    calls = {"n": 0}

    def _fake_snapshot(loop, anyio_limiter=None):
        calls["n"] += 1
        if calls["n"] >= 5:
            polled_enough.set()
        return {"default": IDLE, "plugin": DEAD, "anyio": UNKNOWN}

    monkeypatch.setattr(eh, "executors_snapshot", _fake_snapshot)

    handler = _CaptureHandler()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)
    try:
        stop, thread = eh.start_executor_saturation_watch(
            None, threshold=0.5, poll=0.01)
        try:
            assert polled_enough.wait(10), "the watch never polled"
        finally:
            stop.set()
            thread.join(timeout=5)
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    shutdown_lines = [r for r in handler.records if "SHUT DOWN" in r[1]]
    assert len(shutdown_lines) == 1, (
        f"expected exactly one shut-down WARNING across {calls['n']} polls, "
        f"got {len(shutdown_lines)}: {shutdown_lines}")
    assert shutdown_lines[0][0] == "WARNING"
    assert "plugin" in shutdown_lines[0][1]
    # The unknown-state pool must stay silent: None means "cannot say", and a
    # monitor that warns on what it could not measure is noise, not a signal.
    assert not any("anyio" in r[1] and "SHUT DOWN" in r[1] for r in handler.records)
    assert not any("default" in r[1] and "SHUT DOWN" in r[1] for r in handler.records)
