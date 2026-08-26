# SPDX-License-Identifier: AGPL-3.0-or-later
"""run_in_threadpool_bounded() (_threadpool_timeout.py) bounds how long an HTTP
request waits on a genuinely stuck run_in_threadpool call, instead of hanging
forever.

- test_wrapping_run_in_threadpool_itself_does_nothing: wrapping
  run_in_threadpool in a deadline does not bound the call at all, because
  starlette never sets abandon_on_cancel.
- test_a_functions_own_timeouterror_propagates_unrelabeled: an
  ``except TimeoutError`` implementation cannot distinguish "our deadline
  fired" from "the wrapped function raised its own TimeoutError well within
  budget" (localm.config.update_config's cross-process-lock timeout does
  exactly this), and must not relabel the latter.
- test_abandoning_releases_the_capacity_limiter_token_immediately: pins the
  anyio behaviour this module's "safe to timeout" reasoning depends on.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import anyio
import anyio.to_thread
import pytest

from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded

pytestmark = pytest.mark.anyio


def _blocking_sleep(seconds, label="x"):
    time.sleep(seconds)
    return f"done:{label}"


async def test_fast_call_within_budget_returns_normally():
    result = await run_in_threadpool_bounded(_blocking_sleep, 0.05, "ok", timeout=5.0)
    assert result == "done:ok"


async def test_slow_call_raises_threadcalltimeout_within_budget_not_full_duration():
    start = time.monotonic()
    with pytest.raises(ThreadCallTimeout):
        await run_in_threadpool_bounded(_blocking_sleep, 3.0, timeout=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5, (
        f"caller waited {elapsed:.2f}s for a 0.3s budget - the deadline did "
        "not actually bound the wait")


async def test_threadcalltimeout_is_a_timeouterror_subclass():
    # Existing `except TimeoutError` call sites (e.g. anything already handling
    # localm.config.update_config's own cross-process TimeoutError) keep working
    # unchanged if they end up wrapping a bounded call too.
    assert issubclass(ThreadCallTimeout, TimeoutError)


async def test_underlying_exception_propagates_unchanged_within_budget():
    def _raises():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_in_threadpool_bounded(_raises, timeout=5.0)


async def test_a_functions_own_timeouterror_propagates_unrelabeled():
    """A wrapped function that raises its OWN plain TimeoutError well within
    budget (like update_config's cross-process-lock timeout) must surface THAT
    exact exception, never a relabeled abandoned/leaked ThreadCallTimeout,
    which would hide the more specific message an operator needs."""
    def _raises_own_timeout():
        time.sleep(0.05)
        raise TimeoutError("held by another localm process")

    with pytest.raises(TimeoutError, match="held by another localm process") as exc_info:
        await run_in_threadpool_bounded(_raises_own_timeout, timeout=5.0)
    assert not isinstance(exc_info.value, ThreadCallTimeout), (
        "a function's own TimeoutError, raised well within budget, was "
        "wrongly relabeled as an abandoned-call ThreadCallTimeout")


async def test_timeout_is_a_required_keyword_argument():
    # No default: a caller must pick a budget rather than inherit a
    # one-size-fits-all number.
    with pytest.raises(TypeError):
        await run_in_threadpool_bounded(_blocking_sleep, 0.01)


async def test_timeout_logs_a_warning_naming_the_call_and_budget(caplog):
    with caplog.at_level(logging.WARNING, logger="localm"):
        with pytest.raises(ThreadCallTimeout):
            await run_in_threadpool_bounded(_blocking_sleep, 2.0, timeout=0.1)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("_blocking_sleep" in r.getMessage() and "0.1" in r.getMessage()
               for r in warnings), (
        f"expected a WARNING naming the function and its budget, got: "
        f"{[r.getMessage() for r in warnings]}")


async def test_wrapping_run_in_threadpool_itself_does_nothing():
    """This module bypasses fastapi.concurrency.run_in_threadpool instead of
    wrapping it: starlette's run_in_threadpool calls
    anyio.to_thread.run_sync(func) with anyio's default abandon_on_cancel=False,
    so a deadline placed around it never actually unblocks the caller."""
    from fastapi.concurrency import run_in_threadpool

    start = time.monotonic()
    with anyio.move_on_after(0.2) as scope:
        await run_in_threadpool(_blocking_sleep, 0.6)
    elapsed = time.monotonic() - start
    assert not scope.cancelled_caught, (
        "a deadline around bare run_in_threadpool fired early - if this "
        "starts passing, starlette's default abandon_on_cancel behaviour "
        "changed and run_in_threadpool_bounded's whole rationale should be "
        "re-checked")
    assert elapsed >= 0.5, (
        f"returned after only {elapsed:.2f}s - expected it to wait out the "
        "full 0.6s blocking call regardless of the 0.2s deadline")


async def test_abandoning_releases_the_capacity_limiter_token_immediately():
    """run_in_threadpool_bounded's "safe against permanently reducing pool
    capacity" claim depends on a non-obvious anyio behaviour: abandoning a call
    frees its CapacityLimiter token right away, not when the real (still
    running) thread eventually returns. This pins it, so a future anyio release
    that changes it fails here rather than making new requests queue forever
    behind a wedged call that should have freed its slot."""
    limiter = anyio.CapacityLimiter(1)

    # anyio.to_thread.run_sync does not take a limiter kwarg through our wrapper
    # (it uses the default pool, matching run_in_threadpool), so exercise the
    # underlying primitive directly here to pin the limiter semantics
    # run_in_threadpool_bounded relies on.
    abandon_start = time.monotonic()
    with pytest.raises(TimeoutError):
        with anyio.move_on_after(0.1) as scope:
            await anyio.to_thread.run_sync(
                _blocking_sleep, 3.0, abandon_on_cancel=True, limiter=limiter)
        if scope.cancelled_caught:
            raise TimeoutError("abandoned")
    abandon_elapsed = time.monotonic() - abandon_start
    assert abandon_elapsed < 1.0, (
        f"abandonment took {abandon_elapsed:.2f}s - expected the 0.1s "
        "deadline to bound it, not the full 3.0s blocking call")
    stats = limiter.statistics()
    assert stats.borrowed_tokens == 0, (
        f"expected the limiter token to be released immediately on "
        f"abandonment, got borrowed_tokens={stats.borrowed_tokens}")

    # Prove it concretely: a second call against the SAME 1-token limiter
    # must succeed immediately rather than queuing behind the "leaked" slot.
    quick_start = time.monotonic()
    result = await anyio.to_thread.run_sync(
        _blocking_sleep, 0.05, "quick", abandon_on_cancel=True, limiter=limiter)
    quick_elapsed = time.monotonic() - quick_start
    assert result == "done:quick"
    assert quick_elapsed < 1.0, (
        f"a second call against the same limiter took {quick_elapsed:.2f}s - "
        "the abandoned call's token appears to still be held")

    # Let the original wedged thread actually finish before the test ends.
    await anyio.sleep(3.0)


# --------------------------------------------------------------------------- #
#  An abandoned call must not defeat an existing lock's serialization          #
#  guarantee.                                                                  #
# --------------------------------------------------------------------------- #

async def test_media_workflows_lock_survives_an_abandoned_writer():
    """An abandoned (timed-out) writer's REAL thread must keep holding
    media_workflows._lock_for's lock for as long as it actually runs, so a
    request that arrives after seeing the timeout (e.g. a user retry) still
    queues behind it instead of racing it.

    Synchronized deterministically throughout (a threading.Event polled from
    async code) rather than with fixed sleeps, so it cannot flake under box
    load."""
    from localm import media_workflows

    key = "test-threadpool-timeout-abandoned-lock"
    media_workflows._media_locks.pop(key, None)
    order = []
    proceed = threading.Event()
    b_attempting = threading.Event()

    def slow_holder():
        with media_workflows._lock_for(key):
            order.append("A-in")
            proceed.wait(timeout=5)
            order.append("A-out")

    def quick_second():
        b_attempting.set()   # signalled BEFORE trying to acquire the lock
        with media_workflows._lock_for(key):
            order.append("B-in")

    try:
        # Awaiting this to completion IS the synchronization: it returns only
        # once run_in_threadpool_bounded's own 0.15s deadline has fired and the
        # caller has given up. A's real thread keeps running in the background,
        # still holding the lock - the abandoned-call state this test needs
        # before dispatching B.
        with pytest.raises(ThreadCallTimeout):
            await run_in_threadpool_bounded(slow_holder, timeout=0.15)
        assert order == ["A-in"], "A should still be inside the lock at this point"

        b_task = asyncio.ensure_future(
            run_in_threadpool_bounded(quick_second, timeout=5.0))
        for _ in range(500):
            if b_attempting.is_set():
                break
            await anyio.sleep(0.005)
        assert b_attempting.is_set(), "B's worker thread never started"
        # B is now either blocked on lock.acquire() or has just barely slipped
        # through, but it cannot have appended B-in yet: `proceed` has not been
        # set and A's real thread still unconditionally holds the lock. That is a
        # lock-semantics guarantee, not a timing race, so no further wait.
        assert "B-in" not in order, (
            "B entered the critical section while A's abandoned thread "
            "still holds the lock - the timeout defeated the existing "
            "serialization guarantee")
        proceed.set()   # let A's real thread finish and release the lock
        await b_task
    finally:
        proceed.set()
        media_workflows._media_locks.pop(key, None)

    assert order == ["A-in", "A-out", "B-in"], order


async def test_remove_managed_comfy_lock_survives_an_abandoned_caller(tmp_path, monkeypatch):
    """The same property for managed_comfy.py's _remove_lock: a client retry
    after a timeout cannot start a second concurrent rmtree against the same
    managed ComfyUI install while an abandoned first call's thread is still
    deleting files. Synchronized deterministically throughout."""
    from localm.media import managed_comfy

    order = []
    proceed = threading.Event()
    b_attempting = threading.Event()
    calls: list = []

    def _fake_rmtree(path):
        calls.append(path)
        if len(calls) == 1:
            order.append("A-in")
            proceed.wait(timeout=5)
            order.append("A-out")
        else:
            order.append("B-in")

    fake_target = tmp_path / "comfyui"
    fake_target.mkdir()
    monkeypatch.setattr(managed_comfy, "managed_comfy_remove_targets",
                        lambda with_models=False: [fake_target])
    monkeypatch.setattr(managed_comfy, "rmtree_robust", _fake_rmtree)

    def run_second():
        # Signalled BEFORE remove_managed_comfy is even called, so it fires
        # right as B's worker thread starts - before B has any chance to
        # contend for _remove_lock.
        b_attempting.set()
        return managed_comfy.remove_managed_comfy(False)

    try:
        with pytest.raises(ThreadCallTimeout):
            await run_in_threadpool_bounded(
                managed_comfy.remove_managed_comfy, False, timeout=0.15)
        assert order == ["A-in"]

        b_task = asyncio.ensure_future(
            run_in_threadpool_bounded(run_second, timeout=5.0))
        for _ in range(500):
            if b_attempting.is_set():
                break
            await anyio.sleep(0.005)
        assert b_attempting.is_set(), "B's worker thread never started"
        assert "B-in" not in order, (
            "a second remove_managed_comfy call proceeded while the "
            "first (abandoned) call's rmtree was still running")
        proceed.set()
        await b_task
    finally:
        proceed.set()

    assert order == ["A-in", "A-out", "B-in"], order
