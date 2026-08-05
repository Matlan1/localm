# SPDX-License-Identifier: AGPL-3.0-or-later
"""run_in_threadpool_bounded() (_threadpool_timeout.py) - the follow-up to
#1057 that bounds how long an HTTP request waits on a genuinely stuck
run_in_threadpool call, instead of hanging forever.

Every test here earned its place from a real, measured failure mode while
building this module - not speculative coverage:

- test_wrapping_run_in_threadpool_itself_does_nothing: proves the naive
  "just wrap run_in_threadpool in a deadline" design does not work at all
  (starlette never sets abandon_on_cancel), which is WHY this module exists
  instead of a two-line change at each call site.
- test_a_functions_own_timeouterror_propagates_unrelabeled: proves a bug
  that was caught live by tests/test_config_reg586_reg566_regressions.py
  during this change - an early ``except TimeoutError`` implementation could
  not distinguish "our deadline fired" from "the wrapped function raised its
  own TimeoutError well within budget" (localm.config.update_config's
  cross-process-lock timeout does exactly this), and relabeled the latter
  with a confusing, wrong message.
- test_abandoning_releases_the_capacity_limiter_token_immediately: a
  regression guard for a non-obvious anyio behaviour this module's whole
  "safe to timeout" reasoning depends on - if a future anyio release changes
  it, this test (not a production incident) should be what catches it.
"""

from __future__ import annotations

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
    # So existing `except TimeoutError` call sites (e.g. anything already
    # handling localm.config.update_config's own cross-process TimeoutError)
    # keep working unchanged if they end up wrapping a bounded call too.
    assert issubclass(ThreadCallTimeout, TimeoutError)


async def test_underlying_exception_propagates_unchanged_within_budget():
    def _raises():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_in_threadpool_bounded(_raises, timeout=5.0)


async def test_a_functions_own_timeouterror_propagates_unrelabeled():
    """THE bug this module's implementation had to be fixed for. A wrapped
    function that raises its OWN plain TimeoutError well within budget (like
    update_config's cross-process-lock timeout) must surface THAT exact
    exception - not get relabeled as an abandoned/leaked ThreadCallTimeout,
    which would hide the real, more specific message an operator needs."""
    def _raises_own_timeout():
        time.sleep(0.05)
        raise TimeoutError("held by another localm process")

    with pytest.raises(TimeoutError, match="held by another localm process") as exc_info:
        await run_in_threadpool_bounded(_raises_own_timeout, timeout=5.0)
    assert not isinstance(exc_info.value, ThreadCallTimeout), (
        "a function's own TimeoutError, raised well within budget, was "
        "wrongly relabeled as an abandoned-call ThreadCallTimeout")


async def test_timeout_is_a_required_keyword_argument():
    # No default on purpose (see the module docstring) - a caller must
    # consciously pick a budget rather than silently inherit a one-size-
    # fits-all number that is wrong for most call sites.
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
    """Documents WHY this module bypasses fastapi.concurrency.run_in_threadpool
    instead of wrapping it: starlette's run_in_threadpool calls
    anyio.to_thread.run_sync(func) with anyio's default abandon_on_cancel=False,
    so a deadline placed around it never actually unblocks the caller - the
    naive fix a reviewer might reach for silently does nothing."""
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
    """Regression guard for the non-obvious anyio behaviour run_in_threadpool_
    bounded's whole "safe against permanently reducing pool capacity" claim
    depends on: CONFIRMED BY DIRECT TEST (see the module docstring) that
    abandoning a call frees its CapacityLimiter token right away, not when
    the real (still-running) thread eventually returns. If a future anyio
    release changes this, THIS test should catch it, not a production
    incident where new requests mysteriously queue forever behind a wedged
    call that "should" have freed its slot."""
    limiter = anyio.CapacityLimiter(1)

    # anyio.to_thread.run_sync doesn't take a limiter kwarg through our
    # wrapper (it uses the default pool, matching run_in_threadpool), so
    # exercise the underlying primitive directly here to pin the limiter
    # semantics run_in_threadpool_bounded relies on.
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        with anyio.move_on_after(0.1) as scope:
            await anyio.to_thread.run_sync(
                _blocking_sleep, 3.0, abandon_on_cancel=True, limiter=limiter)
        if scope.cancelled_caught:
            raise TimeoutError("abandoned")
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
#  RMW-safety fires-control: an abandoned call must not defeat an existing    #
#  lock's serialization guarantee (the #1045 shape).                          #
# --------------------------------------------------------------------------- #

async def test_media_workflows_lock_survives_an_abandoned_writer():
    """Fires-control for the #1045 race shape, now that these routes are
    timeout-wrapped: an abandoned (timed-out) writer's REAL thread must keep
    holding media_workflows._lock_for's lock for as long as it actually
    runs, so a request that arrives after seeing the timeout (e.g. a user
    retry) still queues behind it instead of racing it."""
    from localm import media_workflows

    key = "test-threadpool-timeout-abandoned-lock"
    media_workflows._media_locks.pop(key, None)
    order = []
    proceed = threading.Event()

    def slow_holder():
        with media_workflows._lock_for(key):
            order.append("A-in")
            proceed.wait(timeout=5)
            order.append("A-out")

    def quick_second():
        with media_workflows._lock_for(key):
            order.append("B-in")

    async def run_slow():
        with pytest.raises(ThreadCallTimeout):
            await run_in_threadpool_bounded(slow_holder, timeout=0.15)

    async def run_quick():
        await run_in_threadpool_bounded(quick_second, timeout=5.0)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_slow)
            await anyio.sleep(0.4)   # let A's own timeout fire and the caller give up
            assert order == ["A-in"], "A should still be inside the lock at this point"

            tg.start_soon(run_quick)
            # give B a fair chance to (wrongly) proceed if the lock did not hold
            await anyio.sleep(0.3)
            assert "B-in" not in order, (
                "B entered the critical section while A's abandoned thread "
                "still holds the lock - the timeout defeated the existing "
                "serialization guarantee")
            proceed.set()   # let A's real thread finish and release the lock
    finally:
        proceed.set()
        media_workflows._media_locks.pop(key, None)

    assert order == ["A-in", "A-out", "B-in"], order


async def test_remove_managed_comfy_lock_survives_an_abandoned_caller(tmp_path, monkeypatch):
    """Same fires-control as above, for managed_comfy.py's new _remove_lock -
    proves a client retry after a timeout cannot start a second concurrent
    rmtree against the same managed ComfyUI install while an abandoned
    first call's thread is still deleting files."""
    from localm.media import managed_comfy

    order = []
    proceed = threading.Event()
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

    async def run_slow():
        with pytest.raises(ThreadCallTimeout):
            await run_in_threadpool_bounded(
                managed_comfy.remove_managed_comfy, False, timeout=0.15)

    async def run_quick():
        await run_in_threadpool_bounded(
            managed_comfy.remove_managed_comfy, False, timeout=5.0)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_slow)
            await anyio.sleep(0.4)
            assert order == ["A-in"]

            tg.start_soon(run_quick)
            await anyio.sleep(0.3)
            assert "B-in" not in order, (
                "a second remove_managed_comfy call proceeded while the "
                "first (abandoned) call's rmtree was still running")
            proceed.set()
    finally:
        proceed.set()

    assert order == ["A-in", "A-out", "B-in"], order
