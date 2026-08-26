# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded-latency counterpart to ``fastapi.concurrency.run_in_threadpool``: it
bounds how long the ONE HTTP request depending on a stuck sync call waits for
it.

**Do not wrap ``run_in_threadpool`` itself in a deadline - it does nothing.**
``anyio.to_thread.run_sync`` only abandons a cancelled caller when called with
``abandon_on_cancel=True``, and ``starlette.concurrency.run_in_threadpool``
never passes it, so wrapping THAT call in ``anyio.fail_after()`` /
``asyncio.wait_for()`` waits for the FULL blocking call regardless of the
deadline. This module instead calls ``anyio.to_thread.run_sync`` directly with
``abandon_on_cancel=True``, replicating ``run_in_threadpool``'s own
``functools.partial`` argument handling and default (unnamed) limiter, so these
calls still land in - and stay visible to - the same anyio pool
``_executor_health.py`` watches.

**What "abandon" does, against anyio 4.14.2:**

- The awaiting coroutine stops waiting at the deadline. The real OS thread
  keeps running the original blocking call to completion (or forever), and its
  eventual return value or exception is discarded.
- The anyio ``CapacityLimiter`` token the call had borrowed is released
  IMMEDIATELY when the caller abandons, not when the real thread eventually
  returns, so a wedged call bounded here does not reduce the pool's usable
  capacity for other requests.
- The residual is one real OS thread, pinned inside the original blocking call
  for the remaining life of the process and never returned to anyio's
  idle-worker pool: one thread per timeout that actually fires.
- Because the limiter token frees itself the moment this module's deadline
  fires, a single wedged call bounded here never holds ``borrowed_tokens`` at
  the ceiling long enough to trip ``_executor_health.py``'s sustained-streak
  WARNING. That watch still catches concurrent bursts, and any call this module
  does not wrap.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

import anyio
import anyio.to_thread

T = TypeVar("T")


class ThreadCallTimeout(TimeoutError):
    """Raised by ``run_in_threadpool_bounded`` when *func* has not returned
    within its *timeout* budget. A ``TimeoutError`` subclass, so existing
    ``except TimeoutError`` handling keeps working; catch this type
    specifically for a distinct, more actionable message.

    The real OS thread *func* was running on is NOT stopped by this exception.
    It keeps running independently and its eventual result (or failure) is
    discarded."""


async def run_in_threadpool_bounded(func: Callable[..., T], *args, timeout: float,
                                    **kwargs) -> T:
    """Like ``fastapi.concurrency.run_in_threadpool(func, *args, **kwargs)``,
    but the awaiting caller gives up after *timeout* seconds instead of
    waiting for *func* forever.

    *timeout* is a required keyword-only argument with no default. Pick a budget
    generously larger than *func*'s own worst-case legitimate duration, so this
    only fires for a call that has gone beyond that and never for ordinary
    slow-but-working load.

    Raises ``ThreadCallTimeout`` (a ``TimeoutError`` subclass) on expiry, and
    logs a WARNING first. Propagates whatever *func* itself raises or returns,
    unchanged, when it completes within budget: this wrapper changes only how
    long the caller waits.

    Uses ``anyio.move_on_after`` + ``scope.cancelled_caught``, NOT
    ``anyio.fail_after`` with a bare ``except TimeoutError``. Several wrapped
    callers (notably ``localm.config.update_config``) can raise their OWN plain
    ``TimeoutError`` well within budget, and ``anyio.fail_after`` raises that
    same builtin type for its own deadline. ``cancelled_caught`` is true only
    when THIS scope's deadline expired; any exception *func* raises, including a
    coincidental ``TimeoutError``, propagates through the ``with`` block
    untouched."""
    call = functools.partial(func, *args, **kwargs)
    with anyio.move_on_after(timeout) as scope:
        return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
    if not scope.cancelled_caught:
        # Unreachable as written: the only way to fall through the `with` block
        # above without returning is for THIS scope's own deadline to have
        # fired. Asserted so an edit that breaks the invariant fails loudly
        # rather than raising a wrong exception type.
        raise AssertionError(
            "run_in_threadpool_bounded: reached the timeout path without "
            "scope.cancelled_caught being true - this should be unreachable")
    from localm.debuglog import logger
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", repr(func))
    logger.warning(
        "run_in_threadpool_bounded: %s exceeded its %.1fs budget and was "
        "abandoned - the underlying call may still be running on a "
        "leaked worker thread; see GET /debug/stacks for pool state",
        name, timeout)
    raise ThreadCallTimeout(f"{name} did not complete within {timeout:.1f}s")
