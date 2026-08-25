# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded-latency counterpart to ``fastapi.concurrency.run_in_threadpool`` -
the timeout/cancellation mechanism ``_executor_health.py``'s module docstring
names as a deliberately separate, not-yet-answered question for anyio's pool:
observability alone still leaves a genuinely stuck sync call (a file lock held
by a dead process, a hung filesystem call, a native library that never
returns) hanging the ONE HTTP request depending on it forever.

**Do not wrap ``run_in_threadpool`` itself in a deadline - it does nothing.**
Confirmed by direct test, not assumed: ``anyio.to_thread.run_sync`` only
actually abandons a cancelled caller when called with
``abandon_on_cancel=True``, and ``starlette.concurrency.run_in_threadpool``
never passes it (calls ``anyio.to_thread.run_sync(func)`` with anyio's default
``abandon_on_cancel=False``). Wrapping THAT call in ``anyio.fail_after()`` /
``asyncio.wait_for()`` waits for the FULL blocking call regardless of the
deadline - the cancellation is deferred until the worker thread's function
returns, which is exactly the hang this module exists to bound. This module
instead calls ``anyio.to_thread.run_sync`` directly with
``abandon_on_cancel=True``, replicating ``run_in_threadpool``'s own
``functools.partial`` argument handling and default (unnamed) limiter so
these calls still land in, and are still visible to, the exact same anyio
pool ``_executor_health.py`` already watches.

**What "abandon" actually buys, confirmed by direct test against this anyio
version (4.14.2), not assumed from reading the docs:**

- The awaiting coroutine stops waiting at the deadline. The real OS thread
  keeps running the original blocking call to completion (or forever) -
  Python cannot forcibly stop a thread. Its eventual return value or
  exception is silently discarded (anyio's own documented behaviour).
- The anyio ``CapacityLimiter`` token the call had borrowed is released
  IMMEDIATELY when the caller abandons, not when the real thread eventually
  returns - measured live: ``borrowed_tokens`` drops back to 0 the instant
  ``fail_after`` fires, and two further calls against a 2-token limiter both
  succeed immediately afterward rather than queuing behind the "leaked" slot.
  So a wedged call bounded by this module does NOT permanently reduce the
  pool's usable capacity for other requests - a materially better outcome
  than a naive "the token leaks forever" model would predict.
- The genuinely un-recoverable residual is a single real OS thread, pinned
  inside the original blocking call for the remaining life of the process
  (it is simply never returned to anyio's idle-worker pool). That is a slow,
  bounded-per-incident leak (one thread per timeout that ever actually
  fires), not the unbounded-capacity leak the pre-timeout code had.
- Consequence for ``_executor_health.py``'s saturation watch: because the
  limiter token frees itself the moment THIS module's own deadline fires, a
  single wedged call bounded here will never by itself hold
  ``borrowed_tokens`` at the ceiling long enough to trip that watch's
  sustained-streak WARNING. The two mechanisms are complementary, not
  redundant - the saturation watch still catches genuine concurrent BURSTS
  (many legitimate slow calls at once, or any call this module does not wrap)
  while this module bounds how long any ONE wrapped caller ever waits.
"""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

import anyio
import anyio.to_thread

T = TypeVar("T")


class ThreadCallTimeout(TimeoutError):
    """Raised by ``run_in_threadpool_bounded`` when *func* has not returned
    within its *timeout* budget. A ``TimeoutError`` subclass (not a bare new
    type) so existing ``except TimeoutError`` handling - e.g. a caller that
    already expects ``localm.config.update_config``'s own cross-process-lock
    ``TimeoutError`` - keeps working unchanged; catch this type specifically
    when a route wants a distinct, more actionable message.

    The real OS thread *func* was running on is NOT stopped by this
    exception - see the module docstring. It keeps running independently and
    its eventual result (or failure) is discarded."""


async def run_in_threadpool_bounded(func: Callable[..., T], *args, timeout: float,
                                    **kwargs) -> T:
    """Like ``fastapi.concurrency.run_in_threadpool(func, *args, **kwargs)``,
    but the awaiting caller gives up after *timeout* seconds instead of
    waiting for *func* forever.

    *timeout* is a required keyword-only argument, deliberately with no
    default: the right budget for "check whether a checkpoint exists" and
    "write a multi-MB workflow JSON to a slow disk" differ by orders of
    magnitude (see the call sites in localm/inference/routes/config.py,
    media_workflows.py, etc.), so a single silently-applied default would be
    wrong for most callers rather than right for any of them. Pick a budget
    generously larger than *func*'s own worst-case legitimate duration (its
    own internal timeouts, a configured wait, or a rough measurement) so this
    only ever fires for a call that has gone genuinely beyond that, never for
    ordinary slow-but-working load.

    Raises ``ThreadCallTimeout`` (a ``TimeoutError`` subclass) on expiry, and
    logs a WARNING first (AGENTS.md rule 5 - a request silently timing out
    with no trace is exactly the kind of hidden failure that rule forbids).
    Propagates whatever *func* itself raises or returns, unchanged, when it
    completes within budget - this wrapper changes only how long the caller
    waits, never *func*'s own behaviour or return value.

    Uses ``anyio.move_on_after`` + ``scope.cancelled_caught``, NOT
    ``anyio.fail_after``/a bare ``except TimeoutError`` - confirmed by direct
    test to matter, not a style choice. Several wrapped callers (notably
    ``localm.config.update_config``) can legitimately raise their OWN plain
    ``TimeoutError`` well within budget (a cross-process lock genuinely held
    by another process). ``anyio.fail_after`` raises the SAME builtin
    ``TimeoutError`` type for its own deadline, so an ``except TimeoutError``
    wrapped around it cannot tell "the deadline fired" from "func raised its
    own TimeoutError" and would relabel the latter as an abandoned/leaked
    call, hiding its real, more specific message. ``cancelled_caught`` is
    only ever true when THIS scope's own deadline expired - any exception
    *func* raises (including a coincidental ``TimeoutError``) propagates
    through the ``with`` block untouched, since ``CancelScope`` only
    intercepts its own internal cancellation, never an unrelated exception."""
    call = functools.partial(func, *args, **kwargs)
    with anyio.move_on_after(timeout) as scope:
        return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
    if not scope.cancelled_caught:
        # Structurally unreachable as this function is written today: the
        # only way to fall through the `with` block above without returning
        # is for THIS scope's own deadline to have fired (any exception func
        # itself raises propagates straight through instead - see the
        # docstring). Asserting it explicitly, rather than leaving it as an
        # implicit consequence of CancelScope's own-scope-only suppression
        # rule, turns a future edit that adds logic here (a broader
        # try/except, an early return) and silently breaks that invariant
        # into a loud failure instead of a silently wrong exception type.
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
