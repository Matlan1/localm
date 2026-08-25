# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded-latency counterpart to ``fastapi.concurrency.run_in_threadpool`` - the timeout/cancellation mechanism ``_executor_health.py``'s module docstring names as a deliberately separate, not-yet-answered question for anyio's pool: observability alone still leaves a genuinely stuck sync call (a file..."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

import anyio
import anyio.to_thread

T = TypeVar("T")


class ThreadCallTimeout(TimeoutError):
    """Raised by ``run_in_threadpool_bounded`` when *func* has not returned within its *timeout* budget."""


async def run_in_threadpool_bounded(func: Callable[..., T], *args, timeout: float,
                                    **kwargs) -> T:
    """Like ``fastapi.concurrency.run_in_threadpool(func, *args, **kwargs)``, but the awaiting caller gives up after *timeout* seconds instead of waiting for *func* forever."""
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
