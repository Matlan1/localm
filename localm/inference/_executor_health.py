# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-pool saturation detectability.

A pool can fill up under ordinary load (a burst of legitimate work, or a native
call that blocks) with no signal to an operator beyond "the server got slow,
then stopped responding". This module makes that state visible: a WARNING once
saturation has held for a sustained window, plus the live numbers on
``GET /debug/stacks``. It is an observability signal, not a gate - nothing here
rejects load.

**THIS MODULE BUYS OBSERVABILITY, NOT RECOVERY.** A wedged worker in the
``default``/``plugin`` pools below still hangs FOREVER after this module reports
it: nothing here cancels the stuck call, frees the token/slot it holds, or
returns the HTTP request depending on it. Neither pool has any
timeout/cancellation mechanism.

The anyio pool is PARTIALLY covered elsewhere: every ``run_in_threadpool`` call
site (config updates, media workflow management, comfy management, coder session
deletion) goes through
``localm.inference._threadpool_timeout.run_in_threadpool_bounded``, which bounds
how long the HTTP request depending on a wedged call waits (it calls
``anyio.to_thread.run_sync`` directly with ``abandon_on_cancel=True``; wrapping
``run_in_threadpool`` itself in a deadline does nothing, because starlette never
sets that flag). uvicorn is started with no ``timeout_keep_alive`` or
request-level timeout of its own, and the hang watchdog is scoped to a STALLED
EVENT LOOP, which a hung anyio worker does not cause.

Abandoning a call via ``run_in_threadpool_bounded`` releases its anyio
``CapacityLimiter`` token IMMEDIATELY, not when the real (still-running) worker
thread eventually returns, so a single bounded wedged call never holds
``anyio_pool_health()``'s ``borrowed_tokens`` at the ceiling long enough to trip
the sustained-streak WARNING below once its own timeout has fired. The two
mechanisms are complementary: this module still catches genuine concurrent
BURSTS (many legitimate slow calls at once) and anything
``run_in_threadpool_bounded`` does not wrap. The residual leak after a bounded
call abandons is one permanently-stuck OS thread (Python cannot forcibly stop a
thread), not permanently-reduced pool capacity, so "anyio pool monitored" must
never be read as "anyio pool fully protected".

THREE pools are watched, symmetrically: the asyncio loop's TRUE default executor
(every ``run_in_executor(None, ...)`` call site - GPU/VRAM probes, model
load/unload, embeddings, count_tokens, and the isolated HF/GGUF runner RPCs),
the separate ``get_plugin_executor()`` pool (RAG/web/voice/coder tool calls),
and anyio's default worker-thread pool - the ONE FastAPI's own
``run_in_threadpool`` always uses (it calls ``anyio.to_thread.run_sync(func)``
with no per-call limiter override, so every ``run_in_threadpool`` call site in
this codebase shares this single pool). Exhaustion in one says nothing about the
others, so all three are tracked as independent streaks.

**anyio's pool is NOT a ``concurrent.futures.ThreadPoolExecutor``** - it has
no ``_threads``/``_work_queue``/``_max_workers`` for ``pool_health()`` to
read. It exposes a different, PUBLIC introspection surface instead
(``anyio.CapacityLimiter.statistics()`` -> ``borrowed_tokens``/
``total_tokens``/``tasks_waiting``, the anyio-native equivalent), read by
``anyio_pool_health()`` below.

**And unlike the other two, the anyio limiter cannot be resolved from just
anywhere.** ``anyio.to_thread.current_default_thread_limiter()`` raises
``NoEventLoopError`` unless called from INSIDE a running async event loop. The
saturation-watch thread below is a plain (non-async) daemon thread, so it cannot
fetch the limiter itself; the CAPTURED reference must be obtained once from
async code (see ``http_server.py``'s startup, next to where this watch is
started) and passed in as ``anyio_limiter``. The captured *object* stays valid
and live-readable from any thread afterward - only the initial *lookup* needs
the running loop, not each later read.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


def pool_health(executor) -> dict:
    """``{max_workers, threads_spawned, queued, saturated}`` for a
    ``ThreadPoolExecutor``, or a "nothing to report" shape when *executor*
    is ``None`` (the asyncio default executor is created LAZILY on first
    use - see ``default_executor_ref`` below - so it legitimately does not
    exist yet on an idle server).

    Reads ``_threads``/``_work_queue``/``_max_workers`` - private but
    long-stable ``concurrent.futures.ThreadPoolExecutor`` attributes;
    ``concurrent.futures`` exposes no public equivalent for "how many
    workers exist and how many items are queued behind them". Best-effort:
    never raises, degrades to a "cannot report" shape instead.

    ``saturated`` means every worker the pool is allowed to grow to has
    been spawned AND something is still waiting in the queue - i.e. no
    thread is free to pick up new work right now. This is a proxy for "busy
    right now" (there is no public API for a live busy-count either), not
    an exact count: a pool that grew to max_workers and then went idle
    again reports threads_spawned == max_workers with queued == 0, which is
    NOT saturated by this definition (nothing is actually waiting).
    """
    if executor is None:
        return {"max_workers": None, "threads_spawned": 0, "queued": 0, "saturated": False}
    try:
        max_workers = executor._max_workers
        threads_spawned = len(executor._threads)
        queued = executor._work_queue.qsize()
    except Exception as e:
        # Collapsing "could not introspect" into the same shape as "genuinely
        # idle" would let this feature go silently blind if a future
        # Python/uvloop release renamed these attributes. A debug line, not a
        # warning: this is a best-effort diagnostic helper, not a correctness
        # path.
        from localm.debuglog import logger as _dbg
        _dbg.debug("pool_health: could not introspect %r (%s: %s)",
                   executor, type(e).__name__, e)
        return {"max_workers": None, "threads_spawned": None, "queued": None,
                "saturated": False}
    saturated = bool(max_workers) and threads_spawned >= max_workers and queued > 0
    return {"max_workers": max_workers, "threads_spawned": threads_spawned,
            "queued": queued, "saturated": saturated}


# The "nothing to report" shape shared by pool_health(None) and
# anyio_pool_health(None) - kept as one literal so the two pools' idle/absent
# case can never silently drift into looking different from each other.
_NOTHING_TO_REPORT = {"max_workers": None, "threads_spawned": None,
                      "queued": None, "saturated": False}


def anyio_pool_health(limiter) -> dict:
    """Like ``pool_health()``, but for anyio's default worker-thread pool -
    the one ``fastapi.concurrency.run_in_threadpool`` always uses (see the
    module docstring). *limiter* is an ``anyio.CapacityLimiter`` captured
    from async code (``anyio.to_thread.current_default_thread_limiter()``),
    or None when no capture has ever succeeded - a plain-thread caller (the
    saturation watch below) cannot fetch one itself, so unlike
    ``default_executor_ref()``'s lazy "not created yet" None, this None means
    "cannot observe this pool AT ALL from here", not "legitimately idle".
    Both degrade to the same reported shape (never saturated, no counts). The
    STARTUP capture attempt logs its own failure (see ``http_server.py``), so a
    caller wanting to know WHY a given None-shaped pool is unobservable reads
    the startup log, not this function's return value.

    Same ``{max_workers, threads_spawned, queued, saturated}`` shape as
    ``pool_health()`` (``max_workers`` <- ``total_tokens``, ``threads_spawned``
    <- ``borrowed_tokens``, ``queued`` <- ``tasks_waiting``), so both pools
    render identically in ``GET /debug/stacks`` and drive the same saturation-
    watch logic without a special case for anyio."""
    if limiter is None:
        return dict(_NOTHING_TO_REPORT)
    try:
        stats = limiter.statistics()
        max_workers = stats.total_tokens
        threads_spawned = stats.borrowed_tokens
        queued = stats.tasks_waiting
    except Exception as e:
        # Mirrors pool_health()'s own "never trust in silence" comment: a
        # future anyio release could rename CapacityLimiterStatistics'
        # fields, and this must be discoverable, not silently swallowed into
        # "healthy".
        from localm.debuglog import logger as _dbg
        _dbg.debug("anyio_pool_health: could not introspect %r (%s: %s)",
                   limiter, type(e).__name__, e)
        return dict(_NOTHING_TO_REPORT)
    saturated = bool(max_workers) and threads_spawned >= max_workers and queued > 0
    return {"max_workers": max_workers, "threads_spawned": threads_spawned,
            "queued": queued, "saturated": saturated}


def default_executor_ref(loop):
    """The asyncio loop's OWN default executor object, or None if no
    ``run_in_executor(None, ...)`` call has created one yet.

    There is no public ``loop.get_default_executor()`` (only the reverse,
    ``set_default_executor``) - ``_default_executor`` is the private
    attribute both stdlib ``asyncio`` and uvloop's compatible ``Loop``
    populate lazily inside their own ``run_in_executor``. Read via getattr
    with a None default so an alternate loop implementation that does not
    expose this degrades to "cannot report" rather than raising."""
    return getattr(loop, "_default_executor", None)


def executors_snapshot(loop, anyio_limiter=None) -> dict:
    """``{"default": pool_health(...), "plugin": pool_health(...), "anyio":
    anyio_pool_health(...)}`` - the live payload both the saturation watch
    below and ``GET /debug/stacks`` report, kept as one function so the three
    cannot drift apart about which pools exist or what "saturated" means.

    *anyio_limiter* is passed straight to ``anyio_pool_health()`` - None
    (the default) reports anyio's pool as unobservable from here, which is
    correct for any caller that has not captured a live reference (see that
    function's docstring for why this differs from ``default_executor_ref``'s
    lazy-None case)."""
    from localm.executor import get_plugin_executor
    return {
        "default": pool_health(default_executor_ref(loop)),
        "plugin": pool_health(get_plugin_executor()),
        "anyio": anyio_pool_health(anyio_limiter),
    }


# How long a pool must be CONTINUOUSLY saturated before the first WARNING:
# long enough that a legitimate burst (several concurrent embed calls, a
# handful of tool invocations) recovering on its own stays quiet, short enough
# to warn before a systemic exhaustion accumulates.
_DEFAULT_SATURATION_THRESHOLD = 30.0


def start_executor_saturation_watch(loop, *, threshold: float = _DEFAULT_SATURATION_THRESHOLD,
                                    poll: float = 5.0, anyio_limiter=None):
    """Start a plain (non-async) daemon thread that periodically checks all
    three pools via ``executors_snapshot`` and warns once a pool has been
    continuously saturated for *threshold* seconds - then drops to DEBUG for
    as long as it stays saturated, so a genuinely stuck pool logs one line
    every *poll* interval instead of flooding. Recovering (saturated goes
    false) resets the streak, so a LATER re-saturation warns again rather than
    staying silent forever after the first incident.

    *anyio_limiter* must be captured from ASYNC code before this call (see
    the module docstring - this thread cannot fetch it itself) and passed
    straight through to every ``executors_snapshot`` call this thread makes.
    None (the default) means the caller never captured one, in which case
    anyio's pool reports as unobservable for the life of this watch, same as
    if it were never wired in at all - best-effort, never a hard requirement.

    Runs on its own thread, not folded into the hang-watchdog poller: a
    stalled EVENT LOOP and a saturated THREAD POOL are independent failure
    modes with independent detectors and different natural cadences.

    Returns ``(stop_event, thread)`` for teardown, mirroring
    ``_start_hang_watchdog``'s exact shape."""
    stop = threading.Event()
    # name -> monotonic timestamp the pool was FIRST observed saturated in
    # the CURRENT unbroken streak, or None while not saturated. Tracked per
    # pool name so one pool's exhaustion is never blamed on, or hidden
    # behind, the others' state.
    streak_started: "dict[str, Optional[float]]" = {
        "default": None, "plugin": None, "anyio": None}
    warned: "dict[str, bool]" = {"default": False, "plugin": False, "anyio": False}

    def _run() -> None:
        from localm.debuglog import logger as _dbg
        while not stop.wait(poll):
            try:
                snapshot = executors_snapshot(loop, anyio_limiter)
            except Exception:
                # Must never crash the process it is watching - skip this
                # tick, try again next poll.
                continue
            for name, health in snapshot.items():
                if health["saturated"]:
                    now = time.monotonic()
                    if streak_started[name] is None:
                        streak_started[name] = now
                        warned[name] = False
                        continue
                    elapsed = now - streak_started[name]
                    if elapsed < threshold:
                        continue
                    if not warned[name]:
                        _dbg.warning(
                            "%s thread pool has been fully saturated for "
                            "%.0fs (max_workers=%s, queued=%s) - work "
                            "relying on it may be stalled; see "
                            "GET /debug/stacks for live numbers",
                            name, elapsed, health["max_workers"], health["queued"])
                        warned[name] = True
                    else:
                        _dbg.debug(
                            "%s thread pool still saturated (%.0fs, queued=%s)",
                            name, elapsed, health["queued"])
                else:
                    streak_started[name] = None
                    warned[name] = False

    t = threading.Thread(target=_run, name="localm-executor-saturation-watch",
                         daemon=True)
    t.start()
    return stop, t
