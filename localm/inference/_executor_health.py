# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-pool saturation detectability - the third part of the thread-pool-exhaustion fix (dev-notes/decisions-2026-07-30-release-gate.md, Q2): subprocess-isolating HFBackend (see backends/_hf_runner.py) closes the one KNOWN cause of a permanently-leaked ``run_in_executor(None, ...)`` thread, but a po..."""

from __future__ import annotations

import threading
import time
from typing import Optional


def pool_health(executor) -> dict:
    """``{max_workers, threads_spawned, queued, saturated}`` for a ``ThreadPoolExecutor``, or a 'nothing to report' shape when *executor* is ``None`` (the asyncio default executor is created LAZILY on first use - see ``default_executor_ref`` below - so it legitimately does not exist yet on an idle server)."""
    if executor is None:
        return {"max_workers": None, "threads_spawned": 0, "queued": 0, "saturated": False}
    try:
        max_workers = executor._max_workers
        threads_spawned = len(executor._threads)
        queued = executor._work_queue.qsize()
    except Exception as e:
        # Collapsing "could not introspect" into the same shape as "genuinely
        # idle" would let this whole feature go silently blind if a future
        # Python/uvloop release ever renames these attributes - never trust
        # in silence (rule 5). A debug line, not a warning: this is a
        # best-effort diagnostic helper, not a correctness path, so a broken
        # assumption here should be discoverable, not escalated into noise.
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
    """Like ``pool_health()``, but for anyio's default worker-thread pool - the one ``fastapi.concurrency.run_in_threadpool`` always uses (see the module docstring). *limiter* is an ``anyio.CapacityLimiter`` captured from async code (``anyio.to_thread.current_default_thread_limiter()``), or None when no ca..."""
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
    """The asyncio loop's OWN default executor object, or None if no ``run_in_executor(None, ...)`` call has created one yet."""
    return getattr(loop, "_default_executor", None)


def executors_snapshot(loop, anyio_limiter=None) -> dict:
    """``{'default': pool_health(...), 'plugin': pool_health(...), 'anyio': anyio_pool_health(...)}`` - the live payload both the saturation watch below and ``GET /debug/stacks`` report, kept as one function so the three can never drift out of sync about which pools exist or what 'saturated' means."""
    from localm.executor import get_plugin_executor
    return {
        "default": pool_health(default_executor_ref(loop)),
        "plugin": pool_health(get_plugin_executor()),
        "anyio": anyio_pool_health(anyio_limiter),
    }


# How long a pool must be CONTINUOUSLY saturated before the first WARNING -
# long enough that a legitimate burst (several concurrent embed calls, a
# handful of tool invocations) recovering on its own stays quiet, short
# enough to warn well before a systemic exhaustion (16 hangs on this box's
# default pool size) could ever accumulate. A best-effort observability
# threshold, not a measured/tuned constant - revisit if it proves noisy or
# too slow to fire in practice.
_DEFAULT_SATURATION_THRESHOLD = 30.0


def start_executor_saturation_watch(loop, *, threshold: float = _DEFAULT_SATURATION_THRESHOLD,
                                    poll: float = 5.0, anyio_limiter=None):
    """Start a plain (non-async) daemon thread that periodically checks all three pools via ``executors_snapshot`` and warns once a pool has been continuously saturated for *threshold* seconds - then drops to DEBUG for as long as it stays saturated, so a genuinely stuck pool logs one line every *poll* inte..."""
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
