# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-pool saturation detectability - the third part of the
thread-pool-exhaustion fix (dev-notes/decisions-2026-07-30-release-gate.md,
Q2): subprocess-isolating HFBackend (see backends/_hf_runner.py) closes the
one KNOWN cause of a permanently-leaked ``run_in_executor(None, ...)``
thread, but a pool can still fill up under ordinary load (a burst of
legitimate work, or a future native call this fix did not anticipate) with
no signal to an operator beyond "the server mysteriously got slow, then
stopped responding". This module makes that state visible rather than
silent (AGENTS.md rule 5) - a WARNING once saturation has held for a
sustained window, plus the live numbers on ``GET /debug/stacks`` - without
inventing a hard gate that could reject legitimate load: this is an
observability signal, not a new failure mode.

Two pools are watched, symmetrically, for the same blast-radius reason
``localm/executor.py``'s own docstring already gives for splitting them in
the first place: the asyncio loop's TRUE default executor (every
``run_in_executor(None, ...)`` call site - GPU/VRAM probes, model
load/unload, embeddings, count_tokens, and now the isolated HF/GGUF runner
RPCs) and the separate ``get_plugin_executor()`` pool (RAG/web/voice/coder
tool calls). Exhaustion in one says nothing about the other, so they are
tracked as independent streaks.
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
    workers exist and how many items are queued behind them", and this
    codebase already tolerates the same class of introspection elsewhere
    (e.g. reading a live GGUF worker's queues for diagnostics). Best-effort:
    never raises, degrades to a "cannot report" shape instead - a
    monitoring helper must never be the thing that crashes the server it is
    watching.

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


def executors_snapshot(loop) -> dict:
    """``{"default": pool_health(...), "plugin": pool_health(...)}`` - the
    live payload both the saturation watch below and ``GET /debug/stacks``
    report, kept as one function so the two can never drift out of sync
    about which pools exist or what "saturated" means."""
    from localm.executor import get_plugin_executor
    return {
        "default": pool_health(default_executor_ref(loop)),
        "plugin": pool_health(get_plugin_executor()),
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
                                    poll: float = 5.0):
    """Start a plain (non-async) daemon thread that periodically checks both
    pools via ``executors_snapshot`` and warns once a pool has been
    continuously saturated for *threshold* seconds - then drops to DEBUG for
    as long as it stays saturated, so a genuinely stuck pool logs one line
    every *poll* interval instead of flooding (the log-once-then-throttle
    lesson this codebase already learned the hard way for the VRAM-probe
    daemon - see .claude/rules/hard-won-rules.md). Recovering (saturated
    goes false) resets the streak, so a LATER re-saturation warns again
    rather than staying silent forever after the first incident.

    Runs on its own thread (not folded into the existing hang-watchdog
    poller) because it answers a different question on a different natural
    cadence - a stalled EVENT LOOP vs a saturated THREAD POOL are
    independent failure modes with independent detectors, matching how
    GET /debug/stacks already documents itself as complementing, not
    replacing, the hang watchdog's file-based capture.

    Returns ``(stop_event, thread)`` for teardown, mirroring
    ``_start_hang_watchdog``'s exact shape."""
    stop = threading.Event()
    # name -> monotonic timestamp the pool was FIRST observed saturated in
    # the CURRENT unbroken streak, or None while not saturated. Tracked per
    # pool name so one pool's exhaustion is never blamed on, or hidden
    # behind, the other's state.
    streak_started: "dict[str, Optional[float]]" = {"default": None, "plugin": None}
    warned: "dict[str, bool]" = {"default": False, "plugin": False}

    def _run() -> None:
        from localm.debuglog import logger as _dbg
        while not stop.wait(poll):
            try:
                snapshot = executors_snapshot(loop)
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
