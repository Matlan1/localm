# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel diagnostics ``create_app()`` wires in: the debug-mode request log and
GET /debug/stacks. Both report the event-loop lag the lifespan's heartbeat
measures, and /debug/stacks decides on ``app.state.bind_host``, never on the
request peer."""

from __future__ import annotations

import asyncio
import os
import sys
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

import localm.inference.http_server as _hs


def add_request_logging(app: FastAPI) -> None:
    """In debug mode (decided once, here), log every request with its status,
    timing and loop lag. Added before any other middleware, so it is the
    innermost one."""
    # Debug mode: log every request with timing to the debug log file
    from localm.debuglog import debug_enabled, logger as _dbg
    if debug_enabled():
        @app.middleware("http")
        async def _log_requests(request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            # loop_lag = real scheduling delay (_loop_lag_seconds, see the
            # comment above _hb_monotonic) - ~0 on a healthy server, and only
            # positive when a preceding event-loop stall pushed the last
            # heartbeat tick late. This is NOT time-since-last-tick, which
            # saws 0..1s even when nothing is wrong. Resolution limit: a stall
            # shorter than _HEARTBEAT_INTERVAL_S also reads 0.0 (see
            # _loop_lag_seconds' docstring) - 0.0 means "no stall LONGER than the
            # interval", not "no stall at all". None (cold start, before the
            # heartbeat's first tick) renders as "n/a", never as 0.0, so a request
            # served during startup is not reported as identically healthy to one
            # with a real lag measurement behind it.
            lag = _hs._loop_lag_seconds()
            lag_str = f"{lag:.2f}s" if lag is not None else "n/a"
            _dbg.debug(
                "%s %s -> %d (%.0f ms, loop_lag=%s)",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
                lag_str,
            )
            return response


def add_debug_stacks(app: FastAPI) -> None:
    """Register GET /debug/stacks."""
    # Loopback-only debug endpoint: every thread's stack + the asyncio task list,
    # for diagnosing a hang/slowdown from the SAME machine on demand. 404'd off
    # loopback. NOTE: served ON the event loop, so it answers only while the loop
    # is alive (a partial stall, a task backlog). A FULLY wedged loop cannot
    # respond here at all - that case is captured by the off-loop watchdog file
    # (LOCALM_HANG_WATCHDOG); this endpoint complements it.
    #
    # THREE gates, because the first one alone is not enough:
    #   1. Depends(require_fs_host) - meaningful in PROTECTED mode (a key's
    #      fs_access dial), but in DEFAULT KEYLESS mode effective_fs_access
    #      returns "host" for EVERY caller, so on its own it is a tautology.
    #   2. the open-mode shell-token gate, via _SHELL_TOKEN_GETS in the origin
    #      guard (security.py) - that is what makes gate 1 non-vacuous in
    #      keyless mode.
    #   3. the bind_host loopback check below, on app.state.bind_host (what the
    #      server actually BOUND to) and never request.client.host: behind
    #      portmux the request peer is always 127.0.0.1, so the peer address
    #      cannot distinguish a loopback client from a LAN one.
    # Frame text is path-scrubbed on the way out (see below).
    @app.get("/debug/stacks", include_in_schema=False,
             dependencies=[Depends(_hs.require_fs_host)])
    async def _debug_stacks(request: Request):
        host = getattr(request.app.state, "bind_host", "127.0.0.1")
        if not _hs._is_loopback_host(host):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        import traceback

        from localm.pathscrub import path_scrubber
        # traceback.format_stack emits absolute paths ('File "<install>/localm/
        # inference/http_server.py", line N') which name the install dir and, on
        # a per-user install, the OS account. Redact the DIRECTORY only: the file
        # name, line number, function and source line all survive, so this stays
        # a usable hang diagnosis: scrubbed, not muted. Bound once
        # rather than per string: a dump is hundreds of frames and each prefix
        # resolve is a filesystem call.
        scrub = path_scrubber()
        threads = {str(tid): [scrub(line) for line in traceback.format_stack(frame)]
                   for tid, frame in sys._current_frames().items()}
        tasks = []
        try:
            for task in asyncio.all_tasks():
                tasks.append({
                    "name": task.get_name(),
                    "done": task.done(),
                    "stack": [scrub(str(f)) for f in task.get_stack(limit=20)],
                })
        except RuntimeError:
            pass   # no running loop (should not happen inside an async handler)
        # Thread-pool saturation numbers, live here too and not just in the
        # periodic warning log, so a diagnosis in progress does not have to wait
        # for the threshold to trip a line.
        # This handler IS async with a running loop, so unlike the background
        # saturation watch it can fetch anyio's default thread limiter fresh
        # on every call - no captured reference needed for THIS consumer.
        try:
            from localm.inference._executor_health import executors_snapshot
            import anyio.to_thread
            executors = executors_snapshot(
                asyncio.get_running_loop(),
                anyio_limiter=anyio.to_thread.current_default_thread_limiter())
        except Exception:
            executors = {}
        # None (cold start, before the heartbeat's first tick) renders as
        # JSON null, never as 0.0, so a stack dump taken during startup is not
        # reported as identically healthy to one taken with a real lag
        # measurement behind it.
        lag = _hs._loop_lag_seconds()
        return {"pid": os.getpid(),
                "loop_lag_s": round(lag, 2) if lag is not None else None,
                "threads": threads, "tasks": tasks, "executors": executors}
