# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel diagnostics ``create_app()`` wires in: the debug-mode request log (the
outermost middleware) and GET /debug/stacks. Both report the event-loop lag the lifespan's heartbeat
measures, and /debug/stacks decides on ``app.state.bind_host``, never on the
request peer. Also the ``FastAPI()`` options that keep FastAPI's own telemetry
off."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

import localm.inference.http_server as _hs


def fastapi_telemetry_off() -> dict:
    """Keyword arguments for ``FastAPI()`` that turn off FastAPI's built-in
    OpenTelemetry support: tracing, metrics, logs, and the exporters it would
    otherwise configure at startup from the ``OTEL_*`` environment variables.
    Empty when the installed FastAPI has no ``telemetry`` parameter."""
    if "telemetry" not in inspect.signature(FastAPI.__init__).parameters:
        return {}
    return {"telemetry": {"auto_configure": False, "tracing": False,
                          "metrics": False, "logs": False}}


class RequestLogMiddleware:
    """Pure-ASGI debug request log: exactly one DEBUG line per HTTP request,

        <METHOD> <path> -> <status> (<ms> ms, loop_lag=<lag>)[ refused: <reason>]

    written when the response starts, so ``ms`` is the time to the response
    headers. ``loop_lag`` is the longest event-loop stall that overlapped the
    request, ``n/a`` before the heartbeat's first tick. A request that ends
    before any response is logged with 499, one whose app raised first with
    500. The refusal reason is what a middleware recorded with
    ``http_server._note_refusal``; a refused CORS preflight is named as one.
    Logs the path only: never the query string, headers or body. Passes
    ``receive`` and every message through unchanged."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from localm.debuglog import logger as _dbg
        start = time.perf_counter()
        start_mono = time.monotonic()
        note: dict = {}
        scope[_hs.REQUEST_LOG_SCOPE_KEY] = note
        logged = False

        def _log(status: int) -> None:
            nonlocal logged
            logged = True
            lag = _hs._loop_stall_between(start_mono, time.monotonic())
            lag_str = f"{lag:.2f}s" if lag is not None else "n/a"
            reason = note.get("refusal")
            if reason is None and status >= 400 and _is_cors_preflight(scope):
                reason = "CORS preflight refused"
            _dbg.debug(
                "%s %s -> %d (%.0f ms, loop_lag=%s)%s",
                scope.get("method", "?"), Request(scope).url.path, status,
                (time.perf_counter() - start) * 1000, lag_str,
                f" refused: {reason}" if reason else "",
            )

        async def _send(message) -> None:
            if not logged and message["type"] == "http.response.start":
                _log(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except asyncio.CancelledError:
            if not logged:
                _log(499)
            raise
        except BaseException:
            if not logged:
                _log(500)
            raise
        if not logged:
            _log(499)


def _is_cors_preflight(scope) -> bool:
    """True for an OPTIONS request carrying Origin and
    Access-Control-Request-Method, the shape CORSMiddleware answers itself."""
    if scope.get("method") != "OPTIONS":
        return False
    names = {k for k, _ in scope.get("headers") or ()}
    return b"origin" in names and b"access-control-request-method" in names


def add_request_logging(app: FastAPI) -> None:
    """In debug mode (decided once, here), add :class:`RequestLogMiddleware`.
    Call after every other middleware is added, so it is the outermost layer
    and also logs the requests a middleware answers itself (the Host header
    guard, the body size cap, CORS preflight, the origin and open-mode gates,
    the keyless network gate, the docs gate)."""
    from localm.debuglog import debug_enabled
    if debug_enabled():
        app.add_middleware(RequestLogMiddleware)


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
        # A caller other than the local operator also gets every other host
        # folder replaced (path_formatter).
        own_scrub = path_scrubber()
        show = _hs.path_formatter(request)

        def scrub(text: str) -> str:
            return show(own_scrub(text))

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
        executors_error = None
        try:
            from localm.inference._executor_health import executors_snapshot
            import anyio.to_thread
            executors = executors_snapshot(
                asyncio.get_running_loop(),
                anyio_limiter=anyio.to_thread.current_default_thread_limiter())
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.warning("/debug/stacks: executor snapshot failed: %s", e, exc_info=True)
            executors = {}
            executors_error = "the executor snapshot failed; the server log has the details"
        # Longest loop stall in the last minute; JSON null before the
        # heartbeat's first tick.
        lag = _hs._loop_lag_seconds()
        return {"pid": os.getpid(),
                "loop_lag_s": round(lag, 2) if lag is not None else None,
                "threads": threads, "tasks": tasks, "executors": executors,
                "executors_error": executors_error}
