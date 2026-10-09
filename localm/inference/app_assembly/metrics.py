# SPDX-License-Identifier: AGPL-3.0-or-later
"""The opt-in GET /metrics endpoint and the request-timing middleware behind it.

Both exist only when the ``metrics_enabled`` setting (or ``LOCALM_METRICS``) is
on at startup; with it off the route is not registered and nothing is collected.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.routing import Route

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference import metrics

ENV_VAR = "LOCALM_METRICS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def metrics_enabled() -> bool:
    """Whether /metrics is switched on: ``LOCALM_METRICS`` or the
    ``metrics_enabled`` setting. An unreadable config reads as off."""
    if os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY:
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("metrics_enabled", False))
    except Exception:
        from localm.debuglog import logger as _dbg
        _dbg.warning("metrics: config unreadable, /metrics stays off")
        return False


class MetricsMiddleware:
    """Pure-ASGI request counter and timer. Records the route TEMPLATE the
    router matched, never the requested path."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = 500
        started = time.perf_counter()

        async def _send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        metrics.request_started()
        try:
            await self.app(scope, receive, _send)
        finally:
            route = scope.get("route")
            metrics.request_finished(
                scope.get("method"),
                metrics.route_label(route if isinstance(route, Route) else None),
                status, time.perf_counter() - started)


def _queue_depth() -> Optional[int]:
    """Requests waiting on any model's inference slot, or None when the
    semaphore does not expose its waiters (the series is then left out)."""
    sems = {id(s): s for s in (*_hs._inference_sems.values(), _hs._inference_sem)
            if s is not None}
    if not all(hasattr(s, "_waiters") for s in sems.values()):
        from localm.debuglog import logger as _dbg
        _dbg.warning("metrics: asyncio.Semaphore has no _waiters; "
                     "queue depth not reported")
        return None
    return sum(len(s._waiters or ()) for s in sems.values())


def _models_loaded() -> int:
    return sum(1 for e in list(_hs._engines.values()) if getattr(e, "loaded", False))


def _vram_gauges() -> list:
    """Used and total GPU memory from the cached reading (it never probes on the
    request path); a figure the reading does not trust is left out."""
    from localm import sysstats
    vram = (sysstats._vram() or {}).get("vram") or {}
    return [("localm_vram_used_bytes", vram.get("used")),
            ("localm_vram_total_bytes", vram.get("total"))]


def add_metrics(app: FastAPI) -> None:
    """Register /metrics and the timing middleware when metrics are enabled;
    otherwise reset the collector to off and add nothing. Added last, so the
    middleware is the outermost layer."""
    enabled = metrics_enabled()
    metrics.configure(enabled)
    if not enabled:
        return

    # Keyless servers (no API key anywhere) answer only on a loopback bind,
    # decided on app.state.bind_host and never on the request peer; the
    # open-mode shell-token and cross-origin gates in security.py cover the
    # loopback case. A keyed server needs an admin-scoped key.
    @app.get("/metrics", include_in_schema=False,
             dependencies=[Depends(_hs.require_scope(scopes.ADMIN))])
    async def _metrics(request: Request):
        from localm.auth import any_key_configured
        host = getattr(request.app.state, "bind_host", "127.0.0.1")
        if not any_key_configured() and not _hs._is_loopback_host(host):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        gauges = [("localm_inference_queue_depth", _queue_depth()),
                  ("localm_models_loaded", _models_loaded()),
                  *_vram_gauges()]
        return Response(metrics.render(gauges), media_type=metrics.CONTENT_TYPE,
                        headers={"Cache-Control": "no-store"})

    app.add_middleware(MetricsMiddleware)
