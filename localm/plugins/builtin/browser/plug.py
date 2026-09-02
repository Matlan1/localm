# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser plugin: a live view of the automated browser, and controls for it.

Routes (mounted by the engine, auto-scoped to the ``browser`` capability):
  POST   /api/browser/session    - open a browser and start streaming it
  POST   /api/browser/navigate   - drive the open browser to a URL
  POST   /api/browser/stop       - close the browser and end the stream
  GET    /api/browser/state      - whether one is open, and what it reached

The live view is a background job: its worker owns the browser for the job's
lifetime and pushes one ``frame`` event per rendered frame, which the kernel's
existing ``/api/jobs/{id}/events`` SSE endpoint streams unchanged. Frames do not
accumulate in a job's replay history (see ``jobs.FRAME_EVENT``).

The job is LONG-LIVED and interactive, unlike a generation job that runs once and
returns: the navigate route reaches the same live browser through the session
registry while the worker is still streaming it.

Ships DISABLED by default, and every route refuses unless ``browser_enabled`` is
switched on, so holding the capability is not on its own enough to drive it.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from localm.browser import session as bsession
from localm.inference.http_server import principal_id
from localm.plugins.gui.jobs import FRAME_EVENT

_router = APIRouter()

#: How often the worker checks whether it has been asked to stop.
_TICK = 0.25


class OpenRequest(BaseModel):
    url: str | None = None


class NavigateRequest(BaseModel):
    url: str


def _enabled() -> bool:
    from localm.config import load_config
    try:
        return bool(load_config().get("browser_enabled", False))
    except Exception:
        return False


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(
            409, "Browser automation is switched off. Turn it on in "
                 "Settings > Network before opening a browser.")


def _settings() -> dict:
    from localm.config import load_config
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    custom = bool(cfg.get("browser_custom_domain_rules", False))
    engine = str(cfg.get("browser_engine", "bundled") or "bundled")
    return {
        "headless": bool(cfg.get("browser_headless", True)),
        "engine": engine if engine in ("bundled", "system") else "bundled",
        "deny": list(cfg.get("browser_deny") or []) if custom else [],
        "allow": list(cfg.get("browser_allow") or []) if custom else [],
    }


def _gui_session_id(request: Request) -> str:
    """One live browser per principal, so a second open reuses the first."""
    return "gui-" + (principal_id(request) or "owner")


@_router.post("/api/browser/session")
async def open_browser(req: OpenRequest, request: Request):
    _require_enabled()
    sid = _gui_session_id(request)
    if bsession.get(sid) is not None:
        raise HTTPException(409, "A browser is already open for this key.")
    # Any app built through attach_engine has this registry; reaching the None
    # branch means the router was mounted on an app that never ran it.
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "The live browser view needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    cfg = _settings()

    def _run(job) -> bool:
        live = bsession.BrowserSession(
            sid,
            headless=cfg["headless"],
            engine=cfg["engine"],
            extra_deny=cfg["deny"],
            extra_allow=cfg["allow"],
            on_frame=lambda data: job.push({"type": FRAME_EVENT, "data": data}),
        )
        live.start()
        bsession.register(live)
        job.push({"type": "line", "line": "browser ready"})
        try:
            if req.url:
                res = live.navigate(req.url)
                job.push({"type": "line",
                          "line": ("opened " + str(res.get("url")) if res.get("ok")
                                   else "refused: " + str(res.get("refused")
                                                          or res.get("error")))})
            while not job.cancel_requested:
                time.sleep(_TICK)
        finally:
            bsession.close(sid)
        return True

    job = jobs.start_fn("browser", _run, owner=principal_id(request),
                        label="Browser session")
    return {"job_id": job.id, "session_id": sid}


@_router.post("/api/browser/navigate")
async def navigate(req: NavigateRequest, request: Request):
    _require_enabled()
    live = bsession.get(_gui_session_id(request))
    if live is None:
        raise HTTPException(404, "No browser is open. Open one first.")
    from localm.executor import get_plugin_executor
    import asyncio
    loop = asyncio.get_running_loop()
    # navigate() blocks on the browser's own loop, so it never runs on this one.
    return await loop.run_in_executor(get_plugin_executor(),
                                      lambda: live.navigate(req.url))


@_router.post("/api/browser/stop")
async def stop_browser(request: Request):
    sid = _gui_session_id(request)
    closed = bsession.close(sid)
    return {"closed": closed}


@_router.get("/api/browser/state")
async def state(request: Request):
    live = bsession.get(_gui_session_id(request))
    if live is None:
        return {"open": False, "enabled": _enabled()}
    return {
        "open": True,
        "enabled": _enabled(),
        "headless": live.headless,
        "engine": live.engine,
        "blocked": live.blocked_requests()[-50:],
        "allowed": live.allowed_requests()[-50:],
        "console": live.console_messages()[-50:],
    }


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    bsession.close_all()
