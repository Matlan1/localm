# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser plugin: a live view of the automated browser, and controls for it.

Routes (mounted by the engine, auto-scoped to the ``browser`` capability):
  POST   /api/browser/session    - open a browser and start streaming it
  GET    /api/browser/agent      - whether an agent-driven browser is available to watch
  POST   /api/browser/agent      - stream the coding agent's own browser to this caller
  POST   /api/browser/navigate   - drive the open browser to a URL
  POST   /api/browser/click      - click a point on the open browser's page
  POST   /api/browser/scroll     - scroll the open browser's page
  POST   /api/browser/key        - send a named key press to the open browser
  POST   /api/browser/type       - type text into the open browser
  POST   /api/browser/stop       - close the browser and end the stream
  GET    /api/browser/state      - whether one is open, and what it reached
  GET    /api/browser/engine     - whether the configured browser can start, and why not
  POST   /api/browser/download   - download the bundled browser, as a background job

The live view is a background job: its worker owns the browser for the job's
lifetime and pushes one ``frame`` event per rendered frame, which the kernel's
existing ``/api/jobs/{id}/events`` SSE endpoint streams unchanged. Frames do not
accumulate in a job's replay history (see ``jobs.FRAME_EVENT``).

The job is LONG-LIVED and interactive, unlike a generation job that runs once and
returns: the navigate route reaches the same live browser through the session
registry while the worker is still streaming it.

Ships DISABLED by default, and every route refuses unless ``browser_enabled`` is
switched on, except ``/stop``, ``/state``, ``/engine`` and ``/download``, which
work regardless: the last two are how a user sets the browser up before turning
it on. Holding the capability is not on its own enough to drive a browser.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from localm import scopes
from localm.browser import discovery, provision
from localm.browser import session as bsession
from localm.inference.http_server import caller_scopes, principal_id
from localm.plugins.gui.jobs import FRAME_EVENT

_router = APIRouter()

#: How often the worker checks whether it has been asked to stop.
_TICK = 0.25

#: The largest magnitude, in CSS pixels, of a click coordinate or wheel delta.
_MAX_PIXELS = 1_000_000.0

#: A click coordinate or wheel delta: a finite number within +/- _MAX_PIXELS.
#: Anything else, NaN and Infinity included, is refused with a 422.
_Pixels = Annotated[float, Field(allow_inf_nan=False,
                                 ge=-_MAX_PIXELS, le=_MAX_PIXELS)]

#: The most characters one /type request may carry. Longer text is refused
#: with a 422.
_MAX_TYPED_TEXT = 4096

#: The longest key name one /key request may carry, modifiers included.
_MAX_KEY_NAME = 64


class OpenRequest(BaseModel):
    url: str | None = None


class NavigateRequest(BaseModel):
    url: str


class ClickRequest(BaseModel):
    x: _Pixels
    y: _Pixels
    button: Literal["left", "right", "middle"] = "left"


class ScrollRequest(BaseModel):
    delta_x: _Pixels = 0.0
    delta_y: _Pixels = 0.0


class KeyRequest(BaseModel):
    key: str = Field(max_length=_MAX_KEY_NAME)


class TypeRequest(BaseModel):
    text: str = Field(max_length=_MAX_TYPED_TEXT)


class WatchAgentRequest(BaseModel):
    # None (the default, and an empty or absent POST body) watches whichever
    # agent-driven browser this caller may see first. An explicit id narrows
    # to one coder session's own browser, applied after
    # _viewable_agent_browsers' own scoping. See
    # test_watch_agent_targets_one_session_among_several and
    # test_watch_agent_cannot_target_a_session_outside_the_scoped_list.
    coder_session_id: str | None = None


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


def _inline_live_view() -> bool:
    from localm.config import load_config
    try:
        return bool(load_config().get("browser_inline_live_view", False))
    except Exception:
        return False


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
    """Open this key's browser. The key's id is claimed before the job starts,
    so a second open while the first is still starting is refused with 409."""
    _require_enabled()
    sid = _gui_session_id(request)
    # Any app built through attach_engine has this registry; reaching the None
    # branch means the router was mounted on an app that never ran it.
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "The live browser view needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    cfg = _settings()
    claim = bsession.reserve(sid)
    if claim is None:
        raise HTTPException(409, "A browser is already open for this key.")

    def _run(job) -> bool:
        try:
            live = bsession.BrowserSession(
                sid,
                headless=cfg["headless"],
                engine=cfg["engine"],
                extra_deny=cfg["deny"],
                extra_allow=cfg["allow"],
                on_frame=lambda data: job.push({"type": FRAME_EVENT, "data": data}),
            )
            live.start()
        except BaseException:
            bsession.release_if(sid, claim)
            raise
        try:
            if not bsession.install(claim, live):
                job.push({"type": "line", "line": "stopped before it was ready"})
                return True
            job.push({"type": "line", "line": "browser ready"})
            if req.url:
                res = live.navigate(req.url)
                job.push({"type": "line",
                          "line": ("opened " + str(res.get("url")) if res.get("ok")
                                   else "refused: " + str(res.get("refused")
                                                          or res.get("error")))})
            while not job.cancel_requested and bsession.get(sid) is live:
                time.sleep(_TICK)
        finally:
            bsession.release_if(sid, live)
            live.stop()
        return True

    try:
        job = jobs.start_fn("browser", _run, owner=principal_id(request),
                            label="Browser session")
    except BaseException:
        bsession.release_if(sid, claim)
        raise
    return {"job_id": job.id, "session_id": sid}


def _viewable_agent_browsers(request: Request) -> list:
    """Live agent-driven browsers this caller is allowed to watch.

    The coder builds its browser with no on_frame, so it emits no frames, and it
    registers under "coder-<job_owner>" - a namespace this tab's own
    "gui-<principal>" can never match. Without this the Browser tab could only
    ever show a browser it opened itself.

    Scoping is DELEGATED to SessionManager.list rather than re-implemented: the
    owner sees every coder session, a scoped key sees only the sessions its own
    principal created. A handed-out key must never be able to watch the owner's
    browser."""
    mgr = getattr(request.app.state, "coder_sessions", None)
    if mgr is None:
        return []
    try:
        from localm.plugins.builtin.coder.plug import _principal_from_request
        is_owner, principal = _principal_from_request(request)
    except Exception:                      # noqa: BLE001
        return []                          # no identity: show nothing
    found = []
    for info in mgr.list(principal=principal, is_owner=is_owner):
        sess = mgr.get(info.get("id"))
        if sess is None:
            continue
        owner = str(getattr(getattr(sess, "agent", None), "job_owner", "") or "")
        if not owner:
            continue
        live = bsession.get("coder-" + owner)
        if live is not None:
            found.append({"session_id": info.get("id"), "browser": live})
    return found


@_router.get("/api/browser/agent")
async def agent_browser_status(request: Request):
    """Whether an agent-driven browser is running that this caller may watch."""
    _require_enabled()
    found = _viewable_agent_browsers(request)
    return {"available": bool(found),
            "session_id": found[0]["session_id"] if found else None}


@_router.post("/api/browser/agent")
async def watch_agent_browser(request: Request,
                              req: WatchAgentRequest | None = None):
    """Stream the browser the coding agent is driving, to this caller.

    The agent's session emits no frames until a viewer asks for them, so this
    turns its screencast on for as long as the view is open and off again after,
    leaving the agent's own browsing untouched either way."""
    _require_enabled()
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "The live browser view needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    found = _viewable_agent_browsers(request)
    target = req.coder_session_id if req is not None else None
    if target:
        found = [f for f in found if f["session_id"] == target]
    if not found:
        raise HTTPException(404, "No agent is driving a browser right now.")
    entry = found[0]
    live = entry["browser"]

    def _run(job) -> bool:
        ok = live.enable_live_view(
            lambda data: job.push({"type": FRAME_EVENT, "data": data}))
        if not ok:
            job.push({"type": "line",
                      "text": "this browser build has no live view"})
            return False
        job.push({"type": "line", "line": "watching the agent browser"})
        try:
            while (not job.cancel_requested
                   and bsession.get(live.session_id) is live):
                time.sleep(_TICK)
        finally:
            # The agent keeps browsing; only the viewer goes away.
            live.disable_live_view()
        return True

    job = jobs.start_fn("browser", _run, owner=principal_id(request),
                        label="Agent browser view")
    return {"job_id": job.id, "session_id": entry["session_id"]}


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


@_router.post("/api/browser/click")
async def click_coords(req: ClickRequest, request: Request):
    _require_enabled()
    live = bsession.get(_gui_session_id(request))
    if live is None:
        raise HTTPException(404, "No browser is open. Open one first.")
    from localm.executor import get_plugin_executor
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_plugin_executor(),
        lambda: live.click_coords(req.x, req.y, button=req.button))


@_router.post("/api/browser/scroll")
async def scroll(req: ScrollRequest, request: Request):
    _require_enabled()
    live = bsession.get(_gui_session_id(request))
    if live is None:
        raise HTTPException(404, "No browser is open. Open one first.")
    from localm.executor import get_plugin_executor
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_plugin_executor(),
        lambda: live.scroll(req.delta_x, req.delta_y))


@_router.post("/api/browser/key")
async def press_key(req: KeyRequest, request: Request):
    _require_enabled()
    live = bsession.get(_gui_session_id(request))
    if live is None:
        raise HTTPException(404, "No browser is open. Open one first.")
    from localm.executor import get_plugin_executor
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_plugin_executor(),
        lambda: live.press_key(req.key))


@_router.post("/api/browser/type")
async def type_text(req: TypeRequest, request: Request):
    _require_enabled()
    live = bsession.get(_gui_session_id(request))
    if live is None:
        raise HTTPException(404, "No browser is open. Open one first.")
    from localm.executor import get_plugin_executor
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_plugin_executor(),
        lambda: live.type_text(req.text))


@_router.post("/api/browser/stop")
async def stop_browser(request: Request):
    sid = _gui_session_id(request)
    closed = bsession.close(sid)
    return {"closed": closed}


@_router.get("/api/browser/state")
async def state(request: Request):
    live = bsession.get(_gui_session_id(request))
    if live is None:
        return {"open": False, "enabled": _enabled(),
                "inlineLiveView": _inline_live_view()}
    return {
        "open": True,
        "enabled": _enabled(),
        "inlineLiveView": _inline_live_view(),
        "headless": live.headless,
        "engine": live.engine,
        "browser": live.browser_name,
        "blocked": live.blocked_requests()[-50:],
        "allowed": live.allowed_requests()[-50:],
        "console": live.console_messages()[-50:],
    }


#: The running bundled-browser download as ``(job id, owner)``, or None.
#: Guarded by ``_download_lock``.
_download: tuple | None = None
_download_lock = threading.Lock()


def _playwright_installed() -> bool:
    return importlib.util.find_spec("playwright") is not None


def _engine_status(can_write: bool) -> dict:
    """What the configured engine needs before it can start a browser."""
    engine = _settings()["engine"]
    have_playwright = _playwright_installed()
    bundled = provision.is_chromium_installed() if have_playwright else False
    system = [b.name for b in discovery.find_system_browsers()]
    ready = have_playwright and (bundled if engine == "bundled" else bool(system))
    problem = None
    if not have_playwright:
        problem = "playwright_missing"
    elif not ready:
        problem = "bundled_missing" if engine == "bundled" else "system_missing"
    blocked = None
    if not provision.download_allowed():
        blocked = "network"
    elif not can_write:
        blocked = "permission"
    with _download_lock:
        downloading = _download is not None
    return {
        "engine": engine,
        "ready": ready,
        "problem": problem,
        "bundled_installed": bundled,
        "system_browsers": system,
        "looked_for": list(discovery.LOOKED_FOR),
        "can_download": have_playwright and not bundled and blocked is None,
        "download_blocked": blocked if have_playwright and not bundled else None,
        "downloading": downloading,
    }


def _can_write_config(request: Request) -> bool:
    held = caller_scopes(request)
    return held is None or scopes.grants(held, scopes.CONFIG_WRITE)


@_router.get("/api/browser/engine")
async def engine_status(request: Request):
    """Whether the configured browser engine can start a browser right now.

    ``problem`` is None when it can, else ``bundled_missing`` (the download has
    not been done), ``system_missing`` (no installed browser was found) or
    ``playwright_missing`` (the browser extra is not installed).
    ``can_download`` is a UI hint for the download action; ``download_blocked``
    says why it is off (``network`` or ``permission``). POST
    /api/browser/download re-checks both."""
    from localm.executor import get_plugin_executor
    can_write = _can_write_config(request)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_plugin_executor(),
                                      lambda: _engine_status(can_write))


@_router.post("/api/browser/download")
async def download_browser(request: Request):
    """Download the bundled browser once, as a background job.

    Writes no setting. Needs config:write, the scope that governs the network
    policy itself, so a key that could not lift the policy cannot bypass it
    here. Refused under ``net_mode=off`` unless downloads are exempted, and a
    no-op when the browser is already installed. A second request while one
    download runs from the same key returns that job."""
    from localm.executor import get_plugin_executor
    if not _can_write_config(request):
        raise HTTPException(
            403, "Downloading the browser needs the config:write scope (the "
                 "same permission that governs the network policy).")
    if not _playwright_installed():
        raise HTTPException(
            409, "The browser automation extra is not installed. Install it "
                 f"with:  {provision.PIP_INSTALL_HINT}")
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "The browser download needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    loop = asyncio.get_running_loop()
    if await loop.run_in_executor(get_plugin_executor(),
                                  provision.is_chromium_installed):
        return {"status": "already_installed"}
    if not provision.download_allowed():
        raise HTTPException(
            409, "Network access is disabled (net_mode=off), which blocks even "
                 "an explicitly requested download. Set net_mode to ask or "
                 "allow, or turn on \"Allow model downloads while network "
                 "access is off\", first.")
    global _download
    owner = principal_id(request)

    def _run(job) -> bool:
        global _download
        try:
            job.push({"type": "line", "text": "Downloading the browser "
                      "(one-time, a few hundred MB)..."})
            result = provision.install_chromium(
                on_progress=lambda line: job.push({"type": "line", "text": line}))
            if not result.ok:
                job.push({"type": "line", "text": "error: " + result.message})
                return False
            job.push({"type": "line", "text": "Ready: " + result.message})
            return True
        finally:
            with _download_lock:
                _download = None

    with _download_lock:
        if _download is not None:
            running_id, running_owner = _download
            if running_owner != owner:
                raise HTTPException(409, "A browser download is already running.")
            return {"job_id": running_id, "status": "running"}
        job = jobs.start_fn("browser-download", _run, owner=owner,
                            label="Download the browser")
        _download = (job.id, owner)
    return {"job_id": job.id, "status": "started"}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    bsession.close_all()
