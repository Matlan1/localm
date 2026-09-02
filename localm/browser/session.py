# SPDX-License-Identifier: AGPL-3.0-or-later
"""A live automated-browser session, and the registry that finds one again.

THREADING CONTRACT, which the rest of this package depends on: every Playwright
object belongs to ONE event loop running on ONE thread that this session owns.
The public methods here are ordinary blocking calls, made from whatever thread a
route handler or an agent tool happens to be on; each marshals its work onto that
loop with ``run_coroutine_threadsafe`` and waits for it. Never touch ``_page``,
``_browser`` or ``_ctx`` from outside that loop.

Every request the browser makes is put through ``netgate`` before it may
proceed, and a refusal is recorded so a caller can be told which destination was
blocked and why.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from localm.browser import netgate

logger = logging.getLogger(__name__)

#: How long a marshalled call may take before the caller gives up on it.
DEFAULT_CALL_TIMEOUT = 45.0

#: Console lines, allowed URLs and blocked-request records kept per session.
_LOG_CAP = 500


class BrowserUnavailableError(RuntimeError):
    """Playwright, or the browser build it pins, is not installed."""


@dataclass
class Blocked:
    url: str
    reason: str


@dataclass
class SessionState:
    console: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    requests: list = field(default_factory=list)


def _require_playwright():
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise BrowserUnavailableError(
            'The browser automation extra is not installed. Install it with '
            'pip install "localm[browser]" then download the browser it drives '
            "with:  python -m playwright install chromium") from exc
    return async_playwright


class BrowserSession:
    """One Chromium session, owned by its own thread and event loop."""

    def __init__(self, session_id: str, *, headless: bool = True,
                 extra_deny=(), extra_allow=(), engine: str = "bundled"):
        self.session_id = session_id
        self.headless = headless
        self.engine = engine or "bundled"
        self.extra_deny = tuple(extra_deny)
        self.extra_allow = tuple(extra_allow)
        self.state = SessionState()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._pw = None
        self._browser = None
        self._ctx = None
        self._page = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, timeout: float = 90.0) -> None:
        """Launch the browser and block until it is ready to drive."""
        _require_playwright()
        self._thread = threading.Thread(
            target=self._run_loop, name="browser-" + self.session_id, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise BrowserUnavailableError("the browser did not start in time")
        if self._start_error is not None:
            raise self._start_error

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._launch())
        except BaseException as exc:                 # noqa: BLE001
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _launch(self) -> None:
        async_playwright = _require_playwright()
        self._pw = await async_playwright().start()
        # "system" drives the browser already installed on this machine, which
        # carries its real logged-in sessions; "bundled" launches the build
        # localm downloaded, with a fresh profile.
        launch = {"headless": self.headless}
        if self.engine == "system":
            launch["channel"] = "chrome"
        try:
            self._browser = await self._pw.chromium.launch(**launch)
        except Exception as exc:
            if self.engine == "system":
                raise BrowserUnavailableError(
                    "Could not start the system browser (Google Chrome). Install "
                    "it, or set the browser engine back to 'bundled'. "
                    + str(exc)) from exc
            raise
        self._ctx = await self._browser.new_context()
        self._page = await self._ctx.new_page()
        self._page.on("console", self._on_console)
        # Routed on the CONTEXT rather than the page, so a popup or a second
        # page the site opens is gated too.
        await self._ctx.route("**/*", self._on_route)
        # WebSockets are NOT covered by route(): they need their own handler, and
        # a routed WebSocket does not reach the server unless connect_to_server()
        # is called, so an unhandled one fails closed.
        await self._ctx.route_web_socket("**/*", self._on_ws_route)

    def stop(self, timeout: float = 30.0) -> None:
        """Close the browser and stop the loop. Safe to call more than once."""
        if self._closed or self._loop is None:
            self._closed = True
            return
        self._closed = True
        try:
            fut = asyncio.run_coroutine_threadsafe(self._teardown(), self._loop)
            fut.result(timeout)
        except Exception as exc:                     # noqa: BLE001
            logger.debug("browser %s teardown: %s", self.session_id, exc)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=timeout)

    async def _teardown(self) -> None:
        for closer in (self._ctx, self._browser):
            try:
                if closer is not None:
                    await closer.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass

    # -- marshalling -------------------------------------------------------- #

    def _call(self, make_coro: Callable[[], Any],
              timeout: float = DEFAULT_CALL_TIMEOUT) -> Any:
        """Run *make_coro()* on this session's own loop and return its result."""
        if self._closed or self._loop is None:
            raise BrowserUnavailableError("this browser session is closed")
        fut = asyncio.run_coroutine_threadsafe(make_coro(), self._loop)
        return fut.result(timeout)

    # -- request gating ----------------------------------------------------- #

    async def _on_route(self, route, request) -> None:
        url = request.url
        reason = await netgate.decide_async(
            url, extra_deny=self.extra_deny, extra_allow=self.extra_allow)
        if reason is None:
            if len(self.state.requests) < _LOG_CAP:
                self.state.requests.append(url)
            await route.continue_()
            return
        if len(self.state.blocked) < _LOG_CAP:
            self.state.blocked.append(Blocked(url=url, reason=reason))
        logger.info("browser %s blocked %s: %s", self.session_id, url, reason)
        await route.abort()

    async def _on_ws_route(self, ws) -> None:
        url = getattr(ws, "url", "") or ""
        reason = await netgate.decide_async(
            url, extra_deny=self.extra_deny, extra_allow=self.extra_allow)
        if reason is None:
            if len(self.state.requests) < _LOG_CAP:
                self.state.requests.append(url)
            ws.connect_to_server()
            return
        if len(self.state.blocked) < _LOG_CAP:
            self.state.blocked.append(Blocked(url=url, reason=reason))
        logger.info("browser %s blocked websocket %s: %s",
                    self.session_id, url, reason)
        await ws.close()

    def _on_console(self, msg) -> None:
        if len(self.state.console) < _LOG_CAP:
            try:
                self.state.console.append({"type": msg.type, "text": msg.text})
            except Exception:
                pass

    # -- the driving surface ------------------------------------------------ #

    def navigate(self, url: str, *, timeout_ms: int = 30000) -> dict:
        """Go to *url*. A refusal by the gate is reported as ``refused``."""
        pre = netgate.decide(url, extra_deny=self.extra_deny,
                             extra_allow=self.extra_allow)
        if pre is not None:
            return {"ok": False, "url": url, "refused": pre}

        async def go():
            resp = await self._page.goto(url, timeout=timeout_ms)
            return {"ok": True, "url": self._page.url,
                    "status": resp.status if resp else None,
                    "title": await self._page.title()}
        try:
            return self._call(go)
        except Exception as exc:                     # noqa: BLE001
            blocked = [b.reason for b in self.state.blocked if b.url == url]
            return {"ok": False, "url": url, "error": str(exc),
                    "refused": blocked[0] if blocked else None}

    def read_text(self, *, max_chars: int = 20000) -> str:
        async def read():
            return await self._page.inner_text("body")
        try:
            return self._call(read)[:max_chars]
        except Exception as exc:                     # noqa: BLE001
            return "(could not read the page: " + str(exc) + ")"

    def screenshot(self, *, full_page: bool = False) -> bytes:
        async def shot():
            return await self._page.screenshot(full_page=full_page, type="png")
        return self._call(shot)

    def click(self, selector: str, *, timeout_ms: int = 10000) -> dict:
        async def do():
            await self._page.click(selector, timeout=timeout_ms)
            return {"ok": True, "selector": selector, "url": self._page.url}
        try:
            return self._call(do)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "selector": selector, "error": str(exc)}

    def fill(self, selector: str, value: str, *, timeout_ms: int = 10000) -> dict:
        async def do():
            await self._page.fill(selector, value, timeout=timeout_ms)
            return {"ok": True, "selector": selector}
        try:
            return self._call(do)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "selector": selector, "error": str(exc)}

    def console_messages(self) -> list:
        return list(self.state.console)

    def blocked_requests(self) -> list:
        return [{"url": b.url, "reason": b.reason} for b in self.state.blocked]

    def allowed_requests(self) -> list:
        return list(self.state.requests)


# --------------------------------------------------------------------------- #
#  Registry: the live view and an agent tool call reach the SAME session.      #
# --------------------------------------------------------------------------- #

_SESSIONS: dict = {}
_LOCK = threading.Lock()


def register(session: BrowserSession) -> None:
    with _LOCK:
        _SESSIONS[session.session_id] = session


def get(session_id: str) -> Optional[BrowserSession]:
    with _LOCK:
        return _SESSIONS.get(session_id)


def close(session_id: str) -> bool:
    """Close and forget one session. True when there was one to close."""
    with _LOCK:
        session = _SESSIONS.pop(session_id, None)
    if session is None:
        return False
    session.stop()
    return True


def close_all() -> None:
    with _LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        session.stop()


def active_ids() -> list:
    with _LOCK:
        return sorted(_SESSIONS)
