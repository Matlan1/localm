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
from urllib.parse import urljoin

from localm.browser import netgate

logger = logging.getLogger(__name__)

#: How long a marshalled call may take before the caller gives up on it.
DEFAULT_CALL_TIMEOUT = 45.0

#: Console lines, allowed URLs and blocked-request records kept per session.
_LOG_CAP = 500

#: Redirect hops followed for one request before it is refused.
_MAX_REDIRECTS = 10


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
                 extra_deny=(), extra_allow=(), engine: str = "bundled",
                 on_frame: Optional[Callable[[str], None]] = None):
        #: Called with each base64 JPEG frame when a live view is wanted. None
        #: starts no screencast at all.
        self._on_frame = on_frame
        self._cdp = None
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
            # A launch that got as far as starting Chromium and then failed
            # still owns a browser and a driver, and this session is in no
            # registry, so nothing else can ever close them.
            try:
                loop.run_until_complete(self._teardown())
            except BaseException:                    # noqa: BLE001
                pass
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
            # The bundled engine needs a Chromium build the pip extra does NOT
            # bring: playwright downloads it separately, one build per version.
            # A missing build arrives here as a raw playwright error, so name
            # the command that fixes it instead of passing the raw text on.
            raise BrowserUnavailableError(
                "Could not start the bundled browser. Its Chromium build is "
                "downloaded separately from the Python package; get it with:  "
                "python -m playwright install chromium. " + str(exc)) from exc
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
        if self._on_frame is not None:
            await self._start_screencast()

    async def _start_screencast(self) -> None:
        """Stream the page as JPEG frames to the on_frame callback.

        Best-effort: a browser build without the screencast command still drives
        normally, it just has no live view, and the reason is logged rather than
        raised into the session's startup."""
        try:
            self._cdp = await self._ctx.new_cdp_session(self._page)
            await self._cdp.send("Page.enable")
            self._cdp.on("Page.screencastFrame", self._on_screencast_frame)
            await self._cdp.send("Page.startScreencast", {
                "format": "jpeg", "quality": 55,
                "maxWidth": 1280, "maxHeight": 800,
            })
        except Exception as exc:                     # noqa: BLE001
            self._cdp = None
            logger.warning("browser %s has no live view: %s", self.session_id, exc)

    def _on_screencast_frame(self, params: dict) -> None:
        """Hand one frame on, then acknowledge it.

        Chromium stops sending frames until the previous one is acknowledged, so
        a missed ack silently freezes the live view rather than dropping a frame.
        """
        try:
            data = params.get("data")
            if data and self._on_frame is not None:
                self._on_frame(data)
        except Exception as exc:                     # noqa: BLE001
            logger.debug("browser %s frame callback failed: %s",
                         self.session_id, exc)
        sid = params.get("sessionId")
        if self._cdp is not None and sid is not None:
            asyncio.ensure_future(self._ack_frame(sid))

    async def _ack_frame(self, session_id) -> None:
        try:
            await self._cdp.send("Page.screencastFrameAck",
                                 {"sessionId": session_id})
        except Exception:
            pass

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

    def enable_live_view(self, on_frame) -> bool:
        """Start streaming this ALREADY-RUNNING session to *on_frame*.

        start() only starts the screencast when the session was built with an
        on_frame, so a session created without one (the coder builds its browser
        that way) produces no frames at all and cannot be watched. This attaches
        a viewer to it afterwards.

        Returns False when there is nothing to attach to, or when this build has
        no screencast; it never raises into a caller that is only watching."""
        if self._closed or self._loop is None:
            return False
        self._on_frame = on_frame
        if self._cdp is not None:
            return True                      # already streaming
        try:
            self._call(self._start_screencast)
        except Exception as exc:             # noqa: BLE001
            logger.warning("browser %s could not start a live view: %s",
                           self.session_id, exc)
            return False
        return self._cdp is not None

    def disable_live_view(self) -> None:
        """Stop handing frames to a viewer. The session keeps running."""
        self._on_frame = None

    # -- request gating ----------------------------------------------------- #

    def _refuse(self, url: str, reason: str) -> None:
        if len(self.state.blocked) < _LOG_CAP:
            self.state.blocked.append(Blocked(url=url, reason=reason))
        logger.info("browser %s blocked %s: %s", self.session_id, url, reason)

    async def _on_route(self, route, request) -> None:
        """Decide one request. An unexpected failure aborts it.

        A handler that raises leaves the request hanging until the caller's own
        timeout, which reads as a slow site rather than as a broken gate."""
        try:
            await self._route(route, request)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("browser %s gate failed for %s, refusing it: %s",
                           self.session_id, getattr(request, "url", "?"), exc)
            self._refuse(getattr(request, "url", "") or "",
                         "the network gate failed on this request")
            try:
                await route.abort()
            except Exception:
                pass

    async def _route(self, route, request) -> None:
        url = request.url
        reason = await netgate.decide_async(
            url, extra_deny=self.extra_deny, extra_allow=self.extra_allow)
        if reason is not None:
            self._refuse(url, reason)
            await route.abort()
            return
        if len(self.state.requests) < _LOG_CAP:
            self.state.requests.append(url)
        # A WebSocket handshake is not fetchable, and _on_ws_route owns it. Hand
        # it straight on rather than aborting it on a failed fetch.
        if netgate._scheme_of(url) in netgate.WEBSOCKET_SCHEMES:
            await route.continue_()
            return
        # The redirect chain is walked HERE, one hop at a time, and every hop is
        # decided. max_redirects=0 stops fetch() following them internally, and
        # the final response is fulfilled so the browser never follows one
        # itself: a browser-followed redirect is auto-continued without creating
        # a route, so its target would never be decided.
        current = url
        for _ in range(_MAX_REDIRECTS):
            try:
                response = await route.fetch(url=current, max_redirects=0)
            except Exception as exc:                 # noqa: BLE001
                logger.debug("browser %s fetch failed for %s: %s",
                             self.session_id, current, exc)
                await route.abort()
                return
            if not (300 <= response.status < 400):
                await route.fulfill(response=response)
                return
            target = self._redirect_target(current, response)
            if target is None:
                await route.fulfill(response=response)
                return
            hop = await netgate.decide_async(
                target, extra_deny=self.extra_deny, extra_allow=self.extra_allow)
            if hop is not None:
                self._refuse(target, hop)
                await route.abort()
                return
            if len(self.state.requests) < _LOG_CAP:
                self.state.requests.append(target)
            current = target
        self._refuse(current, "too many redirects")
        await route.abort()

    @staticmethod
    def _redirect_target(url: str, response) -> Optional[str]:
        """The absolute URL a 3xx points at, or None when it names none."""
        headers = response.headers or {}
        location = headers.get("location") or headers.get("Location")
        return urljoin(url, location) if location else None

    async def _on_ws_route(self, ws) -> None:
        """Refuse every WebSocket, with the policy's reason where it has one.

        A routed WebSocket only reaches its server if this handler connects it,
        and connecting does not work while requests are being fulfilled for
        redirect gating: the handshake never leaves the browser. Rather than
        leave a socket that looks connected and silently dies, it is closed here
        and the reason is recorded, so the caller is told instead of guessing.
        """
        url = getattr(ws, "url", "") or ""
        reason = await netgate.decide_async(
            url, extra_deny=self.extra_deny, extra_allow=self.extra_allow)
        if reason is None:
            reason = ("WebSocket connections are not available in the automated "
                      "browser")
        self._refuse(url, reason)
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

        mark = len(self.state.blocked)

        async def go():
            resp = await self._page.goto(url, timeout=timeout_ms)
            return {"ok": True, "url": self._page.url,
                    "status": resp.status if resp else None,
                    "title": await self._page.title()}
        try:
            res = self._call(go)
        except Exception as exc:                     # noqa: BLE001
            res = {"ok": False, "url": url, "error": str(exc)}
        # An error page is not a load. Chromium reports the failed navigation as
        # a completed one sitting on its own error URL.
        if res.get("ok") and str(res.get("url", "")).startswith("chrome-error://"):
            res = {"ok": False, "url": url, "error": "the page did not load"}
        # A refusal recorded DURING this call is the real reason a failed
        # navigation failed, and it names the destination, which after a
        # redirect is not the URL that was asked for. A refusal on a
        # SUBRESOURCE does not fail the navigation: the page itself loaded.
        if not res.get("ok"):
            refused = self.state.blocked[mark:]
            if refused:
                res["refused"] = refused[0].reason
                res["refused_url"] = refused[0].url
        res.setdefault("refused", None)
        return res

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
