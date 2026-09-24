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

A session keeps exactly one page open. A new window that a click opened becomes
the page every call drives and the live view shows, and the page before it is
closed; any other new window is closed at once and recorded as refused.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
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

#: Characters sent to the browser per keyboard call when typing.
_TYPE_CHUNK = 16

#: Seconds after a click or key press this session sent during which a window
#: the page opens counts as opened by it; a window opened while one is still
#: being sent counts too. Matches Chromium's transient user activation lifespan.
_INPUT_ACTIVATION = 5.0

#: Seconds a window opened by a click may take to load and still be shown.
_CLICKED_WINDOW_TTL = 30.0

#: Windows opened by a click that are remembered while they load.
_CLICKED_WINDOW_CAP = 32

#: Milliseconds allowed for attaching a DevTools session to a page, for the
#: window watch and for the screencast.
_PAGE_SETUP_MS = 10000

#: Milliseconds a blank window a click opened is given to start loading a URL
#: before it is shown as it is.
_BLANK_WINDOW_WAIT_MS = 5000


class BrowserUnavailableError(RuntimeError):
    """Playwright, or the browser build it pins, is not installed."""


@dataclass
class Blocked:
    url: str
    reason: str
    #: "request" for a request or WebSocket the network gate refused, "window"
    #: for a new window this session closed.
    kind: str = "request"


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
            'pip install "localm[browser]" then download the browser it '
            "drives with:  localm setup-browser") from exc
    return async_playwright


async def _within(coro, timeout_ms: float, what: str) -> Any:
    """Await *coro*, cancelling it once *timeout_ms* has passed.

    Raises TimeoutError naming *what* and the timeout when it is cancelled."""
    seconds = timeout_ms / 1000.0
    try:
        return await asyncio.wait_for(coro, seconds)
    except TimeoutError:
        raise TimeoutError("%s did not finish within %gs" % (what, seconds)) from None


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
        self._tearing_down = False
        #: Held on the browser loop while the driven page or the screencast
        #: changes.
        self._page_lock = asyncio.Lock()
        #: (url, expiry) for each window a click opened that has not yet loaded.
        self._clicked_windows: list = []
        #: time.monotonic() of the last click or key press sent to the page.
        self._last_input = float("-inf")
        #: Clicks and key presses currently being sent to the page.
        self._inputs_in_flight = 0

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
                "localm setup-browser. " + str(exc)) from exc
        self._ctx = await self._browser.new_context()
        # Routed on the CONTEXT rather than the page, so a popup or a second
        # page the site opens is gated too.
        await self._ctx.route("**/*", self._on_route)
        # WebSockets are NOT covered by route(): they need their own handler, and
        # a routed WebSocket does not reach the server unless connect_to_server()
        # is called, so an unhandled one fails closed.
        await self._ctx.route_web_socket("**/*", self._on_ws_route)
        page = await self._ctx.new_page()
        async with self._page_lock:
            await self._drive(page)
        self._ctx.on("page", self._on_new_page)
        if self._on_frame is not None:
            await self._ensure_screencast()

    async def _start_screencast(self) -> None:
        """Stream the page as JPEG frames to the on_frame callback.

        Best-effort: a browser build without the screencast command still drives
        normally, it just has no live view, and the reason is logged rather than
        raised into the session's startup. Gives up after _PAGE_SETUP_MS.
        ``_cdp`` is set only once the screencast has started; an attach that
        fails or is cancelled detaches its DevTools session and leaves ``_cdp``
        None."""
        attached = []

        async def attach():
            cdp = await self._ctx.new_cdp_session(self._page)
            attached.append(cdp)
            await cdp.send("Page.enable")
            cdp.on("Page.screencastFrame",
                   lambda params: self._on_screencast_frame(params, cdp))
            await cdp.send("Page.startScreencast", {
                "format": "jpeg", "quality": 55,
                "maxWidth": 1280, "maxHeight": 800,
            })
        try:
            await _within(attach(), _PAGE_SETUP_MS, "starting the screencast")
        except BaseException as exc:
            for cdp in attached:
                asyncio.ensure_future(self._detach_quietly(cdp))
            if not isinstance(exc, Exception):
                raise
            logger.warning("browser %s has no live view: %s", self.session_id, exc)
            return
        self._cdp = attached[0]

    def _on_screencast_frame(self, params: dict, cdp) -> None:
        """Hand one frame on, then acknowledge it on *cdp*, the DevTools session
        that sent it.

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
        if sid is not None:
            asyncio.ensure_future(self._ack_frame(cdp, sid))

    @staticmethod
    async def _ack_frame(cdp, session_id) -> None:
        try:
            await cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception:
            pass

    @staticmethod
    async def _detach_quietly(cdp) -> None:
        try:
            await cdp.detach()
        except Exception:
            pass

    async def _ensure_screencast(self) -> bool:
        """Start the screencast on the driven page unless it is already running.
        True when it is running."""
        async with self._page_lock:
            if self._cdp is None:
                await self._start_screencast()
            return self._cdp is not None

    # -- the one page this session keeps ------------------------------------ #

    async def _drive(self, page) -> None:
        """Make *page* the page every call drives and the live view shows.

        Caller must hold ``_page_lock``. A running screencast moves to *page*."""
        self._page = page
        page.on("console", self._on_console)
        page.on("close", self._on_page_closed)
        await self._watch_windows(page)
        if self._cdp is not None:
            previous, self._cdp = self._cdp, None
            await self._detach_quietly(previous)
            await self._start_screencast()

    async def _watch_windows(self, page) -> None:
        """Pass each Page.windowOpen event Chromium reports for *page* to
        _on_window_open. When this cannot be set up within _PAGE_SETUP_MS,
        every window *page* opens is treated as not clicked."""
        async def attach():
            cdp = await self._ctx.new_cdp_session(page)
            cdp.on("Page.windowOpen", self._on_window_open)
            await cdp.send("Page.enable")
        try:
            await _within(attach(), _PAGE_SETUP_MS, "watching for new windows")
        except Exception as exc:                     # noqa: BLE001
            logger.warning("browser %s cannot tell a clicked window from a "
                           "scripted one, so every new window will be closed: %s",
                           self.session_id, exc)

    def _mark_input(self) -> None:
        """Note that a click or key press is being sent to the page."""
        self._last_input = time.monotonic()

    @contextlib.contextmanager
    def _input(self):
        """Count a click or key press as in flight for as long as the block
        runs, and mark it sent both when the block starts and when it ends."""
        self._inputs_in_flight += 1
        self._mark_input()
        try:
            yield
        finally:
            self._inputs_in_flight -= 1
            self._mark_input()

    def _on_window_open(self, params: dict) -> None:
        """Remember a window the page opened as clicked, when Chromium reports a
        user gesture and this session is sending a click or key press or sent
        one within _INPUT_ACTIVATION seconds. See
        test_a_gesture_flag_without_recent_input_is_not_a_click."""
        now = time.monotonic()
        if not params.get("userGesture"):
            return
        if not self._inputs_in_flight and now - self._last_input > _INPUT_ACTIVATION:
            return
        live = [(u, t) for (u, t) in self._clicked_windows if t > now]
        live.append((str(params.get("url") or "about:blank"),
                     now + _CLICKED_WINDOW_TTL))
        self._clicked_windows = live[-_CLICKED_WINDOW_CAP:]

    def _take_clicked_window(self, url: str) -> bool:
        """Consume the record of a click opening *url*. False when there is none."""
        now = time.monotonic()
        for i, (opened, expiry) in enumerate(self._clicked_windows):
            if opened == url and expiry > now:
                del self._clicked_windows[i]
                return True
        return False

    async def _on_new_page(self, page) -> None:
        """Show a loaded window that a click opened, in place of the page before
        it. Close any other new window and record it as refused.

        A clicked window that is still blank is first given
        _BLANK_WINDOW_WAIT_MS to start loading a URL, while the page that opened
        it stays open."""
        try:
            async with self._page_lock:
                if (page is self._page or page.is_closed()
                        or self._closed or self._tearing_down):
                    return
                url = page.url
                if url.startswith("chrome-error:"):
                    self._refuse(url, "a new window the page opened did not "
                                      "load, so it was closed", kind="window")
                    await self._close_quietly(page)
                elif self._take_clicked_window(url):
                    if url == "about:blank":
                        await self._await_navigation(page)
                        if page.is_closed():
                            return
                    previous = self._page
                    previous_url = previous.url if previous is not None else ""
                    await self._drive(page)
                    if previous is not None:
                        await self._close_quietly(previous)
                    self._note("showing the new window %s in place of %s, "
                               "which was closed" % (page.url, previous_url))
                else:
                    self._refuse(url, "the page opened a new window without a "
                                      "click, so it was closed", kind="window")
                    await self._close_quietly(page)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("browser %s could not handle a new window: %s",
                           self.session_id, exc)

    def _on_page_closed(self, page) -> None:
        if page is self._page and not (self._closed or self._tearing_down):
            asyncio.ensure_future(self._replace_closed_page(page))

    async def _replace_closed_page(self, closed) -> None:
        """Open a blank page in place of a driven page that closed itself."""
        try:
            async with self._page_lock:
                if (closed is not self._page
                        or self._closed or self._tearing_down):
                    return
                page = await self._ctx.new_page()
                await self._drive(page)
                self._note("the page %s closed itself; a blank page is open in "
                           "its place" % closed.url)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("browser %s could not replace a page that closed "
                           "itself: %s", self.session_id, exc)

    async def _await_navigation(self, page) -> None:
        """Wait up to _BLANK_WINDOW_WAIT_MS for a blank *page* to commit a
        navigation to another URL. Returns either way."""
        try:
            await page.wait_for_url(lambda u: u != "about:blank",
                                    wait_until="commit",
                                    timeout=_BLANK_WINDOW_WAIT_MS)
        except Exception as exc:                     # noqa: BLE001
            logger.debug("browser %s: a new window stayed blank: %s",
                         self.session_id, exc)

    @staticmethod
    async def _close_quietly(page) -> None:
        try:
            await page.close()
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
        self._tearing_down = True
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
              timeout: float = DEFAULT_CALL_TIMEOUT, *,
              what: str = "the browser call") -> Any:
        """Run *make_coro()* on this session's own loop and return its result.

        When it has not finished within *timeout* seconds it is cancelled on the
        loop and TimeoutError is raised, naming *what* and the timeout."""
        if self._closed or self._loop is None:
            raise BrowserUnavailableError("this browser session is closed")
        fut = asyncio.run_coroutine_threadsafe(make_coro(), self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            if not fut.cancel():
                return fut.result()
            raise TimeoutError("%s did not finish within %gs"
                               % (what, timeout)) from None

    def enable_live_view(self, on_frame, *, timeout_ms: int = 15000) -> bool:
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
        what = "starting the live view"
        try:
            return bool(self._call(
                lambda: _within(self._ensure_screencast(), timeout_ms, what),
                what=what))
        except Exception as exc:             # noqa: BLE001
            logger.warning("browser %s could not start a live view: %s",
                           self.session_id, exc)
            return False

    def disable_live_view(self) -> None:
        """Stop handing frames to a viewer. The session keeps running."""
        self._on_frame = None

    # -- request gating ----------------------------------------------------- #

    def _refuse(self, url: str, reason: str, *, kind: str = "request") -> None:
        if len(self.state.blocked) < _LOG_CAP:
            self.state.blocked.append(Blocked(url=url, reason=reason, kind=kind))
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

    def _note(self, text: str) -> None:
        """Record something the session did on its own, as a console line of
        type ``localm``."""
        if len(self.state.console) < _LOG_CAP:
            self.state.console.append({"type": "localm", "text": text})
        logger.info("browser %s: %s", self.session_id, text)

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
            res = self._call(go, what="loading the page")
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
        # A window closed during the call is not a network refusal.
        if not res.get("ok"):
            refused = [b for b in self.state.blocked[mark:] if b.kind == "request"]
            if refused:
                res["refused"] = refused[0].reason
                res["refused_url"] = refused[0].url
        res.setdefault("refused", None)
        return res

    def read_text(self, *, max_chars: int = 20000, timeout_ms: int = 30000) -> str:
        async def read():
            return await self._page.inner_text("body", timeout=timeout_ms)
        try:
            return self._call(read, what="reading the page")[:max_chars]
        except Exception as exc:                     # noqa: BLE001
            return "(could not read the page: " + str(exc) + ")"

    def screenshot(self, *, full_page: bool = False,
                   timeout_ms: int = 30000) -> bytes:
        async def shot():
            return await self._page.screenshot(full_page=full_page, type="png",
                                               timeout=timeout_ms)
        return self._call(shot, what="the screenshot")

    def click(self, selector: str, *, timeout_ms: int = 10000) -> dict:
        async def do():
            with self._input():
                await self._page.click(selector, timeout=timeout_ms)
            return {"ok": True, "selector": selector, "url": self._page.url}
        try:
            return self._call(do, what="the click")
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "selector": selector, "error": str(exc)}

    def fill(self, selector: str, value: str, *, timeout_ms: int = 10000) -> dict:
        async def do():
            await self._page.fill(selector, value, timeout=timeout_ms)
            return {"ok": True, "selector": selector}
        try:
            return self._call(do, what="filling the field")
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "selector": selector, "error": str(exc)}

    def click_coords(self, x: float, y: float, button: str = "left", *,
                     timeout_ms: int = 10000) -> dict:
        async def do():
            with self._input():
                await self._page.mouse.click(x, y, button=button)
            return {"ok": True, "x": x, "y": y, "url": self._page.url}
        what = "the click"
        try:
            return self._call(lambda: _within(do(), timeout_ms, what), what=what)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "x": x, "y": y, "error": str(exc)}

    def scroll(self, delta_x: float, delta_y: float, *,
               timeout_ms: int = 10000) -> dict:
        async def do():
            await self._page.mouse.wheel(delta_x, delta_y)
            return {"ok": True}
        what = "the scroll"
        try:
            return self._call(lambda: _within(do(), timeout_ms, what), what=what)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def type_text(self, text: str, *, timeout_ms: int = 30000) -> dict:
        """Type *text* into the focused element, _TYPE_CHUNK characters per
        keyboard call. When *timeout_ms* runs out no further characters are
        sent; the ones already typed stay."""
        async def do():
            for start in range(0, len(text), _TYPE_CHUNK):
                with self._input():
                    await self._page.keyboard.type(text[start:start + _TYPE_CHUNK])
            return {"ok": True}
        what = "typing"
        try:
            return self._call(lambda: _within(do(), timeout_ms, what), what=what)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def press_key(self, key: str, *, timeout_ms: int = 10000) -> dict:
        async def do():
            with self._input():
                await self._page.keyboard.press(key)
            return {"ok": True}
        what = "the key press"
        try:
            return self._call(lambda: _within(do(), timeout_ms, what), what=what)
        except Exception as exc:                     # noqa: BLE001
            return {"ok": False, "error": str(exc)}

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


class Claim:
    """Holds a session id in the registry while the browser for it starts."""

    __slots__ = ("session_id",)

    def __init__(self, session_id: str):
        self.session_id = session_id


def register(session: BrowserSession) -> None:
    with _LOCK:
        _SESSIONS[session.session_id] = session


def reserve(session_id: str) -> Optional[Claim]:
    """Claim *session_id* for a browser about to start.

    Returns the claim, or None when the id is already registered or claimed."""
    with _LOCK:
        if session_id in _SESSIONS:
            return None
        claim = Claim(session_id)
        _SESSIONS[session_id] = claim
        return claim


def install(claim: Claim, session: BrowserSession) -> bool:
    """Register *session* in place of *claim*.

    False, registering nothing, when the claim is no longer held because it was
    closed or released."""
    with _LOCK:
        if _SESSIONS.get(claim.session_id) is not claim:
            return False
        _SESSIONS[claim.session_id] = session
        return True


def release_if(session_id: str, entry) -> bool:
    """Forget *entry*, a session or a claim, if it is still the one registered
    under *session_id*. Stops nothing. True when it was removed."""
    with _LOCK:
        if _SESSIONS.get(session_id) is not entry:
            return False
        del _SESSIONS[session_id]
        return True


def get(session_id: str) -> Optional[BrowserSession]:
    """The running session registered under *session_id*. None when there is
    none, including while one is still starting."""
    with _LOCK:
        entry = _SESSIONS.get(session_id)
    return None if isinstance(entry, Claim) else entry


def close(session_id: str) -> bool:
    """Close and forget one session, or release the claim of one still
    starting. True when there was either."""
    with _LOCK:
        entry = _SESSIONS.pop(session_id, None)
    if entry is None:
        return False
    if not isinstance(entry, Claim):
        entry.stop()
    return True


def close_all() -> None:
    """Close every running session and release every claim."""
    with _LOCK:
        entries = list(_SESSIONS.values())
        _SESSIONS.clear()
    for entry in entries:
        if not isinstance(entry, Claim):
            entry.stop()


def active_ids() -> list:
    """Ids with a running session or one still starting."""
    with _LOCK:
        return sorted(_SESSIONS)
