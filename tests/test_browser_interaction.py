# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for interactive controls in the automated browser live view."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _isolate_home(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    _cfg.ensure_dirs()


@pytest.fixture
def app(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    from localm.plugins.builtin.browser import plug
    from localm.plugins.engine import attach_engine
    application = FastAPI()
    attach_engine(application)
    application.include_router(plug._router)
    return application


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The browser routes on the real server app, so a refused request is
    rendered by the same validation handler a live request meets."""
    _isolate_home(tmp_path, monkeypatch)
    from localm.inference.http_server import create_app
    from localm.plugins.builtin.browser import plug
    application = create_app(None)
    application.include_router(plug._router)
    return application


def _post_raw(application, route: str, body: str):
    """POST *body* verbatim as JSON, with the GUI shell token open mode asks for.
    Server errors come back as responses rather than being re-raised."""
    with TestClient(application, raise_server_exceptions=False) as c:
        return c.post(route, content=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {application.state.shell_token}",
        })


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


class _InteractiveFakeSession:
    def __init__(self, sid):
        self.session_id = sid
        self.clicks = []
        self.scrolls = []
        self.keys = []
        self.typed = []

    def click_coords(self, x: float, y: float, button: str = "left") -> dict:
        self.clicks.append((x, y, button))
        return {"ok": True, "x": x, "y": y, "url": "https://example.com/"}

    def scroll(self, delta_x: float, delta_y: float) -> dict:
        self.scrolls.append((delta_x, delta_y))
        return {"ok": True}

    def press_key(self, key: str) -> dict:
        self.keys.append(key)
        return {"ok": True}

    def type_text(self, text: str) -> dict:
        self.typed.append(text)
        return {"ok": True}

    def stop(self):
        pass


class TestInteractionRoutesGate:
    def test_routes_refused_when_browser_disabled(self, app):
        with TestClient(app) as c:
            for route, payload in [
                ("/api/browser/click", {"x": 100, "y": 200}),
                ("/api/browser/scroll", {"delta_x": 0, "delta_y": 100}),
                ("/api/browser/key", {"key": "Enter"}),
                ("/api/browser/type", {"text": "hello"}),
            ]:
                r = c.post(route, json=payload)
                assert r.status_code == 409, f"{route} expected 409, got {r.status_code}"

    def test_routes_return_404_when_no_browser_open(self, app):
        _set(browser_enabled=True)
        with TestClient(app) as c:
            for route, payload in [
                ("/api/browser/click", {"x": 100, "y": 200}),
                ("/api/browser/scroll", {"delta_x": 0, "delta_y": 100}),
                ("/api/browser/key", {"key": "Enter"}),
                ("/api/browser/type", {"text": "hello"}),
            ]:
                r = c.post(route, json=payload)
                assert r.status_code == 404, f"{route} expected 404, got {r.status_code}"


class TestInteractionRoutesDispatch:
    def test_routes_forward_to_active_gui_session(self, app, monkeypatch):
        _set(browser_enabled=True)
        from localm.browser import session as bsession

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            with TestClient(app) as c:
                r1 = c.post("/api/browser/click", json={"x": 150.0, "y": 300.0, "button": "left"})
                assert r1.status_code == 200, r1.text
                assert r1.json()["ok"] is True
                assert fake.clicks == [(150.0, 300.0, "left")]

                r2 = c.post("/api/browser/scroll", json={"delta_x": 10.0, "delta_y": 120.0})
                assert r2.status_code == 200, r2.text
                assert r2.json()["ok"] is True
                assert fake.scrolls == [(10.0, 120.0)]

                r3 = c.post("/api/browser/key", json={"key": "Tab"})
                assert r3.status_code == 200, r3.text
                assert r3.json()["ok"] is True
                assert fake.keys == ["Tab"]

                r4 = c.post("/api/browser/type", json={"text": "localm"})
                assert r4.status_code == 200, r4.text
                assert r4.json()["ok"] is True
                assert fake.typed == ["localm"]
        finally:
            bsession.close("gui-owner")


#: Bodies the click and scroll routes must refuse. 1e400 is valid JSON that
#: parses to Infinity; the quoted words are what pydantic reads as NaN and inf.
_REFUSED = [
    ("/api/browser/click", '{"x": 1e400, "y": 0}'),
    ("/api/browser/click", '{"x": "NaN", "y": 0}'),
    ("/api/browser/click", '{"x": "Infinity", "y": 0}'),
    ("/api/browser/click", '{"x": 0, "y": "-Infinity"}'),
    ("/api/browser/click", '{"x": 10000000, "y": 0}'),
    ("/api/browser/click", '{"x": 10, "y": 20, "button": "back"}'),
    ("/api/browser/scroll", '{"delta_x": 0, "delta_y": 1e400}'),
    ("/api/browser/scroll", '{"delta_x": 0, "delta_y": "NaN"}'),
    ("/api/browser/scroll", '{"delta_x": "-Infinity", "delta_y": 0}'),
]


class TestPointerValuesAreBounded:
    @pytest.mark.parametrize("route,body", _REFUSED)
    def test_refused_before_it_reaches_the_session(self, served, route, body):
        _set(browser_enabled=True)
        from localm.browser import session as bsession

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            r = _post_raw(served, route, body)
            assert fake.clicks == [] and fake.scrolls == [], (
                f"{body} reached the browser session as {fake.clicks or fake.scrolls}, "
                "so a non-finite or out-of-range value goes on to Playwright's "
                "driver pipe")
            assert r.status_code == 422, r.text
        finally:
            bsession.close("gui-owner")

    def test_a_finite_in_range_click_and_scroll_still_reach_the_session(self, served):
        _set(browser_enabled=True)
        from localm.browser import session as bsession

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            r1 = _post_raw(served, "/api/browser/click",
                           '{"x": 640.5, "y": 400, "button": "middle"}')
            r2 = _post_raw(served, "/api/browser/scroll",
                           '{"delta_x": -1000000, "delta_y": 1000000}')
            assert fake.clicks == [(640.5, 400.0, "middle")]
            assert fake.scrolls == [(-1000000.0, 1000000.0)]
            assert r1.status_code == 200, r1.text
            assert r2.status_code == 200, r2.text
        finally:
            bsession.close("gui-owner")


class TestTypedInputIsBounded:
    """The type and key routes refuse an oversized body before any of it
    reaches the browser session."""

    def test_text_longer_than_the_cap_is_refused(self, served):
        _set(browser_enabled=True)
        from localm.browser import session as bsession
        from localm.plugins.builtin.browser import plug

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            body = json.dumps({"text": "a" * (plug._MAX_TYPED_TEXT + 1)})
            r = _post_raw(served, "/api/browser/type", body)
            assert fake.typed == [], (
                "an oversized text reached the browser session, so the caller "
                "decides how long typing runs")
            assert r.status_code == 422, r.text
        finally:
            bsession.close("gui-owner")

    def test_text_at_the_cap_is_still_typed(self, served):
        _set(browser_enabled=True)
        from localm.browser import session as bsession
        from localm.plugins.builtin.browser import plug

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            text = "a" * plug._MAX_TYPED_TEXT
            r = _post_raw(served, "/api/browser/type", json.dumps({"text": text}))
            assert fake.typed == [text]
            assert r.status_code == 200, r.text
        finally:
            bsession.close("gui-owner")

    def test_a_key_name_longer_than_the_cap_is_refused(self, served):
        _set(browser_enabled=True)
        from localm.browser import session as bsession
        from localm.plugins.builtin.browser import plug

        fake = _InteractiveFakeSession("gui-owner")
        bsession.register(fake)
        try:
            body = json.dumps({"key": "K" * (plug._MAX_KEY_NAME + 1)})
            r = _post_raw(served, "/api/browser/key", body)
            assert fake.keys == [], "an oversized key name reached the session"
            assert r.status_code == 422, r.text
        finally:
            bsession.close("gui-owner")

    def test_every_key_the_live_view_sends_fits_the_cap(self):
        from localm.plugins.builtin.browser import plug
        sent = ["Backspace", "Enter", "Tab", "ArrowUp", "ArrowDown", "ArrowLeft",
                "ArrowRight", "PageUp", "PageDown", "Home", "End", "Delete"]
        assert max(len(k) for k in sent) <= plug._MAX_KEY_NAME


@pytest.fixture
def looped():
    """BrowserSessions on a real event loop thread, driving a fake page.

    No Chromium is launched: only the marshalling onto the session's loop, and
    what happens to work that outlives its caller, are exercised."""
    from localm.browser.session import BrowserSession

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    made = []

    def _make(page):
        sess = BrowserSession("t-looped")
        sess._loop = loop
        sess._page = page
        made.append(sess)
        return sess
    yield _make
    for sess in made:
        sess._closed = True
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


class _Hung:
    """Records each input call, then waits far longer than any test runs."""

    def __init__(self):
        self.started = []
        self.cancelled = []

    async def wait(self, name):
        self.started.append(name)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.append(name)
            raise


class _HungPage:
    def __init__(self):
        self.url = "about:blank"
        self.rec = _Hung()
        rec = self.rec

        class _Keyboard:
            async def type(self, text):
                await rec.wait("type")

            async def press(self, key):
                await rec.wait("press")

        class _Mouse:
            async def click(self, x, y, button="left"):
                await rec.wait("click")

            async def wheel(self, dx, dy):
                await rec.wait("wheel")

        self.keyboard = _Keyboard()
        self.mouse = _Mouse()


class _DriverKeyboard:
    """Types one character at a time the way Playwright's driver does: a call
    keeps typing on its own after the caller awaiting it is cancelled."""

    def __init__(self):
        self.landed = []

    async def type(self, text):
        async def run():
            for ch in text:
                await asyncio.sleep(0.005)
                self.landed.append(ch)
        await asyncio.shield(asyncio.ensure_future(run()))


class _TypingPage:
    def __init__(self):
        self.url = "about:blank"
        self.keyboard = _DriverKeyboard()


_HUNG_INPUTS = [
    ("type", lambda s: s.type_text("abc", timeout_ms=300)),
    ("press", lambda s: s.press_key("Enter", timeout_ms=300)),
    ("click", lambda s: s.click_coords(1, 2, timeout_ms=300)),
    ("wheel", lambda s: s.scroll(0, 50, timeout_ms=300)),
]


class TestAbandonedWorkIsCancelled:
    """A call that runs past its deadline is cancelled on the browser loop and
    reported with a message naming what timed out."""

    @pytest.mark.parametrize("name,call", _HUNG_INPUTS,
                             ids=[n for n, _ in _HUNG_INPUTS])
    def test_a_hung_input_is_cancelled_and_named(self, looped, name, call):
        page = _HungPage()
        sess = looped(page)
        started = time.monotonic()
        res = call(sess)
        elapsed = time.monotonic() - started
        time.sleep(0.2)
        assert page.rec.cancelled == [name], (
            "the %s call was left running on the browser loop after its "
            "caller was told it failed" % name)
        assert res["ok"] is False, res
        assert res["error"], "the timeout came back with an empty message"
        assert "0.3s" in res["error"], res
        assert elapsed < 10, "the call was not bounded by its own timeout"

    def test_typing_stops_soon_after_the_call_gives_up(self, looped):
        page = _TypingPage()
        sess = looped(page)
        res = sess.type_text("a" * 2000, timeout_ms=300)
        after = len(page.keyboard.landed)
        time.sleep(1.5)
        later = len(page.keyboard.landed)
        assert later - after <= 32, (
            "%d characters kept landing after type_text reported failure"
            % (later - after))
        assert res["ok"] is False, res
        assert "typing" in res["error"], res

    def test_a_call_past_the_marshalling_timeout_is_cancelled(self, looped):
        sess = looped(_HungPage())
        state = {"cancelled": False}

        async def hang():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
        err = None
        try:
            sess._call(hang, timeout=0.3, what="typing")
        except Exception as exc:                     # noqa: BLE001
            err = exc
        time.sleep(0.2)
        assert state["cancelled"] is True, (
            "the coroutine kept running on the browser loop after _call gave up")
        assert isinstance(err, TimeoutError), repr(err)
        assert "typing" in str(err) and "0.3s" in str(err), str(err)

    def test_a_result_that_lands_on_time_is_returned(self, looped):
        sess = looped(_HungPage())

        async def quick():
            return 42
        assert sess._call(quick, timeout=5, what="a quick call") == 42


class _StallingCDP:
    """A DevTools session whose Page.enable never answers."""

    def __init__(self):
        self.sent = []
        self.detached = False

    def on(self, event, handler):
        pass

    async def send(self, method, params=None):
        self.sent.append((method, params))
        if method == "Page.enable":
            await asyncio.sleep(60)

    async def detach(self):
        self.detached = True


class _RecordingCDP(_StallingCDP):
    async def send(self, method, params=None):
        self.sent.append((method, params))


class _FakeContext:
    def __init__(self):
        self.sessions = []

    async def new_cdp_session(self, page):
        cdp = _StallingCDP()
        self.sessions.append(cdp)
        return cdp


class TestLiveViewAttach:
    def test_a_stalled_attach_is_retried_not_reported_running(
            self, looped, monkeypatch):
        from localm.browser import session as bsession
        monkeypatch.setattr(bsession, "_PAGE_SETUP_MS", 60000)
        sess = looped(_HungPage())
        sess._ctx = _FakeContext()
        first = sess.enable_live_view(lambda data: None, timeout_ms=300)
        second = sess.enable_live_view(lambda data: None, timeout_ms=300)
        time.sleep(0.2)
        assert sess._cdp is None, (
            "a screencast that never started is recorded as running")
        assert len(sess._ctx.sessions) == 2, "the second call did not try again"
        assert all(c.detached for c in sess._ctx.sessions), (
            "a half-attached DevTools session was left behind")
        assert first is False and second is False, (first, second)

    def test_a_frame_is_acknowledged_on_the_session_that_sent_it(self, looped):
        sess = looped(_HungPage())
        cdp = _RecordingCDP()
        frames = []
        sess._on_frame = frames.append

        async def deliver():
            sess._on_screencast_frame({"data": "jpeg", "sessionId": 7}, cdp)
            await asyncio.sleep(0.05)
        sess._call(deliver)
        assert ("Page.screencastFrameAck", {"sessionId": 7}) in cdp.sent, cdp.sent
        assert frames == ["jpeg"]


class _Popup:
    def __init__(self, url):
        self.url = url
        self.closed = False

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True


class _GotoTimeout(Exception):
    pass


class TestNavigationFailureAttribution:
    def test_a_refused_window_is_not_reported_as_the_navigation_refusal(
            self, looped, monkeypatch):
        from localm.browser import session as bsession
        monkeypatch.setattr(bsession.netgate, "decide", lambda *a, **k: None)
        popup = _Popup("https://popup.example/")
        holder = {}

        class _Page:
            url = "about:blank"

            async def goto(self, url, timeout):
                await holder["sess"]._on_new_page(popup)
                raise _GotoTimeout("Timeout 3000ms exceeded")

        sess = looped(_Page())
        holder["sess"] = sess
        res = sess.navigate("https://slow.example/", timeout_ms=3000)
        assert popup.closed is True
        assert any(b["url"] == popup.url for b in sess.blocked_requests()), (
            "the refused window is no longer listed")
        assert res["ok"] is False, res
        assert res["refused"] is None, (
            "a navigation timeout was reported as a refusal: %r" % (res,))
        assert "Timeout" in res["error"], res


class TestSlowSelectorClick:
    def test_a_window_opened_by_a_click_that_waited_long_counts(
            self, looped, monkeypatch):
        from localm.browser import session as bsession
        monkeypatch.setattr(bsession, "_INPUT_ACTIVATION", 0.1)
        url = "https://popup.example/"
        holder = {}

        class _Page:
            url = "about:blank"

            async def click(self, selector, timeout):
                await asyncio.sleep(0.3)
                holder["sess"]._on_window_open({"url": url, "userGesture": True})

        sess = looped(_Page())
        holder["sess"] = sess
        assert sess.click("#late")["ok"] is True
        assert sess._take_clicked_window(url) is True, (
            "a window opened while a selector click was still in progress was "
            "not counted as clicked")


class TestClickedWindows:
    """Which new windows count as opened by a click: Chromium must report a
    user gesture AND this session must have sent a click or key press within
    the activation window. A gesture flag with no recent input from this
    session does not count."""

    URL = "https://popup.example/"

    def _session(self):
        from localm.browser.session import BrowserSession
        return BrowserSession("t-windows")

    def test_a_gesture_flag_without_recent_input_is_not_a_click(self):
        s = self._session()
        s._on_window_open({"url": self.URL, "userGesture": True})
        assert s._take_clicked_window(self.URL) is False

    def test_a_gesture_right_after_input_is_a_click_once(self):
        s = self._session()
        s._mark_input()
        s._on_window_open({"url": self.URL, "userGesture": True})
        assert s._take_clicked_window(self.URL) is True
        assert s._take_clicked_window(self.URL) is False

    def test_no_gesture_after_input_is_not_a_click(self):
        s = self._session()
        s._mark_input()
        s._on_window_open({"url": self.URL, "userGesture": False})
        assert s._take_clicked_window(self.URL) is False

    def test_input_older_than_the_activation_window_does_not_count(self):
        from localm.browser import session as bsession
        s = self._session()
        s._last_input = time.monotonic() - bsession._INPUT_ACTIVATION - 1
        s._on_window_open({"url": self.URL, "userGesture": True})
        assert s._take_clicked_window(self.URL) is False

    def test_a_click_is_matched_by_url(self):
        s = self._session()
        s._mark_input()
        s._on_window_open({"url": self.URL, "userGesture": True})
        assert s._take_clicked_window("https://other.example/") is False
        assert s._take_clicked_window(self.URL) is True

    def test_the_record_is_bounded(self):
        from localm.browser import session as bsession
        s = self._session()
        s._mark_input()
        for i in range(bsession._CLICKED_WINDOW_CAP * 3):
            s._on_window_open({"url": "https://p%d.example/" % i,
                               "userGesture": True})
        assert len(s._clicked_windows) == bsession._CLICKED_WINDOW_CAP


class TestBrowserSessionInteractionMethods:
    def test_session_interaction_handles_closed_session(self):
        from localm.browser.session import BrowserSession
        sess = BrowserSession("test-closed")
        sess._closed = True

        res_click = sess.click_coords(10, 20)
        assert res_click["ok"] is False
        assert "closed" in res_click["error"]

        res_scroll = sess.scroll(0, 50)
        assert res_scroll["ok"] is False
        assert "closed" in res_scroll["error"]

        res_key = sess.press_key("Enter")
        assert res_key["ok"] is False
        assert "closed" in res_key["error"]

        res_type = sess.type_text("test")
        assert res_type["ok"] is False
        assert "closed" in res_type["error"]
