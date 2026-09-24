# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for interactive controls in the automated browser live view."""

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
