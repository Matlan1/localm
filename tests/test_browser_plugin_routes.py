# SPDX-License-Identifier: AGPL-3.0-or-later
"""The browser plugin's own routes, and the switch in front of them.

Holding the browser capability is not enough on its own: ``browser_enabled``
must also be on. That is the ADR's "scope grants eligibility, a separate switch
grants use" shape, and it is enforced here as well as in the coder tools, so a
GUI caller cannot reach a browser the setting says is off.

These need no browser: the switch is checked before anything is launched.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path, monkeypatch):
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
    from localm.plugins.builtin.browser import plug
    from localm.plugins.engine import attach_engine
    application = FastAPI()
    attach_engine(application)
    application.include_router(plug._router)
    return application


def _set(**values):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg.update(values)
    save_config(cfg)


class TestTheSwitchGuardsTheRoutes:
    def test_opening_is_refused_while_the_setting_is_off(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/session", json={})
            assert r.status_code == 409, r.text
            assert "switched off" in r.json()["detail"]

    def test_navigating_is_refused_while_the_setting_is_off(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/navigate", json={"url": "https://example.com/"})
            assert r.status_code == 409, r.text

    def test_with_the_setting_on_navigate_reports_no_open_browser(self, app):
        """Past the switch, and refusing for the right reason: nothing is open.
        Without this the 409 above would equally pass on a route that always
        refuses."""
        with TestClient(app) as c:
            _set(browser_enabled=True)
            r = c.post("/api/browser/navigate", json={"url": "https://example.com/"})
            assert r.status_code == 404, r.text
            assert "No browser is open" in r.json()["detail"]

    def test_state_reports_the_switch_without_opening_anything(self, app):
        with TestClient(app) as c:
            body = c.get("/api/browser/state").json()
            assert body == {"open": False, "enabled": False,
                            "inlineLiveView": False}
            _set(browser_enabled=True)
            assert c.get("/api/browser/state").json()["enabled"] is True

    def test_state_reports_the_inline_live_view_setting(self, app):
        with TestClient(app) as c:
            assert c.get("/api/browser/state").json()["inlineLiveView"] is False
            _set(browser_inline_live_view=True)
            assert c.get("/api/browser/state").json()["inlineLiveView"] is True

    def test_stopping_when_nothing_is_open_is_not_an_error(self, app):
        with TestClient(app) as c:
            r = c.post("/api/browser/stop")
            assert r.status_code == 200
            assert r.json() == {"closed": False}


class _FakeSession:
    """Stands in for a real BrowserSession: no Chromium, no Playwright."""

    def __init__(self, sid, **kw):
        self.session_id = sid

    def start(self):
        pass

    def navigate(self, url):
        return {"ok": True, "url": url}

    def stop(self):
        pass


def _h(key):
    return {"Authorization": f"Bearer {key}"}


class TestLiveViewJobOwnership:
    """The ADR's live-view auth follow-up (issues.txt NEW-CAP-BROWSER item 2):
    the browser plugin adds no ownership logic of its own for streaming a live
    view. It creates its job with owner=principal_id(request) and leaves
    streaming entirely to the kernel's job_owner_ok gate
    (tests/test_key_scope_jobs.py::TestJobOwnerBinding pins that gate
    generically, including that a non-creator gets 404 on
    /api/jobs/{id}/events). This is the ONE test that exercises the browser
    plugin's OWN job-creation path rather than a synthetic injected job, so
    the claim "the existing owner check is sufficient" rests on something
    more direct than the generic test alone.

    Asserts the job's owner attribute directly rather than actually
    streaming /api/jobs/{id}/events as a foreign key: this job's worker
    loops until cancelled and never sends an "end" event on its own, and
    starlette's TestClient does not hand back a response until the ASGI
    cycle progresses past that, so a real stream read here - correct or
    refused - blocks the test runner indefinitely. Confirmed live: with
    owner deliberately set to None, both client.get() and client.stream()
    on this route hung past 20 seconds instead of returning 200. The
    generic 404-on-mismatch behaviour itself is what
    TestJobOwnerBinding already covers with a short-lived synthetic job."""

    def test_the_job_is_owned_by_the_key_that_opened_it(self, app, monkeypatch):
        from localm import auth
        from localm import scopes as S
        from localm.browser import session as bsession
        monkeypatch.setattr(bsession, "BrowserSession", _FakeSession)
        _set(browser_enabled=True)
        a = auth.create_key("A", [S.BROWSER])["key"]
        b = auth.create_key("B", [S.BROWSER])["key"]
        with TestClient(app) as c:
            r = c.post("/api/browser/session", headers=_h(a), json={})
            assert r.status_code == 200, r.text
            job_id = r.json()["job_id"]
            try:
                job = app.state.jobs.get(job_id)
                assert job is not None, "the job vanished right after creation"
                assert job.owner == auth._hash_key(a), (
                    "the browser session's job is not bound to the key that "
                    "opened it, so job_owner_ok's creator-or-admin gate would "
                    "not restrict who may stream it")
                assert job.owner != auth._hash_key(b)
            finally:
                c.post(f"/api/jobs/{job_id}/cancel", headers=_h(a))


class _SlowStartSession:
    """A BrowserSession stand-in whose start() waits until the test releases it.
    Every instance is recorded on the class, in construction order."""

    made: list = []
    release: threading.Event = threading.Event()
    fail_start: bool = False

    def __init__(self, sid, **kw):
        self.session_id = sid
        self.stopped = False
        type(self).made.append(self)

    def start(self):
        type(self).release.wait(10)
        if type(self).fail_start:
            from localm.browser.session import BrowserUnavailableError
            raise BrowserUnavailableError("no browser here")

    def navigate(self, url):
        return {"ok": True, "url": url}

    def stop(self):
        self.stopped = True


class _StandIn:
    def __init__(self, sid):
        self.session_id = sid
        self.stopped = False

    def stop(self):
        self.stopped = True


def _wait_until(predicate, timeout=5.0):
    """Poll *predicate* until it holds or *timeout* passes. Returns its last value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(0.02)


@pytest.fixture
def slow_start(monkeypatch):
    from localm.browser import session as bsession
    _SlowStartSession.made = []
    _SlowStartSession.release = threading.Event()
    _SlowStartSession.fail_start = False
    monkeypatch.setattr(bsession, "BrowserSession", _SlowStartSession)
    yield _SlowStartSession
    _SlowStartSession.release.set()
    bsession.close_all()


def _job_status(app, job_id):
    job = app.state.jobs.get(job_id)
    return None if job is None else job.status


class TestOpeningIsAtomic:
    """One key gets one browser, and a worker's teardown touches only the
    browser that worker started."""

    def test_a_second_open_while_the_first_starts_launches_nothing(
            self, app, slow_start):
        from localm.browser import session as bsession
        _set(browser_enabled=True)
        with TestClient(app) as c:
            r1 = c.post("/api/browser/session", json={})
            _wait_until(lambda: len(slow_start.made) >= 1)
            r2 = c.post("/api/browser/session", json={})
            if r2.status_code == 200:
                _wait_until(lambda: len(slow_start.made) >= 2)
            made = list(slow_start.made)
            slow_start.release.set()
            try:
                assert len(made) == 1, (
                    "two browsers were launched for one key: %d" % len(made))
                assert r2.status_code == 409, r2.text
                assert r1.status_code == 200, r1.text
            finally:
                for r in (r1, r2):
                    if r.status_code == 200:
                        c.post("/api/jobs/%s/cancel" % r.json()["job_id"])
                assert _wait_until(lambda: all(s.stopped for s in made))
                assert bsession.active_ids() == []

    def test_a_workers_teardown_stops_only_its_own_browser(self, app, slow_start):
        from localm.browser import session as bsession
        _set(browser_enabled=True)
        slow_start.release.set()
        with TestClient(app) as c:
            r = c.post("/api/browser/session", json={})
            assert r.status_code == 200, r.text
            sid = r.json()["session_id"]
            first = _wait_until(lambda: bsession.get(sid))
            assert first is slow_start.made[0], "the first browser never registered"
            other = _StandIn(sid)
            bsession.register(other)
            try:
                c.post("/api/jobs/%s/cancel" % r.json()["job_id"])
                _wait_until(lambda: first.stopped)
                assert other.stopped is False, (
                    "the first worker's teardown stopped a browser it did not start")
                assert bsession.get(sid) is other
                assert first.stopped is True, "the first worker never stopped its own browser"
            finally:
                bsession.close(sid)

    def test_stop_while_starting_stops_the_browser_once_it_is_up(
            self, app, slow_start):
        from localm.browser import session as bsession
        _set(browser_enabled=True)
        with TestClient(app) as c:
            r = c.post("/api/browser/session", json={})
            assert r.status_code == 200, r.text
            sid = r.json()["session_id"]
            job_id = r.json()["job_id"]
            _wait_until(lambda: len(slow_start.made) >= 1)
            s = c.post("/api/browser/stop")
            slow_start.release.set()
            _wait_until(lambda: slow_start.made[0].stopped)
            _wait_until(lambda: _job_status(app, job_id) not in (None, "running"))
            assert slow_start.made[0].stopped is True, (
                "a browser stopped while starting kept running once it was up")
            assert bsession.get(sid) is None
            assert sid not in bsession.active_ids()
            assert s.json() == {"closed": True}
            assert _job_status(app, job_id) == "done"

    def test_a_failed_start_frees_the_key(self, app, slow_start):
        from localm.browser import session as bsession
        _set(browser_enabled=True)
        slow_start.fail_start = True
        slow_start.release.set()
        with TestClient(app) as c:
            r1 = c.post("/api/browser/session", json={})
            assert r1.status_code == 200, r1.text
            job_id = r1.json()["job_id"]
            _wait_until(lambda: _job_status(app, job_id) == "failed")
            sid = r1.json()["session_id"]
            assert sid not in bsession.active_ids(), (
                "a browser that failed to start still holds its key")
            slow_start.fail_start = False
            r2 = c.post("/api/browser/session", json={})
            try:
                assert r2.status_code == 200, r2.text
            finally:
                if r2.status_code == 200:
                    c.post("/api/jobs/%s/cancel" % r2.json()["job_id"])

    def test_a_failure_before_the_launch_does_not_hold_the_key(
            self, app, slow_start, monkeypatch):
        from localm.browser import session as bsession
        from localm.plugins.builtin.browser import plug
        _set(browser_enabled=True)
        slow_start.release.set()
        real = plug._settings
        calls = {"n": 0}

        def failing_once():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("settings unreadable")
            return real()
        monkeypatch.setattr(plug, "_settings", failing_once)
        with TestClient(app, raise_server_exceptions=False) as c:
            r1 = c.post("/api/browser/session", json={})
            held = bsession.active_ids()
            r2 = c.post("/api/browser/session", json={})
            try:
                assert held == [], "a failed open still holds the key: %r" % held
                assert r1.status_code == 500, r1.text
                assert r2.status_code == 200, r2.text
            finally:
                if r2.status_code == 200:
                    c.post("/api/jobs/%s/cancel" % r2.json()["job_id"])

    def test_the_stop_route_ends_the_job(self, app, slow_start):
        from localm.browser import session as bsession
        _set(browser_enabled=True)
        slow_start.release.set()
        with TestClient(app) as c:
            r = c.post("/api/browser/session", json={})
            sid = r.json()["session_id"]
            job_id = r.json()["job_id"]
            live = _wait_until(lambda: bsession.get(sid))
            assert live is not None
            assert c.post("/api/browser/stop").json() == {"closed": True}
            _wait_until(lambda: _job_status(app, job_id) not in (None, "running"))
            assert live.stopped is True
            assert _job_status(app, job_id) == "done", (
                "the job outlived the browser it streams")


class TestTheRegistry:
    def test_a_claimed_id_reads_as_not_open(self):
        from localm.browser import session as bsession
        claim = bsession.reserve("gui-claimed")
        try:
            assert claim is not None
            assert bsession.get("gui-claimed") is None
            assert bsession.reserve("gui-claimed") is None, "one id was claimed twice"
        finally:
            assert bsession.release_if("gui-claimed", claim) is True

    def test_install_needs_the_claim_to_still_be_held(self):
        from localm.browser import session as bsession
        claim = bsession.reserve("gui-install")
        assert claim is not None
        assert bsession.close("gui-install") is True
        late = _StandIn("gui-install")
        assert bsession.install(claim, late) is False
        assert bsession.get("gui-install") is None
        assert late.stopped is False

    def test_release_if_removes_only_the_named_entry(self):
        from localm.browser import session as bsession
        a = _StandIn("gui-rel")
        b = _StandIn("gui-rel")
        bsession.register(a)
        bsession.register(b)
        try:
            assert bsession.release_if("gui-rel", a) is False
            assert bsession.get("gui-rel") is b
        finally:
            assert bsession.release_if("gui-rel", b) is True
        assert a.stopped is False and b.stopped is False

    def test_close_all_skips_a_claim(self):
        from localm.browser import session as bsession
        claim = bsession.reserve("gui-closeall")
        running = _StandIn("gui-running")
        bsession.register(running)
        bsession.close_all()
        assert running.stopped is True
        assert bsession.active_ids() == []
        assert bsession.install(claim, _StandIn("gui-closeall")) is False


class TestTheDocstringInventory:
    """A route missing from the module's own route inventory in its docstring
    is a route an auditor reading that inventory would never learn exists."""

    def test_every_route_is_listed_in_the_module_docstring(self):
        from localm.plugins.builtin.browser import plug
        doc = plug.__doc__ or ""
        paths = sorted({route.path for route in plug._router.routes})
        assert paths, "no routes registered; the router import is wrong"
        missing = [p for p in paths if p not in doc]
        assert missing == [], (
            f"routes missing from the module docstring's inventory: {missing}")


class TestTheManifest:
    def test_it_ships_disabled_and_declares_its_extra(self):
        from localm.plugins.engine import parse_spec
        spec = parse_spec(Path("localm/plugins/builtin/browser"), builtin=True)
        assert spec.default_enabled is False, "a browser must not be on by default"
        assert spec.scope == "browser"
        assert spec.requires_extras == ["browser"]

    def test_it_joins_the_coder_nav_category(self):
        from localm.plugins.engine import parse_spec
        browser = parse_spec(Path("localm/plugins/builtin/browser"), builtin=True)
        coder = parse_spec(Path("localm/plugins/builtin/coder"), builtin=True)
        assert browser.surface.group == "coder"
        assert coder.surface.group == "coder", (
            "the agent must join the category too, or the category has one member "
            "and the browser renders as a flat tab beside it")

    def test_the_manifest_parses_without_warnings(self):
        from localm.plugins.engine import parse_spec
        warns: list = []
        parse_spec(Path("localm/plugins/builtin/browser"), builtin=True,
                   warnings=warns)
        assert warns == [], warns
