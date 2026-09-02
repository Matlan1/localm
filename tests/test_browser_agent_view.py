# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watching the browser the coding agent drives.

The changelog claims "the Browser tab shows the page live as the agent drives
it". It could not: the tab registers its browser as "gui-<principal>" and the
coder registers its own as "coder-<job_owner>", two namespaces that never meet,
so the tab could only ever show a browser it had opened itself.

The negative test is the one that matters. A handed-out scoped key must never be
able to watch the owner's browser, so the lookup delegates its scoping to
SessionManager.list rather than re-implementing it.
"""

from __future__ import annotations

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


def _enable():
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["browser_enabled"] = True
    save_config(cfg)


class _FakeBrowser:
    def __init__(self, sid):
        self.session_id = sid
        self.viewer = None

    def enable_live_view(self, on_frame):
        self.viewer = on_frame
        return True

    def disable_live_view(self):
        self.viewer = None


class _FakeAgent:
    def __init__(self, owner):
        self.job_owner = owner


class _FakeSession:
    def __init__(self, sid, owner, principal):
        self.id = sid
        self.agent = _FakeAgent(owner)
        self.principal = principal

    def info(self):
        return {"id": self.id}


class _FakeManager:
    """Mirrors SessionManager.list's scoping exactly."""

    def __init__(self, sessions):
        self._s = {s.id: s for s in sessions}

    def list(self, *, principal=None, is_owner=True):
        vals = list(self._s.values())
        if not is_owner:
            vals = [s for s in vals if s.principal == principal]
        return [s.info() for s in vals]

    def get(self, sid):
        return self._s.get(sid)


def _wire(app, monkeypatch, *, sessions, live_ids, is_owner, principal):
    from localm.browser import session as bsession
    from localm.plugins.builtin.browser import plug

    app.state.coder_sessions = _FakeManager(sessions)
    browsers = {sid: _FakeBrowser(sid) for sid in live_ids}
    monkeypatch.setattr(bsession, "get", lambda sid: browsers.get(sid))
    monkeypatch.setattr(
        "localm.plugins.builtin.coder.plug._principal_from_request",
        lambda request: (is_owner, principal))
    return plug, browsers


def test_the_owner_can_watch_the_agents_browser(app, monkeypatch):
    _enable()
    sessions = [_FakeSession("s1", "owner-job", None)]
    _wire(app, monkeypatch, sessions=sessions, live_ids=["coder-owner-job"],
          is_owner=True, principal=None)

    with TestClient(app) as c:
        r = c.get("/api/browser/agent")
    assert r.status_code == 200, r.text
    assert r.json()["available"] is True, (
        "the owner could not see the agent browser that is running")
    assert r.json()["session_id"] == "s1"


def test_a_scoped_key_cannot_watch_the_owners_browser(app, monkeypatch):
    """THE ONE THAT MATTERS. The owner's coder session has principal None; a
    handed-out key is some other principal and must see nothing."""
    _enable()
    sessions = [_FakeSession("s1", "owner-job", None)]      # the OWNER's session
    _wire(app, monkeypatch, sessions=sessions, live_ids=["coder-owner-job"],
          is_owner=False, principal="a-handed-out-key")

    with TestClient(app) as c:
        status = c.get("/api/browser/agent")
        watch = c.post("/api/browser/agent")

    assert status.json()["available"] is False, (
        "a scoped key was told the owner's agent browser is watchable")
    assert watch.status_code == 404, (
        f"a scoped key reached the owner's agent browser: {watch.status_code}")


def test_a_scoped_key_can_watch_its_own_session(app, monkeypatch):
    _enable()
    sessions = [_FakeSession("s2", "mine-job", "my-key")]
    _wire(app, monkeypatch, sessions=sessions, live_ids=["coder-mine-job"],
          is_owner=False, principal="my-key")

    with TestClient(app) as c:
        r = c.get("/api/browser/agent")
    assert r.json()["available"] is True
    assert r.json()["session_id"] == "s2"


def test_nothing_to_watch_when_the_agent_has_no_browser(app, monkeypatch):
    _enable()
    sessions = [_FakeSession("s1", "owner-job", None)]
    _wire(app, monkeypatch, sessions=sessions, live_ids=[],   # none running
          is_owner=True, principal=None)

    with TestClient(app) as c:
        assert c.get("/api/browser/agent").json()["available"] is False
        assert c.post("/api/browser/agent").status_code == 404
