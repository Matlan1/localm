# SPDX-License-Identifier: AGPL-3.0-or-later
"""Routes + helpers for the GUI's host-side dependency install.

Key security property under test: a NON-local (remote) client is refused (403)
even with the PLUGINS_ADMIN scope - a remote client must never trigger a
server-side pip. The happy path (local operator) starts a background task and
streams its progress over SSE.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins import deps, deps_task


@pytest.fixture
def app_mgr(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    mgr = attach_engine(app)
    return app, mgr


def _be_local(monkeypatch, local=True):
    monkeypatch.setattr(deps_task, "is_local_request", lambda req: local)


def _fake_install(monkeypatch, *, ok=True, lines=("resolving...", "installed x")):
    def fake(extras, on_progress=None):
        for ln in lines:
            if on_progress:
                on_progress(ln)
        return deps.InstallResult(ok=ok, installed=["x>=1"] if ok else [],
                                  failed=[] if ok else ["x>=1"],
                                  error="" if ok else "boom")
    monkeypatch.setattr(deps, "install_plugin_extras", fake)


# --------------------------------------------------------------------------- #
#  Host-local security gate                                                   #
# --------------------------------------------------------------------------- #

def test_install_deps_remote_is_forbidden(app_mgr, monkeypatch):
    app, _ = app_mgr
    _be_local(monkeypatch, local=False)         # simulate a remote client
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/install-deps")
    assert r.status_code == 403
    assert "host only" in r.json()["detail"].lower()


def test_events_remote_is_forbidden(app_mgr, monkeypatch):
    app, _ = app_mgr
    _be_local(monkeypatch, local=False)
    with TestClient(app) as c:
        r = c.get("/api/plugins/voice/install-deps/events")
    assert r.status_code == 403


def test_install_deps_unknown_plugin_404(app_mgr, monkeypatch):
    app, _ = app_mgr
    _be_local(monkeypatch, local=True)
    with TestClient(app) as c:
        r = c.post("/api/plugins/nope/install-deps")
    assert r.status_code == 404


def test_events_without_task_404(app_mgr, monkeypatch):
    app, _ = app_mgr
    _be_local(monkeypatch, local=True)
    with TestClient(app) as c:
        r = c.get("/api/plugins/voice/install-deps/events")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
#  Happy path: start + stream                                                 #
# --------------------------------------------------------------------------- #

def test_install_deps_starts_and_streams(app_mgr, monkeypatch):
    app, mgr = app_mgr
    _be_local(monkeypatch, local=True)
    _fake_install(monkeypatch, ok=True, lines=("resolving...", "installed x"))
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/install-deps")
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "voice"
        # The background task finishes quickly; the SSE replays the full log.
        ev = c.get("/api/plugins/voice/install-deps/events")
        assert ev.status_code == 200
        body = ev.text
    assert "resolving..." in body and "installed x" in body
    assert '"type": "end"' in body and '"ok": true' in body


def test_install_deps_failure_streams_error(app_mgr, monkeypatch):
    app, mgr = app_mgr
    _be_local(monkeypatch, local=True)
    _fake_install(monkeypatch, ok=False, lines=("resolving...", "ERROR: boom"))
    with TestClient(app) as c:
        c.post("/api/plugins/voice/install-deps")
        body = c.get("/api/plugins/voice/install-deps/events").text
    assert '"ok": false' in body and "boom" in body


# --------------------------------------------------------------------------- #
#  Unit: is_local_request + DepInstallTask + worker                           #
# --------------------------------------------------------------------------- #

class _Req:
    def __init__(self, host):
        self.client = type("C", (), {"host": host})() if host is not None else None


@pytest.mark.parametrize("host,local", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("127.0.0.5", True), ("10.0.0.7", False), ("192.168.1.4", False),
    ("testclient", False), ("", False), (None, False),
])
def test_is_local_request(host, local):
    assert deps_task.is_local_request(_Req(host)) is local


def test_dep_task_lifecycle():
    t = deps_task.DepInstallTask("voice")
    assert t.status == "running"
    t.emit("a")
    t.emit("b")
    assert t.snapshot() == ["a", "b"]
    t.finish(deps.InstallResult(ok=True, installed=["x>=1"]))
    assert t.status == "done"
    ev = t.end_event()
    assert ev["type"] == "end" and ev["ok"] is True and ev["installed"] == ["x>=1"]


def test_run_dep_install_captures_exception(monkeypatch):
    class _Mgr:
        def install_plugin_deps(self, name, on_progress=None):
            on_progress("starting")
            raise RuntimeError("kaboom")

    t = deps_task.DepInstallTask("voice")
    deps_task.run_dep_install(_Mgr(), "voice", t)
    assert t.status == "error"
    assert t.result.ok is False
    assert any("kaboom" in ln for ln in t.snapshot())


def test_start_dep_install_idempotent_while_running(app_mgr, monkeypatch):
    _, mgr = app_mgr
    # A task that blocks until released, so the second call sees it "running".
    import threading
    gate = threading.Event()

    def blocking(extras, on_progress=None):
        gate.wait(timeout=2)
        return deps.InstallResult(ok=True)

    monkeypatch.setattr(deps, "install_plugin_extras", blocking)
    t1 = mgr.start_dep_install("voice")
    t2 = mgr.start_dep_install("voice")
    assert t1 is t2                         # no second pip launched
    gate.set()
