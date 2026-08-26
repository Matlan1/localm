# SPDX-License-Identifier: AGPL-3.0-or-later
"""Routes + helpers for the GUI's host-side dependency install.

Key security property under test: a NON-local (remote) client is refused (403)
even with the PLUGINS_ADMIN scope - a remote client must never trigger a
server-side pip. The happy path (local operator) starts a background task and
streams its progress over SSE.
"""

import time

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
    # Default to a loopback bind (the `localm gui` default), so every client is
    # truly on this host and the pip path is allowed. Network-bind tests override.
    app.state.bind_host = "127.0.0.1"
    return app, mgr


def _set_bind(app, host):
    app.state.bind_host = host


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
    _set_bind(app, "0.0.0.0")                    # a network bind
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/install-deps")
    assert r.status_code == 403
    assert "host only" in r.json()["detail"].lower()


def test_events_remote_is_forbidden(app_mgr, monkeypatch):
    app, _ = app_mgr
    _set_bind(app, "0.0.0.0")
    with TestClient(app) as c:
        r = c.get("/api/plugins/voice/install-deps/events")
    assert r.status_code == 403


def test_network_bind_denies_even_from_loopback_peer(app_mgr, monkeypatch):
    """The GUI runs behind portmux, so request.client.host is always 127.0.0.1
    (a loopback peer) even for a genuinely remote client. The gate MUST key off
    the bind host, not the peer - a network bind is refused regardless of the
    (loopback-looking) peer the TestClient presents."""
    app, mgr = app_mgr
    _set_bind(app, "0.0.0.0")                    # network bind; TestClient peer is loopback
    _fake_install(monkeypatch)                  # would run if the gate were wrong
    with TestClient(app) as c:
        r = c.post("/api/plugins/voice/install-deps")
    assert r.status_code == 403
    # And no background install task was ever created.
    assert mgr.get_dep_task("voice") is None


def test_install_deps_unknown_plugin_404(app_mgr, monkeypatch):
    app, _ = app_mgr
    with TestClient(app) as c:
        r = c.post("/api/plugins/nope/install-deps")
    assert r.status_code == 404


def test_events_without_task_404(app_mgr, monkeypatch):
    app, _ = app_mgr
    with TestClient(app) as c:
        r = c.get("/api/plugins/voice/install-deps/events")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
#  Happy path: start + stream                                                 #
# --------------------------------------------------------------------------- #

def test_install_deps_starts_and_streams(app_mgr, monkeypatch):
    app, mgr = app_mgr
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
    _fake_install(monkeypatch, ok=False, lines=("resolving...", "ERROR: boom"))
    with TestClient(app) as c:
        c.post("/api/plugins/voice/install-deps")
        body = c.get("/api/plugins/voice/install-deps/events").text
    assert '"ok": false' in body and "boom" in body


# --------------------------------------------------------------------------- #
#  Unit: bind-host gate + DepInstallTask + worker                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("host,loop", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("127.0.0.5", True), ("10.0.0.7", False), ("192.168.1.4", False),
    ("0.0.0.0", False), ("testclient", False), ("", False), (None, False),
])
def test_is_loopback_host(host, loop):
    assert deps_task.is_loopback_host(host) is loop


class _App:
    def __init__(self, bind_host="__unset__"):
        self.state = type("S", (), {})()
        if bind_host != "__unset__":
            self.state.bind_host = bind_host


@pytest.mark.parametrize("bind,allowed", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("0.0.0.0", False), ("192.168.1.10", False),  # network binds -> deny
    (None, False), ("__unset__", False),          # unknown -> fail closed
])
def test_host_pip_allowed(bind, allowed):
    # Fail closed on a network or unknown bind: only a loopback bind may allow
    # the host-only pip path, since the portmux relay makes the peer useless.
    assert deps_task.host_pip_allowed(_App(bind)) is allowed


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

    # WAIT for the worker before returning: monkeypatch undoes the `blocking`
    # stub at teardown, and run_dep_install resolves deps.install_plugin_extras
    # at CALL time, so a thread still in flight would reach the real installer.
    # start_dep_install keeps the TASK, not the thread, so poll the task's status.
    deadline = time.time() + 10
    while t1.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert t1.status != "running", (
        "the dep-install worker outlived its stub; with the real "
        "install_plugin_extras restored it would run a REAL pip install against "
        "the interpreter running this suite")
