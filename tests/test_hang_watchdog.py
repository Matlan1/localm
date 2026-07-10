# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hang-capture instrumentation: the event-loop stall watchdog and the
loopback-only GET /debug/stacks endpoint.

The watchdog is the load-bearing tool for the diagnosed intermittent server
hang: it runs OFF the event loop and dumps every thread's stack to a file when
the loop stops ticking, so a freeze that would otherwise be lost is captured.
/debug/stacks is the on-demand complement (usable while the loop is still alive).
"""

import time

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


def test_watchdog_dumps_stacks_on_stall(tmp_path, monkeypatch):
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    # Freeze the heartbeat far in the past so the loop looks permanently stalled;
    # the watchdog thread reads this module global every second.
    monkeypatch.setattr(hs, "_hb_monotonic", 0.0)
    stop, thread, fh = hs._start_hang_watchdog(
        threshold=0.2, trace_path=trace, poll=0.05)
    try:
        time.sleep(0.4)     # several 0.05s polls: the watchdog observes + dumps
    finally:
        stop.set()
        thread.join(timeout=2)
        fh.close()
    text = trace.read_text(encoding="utf-8", errors="replace")
    assert "LOCALM HANG WATCHDOG" in text, text
    assert "stalled" in text
    # faulthandler.dump_traceback (or the pure-Python fallback) writes per-thread
    # stack frames; either way a "File ..." / "Thread" / "--- thread" marker shows.
    assert ("File " in text or "Thread" in text or "--- thread" in text), text


def test_watchdog_quiet_while_loop_ticks(tmp_path, monkeypatch):
    """No stall -> no dump. A fresh heartbeat must not trip the watchdog."""
    from localm.inference import http_server as hs
    trace = tmp_path / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())
    stop, thread, fh = hs._start_hang_watchdog(
        threshold=0.3, trace_path=trace, poll=0.05)
    try:
        # Keep the heartbeat fresh across many watchdog polls (lag stays < 0.3s).
        for _ in range(8):
            monkeypatch.setattr(hs, "_hb_monotonic", time.monotonic())
            time.sleep(0.05)
    finally:
        stop.set()
        thread.join(timeout=2)
        fh.close()
    assert trace.read_text(encoding="utf-8", errors="replace") == ""


@pytest.fixture
def app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return create_app(None, api_landing=True)


def test_debug_stacks_open_mode_loopback(app):
    # Open mode (no key configured) + loopback bind -> require_fs_host passes.
    app.state.bind_host = "127.0.0.1"
    c = TestClient(app)
    r = c.get("/debug/stacks")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(("pid", "loop_lag_s", "threads", "tasks")) <= set(body)
    assert isinstance(body["threads"], dict) and body["threads"]   # at least one thread


def test_debug_stacks_hidden_on_network_bind(app):
    for host in ("0.0.0.0", "192.168.1.50", "10.0.0.7"):
        app.state.bind_host = host
        r = TestClient(app).get("/debug/stacks")
        assert r.status_code == 404, f"{host} -> {r.status_code}"


def test_debug_stacks_requires_fs_host_key(tmp_path, monkeypatch):
    # With a key configured, an unauthenticated caller is refused; the owner key
    # (fs_access=host) is allowed.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    app = create_app(None, api_landing=True)
    app.state.bind_host = "127.0.0.1"
    c = TestClient(app)
    assert c.get("/debug/stacks").status_code in (401, 403)     # no credential
    r = c.get("/debug/stacks", headers={"Authorization": "Bearer ownersecret"})
    assert r.status_code == 200, r.text
    assert "threads" in r.json()
