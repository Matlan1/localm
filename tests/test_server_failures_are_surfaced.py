# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-side best-effort steps that fail say so: a log line at WARNING, and
where the caller reads a result, a field that tells "could not check" apart from
"checked and fine"."""

import logging
import time

import pytest
from fastapi.testclient import TestClient

from localm import auth


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


def _owner_client():
    from localm.inference.http_server import create_app
    key = auth.regenerate_key()
    app = create_app(None)
    app.state.bind_host = "127.0.0.1"
    return TestClient(app), {"Authorization": f"Bearer {key}"}


def test_key_create_reports_an_unchecked_scope_readiness(monkeypatch, caplog):
    from localm.plugins.engine import PluginManager

    def _boom(self, scopes):
        raise RuntimeError("catalog unreadable")
    monkeypatch.setattr(PluginManager, "scope_deps_warnings", _boom)
    c, owner = _owner_client()

    with caplog.at_level(logging.WARNING, logger="localm"):
        r = c.post("/v1/keys", json={"name": "dash", "scopes": ["models:read"]},
                   headers=owner)

    assert r.status_code == 200, r.text
    assert any("Could not check" in w for w in r.json().get("warnings", []))
    assert any("plugin readiness" in rec.getMessage() for rec in caplog.records)


def test_debug_stacks_reports_a_failed_executor_snapshot(monkeypatch, caplog):
    import localm.inference._executor_health as eh

    def _boom(*a, **kw):
        raise AttributeError("_work_queue")
    monkeypatch.setattr(eh, "executors_snapshot", _boom)
    c, owner = _owner_client()

    with caplog.at_level(logging.WARNING, logger="localm"):
        r = c.get("/debug/stacks", headers=owner)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["executors"] == {}
    assert body["executors_error"] == "AttributeError: '_work_queue'"
    assert any("executor snapshot failed" in rec.getMessage() for rec in caplog.records)


def test_debug_stacks_executors_error_is_null_when_the_snapshot_works():
    c, owner = _owner_client()
    body = c.get("/debug/stacks", headers=owner).json()
    assert body["executors_error"] is None
    assert isinstance(body["executors"], dict)


def test_load_gpu_index_fallback_is_logged(monkeypatch, caplog):
    from localm.inference import http_server as hs

    def _boom():
        raise OSError("config dir not writable")
    monkeypatch.setattr("localm.config.load_config", _boom)

    with caplog.at_level(logging.WARNING, logger="localm"):
        assert hs._current_gpu_index() == 0

    assert any("assuming device 0" in rec.getMessage() for rec in caplog.records)


def test_hang_trace_write_failure_is_logged_once(tmp_path, monkeypatch, caplog):
    from localm.inference import http_server as hs
    trace = tmp_path / "no-such-dir" / "hang.log"
    monkeypatch.setattr(hs, "_hb_monotonic", 0.0)

    with caplog.at_level(logging.WARNING, logger="localm"):
        stop, thread = hs._start_hang_watchdog(
            threshold=0.01, trace_path=trace, poll=0.01)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not any(
                    "could not write the stall trace" in r.getMessage()
                    for r in caplog.records):
                time.sleep(0.02)
        finally:
            stop.set()
            thread.join(timeout=2)

    hits = [r for r in caplog.records if "could not write the stall trace" in r.getMessage()]
    assert len(hits) == 1
    assert hits[0].levelno == logging.WARNING
    assert not trace.exists()


def test_revoke_warns_when_session_cleanup_raises(monkeypatch, caplog):
    from localm import sessions
    made = auth.create_key("device", scopes=["admin"])

    def _boom(key_hash):
        raise AttributeError("'list' object has no attribute 'get'")
    monkeypatch.setattr(sessions, "revoke_by_key_hash", _boom)

    with caplog.at_level(logging.WARNING, logger="localm"):
        assert auth.revoke_key(made["id"]) is True

    warned = [r for r in caplog.records if r.levelno == logging.WARNING
              and "browser sessions failed" in r.getMessage()]
    assert warned, [r.getMessage() for r in caplog.records]
