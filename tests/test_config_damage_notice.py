# SPDX-License-Identifier: AGPL-3.0-or-later
"""A damaged config.json locks the server (fail closed) and must say so: the
gate, the login answer and the session state name the file instead of asking
for an API key that does not exist. Session transcripts and audit trails of one
session carry one id."""

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _point_home(tmp_path, monkeypatch):
    for var in ("LOCALM_API_KEY", "LOCALM_REQUIRE_AUTH", "LOCALM_MODE"):
        monkeypatch.delenv(var, raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")


@pytest.fixture
def damaged_home(tmp_path, monkeypatch):
    _point_home(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text('{ "net_mode": ', encoding="utf-8")
    return tmp_path


def _client():
    from localm.inference.http_server import create_app
    engine = MagicMock()
    engine.display_name = "m"
    app = create_app(engine)
    return app, TestClient(app)


def test_locked_config_is_named_in_the_401_and_the_session_state(damaged_home):
    app, client = _client()
    with client:
        r = client.get("/v1/config")
        assert r.status_code == 401
        assert "config.json" in r.text and "damaged" in r.text
        assert "No API key is involved" in r.text
        assert r.headers["X-Localm-Config-Unreadable"] == "1"
        assert client.get("/api/session").json()["config_unreadable"] is True


def test_login_on_a_locked_config_does_not_claim_open_mode(damaged_home):
    app, client = _client()
    with client:
        r = client.post("/api/session", json={"key": "anything"})
        assert r.status_code == 503
        assert "open mode" not in r.text
        assert "config.json" in r.text


def test_shutdown_accepts_the_attach_token_on_a_locked_config_only(damaged_home):
    with patch("localm.inference.http_server._request_shutdown") as stop:
        app, client = _client()
    token = "t" * 32
    app.state.instance_token = token
    with client:
        bad = client.post("/v1/server/shutdown",
                          headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401
        other = client.post("/v1/server/restart",
                            headers={"Authorization": "Bearer " + token})
        assert other.status_code == 401
        stop.assert_not_called()
        ok = client.post("/v1/server/shutdown",
                         headers={"Authorization": "Bearer " + token})
        assert ok.status_code == 200
        stop.assert_called_once()


def test_readable_config_reports_no_lock(tmp_path, monkeypatch):
    _point_home(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _app, client = _client()
    with client:
        assert client.get("/api/session").json()["config_unreadable"] is False


def test_transcript_and_audit_trail_share_one_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "full")
    from localm.audit import SessionMode, SessionRecorder, make_session_recording
    with patch("localm.audit._SESSIONS_DIR", tmp_path):
        recorder = SessionRecorder("server", label="server")
        _audit, transcript, _mode = recorder.current()
        transcript.exchange("hi", "there")
        recorder.close()
        audit2, transcript2 = make_session_recording(SessionMode.FULL, label="chat")
        transcript2.exchange("a", "b")
        audit2.close()

    for label in ("server", "chat"):
        found = {}
        for p in tmp_path.iterdir():
            m = re.match(r"\d{4}-\d\d-\d\d_\d{6}_\d+_([0-9a-f]+)_(\w+)\.(jsonl|md)$", p.name)
            assert m, p.name
            if m.group(2) == label:
                found.setdefault(m.group(3), set()).add(m.group(1))
        assert found["jsonl"] == found["md"] and len(found["md"]) == 1
