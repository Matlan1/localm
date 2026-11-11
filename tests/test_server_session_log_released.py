"""The server's session log file is released when the app that opened it is
replaced by a later create_app() or shut down by its lifespan."""
import os

import pytest
from fastapi.testclient import TestClient

from localm import audit
from localm.inference import http_server as hs


@pytest.fixture
def full_mode_sessions(tmp_path, monkeypatch):
    """Full session mode with the sessions folder in *tmp_path*. Closes the
    recorder the test left published."""
    monkeypatch.setenv("LOCALM_MODE", "full")
    monkeypatch.setattr(audit, "_SESSIONS_DIR", tmp_path)
    yield tmp_path
    if hs._session_recorder is not None:
        hs._session_recorder.close()


def _server_logs(d):
    return sorted(p for p in d.iterdir() if p.name.endswith("_server.jsonl"))


def test_a_second_create_app_releases_the_first_apps_session_log(full_mode_sessions):
    first = hs.create_app(None)
    assert TestClient(first).get("/health").status_code == 503
    [first_log] = _server_logs(full_mode_sessions)

    hs.create_app(None)

    assert len(_server_logs(full_mode_sessions)) == 2
    os.remove(first_log)
    assert not first_log.exists()


def test_a_second_create_app_closes_the_first_recorder(full_mode_sessions):
    hs.create_app(None)
    first = hs._session_recorder
    first_audit = first.current()[0]
    assert isinstance(first_audit, audit.AuditLog)

    hs.create_app(None)

    assert hs._session_recorder is not first
    assert first_audit._fh.closed
    assert isinstance(first.current()[0], audit.NullAuditLog)
    assert isinstance(hs._session_recorder.current()[0], audit.AuditLog)


def test_lifespan_shutdown_releases_the_session_log(full_mode_sessions):
    app = hs.create_app(None)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
    [log] = _server_logs(full_mode_sessions)

    os.remove(log)
    assert not log.exists()
