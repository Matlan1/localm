# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two coder sessions created close together, in the same process, must not
share an audit log file.

AuditLog named its file by timestamp+pid alone (second resolution, pid shared
by every session in one server process). Two CoderSessions constructed within
the same second landed on the identical filename and their JSONL records
interleaved into one file, so GET /api/coder/sessions/{id}/log returned the
same merged content for both sessions regardless of which one, or which
project, actually produced a given line.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.coder.sessions import CoderSession


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _session(cwd: Path, **kw) -> CoderSession:
    return CoderSession(cwd, _StubBackend(), mode="log", auto_verify=False, **kw)


def _coder_app(tmp_path, monkeypatch, *, api_key="ownersecret"):
    home_dir = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home_dir))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home_dir)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home_dir / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home_dir / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home_dir / "registry.json")
    monkeypatch.setattr("localm.audit._SESSIONS_DIR", home_dir / "sessions")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def test_sessions_in_different_projects_get_distinct_audit_logs(tmp_path):
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    proj_a.mkdir()
    proj_b.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj_a)
        s2 = _session(proj_b)
    try:
        assert s1.agent._checkpoint_id != s2.agent._checkpoint_id
        assert s1.audit_log_path() != s2.audit_log_path()
        assert s1.audit_log_path().exists()
        assert s2.audit_log_path().exists()
    finally:
        s1.agent.close()
        s2.agent.close()


def test_sessions_in_the_same_project_get_distinct_audit_logs(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj)
        s2 = _session(proj)
    try:
        assert s1.audit_log_path() != s2.audit_log_path()
    finally:
        s1.agent.close()
        s2.agent.close()


def test_each_sessions_log_holds_only_its_own_records(tmp_path):
    """The reported symptom: two sessions' logs must not merge, and a
    consumer reading either file back can attribute every line."""
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    proj_a.mkdir()
    proj_b.mkdir()
    with patch("localm.audit._SESSIONS_DIR", tmp_path / "sessions"), \
         patch("time.strftime", return_value="2026-01-01_000000"):
        s1 = _session(proj_a)
        s2 = _session(proj_b)
        s1.agent._audit.notice("test", "from session A")
        s2.agent._audit.notice("test", "from session B")
        s1.agent.close()
        s2.agent.close()

    entries_a = [json.loads(line) for line in
                 s1.audit_log_path().read_text(encoding="utf-8").splitlines()]
    entries_b = [json.loads(line) for line in
                 s2.audit_log_path().read_text(encoding="utf-8").splitlines()]

    assert all(e["session"] == s1.agent._checkpoint_id for e in entries_a)
    assert all(e["session"] == s2.agent._checkpoint_id for e in entries_b)
    assert not any(e["data"].get("message") == "from session B" for e in entries_a)
    assert not any(e["data"].get("message") == "from session A" for e in entries_b)


def test_http_api_reproduction_two_sessions_created_close_together(
        tmp_path, monkeypatch):
    """The exact reported reproduction: two coder sessions created in
    different project directories, within the same second, on one running
    server. Each session's GET .../log must return only its own entries, not
    the other's merged in."""
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    proj_a.mkdir()
    proj_b.mkdir()
    app = _coder_app(tmp_path, monkeypatch)
    owner = {"Authorization": "Bearer ownersecret"}

    with patch("time.strftime", return_value="2026-01-01_000000"), \
         TestClient(app) as client:
        r1 = client.post("/api/coder/sessions", headers=owner,
                         json={"cwd": str(proj_a), "mode": "log"})
        r2 = client.post("/api/coder/sessions", headers=owner,
                         json={"cwd": str(proj_b), "mode": "log"})
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        id1, id2 = r1.json()["id"], r2.json()["id"]

        log1 = client.get(f"/api/coder/sessions/{id1}/log", headers=owner)
        log2 = client.get(f"/api/coder/sessions/{id2}/log", headers=owner)
        assert log1.status_code == 200, log1.text
        assert log2.status_code == 200, log2.text

    body1, body2 = log1.json(), log2.json()
    assert body1["path"] != body2["path"]
    assert body1["entries"] and body2["entries"]
    sessions1 = {e["session"] for e in body1["entries"]}
    sessions2 = {e["session"] for e in body2["entries"]}
    assert len(sessions1) == 1, "session 1's log must not mix in another session's records"
    assert len(sessions2) == 1, "session 2's log must not mix in another session's records"
    assert sessions1 != sessions2
