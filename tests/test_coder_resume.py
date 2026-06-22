# SPDX-License-Identifier: AGPL-3.0-or-later
"""CODER-2: resume a past coder session. A checkpoint saved for a cwd can be
restored into a new session (owner / coder:full only); the GET /api/coder/resumable
probe is owner-gated and reflects whether a checkpoint exists for a directory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.coder.agent import Agent


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _coder_app(tmp_path, monkeypatch, *, api_key):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def _seed_checkpoint(app, sid, messages):
    """Put a saved conversation on a live session and persist it to disk."""
    sess = app.state.coder_sessions.get(sid)
    sess.agent._messages = messages
    sess.agent._turns = len(messages)
    sess.agent._total_tokens = 42
    sess.persist_checkpoint()
    return sess


def test_resume_restores_a_saved_conversation_for_the_owner(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    msgs = [{"role": "user", "content": "build a calculator"},
            {"role": "assistant", "content": "Here is the plan."}]

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        assert a.status_code == 200
        _seed_checkpoint(app, a.json()["id"], msgs)

        # The probe now sees a resumable checkpoint for this dir.
        probe = client.get("/api/coder/resumable",
                           headers=owner, params={"cwd": str(proj)}).json()
        assert probe["resumable"] is True
        assert probe["turns"] == 2 and probe["messages"] == 2

        # A fresh session with resume=true restores the conversation.
        b = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log", "resume": True})
        assert b.status_code == 200
        assert b.json()["resumed"] is True
        restored = app.state.coder_sessions.get(b.json()["id"]).agent._messages
        assert restored == msgs


def test_resume_false_does_not_restore(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], [{"role": "user", "content": "x"}])
        # Default (no resume) starts fresh.
        b = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        assert b.json().get("resumed") is False
        assert app.state.coder_sessions.get(b.json()["id"]).agent._messages == []


def test_resumable_and_resume_are_owner_only(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])
    sh = {"Authorization": f"Bearer {scoped['key']}"}

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], [{"role": "user", "content": "secret"}])

        # A scoped key is never told a resumable checkpoint exists.
        s = client.get("/api/coder/resumable", headers=sh,
                       params={"cwd": str(proj)}).json()
        assert s["resumable"] is False

        # And a restricted (scoped) session cannot resume - it starts fresh, NOT
        # loading the owner's prior conversation.
        b = client.post("/api/coder/sessions", headers=sh,
                        json={"cwd": str(proj), "resume": True})
        assert b.status_code == 200
        assert b.json().get("resumed") is False
        assert app.state.coder_sessions.get(b.json()["id"]).agent._messages == []


def test_restricted_session_does_not_clobber_the_owner_checkpoint(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])
    sh = {"Authorization": f"Bearer {scoped['key']}"}
    owner_msgs = [{"role": "user", "content": "owner work"}]

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], owner_msgs)

        # A restricted scoped session (forced to the project root) runs + persists.
        b = client.post("/api/coder/sessions", headers=sh,
                        json={"cwd": str(proj), "mode": "log"})
        sess = app.state.coder_sessions.get(b.json()["id"])
        assert sess.restricted is True
        sess.agent._messages = [{"role": "user", "content": "scoped work"}]
        sess.persist_checkpoint()                 # must be a NO-OP for restricted

        # The owner can still resume THEIR conversation, not the scoped one.
        c = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log", "resume": True})
        assert c.json()["resumed"] is True
        assert app.state.coder_sessions.get(c.json()["id"]).agent._messages == owner_msgs


def test_resumable_validates_cwd(tmp_path, monkeypatch):
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        # Missing cwd -> 400.
        assert client.get("/api/coder/resumable", headers=owner).status_code == 400
        # A non-existent directory -> not resumable (not an error).
        r = client.get("/api/coder/resumable", headers=owner,
                       params={"cwd": str(tmp_path / "nope")}).json()
        assert r["resumable"] is False


def test_agent_persist_then_resume_roundtrip_offline(tmp_path, monkeypatch):
    # The persist/resume mechanism works without the HTTP layer (LOG mode).
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    from unittest.mock import patch
    from localm.plugins.coder.audit import SessionMode

    proj = tmp_path / "proj"; proj.mkdir()
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        a = Agent(_StubBackend(), cwd=proj, mode=SessionMode.LOG,
                  auto_approve=True, self_verify=False)
        a._messages = [{"role": "user", "content": "hi"}]
        a._turns = 1
        a.save_checkpoint()

        b = Agent(_StubBackend(), cwd=proj, mode=SessionMode.LOG,
                  auto_approve=True, self_verify=False)
        data = b.load_checkpoint()
        assert data is not None
        b.resume_checkpoint(data)
        assert b._messages == [{"role": "user", "content": "hi"}]
