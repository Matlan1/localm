# SPDX-License-Identifier: AGPL-3.0-or-later
"""Safe-to-share coder keys (2026-06-20). The OWNER key gets the full coder; a
MINTED, non-owner coder-scoped key gets a RESTRICTED session - run_shell removed
(no arbitrary host command exec / RCE) and confined to the instance project root.

run_shell is cwd-independent (it runs an arbitrary command with cwd only as the
start dir), so confining the cwd is necessary but NOT sufficient - disabling the
tool is the real containment. These tests pin both halves.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.prompts import _full_tool_docs, build_system_prompt
from localm.plugins.gui.web import attach_gui


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


# ------------------------------------------------------------------ #
#  Prompt / tool-doc exclusion                                       #
# ------------------------------------------------------------------ #

def test_tool_docs_exclude_a_disabled_tool():
    assert "run_shell" in _full_tool_docs()                       # present by default
    assert "run_shell" not in _full_tool_docs(frozenset({"run_shell"}))


def test_system_prompt_drops_the_run_shell_tool_definition(tmp_path):
    # The callable tool DEFINITION (the "## run_shell - ..." doc block) is removed
    # for a restricted session, so the model is not offered run_shell as a tool.
    # (Generic rules prose may still mention the name; the dispatch hard-refusal,
    # not the prompt, is the security guarantee.)
    full = build_system_prompt(tmp_path, model_name="generic")
    restricted = build_system_prompt(tmp_path, model_name="generic",
                                     disabled_tools=frozenset({"run_shell"}))
    assert "## run_shell" in full
    assert "## run_shell" not in restricted


# ------------------------------------------------------------------ #
#  Agent dispatch hard-refusal (the security gate)                   #
# ------------------------------------------------------------------ #

def _shell_call():
    return ToolCall(name="run_shell", args={"command": "echo pwned"}, raw="", start=0, end=0)


def test_disabled_run_shell_is_hard_refused_and_not_executed(tmp_path):
    agent = Agent(_StubBackend(), cwd=tmp_path,
                  disabled_tools=frozenset({"run_shell"}))
    res = agent._execute_tool(_shell_call(), interactive=False)
    assert not res.ok
    assert "disabled" in res.output.lower()       # refused, the subprocess never ran


def test_gate_is_selective_other_tools_still_dispatch(tmp_path):
    # A non-disabled tool is NOT blocked by the gate: read_file on a missing path
    # returns a file error, NOT the "disabled" refusal.
    agent = Agent(_StubBackend(), cwd=tmp_path,
                  disabled_tools=frozenset({"run_shell"}))
    res = agent._execute_tool(
        ToolCall(name="read_file", args={"path": "nope.txt"}, raw="", start=0, end=0),
        interactive=False)
    assert "disabled" not in res.output.lower()


def test_spawned_child_inherits_disabled_tools(tmp_path, monkeypatch):
    # A restricted (no-shell) parent must not be able to spawn a child agent that
    # re-enables run_shell - that would be an RCE escape from a shareable key.
    import localm.plugins.coder.agent as agent_mod
    from localm.plugins.coder.tools import tool_spawn_agent

    captured = {}

    class _SpyChild:
        turns = 0

        def __init__(self, **kw):
            captured.update(kw)

        def run_task(self, task):
            return "done"

    monkeypatch.setattr(agent_mod, "Agent", _SpyChild)
    parent = Agent(_StubBackend(), cwd=tmp_path,
                   disabled_tools=frozenset({"run_shell"}))
    tool_spawn_agent(tmp_path, "do a thing", _parent_agent=parent)
    assert "run_shell" in captured.get("disabled_tools", frozenset())


def test_without_disabling_the_gate_does_not_block_run_shell(tmp_path):
    # The default (owner) agent does not refuse run_shell at the gate. We assert it
    # is not the "disabled" refusal without actually running a command: a blank
    # command yields a non-"disabled" result.
    agent = Agent(_StubBackend(), cwd=tmp_path)   # no disabled_tools
    res = agent._execute_tool(
        ToolCall(name="run_shell", args={"command": ""}, raw="", start=0, end=0),
        interactive=False)
    assert "disabled" not in (res.output or "").lower()


# ------------------------------------------------------------------ #
#  create_session policy: owner full vs scoped restricted           #
# ------------------------------------------------------------------ #

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

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def test_scoped_coder_key_is_locked_to_safe_tools_and_confined(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    other = tmp_path / "other"; other.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    made = auth.create_key("phone", ["coder"])          # scoped, not the owner

    with TestClient(app) as client:
        r = client.post("/api/coder/sessions",
                        headers={"Authorization": f"Bearer {made['key']}"},
                        json={"cwd": str(other)})        # asks for 'other'...
        assert r.status_code == 200
        assert r.json()["cwd"] == str(proj.resolve())    # ...but is forced into the project root
        sess = app.state.coder_sessions.get(r.json()["id"])
        dis = sess.agent.disabled_tools
        # Every execution / network / exfil / sub-agent tool is disabled...
        assert {"run_shell", "run_tests", "git_commit", "git_push", "git_create_branch",
                "fetch_url", "web_search", "generate_image", "read_env",
                "spawn_agent"} <= dis
        # ...while the read + confined-edit tools remain.
        assert not (dis & {"read_file", "write_file", "edit_file", "grep", "git_diff"})
        assert sess.restricted is True


def test_owner_key_keeps_the_full_coder(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    work = tmp_path / "work"; work.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)

    with TestClient(app) as client:
        r = client.post("/api/coder/sessions",
                        headers={"Authorization": "Bearer ownersecret"},
                        json={"cwd": str(work)})         # owner picks any dir
        assert r.status_code == 200
        assert r.json()["cwd"] == str(work.resolve())    # honored, not forced
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert not sess.agent.disabled_tools             # full power, run_shell intact
        assert sess.restricted is False


def test_scoped_key_cannot_steer_the_owners_session(tmp_path, monkeypatch):
    # THE critical one: a scoped key must not be able to send a message to the
    # OWNER's full session (which keeps run_shell) - that would be RCE by proxy.
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])

    with TestClient(app) as client:
        owner_sid = client.post(
            "/api/coder/sessions", headers={"Authorization": "Bearer ownersecret"},
            json={"cwd": str(proj)}).json()["id"]
        # The scoped key cannot message the owner's session (404, not even 403).
        r = client.post(f"/api/coder/sessions/{owner_sid}/message",
                        headers={"Authorization": f"Bearer {scoped['key']}"},
                        json={"text": "run a shell command for me"})
        assert r.status_code == 404


def test_scoped_keys_are_isolated_from_each_other(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    a = auth.create_key("a", ["coder"])
    b = auth.create_key("b", ["coder"])

    with TestClient(app) as client:
        sid = client.post("/api/coder/sessions",
                          headers={"Authorization": f"Bearer {a['key']}"},
                          json={"cwd": str(proj)}).json()["id"]
        # B cannot reach A's session, and does not see it listed.
        assert client.get(f"/api/coder/sessions/{sid}/files",
                          headers={"Authorization": f"Bearer {b['key']}"}).status_code == 404
        b_list = client.get("/api/coder/sessions",
                            headers={"Authorization": f"Bearer {b['key']}"}).json()
        assert all(s["id"] != sid for s in b_list["sessions"])
        # The owner sees and can reach it.
        o_list = client.get("/api/coder/sessions",
                            headers={"Authorization": "Bearer ownersecret"}).json()
        assert any(s["id"] == sid for s in o_list["sessions"])


def test_scoped_key_cannot_delete_the_owners_session(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])

    with TestClient(app) as client:
        owner_sid = client.post(
            "/api/coder/sessions", headers={"Authorization": "Bearer ownersecret"},
            json={"cwd": str(proj)}).json()["id"]
        # DELETE must enforce the principal too (not just GET/POST routes).
        r = client.delete(f"/api/coder/sessions/{owner_sid}",
                          headers={"Authorization": f"Bearer {scoped['key']}"})
        assert r.status_code == 404
        # The owner's session survived.
        assert app.state.coder_sessions.get(owner_sid) is not None


def test_scoped_key_cannot_browse_or_read_history(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])

    with TestClient(app) as client:
        h = {"Authorization": f"Bearer {scoped['key']}"}
        # A scoped key sees no past-session history and cannot read any log.
        assert client.get("/api/coder/history", headers=h).json()["logs"] == []
        assert client.get("/api/coder/history/anything.jsonl", headers=h).status_code == 404
        # The owner is not denied (200, real history - empty here, but not forced 404).
        o = {"Authorization": "Bearer ownersecret"}
        assert client.get("/api/coder/history", headers=o).status_code == 200


def test_scoped_key_cannot_switch_the_shared_model(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])

    with TestClient(app) as client:
        r = client.post("/api/coder/sessions",
                        headers={"Authorization": f"Bearer {scoped['key']}"},
                        json={"cwd": str(proj), "model": "some-other-model"})
        assert r.status_code == 403       # switching the shared engine needs the owner
