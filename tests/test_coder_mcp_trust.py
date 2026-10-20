# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project MCP servers start only once the project config is trusted
(coder/mcp.py, coder/hooks.py, coder/hook_trust.py, agent/core.py): a real MCP
server process that leaves a marker file when it is spawned, a real trust store
under the per-test LOCALM_HOME."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder import hook_trust
from localm.plugins.coder import mcp as mcp_mod
from localm.plugins.coder.hook_trust import commands_digest, hooks_digest
from localm.plugins.coder.hooks import ToolHooks
from localm.plugins.coder.project_config import read_project_config_file
from localm.plugins.coder.tools import TOOL_REGISTRY

_MARKER_SERVER = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    with open(sys.argv[1], "a", encoding="utf-8") as f:
        f.write("spawned\\n")

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                "serverInfo": {"name": "mark", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
                "name": "ping", "description": "ping",
                "inputSchema": {"type": "object", "properties": {}}}]}})
        elif method == "tools/call":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "pong"}]}})
""")

TOOL = "mcp_mark_ping"


@pytest.fixture()
def proj(tmp_path):
    """A project declaring one MCP server that appends to a marker file outside
    the project each time it is spawned. Yields (project, marker, script)."""
    project = tmp_path / "proj"
    (project / ".localcoder").mkdir(parents=True)
    script = tmp_path / "marker_server.py"
    script.write_text(_MARKER_SERVER, encoding="utf-8")
    marker = tmp_path / "SPAWNED.txt"
    _write_config(project, script, marker)
    mcp_mod.stop_pooled_servers()
    TOOL_REGISTRY.pop(TOOL, None)
    try:
        yield project, marker, script
    finally:
        TOOL_REGISTRY.pop(TOOL, None)
        mcp_mod.stop_pooled_servers()


def _write_config(project: Path, script: Path, marker: Path, *, extra_args=(),
                  env=None, hooks: str = "") -> None:
    args = [str(script), str(marker), *extra_args]
    body = (f"[mcp.servers.mark]\ncommand = {json.dumps(sys.executable)}\n"
            f"args = [{', '.join(json.dumps(a) for a in args)}]\n")
    if env:
        body += "env = {" + ", ".join(f"{k} = {json.dumps(v)}" for k, v in env.items()) + "}\n"
    (project / ".localcoder" / "config.toml").write_text(hooks + body, encoding="utf-8")


def _config_path(project: Path) -> Path:
    return project / ".localcoder" / "config.toml"


def _trust(project: Path) -> None:
    path = _config_path(project)
    hook_trust.remember_trust(path, commands_digest(read_project_config_file(path)))


def _spawned(marker: Path) -> int:
    return len(marker.read_text(encoding="utf-8").splitlines()) if marker.exists() else 0


def _agent(cwd: Path, *, native: bool = False, **kw):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = native
    kw.setdefault("auto_approve", True)
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        PM.build.return_value.dirty = False
        return Agent(backend=backend, cwd=cwd, **kw)


def _prompt(agent) -> str:
    return str(agent._system_prompt)


# --------------------------------------------------------------------------- #
#  digest
# --------------------------------------------------------------------------- #

def test_a_hooks_only_config_keeps_the_hooks_digest():
    hooks = {"pre_tool_use": [{"command": "x", "matcher": "write_file"}]}
    assert commands_digest({"hooks": hooks, "model": "m"}) == hooks_digest(hooks)
    assert commands_digest({"model": "m"}) == hooks_digest(None)


def test_a_hooks_only_toolhooks_digest_is_the_hooks_digest(tmp_path):
    project = tmp_path / "p"
    (project / ".localcoder").mkdir(parents=True)
    _config_path(project).write_text(
        '[[hooks.pre_tool_use]]\ncommand = "x"\nmatcher = "write_file"\n', encoding="utf-8")
    raw = read_project_config_file(_config_path(project))
    loaded = ToolHooks.load(project)
    assert loaded is not None and loaded.digest == hooks_digest(raw["hooks"])
    assert not loaded.has_mcp()


@pytest.mark.parametrize("change", [
    {"command": "other"}, {"args": ["a", "b"]}, {"env": {"NODE_OPTIONS": "--require x.js"}},
    {"trusted": True},
])
def test_any_server_change_changes_the_digest(change):
    base = {"mcp": {"servers": {"s": {"command": "c", "args": ["a"]}}}}
    changed = json.loads(json.dumps(base))
    changed["mcp"]["servers"]["s"].update(change)
    assert commands_digest(base) != commands_digest(changed)


def test_an_mcp_digest_never_equals_a_hooks_only_digest():
    raw = {"mcp": {"servers": {"s": {"command": "c"}}}}
    lookalike = {"hooks": {"mcp": raw["mcp"]}}
    assert commands_digest(raw) != commands_digest(lookalike)
    assert commands_digest(raw) != hooks_digest(raw)


# --------------------------------------------------------------------------- #
#  the gate at session start
# --------------------------------------------------------------------------- #

def test_an_untrusted_server_is_not_spawned_and_registers_nothing(proj):
    project, marker, _ = proj
    agent = _agent(project)
    assert _spawned(marker) == 0
    assert TOOL not in TOOL_REGISTRY
    assert agent._mcp_tool_names == frozenset()
    assert mcp_mod._POOL == {}
    assert TOOL not in _prompt(agent)
    assert agent._mcp_deferred is agent._tool_hooks


def test_a_trusted_server_starts_with_the_session(proj):
    project, marker, _ = proj
    _trust(project)
    agent = _agent(project)
    assert _spawned(marker) == 1
    assert agent._mcp_tool_names == {TOOL}
    assert TOOL in _prompt(agent)
    assert agent._mcp_deferred is None


@pytest.mark.parametrize("edit", ["args", "env"])
def test_editing_a_trusted_server_needs_trust_again(proj, edit):
    project, marker, script = proj
    _trust(project)
    if edit == "args":
        _write_config(project, script, marker, extra_args=["--extra"])
    else:
        _write_config(project, script, marker, env={"EXTRA": "1"})
    agent = _agent(project)
    assert _spawned(marker) == 0
    assert TOOL not in TOOL_REGISTRY and agent._mcp_tool_names == frozenset()


def test_an_unreadable_trust_store_starts_nothing(proj):
    project, marker, _ = proj
    hook_trust._store().parent.mkdir(parents=True, exist_ok=True)
    hook_trust._store().write_text("{not json", encoding="utf-8")
    _agent(project)
    assert _spawned(marker) == 0


def test_register_mcp_tools_alone_refuses_an_untrusted_config(proj):
    project, marker, _ = proj
    names, warnings = mcp_mod.register_mcp_tools(project)
    assert names == [] and _spawned(marker) == 0
    assert len(warnings) == 1 and "not trusted" in warnings[0] and "'mark'" in warnings[0]
    _trust(project)
    names, warnings = mcp_mod.register_mcp_tools(project)
    assert names == [TOOL] and warnings == [] and _spawned(marker) == 1


def test_the_servers_started_are_the_ones_trusted_not_a_later_edit(proj):
    """A config edited after the session read and trusted it does not change
    what the session starts."""
    project, marker, script = proj
    agent = _agent(project)
    _write_config(project, script, marker, extra_args=["--changed"])
    agent._tool_hooks.trust_for_session()
    agent.start_deferred_mcp()
    server = TOOL_REGISTRY[TOOL].fn._mcp_server
    assert "--changed" not in server.args


def test_a_child_agent_starts_nothing_of_its_own(proj):
    from localm.plugins.coder.tools.agents import inherited_child_kwargs
    project, marker, _ = proj
    parent = _agent(project)
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.dirty = False
        child = Agent(**inherited_child_kwargs(
            parent, backend=parent.backend, cwd=project, name="kid",
            max_turns=3, confirm_handler=None))
    assert _spawned(marker) == 0
    assert child._mcp_tool_names == frozenset() and child._mcp_deferred is None


# --------------------------------------------------------------------------- #
#  starting held-back servers once trusted
# --------------------------------------------------------------------------- #

def test_session_trust_starts_the_held_back_servers(proj):
    project, marker, _ = proj
    agent = _agent(project, native=True)
    agent.backend.set_tools.reset_mock()
    assert agent.start_deferred_mcp() == []          # still untrusted
    assert _spawned(marker) == 0
    agent._tool_hooks.trust_for_session()
    assert agent.start_deferred_mcp() == [TOOL]
    assert _spawned(marker) == 1
    assert agent._mcp_tool_names == {TOOL} and agent._mcp_deferred is None
    assert TOOL in _prompt(agent)
    agent.backend.set_tools.assert_called_once()
    assert agent.backend.set_tools.call_args.args[0] == agent._native_tool_defs()
    assert agent.start_deferred_mcp() == []          # only once
    assert _spawned(marker) == 1


def test_trust_granted_elsewhere_starts_them_at_the_next_task(proj):
    project, marker, _ = proj
    agent = _agent(project)
    ToolHooks.load(project).remember_trust()          # what the GUI route does
    assert _spawned(marker) == 0
    class _Stop(Exception):
        pass

    with patch.object(type(agent), "report_deferred_mcp", side_effect=_Stop):
        with pytest.raises(_Stop):
            agent._loop(interactive=False)
    assert _spawned(marker) == 1 and agent._mcp_tool_names == {TOOL}


def test_a_restricted_session_never_starts_them_even_when_trusted(proj):
    project, marker, _ = proj
    _trust(project)
    agent = _agent(project, restricted=True)
    assert agent.start_deferred_mcp() == []
    assert _spawned(marker) == 0 and agent._tool_hooks is None


def test_the_untrusted_notice_is_given_once(proj):
    project, _, _ = proj
    agent = _agent(project)
    events = []
    agent.on_event = events.append
    with patch("localm.plugins.coder.agent.print_warning") as warn:
        agent.report_deferred_mcp(interactive=False)
        agent.report_deferred_mcp(interactive=False)
    assert warn.call_count == 1
    assert warn.call_args.kwargs.get("to_stderr") is True
    assert "not trusted" in warn.call_args.args[0] and "'mark'" in warn.call_args.args[0]
    assert [e["type"] for e in events] == ["info"]


# --------------------------------------------------------------------------- #
#  CLI trust paths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("answer, spawned, remembered", [
    ("a", 1, True), ("y", 1, False), ("", 0, False), ("n", 0, False),
])
def test_the_startup_prompt_lists_the_server_and_obeys_the_answer(
        proj, answer, spawned, remembered):
    from localm.plugins.coder.cli import hooks_cli
    project, marker, _ = proj
    agent = _agent(project)
    printed = []
    with patch.object(hooks_cli.console, "input", return_value=answer), \
         patch.object(hooks_cli.console, "print", side_effect=lambda *a, **k: printed.append(
             " ".join(str(x) for x in a))):
        hooks_cli.resolve_hook_trust(agent, trust_flag=False, can_prompt=True)
    shown = "\n".join(printed)
    assert "MCP server" in shown and "marker_server.py" in shown
    assert _spawned(marker) == spawned
    assert (TOOL in agent._mcp_tool_names) is bool(spawned)
    assert ToolHooks.load(project).trusted is remembered


def test_the_flag_trusts_and_starts_them(proj):
    from localm.plugins.coder.cli.hooks_cli import resolve_hook_trust
    project, marker, _ = proj
    agent = _agent(project)
    resolve_hook_trust(agent, trust_flag=True, can_prompt=False)
    assert _spawned(marker) == 1 and ToolHooks.load(project).trusted


def test_without_a_terminal_it_says_so_once_and_starts_nothing(proj):
    from localm.plugins.coder.cli import hooks_cli
    project, marker, _ = proj
    agent = _agent(project)
    with patch.object(hooks_cli, "print_warning") as warn, \
         patch("localm.plugins.coder.agent.print_warning") as agent_warn:
        hooks_cli.resolve_hook_trust(agent, trust_flag=False, can_prompt=False)
        agent.report_deferred_mcp(interactive=False)
    assert warn.call_count == 1 and "--trust-hooks" in warn.call_args.args[0]
    assert warn.call_args.kwargs.get("to_stderr") is True
    assert agent_warn.call_count == 0
    assert _spawned(marker) == 0


def test_hooks_command_trust_starts_them_and_forget_withdraws(proj):
    from localm.plugins.coder.cli.hooks_cli import handle_hooks_command
    project, marker, _ = proj
    agent = _agent(project)
    handle_hooks_command("trust", agent)
    assert _spawned(marker) == 1 and agent._mcp_tool_names == {TOOL}
    handle_hooks_command("forget", agent)
    assert not ToolHooks.load(project).trusted
    agent2 = _agent(project)
    assert agent2._mcp_tool_names == frozenset() and _spawned(marker) == 1


def test_trusting_another_folders_servers_after_cd_waits_for_the_next_session(proj, tmp_path):
    from localm.plugins.coder.cli import hooks_cli
    project, marker, _ = proj
    other = tmp_path / "plain"
    other.mkdir()
    agent = _agent(other)
    agent.set_cwd(project)
    with patch.object(hooks_cli, "print_info") as info:
        hooks_cli.handle_hooks_command("trust", agent)
    assert _spawned(marker) == 0
    assert any("next session" in c.args[0] for c in info.call_args_list)
    assert ToolHooks.load(project).trusted


@pytest.mark.parametrize("answer", ["a", "y"])
def test_cd_within_the_project_after_declining_then_trusting_starts_them(proj, answer):
    from localm.plugins.coder.cli import hooks_cli
    project, marker, _ = proj
    sub = project / "src"
    sub.mkdir()
    agent = _agent(project)
    with patch.object(hooks_cli.console, "input", return_value="n"):
        hooks_cli.resolve_hook_trust(agent, trust_flag=False, can_prompt=True)
    agent.set_cwd(sub)
    with patch.object(hooks_cli.console, "input", return_value=answer):
        hooks_cli.resolve_hook_trust(agent, trust_flag=False, can_prompt=True)
    assert _spawned(marker) == 1 and agent._mcp_tool_names == {TOOL}


def test_session_trust_of_another_folders_servers_says_they_do_not_start(proj, tmp_path):
    from localm.plugins.coder.cli import hooks_cli
    project, marker, _ = proj
    other = tmp_path / "plain"
    other.mkdir()
    agent = _agent(other)
    agent.set_cwd(project)
    with patch.object(hooks_cli.console, "input", return_value="y"), \
         patch.object(hooks_cli, "print_info") as info:
        hooks_cli.resolve_hook_trust(agent, trust_flag=False, can_prompt=True)
    said = " ".join(c.args[0] for c in info.call_args_list)
    assert "are not started" in said and "next session" not in said
    assert _spawned(marker) == 0 and not ToolHooks.load(project).trusted


@pytest.mark.parametrize("args_toml, argv_tail", [
    ("5", ["5"]), ("true", ["True"]), ('"ab"', ["a", "b"]), ("[1, \"x\"]", ["1", "x"]),
])
def test_a_scalar_args_value_is_shown_not_a_crash(tmp_path, args_toml, argv_tail):
    project = tmp_path / "p"
    (project / ".localcoder").mkdir(parents=True)
    _config_path(project).write_text(
        f'[mcp.servers.s]\ncommand = "c"\nargs = {args_toml}\n', encoding="utf-8")
    hooks = ToolHooks.load(project)
    assert mcp_mod.mcp_argv(hooks.mcp_servers["s"]) == ["c", *argv_tail]
    assert hooks.describe_mcp()[0].startswith('MCP server "s": ["c"')


def test_the_server_line_escapes_control_characters(tmp_path):
    project = tmp_path / "p"
    (project / ".localcoder").mkdir(parents=True)
    _config_path(project).write_text(
        '[mcp.servers.s]\ncommand = "safe\\u001b[2K\\revil"\n', encoding="utf-8")
    line = ToolHooks.load(project).describe_mcp()[0]
    assert "\x1b" not in line and "\r" not in line
    assert "\\u001b" in line and "\\r" in line


# --------------------------------------------------------------------------- #
#  doctor
# --------------------------------------------------------------------------- #

def test_doctor_starts_only_trusted_servers(proj):
    from localm.plugins.coder.doctor import WARN, OK, check_mcp
    project, marker, _ = proj
    checks = check_mcp(project)
    assert [(c.name, c.status) for c in checks] == [("mcp:mark", WARN)]
    assert _spawned(marker) == 0
    _trust(project)
    checks = check_mcp(project)
    assert [(c.name, c.status) for c in checks] == [("mcp:mark", OK)]
    assert _spawned(marker) == 1


def test_doctor_probes_servers_the_session_trusts(proj):
    from localm.plugins.coder.doctor import OK, check_mcp
    project, marker, _ = proj
    servers = ToolHooks.load(project).mcp_servers
    checks = check_mcp(project, trusted_servers=servers)
    assert [(c.name, c.status) for c in checks] == [("mcp:mark", OK)]


# --------------------------------------------------------------------------- #
#  web routes
# --------------------------------------------------------------------------- #

def _owner_app(tmp_path, monkeypatch, project):
    from fastapi import FastAPI
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
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
    app.state.root_dir = str(project)
    return app, {"Authorization": "Bearer ownersecret"}


def test_the_routes_list_trust_and_start_mcp_servers(proj, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    project, marker, script = proj
    app, owner = _owner_app(tmp_path, monkeypatch, project)
    with TestClient(app) as client:
        data = client.get("/api/coder/hooks", headers=owner,
                          params={"cwd": str(project)}).json()
        assert data["configured"] is True and data["trusted"] is False
        assert data["hooks"] == []
        (srv,) = data["mcp_servers"]
        assert srv["name"] == "mark" and srv["argv"][1] == str(script)
        assert srv["trusted_calls"] is False and "marker_server.py" in srv["line"]
        assert _spawned(marker) == 0

        _write_config(project, script, marker, extra_args=["--sneaky"])
        stale = client.post("/api/coder/hooks/trust", headers=owner,
                            json={"cwd": str(project), "digest": data["digest"]})
        assert stale.status_code == 409
        assert not ToolHooks.load(project).trusted

        fresh = client.get("/api/coder/hooks", headers=owner,
                           params={"cwd": str(project)}).json()
        ok = client.post("/api/coder/hooks/trust", headers=owner,
                         json={"cwd": str(project), "digest": fresh["digest"]})
        assert ok.status_code == 200, ok.text
        assert ToolHooks.load(project).trusted
    assert _spawned(marker) == 0
    agent = _agent(project)
    assert _spawned(marker) == 1 and agent._mcp_tool_names == {TOOL}
