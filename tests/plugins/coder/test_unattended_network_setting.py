# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whether an unattended coder run may use network tools under net_mode "ask".

The ``coder_unattended_network`` setting ("deny" by default, or "allow") and the
``localm coder --allow-network`` flag decide it for a run nobody can be asked
in. net_mode "off" refuses network tools either way, a run that can ask still
asks, and the MCP ``run_coder_task`` caller cannot loosen it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.plugins.coder import runner as coder_runner
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import TOOL_REGISTRY, ToolResult

URL = "http://127.0.0.1:9/never-fetched"


class _Stub:
    model_id = "stub-model"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _set_setting(home: Path, value: str) -> None:
    (home / "config.json").write_text(
        json.dumps({"coder_unattended_network": value}), encoding="utf-8")


@pytest.fixture
def ran(monkeypatch):
    """Replace the network tools' bodies with recorders; returns the names run."""
    calls: list[str] = []
    for name in ("fetch_url", "web_search", "git_push"):
        orig = TOOL_REGISTRY[name]

        def fn(cwd, _name=name, **kw):
            calls.append(_name)
            return ToolResult.success(f"{_name} ran")

        monkeypatch.setitem(TOOL_REGISTRY, name, dataclasses.replace(orig, fn=fn))
    return calls


@pytest.fixture
def net_ask(monkeypatch):
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")


def _patched():
    pm = patch("localm.plugins.coder.agent.ProjectMap")
    return pm, patch("localm.plugins.coder.agent.make_audit_log"), \
        patch("localm.plugins.coder.agent.load_memory", return_value="")


def _build(cwd, *, task="do x", allow_network=False, auto_approve=False):
    pm, audit, mem = _patched()
    with pm as PM, audit, mem:
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        return coder_runner.build_agent(
            _Stub(), cwd, task=task, max_turns=3, auto_approve=auto_approve,
            always_confirm=None, session_mode=SessionMode.LOG,
            allow_network=allow_network)


def _call(name, **args):
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir(exist_ok=True)
    return p


class TestUnattendedRun:
    def test_denied_by_default(self, home, tmp_path, net_ask, ran):
        a = _build(_project(tmp_path))
        assert a.unattended_network is False

        res = a._execute_tool(_call("fetch_url", url=URL), interactive=False)

        assert res.ok is False
        assert "requires confirmation" in res.output, res.output
        assert ran == []
        assert a.denied_unconfirmed == [("fetch_url", "unconfirmable")]

    def test_allowed_by_the_setting(self, home, tmp_path, net_ask, ran):
        _set_setting(home, "allow")
        a = _build(_project(tmp_path))

        for call in (_call("fetch_url", url=URL), _call("web_search", query="q"),
                     _call("git_push")):
            res = a._execute_tool(call, interactive=False)
            assert res.ok is True, res.output

        assert ran == ["fetch_url", "web_search", "git_push"]
        assert a.denied_unconfirmed == []

    def test_setting_deny_value_denies(self, home, tmp_path, net_ask, ran):
        _set_setting(home, "deny")
        a = _build(_project(tmp_path))

        res = a._execute_tool(_call("web_search", query="q"), interactive=False)

        assert res.ok is False
        assert ran == []

    def test_allowed_by_the_flag_through_the_cli(self, home, tmp_path, monkeypatch,
                                                  net_ask, ran):
        from localm.plugins.coder.cli import _main
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("coder", True)
        proj = _project(tmp_path)
        built = []

        def build_agent(*a, **k):
            pm, audit, mem = _patched()
            with pm as PM, audit, mem:
                PM.build.return_value.file_count.return_value = 0
                PM.build.return_value.truncated = False
                built.append(coder_runner.build_agent(*a, **k))
            return built[-1]

        def run_single_task(agent, task):
            agent.flag_result = agent._execute_tool(
                _call("fetch_url", url=URL), interactive=False)
            return MagicMock(success=True, response="", turns=1, total_tokens=0,
                             denied=[])

        monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: _Stub())
        monkeypatch.setattr(_main, "build_agent", build_agent)
        monkeypatch.setattr(_main, "run_single_task", run_single_task)
        monkeypatch.setattr(_main, "finish_agent", lambda a: None)

        def invoke(*extra):
            return CliRunner().invoke(
                _main.main, ["-m", "stub-model", "--cwd", str(proj), *extra, "do x"])

        result = invoke()
        assert result.exit_code == 0, result.output
        result = invoke("--allow-network")
        assert result.exit_code == 0, result.output

        assert [b.unattended_network for b in built] == [False, True]
        assert built[0].flag_result.ok is False
        assert built[1].flag_result.ok is True, built[1].flag_result.output
        assert ran == ["fetch_url"]

    @pytest.mark.parametrize("setting,flag", [("allow", False), ("deny", True),
                                              ("allow", True)])
    def test_net_mode_off_beats_setting_and_flag(self, home, tmp_path, monkeypatch,
                                                 ran, setting, flag):
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")
        _set_setting(home, setting)
        a = _build(_project(tmp_path), allow_network=flag)
        assert a.unattended_network is True

        res = a._execute_tool(_call("fetch_url", url=URL), interactive=False)

        assert res.ok is False
        assert "net_mode=off" in res.output, res.output
        assert ran == []

    def test_untrusted_content_still_denies_when_allowed(self, home, tmp_path,
                                                         net_ask, ran):
        _set_setting(home, "allow")
        a = _build(_project(tmp_path))
        a._untrusted_in_context = True

        res = a._execute_tool(_call("fetch_url", url=URL), interactive=False)

        assert res.ok is False
        assert ran == []
        assert a.denied_unconfirmed == [("fetch_url", "untrusted-content")]

    def test_a_child_agent_inherits_the_allowance(self, home, tmp_path):
        from localm.plugins.coder.tools.agents import inherit_session_state
        parent = _build(_project(tmp_path), allow_network=True)
        child = MagicMock()

        inherit_session_state(parent, child)

        assert child.unattended_network is True


class TestRunsThatCanAskStillAsk:
    def test_a_confirm_handler_is_asked_with_the_setting_on(
            self, home, tmp_path, net_ask, ran):
        _set_setting(home, "allow")
        a = _build(_project(tmp_path), allow_network=True)
        asked = []

        def handler(call):
            asked.append(call.name)
            return False

        a.confirm_handler = handler
        res = a._execute_tool(_call("fetch_url", url=URL), interactive=False)

        assert asked == ["fetch_url"]
        assert res.ok is False
        assert ran == []

    def test_an_interactive_run_is_asked_with_the_setting_on(
            self, home, tmp_path, net_ask, ran):
        _set_setting(home, "allow")
        a = _build(_project(tmp_path), task="", allow_network=True,
                   auto_approve=True)
        asked = []

        def confirm(call):
            asked.append(call.name)
            return False

        a._confirm_tool = confirm
        res = a._execute_tool(_call("web_search", query="q"), interactive=True)

        assert asked == ["web_search"]
        assert res.ok is False
        assert ran == []


class TestSettingValidation:
    def test_bad_value_is_rejected(self):
        from localm.settings_schema import validate_update
        with pytest.raises(ValueError, match="coder_unattended_network"):
            validate_update({"coder_unattended_network": "sometimes"})

    def test_both_values_are_accepted(self):
        from localm.settings_schema import validate_update
        for v in ("deny", "allow"):
            assert validate_update({"coder_unattended_network": v}) == {
                "coder_unattended_network": v}

    def test_default_is_deny(self, home):
        from localm.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["coder_unattended_network"] == "deny"
        assert coder_runner.unattended_network_setting() is False

    def test_localm_config_rejects_a_bad_value_and_stores_a_good_one(self, home):
        from localm.cli.models import config_cmd
        bad = CliRunner().invoke(config_cmd, ["coder_unattended_network", "yes"])
        assert bad.exit_code != 0
        assert coder_runner.unattended_network_setting() is False

        good = CliRunner().invoke(config_cmd, ["coder_unattended_network", "allow"])
        assert good.exit_code == 0, good.output
        assert coder_runner.unattended_network_setting() is True

    def test_the_setting_is_owner_only(self):
        from localm.settings_schema import admin_only_keys
        assert "coder_unattended_network" in admin_only_keys()


# --------------------------------------------------------------------------- #
#  MCP run_coder_task follows the setting and cannot loosen it                #
# --------------------------------------------------------------------------- #

def _scripted_factory(script):
    def factory(model_name):
        engine = MagicMock()
        engine.display_name = model_name
        engine.active_requests = 0
        engine.unloading = False
        engine.supports_grammar = False
        engine.context_capacity.return_value = 8192
        engine.count_messages_tokens.return_value = 10
        engine.count_tokens.return_value = 2
        calls = []

        def chat_stream(messages, **kw):
            reply = script[min(len(calls), len(script) - 1)]
            calls.append(messages)
            return iter([reply])

        engine.chat_stream.side_effect = chat_stream
        return engine
    return factory


@pytest.fixture
def coder_mcp(home, monkeypatch):
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)
    return home


def _mcp_run(tmp_path, *, coder_shell, args):
    from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools
    script = ["<tool_call>" + json.dumps({"name": "fetch_url", "args": {"url": URL}})
              + "</tool_call>", "done"]
    engines = EngineCache(default_model="stub-model",
                          engine_factory=_scripted_factory(script))
    server = MCPStdioServer(build_tools(engines, enable_images=False,
                                        coder_shell=coder_shell))
    full = {"task": "fetch it", "cwd": str(_project(tmp_path))}
    full.update(args)
    pm, audit, mem = _patched()
    with pm as PM, audit, mem:
        PM.build.return_value.file_count.return_value = 0
        return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "run_coder_task",
                                         "arguments": full}})


class TestMcpRunCoderTask:
    @pytest.mark.parametrize("coder_shell", [False, True])
    @pytest.mark.parametrize("args", [
        {},
        {"yes": True},
        {"yes": True, "allow_network": True, "auto_approve": True,
         "net_mode": "allow", "coder_unattended_network": "allow"},
    ])
    def test_setting_deny_denies_whatever_the_caller_sends(
            self, coder_mcp, tmp_path, net_ask, ran, coder_shell, args):
        resp = _mcp_run(tmp_path, coder_shell=coder_shell, args=args)
        text = resp["result"]["content"][0]["text"]

        assert ran == [], text
        assert "[denied]" in text and "fetch_url" in text, text

    def test_setting_allow_lets_the_run_fetch(self, coder_mcp, tmp_path, net_ask,
                                              ran):
        _set_setting(coder_mcp, "allow")
        resp = _mcp_run(tmp_path, coder_shell=False, args={})
        text = resp["result"]["content"][0]["text"]

        assert ran == ["fetch_url"], text
        assert "[denied]" not in text, text

    def test_the_tool_schema_has_no_network_argument(self, coder_mcp):
        from localm.plugins.mcpserver.server import EngineCache, build_tools
        engines = EngineCache(default_model="stub-model",
                              engine_factory=_scripted_factory(["done"]))
        for shell in (False, True):
            tools = build_tools(engines, enable_images=False, coder_shell=shell)
            props = set(tools["run_coder_task"]["inputSchema"]["properties"])
            assert props == {"task", "cwd", "model", "max_turns", "yes",
                             "timeout_seconds"}
