# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the localm MCP server plugin (plugins/mcpserver)."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.mcpserver.server import (
    EngineCache,
    MCPStdioServer,
    build_tools,
)


def _stub_engine_factory(model_name):
    engine = MagicMock()
    engine.display_name = model_name
    engine.chat_stream.side_effect = lambda messages, **kw: iter(
        [f"reply-from-{model_name}"])
    engine.embed.return_value = [[0.1, 0.2]]
    return engine


def _server(default_model="stub-model", enable_images=True):
    engines = EngineCache(default_model=default_model,
                          engine_factory=_stub_engine_factory)
    return MCPStdioServer(build_tools(engines, enable_images=enable_images)), engines


def _req(server, method, params=None, mid=1):
    return server.handle(
        {"jsonrpc": "2.0", "id": mid, "method": method,
         "params": params or {}})


class TestProtocol:
    def test_initialize(self):
        server, _ = _server()
        resp = _req(server, "initialize")
        assert resp["result"]["serverInfo"]["name"] == "localm"
        assert resp["result"]["capabilities"] == {"tools": {}}

    def test_notification_gets_no_reply(self):
        server, _ = _server()
        assert server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_ping(self):
        server, _ = _server()
        assert _req(server, "ping")["result"] == {}

    def test_unknown_method_errors(self):
        server, _ = _server()
        resp = _req(server, "no/such/method")
        assert resp["error"]["code"] == -32601

    def test_tools_list_includes_all(self):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert names == {"chat", "list_models", "embed", "generate_image"}

    def test_no_images_flag_hides_tool(self):
        server, _ = _server(enable_images=False)
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "generate_image" not in names


class TestToolCalls:
    def test_chat_returns_model_output(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {"prompt": "hi"}})
        result = resp["result"]
        assert result["isError"] is False
        assert result["content"][0]["text"] == "reply-from-stub-model"

    def test_chat_model_override(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat",
                     "arguments": {"prompt": "hi", "model": "other-model"}})
        assert resp["result"]["content"][0]["text"] == "reply-from-other-model"

    def test_chat_missing_prompt_is_error(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {}})
        assert resp["result"]["isError"] is True

    def test_embed_returns_vectors(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "embed", "arguments": {"texts": ["a"]}})
        assert json.loads(resp["result"]["content"][0]["text"]) == [[0.1, 0.2]]

    def test_unknown_tool_is_jsonrpc_error(self):
        server, _ = _server()
        resp = _req(server, "tools/call", {"name": "nope", "arguments": {}})
        assert resp["error"]["code"] == -32602

    def test_handler_crash_becomes_tool_error_not_protocol_error(self):
        server, engines = _server()
        server.tools["chat"]["handler"] = MagicMock(side_effect=RuntimeError("boom"))
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {"prompt": "x"}})
        assert "error" not in resp
        assert resp["result"]["isError"] is True
        assert "boom" in resp["result"]["content"][0]["text"]

    def test_list_models_reads_registry(self):
        server, _ = _server()
        with patch("localm.config.load_registry",
                   return_value={"m1": {"path": "x", "source": "local"}}):
            resp = _req(server, "tools/call",
                        {"name": "list_models", "arguments": {}})
        assert "m1" in resp["result"]["content"][0]["text"]


class TestEngineCache:
    def test_same_model_reuses_engine(self):
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        assert cache.get(None) is cache.get(None)

    def test_model_switch_unloads_previous(self):
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        first = cache.get("a")
        second = cache.get("b")
        assert first is not second
        first.unload.assert_called_once()

    def test_no_default_falls_back_to_first_registered(self):
        cache = EngineCache(None, engine_factory=_stub_engine_factory)
        with patch("localm.config.load_registry",
                   return_value={"zeta": {}, "alpha": {}}):
            assert cache.resolve_model(None) == "alpha"

    def test_empty_registry_raises_helpfully(self):
        cache = EngineCache(None, engine_factory=_stub_engine_factory)
        with patch("localm.config.load_registry", return_value={}):
            with pytest.raises(ValueError, match="localm pull"):
                cache.resolve_model(None)


class TestStdioLoop:
    def test_full_round_trip(self):
        """Feed a complete client session through the stdio loop."""
        server, _ = _server()
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {}}),
            json.dumps({"jsonrpc": "2.0",
                        "method": "notifications/initialized"}),
            "this is not json",   # must be skipped, not crash
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "chat",
                                   "arguments": {"prompt": "hello"}}}),
        ]
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        server.run_stdio(stdin=stdin, stdout=stdout)

        responses = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert len(responses) == 2          # init + call; notification silent
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
        assert responses[1]["result"]["content"][0]["text"] == "reply-from-stub-model"

    def test_stdout_is_pure_jsonrpc(self):
        """Every stdout line must parse as JSON - no log contamination."""
        server, _ = _server()
        stdin = io.StringIO(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
        stdout = io.StringIO()
        server.run_stdio(stdin=stdin, stdout=stdout)
        for line in stdout.getvalue().splitlines():
            json.loads(line)   # raises if anything non-JSON leaked


class TestClientServerIntegration:
    def test_own_client_can_drive_own_server(self, tmp_path):
        """localcoder's MCP client talks to localm's MCP server in-process
        logic via subprocess - the two halves must interoperate."""
        import sys
        import textwrap
        bridge = tmp_path / "bridge.py"
        bridge.write_text(textwrap.dedent("""\
            from unittest.mock import MagicMock
            from localm.plugins.mcpserver.server import (
                EngineCache, MCPStdioServer, build_tools)

            def factory(name):
                e = MagicMock()
                e.chat_stream.side_effect = lambda m, **k: iter(["pong"])
                return e

            engines = EngineCache("stub", engine_factory=factory)
            MCPStdioServer(build_tools(engines, enable_images=False)).run_stdio()
        """), encoding="utf-8")

        from localm.plugins.coder.mcp import MCPServer
        client = MCPServer("self", sys.executable, [str(bridge)])
        try:
            client.start()
            assert {t["name"] for t in client.tools} == \
                {"chat", "list_models", "embed"}
            res = client.call_tool("chat", {"prompt": "ping"})
            assert res.ok
            assert res.output == "pong"
        finally:
            client.stop()


class TestMcpCliGate:
    """The MCP server became an optional plugin (Phase 3): `localm mcp` refuses to
    serve unless the mcp plugin is enabled, but --print-config always works."""

    @pytest.fixture
    def cfg_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", tmp_path)
        monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", tmp_path / "registry.json")
        return _cfg

    def test_refuses_when_not_installed(self, cfg_env):
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 1
        assert "not active" in result.output.lower()
        assert "localm plugin install mcp" in result.output

    def test_print_config_works_when_not_installed(self, cfg_env):
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, ["--print-config"])
        assert result.exit_code == 0
        assert "mcpServers" in result.output
        assert "localm plugin install mcp" in result.output

    def test_serves_when_installed(self, cfg_env, monkeypatch):
        from click.testing import CliRunner
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("mcp", True)   # copy store->installed + enable
        called = {}
        import localm.plugins.mcpserver.server as server
        monkeypatch.setattr(server, "serve_stdio",
                            lambda **k: called.setdefault("ok", True))
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 0
        assert called.get("ok") is True

    def test_installed_but_disabled_refuses(self, cfg_env):
        """Installed-but-disabled (the two-axis case) must NOT serve."""
        from click.testing import CliRunner
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        mgr.set_installed_state("mcp", True)
        mgr.set_enabled_state("mcp", False)                   # installed, disabled
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 1
        assert "not active" in result.output.lower()

    def test_mcp_plugin_available_not_installed_by_default(self, cfg_env):
        from localm.plugins.engine import PluginManager
        state = {p["name"]: p for p in PluginManager(None).api_state()["plugins"]}
        assert "mcp" in state                   # in the available catalog
        assert state["mcp"]["installed"] is False
        assert state["mcp"]["available"] is True
        assert state["mcp"]["scope"] == "mcp"

    def test_print_config_uses_os_correct_path(self, cfg_env, monkeypatch):
        """FAC-14: the Claude Desktop path must match the OS, not hardcode %APPDATA%.

        Drives all three OS branches by patching sys.platform (cli.py reads it at
        call time), so the test proves the fix regardless of the host platform -
        the previous version only asserted the host's own branch, which on
        Windows matched the pre-fix hardcoded %APPDATA% too (a weak guard)."""
        import sys
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        cases = {
            "win32": "%APPDATA%\\Claude\\claude_desktop_config.json",
            "darwin": "Library/Application Support/Claude/claude_desktop_config.json",
            "linux": "/.config/Claude/claude_desktop_config.json",
        }
        for plat, expected in cases.items():
            monkeypatch.setattr(sys, "platform", plat)
            out = CliRunner().invoke(main, ["--print-config"]).output
            assert expected in out, f"{plat}: expected {expected!r} in output"
        # the macOS path must NOT be the windows one (catches a hardcode regression)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert "%APPDATA%" not in CliRunner().invoke(main, ["--print-config"]).output


# --------------------------------------------------------------------------- #
#  Robustness: a malformed JSON-RPC payload must not crash the stdio loop      #
# --------------------------------------------------------------------------- #

class TestStdioRobustness:
    def test_batch_array_is_handled_not_crashed(self):
        """BUG-12: a JSON-RPC batch array must be processed element by element."""
        server, _ = _server()
        batch = json.dumps([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])
        stdout = io.StringIO()
        server.run_stdio(stdin=io.StringIO(batch + "\n"), stdout=stdout)
        ids = {json.loads(l)["id"] for l in stdout.getvalue().splitlines()}
        assert ids == {1, 2}

    def test_scalar_and_null_lines_do_not_crash(self):
        """BUG-12: a bare scalar / null parses but is not a request object."""
        server, _ = _server()
        stdout = io.StringIO()
        server.run_stdio(stdin=io.StringIO('123\n"hi"\ntrue\nnull\n'), stdout=stdout)
        responses = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert len(responses) == 4
        assert all(r["error"]["code"] == -32600 for r in responses)

    def test_handle_rejects_non_dict_directly(self):
        server, _ = _server()
        assert server.handle(123)["error"]["code"] == -32600
        assert server.handle([{"x": 1}])["error"]["code"] == -32600


# --------------------------------------------------------------------------- #
#  generate_image safety: stdout purity, privacy sidecar, path confinement     #
# --------------------------------------------------------------------------- #

class TestGenerateImageSafety:
    def _call(self, server, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "generate_image", "arguments": args}})

    def test_output_path_confined_to_home(self, tmp_path):
        """SEC-7: an output_path outside the localm data dir is refused."""
        server, _ = _server()
        outside = str(tmp_path / "evil.png")        # sibling of LOCALM_HOME (.localm)
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "output_path": outside})
        assert r["result"]["isError"] is True
        assert "data dir" in r["result"]["content"][0]["text"]
        mock_gen.assert_not_called()

    def test_input_image_confined_to_home(self, tmp_path):
        server, _ = _server()
        outside = str(tmp_path / "secret.png")
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "input_image": outside})
        assert r["result"]["isError"] is True
        mock_gen.assert_not_called()

    def test_privacy_mode_suppresses_sidecar(self, monkeypatch):
        """BUG-14: privacy mode must pass write_sidecar=False to generate_image."""
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            self._call(server, {"prompt": "x"})
        assert mock_gen.call_args.kwargs.get("write_sidecar") is False

    def test_logmode_keeps_sidecar(self, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            self._call(server, {"prompt": "x"})
        assert mock_gen.call_args.kwargs.get("write_sidecar") is True

    def test_generate_image_keeps_stdout_clean(self, capsys):
        """BUG-11: comfy progress output must go to stderr, not the JSON-RPC stdout."""
        server, _ = _server()

        def noisy(*a, **k):
            print("PROGRESS_NOISE_ON_STDOUT")
            return (True, "ok")

        with patch("localm.image_gen.comfy.generate_image", noisy):
            self._call(server, {"prompt": "x"})
        captured = capsys.readouterr()
        assert "PROGRESS_NOISE_ON_STDOUT" not in captured.out
        assert "PROGRESS_NOISE_ON_STDOUT" in captured.err
