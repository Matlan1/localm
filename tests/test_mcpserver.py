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
        import subprocess, sys, textwrap
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
