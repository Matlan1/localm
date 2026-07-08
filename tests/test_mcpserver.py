# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the localm MCP server plugin (plugins/mcpserver)."""

import io
import json
import sys
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
        expected = {
            "chat", "list_models", "system_stats", "search_models",
            "list_model_files", "pull_model", "embed", "generate_image",
            "setup_embeddings", "remove_model", "run_doctor", "list_plugins",
            "install_plugin", "enable_plugin", "disable_plugin", "uninstall_plugin"
        }
        assert names == expected

    def test_no_images_flag_hides_tool(self):
        server, _ = _server(enable_images=False)
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "generate_image" not in names


class TestToolAnnotations:
    """MCP tool annotations (audit finding C). Confirmation for destructive
    tools belongs at the CLIENT, so the server must DECLARE intent via the
    standard MCP annotations (destructiveHint / readOnlyHint) in tools/list.
    Without this, an MCP client has no signal that remove_model / uninstall_plugin
    delete files."""

    def _annotations_by_name(self, server):
        return {t["name"]: t.get("annotations")
                for t in _req(server, "tools/list")["result"]["tools"]}

    def test_destructive_tools_declare_destructive_hint(self):
        """The primary oracle: the two file-deleting tools carry
        annotations.destructiveHint == true with a human-readable title. Fails
        on pre-fix master, where tools/list emits no annotations at all."""
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("remove_model", "uninstall_plugin"):
            assert ann[name] is not None, f"{name} advertises no annotations"
            assert ann[name]["destructiveHint"] is True, f"{name} not destructiveHint"
            assert ann[name].get("title"), f"{name} has no human title"

    def test_read_only_tools_declare_read_only_hint(self):
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("list_models", "list_plugins", "list_model_files",
                     "search_models", "system_stats", "run_doctor"):
            assert ann[name] is not None, f"{name} advertises no annotations"
            assert ann[name]["readOnlyHint"] is True, f"{name} not readOnlyHint"

    def test_destructive_tools_are_not_marked_read_only(self):
        """Negative guard: a destructive tool must never also claim readOnlyHint
        (readOnlyHint true would tell the client destructiveHint is meaningless)."""
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("remove_model", "uninstall_plugin"):
            assert ann[name].get("readOnlyHint") is not True

    def test_plain_tools_have_no_annotations_key_leaked(self):
        """A tool with no annotations (e.g. chat) must not carry an empty/None
        annotations field - the key is present only when there is a hint."""
        server, _ = _server()
        entries = {t["name"]: t for t in _req(server, "tools/list")["result"]["tools"]}
        assert "annotations" not in entries["chat"]


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

    def test_model_switch_waits_for_vram_release(self):
        """A model switch must poll for the previous model's VRAM to actually
        free before loading the next one (TDR-hang guard, shared with the
        /v1/models/unload endpoint's own use of wait_for_vram_release)."""
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.vram_info", return_value={"free": 1_000_000_000}), \
             patch("localm.vram.wait_for_vram_release",
                   return_value=(True, 2_000_000_000)) as mock_wait:
            cache.get("b")
        mock_wait.assert_called_once()
        assert mock_wait.call_args.kwargs["before_bytes"] == 1_000_000_000

    def test_vram_unmeasurable_does_not_block_switch(self):
        """When VRAM cannot be read at all (no GPU/torch), the wait is a no-op -
        matches wait_for_vram_release's own before_bytes=None contract."""
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.vram_info", return_value={}):
            second = cache.get("b")   # must not hang/raise
        assert second is not None

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
            expected = {
                "chat", "list_models", "system_stats", "search_models",
                "list_model_files", "pull_model", "embed",
                "setup_embeddings", "remove_model", "run_doctor", "list_plugins",
                "install_plugin", "enable_plugin", "disable_plugin", "uninstall_plugin"
            }
            assert {t["name"] for t in client.tools} == expected
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


class TestModelDiscoveryTools:
    """system_stats / search_models / list_model_files / pull_model - always
    advertised (no plugin gate, unlike run_coder_task) since they only need
    core localm functionality."""

    def _call(self, server, name, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args}})

    def test_system_stats_returns_live_stats(self):
        server, _ = _server()
        stats = {"cpu": {"percent": 12.0}, "ram": {"used": 1, "total": 2, "percent": 50.0},
                 "vram": {"used": 1, "total": 8, "percent": 12.5}}
        with patch("localm.sysstats.system_stats", return_value=stats):
            r = self._call(server, "system_stats", {})
        assert json.loads(r["result"]["content"][0]["text"]) == stats

    def test_search_models_wraps_hf_search(self):
        server, _ = _server()
        results = [{"id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "downloads": 999,
                   "likes": 10, "updated": "2026-01-01"}]
        with patch("localm.discover.hf_search", return_value=results) as mock_search:
            r = self._call(server, "search_models", {"query": "qwen", "limit": 5})
        assert json.loads(r["result"]["content"][0]["text"]) == results
        mock_search.assert_called_once_with("qwen", limit=5)

    def test_search_models_surfaces_discover_error(self):
        server, _ = _server()
        from localm.discover import DiscoverError
        with patch("localm.discover.hf_search", side_effect=DiscoverError("offline")):
            r = self._call(server, "search_models", {})
        assert r["result"]["isError"] is True
        assert "offline" in r["result"]["content"][0]["text"]

    def test_list_model_files_missing_repo_is_error(self):
        server, _ = _server()
        r = self._call(server, "list_model_files", {})
        assert r["result"]["isError"] is True
        assert "repo" in r["result"]["content"][0]["text"]

    def test_list_model_files_includes_fit_label(self):
        server, _ = _server()
        files = [{"file": "model.Q4_K_M.gguf", "quant": "Q4_K_M",
                 "size_bytes": 4_000_000_000, "n_parts": 1}]
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.vram_info", return_value={"total": 8_000_000_000}), \
             patch("localm.discover.fit_label", return_value="fits") as mock_fit:
            r = self._call(server, "list_model_files", {"repo": "owner/repo"})
        out = json.loads(r["result"]["content"][0]["text"])
        assert out[0]["fit"] == "fits"
        mock_fit.assert_called_once_with(4_000_000_000, 8_000_000_000)

    def test_list_model_files_surfaces_discover_error(self):
        server, _ = _server()
        from localm.discover import DiscoverError
        with patch("localm.discover.hf_gguf_files", side_effect=DiscoverError("not a repo")):
            r = self._call(server, "list_model_files", {"repo": "nope"})
        assert r["result"]["isError"] is True
        assert "not a repo" in r["result"]["content"][0]["text"]

    def test_pull_model_missing_repo_is_error(self):
        server, _ = _server()
        r = self._call(server, "pull_model", {"name": "m"})
        assert r["result"]["isError"] is True
        assert "repo" in r["result"]["content"][0]["text"]

    def test_pull_model_missing_name_is_error(self):
        server, _ = _server()
        r = self._call(server, "pull_model", {"repo": "owner/repo"})
        assert r["result"]["isError"] is True
        assert "name" in r["result"]["content"][0]["text"]

    def test_pull_model_success_loads_by_default(self):
        server, engines = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=True) as mock_pull:
            r = self._call(server, "pull_model",
                           {"repo": "owner/repo", "file": "m.Q4_K_M.gguf", "name": "m"})
        assert r["result"]["isError"] is False
        assert "loaded" in r["result"]["content"][0]["text"]
        mock_pull.assert_called_once_with("owner/repo:m.Q4_K_M.gguf", name="m")
        assert engines._loaded_name == "m"

    def test_pull_model_load_false_skips_loading(self):
        # _server() eagerly resolves the default engine once already (the embed-
        # capability probe in build_tools()), so "never loaded" is asserted as
        # "the newly pulled name never became active", not "nothing is loaded".
        server, engines = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=True):
            r = self._call(server, "pull_model",
                           {"repo": "owner/repo", "name": "m", "load": False})
        assert r["result"]["isError"] is False
        assert "not loaded" in r["result"]["content"][0]["text"]
        assert engines._loaded_name != "m"

    def test_pull_model_failure_is_surfaced(self):
        server, _ = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=False):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "pull failed" in r["result"]["content"][0]["text"]

    def test_pull_model_exception_is_surfaced(self):
        server, _ = _server()
        with patch("localm.model_manager.pull.pull_model",
                   side_effect=RuntimeError("network down")):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "network down" in r["result"]["content"][0]["text"]

    def test_pull_model_load_failure_is_distinguished_from_pull_failure(self):
        engines = EngineCache(default_model=None, engine_factory=_stub_engine_factory)
        server = MCPStdioServer(build_tools(engines))
        with patch("localm.model_manager.pull.pull_model", return_value=True), \
             patch.object(engines, "get", side_effect=RuntimeError("no GPU memory")):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "loading it failed" in r["result"]["content"][0]["text"]
        assert "no GPU memory" in r["result"]["content"][0]["text"]


class TestRunCoderTask:
    """run_coder_task shells out to `localm coder --output-format json` and is
    only advertised when the coder plugin is installed+enabled (mirrors the
    embed tool's capability-gated advertisement)."""

    @pytest.fixture
    def coder_active(self):
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("coder", True)

    def _call(self, server, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "run_coder_task", "arguments": args}})

    def test_hidden_when_coder_not_active(self):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" not in names

    def test_listed_when_coder_active(self, coder_active):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" in names

    def test_no_coder_flag_hides_tool_even_when_active(self, coder_active):
        engines = EngineCache(default_model="stub", engine_factory=_stub_engine_factory)
        server = MCPStdioServer(build_tools(engines, enable_coder=False))
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" not in names

    def test_missing_task_is_error(self, coder_active, tmp_path):
        server, _ = _server()
        r = self._call(server, {"cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "task" in r["result"]["content"][0]["text"]

    def test_missing_cwd_is_error(self, coder_active):
        server, _ = _server()
        r = self._call(server, {"task": "do a thing"})
        assert r["result"]["isError"] is True
        assert "cwd" in r["result"]["content"][0]["text"]

    def test_nonexistent_cwd_is_error(self, coder_active, tmp_path):
        server, _ = _server()
        r = self._call(server, {"task": "x", "cwd": str(tmp_path / "nope")})
        assert r["result"]["isError"] is True
        assert "directory" in r["result"]["content"][0]["text"]

    def test_successful_run_parses_json_payload(self, coder_active, tmp_path):
        # Real `--output-format json` pretty-prints (indent=2, multi-line) - a
        # single-line json.dumps() here would silently hide the exact bug this
        # parsing once had (see test_console_messages_before_json_are_ignored).
        server, _ = _server()
        payload = {"success": True, "response": "done: added type hints",
                   "turns": 3, "total_tokens": 512}
        fake = MagicMock(stdout=json.dumps(payload, indent=2) + "\n", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake) as mock_run:
            r = self._call(server, {"task": "add type hints", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        assert "done: added type hints" in r["result"]["content"][0]["text"]
        assert "turns=3" in r["result"]["content"][0]["text"]
        cmd = mock_run.call_args.args[0]
        assert cmd[:4] == [sys.executable, "-m", "localm", "coder"]
        assert "add type hints" in cmd
        assert "--cwd" in cmd and str(tmp_path) in cmd
        assert "--output-format" in cmd and "json" in cmd
        assert "--yes" not in cmd            # default off (R19a fail-closed)
        # The subprocess's OWN OS cwd must also be the task dir, not just the
        # --cwd flag: if the coder auto-spawns a background server (no instance
        # running yet), it identifies "this project" by ITS OWN inherited
        # working directory. Passing --cwd alone registers the auto-spawned
        # server under the WRONG project root, so attach-back never finds it.
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)

    def test_agent_reported_failure_is_surfaced_as_error(self, coder_active, tmp_path):
        server, _ = _server()
        payload = {"success": False, "response": "hit max turns", "turns": 40,
                   "total_tokens": 9001}
        fake = MagicMock(stdout=json.dumps(payload, indent=2) + "\n", stderr="", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "hit max turns" in r["result"]["content"][0]["text"]

    def test_yes_model_max_turns_thread_through_to_cmd(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout=json.dumps({"success": True, "response": "ok"}, indent=2) + "\n",
                         stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake) as mock_run:
            self._call(server, {"task": "x", "cwd": str(tmp_path), "model": "qwen2.5-7b",
                                "max_turns": 10, "yes": True})
        cmd = mock_run.call_args.args[0]
        assert "--model" in cmd and "qwen2.5-7b" in cmd
        assert "--max-turns" in cmd and "10" in cmd
        assert "--yes" in cmd

    def test_console_messages_before_json_are_ignored(self, coder_active, tmp_path):
        """Regression guard: a live run against a real model produced exactly
        this shape - console.print() messages (e.g. attaching to a running
        server) print to stdout BEFORE the final --output-format json dump.
        Parsing must find the JSON, not mistake the console text or the bare
        closing brace for the payload."""
        server, _ = _server()
        payload = {"success": True, "response": "created hello.txt", "turns": 2,
                   "total_tokens": 123}
        stdout = (
            "Using the localm already running for /some/project "
            "(port 8642, mode api) - sharing its loaded model.\n"
            + json.dumps(payload, indent=2) + "\n"
        )
        fake = MagicMock(stdout=stdout, stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        assert "created hello.txt" in r["result"]["content"][0]["text"]

    def test_no_json_object_in_stdout_falls_back_cleanly(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout="some console message, no JSON at all\n",
                         stderr="", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "some console message" in r["result"]["content"][0]["text"]

    def test_timeout_is_reported_not_raised(self, coder_active, tmp_path):
        import subprocess as _sp
        server, _ = _server()
        with patch("localm.plugins.mcpserver.server.subprocess.run",
                   side_effect=_sp.TimeoutExpired(cmd="x", timeout=5)):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path), "timeout_seconds": 5})
        assert r["result"]["isError"] is True
        assert "timed out" in r["result"]["content"][0]["text"]

    def test_non_json_output_falls_back_to_stderr_detail(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout="", stderr="coder plugin is not active\n", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "coder plugin is not active" in r["result"]["content"][0]["text"]


class TestMcpCliWiring:
    """Regression guard: `localm mcp` must resolve to the real plugin command
    (plugins/mcpserver/cli.py), not a broken hand-rolled duplicate. A duplicate
    in localm/cli/maintenance.py once shadowed it and made `localm mcp` crash
    with an ImportError before it could even print its 'not enabled' message.
    """

    def test_mcp_command_is_the_plugin_command(self):
        from localm.cli import main
        cmd = main.commands.get("mcp")
        assert cmd is not None, "`localm mcp` is not registered"
        # The real command lives in the mcpserver plugin, not in cli.maintenance.
        assert cmd.callback.__module__ == "localm.plugins.mcpserver.cli"

    def test_print_config_works_and_needs_no_active_plugin(self):
        # --print-config is exempt from the active-plugin check, so it works on a
        # fresh install and proves the real (option-bearing) command is wired.
        from click.testing import CliRunner

        from localm.cli import main
        r = CliRunner().invoke(main, ["mcp", "--print-config"])
        assert r.exit_code == 0, r.output
        assert "mcpServers" in r.output and "localm" in r.output
        # the broken duplicate had no such option and never touched winconsole
        assert "winconsole" not in r.output
        assert "unexpected error" not in r.output.lower()


class TestNewToolCalls:
    def test_setup_embeddings(self):
        server, _ = _server()
        with patch("localm.inference.embedder.resolve_embedding_model_path", return_value="fake-path") as mock_resolve:
            resp = _req(server, "tools/call",
                        {"name": "setup_embeddings", "arguments": {"model": "fake-model"}})
        assert resp["result"]["isError"] is False
        assert "ready" in resp["result"]["content"][0]["text"]
        mock_resolve.assert_called_once()

    def test_remove_model(self):
        server, _ = _server()
        with patch("localm.config.load_registry", return_value={"to-remove": {"path": "x"}}), \
             patch("localm.model_manager.remove_model") as mock_remove:
            resp = _req(server, "tools/call",
                        {"name": "remove_model", "arguments": {"model": "to-remove"}})
        assert resp["result"]["isError"] is False
        assert "removed" in resp["result"]["content"][0]["text"]
        mock_remove.assert_called_once_with("to-remove")

    def test_run_doctor(self):
        server, _ = _server()
        fake_proc = MagicMock(stdout="doctor-ok", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake_proc) as mock_run:
            resp = _req(server, "tools/call", {"name": "run_doctor", "arguments": {}})
        assert resp["result"]["isError"] is False
        assert "doctor-ok" in resp["result"]["content"][0]["text"]
        mock_run.assert_called_once()

    def test_list_plugins(self):
        server, _ = _server()
        fake_state = {"plugins": [{"name": "fake", "description": "d", "installed": True, "active": True}]}
        with patch("localm.plugins.engine.PluginManager.api_state", return_value=fake_state):
            resp = _req(server, "tools/call", {"name": "list_plugins", "arguments": {}})
        assert resp["result"]["isError"] is False
        assert "fake" in resp["result"]["content"][0]["text"]
        assert "enabled" in resp["result"]["content"][0]["text"]

    def test_install_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.set_installed_state") as mock_install, \
             patch("localm.plugins.engine.PluginManager.plugin_missing_deps", return_value=False):
            resp = _req(server, "tools/call",
                        {"name": "install_plugin", "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is False
        assert "installed" in resp["result"]["content"][0]["text"]
        mock_install.assert_called_once_with("coder", True)

    def test_enable_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.set_enabled_state") as mock_enable:
            resp = _req(server, "tools/call",
                        {"name": "enable_plugin", "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is False
        assert "enabled" in resp["result"]["content"][0]["text"]
        mock_enable.assert_called_once_with("coder", True)

    def test_disable_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.set_enabled_state") as mock_disable:
            resp = _req(server, "tools/call",
                        {"name": "disable_plugin", "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is False
        assert "disabled" in resp["result"]["content"][0]["text"]
        mock_disable.assert_called_once_with("coder", False)

    def test_uninstall_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.uninstall") as mock_uninstall:
            resp = _req(server, "tools/call",
                        {"name": "uninstall_plugin", "arguments": {"plugin": "coder", "delete_data": True}})
        assert resp["result"]["isError"] is False
        assert "uninstalled" in resp["result"]["content"][0]["text"]
        mock_uninstall.assert_called_once_with("coder", delete_data=True)
