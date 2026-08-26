# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server / embeddings family. Three properties:

  - /v1/embeddings honors encoding_format="base64" (returning base64
    little-endian float32 per the OpenAI spec) or rejects an unknown format
    with 400, never silently returning float arrays.

  - the MCP 'embed' tool is not advertised when the active backend cannot embed
    (the common GGUF backend raises NotImplementedError), and calling it on
    such a backend reports unsupported rather than crashing.

  - PATCH /v1/config refuses to set require_auth=true while no API key is
    configured (a one-way self-lockout), with a clear 400; once a key exists
    the same PATCH succeeds.
"""

import base64
import os
import struct
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


SECRET = "phase3-secret-key-xyz789"


# --------------------------------------------------------------------------- #
#  Engine doubles                                                              #
# --------------------------------------------------------------------------- #

def _make_engine(embed_value=None):
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 3
    if embed_value is not None:
        engine.embed.return_value = embed_value
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])
    return engine


# --------------------------------------------------------------------------- #
#  encoding_format=base64                                                      #
# --------------------------------------------------------------------------- #

class TestEmbeddingsEncodingFormat:
    VEC = [0.1, -0.5, 1.25, 0.0]

    def _client(self):
        os.environ.pop("LOCALM_API_KEY", None)
        engine = _make_engine(embed_value=[self.VEC])
        return TestClient(create_app(engine), raise_server_exceptions=True)

    def test_float_default_still_returns_list(self):
        with self._client() as c:
            r = c.post("/v1/embeddings",
                       json={"model": "m", "input": "hi"})
        assert r.status_code == 200
        emb = r.json()["data"][0]["embedding"]
        assert isinstance(emb, list)
        assert emb == pytest.approx(self.VEC)

    def test_base64_returns_decodable_float32_string(self):
        with self._client() as c:
            r = c.post("/v1/embeddings",
                       json={"model": "m", "input": "hi",
                             "encoding_format": "base64"})
        assert r.status_code == 200
        emb = r.json()["data"][0]["embedding"]
        # Per the OpenAI spec, base64 mode returns a string, not a list.
        assert isinstance(emb, str)
        raw = base64.b64decode(emb)
        # little-endian float32 -> 4 bytes per element
        assert len(raw) == 4 * len(self.VEC)
        decoded = list(struct.unpack("<%df" % len(self.VEC), raw))
        assert decoded == pytest.approx(self.VEC, abs=1e-6)

    def test_unknown_encoding_format_rejected_with_400(self):
        with self._client() as c:
            r = c.post("/v1/embeddings",
                       json={"model": "m", "input": "hi",
                             "encoding_format": "protobuf"})
        assert r.status_code == 400
        assert "encoding_format" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  MCP embed tool degrades on a non-embedding backend                          #
# --------------------------------------------------------------------------- #

def _mcp(default_model="stub", *, can_embed):
    """Build an MCP server whose default engine's backend reports *can_embed*."""
    from localm.plugins.mcpserver.server import (
        EngineCache, MCPStdioServer, build_tools)

    def factory(model_name):
        engine = MagicMock()
        engine.display_name = model_name
        # The capability probe inspects the backend's can_embed flag.
        engine._backend.can_embed = can_embed
        if can_embed:
            engine.embed.return_value = [[0.1, 0.2]]
        else:
            engine.embed.side_effect = NotImplementedError(
                "This backend does not support embedding.")
        return engine

    engines = EngineCache(default_model=default_model, engine_factory=factory)
    server = MCPStdioServer(build_tools(engines, enable_images=False))
    return server, engines


def _tool_names(server):
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    return {t["name"] for t in resp["result"]["tools"]}


class TestMcpEmbedToolCapability:
    def test_embed_listed_when_backend_can_embed(self):
        server, _ = _mcp(can_embed=True)
        assert "embed" in _tool_names(server)

    def test_embed_hidden_when_backend_cannot_embed(self):
        server, _ = _mcp(can_embed=False)
        assert "embed" not in _tool_names(server)
        # The other tools are unaffected.
        assert {"chat", "list_models"} <= _tool_names(server)

    def test_embed_call_when_hidden_returns_graceful_error(self):
        # When embed is hidden, invoking it directly must produce a clean
        # JSON-RPC "unknown tool" error rather than crashing the server.
        server, _ = _mcp(can_embed=False)
        resp = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                              "params": {"name": "embed",
                                         "arguments": {"texts": ["a"]}}})
        assert "result" not in resp
        assert resp["error"]["code"] == -32602
        assert "embed" in resp["error"]["message"].lower()

    def test_embed_handler_degrades_when_backend_raises(self):
        # If embed IS advertised but the backend nonetheless raises at call time
        # (e.g. a model that only reveals it once loaded), the handler reports
        # the error rather than crashing. The tool is registered by claiming
        # can_embed=True, then the engine raises NotImplementedError.
        from localm.plugins.mcpserver.server import (
            EngineCache, MCPStdioServer, build_tools)

        def factory(model_name):
            engine = MagicMock()
            engine.display_name = model_name
            engine._backend.can_embed = True
            engine.embed.side_effect = NotImplementedError(
                "This backend does not support embedding.")
            return engine

        engines = EngineCache(default_model="stub", engine_factory=factory)
        server = MCPStdioServer(build_tools(engines, enable_images=False))
        resp = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                              "params": {"name": "embed",
                                         "arguments": {"texts": ["a"]}}})
        result = resp["result"]
        assert result["isError"] is True
        assert "support" in result["content"][0]["text"].lower()


# --------------------------------------------------------------------------- #
#  PATCH /v1/config require_auth self-lockout guard                            #
# --------------------------------------------------------------------------- #

class TestRequireAuthLockoutGuard:
    @pytest.fixture(autouse=True)
    def _isolated_config_file(self, tmp_path, monkeypatch):
        # save_config/load_config/update_config all read the FROZEN CONFIG_FILE
        # and HOME_DIR module attributes, not the lazy home_dir() the autouse
        # per-test LOCALM_HOME fixture affects, so repoint them explicitly.
        import localm.config as cfg
        home = tmp_path / ".localm"
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    def _client(self):
        # PATCH /v1/config is a management route; in open mode it needs the
        # loopback shell token. Seed it by default (tests that set their own key
        # override the header and hit protected-mode bearer auth instead).
        app = create_app(_make_engine())
        return TestClient(
            app, raise_server_exceptions=True,
            headers={"Authorization": f"Bearer {app.state.shell_token}"})

    # patch_config() persists via config.update_config(), the atomic
    # read-modify-write helper, so these tests verify against the REAL persisted
    # config; the autouse per-test LOCALM_HOME isolates the file.

    def test_enable_require_auth_without_key_rejected_400(self):
        os.environ.pop("LOCALM_API_KEY", None)
        from localm.config import load_config, save_config
        save_config({"require_auth": False})
        with patch("localm.auth.any_key_configured", return_value=False):
            with self._client() as c:
                r = c.patch("/v1/config", json={"require_auth": True})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "key" in detail.lower()
        # Crucially, the lockout config was never persisted.
        assert load_config()["require_auth"] is False

    def test_enable_require_auth_with_key_succeeds_200(self):
        os.environ.pop("LOCALM_API_KEY", None)
        from localm.config import load_config, save_config
        save_config({"require_auth": False})
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            # An owner key is now configured (env), so the guard allows it and
            # the owner key authorises the CONFIG_WRITE scope.
            with self._client() as c:
                r = c.patch("/v1/config", json={"require_auth": True},
                            headers={"Authorization": f"Bearer {SECRET}"})
        assert r.status_code == 200
        assert r.json().get("require_auth") is True
        assert load_config()["require_auth"] is True

    def test_setting_require_auth_false_without_key_not_blocked(self):
        # The guard only blocks ENABLING the lockout. Writing require_auth=False
        # (or any unrelated key) while keyless is never blocked.
        os.environ.pop("LOCALM_API_KEY", None)
        from localm.config import load_config, save_config
        save_config({"require_auth": False})
        with patch("localm.auth.any_key_configured", return_value=False):
            with self._client() as c:
                r = c.patch("/v1/config", json={"require_auth": False})
        assert r.status_code == 200
        assert load_config()["require_auth"] is False

    def test_unrelated_patch_without_key_not_blocked(self):
        # A PATCH that does not touch require_auth is unaffected by the guard.
        os.environ.pop("LOCALM_API_KEY", None)
        from localm.config import load_config, save_config
        save_config({"require_auth": False, "max_tokens": 1024})
        with patch("localm.auth.any_key_configured", return_value=False):
            with self._client() as c:
                r = c.patch("/v1/config", json={"max_tokens": 2048})
        assert r.status_code == 200
        assert load_config()["max_tokens"] == 2048
