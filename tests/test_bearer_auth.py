# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for bearer-token authentication in localm.inference.http_server.

Covers:
  - Open mode (no LOCALM_API_KEY set) - all requests allowed
  - Protected mode (LOCALM_API_KEY set) - wrong/missing token → 401, correct → 200
  - /health stays open in both modes; /v1/models requires models:read once a key is set

Key invariant: _require_auth reads os.environ at REQUEST time, so env vars
must be active when the request is processed (not just at app-creation time).
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

SECRET = "test-secret-key-abc123"

CHAT_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


def _make_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5

    _state = {"loaded": True}

    def _unload():
        _state["loaded"] = False

    def _load():
        _state["loaded"] = True

    def _chat_stream(messages, **kwargs):
        yield "ok"

    engine.unload.side_effect = _unload
    engine.load.side_effect = _load
    engine.chat_stream.side_effect = _chat_stream
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


@pytest.fixture()
def client():
    """TestClient with no API key set (open mode). Context manager triggers ASGI lifespan."""
    os.environ.pop("LOCALM_API_KEY", None)
    with TestClient(create_app(_make_engine()), raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def protected_client():
    """TestClient for protected-mode tests. Lifespan triggered; env var patched per-request."""
    os.environ.pop("LOCALM_API_KEY", None)
    with TestClient(create_app(_make_engine()), raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
#  Open mode (no API key configured)
# ---------------------------------------------------------------------------

class TestOpenMode:
    def test_chat_completions_allowed_without_auth(self, client):
        os.environ.pop("LOCALM_API_KEY", None)
        r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert r.status_code == 200

    def test_model_unload_with_shell_token(self, client):
        # Model load/unload are MODELS_WRITE management routes; in open mode they
        # require the loopback shell token (the GUI shell injects it).
        os.environ.pop("LOCALM_API_KEY", None)
        token = client.app.state.shell_token
        r = client.post("/v1/models/unload",
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_model_unload_without_token_refused(self, client):
        os.environ.pop("LOCALM_API_KEY", None)
        assert client.post("/v1/models/unload").status_code == 403

    def test_model_load_with_shell_token(self, client):
        os.environ.pop("LOCALM_API_KEY", None)
        token = client.app.state.shell_token
        hdr = {"Authorization": f"Bearer {token}"}
        client.post("/v1/models/unload", headers=hdr)  # unload first
        r = client.post("/v1/models/load", headers=hdr)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
#  Protected mode (API key configured)
# ---------------------------------------------------------------------------

class TestProtectedMode:
    def test_correct_key_allows_chat(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/chat/completions",
                json=CHAT_PAYLOAD,
                headers={"Authorization": f"Bearer {SECRET}"},
            )
        assert r.status_code == 200

    def test_wrong_key_returns_401_on_chat(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/chat/completions",
                json=CHAT_PAYLOAD,
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert r.status_code == 401

    def test_missing_auth_header_returns_401_on_chat(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert r.status_code == 401

    def test_wrong_key_returns_401_on_unload(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/models/unload",
                headers={"Authorization": "Bearer bad"},
            )
        assert r.status_code == 401

    def test_correct_key_allows_unload(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/models/unload",
                headers={"Authorization": f"Bearer {SECRET}"},
            )
        assert r.status_code == 200

    def test_wrong_key_returns_401_on_load(self, protected_client):
        protected_client.post(
            "/v1/models/unload",
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/models/load",
                headers={"Authorization": "Bearer wrong"},
            )
        assert r.status_code == 401

    def test_correct_key_allows_load(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            protected_client.post(
                "/v1/models/unload",
                headers={"Authorization": f"Bearer {SECRET}"},
            )
            r = protected_client.post(
                "/v1/models/load",
                headers={"Authorization": f"Bearer {SECRET}"},
            )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
#  /health stays open with a key set; /v1/models is gated by models:read
# ---------------------------------------------------------------------------

class TestUnprotectedEndpoints:
    def test_health_accessible_without_key(self, protected_client):
        """/health is an intentionally open liveness probe."""
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.get("/health")
        assert r.status_code == 200

    def test_models_list_requires_key_when_configured(self, protected_client):
        """With a key set, /v1/models requires it (models:read); the owner key
        (implicit ADMIN) is accepted. It is no longer open like /health."""
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            denied = protected_client.get("/v1/models")
            assert denied.status_code == 401
            ok = protected_client.get(
                "/v1/models", headers={"Authorization": f"Bearer {SECRET}"})
            assert ok.status_code == 200


# ---------------------------------------------------------------------------
#  /v1/completions endpoint
# ---------------------------------------------------------------------------

COMPLETION_PAYLOAD = {"model": "test-model", "prompt": "Hello"}


class TestCompletionsEndpoint:
    def test_non_streaming_returns_text(self, client):
        os.environ.pop("LOCALM_API_KEY", None)
        r = client.post("/v1/completions", json=COMPLETION_PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert data["choices"][0]["text"] == "ok"

    def test_returns_usage_info(self, client):
        os.environ.pop("LOCALM_API_KEY", None)
        r = client.post("/v1/completions", json=COMPLETION_PAYLOAD)
        assert r.status_code == 200
        usage = r.json().get("usage", {})
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_requires_auth_when_key_set(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post("/v1/completions", json=COMPLETION_PAYLOAD)
        assert r.status_code == 401

    def test_accepts_correct_key(self, protected_client):
        with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
            r = protected_client.post(
                "/v1/completions",
                json=COMPLETION_PAYLOAD,
                headers={"Authorization": f"Bearer {SECRET}"},
            )
        assert r.status_code == 200

    def test_seed_param_accepted(self, client):
        """seed field is accepted without error."""
        os.environ.pop("LOCALM_API_KEY", None)
        r = client.post("/v1/completions", json={**COMPLETION_PAYLOAD, "seed": 42})
        assert r.status_code == 200


class TestSeedParam:
    def test_seed_in_chat_completions(self, client):
        """seed field is accepted in /v1/chat/completions without error."""
        os.environ.pop("LOCALM_API_KEY", None)
        r = client.post(
            "/v1/chat/completions",
            json={**CHAT_PAYLOAD, "seed": 123},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
#  /v1/embeddings endpoint
# ---------------------------------------------------------------------------

class TestEmbeddingsEndpoint:
    def _client_with_embed(self):
        """Client whose engine supports embed()."""
        os.environ.pop("LOCALM_API_KEY", None)
        engine = _make_engine()
        engine.embed.return_value = [[0.1, 0.2, 0.3]]
        app = create_app(engine)
        return TestClient(app, raise_server_exceptions=True)

    def test_single_string_input(self):
        with self._client_with_embed() as client:
            r = client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": "hello world"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_list_input(self):
        engine = _make_engine()
        engine.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        app = create_app(engine)
        os.environ.pop("LOCALM_API_KEY", None)
        with TestClient(app, raise_server_exceptions=True) as client:
            r = client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": ["text a", "text b"]},
            )
        assert r.status_code == 200
        assert len(r.json()["data"]) == 2

    def test_not_implemented_returns_422(self):
        engine = _make_engine()
        engine.embed.side_effect = NotImplementedError("no embeddings")
        app = create_app(engine)
        os.environ.pop("LOCALM_API_KEY", None)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                "/v1/embeddings",
                json={"model": "test-model", "input": "hi"},
            )
        assert r.status_code == 422

    def test_requires_auth_when_key_set(self):
        engine = _make_engine()
        engine.embed.return_value = [[0.0]]
        app = create_app(engine)
        with TestClient(app, raise_server_exceptions=True) as client:
            with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
                r = client.post(
                    "/v1/embeddings",
                    json={"model": "test-model", "input": "hi"},
                )
        assert r.status_code == 401
