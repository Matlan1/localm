# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/embeddings must route to the dedicated embedder for the model named by the
``embedding_model`` config key, not only for a registry entry whose NAME happens
to match that key exactly.

`localm setup-embeddings` registers a freshly downloaded known-key model (the
default "bge-small-en-v1.5") under a PREFIXED alias
("embedding-bge-small-en-v1.5", cli/maintenance.py), while `embedding_model` in
config stays the raw known-key string. `_make_self_embed` (rag/plug.py) sends that
raw config value as `model`. A registry-name-only routing check therefore never
matches the default setup flow and falls through to get_engine(), which 404s ("not
registered") since "bge-small-en-v1.5" is not itself a registry key, breaking
indexing/query even when a chat model IS loaded."""

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs


@pytest.fixture
def app_client(monkeypatch):
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    app = hs.create_app(None)
    return TestClient(app)


def test_default_embedding_model_routes_despite_prefixed_registry_alias(app_client, monkeypatch):
    """embedding_model="bge-small-en-v1.5", registered only as
    "embedding-bge-small-en-v1.5" (what setup-embeddings actually does) - a request
    naming the raw config value must still route to the dedicated embedder, not
    404 through get_engine()."""
    registry = {
        "embedding-bge-small-en-v1.5": {
            "path": "models/embeddings/bge-small-en-v1.5-q4_k_m.gguf",
            "source": "setup-embeddings",
            "model_type": "embedding",
        },
        "some-chat-model": {"path": "models/some-chat-model.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})

    def fake_embed_texts(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr("localm.inference.embedder.embed_texts", fake_embed_texts)

    r = app_client.post("/v1/embeddings",
                        json={"model": "bge-small-en-v1.5", "input": "hello world"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_default_embedding_model_routes_even_with_a_chat_model_active(app_client, monkeypatch):
    """Same as above, but with a chat model resident - must still route via the
    dedicated embedder, never attempt to resolve "bge-small-en-v1.5" as a chat
    engine."""
    registry = {
        "embedding-bge-small-en-v1.5": {
            "path": "models/embeddings/bge-small-en-v1.5-q4_k_m.gguf",
            "source": "setup-embeddings",
            "model_type": "embedding",
        },
        "some-chat-model": {"path": "models/some-chat-model.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: [[0.4, 0.5] for _ in texts])
    hs._active_model_name = "some-chat-model"

    r = app_client.post("/v1/embeddings",
                        json={"model": "bge-small-en-v1.5", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.4, 0.5]


def test_explicitly_registered_embedding_model_still_routes_by_exact_name(app_client, monkeypatch):
    """A model the user picked "from your models" (config stores the exact
    registered name) still routes via the registry model_type=="embedding"
    check."""
    registry = {
        "Qwen3-Embedding-4B-Q8_0": {
            "path": "models/Qwen3-Embedding-4B-Q8_0.gguf",
            "source": "hf:some/repo",
            "model_type": "embedding",
        },
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "Qwen3-Embedding-4B-Q8_0"})
    monkeypatch.setattr("localm.inference.embedder.embed_texts",
                        lambda texts: [[0.9] for _ in texts])

    r = app_client.post("/v1/embeddings",
                        json={"model": "Qwen3-Embedding-4B-Q8_0", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.9]


def test_unrelated_chat_model_name_does_not_spuriously_route_to_embedder(app_client, monkeypatch):
    """A request naming an ordinary registered chat model (not the configured
    embedding model, not model_type="embedding") must NOT be diverted to
    embed_texts() - it should fall through to the normal engine path."""
    registry = {
        "embedding-bge-small-en-v1.5": {
            "path": "models/embeddings/bge-small-en-v1.5-q4_k_m.gguf",
            "source": "setup-embeddings", "model_type": "embedding",
        },
        "some-chat-model": {"path": "models/some-chat-model.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})

    def must_not_be_called(texts):
        raise AssertionError("embed_texts must not be called for a plain chat model request")
    monkeypatch.setattr("localm.inference.embedder.embed_texts", must_not_be_called)

    class FakeBackend:
        can_embed = False

    class FakeEngine:
        display_name = "some-chat-model"
        loaded = True
        active_requests = 0
        _backend = FakeBackend()
        def embed(self, texts):
            return [[0.7] for _ in texts]
        def count_tokens(self, text):
            return len(text.split())

    monkeypatch.setattr(hs, "_engine_factory", lambda n: FakeEngine())
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: ("models/some-chat-model.gguf", "h"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda n: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})

    r = app_client.post("/v1/embeddings",
                        json={"model": "some-chat-model", "input": "hello"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["embedding"] == [0.7]
