# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: F811
"""API route defects found by the functional audit: JSON bodies on the model
lifecycle routes, embeddings reporting, key scope validation, empty chat
messages, a corrupt attached image, and the RAG query ``k`` bound."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from localm.inference import http_server as hs
from tests.test_model_unload_targeted import (  # noqa: F401
    FakeEngine, _authed_client, _load, setup_multi_model)


def _post_json(client, path, body):
    return client.post(path, content=json.dumps(body),
                       headers={"content-type": "application/json"})


# ----------------------------------------------------------------- F3

def test_unload_with_json_body_unloads_only_the_named_model(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    _load(client, "model-b")
    r = _post_json(client, "/v1/models/unload", {"model": "model-a"})
    assert r.status_code == 200
    assert r.json()["model"] == "model-a"
    assert hs._engines["model-a"].loaded is False
    assert hs._engines["model-b"].loaded is True


def test_unload_with_json_body_unknown_model_is_404_and_unloads_nothing(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    r = _post_json(client, "/v1/models/unload", {"model": "nope"})
    assert r.status_code == 404
    assert hs._engines["model-a"].loaded is True


def test_unload_body_with_unread_key_is_400_naming_it(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    r = _post_json(client, "/v1/models/unload", {"model": "model-a", "modle": "x"})
    assert r.status_code == 400
    assert "modle" in r.json()["detail"]
    assert hs._engines["model-a"].loaded is True


def test_unload_body_that_is_not_an_object_is_400(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    r = _post_json(client, "/v1/models/unload", ["model-a"])
    assert r.status_code == 400


def test_unload_query_wins_over_body(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    _load(client, "model-b")
    r = client.post("/v1/models/unload", params={"model": "model-b"},
                    content=json.dumps({"model": "model-a"}),
                    headers={"content-type": "application/json"})
    assert r.json()["model"] == "model-b"
    assert hs._engines["model-a"].loaded is True


def test_unload_empty_body_still_unloads_everything(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    _load(client, "model-b")
    assert client.post("/v1/models/unload").status_code == 200
    assert not hs._engines["model-a"].loaded and not hs._engines["model-b"].loaded


def test_load_with_json_body_loads_the_named_model(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    r = _post_json(client, "/v1/models/load", {"model": "model-b"})
    assert r.status_code == 200
    assert r.json()["model"] == "model-b"
    assert hs._engines["model-b"].loaded is True


def test_load_with_json_body_unknown_model_is_not_a_success(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    r = _post_json(client, "/v1/models/load", {"model": "nope"})
    assert r.status_code == 404
    assert "nope" not in hs._engines


def test_load_body_with_unread_key_is_400(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    r = _post_json(client, "/v1/models/load", {"model": "model-a", "force": True})
    assert r.status_code == 400 and "force" in r.json()["detail"]


def test_rename_with_json_body(setup_multi_model, monkeypatch):
    seen = {}

    async def fake_rename(model, new_name):
        seen.update(model=model, new_name=new_name)
        return {"status": "renamed", "model": model, "new_name": new_name, "notes": []}

    monkeypatch.setattr(hs, "rename_registered_model", fake_rename)
    client = _authed_client(hs.create_app(None))
    r = _post_json(client, "/v1/models/rename", {"model": "model-a", "new_name": "z"})
    assert r.status_code == 200
    assert seen == {"model": "model-a", "new_name": "z"}


def test_rename_without_names_is_400(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    assert _post_json(client, "/v1/models/rename", {"model": "model-a"}).status_code == 400


# ----------------------------------------------------------------- F11

def test_chat_completions_with_no_messages_is_400(setup_multi_model):
    client = _authed_client(hs.create_app(None))
    r = client.post("/v1/chat/completions", json={"model": "model-a", "messages": []})
    assert r.status_code == 400
    assert "messages" in r.json()["detail"]


# ----------------------------------------------------------------- F4

class _DedicatedEngine(FakeEngine):
    class _Backend:
        can_embed = False

    _backend = _Backend()

    def embeds_via_dedicated(self):
        return True


def test_embeddings_with_a_chat_model_name_the_embedder_that_answered(
        setup_multi_model, monkeypatch):
    monkeypatch.setattr(hs, "_engine_factory", lambda name: _DedicatedEngine(name))
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})
    client = _authed_client(hs.create_app(None))
    r = client.post("/v1/embeddings", json={"model": "model-a", "input": "hello world"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "bge-small-en-v1.5"
    assert r.headers["x-localm-embedding-model"] == "bge-small-en-v1.5"
    assert r.headers["x-localm-requested-model"] == "model-a"
    assert body["usage"]["prompt_tokens"] > 0


def test_embeddings_with_an_embedding_model_fills_usage(setup_multi_model, monkeypatch):
    import localm.inference.embedder as emb
    import localm.inference.reranker as rr
    monkeypatch.setattr("localm.config.load_registry", lambda: {
        "emb": {"path": "Z:/models/emb.gguf", "source": "local", "model_type": "embedding"}})
    monkeypatch.setattr("localm.model_manager.registry.entry_is_reranker",
                        lambda *a, **k: False)
    monkeypatch.setattr(rr, "named_embedding_path", lambda name, entry: "Z:/models/emb.gguf")
    monkeypatch.setattr(rr, "embed", lambda path, texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(emb, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])
    client = _authed_client(hs.create_app(None))
    r = client.post("/v1/embeddings",
                    json={"model": "emb", "input": ["some text here", "more"]})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "emb"
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"]


# ----------------------------------------------------------------- F8

def _owner_client():
    app = hs.create_app(None)
    return TestClient(app, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def test_key_with_unknown_scope_is_refused_listing_valid_ones():
    client = _owner_client()
    r = client.post("/v1/keys", json={"name": "b", "scopes": ["nonsense"]})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "nonsense" in detail and "models:read" in detail
    listing = client.get("/v1/keys")
    assert all("nonsense" not in k.get("scopes", []) for k in listing.json()["keys"])


def test_key_with_known_scopes_is_still_minted():
    r = _owner_client().post("/v1/keys",
                             json={"name": "ok", "scopes": ["chat", "models:read"]})
    assert r.status_code == 200, r.text


# ----------------------------------------------------------------- F10

@pytest.mark.parametrize("k", [0, -3])
def test_rag_query_k_below_one_is_rejected(k):
    from localm.plugins.builtin.rag.plug import RagQueryRequest
    with pytest.raises(ValidationError):
        RagQueryRequest(query="bread", k=k)
    assert RagQueryRequest(query="bread").k == 4


# ----------------------------------------------------------------- F13

def test_corrupt_attached_image_is_a_400_with_the_decoder_reason():
    from localm.inference.backends.base import ImageLimitError, VisionInputError
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    with pytest.raises(VisionInputError) as ei:
        LlamaCpp._messages_with_markers(msgs, "<m>")
    assert not isinstance(ei.value, ImageLimitError)
    assert hs.backend_error_status(ei.value) == 400
    assert "could not be read: " in str(ei.value)
    assert not str(ei.value).endswith(": ")


# ----------------------------------------------------------------- F15

def _remove_from_registry(monkeypatch, name):
    reg = {"model-b": {"path": "Z:/models/model-b.gguf", "source": "local"}}
    assert name not in reg
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)


def test_a_removed_model_is_404_by_name_and_not_resolved_for_an_unnamed_request(
        setup_multi_model, monkeypatch):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    assert client.post("/v1/models/unload", params={"model": "model-a"}).status_code == 200
    _remove_from_registry(monkeypatch, "model-a")

    named = client.post("/v1/chat/completions", json={
        "model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    assert named.status_code == 404, named.text

    unnamed = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert unnamed.status_code != 200 or "model-a" not in unnamed.text
    assert unnamed.status_code in (400, 404, 503), unnamed.text
    assert "model-a" not in hs._engines
    assert hs._last_active_model_name != "model-a"


def test_a_removed_model_that_is_still_loaded_is_left_alone(setup_multi_model, monkeypatch):
    client = _authed_client(hs.create_app(None))
    _load(client, "model-a")
    _remove_from_registry(monkeypatch, "model-a")
    assert hs._forget_removed_model("model-a") is False
    assert "model-a" in hs._engines
