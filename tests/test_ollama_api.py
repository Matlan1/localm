# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Ollama-native routes against a real create_app() with a mock engine:
shapes, streaming, options, auth refusals in protected mode, the open-mode
origin guard, and the 501s."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import localm
from localm import scopes as S
from localm.inference import ollama_protocol as P
from localm.inference.http_server import create_app

MODEL = "test-model"
TINY_PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde").decode()


def _mock_engine(tokens=("Hello", " world")):
    engine = MagicMock()
    state = {"loaded": True}
    seen: dict = {"calls": []}

    def _chat_stream(messages, **kwargs):
        seen["calls"].append((messages, kwargs))
        yield from tokens

    engine.unload.side_effect = lambda: state.update(loaded=False)
    engine.load.side_effect = lambda: state.update(loaded=True)
    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = MODEL
    engine.model_path = ""
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    engine.gpu_placement = None
    engine.supports_images = True
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: state["loaded"])
    engine.seen = seen
    return engine


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".localm"
    root.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(root))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", root)
    monkeypatch.setattr(_cfg, "MODELS_DIR", root / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", root / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", root / "registry.json")
    return root


def _write_registry(home: Path, entries: dict) -> None:
    (home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.fixture
def engine():
    return _mock_engine()


@pytest.fixture
def app(home, engine):
    return create_app(engine)


@pytest.fixture
def client(app):
    return TestClient(app, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def _error_of(response) -> str:
    return response.json()["error"]


def _has_key(response, key: str) -> bool:
    return key in response.json()


def _lines(response):
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.split("\n") if line]


def _chat(client, **body):
    body.setdefault("model", MODEL)
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post("/api/chat", json=body)


def _generate(client, **body):
    body.setdefault("model", MODEL)
    body.setdefault("prompt", "hi")
    return client.post("/api/generate", json=body)


# ------------------------------------------------------------------ routes

OLLAMA_ROUTES = {
    ("GET", "/api/version"), ("GET", "/api/tags"), ("GET", "/api/ps"),
    ("POST", "/api/show"), ("POST", "/api/chat"), ("POST", "/api/generate"),
    ("POST", "/api/embed"), ("POST", "/api/embeddings"), ("POST", "/api/copy"),
    ("POST", "/api/pull"), ("POST", "/api/push"), ("POST", "/api/create"),
    ("DELETE", "/api/delete"), ("POST", "/api/blobs/{digest}"),
    ("HEAD", "/api/blobs/{digest}"),
}


def test_every_ollama_route_is_registered_exactly_once(app):
    registered = [(m, r.path) for r in app.routes if isinstance(r, APIRoute)
                  for m in sorted(r.methods)]
    for expected in OLLAMA_ROUTES:
        assert registered.count(expected) == 1, expected
    assert len(registered) == len(set(registered)), "a route is registered twice"


def test_no_gui_route_shares_an_ollama_path(app):
    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda n: None, active_model=lambda: MODEL)
    registered = [(m, r.path) for r in app.routes if isinstance(r, APIRoute)
                  for m in sorted(r.methods)]
    for expected in OLLAMA_ROUTES:
        assert registered.count(expected) == 1, expected


# ------------------------------------------------------------------ reads

def test_version(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    assert r.json() == {"version": localm.__version__}


def test_tags_lists_registered_chat_and_embedding_models_only(client, home):
    weights = home / "m.gguf"
    weights.write_bytes(b"x" * 1234)
    _write_registry(home, {
        "alpha": {"path": str(weights), "model_type": "llm", "sha256": "sha256:abc",
                  "architecture": "llama"},
        "embed": {"path": str(weights), "model_type": "embedding"},
        "proj": {"path": str(weights), "model_type": "mmproj"},
        "vae": {"path": str(weights), "model_type": "vae"},
        "broken": "not a dict",
    })
    body = client.get("/api/tags").json()
    by_name = {m["name"]: m for m in body["models"]}
    assert set(by_name) == {"alpha", "embed", MODEL}
    alpha = by_name["alpha"]
    assert alpha["model"] == "alpha" and alpha["size"] == 1234
    assert alpha["digest"] == "abc"
    assert alpha["details"]["format"] == "gguf" and alpha["details"]["family"] == "llama"
    assert alpha["modified_at"].endswith("Z")


def test_tags_survives_an_entry_whose_file_is_gone(client, home):
    _write_registry(home, {"ghost": {"path": str(home / "missing.gguf")}})
    ghost = {m["name"]: m for m in client.get("/api/tags").json()["models"]}["ghost"]
    assert ghost["size"] == 0 and ghost["modified_at"] is None


def test_ps_lists_only_loaded_models(client, engine):
    names = [m["name"] for m in client.get("/api/ps").json()["models"]]
    assert names == [MODEL]
    engine.unload()
    assert client.get("/api/ps").json() == {"models": []}


def test_ps_reports_the_loaded_context_length(client):
    model = client.get("/api/ps").json()["models"][0]
    assert model["context_length"] == 4096 and "modified_at" not in model


def test_ps_expiry_is_far_future_without_an_idle_timeout(client):
    model = client.get("/api/ps").json()["models"][0]
    assert model["expires_at"] == P.NEVER_EXPIRES


def test_ps_expiry_follows_the_idle_timeout_and_the_last_request(client):
    import time

    import localm.inference.http_server as hs
    with patch.object(hs, "_idle_unload_ttl", return_value=600):
        hs._last_activity_per_model[MODEL] = time.monotonic() - 100
        model = client.get("/api/ps").json()["models"][0]
    from datetime import datetime
    remaining = datetime.fromisoformat(
        model["expires_at"].replace("Z", "+00:00")).timestamp() - time.time()
    assert 480 < remaining <= 500


def test_show_describes_a_registered_model(client, home):
    weights = home / "m.gguf"
    weights.write_bytes(b"x" * 10)
    _write_registry(home, {"alpha": {"path": str(weights), "architecture": "llama",
                                     "context_length": 8192}})
    body = client.post("/api/show", json={"model": "alpha"}).json()
    assert body["capabilities"] == ["completion"]
    assert body["details"]["format"] == "gguf"
    assert body["model_info"]["general.architecture"] == "llama"
    assert body["model_info"]["llama.context_length"] == 8192
    assert {"modelfile", "parameters", "template"} <= set(body)


def test_show_accepts_the_legacy_name_field_and_the_latest_tag(client, home):
    _write_registry(home, {"alpha": {"path": str(home / "x.gguf")}})
    assert client.post("/api/show", json={"name": "alpha"}).status_code == 200
    assert client.post("/api/show", json={"model": "alpha:latest"}).status_code == 200


def test_show_never_claims_tools(client, home):
    _write_registry(home, {"alpha": {"path": str(home / "x.gguf"), "tool_use": True}})
    caps = client.post("/api/show", json={"model": "alpha"}).json()["capabilities"]
    assert "tools" not in caps


def test_show_marks_an_embedding_model(client, home):
    _write_registry(home, {"emb": {"path": str(home / "x.gguf"), "model_type": "embedding"}})
    caps = client.post("/api/show", json={"model": "emb"}).json()["capabilities"]
    assert caps == ["embedding"]


def test_show_unknown_model_is_a_404_with_the_ollama_error_body(client):
    r = client.post("/api/show", json={"model": "nope"})
    assert r.status_code == 404
    assert r.json() == {"error": "model 'nope' not found"}


def test_show_without_a_model_is_a_400(client):
    r = client.post("/api/show", json={})
    assert r.status_code == 400 and r.json() == {"error": "model is required"}


# ------------------------------------------------------------------ chat

def test_chat_non_streaming(client):
    r = _chat(client, stream=False)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == MODEL
    assert body["message"] == {"role": "assistant", "content": "Hello world"}
    assert body["done"] is True and body["done_reason"] == "stop"
    assert body["prompt_eval_count"] == 3 and body["eval_count"] > 0
    assert body["total_duration"] > 0
    assert body["created_at"].endswith("Z")


def test_chat_streams_ndjson_by_default(client):
    r = _chat(client)
    objs = _lines(r)
    assert [o["message"]["content"] for o in objs] == ["Hello", " world", ""]
    assert [o["done"] for o in objs] == [False, False, True]
    assert objs[-1]["done_reason"] == "stop"
    assert "\n\n" not in r.text and r.text.endswith("\n")


def test_chat_stream_false_is_one_json_document(client):
    r = _chat(client, stream=False)
    assert r.headers["content-type"].startswith("application/json")
    assert r.text.count("\n") <= 1


def test_chat_echoes_the_model_name_the_client_sent(client, home):
    r = _chat(client, model=MODEL + ":latest", stream=False)
    assert r.status_code == 200 and r.json()["model"] == MODEL + ":latest"


def test_chat_maps_options_to_the_engine(client, engine):
    r = _chat(client, stream=False, options={
        "temperature": 0.3, "top_p": 0.8, "top_k": 30, "repeat_penalty": 1.2,
        "seed": 11, "num_predict": 17, "num_ctx": 9999})
    assert r.status_code == 200
    _messages, kwargs = engine.seen["calls"][-1]
    assert kwargs["temperature"] == 0.3 and kwargs["top_p"] == 0.8
    assert kwargs["top_k"] == 30 and kwargs["repeat_penalty"] == 1.2
    assert kwargs["seed"] == 11 and kwargs["max_tokens"] == 17


def test_chat_passes_the_messages_through(client, engine):
    _chat(client, stream=False, messages=[
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}])
    messages, _kwargs = engine.seen["calls"][-1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"].startswith("be brief")
    assert [m["content"] for m in messages[1:]] == ["one", "two", "three"]


def test_chat_stop_sequence_cuts_the_reply(client):
    body = _chat(client, stream=False, options={"stop": ["wor"]}).json()
    assert body["message"]["content"] == "Hello "
    assert body["done_reason"] == "stop"
    assert body["eval_count"] > 0 and body["total_duration"] > 0
    objs = _lines(_chat(client, options={"stop": ["wor"]}))
    assert "".join(o["message"]["content"] for o in objs) == "Hello "
    assert objs[-1]["done"] is True and objs[-1]["done_reason"] == "stop"
    assert objs[-1]["eval_count"] > 0


def test_chat_format_json_reaches_the_engine_as_a_grammar(client, engine):
    r = _chat(client, stream=False, format="json")
    assert r.status_code == 200
    _messages, kwargs = engine.seen["calls"][-1]
    assert kwargs["grammar"] == P.JSON_GRAMMAR
    engine.validate_grammar.assert_called()


def test_chat_format_schema_reaches_the_engine_as_a_grammar(client, engine):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    r = _chat(client, stream=False, format=schema)
    assert r.status_code == 200
    _messages, kwargs = engine.seen["calls"][-1]
    assert kwargs["grammar"].startswith("root ::=") and "name" in kwargs["grammar"]


def test_chat_format_schema_the_grammar_cannot_enforce_is_a_400(client):
    r = _chat(client, format={"type": "string", "pattern": "^a"})
    assert r.status_code == 400 and "pattern" in _error_of(r)


def test_chat_tools_are_refused_not_dropped(client):
    r = _chat(client, tools=[{"type": "function", "function": {"name": "f"}}])
    assert r.status_code == 400 and "tools" in _error_of(r)


def test_chat_think_false_disables_thinking(client, engine):
    _chat(client, stream=False, think=False)
    _messages, kwargs = engine.seen["calls"][-1]
    assert kwargs["thinking"] is False


def test_chat_thinking_is_returned_only_when_asked():
    engine = _mock_engine(tokens=("<think>because</think>", "answer"))
    app = create_app(engine)
    client = TestClient(app, headers={"Authorization": f"Bearer {app.state.shell_token}"})
    plain = _chat(client, stream=False).json()["message"]
    assert plain["content"] == "answer" and "thinking" not in plain
    asked = _chat(client, stream=False, think=True).json()["message"]
    assert asked["content"] == "answer" and asked["thinking"] == "because"
    objs = _lines(_chat(client, think=True))
    assert any(o["message"].get("thinking") == "because" for o in objs)


def test_chat_images_reach_the_engine_as_image_parts(client, engine):
    r = _chat(client, stream=False, messages=[
        {"role": "user", "content": "what is this", "images": [TINY_PNG]}])
    assert r.status_code == 200, r.text
    messages, _kwargs = engine.seen["calls"][-1]
    parts = messages[-1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_chat_length_finish_is_reported(client, engine):
    engine.last_finish_reason = "length"
    assert _chat(client, stream=False).json()["done_reason"] == "length"
    assert _lines(_chat(client))[-1]["done_reason"] == "length"


def test_chat_unknown_model_is_a_404_in_the_ollama_error_shape(client, home):
    _write_registry(home, {"alpha": {"path": str(home / "x.gguf")}})
    r = _chat(client, model="nope", stream=False)
    assert r.status_code == 404
    assert "nope" in _error_of(r) and not _has_key(r, "detail")


def test_chat_missing_model_and_bad_bodies_are_400s_with_an_error_key(client):
    assert client.post("/api/chat", json={"messages": []}).json() == {
        "error": "model is required"}
    r = client.post("/api/chat", json={"model": MODEL, "messages": "oops"})
    assert r.status_code == 400 and isinstance(_error_of(r), str)
    r = _chat(client, options={"temperature": "hot"}, stream=False)
    assert r.status_code == 400 and _has_key(r, "error")


def test_chat_generation_failure_is_a_500_not_a_200(client, engine):
    def boom(messages, **kwargs):
        raise RuntimeError("out of memory")
        yield  # pragma: no cover
    engine.chat_stream.side_effect = boom
    r = _chat(client, stream=False)
    assert r.status_code == 500 and "out of memory" in _error_of(r)


def test_chat_stream_generation_failure_ends_with_an_error_line(client, engine):
    def boom(messages, **kwargs):
        yield "par"
        raise RuntimeError("out of memory")
    engine.chat_stream.side_effect = boom
    objs = _lines(_chat(client))
    assert objs[0]["message"]["content"] == "par"
    assert "out of memory" in objs[-1]["error"]
    assert not any(o.get("done") for o in objs)


def test_chat_load_idiom_loads_without_generating(client, engine):
    engine.unload()
    r = client.post("/api/chat", json={"model": MODEL, "messages": []})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True and body["done_reason"] == "load"
    assert body["message"]["content"] == ""
    assert engine.loaded is True
    assert engine.seen["calls"] == []


def test_load_idiom_leaves_a_peer_routed_model_to_its_peer(client, engine):
    engine.unload()
    with patch("localm.inference.routes.ollama.peer_routing.get_route",
               return_value=object()):
        r = client.post("/api/chat", json={"model": MODEL, "messages": []})
    assert r.status_code == 200 and r.json()["done_reason"] == "load"
    assert engine.loaded is False


def test_chat_unload_idiom_unloads(client, engine):
    with patch("localm.discover.vram_info", return_value={}):
        r = client.post("/api/chat",
                        json={"model": MODEL, "messages": [], "keep_alive": 0})
    assert r.status_code == 200 and r.json()["done_reason"] == "unload"
    assert engine.loaded is False


def test_unknown_request_keys_are_ignored_not_fatal(client):
    r = _chat(client, stream=False, mystery_field=1, options={"mirostat": 2})
    assert r.status_code == 200


def test_ndjson_survives_unicode_line_breaks_in_the_reply():
    engine = _mock_engine(tokens=("a\u2028b", "c\u2029d\x85"))
    app = create_app(engine)
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"model": MODEL, "stream": True,
                                      "messages": [{"role": "user", "content": "hi"}]})
    lines = r.text.splitlines()
    assert all(json.loads(line) for line in lines)
    text = "".join(json.loads(line)["message"]["content"] for line in lines)
    assert text == "a\u2028bc\u2029d\x85"


# ------------------------------------------------------------------ generate

def test_generate_non_streaming(client):
    body = _generate(client, stream=False).json()
    assert body["response"] == "Hello world" and body["done"] is True
    assert body["done_reason"] == "stop" and "message" not in body


def test_generate_streams_response_fragments(client):
    objs = _lines(_generate(client))
    assert [o["response"] for o in objs] == ["Hello", " world", ""]
    assert objs[-1]["done"] is True


def test_generate_sends_system_then_prompt(client, engine):
    _generate(client, stream=False, system="Be brief.", prompt="Why?")
    messages, _kwargs = engine.seen["calls"][-1]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("Be brief.")
    assert messages[1]["content"] == "Why?"


@pytest.mark.parametrize("extra", [
    {"raw": True}, {"suffix": "tail"}, {"template": "{{ .Prompt }}"},
    {"context": [1, 2]}])
def test_generate_refuses_what_it_cannot_honour(client, extra):
    r = _generate(client, stream=False, **extra)
    assert r.status_code == 400 and _has_key(r, "error")


def test_generate_load_idiom(client, engine):
    engine.unload()
    r = client.post("/api/generate", json={"model": MODEL})
    assert r.status_code == 200
    assert r.json()["response"] == "" and r.json()["done_reason"] == "load"
    assert engine.loaded is True


def test_generate_unload_idiom(client, engine):
    with patch("localm.discover.vram_info", return_value={}):
        r = client.post("/api/generate", json={"model": MODEL, "keep_alive": "0s"})
    assert r.json()["done_reason"] == "unload" and engine.loaded is False


# ------------------------------------------------------------------ embeddings

@pytest.fixture
def embedding_engine(engine):
    engine.embed.side_effect = None
    engine.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
    engine.count_tokens.return_value = 2
    return engine


def test_embed_returns_one_vector_per_input(client, embedding_engine):
    body = client.post("/api/embed", json={"model": MODEL, "input": ["a", "b"]}).json()
    assert body["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert body["model"] == MODEL and body["prompt_eval_count"] == 4
    assert body["total_duration"] > 0


def test_embed_leaves_out_a_token_count_the_backend_did_not_measure(client, embedding_engine):
    embedding_engine.count_tokens.return_value = 0
    body = client.post("/api/embed", json={"model": MODEL, "input": ["a", "b"]}).json()
    assert "prompt_eval_count" not in body and len(body["embeddings"]) == 2


def test_embed_accepts_a_single_string(client, embedding_engine):
    embedding_engine.embed.return_value = [[0.5, 0.6]]
    body = client.post("/api/embed", json={"model": MODEL, "input": "a"}).json()
    assert body["embeddings"] == [[0.5, 0.6]]


def test_embed_empty_input_list_is_an_empty_result(client):
    body = client.post("/api/embed", json={"model": MODEL, "input": []}).json()
    assert body == {"model": MODEL, "embeddings": []}


def test_embed_requires_input_and_refuses_dimensions(client, embedding_engine):
    assert client.post("/api/embed", json={"model": MODEL}).status_code == 400
    r = client.post("/api/embed", json={"model": MODEL, "input": "a", "dimensions": 8})
    assert r.status_code == 400 and "dimensions" in _error_of(r)


def test_legacy_embeddings_returns_a_single_vector(client, embedding_engine):
    embedding_engine.embed.return_value = [[0.7, 0.8]]
    body = client.post("/api/embeddings", json={"model": MODEL, "prompt": "a"}).json()
    assert body == {"embedding": [0.7, 0.8]}
    assert client.post("/api/embeddings", json={"model": MODEL}).status_code == 400


# ------------------------------------------------------------------ copy, 501s

def test_copy_creates_an_alias(client, home):
    weights = home / "m.gguf"
    weights.write_bytes(b"x")
    _write_registry(home, {"alpha": {"path": str(weights)}})
    r = client.post("/api/copy", json={"source": "alpha", "destination": "beta"})
    assert r.status_code == 200 and r.json() == {"status": "success"}
    assert set(json.loads((home / "registry.json").read_text())) == {"alpha", "beta"}


def test_copy_errors(client, home):
    _write_registry(home, {"alpha": {"path": "x"}, "beta": {"path": "y"}})
    missing = client.post("/api/copy", json={"source": "nope", "destination": "z"})
    assert missing.status_code == 404 and "nope" in _error_of(missing)
    taken = client.post("/api/copy", json={"source": "alpha", "destination": "beta"})
    assert taken.status_code == 409
    assert client.post("/api/copy", json={"source": "alpha"}).status_code == 400
    odd = client.post("/api/copy", json={"source": "alpha", "destination": "../evil"})
    assert odd.status_code == 400
    assert set(json.loads((home / "registry.json").read_text())) == {"alpha", "beta"}


@pytest.mark.parametrize("method,path,needle", [
    ("POST", "/api/pull", "localm pull"),
    ("POST", "/api/push", "registry"),
    ("POST", "/api/create", "Modelfile"),
    ("DELETE", "/api/delete", "localm rm"),
    ("POST", "/api/blobs/sha256:abc", "/api/create"),
])
def test_unsupported_verbs_answer_501_with_the_localm_equivalent(
        client, method, path, needle):
    r = client.request(method, path, json={"model": "x"})
    assert r.status_code == 501
    assert needle in _error_of(r)


# ------------------------------------------------------------------ auth: protected mode

EVIL = {"Origin": "http://localhost:6666"}
CHAT_BODY = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
             "stream": False}
CASES = [
    # (method, path, json body, scope the route needs: None = any key)
    ("GET", "/api/version", None, None),
    ("POST", "/api/chat", CHAT_BODY, None),
    ("POST", "/api/generate", {"model": MODEL, "prompt": "hi", "stream": False}, None),
    ("POST", "/api/embed", {"model": MODEL, "input": "a"}, None),
    ("POST", "/api/embeddings", {"model": MODEL, "prompt": "a"}, None),
    ("GET", "/api/tags", None, S.MODELS_READ),
    ("GET", "/api/ps", None, S.MODELS_READ),
    ("POST", "/api/show", {"model": MODEL}, S.MODELS_READ),
    ("POST", "/api/copy", {"source": "a", "destination": "b"}, S.MODELS_WRITE),
    ("POST", "/api/pull", {"model": "x"}, S.MODELS_WRITE),
    ("POST", "/api/push", {"model": "x"}, S.MODELS_WRITE),
    ("POST", "/api/create", {"model": "x"}, S.MODELS_WRITE),
    ("DELETE", "/api/delete", {"model": "x"}, S.MODELS_WRITE),
    ("POST", "/api/blobs/sha256:abc", None, S.MODELS_WRITE),
    ("HEAD", "/api/blobs/sha256:abc", None, S.MODELS_WRITE),
]


@pytest.fixture
def protected(home, engine):
    from localm import auth
    app = create_app(engine)
    keys = {
        "owner": auth.create_key("owner", [S.ADMIN], allow_privileged=True)["key"],
        "chat": auth.create_key("chat", [S.CHAT])["key"],
        "reader": auth.create_key("reader", [S.MODELS_READ])["key"],
    }
    return app, keys


def _call(client, method, path, body, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    kwargs = {"json": body} if body is not None else {}
    return client.request(method, path, headers=headers, **kwargs)


@pytest.mark.parametrize("method,path,body,scope", CASES)
def test_no_key_is_refused_with_a_401_error_body(protected, method, path, body, scope):
    app, _keys = protected
    with TestClient(app) as c:
        r = _call(c, method, path, body)
    assert r.status_code == 401
    if method != "HEAD":
        assert isinstance(_error_of(r), str) and not _has_key(r, "detail")


@pytest.mark.parametrize("method,path,body,scope", CASES)
def test_a_wrong_key_is_refused(protected, method, path, body, scope):
    app, _keys = protected
    with TestClient(app) as c:
        assert _call(c, method, path, body, key="not-a-key").status_code == 401


@pytest.mark.parametrize("method,path,body,scope", [c for c in CASES if c[3]])
def test_a_key_without_the_scope_is_refused_with_a_403(protected, method, path, body, scope):
    app, keys = protected
    with TestClient(app) as c:
        r = _call(c, method, path, body, key=keys["chat"])
    assert r.status_code == 403
    if method != "HEAD":
        assert scope in _error_of(r)


@pytest.mark.parametrize("method,path,body,scope", [c for c in CASES if c[3] == S.MODELS_WRITE])
def test_a_read_only_key_cannot_use_a_mutating_verb(protected, method, path, body, scope):
    app, keys = protected
    with TestClient(app) as c:
        assert _call(c, method, path, body, key=keys["reader"]).status_code == 403


@pytest.mark.parametrize("method,path,body,scope", [c for c in CASES if c[3] is None])
def test_any_valid_key_reaches_the_inference_routes(protected, method, path, body, scope):
    app, keys = protected
    with TestClient(app) as c:
        r = _call(c, method, path, body, key=keys["chat"])
    assert r.status_code == 200, r.text


def test_a_models_read_key_reads_the_listings(protected):
    app, keys = protected
    with TestClient(app) as c:
        for method, path, body in (("GET", "/api/tags", None), ("GET", "/api/ps", None),
                                   ("POST", "/api/show", {"model": MODEL})):
            assert _call(c, method, path, body, key=keys["reader"]).status_code == 200


def test_the_owner_key_reaches_a_mutating_verb(protected):
    app, keys = protected
    with TestClient(app) as c:
        assert _call(c, "POST", "/api/pull", {"model": "x"}, key=keys["owner"]).status_code == 501


UNLOAD_BODY = {"model": MODEL, "messages": [], "keep_alive": 0}


def test_open_mode_unload_idiom_is_a_management_action(app, engine):
    token = {"Authorization": f"Bearer {app.state.shell_token}"}
    with patch("localm.discover.vram_info", return_value={}), TestClient(app) as c:
        bare = c.post("/api/chat", json=UNLOAD_BODY)
        foreign = c.post("/api/chat", json=UNLOAD_BODY, headers={**token, **EVIL})
        wrong = c.post("/api/chat", json=UNLOAD_BODY,
                       headers={"Authorization": "Bearer not-the-token"})
        assert engine.loaded is True, "a refused unload must not unload"
        for refused in (bare, foreign, wrong):
            assert refused.status_code == 403 and "API key" in _error_of(refused)
        allowed = c.post("/api/chat", json=UNLOAD_BODY, headers=token)
    assert allowed.status_code == 200 and allowed.json()["done_reason"] == "unload"
    assert engine.loaded is False


def test_open_mode_generate_unload_idiom_is_a_management_action(app, engine):
    body = {"model": MODEL, "keep_alive": 0}
    with TestClient(app) as c:
        r = c.post("/api/generate", json=body, headers=EVIL)
    assert r.status_code == 403 and engine.loaded is True


def test_unload_idiom_needs_models_write_in_protected_mode(protected, engine):
    app, keys = protected
    body = {"model": MODEL, "messages": [], "keep_alive": 0}
    with TestClient(app) as c:
        denied = _call(c, "POST", "/api/chat", body, key=keys["chat"])
        assert denied.status_code == 403 and engine.loaded is True
        with patch("localm.discover.vram_info", return_value={}):
            allowed = _call(c, "POST", "/api/chat", body, key=keys["owner"])
    assert allowed.status_code == 200 and engine.loaded is False


def test_non_ollama_paths_keep_their_detail_error_shape(protected):
    app, _keys = protected
    with TestClient(app) as c:
        r = c.get("/v1/models")
    assert r.status_code == 401 and _has_key(r, "detail") and not _has_key(r, "error")


# ------------------------------------------------------------------ open mode: origin guard

def test_open_mode_inference_routes_are_callable_cross_origin_like_v1(app):
    with TestClient(app) as c:
        chat = c.post("/api/chat", json=CHAT_BODY, headers=EVIL)
        v1 = c.post("/v1/chat/completions", json=CHAT_BODY, headers=EVIL)
    assert v1.status_code == 200
    assert chat.status_code == 200


def test_open_mode_reads_need_no_shell_token_like_v1_models(app):
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 200
        for path in ("/api/tags", "/api/ps", "/api/version"):
            assert c.get(path).status_code == 200, path
        assert c.post("/api/show", json={"model": MODEL}).status_code == 200


def test_open_mode_cross_origin_page_cannot_use_mutating_verbs(app):
    with TestClient(app) as c:
        for method, path, body in (
                ("POST", "/api/copy", {"source": "a", "destination": "b"}),
                ("POST", "/api/pull", {"model": "x"}),
                ("DELETE", "/api/delete", {"model": "x"}),
                ("POST", "/api/create", {"model": "x"})):
            r = c.request(method, path, json=body, headers=EVIL)
            assert r.status_code == 403, path
            assert "Cross-origin" in r.json()["detail"], path


def test_open_mode_mutating_verbs_need_the_shell_token(app):
    with TestClient(app) as c:
        refused = c.post("/api/pull", json={"model": "x"})
        assert refused.status_code == 403
        allowed = c.post("/api/pull", json={"model": "x"},
                         headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert allowed.status_code == 501


def test_the_gui_embedding_warmup_route_is_not_exempted_by_the_embed_paths(app):
    with TestClient(app) as c:
        r = c.post("/api/embedding/warmup", json={}, headers=EVIL)
    assert r.status_code == 403
    assert "Cross-origin" in r.json()["detail"]
