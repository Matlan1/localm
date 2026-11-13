# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: F811
"""API protocol answers: the default-model sentinel on /v1/models/load, Ollama
model details and a mistyped ``think``, an empty Responses input, and a reply
cap of 0 on the OpenAI routes."""

import json

import pytest
from pydantic import ValidationError

from localm.inference import http_server as hs
from localm.inference import responses_protocol as R
from localm.inference.protocol import ChatRequest, CompletionRequest
from tests.test_model_unload_targeted import (  # noqa: F401
    FakeEngine, _authed_client, _load, setup_multi_model)
from tests.test_ollama_api import (  # noqa: F401
    MODEL, _write_registry, app, client, engine, home)


# ------------------------------------------------- /v1/models/load sentinel

def test_load_of_the_localm_sentinel_reports_the_model_already_loaded(setup_multi_model):
    c = _authed_client(hs.create_app(None))
    _load(c, "model-a")
    r = c.post("/v1/models/load", params={"model": "localm"})
    assert r.status_code == 200
    assert r.json()["status"] == "already_loaded"
    assert r.json()["model"] == "model-a"


def test_load_of_the_localm_sentinel_with_nothing_loaded_still_loads(setup_multi_model):
    c = _authed_client(hs.create_app(None))
    _load(c, "model-b")
    assert c.post("/v1/models/unload").status_code == 200
    r = c.post("/v1/models/load", params={"model": "localm"})
    assert r.status_code == 200
    assert r.json() == {**r.json(), "status": "loaded", "model": "model-b"}
    assert hs._engines["model-b"].loaded is True


# ------------------------------------------------------ Ollama details

@pytest.mark.parametrize("filename, size, quant", [
    ("Qwen3-8B-Q4_K_M.gguf", "8B", "Q4_K_M"),
    ("Qwen3-30B-A3B-Q8_0.gguf", "30B", "Q8_0"),
    ("Qwen3-0.6B-IQ4_XS.gguf", "0.6B", "IQ4_XS"),
    ("Mixtral-8x7B-Instruct-v0.1.Q5_K_S.gguf", "8x7B", "Q5_K_S"),
    ("gemma-3n-E2B-it-BF16.gguf", "", "BF16"),
    ("plain.gguf", "", ""),
])
def test_tags_and_show_report_size_and_quant_from_the_file_name(client, home,
                                                                filename, size, quant):
    weights = home / filename
    weights.write_bytes(b"x" * 10)
    _write_registry(home, {"alpha": {"path": str(weights), "architecture": "llama"}})
    tags = {m["name"]: m for m in client.get("/api/tags").json()["models"]}
    assert tags["alpha"]["details"]["parameter_size"] == size
    assert tags["alpha"]["details"]["quantization_level"] == quant
    show = client.post("/api/show", json={"model": "alpha"}).json()
    assert show["details"]["parameter_size"] == size
    assert show["details"]["quantization_level"] == quant


def test_ollama_think_of_the_wrong_type_gets_the_written_400(client):
    r = client.post("/api/chat", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "think": 5, "stream": False})
    assert r.status_code == 400
    assert r.json()["error"] == "think must be a boolean or a string"


def test_ollama_think_true_still_reaches_the_template(client, engine):
    r = client.post("/api/chat", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "think": True, "stream": False})
    assert r.status_code == 200


# ------------------------------------------------------ Responses input

def test_responses_blank_string_input_is_refused():
    with pytest.raises(R.ResponsesError) as ei:
        R.plan_chat(R.ResponsesRequest(model="m", input="   "), [])
    assert ei.value.status == 400
    assert str(ei.value) == "input must not be empty"


def test_responses_blank_input_answers_400_on_the_route(client):
    r = client.post("/v1/responses", json={"model": MODEL, "input": ""})
    assert r.status_code == 400
    assert "input must not be empty" in json.dumps(r.json())


# ------------------------------------------------------ reply cap 0

@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_chat_reply_cap_zero_reads_as_omitted(field):
    assert getattr(ChatRequest(messages=[{"role": "user", "content": "hi"}],
                               **{field: 0}), field) is None


def test_completion_reply_cap_zero_reads_as_omitted():
    assert CompletionRequest(prompt="x", max_tokens=0).max_tokens is None


@pytest.mark.parametrize("value", [-1, False])
def test_chat_reply_cap_negative_or_bool_is_still_refused(value):
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=value)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_chat_reply_cap_float_zero_reads_as_omitted(field, zero):
    assert getattr(ChatRequest(messages=[{"role": "user", "content": "hi"}],
                               **{field: zero}), field) is None


def test_completion_reply_cap_float_zero_reads_as_omitted():
    assert CompletionRequest(prompt="x", max_tokens=0.0).max_tokens is None


@pytest.mark.parametrize("value", [-1.0, 0.5, 1.5])
def test_chat_reply_cap_negative_or_fractional_float_is_still_refused(value):
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=value)


def test_chat_reply_cap_whole_float_is_kept_as_an_int():
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=64.0)
    assert req.max_tokens == 64 and isinstance(req.max_tokens, int)


def test_chat_route_with_max_tokens_zero_answers_without_a_cap(client, engine):
    r = client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 0})
    assert r.status_code == 200, r.text
    _messages, kwargs = engine.seen["calls"][-1]
    assert "max_tokens" not in kwargs


def test_chat_route_with_max_tokens_float_zero_answers_without_a_cap(client, engine):
    r = client.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 0.0})
    assert r.status_code == 200, r.text
    _messages, kwargs = engine.seen["calls"][-1]
    assert "max_tokens" not in kwargs
