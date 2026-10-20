# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registry entries the chat engine can never load are refused by name, before
the resident model is evicted, and the ones nothing under /v1 accepts are not
listed by GET /v1/models."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.model_manager.registry import (NON_CHAT_TYPES, UNLISTED_TYPES,
                                           chat_load_refusal, is_unlisted_entry)
from localm.plugins.gui.web import attach_gui
from tests.conftest import probe_double
from tests.test_projector_not_a_model import FakeEngine

COMPONENT_TYPES = sorted(NON_CHAT_TYPES - {"mmproj"})
WORDS = {
    "diffusion-unet": "an image or video checkpoint",
    "text-encoder": "a text encoder",
    "vae": "a VAE",
    "lora": "a LoRA adapter",
    "upscaler": "an upscaler",
    "tts": "a text-to-speech model",
}


def _name(kind):
    return f"zz-{kind}"


def _registry():
    reg = {"aaa-working": {"path": "models/aaa-working.gguf", "source": "local",
                           "model_type": "llm"},
           "emb": {"path": "models/emb.gguf", "model_type": "embedding"},
           "unk": {"path": "models/unk.gguf", "model_type": "unknown"},
           "proj": {"path": "models/proj-mmproj.gguf", "model_type": "mmproj"}}
    for kind in COMPONENT_TYPES:
        reg[_name(kind)] = {"path": f"models/{kind}.safetensors", "source": "local",
                            "model_type": kind}
    return reg


@pytest.fixture
def served(monkeypatch):
    reg = _registry()
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        engines[name] = FakeEngine(name)
        return engines[name]

    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", factory)
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    app = hs.create_app(None)
    client = TestClient(
        app, headers={"Authorization": f"Bearer {app.state.shell_token}"})
    r = client.post("/v1/chat/completions", json={
        "model": "aaa-working", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert hs._engines["aaa-working"].loaded
    return client, engines


def _assert_untouched(engines, name):
    assert name not in engines, "a component must never get an engine built for it"
    assert hs._engines["aaa-working"].loaded is True
    assert hs._active_model_name == "aaa-working"


@pytest.mark.parametrize("kind", COMPONENT_TYPES)
@pytest.mark.parametrize("stream", [False, True])
def test_chat_naming_a_component_is_refused_and_keeps_the_loaded_model(served, kind, stream):
    client, engines = served
    r = client.post("/v1/chat/completions", json={
        "model": _name(kind), "stream": stream,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422
    assert WORDS[kind] in r.json()["detail"]
    _assert_untouched(engines, _name(kind))


@pytest.mark.parametrize("kind", COMPONENT_TYPES)
def test_models_load_naming_a_component_is_refused(served, kind):
    client, engines = served
    r = client.post("/v1/models/load", params={"model": _name(kind)})
    assert r.status_code == 422
    _assert_untouched(engines, _name(kind))


@pytest.mark.parametrize("kind", COMPONENT_TYPES)
def test_get_engine_without_loading_refuses_a_component(served, kind):
    _, engines = served
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.get_engine(_name(kind), load=False))
    assert exc.value.status_code == 422
    _assert_untouched(engines, _name(kind))


@pytest.mark.parametrize("kind", COMPONENT_TYPES)
def test_switch_engine_refuses_a_component_before_cancelling_a_pending_switch(served, kind):
    _, engines = served
    hs._switch_desired = "aaa-working"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.switch_engine(_name(kind), hs._engine_factory))
    assert exc.value.status_code == 422
    assert hs._switch_desired == "aaa-working"
    _assert_untouched(engines, _name(kind))


@pytest.mark.parametrize("kind", COMPONENT_TYPES)
def test_gui_load_route_refuses_a_component_with_422(monkeypatch, kind):
    reg = _registry()
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    switched = []

    async def switch_model(name, **kw):
        switched.append(name)
        return {"status": "loaded", "model": name}

    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "aaa-working")
    with TestClient(app) as client:
        r = client.post("/api/models/load", json={"model": _name(kind)})
        ok = client.post("/api/models/load", json={"model": "aaa-working"})
    assert r.status_code == 422
    assert WORDS[kind] in r.json()["detail"]
    assert switched == ["aaa-working"]
    assert ok.status_code == 200


def test_v1_models_lists_what_v1_accepts_by_name(served):
    client, _ = served
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert ids == {"aaa-working", "emb", "unk", _name("tts")}
    assert UNLISTED_TYPES == NON_CHAT_TYPES - {"tts"}


def test_tts_message_points_at_the_speech_routes():
    msg = chat_load_refusal(_name("tts"), _registry())
    assert "localm speak" in msg and "/v1/audio/speech" in msg


@pytest.mark.parametrize("name", ["aaa-working", "emb", "unk", "not-registered"])
def test_loadable_types_and_unknown_names_are_not_refused(name):
    assert chat_load_refusal(name, _registry()) is None


def test_embedding_and_unknown_entries_still_load(served):
    client, _ = served
    for name in ("emb", "unk"):
        r = client.post("/v1/models/load", params={"model": name})
        assert r.status_code == 200 and r.json()["status"] == "loaded"


@pytest.mark.parametrize("bad", [None, "x", 5, [], {"model_type": None},
                                 {"model_type": ["vae"]}])
def test_malformed_entries_are_not_refused(bad):
    assert chat_load_refusal("n", {"n": bad}) is None
    assert is_unlisted_entry(bad) is False


def test_the_projector_keeps_its_own_message_naming_its_chat_model():
    reg = _registry()
    reg["chat"] = {"path": "models/proj-chat.gguf", "model_type": "llm"}
    assert "vision projector" in chat_load_refusal("proj", reg)
