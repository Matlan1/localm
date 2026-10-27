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

COMPONENT_TYPES = ["diffusion-unet", "lora", "text-encoder", "tts", "upscaler", "vae",
                   "controlnet", "textual-inversion", "segmentation"]
WORDS = {
    "diffusion-unet": "an image or video checkpoint",
    "text-encoder": "a text encoder",
    "vae": "a VAE",
    "lora": "a LoRA adapter",
    "upscaler": "an upscaler",
    "tts": "a text-to-speech model",
    "controlnet": "a ControlNet",
    "textual-inversion": "a textual-inversion embedding",
    "segmentation": "a segmentation model",
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
def unknown_file(tmp_path):
    path = tmp_path / "RealESRGAN_x4plus.safetensors"
    path.write_bytes(b"x" * 16)
    return path


@pytest.fixture
def served(monkeypatch, unknown_file):
    reg = _registry()
    reg["zz-unknown-file"] = {"path": str(unknown_file), "model_type": "unknown"}
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
    assert ids == {"aaa-working", "emb", _name("tts")}


def test_the_refused_and_unlisted_type_sets():
    assert NON_CHAT_TYPES == {"mmproj", *COMPONENT_TYPES}
    assert UNLISTED_TYPES == (NON_CHAT_TYPES - {"tts"}) | {"unknown"}


def test_tts_message_points_at_the_speech_route_only_when_the_tts_plugin_is_active(monkeypatch):
    monkeypatch.setattr("localm.model_manager.registry._speech_route_available", lambda: True)
    msg = chat_load_refusal(_name("tts"), _registry())
    assert "localm speak" in msg and "/v1/audio/speech" in msg
    monkeypatch.setattr("localm.model_manager.registry._speech_route_available", lambda: False)
    msg = chat_load_refusal(_name("tts"), _registry())
    assert "localm speak" in msg and "/v1/audio/speech" not in msg


def test_speech_route_availability_follows_the_installed_and_enabled_tts_plugin(
        monkeypatch, tmp_path):
    from localm.model_manager.registry import _speech_route_available
    from localm.plugins import engine as plugin_engine
    installed = tmp_path / "installed"
    (installed / "tts").mkdir(parents=True)
    (installed / "tts" / "plugin.toml").write_text("name = 'tts'\n", encoding="utf-8")
    enabled: list = []
    real_init = plugin_engine.PluginManager.__init__

    def init(self, app, *a, **kw):
        real_init(self, app, installed_root=installed)

    monkeypatch.setattr(plugin_engine.PluginManager, "__init__", init)
    monkeypatch.setattr(plugin_engine.PluginManager, "_enabled_set", lambda self: set(enabled))
    assert _speech_route_available() is False
    enabled.append("tts")
    assert _speech_route_available() is True
    (installed / "tts" / "plugin.toml").unlink()
    assert _speech_route_available() is False


def test_chat_naming_an_unknown_typed_file_no_backend_runs_is_a_clean_422(served, unknown_file):
    client, engines = served
    r = client.post("/v1/chat/completions", json={
        "model": "zz-unknown-file", "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "unknown type" in detail and "localm set-type zz-unknown-file" in detail
    assert "\n" not in detail
    _assert_untouched(engines, "zz-unknown-file")


def test_an_unknown_typed_entry_a_backend_can_run_is_not_refused(tmp_path):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x" * 16)
    gguf = tmp_path / "odd.gguf"
    gguf.write_bytes(b"x" * 16)
    reg = {"snap": {"path": str(folder), "model_type": "unknown"},
           "odd": {"path": str(gguf), "model_type": "unknown"}}
    assert chat_load_refusal("snap", reg) is None
    assert chat_load_refusal("odd", reg) is None


def test_an_unknown_typed_file_is_refused_unless_a_plugin_backend_claims_it(
        monkeypatch, unknown_file):
    from localm.inference import backend_registry
    reg = {"u": {"path": str(unknown_file), "model_type": "unknown"}}
    assert "unknown type" in chat_load_refusal("u", reg)
    monkeypatch.setattr(backend_registry, "backend_for", lambda path: object())
    assert chat_load_refusal("u", reg) is None


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


def test_an_unknown_typed_folder_no_backend_runs_is_refused(tmp_path):
    folder = tmp_path / "not-a-snapshot"
    folder.mkdir()
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    reg = {"u": {"path": str(folder), "model_type": "unknown"}}
    assert "unknown type" in chat_load_refusal("u", reg)


def test_an_unknown_typed_path_that_does_not_exist_is_not_refused(tmp_path):
    reg = {"u": {"path": str(tmp_path / "gone.safetensors"), "model_type": "unknown"}}
    assert chat_load_refusal("u", reg) is None


@pytest.fixture
def refusal_probe(monkeypatch):
    """Records, per call of the file-system check, whether an event loop was
    running on the calling thread."""
    from localm.model_manager import registry as R
    real = R._no_backend_runs_file
    on_loop: list[bool] = []

    def spy(path):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(path)

    monkeypatch.setattr(R, "_no_backend_runs_file", spy)
    return on_loop


def test_get_engine_checks_an_unknown_entry_off_the_event_loop(served, refusal_probe):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.get_engine("zz-unknown-file", load=False))
    assert exc.value.status_code == 422
    assert refusal_probe == [False]


def test_switch_engine_checks_an_unknown_entry_off_the_event_loop(served, refusal_probe):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.switch_engine("zz-unknown-file", hs._engine_factory))
    assert exc.value.status_code == 422
    assert refusal_probe == [False]


def test_gui_load_route_checks_an_unknown_entry_off_the_event_loop(served, refusal_probe):
    async def switch_model(name, **kw):
        return {"status": "loaded", "model": name}

    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "aaa-working")
    with TestClient(app) as client:
        r = client.post("/api/models/load", json={"model": "zz-unknown-file"})
    assert r.status_code == 422
    assert refusal_probe == [False]


def test_a_component_type_needs_no_worker_thread(served, refusal_probe):
    with pytest.raises(HTTPException):
        asyncio.run(hs.get_engine(_name("vae"), load=False))
    assert refusal_probe == []
