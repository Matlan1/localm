# SPDX-License-Identifier: AGPL-3.0-or-later
"""A registered vision projector (model_type "mmproj") is not a model.

It is not listed by GET /v1/models, and a request that names it is refused with
a 422 that names the chat model it belongs to, BEFORE the resident model is
evicted or any load is attempted.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.model_manager.registry import projector_owners, projector_refusal
from localm.plugins.gui.web import attach_gui
from tests.conftest import probe_double

PROJECTOR = "Ministral-3-3B-Instruct-2512-BF16-mmproj"
OWNER = "Ministral-3-3B-Instruct-2512-Q4_K_M"


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.can_be_multimodal = False
        self.load_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 10

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **kw):
        yield "hi from " + self.display_name


def _registry():
    return {
        "aaa-working": {"path": "models/aaa-working.gguf", "source": "local",
                        "model_type": "llm"},
        OWNER: {"path": "models/ministral/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
                "source": "hf:x/y", "model_type": "llm"},
        PROJECTOR: {"path": "models/ministral/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf",
                    "source": "hf:x/y", "model_type": "mmproj", "architecture": "clip"},
    }


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


def _assert_working_model_untouched(engines):
    assert PROJECTOR not in engines, "a projector must never get an engine built for it"
    assert hs._engines["aaa-working"].loaded is True
    assert hs._active_model_name == "aaa-working"


def test_v1_models_does_not_list_the_projector(served):
    client, _ = served
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert PROJECTOR not in ids
    assert ids == sorted(["aaa-working", OWNER])


def test_chat_naming_the_projector_is_refused_and_keeps_the_loaded_model(served):
    client, engines = served
    r = client.post("/v1/chat/completions", json={
        "model": PROJECTOR, "stream": False,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422
    assert OWNER in r.json()["detail"]
    _assert_working_model_untouched(engines)


def test_streaming_chat_naming_the_projector_is_refused(served):
    client, engines = served
    r = client.post("/v1/chat/completions", json={
        "model": PROJECTOR, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422
    _assert_working_model_untouched(engines)


def test_models_load_naming_the_projector_is_refused(served):
    client, engines = served
    r = client.post("/v1/models/load", params={"model": PROJECTOR})
    assert r.status_code == 422
    assert OWNER in r.json()["detail"]
    _assert_working_model_untouched(engines)


def test_get_engine_without_loading_refuses_the_projector(served):
    _, engines = served
    import asyncio
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.get_engine(PROJECTOR, load=False))
    assert exc.value.status_code == 422
    _assert_working_model_untouched(engines)


def test_switch_engine_refuses_the_projector_before_cancelling_a_pending_switch(served):
    _, engines = served
    import asyncio
    from fastapi import HTTPException
    hs._switch_desired = "aaa-working"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(hs.switch_engine(PROJECTOR, hs._engine_factory))
    assert exc.value.status_code == 422
    assert hs._switch_desired == "aaa-working"
    _assert_working_model_untouched(engines)


def test_a_chat_model_beside_the_projector_still_loads(served):
    client, _ = served
    r = client.post("/v1/models/load", params={"model": OWNER})
    assert r.status_code == 200
    assert r.json()["status"] == "loaded"


def test_gui_load_route_refuses_the_projector_with_422(monkeypatch):
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
        r = client.post("/api/models/load", json={"model": PROJECTOR})
        ok = client.post("/api/models/load", json={"model": OWNER})
    assert r.status_code == 422
    assert OWNER in r.json()["detail"]
    assert switched == [OWNER]
    assert ok.status_code == 200


def test_owner_is_found_by_recorded_mmproj_over_folder_siblings():
    reg = _registry()
    reg["other-in-same-folder"] = {
        "path": "models/ministral/other.gguf", "model_type": "llm"}
    reg[OWNER]["mmproj"] = reg[PROJECTOR]["path"]
    assert projector_owners(PROJECTOR, reg) == [OWNER]


def test_owner_falls_back_to_chat_models_in_the_same_folder():
    reg = _registry()
    reg["elsewhere"] = {"path": "models/elsewhere/x.gguf", "model_type": "llm"}
    assert projector_owners(PROJECTOR, reg) == [OWNER]


def test_refusal_is_none_for_chat_models_and_unknown_names():
    reg = _registry()
    assert projector_refusal(OWNER, reg) is None
    assert projector_refusal("not-registered", reg) is None
    assert projector_refusal(PROJECTOR, reg) is not None


def test_refusal_without_a_known_owner_still_refuses():
    reg = {PROJECTOR: {"path": "models/lonely/p-mmproj.gguf", "model_type": "mmproj"}}
    msg = projector_refusal(PROJECTOR, reg)
    assert msg is not None and "vision projector" in msg and "belongs to" not in msg


@pytest.mark.parametrize("bad", [None, "x", 5, [], {"model_type": None}])
def test_malformed_entries_are_not_projectors(bad):
    assert projector_refusal("n", {"n": bad}) is None
    assert projector_owners("n", {"n": bad}) == []
