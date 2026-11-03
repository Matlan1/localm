# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat request that names a model that is not loaded, after the user switched
to another model, makes room the way a switch does.

The model the user switched to is idle and holds the VRAM the named model needs:
it is evicted and the named model loads on the GPU, instead of the named model
loading beside it with whatever VRAM is left.
"""

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double

GIB = 1024 ** 3


class FakeEngine:
    def __init__(self, name, vram):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.model_path = f"Z:/models/{name}.gguf"
        self._vram = vram
        self.held = 0

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, cancel):
        pass

    def load(self):
        self._loaded = True
        self.held = self._vram["want"][self.display_name]
        self._vram["free"] -= self.held

    def unload(self):
        if self._loaded:
            self._vram["free"] += self.held
        self._loaded = False
        self.held = 0

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 10

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **gen_kwargs):
        yield "Hello from " + self.display_name


@pytest.fixture
def two_models(monkeypatch):
    """Model "a" resident and active, holding 13 of 16 GB; model "b" needs 9."""
    vram = {"free": 16 * GIB, "want": {"a": 13 * GIB, "b": 9 * GIB}}
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name, vram))

    reg = {"a": {"path": "Z:/models/a.gguf", "source": "local"},
           "b": {"path": "Z:/models/b.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_capacity",
                        probe_double(lambda: {"free": vram["free"], "total": 16 * GIB}))
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (True, before_bytes))
    monkeypatch.setattr("localm.inference.residency.model_footprint_bytes",
                        lambda path: vram["want"][str(path).split("/")[-1][:-5]])
    monkeypatch.setattr(hs, "_engine_factory", factory)

    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None

    a = factory("a")
    a.load()
    app = hs.create_app(a)
    return TestClient(app), engines, vram


def _chat(client, model, **extra):
    return client.post("/v1/chat/completions",
                       json={"model": model, "stream": False, **extra,
                             "messages": [{"role": "user", "content": "hi"}]})


def test_named_unloaded_model_evicts_the_idle_active_model(two_models):
    client, engines, vram = two_models

    r = _chat(client, "b", pin_model=False)

    assert r.status_code == 200, r.text
    assert engines["b"].loaded
    assert not engines["a"].loaded, "the idle model that held the VRAM was not evicted"
    assert vram["free"] == 16 * GIB - 9 * GIB
    assert hs._active_model_name == "b"


def test_a_load_beside_a_pinned_model_says_so(two_models, monkeypatch):
    import asyncio

    client, engines, vram = two_models
    monkeypatch.setattr("localm.config.load_config",
                        lambda *a, **k: {"pinned_models": ["a"]})
    said = []

    asyncio.run(hs.get_engine("b", on_status=said.append))

    assert engines["a"].loaded and engines["b"].loaded
    assert hs.LOADING_BESIDE_STATUS.format(model="b", resident="'a'") in said


@pytest.mark.parametrize("activate, active_after", [(False, "a"), (True, "b")])
def test_reloading_a_resident_model_keeps_the_active_one_unless_asked(
        two_models, activate, active_after):
    import asyncio

    client, engines, vram = two_models
    vram["want"]["b"] = 2 * GIB
    asyncio.run(hs.switch_engine("b", None, activate=False))
    assert engines["a"].loaded and engines["b"].loaded
    assert hs._active_model_name == "a"

    async def reload_b():
        await hs.unload_one_model("b")
        return await hs.switch_engine("b", None, activate=activate)

    res = asyncio.run(reload_b())

    assert res["status"] == "loaded", res
    assert engines["a"].loaded and engines["b"].loaded
    assert hs._active_model_name == active_after


def test_the_load_beside_status_streams_its_code_and_args(two_models, monkeypatch):
    import json

    client, engines, vram = two_models
    monkeypatch.setattr("localm.config.load_config",
                        lambda *a, **k: {"pinned_models": ["a"]})

    r = client.post("/v1/chat/completions",
                    json={"model": "b", "stream": True, "pin_model": True,
                          "messages": [{"role": "user", "content": "hi"}]})

    assert r.status_code == 200, r.text
    deltas = [json.loads(line[6:])["choices"][0]["delta"]
              for line in r.text.splitlines()
              if line.startswith("data: {")]
    beside = [d for d in deltas if " beside " in (d.get("status") or "")]
    assert beside, [d.get("status") for d in deltas]
    assert beside[0]["status"] == hs.LOADING_BESIDE_STATUS.format(model="b", resident="'a'")
    assert beside[0]["status_code"] == "loading_beside"
    assert beside[0]["status_args"] == {"model": "b", "resident": "'a'"}
    assert all(d.get("status_args") is None for d in deltas if d is not beside[0])


def test_a_load_that_evicts_does_not_claim_to_load_beside(two_models):
    import asyncio

    client, engines, vram = two_models
    said = []

    asyncio.run(hs.get_engine("b", on_status=said.append))

    assert engines["b"].loaded and not engines["a"].loaded
    assert not [s for s in said if " beside " in s]
