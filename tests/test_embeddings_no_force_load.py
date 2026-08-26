# SPDX-License-Identifier: AGPL-3.0-or-later
"""/v1/embeddings must not force-load the chat model.

The embeddings endpoint calls get_engine(req.model), which LOADS the model via
switch_engine (with eviction). But a GGUF backend (can_embed=False, the default
runtime) embeds via a small DEDICATED embedder and never needs the multi-GB chat
model resident, so loading it - and, under VRAM pressure, evicting the active
chat model - is pure waste.
"""

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeBackend:
    can_embed = False   # GGUF-style: cannot embed its own weights


class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self._backend = FakeBackend()
        self.load_calls = 0
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, ev):
        pass

    def embed(self, texts):
        # Mirrors Engine.embed for can_embed=False: uses the dedicated embedder,
        # never loads this (chat) model.
        return [[0.1, 0.2, 0.3] for _ in texts]

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def setup(monkeypatch):
    reg = {"model-y": {"path": "models/model-y.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "h"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda n: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    engines = {"model-y": FakeEngine("model-y")}
    monkeypatch.setattr(hs, "_engine_factory", lambda n: engines[n])
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    return engines


def test_embeddings_do_not_load_a_gguf_chat_model(setup):
    engines = setup
    app = hs.create_app(None)
    client = TestClient(app)

    r = client.post("/v1/embeddings", json={"model": "model-y", "input": "hello world"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 1 and len(data[0]["embedding"]) == 3

    assert engines["model-y"].load_calls == 0, (
        "embeddings force-loaded the GGUF chat model; it should embed via the "
        "dedicated embedder without loading the chat model")
    assert not engines["model-y"].loaded
