# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, display_name):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self.model_path = f"Z:/models/{display_name}.gguf"

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def count_tokens(self, text):
        return len(text.split())

    def count_messages_tokens(self, messages):
        return 100

    def context_capacity(self):
        return 4096

    def chat_stream(self, messages, **gen_kwargs):
        yield "Hello from " + self.display_name

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def setup_multi_model(monkeypatch):
    fake_registry = {
        "model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
        "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"},
        "model-c": {"path": "Z:/models/model-c.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    
    monkeypatch.setattr(hs, "_engine_factory", lambda name: FakeEngine(name))
    
    # Clear active server state
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None


def test_concurrent_loading_and_routing(setup_multi_model):
    app = hs.create_app(None)
    client = TestClient(app)
    
    # 1. Query model-a
    r = client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "Hello from model-a" in r.text
    
    # 2. Query model-b
    r = client.post("/v1/chat/completions", json={
        "model": "model-b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "Hello from model-b" in r.text
    
    # Both should be resident in memory
    assert "model-a" in hs._engines
    assert "model-b" in hs._engines
    assert hs._engines["model-a"].loaded
    assert hs._engines["model-b"].loaded
    
    # Active model should be model-b (most recently used)
    assert hs._active_model_name == "model-b"


def test_list_models_status(setup_multi_model):
    app = hs.create_app(None)
    client = TestClient(app)
    
    # Load model-a first
    client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
    })
    
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    
    model_a = next(m for m in data if m["id"] == "model-a")
    model_b = next(m for m in data if m["id"] == "model-b")
    
    assert model_a["loaded"] is True
    assert model_b["loaded"] is False


def test_lru_eviction(setup_multi_model, monkeypatch):
    app = hs.create_app(None)
    client = TestClient(app)
    
    # 4.8 GB per model. With 10 GB total, loading a second model exceeds VRAM limit.
    def dynamic_vram():
        loaded_count = sum(1 for e in hs._engines.values() if e.loaded)
        free = (10 * 1024 ** 3) - int(loaded_count * 4.8 * 1024 ** 3)
        return {"free": free, "total": 10 * 1024 ** 3}
        
    monkeypatch.setattr("localm.discover.vram_info", probe_double(dynamic_vram))
    
    # Load model-a
    r = client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "model-a" in hs._engines
    
    # Load model-b. This must trigger eviction of model-a.
    r = client.post("/v1/chat/completions", json={
        "model": "model-b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    
    assert "model-b" in hs._engines
    assert "model-a" not in hs._engines


def test_active_requests_protection_from_eviction(setup_multi_model, monkeypatch):
    app = hs.create_app(None)
    client = TestClient(app)

    # 4.8 GB per model. With 10 GB total, loading a second model exceeds VRAM limit.
    def dynamic_vram():
        loaded_count = sum(1 for e in hs._engines.values() if e.loaded)
        free = (10 * 1024 ** 3) - int(loaded_count * 4.8 * 1024 ** 3)
        return {"free": free, "total": 10 * 1024 ** 3}

    monkeypatch.setattr("localm.discover.vram_info", probe_double(dynamic_vram))

    # Load model-a
    r = client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200

    # Simulate active requests on model-a
    hs._engines["model-a"].active_requests = 1

    # switch_engine falls through to the backend's own sizing once local and
    # cooperative eviction are exhausted, so model-b's engine is made to fail its
    # load outright: eviction protection is then the only thing between the
    # request and a 503.
    def _failing_engine_factory(name):
        engine = FakeEngine(name)
        engine.load = lambda: (_ for _ in ()).throw(
            RuntimeError("VRAM exhausted: cannot fit even at 0 GPU layers"))
        return engine
    monkeypatch.setattr(hs, "_engine_factory", _failing_engine_factory)

    r = client.post("/v1/chat/completions", json={
        "model": "model-b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 503
    assert "VRAM exhausted" in r.text
    assert hs._engines["model-a"].loaded, "the busy model-a must not be evicted"

    hs._engines["model-a"].active_requests = 0


def test_localm_resolves_to_active(setup_multi_model):
    app = hs.create_app(None)
    client = TestClient(app)
    
    # Load model-b first
    client.post("/v1/chat/completions", json={
        "model": "model-b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    
    # Sending request to "localm" should route to model-b
    r = client.post("/v1/chat/completions", json={
        "model": "localm",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert "Hello from model-b" in r.text


def test_unregistered_model_404(setup_multi_model):
    app = hs.create_app(None)
    client = TestClient(app)
    
    r = client.post("/v1/chat/completions", json={
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404
    assert "is not registered" in r.text
    assert "model-a" in r.text


def test_empty_model_400(setup_multi_model):
    app = hs.create_app(None)
    client = TestClient(app)
    
    r = client.post("/v1/chat/completions", json={
        "model": "",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400
    assert "Model parameter is required" in r.text
