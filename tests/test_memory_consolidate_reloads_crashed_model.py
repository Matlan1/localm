# SPDX-License-Identifier: AGPL-3.0-or-later
"""`POST /api/memory/consolidate` reloads a model whose worker died, as a chat
request does, instead of answering "Load a model first".

The engine is the server's resident one: its worker dies mid-stream (the stream
raises and the backend drops its model), so ``loaded`` turns False while the
server still has the model selected. The route runs over the real ``get_engine``
reload path with a fake model backend.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.plugins.builtin.memory import plug
from tests.conftest import probe_double

_FACTS = json.dumps({"facts": [{"fact": "User's name is Ada", "confidence": 0.9}]})
_NAME = "model-a"


class FakeEngine:
    """Model backend stand-in: the first stream after ``crash_next`` is set dies
    like a killed worker and leaves the model unloaded."""

    def __init__(self, display_name: str, load_error: Exception | None = None):
        self.display_name = display_name
        self.model_path = f"Z:/models/{display_name}.gguf"
        self.load_error = load_error
        self._loaded = False
        self.active_requests = 0
        self.load_calls = 0
        self.crash_next = False
        self.streamed_while_unloaded = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass

    def chat_stream(self, messages, **kw):
        if self.crash_next:
            self.crash_next = False
            self._loaded = False
            raise RuntimeError("The model worker died")
        if not self._loaded:
            self.streamed_while_unloaded += 1
        yield _FACTS


class _Host:
    """PluginManager.inference_engine: the server's active resident engine."""

    def mount_router(self, router):
        pass

    def register_chat_hook(self, phase, fn, priority=0):
        pass

    def engine(self):
        name = hs._active_model_name
        return hs._engines.get(name) if name else None


@pytest.fixture
def server(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_memory_root", lambda: tmp_path / "memory")
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    rows = [{"type": "user", "data": {"content": "my name is Ada and I prefer metric units"}},
            {"type": "llm", "data": {"content": "Noted, Ada!"}}]
    (sdir / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {_NAME: {"path": f"Z:/models/{_NAME}.gguf",
                                         "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))

    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    for attr, value in (("_engines", {}), ("_engines_lru", []), ("_inference_sems", {}),
                        ("_last_activity_per_model", {}), ("_active_model_name", None),
                        ("_last_active_model_name", None), ("_engine", None),
                        ("_inference_sem", None), ("_engine_factory", factory)):
        monkeypatch.setattr(hs, attr, value)

    resident = factory(_NAME)
    resident.load()
    hs._engines[_NAME] = resident
    hs._engines_lru.append(_NAME)
    hs._inference_sems[_NAME] = hs.InferenceGate()
    hs._active_model_name = _NAME
    hs._engine = resident
    hs._inference_sem = hs._inference_sems[_NAME]

    monkeypatch.setattr(plug, "_HOST", None)
    plug.register(_Host())
    app = FastAPI()
    app.include_router(plug._router)
    return TestClient(app), resident, engines


def _kill_worker(eng: FakeEngine) -> None:
    eng.crash_next = True
    with pytest.raises(RuntimeError):
        list(eng.chat_stream([{"role": "user", "content": "hi"}]))
    assert not eng.loaded, "the simulated worker death did not unload the model"
    assert hs._active_model_name == _NAME, "test premise: the model stays selected"


def test_synthesize_reloads_a_model_whose_worker_died(server):
    client, eng, _ = server
    _kill_worker(eng)
    loads_before = eng.load_calls

    r = client.post("/api/memory/consolidate")

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
    assert eng.load_calls == loads_before + 1, "the route did not reload the model"
    assert eng.streamed_while_unloaded == 0
    assert eng.loaded
    assert any("Ada" in rec.text for rec in plug._chat_store().all())


def test_synthesize_does_not_reload_a_loaded_model(server):
    client, eng, _ = server
    loads_before = eng.load_calls

    r = client.post("/api/memory/consolidate")

    assert r.status_code == 200, r.text
    assert eng.load_calls == loads_before


def test_a_failed_reload_is_an_error_and_saves_nothing(server):
    client, eng, _ = server
    _kill_worker(eng)
    eng.load_error = RuntimeError("out of VRAM")

    r = client.post("/api/memory/consolidate")

    assert r.status_code == 503, r.text
    assert "Load a model first" not in r.text
    assert plug._chat_store().all() == []
    assert eng.active_requests == 0


def test_a_model_unloaded_on_request_is_not_loaded_behind_the_users_back(server):
    client, eng, _ = server
    asyncio.run(hs.unload_all_models())
    assert hs._active_model_name is None, "test premise: eviction clears the selection"
    loads_before = eng.load_calls

    r = client.post("/api/memory/consolidate")

    assert r.status_code == 503
    assert "Load a model first" in r.text
    assert eng.load_calls == loads_before
