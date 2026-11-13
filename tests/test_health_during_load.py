# SPDX-License-Identifier: AGPL-3.0-or-later
"""/health while a model is being loaded and no engine is active (a routed
load after the active model was evicted) reports the loading model with 200,
not 503 "No engine initialised"."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs


class _BlockingEngine:
    def __init__(self, name: str, release: threading.Event, entered: threading.Event):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
        self._release = release
        self._entered = entered

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._entered.set()
        self._release.wait(20)
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    saved = (dict(hs._engines), list(hs._engines_lru), dict(hs._inference_sems),
             hs._active_model_name, hs._last_active_model_name, hs._engine,
             hs._inference_sem, hs._default_model_name, hs._engine_factory)
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._default_model_name = None
    hs._loads_in_flight.clear()
    yield
    hs._loads_in_flight.clear()
    (engines, lru, sems, hs._active_model_name, hs._last_active_model_name, hs._engine,
     hs._inference_sem, hs._default_model_name, hs._engine_factory) = saved
    hs._engines.clear()
    hs._engines.update(engines)
    hs._engines_lru[:] = lru
    hs._inference_sems.clear()
    hs._inference_sems.update(sems)


def test_health_reports_the_model_being_loaded(clean_state):
    client = TestClient(hs.create_app(None))
    assert client.get("/health").status_code == 503
    hs._loads_in_flight["vision-model"] = 1
    r = client.get("/health")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "model": "vision-model", "loaded": False,
                        "loading": True}


def test_switch_engine_registers_its_load_until_it_returns(clean_state, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    engine = _BlockingEngine("vision-model", release, entered)

    async def no_refusal(*a, **k):
        return None

    monkeypatch.setattr(hs, "_refuse_non_chat_model", no_refusal)

    async def scenario():
        task = asyncio.ensure_future(
            hs.switch_engine("vision-model", lambda n: engine, preempt=False,
                             activate=False))
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, entered.wait, 10)
        during = hs.loading_model_name()
        release.set()
        result = await task
        return during, result, hs.loading_model_name()

    during, result, after = asyncio.run(scenario())
    assert during == "vision-model"
    assert result["status"] == "loaded"
    assert after is None
