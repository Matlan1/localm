# SPDX-License-Identifier: AGPL-3.0-or-later
"""_default_model_name is write-once at startup; an unnamed request after an
eviction must resolve to the model actually last in use, not the one the
process booted with - and GET /health must agree with that same resolution
instead of reporting a flat 503 during a state chat can already recover from
by itself.

switch_engine updates _active_model_name on every load but not
_default_model_name, so get_engine's fallback (_active_model_name or
_default_model_name) reverts to the STARTUP model once an eviction clears the
active pointer. _default_model_name has exactly two assignments in the whole
tree, both inside create_app.

FIXTURE PREMISE: the failing case needs a server that switched to a SECOND
model before the eviction. A fixture that only ever loads one model cannot
distinguish "resolved to the right model" from "resolved to the only model
there is" - both read identically. These tests switch to model-b before
evicting, so a fix that falls back to model-a instead cannot pass.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from tests.conftest import probe_double


class FakeEngine:
    """Stands in for the model backend only."""

    def __init__(self, display_name: str):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.supports_images = False
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


def _reset_server_state():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._last_active_model_name = None
    hs._engine = None
    hs._inference_sem = None


@pytest.fixture
def two_model_server(monkeypatch):
    """Boots with model-a (the startup/default model), then switches to
    model-b before the eviction under test."""
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    monkeypatch.setattr(
        "localm.config.load_registry",
        lambda: {"model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
                 "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3,
                                      "total": 16 * 1024 ** 3}))
    monkeypatch.setattr(hs, "_engine_factory", factory)

    _reset_server_state()

    startup = factory("model-a")
    startup.load()
    app = hs.create_app(startup)

    asyncio.run(hs.switch_engine("model-b", hs._engine_factory, preempt=False))
    assert hs._active_model_name == "model-b", "test premise: switch must succeed"
    assert hs._default_model_name == "model-a", (
        "test premise: _default_model_name must still name the STARTUP "
        "model - it is write-once, that is the whole defect")

    return TestClient(app), engines


def _evict():
    """The real eviction, through the real function."""
    result = asyncio.run(hs.unload_all_models())
    assert hs._active_model_name is None, (
        "test premise: eviction must clear the active model pointer")
    return result


def test_get_engine_resolves_switched_model_not_startup_model(two_model_server):
    _client, engines = two_model_server
    _evict()
    assert not engines["model-a"].loaded
    assert not engines["model-b"].loaded

    engine = asyncio.run(hs.get_engine(""))

    assert engine.display_name == "model-b", (
        f"unnamed reload resolved to {engine.display_name!r} - the STARTUP "
        f"model - instead of model-b, the model actually in use before the "
        f"eviction")
    assert engines["model-b"].loaded, "the last-active model was never reloaded"
    assert not engines["model-a"].loaded, (
        "the startup model was reloaded instead of the last-active one")


def test_health_reports_switched_model_not_startup_model(two_model_server):
    client, engines = two_model_server
    _evict()

    r = client.get("/health")

    assert r.status_code == 200, (
        f"health reported unrecoverable for a model still retained for lazy "
        f"reload: {r.status_code} {r.text}")
    body = r.json()
    assert body["model"] == "model-b", (
        f"health named {body['model']!r} as the recoverable model, not "
        f"model-b - the one actually in use before the eviction")
    assert body["loaded"] is False
    assert not engines["model-a"].loaded and not engines["model-b"].loaded, (
        "GET /health must never itself trigger a load - it only reports")


def test_health_recoverable_after_plain_eviction_no_switch(monkeypatch):
    """The minimal case, no switch involved: /health must not 503 after an
    eviction when the very next chat turn reloads the model successfully."""
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    monkeypatch.setattr(
        "localm.config.load_registry",
        lambda: {"model-a": {"path": "Z:/models/model-a.gguf", "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr(hs, "_engine_factory", factory)

    _reset_server_state()
    startup = factory("model-a")
    startup.load()
    client = TestClient(hs.create_app(startup))

    _evict()

    r = client.get("/health")

    assert r.status_code == 200, (
        f"a model kept in _engines for lazy reload was reported as an "
        f"unrecoverable 503: {r.status_code} {r.text}")
    body = r.json()
    assert body["model"] == "model-a"
    assert body["loaded"] is False


def test_health_still_503_when_nothing_is_recoverable(monkeypatch):
    """A server with no model loaded and none ever configured is genuinely
    unserveable, and /health must still say so plainly rather than inventing a
    fake 200."""
    monkeypatch.setattr("localm.config.load_registry", lambda: {})
    _reset_server_state()

    client = TestClient(hs.create_app(None))
    r = client.get("/health")

    assert r.status_code == 503
    assert "No engine initialised" in r.text
