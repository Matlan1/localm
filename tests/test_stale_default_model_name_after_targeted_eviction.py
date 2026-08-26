# SPDX-License-Identifier: AGPL-3.0-or-later
"""_last_active_model_name must be updated by every eviction path.

unload_one_model and _idle_unload_once clear _active_model_name to None under the
same "retain the Engine in _engines for lazy reload" contract as
unload_all_models, so an unnamed request after either of those eviction paths must
not resolve to a stale model name.

FIXTURE PREMISE: the failing case needs a server that switched to a SECOND model
before the eviction under test. A fixture that only ever loads one model cannot
distinguish "resolved to the right model" from "resolved to the only model there
is" - both read identically."""

from __future__ import annotations

import asyncio
import time

import pytest

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
    """Boots with model-a (the startup/default model), then switches to model-b
    before the eviction under test. Plenty of free VRAM means the switch does not
    evict model-a, so both stay resident with model-b active."""
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
    hs.create_app(startup)

    asyncio.run(hs.switch_engine("model-b", hs._engine_factory, preempt=False))
    assert hs._active_model_name == "model-b", "test premise: switch must succeed"

    return engines


def test_get_engine_resolves_switched_model_after_unload_one_model(two_model_server):
    """Evicting the ACTIVE model by name with unload_one_model(name) - leaving
    the background model-a resident and untouched, since unload_one_model only
    touches its target - must still leave model-b resolvable by an unnamed
    request, exactly like unload_all_models."""
    engines = two_model_server
    assert engines["model-a"].loaded, "test premise: model-a stays resident (plenty of VRAM)"

    result = asyncio.run(hs.unload_one_model("model-b"))

    assert result["was_active"] is True, (
        "test premise: model-b must be the active model unload_one_model clears")
    assert hs._active_model_name is None, (
        "test premise: unload_one_model must clear the active pointer for the "
        "model it just unloaded")
    assert not engines["model-b"].loaded

    engine = asyncio.run(hs.get_engine(""))

    assert engine.display_name == "model-b", (
        f"unnamed reload resolved to {engine.display_name!r} instead of "
        f"model-b, the model actually in use before unload_one_model evicted "
        f"it - a stale (or absent) _last_active_model_name fell back to "
        f"model-a, the untouched background model, instead")
    assert engines["model-b"].loaded, "the last-active model was never reloaded"


def test_get_engine_resolves_switched_model_after_idle_unload(two_model_server):
    """When the idle-evicted model was the LAST one resident,
    _idle_unload_once(ttl)'s own partial-recovery fallback (_engines_lru[-1])
    finds nothing and _active_model_name goes to None; the model must still be
    recoverable via _last_active_model_name.

    model-a is unloaded directly rather than through unload_one_model, so
    _engines_lru is genuinely empty once model-b, the only remaining resident
    model, is idle-evicted."""
    engines = two_model_server
    engines["model-a"].unload()
    hs._engines_lru.remove("model-a")
    hs._last_activity_per_model["model-b"] = time.monotonic() - 1000

    unloaded = asyncio.run(hs._idle_unload_once(60))

    assert unloaded is True, "test premise: the idle check must actually evict model-b"
    assert hs._active_model_name is None, (
        "test premise: idle-unloading the only resident model must clear the "
        "active pointer - the LRU fallback has nothing left to fall back to")
    assert not engines["model-b"].loaded

    engine = asyncio.run(hs.get_engine(""))

    assert engine.display_name == "model-b", (
        f"unnamed reload resolved to {engine.display_name!r} instead of "
        f"model-b, the model actually in use before the idle-unload evicted "
        f"it - a stale (or absent) _last_active_model_name fell back to "
        f"model-a, the already-unloaded startup model, instead")
    assert engines["model-b"].loaded, "the last-active model was never reloaded"
    assert not engines["model-a"].loaded, (
        "the already-unloaded startup model was reloaded instead of the "
        "last-active one")
