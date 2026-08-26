# SPDX-License-Identifier: AGPL-3.0-or-later
"""Idle-unload must keep the Engine object for lazy reload.

Deleting the engine from _engines after unloading it breaks the "the next
inference reloads the model lazily" contract:

  - a direct-path served model (``localm serve <path>``) whose display name is
    NOT in the registry cannot be rebuilt by the default factory, so it 404s
    forever after one idle timeout;
  - a registered model gets rebuilt by the default factory, silently dropping
    the user's original --ctx / --gpu-layers / --device / --mmproj settings.

Keeping the (now-unloaded) object in _engines lets the next request reuse it
with its original constructor settings, and never lose a direct-path model.
"""

import asyncio
import time

from localm.inference import http_server as hs


class FakeEngine:
    def __init__(self, display_name, marker="orig"):
        self.display_name = display_name
        self._loaded = False
        self.active_requests = 0
        self.marker = marker  # proves object identity survives an idle unload

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


def _reset():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None


def _install_loaded(name, marker="orig"):
    eng = FakeEngine(name, marker=marker)
    eng.load()
    hs._engines[name] = eng
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    hs._active_model_name = name
    hs._engine = eng
    hs._inference_sem = hs._inference_sems[name]
    # Idle far in the past so the ttl check fires.
    hs._last_activity_per_model[name] = time.monotonic() - 10_000
    return eng


def test_idle_unload_keeps_engine_object(monkeypatch):
    _reset()
    eng = _install_loaded("served-model")
    try:
        did = asyncio.run(hs._idle_unload_once(ttl=1))
        assert did is True
        # The object is kept (unloaded) for lazy reload, NOT deleted.
        assert "served-model" in hs._engines, "idle-unload deleted the engine (breaks lazy reload)"
        assert hs._engines["served-model"] is eng
        assert hs._engines["served-model"].loaded is False
        # ...but it no longer holds a VRAM/LRU slot while unloaded.
        assert "served-model" not in hs._engines_lru
    finally:
        _reset()


def test_lazy_reload_reuses_kept_object_not_the_factory(monkeypatch):
    """After an idle unload, a new request must reload the SAME kept object (its
    original settings), never fall through to a factory that cannot rebuild a
    direct-path / non-registered model."""
    _reset()
    eng = _install_loaded("served-model", marker="had-user-flags")

    # A factory that FAILS models the direct-path case: the served model's name
    # is not in the registry, so the default factory would raise. If reload
    # reuses the kept object, the factory is never called.
    def exploding_factory(name):
        raise AssertionError("factory must not be called - kept object should be reused")

    monkeypatch.setattr(hs, "_engine_factory", exploding_factory)
    # Empty registry => get_engine routes to the active/loaded engine path.
    monkeypatch.setattr("localm.config.load_registry", lambda: {})

    try:
        asyncio.run(hs._idle_unload_once(ttl=1))
        assert hs._engines["served-model"].loaded is False

        reloaded = asyncio.run(hs.get_engine("served-model"))
        assert reloaded is eng, "reload must reuse the kept object"
        assert reloaded.marker == "had-user-flags", "original settings preserved"
        assert reloaded.loaded is True
    finally:
        _reset()
