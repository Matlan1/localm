# SPDX-License-Identifier: AGPL-3.0-or-later
"""VRAM eviction must not crash when the victim is removed during its unload.

switch_engine picks an idle victim, then `await`s its unload (an executor call -
an event-loop yield). During that yield a CONCURRENT remover (another API load
that picked the SAME idle victim - get_engine loads with preempt=False so they
coexist - or an idle/explicit unload) may have already dropped the victim from
`_engines`/`_engines_lru`. The removal step then used a bare
`del _engines[evict_name]` / `_engines_lru.remove(evict_name)`, which raises
KeyError/ValueError and surfaces as an HTTP 500 with a traceback (plus a leaked
`_inference_sems` entry). This pins the guarded removal.

The concurrent removal is simulated deterministically by the victim's own
unload() dropping itself from the registry (exactly what a racing remover does
during the same await window) and freeing enough VRAM for the incoming load.
"""

import asyncio

import pytest

from localm.inference import http_server as hs


class _IncomingEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, ev):
        pass

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False


class _RacyVictim:
    """An idle victim whose unload() ALSO removes it from the registry (like a
    concurrent remover during the same await window) and frees VRAM."""

    def __init__(self, name, vram_state):
        self.display_name = name
        self._loaded = True
        self.active_requests = 0
        self._vram = vram_state

    @property
    def loaded(self):
        return self._loaded

    def unload(self):
        self._loaded = False
        # Simulate a concurrent removal landing during this (executor) unload...
        hs._engines.pop(self.display_name, None)
        if self.display_name in hs._engines_lru:
            hs._engines_lru.remove(self.display_name)
        # ...and the freed VRAM so the incoming load now fits and the loop breaks.
        self._vram["free"] = 8 * 1024 ** 3


@pytest.fixture
def evicting(monkeypatch):
    vram = {"free": 3 * 1024 ** 3}   # below the ~5.8 GB the incoming load needs
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": vram["free"], "total": 16 * 1024 ** 3})
    monkeypatch.setattr("localm.discover.gpu_split_shortfall", lambda need: [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    reg = {"victim": {"path": "models/victim.gguf", "source": "local"},
           "incoming": {"path": "models/incoming.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "hint"))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return vram


def test_eviction_survives_victim_removed_during_unload(evicting):
    vram = evicting
    victim = _RacyVictim("victim", vram)
    hs._engines["victim"] = victim
    hs._engines_lru.append("victim")
    hs._inference_sems["victim"] = asyncio.Semaphore(1)
    hs._active_model_name = "victim"

    incoming = _IncomingEngine("incoming")
    from unittest.mock import patch
    with patch.object(hs, "_engine_factory", lambda n: incoming):
        # Without the guarded removal this raises KeyError (-> HTTP 500); with it
        # the load completes.
        engine = asyncio.run(hs.get_engine("incoming"))

    assert engine is incoming and incoming.loaded, "incoming model should have loaded"
    assert "victim" not in hs._engines, "victim was evicted"
    assert "victim" not in hs._inference_sems, "victim's semaphore must not leak"
