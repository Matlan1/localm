# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GGUF model's context window never grows past the window it was trained
for (unless its starting window is already larger), and a refusal for an
over-long prompt names the number its reason is about. A load whose GPU
layers setting puts no layer on the GPU needs no VRAM, so it never evicts a
model to make VRAM room, and ``GET /health`` reports the model the next
unnamed request reloads instead of 503.

The context ceilings come from real GGUF backends over byte-built model
files; switch_engine runs against the simulated box of the transition-matrix
rows; /health and the chat route run in the real app."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference import switch_admission as sa
from localm.inference.backends.gguf import GgufBackend
from tests.test_context_capacity_guard import _EngineWithBackend
from tests.test_context_window_growth import _GrowableBackend, _post_chat
from tests.test_eviction_victim_race import GatedEngine
from tests.test_moe_auto_placement import N_CTX, _moe_model, _Vram
from tests.test_moe_context_first import _trained_moe
from tests.test_switch_engine_transition_matrix import (  # noqa: F401
    GB, _fresh_switch_state, _install, _registered_anywhere, _seat)
from tests.test_vram_eviction_safety import _knobs

TRAINED = 8192


def _backend(path, **kw):
    kw.setdefault("n_gpu_layers", 99)
    kw.setdefault("n_gpu_layers_auto", True)
    kw.setdefault("ctx_auto", True)
    kw.setdefault("n_ctx_max", 16384)
    kw.setdefault("n_ctx", N_CTX)
    return GgufBackend(str(path), **kw)


class TestTheCeilingStopsAtTheTrainedWindow:
    def test_the_automatic_ceiling_is_held_to_the_trained_window(self, tmp_path):
        b = _backend(_trained_moe(tmp_path, TRAINED))
        with _Vram(10 ** 13):
            assert b._auto_ctx_sized()[0] > TRAINED, "the memory figure alone is larger"
            assert b._effective_ctx_max() == TRAINED

    def test_no_maximum_is_held_to_the_trained_window_too(self, tmp_path):
        b = _backend(_trained_moe(tmp_path, TRAINED), n_ctx_max=0)
        with _Vram(10 ** 13):
            assert b._effective_ctx_max() == TRAINED

    def test_a_fixed_ceiling_above_it_is_held_and_one_below_it_kept(self, tmp_path):
        above = _backend(_trained_moe(tmp_path, TRAINED), ctx_auto=False, n_ctx_max=65536)
        assert above._effective_ctx_max() == TRAINED
        below = _backend(_trained_moe(tmp_path, TRAINED), ctx_auto=False, n_ctx_max=4096)
        assert below._effective_ctx_max() == 4096
        unlimited = _backend(_trained_moe(tmp_path, TRAINED), ctx_auto=False, n_ctx_max=0)
        assert unlimited._effective_ctx_max() == TRAINED

    def test_a_starting_window_above_the_trained_one_is_kept(self, tmp_path):
        b = _backend(_trained_moe(tmp_path, TRAINED), ctx_auto=False, n_ctx_max=0,
                     n_ctx=16384)
        assert b._effective_ctx_max() == 16384

    def test_an_unknown_trained_window_leaves_the_ceiling_alone(self, tmp_path):
        b = _backend(_moe_model(tmp_path), ctx_auto=False, n_ctx_max=0)
        assert b._effective_ctx_max() == 0
        assert _backend(_moe_model(tmp_path, "b.gguf"), ctx_auto=False,
                        n_ctx_max=65536)._effective_ctx_max() == 65536


class TestTheRefusalNamesItsOwnNumber:
    def test_the_reason_only_when_the_window_is_that_limit(self):
        class Limited:
            def context_reserve_limit(self):
                return TRAINED, "trained"
        assert hs.context_limit_reason(Limited(), TRAINED) == "trained"
        assert hs.context_limit_reason(Limited(), 65536) is None
        assert hs.context_limit_reason(Limited(), 4096) is None
        assert hs.context_limit_reason(object(), TRAINED) is None

    def test_a_window_past_the_trained_one_is_not_called_trained(self):
        """The window holds 200 tokens while the model was trained for 150:
        the refusal names the 200-token window without calling it trained."""

        class Wide(_GrowableBackend):
            def __init__(self):
                super().__init__(capacity=200, limit=(150, "trained"))

        backend = Wide()
        client = TestClient(hs.create_app(_EngineWithBackend(backend)))
        resp = _post_chat(client, "x" * 400)
        assert resp.status_code == 413, resp.text
        detail = resp.json()["detail"]
        assert "the model's 200-token context window" in detail
        assert "trained for" not in detail

    def test_a_window_at_the_trained_one_still_says_so(self):
        backend = _GrowableBackend(capacity=150, limit=(150, "trained"))
        client = TestClient(hs.create_app(_EngineWithBackend(backend)))
        resp = _post_chat(client, "x" * 400)
        assert resp.status_code == 413, resp.text
        assert "150-token context window this model was trained for" in resp.json()["detail"]


class _CpuOnlyEngine(GatedEngine):
    """A model whose GPU layers setting puts no layer on the GPU."""

    def configured_gpu_layers(self):
        return 0


class TestACpuOnlyLoadNeedsNoVram:
    def test_the_admission_admits_it_whatever_the_reading(self):
        budget = sa.LoadBudget(name="b", vram_required=GB, headroom=0, resident_cap=None,
                               pinned=frozenset(), check_split_fit=True, cpu_only=True)
        probe = sa.VramProbe(free=0, probe_ok=True, process_scoped=False,
                             shortfall=None, shares_adaptive=False)
        engines = {"a": GatedEngine("a")}
        engines["a"]._loaded = True
        assert sa.decide_admission(probe, budget, ["a"], engines).action == sa.ADMIT
        gpu = sa.decide_admission(probe, sa.LoadBudget(
            name="b", vram_required=GB, headroom=0, resident_cap=None,
            pinned=frozenset(), check_split_fit=True), ["a"], engines)
        assert gpu.action == sa.EVICT_IDLE and gpu.victim == "a"

    def test_a_job_load_keeps_the_active_model(self, monkeypatch):
        a = GatedEngine("model-a")
        b = _CpuOnlyEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False,
                                           activate=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert a.loaded and a.unload_calls == 0
        assert hs._active_model_name == "model-a" and hs._engine is a
        assert set(hs._engines) == {"model-a", "model-b"}

    def test_the_resident_cap_still_evicts(self, monkeypatch):
        a = GatedEngine("model-a")
        b = _CpuOnlyEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _knobs(monkeypatch, max_resident_models=1)
        _seat("model-a", a)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()


class TestHealthAfterTheActiveModelWasEvicted:
    def _evicted(self, monkeypatch, registry):
        """The state a non-activating load leaves after evicting the active
        model-a: nothing active, model-a remembered, not in _engines."""
        monkeypatch.setattr("localm.config.load_registry", lambda: registry)
        monkeypatch.setattr(hs, "_default_model_name", None)
        client = TestClient(hs.create_app(None))
        hs._last_active_model_name = "model-a"
        return client

    def test_the_model_the_next_request_reloads_is_reported(self, monkeypatch):
        client = self._evicted(monkeypatch, {"model-a": {"path": "models/model-a.gguf"}})
        r = client.get("/health")
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok", "model": "model-a", "loaded": False}

    def test_it_is_reported_loading_while_it_reloads(self, monkeypatch):
        client = self._evicted(monkeypatch, {"model-a": {"path": "models/model-a.gguf"}})
        hs._loads_in_flight["model-a"] = 1
        try:
            r = client.get("/health")
        finally:
            hs._loads_in_flight.pop("model-a", None)
        assert r.json() == {"status": "ok", "model": "model-a", "loaded": False,
                            "loading": True}

    def test_without_a_registry_a_model_no_longer_kept_is_not_recoverable(self, monkeypatch):
        client = self._evicted(monkeypatch, {})
        r = client.get("/health")
        assert r.status_code == 503
        assert "No engine initialised" in r.text
