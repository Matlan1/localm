# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characterization of http_server.switch_engine() as a transition matrix.

Every row drives switch_engine() from one starting state (who is resident,
what the VRAM probe reports, how the split and residency knobs are set) and
pins what a caller and the rest of the server can observe afterwards:

  - the returned dict, or the HTTPException status and its exact detail text;
  - the live registries (_engines, _engines_lru, _inference_sems) and the
    active-model pointers (_active_model_name, _last_active_model_name,
    _engine, _inference_sem);
  - which fake engines were built, loaded, unloaded or signalled to cancel,
    and how many VRAM probes the transition took.

The rows are grouped by transition: already active, load success (with and
without eviction), superseded, cancelled, missing model files, VRAM that
cannot be measured, an inconclusive probe (retry, refuse, restart, recover),
pinned and busy victims, the unload race, load errors, and the static vs
adaptive multi-GPU split handling.

Patched boundaries: localm.discover.vram_capacity and gpu_split_shortfall,
the registry and config readers, residency.cancel_all, the embedder's
loaded_dim and localm.vram.wait_for_vram_release.
"""

import asyncio
import re
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from localm.discover import GPU_PROBE_OK, GPU_PROBE_TIMEOUT
from localm.inference import http_server as hs
from localm.inference import residency
from tests.conftest import probe_double
from tests.test_eviction_victim_race import GatedEngine, _RacyVictim, _wait_entered
from tests.test_gpu_load_placement_reporting import _FakeEngine as PlacementEngine
from tests.test_gpu_registry import _UnfittableEngine
from tests.test_model_switch_preempt import FakeEngine as GateableLoadEngine
from tests.test_model_switch_preempt import _AlwaysCancelsEngine, _await_started
from tests.test_vram_eviction_safety import _FakeHangAlarm, _knobs

MB = 1024 ** 2
GB = 1024 ** 3

# The registered fake paths do not exist on disk, so every load is sized at
# residency's unknown-footprint default. NEED is the whole-model estimate
# switch_engine demands; a resident fake model occupies exactly that much in
# the simulated device below. At the default 10 GB total, one model fits
# (10 GB free >= NEED + headroom) and a second one does not.
NEED = residency.required_vram_bytes(residency.UNKNOWN_FOOTPRINT_BYTES)
HEADROOM = residency.DEFAULT_HEADROOM_BYTES


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
    """Every row starts from an empty server and leaves one behind."""
    def _clear():
        for d in (hs._engines, hs._engines_lru, hs._inference_sems,
                  hs._last_activity_per_model, hs._evicting_names):
            d.clear()

    _clear()
    for name, value in (("_active_model_name", None),
                        ("_last_active_model_name", None),
                        ("_engine", None),
                        ("_inference_sem", None),
                        ("_switch_desired", None),
                        ("_switch_loading", None),
                        ("_switch_cancel", None),
                        ("_hang_alarm_instance", None),
                        ("_gpu_coord", None),
                        ("_engine_factory", hs._engine_factory),
                        ("_INCONCLUSIVE_LOAD_RETRY_DELAY", 0),
                        ("_BUSY_VICTIM_IDLE_WAIT_S", 0.2),
                        ("_VICTIM_RELEASE_EXTRA_WAIT_S", 0)):
        monkeypatch.setattr(hs, name, value)
    yield
    _clear()


def _install(monkeypatch, engines, *, total=10 * GB, measurable=True,
             statuses=(GPU_PROBE_OK,), reading=None, registered=None,
             suffix=".gguf", shortfall=(), adaptive=False, on_cancel=None):
    """Wire switch_engine's collaborators to a simulated box.

    *engines* maps a model name to the fake engine the factory hands out for
    it. *registered* names the registry entries (default: every engine; pass
    () for the empty-registry startup). The device reports *total* bytes and,
    when *measurable*, free = total minus NEED per loaded fake engine (or
    whatever *reading* returns). *statuses* is the probe status per probe,
    the last one repeating. *shortfall* (a list, or a callable returning one)
    and *adaptive* are the configured split's per-device answer.

    Returns a namespace recording what happened: ``built`` (factory calls),
    ``probes`` (VRAM probes taken), ``release_waits`` ((before, after) free
    readings of each post-eviction wait), ``shortfall_calls`` and
    ``cancelled`` (models whose generations were signalled to stop)."""
    env = SimpleNamespace(built=[], probes=0, release_waits=[],
                          shortfall_calls=0, cancelled=[])

    names = list(engines) if registered is None else list(registered)
    registry = {n: {"path": f"models/{n}{suffix}", "source": "local"} for n in names}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr(
        "localm.model_manager.get_model_info",
        lambda name: (f"models/{name}{suffix}", "hint") if name in registry else None)

    def _default_reading():
        info = {"total": total}
        if measurable:
            loaded = sum(1 for e in engines.values() if e.loaded)
            info["free"] = total - loaded * NEED
        return info

    read = reading or _default_reading

    def _probe(*args, **kwargs):
        if kwargs.get("return_status"):
            env.probes += 1
            status = statuses[min(env.probes, len(statuses)) - 1]
            return probe_double(read, status=status)(*args, **kwargs)
        return read()

    monkeypatch.setattr("localm.discover.vram_capacity", _probe)

    def _split(vram_required, *args, **kwargs):
        env.shortfall_calls += 1
        found = list(shortfall() if callable(shortfall) else shortfall)
        return (found, adaptive) if kwargs.get("return_shares_adaptive") else found

    monkeypatch.setattr("localm.discover.gpu_split_shortfall", _split)

    def _wait_for_release(read_free, *, before_bytes, **kwargs):
        after = read_free()
        env.release_waits.append((before_bytes, after))
        return True, after

    monkeypatch.setattr("localm.vram.wait_for_vram_release", _wait_for_release)
    monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: None)

    real_cancel_all = residency.cancel_all

    def _cancel_all(name):
        env.cancelled.append(name)
        if on_cancel is not None:
            on_cancel(name)
        return real_cancel_all(name)

    monkeypatch.setattr(residency, "cancel_all", _cancel_all)

    def _factory(name):
        env.built.append(name)
        return engines[name]

    env.factory = _factory
    return env


def _seat(name, engine, *, active=False, busy=0):
    """Make *engine* resident exactly as a completed earlier load leaves it."""
    engine._loaded = True
    if busy:
        engine.active_requests = busy
    hs._engines[name] = engine
    hs._engines_lru.append(name)
    sem = hs._inference_sems.setdefault(name, asyncio.Semaphore(1))
    if active:
        hs._active_model_name = name
        hs._engine = engine
        hs._inference_sem = sem


def _registered_anywhere(name):
    """Which live registries or active pointers still name *name*."""
    return {where for where, hit in (
        ("engines", name in hs._engines),
        ("lru", name in hs._engines_lru),
        ("sems", name in hs._inference_sems),
        ("active", hs._active_model_name == name),
    ) if hit}


def _inconclusive_refusal(name, attempts):
    return re.compile(
        rf"Cannot load '{re.escape(name)}': tried measuring free VRAM {attempts} "
        r"times over about \d+s without a conclusive reading, could not free "
        r"anything, and automatic recovery is unavailable\. Please file a bug "
        r"report if this keeps happening\.")


def _estimate_confirm_detail(name, free_bytes):
    return (f"'{name}' does not fit the estimated free VRAM "
            f"(need ~{NEED // MB} MB, {free_bytes // MB} MB free) even after "
            "eviction; loading it anyway will let the backend fall back to "
            "partial CPU offload, which is slower")


# --------------------------------------------------------------------------- #
#  Already active                                                             #
# --------------------------------------------------------------------------- #

class TestAlreadyActive:
    def test_resident_model_is_reactivated_without_probing_or_loading(self, monkeypatch):
        placement = {"gpu_layers_offloaded": 20, "gpu_layers_total": 32,
                     "degraded": True}
        a = PlacementEngine("model-a", gpu_placement=placement)
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a)
        _seat("model-b", b, active=True)
        hs._last_active_model_name = "model-gone"
        seen = []

        res = asyncio.run(hs.switch_engine("model-a", env.factory, on_active=seen.append))

        assert res == {"status": "already_active", "model": "model-a", **placement}
        assert env.built == [] and env.probes == 0
        assert hs._engines_lru == ["model-b", "model-a"], "touched to most recent"
        assert hs._active_model_name == "model-a"
        assert hs._last_active_model_name is None
        assert hs._engine is a
        assert hs._inference_sem is hs._inference_sems["model-a"]
        assert seen == ["model-a"]
        assert b.loaded and b.unload_calls == 0

    def test_non_activating_call_leaves_the_active_model_alone(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a)
        _seat("model-b", b, active=True)
        seen = []

        res = asyncio.run(hs.switch_engine("model-a", env.factory, activate=False,
                                           on_active=seen.append))

        assert res == {"status": "already_active", "model": "model-a"}
        assert hs._engines_lru == ["model-b", "model-a"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert seen == []

    def test_a_registered_engine_mid_unload_is_reloaded_not_reported_active(
            self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=20 * GB)
        _seat("model-a", a)
        a.unloading = True

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-a"}
        assert env.built == [], "the registered engine object is reused"
        assert env.probes == 1
        assert hs._engines["model-a"] is a


# --------------------------------------------------------------------------- #
#  Load success                                                               #
# --------------------------------------------------------------------------- #

class TestLoadSuccess:
    def test_fresh_load_that_fits_registers_and_activates(self, monkeypatch):
        placement = {"gpu_layers_offloaded": 32, "gpu_layers_total": 32,
                     "degraded": False}
        a = PlacementEngine("model-a", gpu_placement=placement)
        env = _install(monkeypatch, {"model-a": a})
        seen = []

        res = asyncio.run(hs.switch_engine("model-a", env.factory, on_active=seen.append))

        assert res == {"status": "loaded", "model": "model-a", **placement}
        assert env.built == ["model-a"] and env.probes == 1
        assert hs._engines == {"model-a": a} and a.loaded
        assert hs._engines_lru == ["model-a"]
        assert hs._active_model_name == "model-a" and hs._engine is a
        assert hs._inference_sem is hs._inference_sems["model-a"]
        assert "model-a" in hs._last_activity_per_model
        assert seen == ["model-a"]
        assert env.release_waits == []
        assert hs._switch_desired == "model-a"
        assert hs._switch_cancel is None and hs._switch_loading is None

    def test_fresh_load_without_placement_reports_no_placement_keys(self, monkeypatch):
        a = PlacementEngine("model-a", gpu_placement=None)
        env = _install(monkeypatch, {"model-a": a})

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-a"}

    def test_empty_registry_loads_with_no_vram_gate_at_all(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, registered=(),
                       statuses=(GPU_PROBE_TIMEOUT,))

        res = asyncio.run(hs.switch_engine("model-a", env.factory))

        assert res == {"status": "loaded", "model": "model-a"}
        assert env.probes == 0 and env.shortfall_calls == 0
        assert hs._engines == {"model-a": a} and a.loaded

    def test_non_activating_load_keeps_the_current_active_model(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        _seat("model-b", b, active=True)
        seen = []

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False,
                                           activate=False, on_active=seen.append))

        assert res == {"status": "loaded", "model": "model-a"}
        assert hs._engines_lru == ["model-b", "model-a"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert seen == []

    def test_non_activating_load_still_activates_when_nothing_else_would_answer(
            self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a})
        seen = []

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False,
                                           activate=False, on_active=seen.append))

        assert res == {"status": "loaded", "model": "model-a"}
        assert hs._active_model_name == "model-a" and hs._engine is a
        assert seen == ["model-a"]

    def test_idle_victim_is_detached_from_live_registries_before_its_native_unload(
            self, monkeypatch):
        a = GatedEngine("model-a", gate=True)
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)
        free_before = 10 * GB - NEED

        async def scenario():
            task = asyncio.create_task(hs.switch_engine("model-b", env.factory))
            try:
                await _wait_entered(a)
                during = {
                    "registered": _registered_anywhere("model-a"),
                    "engine_pointer_cleared": hs._engine is None,
                    "sem_pointer_cleared": hs._inference_sem is None,
                    "flagged_unloading": a.unloading,
                    "still_natively_loaded": a.loaded,
                }
            finally:
                a.release.set()
            return during, await task

        during, res = asyncio.run(scenario())

        assert during == {"registered": set(), "engine_pointer_cleared": True,
                          "sem_pointer_cleared": True, "flagged_unloading": True,
                          "still_natively_loaded": True}
        assert res == {"status": "loaded", "model": "model-b"}
        assert a.unload_calls == 1 and not a.loaded
        assert hs._engines == {"model-b": b} and hs._engines_lru == ["model-b"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert env.probes == 2
        assert env.release_waits == [(free_before, 10 * GB)], (
            "the release wait starts from the pre-eviction reading, after the free")

    def test_evicting_the_active_model_for_a_non_activating_load_remembers_it(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False,
                                           activate=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert _registered_anywhere("model-a") == set()
        assert hs._active_model_name is None and hs._engine is None
        assert hs._last_active_model_name == "model-a"
        assert hs._engines == {"model-b": b}

    def test_resident_cap_evicts_an_idle_model_even_though_vram_fits(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        _knobs(monkeypatch, max_resident_models=1)
        _seat("model-a", a)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b}


# --------------------------------------------------------------------------- #
#  Superseded                                                                 #
# --------------------------------------------------------------------------- #

class TestSuperseded:
    def test_newer_switch_aborts_the_in_flight_load(self, monkeypatch):
        never = threading.Event()
        ready = threading.Event()
        ready.set()
        a = GateableLoadEngine("model-a", load_gate=never)
        b = GateableLoadEngine("model-b", load_gate=ready)
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)

        async def scenario():
            ta = asyncio.create_task(hs.switch_engine("model-a", env.factory))
            await _await_started(a)
            tb = asyncio.create_task(hs.switch_engine("model-b", env.factory))
            return await asyncio.gather(ta, tb)

        res_a, res_b = asyncio.run(scenario())

        assert res_a == {"status": "superseded", "model": "model-a", "by": "model-b"}
        assert res_b == {"status": "loaded", "model": "model-b"}
        assert not a.loaded and "model-a" not in hs._engines
        assert hs._engines_lru == ["model-b"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert hs._switch_desired == "model-b"
        assert hs._switch_cancel is None and hs._switch_loading is None

    def test_queued_switch_overtaken_before_its_turn_does_no_work(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        # An earlier load of model-a still holds that model's slot.
        slot = asyncio.Semaphore(1)
        hs._inference_sems["model-a"] = slot

        async def scenario():
            await slot.acquire()
            ta = asyncio.create_task(hs.switch_engine("model-a", env.factory))
            while hs._switch_desired != "model-a":
                await asyncio.sleep(0)
            res_b = await hs.switch_engine("model-b", env.factory)
            probes_before_turn = env.probes
            slot.release()
            return await ta, res_b, probes_before_turn

        res_a, res_b, probes_before_turn = asyncio.run(scenario())

        assert res_a == {"status": "superseded", "model": "model-a", "by": "model-b"}
        assert res_b == {"status": "loaded", "model": "model-b"}
        assert env.probes == probes_before_turn, "no probe once overtaken"
        assert env.built == ["model-b"]
        assert not a.loaded and "model-a" not in hs._engines
        assert hs._active_model_name == "model-b"


# --------------------------------------------------------------------------- #
#  Cancelled                                                                  #
# --------------------------------------------------------------------------- #

class TestCancelled:
    @pytest.mark.parametrize("preempt", [False, True])
    def test_cancellation_not_caused_by_a_newer_switch_reports_its_reason(
            self, monkeypatch, preempt):
        reason = "the model was unloaded while it was still loading"
        a = _AlwaysCancelsEngine("model-a", reason)
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        _seat("model-b", b, active=True)

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=preempt))

        assert res == {"status": "cancelled", "model": "model-a", "reason": reason}
        assert "model-a" not in hs._engines and hs._engines_lru == ["model-b"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert hs._switch_cancel is None and hs._switch_loading is None


# --------------------------------------------------------------------------- #
#  Missing model files                                                        #
# --------------------------------------------------------------------------- #

class TestMissingModelFiles:
    def test_registered_registry_without_the_models_files_is_a_404(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a})
        _seat("model-a", a, active=True)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("ghost", env.factory))

        assert ei.value.status_code == 404
        assert ei.value.detail == "Model files not found: ghost"
        assert env.built == [] and env.probes == 0
        assert hs._engines == {"model-a": a} and hs._engines_lru == ["model-a"]
        assert hs._active_model_name == "model-a" and a.unload_calls == 0


# --------------------------------------------------------------------------- #
#  VRAM that cannot be measured (probe OK, no free figure)                    #
# --------------------------------------------------------------------------- #

class TestCannotMeasure:
    def test_idle_residents_are_evicted_then_the_load_goes_ahead_best_effort(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, measurable=False)
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "loaded", "model": "model-b"}
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b}
        assert env.release_waits == [], "nothing measurable to wait on"
        assert env.probes == 2

    def test_a_busy_resident_is_left_alone_even_for_an_explicit_switch(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, measurable=False)
        _seat("model-a", a, busy=1)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "loaded", "model": "model-b"}
        assert env.cancelled == []
        assert hs._engines == {"model-a": a, "model-b": b}
        assert a.loaded and a.unload_calls == 0 and a.active_requests == 1


# --------------------------------------------------------------------------- #
#  Inconclusive probe                                                         #
# --------------------------------------------------------------------------- #

class TestInconclusiveProbe:
    @pytest.mark.parametrize("measurable", [False, True],
                             ids=["no-reading", "stale-high-reading"])
    def test_refuses_after_the_automatic_retries(self, monkeypatch, measurable):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=16 * GB,
                       measurable=measurable, statuses=(GPU_PROBE_TIMEOUT,))
        attempts = hs._INCONCLUSIVE_LOAD_RETRIES + 1

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert ei.value.status_code == 503
        assert _inconclusive_refusal("model-a", attempts).fullmatch(ei.value.detail), (
            ei.value.detail)
        assert env.probes == attempts
        assert env.built == [] and not a.loaded and hs._engines == {}

    def test_a_probe_that_clears_within_the_retries_loads(self, monkeypatch):
        a = GatedEngine("model-a")
        retries = hs._INCONCLUSIVE_LOAD_RETRIES
        env = _install(monkeypatch, {"model-a": a},
                       statuses=(GPU_PROBE_TIMEOUT,) * retries + (GPU_PROBE_OK,))

        res = asyncio.run(hs.switch_engine("model-a", env.factory))

        assert res == {"status": "loaded", "model": "model-a"}
        assert env.probes == retries + 1
        assert hs._engines == {"model-a": a}

    def test_idle_residents_are_evicted_before_the_refusal(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b},
                       measurable=False, statuses=(GPU_PROBE_TIMEOUT,))
        _seat("model-a", a, active=True)
        attempts = hs._INCONCLUSIVE_LOAD_RETRIES + 1

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-b", env.factory))

        assert ei.value.status_code == 503
        assert _inconclusive_refusal("model-b", attempts).fullmatch(ei.value.detail), (
            ei.value.detail)
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert env.probes == attempts + 1, "the eviction pass is not a retry"
        assert hs._engines == {} and not b.loaded

    def test_a_busy_resident_is_never_cancelled_on_an_inconclusive_probe(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b},
                       measurable=False, statuses=(GPU_PROBE_TIMEOUT,))
        _seat("model-a", a, busy=1)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-b", env.factory, force=True))

        assert ei.value.status_code == 503
        assert env.cancelled == []
        assert hs._engines == {"model-a": a} and a.loaded and a.unload_calls == 0

    def test_exhausted_retries_escalate_to_the_self_restart(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, statuses=(GPU_PROBE_TIMEOUT,))
        alarm = _FakeHangAlarm(will_restart=True)
        monkeypatch.setattr(hs, "_hang_alarm_instance", alarm)
        attempts = hs._INCONCLUSIVE_LOAD_RETRIES + 1

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert ei.value.status_code == 503
        assert ei.value.detail == (
            "Cannot load 'model-a' right now: the server detected a stuck GPU "
            "check and is restarting automatically. This page will reconnect "
            "once it comes back up.")
        assert alarm.calls == [
            f"GPU probe still inconclusive after {attempts} attempts loading 'model-a'"]
        assert hs._engines == {}

    def test_a_declined_self_restart_falls_back_to_the_plain_refusal(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, statuses=(GPU_PROBE_TIMEOUT,))
        alarm = _FakeHangAlarm(will_restart=False)
        monkeypatch.setattr(hs, "_hang_alarm_instance", alarm)
        attempts = hs._INCONCLUSIVE_LOAD_RETRIES + 1

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert ei.value.status_code == 503
        assert _inconclusive_refusal("model-a", attempts).fullmatch(ei.value.detail), (
            ei.value.detail)
        assert len(alarm.calls) == 1


# --------------------------------------------------------------------------- #
#  Pinned victim                                                              #
# --------------------------------------------------------------------------- #

class TestPinnedVictim:
    def test_api_routed_load_defers_to_the_backend_and_keeps_the_pin(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _knobs(monkeypatch, pinned_models=["model-a"])
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert hs._engines == {"model-a": a, "model-b": b}
        assert a.loaded and a.unload_calls == 0
        assert env.cancelled == []

    def test_explicit_switch_asks_before_a_degraded_load(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _knobs(monkeypatch, pinned_models=["model-a"])
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "confirm_required", "model": "model-b",
                       "detail": _estimate_confirm_detail("model-b", 10 * GB - NEED)}
        assert env.built == [] and env.cancelled == []
        assert hs._engines == {"model-a": a} and a.unload_calls == 0
        assert hs._active_model_name == "model-a"

    def test_force_loads_without_ever_touching_the_pinned_model(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _knobs(monkeypatch, pinned_models=["model-a"])
        _seat("model-a", a, active=True)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, force=True))

        assert res == {"status": "loaded", "model": "model-b"}
        assert env.cancelled == []
        assert hs._engines == {"model-a": a, "model-b": b}
        assert a.loaded and a.unload_calls == 0
        assert hs._active_model_name == "model-b"

    def test_an_unmeetable_cap_loads_over_it_instead_of_evicting_the_pin(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        _knobs(monkeypatch, max_resident_models=1, pinned_models=["model-a"])
        _seat("model-a", a)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "loaded", "model": "model-b"}
        assert hs._engines == {"model-a": a, "model-b": b}
        assert a.unload_calls == 0 and env.cancelled == []


# --------------------------------------------------------------------------- #
#  Busy victim                                                                #
# --------------------------------------------------------------------------- #

class TestBusyVictim:
    def test_api_routed_load_never_cancels_a_busy_model(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert env.cancelled == []
        assert hs._engines == {"model-a": a, "model-b": b}
        assert a.loaded and a.unload_calls == 0 and a.active_requests == 1

    def test_explicit_switch_asks_when_the_busy_model_does_not_stop(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "confirm_required", "model": "model-b",
                       "detail": "loading 'model-b' needs to free 'model-a', "
                                 "which is 1 other active request"}
        assert env.cancelled == ["model-a"], "its generation was asked to stop"
        assert env.built == []
        assert hs._engines == {"model-a": a} and hs._engines_lru == ["model-a"]
        assert a.loaded and a.unload_calls == 0 and a.unloading is False
        assert hs._active_model_name == "model-a"

    def test_explicit_switch_evicts_a_busy_model_that_stops_when_asked(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")

        def _generation_stops(name):
            a.active_requests = 0

        env = _install(monkeypatch, {"model-a": a, "model-b": b},
                       on_cancel=_generation_stops)
        _seat("model-a", a, active=True, busy=1)

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "loaded", "model": "model-b"}
        assert env.cancelled == ["model-a"]
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b} and hs._active_model_name == "model-b"

    def test_force_evicts_a_busy_model_that_never_stops(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, force=True))

        assert res == {"status": "loaded", "model": "model-b"}
        assert env.cancelled == ["model-a"]
        assert a.unload_calls == 1 and a.unloading is True
        assert _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b} and hs._active_model_name == "model-b"


# --------------------------------------------------------------------------- #
#  Unload race                                                                #
# --------------------------------------------------------------------------- #

class TestUnloadRace:
    def test_reloading_a_victim_during_its_native_free_is_refused(self, monkeypatch):
        a = GatedEngine("model-a", gate=True)
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a)

        async def scenario():
            tb = asyncio.create_task(
                hs.switch_engine("model-b", env.factory, preempt=False))
            try:
                await _wait_entered(a)
                try:
                    await hs.switch_engine("model-a", env.factory, preempt=False)
                except HTTPException as e:
                    refused = e
                else:
                    refused = None
            finally:
                a.release.set()
            return refused, await tb

        refused, res_b = asyncio.run(scenario())

        assert refused is not None, "a reload raced the still-running native free"
        assert refused.status_code == 503
        assert refused.detail == ("'model-a' is currently being freed by another "
                                  "request; retry shortly.")
        assert res_b == {"status": "loaded", "model": "model-b"}
        assert env.built == ["model-b"]
        assert a.unload_calls == 1 and not a.loaded

        # Once the free lands the name is loadable again.
        again = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))
        assert again == {"status": "loaded", "model": "model-a"}

    def test_victim_removed_by_a_concurrent_remover_mid_unload(self, monkeypatch):
        vram = {"free": 3 * GB}
        victim = _RacyVictim("victim", vram)
        incoming = GatedEngine("incoming")
        env = _install(monkeypatch, {"victim": victim, "incoming": incoming},
                       reading=lambda: {"total": 16 * GB, "free": vram["free"]})
        _seat("victim", victim, active=True)

        res = asyncio.run(hs.switch_engine("incoming", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "incoming"}
        assert _registered_anywhere("victim") == set()
        assert hs._engines == {"incoming": incoming}
        assert hs._active_model_name == "incoming"

    def test_a_failing_native_free_propagates_and_does_not_strand_the_name(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)

        def _free_fails():
            raise RuntimeError("native free failed")

        a.unload = _free_fails

        with pytest.raises(RuntimeError, match="native free failed"):
            asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert _registered_anywhere("model-a") == set(), "stays detached"
        assert a.unloading is True
        assert "model-b" not in hs._engines and env.built == []
        assert hs._active_model_name is None and hs._engine is None

        again = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))
        assert again == {"status": "loaded", "model": "model-a"}


# --------------------------------------------------------------------------- #
#  Load error                                                                 #
# --------------------------------------------------------------------------- #

class TestLoadError:
    def test_backend_refusal_becomes_a_clean_503(self, monkeypatch):
        a = _UnfittableEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        _seat("model-b", b, active=True)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert ei.value.status_code == 503
        assert ei.value.detail == ("Failed to load 'model-a': VRAM exhausted: "
                                   "cannot fit even at 0 GPU layers")
        assert isinstance(ei.value.__cause__, RuntimeError)
        assert "model-a" not in hs._engines and hs._engines_lru == ["model-b"]
        assert hs._active_model_name == "model-b" and hs._engine is b
        assert hs._switch_cancel is None and hs._switch_loading is None

    def test_a_failed_load_after_evicting_the_active_model_leaves_none_active(
            self, monkeypatch):
        a = GatedEngine("model-a")
        b = _UnfittableEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert ei.value.status_code == 503
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {} and hs._engines_lru == []
        assert hs._active_model_name is None and hs._engine is None
        assert hs._inference_sem is None
        assert hs._last_active_model_name is None

    def test_a_non_runtime_load_failure_propagates_unconverted(self, monkeypatch):
        a = GatedEngine("model-a")

        def _load_breaks():
            raise ValueError("corrupt header")

        a.load = _load_breaks
        env = _install(monkeypatch, {"model-a": a})

        with pytest.raises(ValueError, match="corrupt header"):
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert hs._engines == {} and hs._active_model_name is None
        assert hs._switch_cancel is None and hs._switch_loading is None


# --------------------------------------------------------------------------- #
#  Static vs adaptive multi-GPU split                                         #
# --------------------------------------------------------------------------- #

_TWO_SHORT_DEVICES = [
    {"index": 0, "needed": 3000 * MB, "free": 2048 * MB},
    {"index": 1, "needed": 3000 * MB, "free": 1024 * MB},
]


class TestSplitShortfall:
    @pytest.mark.parametrize("preempt,force", [(False, False), (True, False),
                                               (True, True)])
    def test_static_shares_short_on_a_device_is_a_hard_refusal(
            self, monkeypatch, preempt, force):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=20 * GB,
                       shortfall=_TWO_SHORT_DEVICES, adaptive=False)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", env.factory, preempt=preempt,
                                         force=force))

        assert ei.value.status_code == 503
        assert ei.value.detail == (
            "Not enough VRAM on the configured split device(s) to load 'model-a' "
            "(GPU 0 needs ~3000 MB, 2048 MB free; GPU 1 needs ~3000 MB, 1024 MB free).")
        assert env.built == [] and hs._engines == {}

    def test_adaptive_shares_short_defer_to_the_backend_for_an_api_load(
            self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=20 * GB,
                       shortfall=_TWO_SHORT_DEVICES, adaptive=True)

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-a"}
        assert hs._engines == {"model-a": a}

    def test_adaptive_shares_short_ask_first_on_an_explicit_switch(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=20 * GB,
                       shortfall=_TWO_SHORT_DEVICES, adaptive=True)

        res = asyncio.run(hs.switch_engine("model-a", env.factory))
        forced = asyncio.run(hs.switch_engine("model-a", env.factory, force=True))

        assert res == {"status": "confirm_required", "model": "model-a",
                       "detail": _estimate_confirm_detail("model-a", 20 * GB)}
        assert forced == {"status": "loaded", "model": "model-a"}

    def test_a_device_shortfall_drives_eviction_until_it_clears(self, monkeypatch):
        a = GatedEngine("model-a")
        b = GatedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB,
                       shortfall=lambda: _TWO_SHORT_DEVICES[:1] if a.loaded else [],
                       adaptive=False)
        _seat("model-a", a)

        res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-b"}
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b}
        assert env.shortfall_calls == 2

    def test_a_non_gguf_model_skips_the_per_device_check(self, monkeypatch):
        a = GatedEngine("model-a")
        env = _install(monkeypatch, {"model-a": a}, total=20 * GB, suffix="",
                       shortfall=_TWO_SHORT_DEVICES, adaptive=False)

        res = asyncio.run(hs.switch_engine("model-a", env.factory, preempt=False))

        assert res == {"status": "loaded", "model": "model-a"}
        assert env.shortfall_calls == 0


# --------------------------------------------------------------------------- #
#  Load placement log                                                         #
# --------------------------------------------------------------------------- #

def _switch_lines(caplog):
    return [(r.levelname, r.getMessage()) for r in caplog.records
            if r.getMessage().startswith("switch_engine: loaded ")]


class TestLoadPlacementLog:
    def test_a_routed_load_logs_its_placement_and_the_eviction_that_made_room(
            self, monkeypatch, caplog):
        a = GatedEngine("model-a")
        b = PlacementEngine("model-b", gpu_placement={
            "gpu_layers_offloaded": 32, "gpu_layers_total": 32, "degraded": False})
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True)

        with caplog.at_level("INFO", logger="localm"):
            res = asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False,
                                               activate=False))

        assert res["status"] == "loaded" and a.unload_calls == 1
        [(level, line)] = _switch_lines(caplog)
        assert level == "INFO"
        free_before = (10 * GB - NEED) // MB
        assert line.startswith("switch_engine: loaded 'model-b': 32/32 layers on the GPU; ")
        assert (f"evicting 'model-a': VRAM release confirmed ({free_before} -> "
                f"{10 * GB // MB} MB free in ") in line

    def test_a_load_left_partly_on_the_cpu_is_logged_as_a_warning(
            self, monkeypatch, caplog):
        b = PlacementEngine("model-b", gpu_placement={
            "gpu_layers_offloaded": 12, "gpu_layers_total": 32, "degraded": True})
        b.gpu_sizing = {"mode": "auto", "layers": 12, "n_ctx": 4096,
                        "free_bytes": 6 * GB, "total_bytes": 16 * GB,
                        "model_bytes": 8 * GB, "kv_bytes": GB // 2,
                        "overhead_bytes": GB // 2,
                        "cause": "only 6.0 of 16.0 GB free - another app holds VRAM"}
        env = _install(monkeypatch, {"model-b": b})

        with caplog.at_level("INFO", logger="localm"):
            asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        [(level, line)] = _switch_lines(caplog)
        assert level == "WARNING"
        assert line == (
            "switch_engine: loaded 'model-b': 12/32 layers on the GPU, the rest on the "
            "CPU (slower); n_ctx 4096, sized against 6144 MB free VRAM (full offload "
            "needs ~9216 MB: weights 8192 + KV 512 + overhead 512) - only 6.0 of 16.0 "
            "GB free - another app holds VRAM")

    def test_an_already_resident_model_logs_no_load_line(self, monkeypatch, caplog):
        a = PlacementEngine("model-a")
        env = _install(monkeypatch, {"model-a": a})
        _seat("model-a", a, active=True)

        with caplog.at_level("INFO", logger="localm"):
            asyncio.run(hs.switch_engine("model-a", env.factory))

        assert _switch_lines(caplog) == []


# --------------------------------------------------------------------------- #
#  Busy resident model and placement heal                                     #
# --------------------------------------------------------------------------- #

_REAL_BUSY_WAIT_S = hs._BUSY_VICTIM_IDLE_WAIT_S
_REAL_EXTRA_RELEASE_WAIT_S = hs._VICTIM_RELEASE_EXTRA_WAIT_S
_FULL = {"gpu_layers_offloaded": 32, "gpu_layers_total": 32, "degraded": False}
_PARTIAL = {"gpu_layers_offloaded": 12, "gpu_layers_total": 32, "degraded": True}


class _SizedEngine:
    """Fake engine whose load N gets ``placements[N]`` (the last one repeats)
    and an auto-sized ``gpu_sizing``, counting loads and unloads."""

    def __init__(self, name, placements=(_FULL,)):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0
        self.unloading = False
        self.placement_heal = None
        self._placements = list(placements)
        self.gpu_placement = None
        self.gpu_sizing = None
        self.load_calls = 0
        self.unload_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, event):
        pass

    def load(self):
        placement = self._placements[min(self.load_calls, len(self._placements) - 1)]
        self.load_calls += 1
        self.gpu_placement = dict(placement)
        self.gpu_sizing = {"mode": "auto", "layers": placement["gpu_layers_offloaded"],
                           "n_ctx": 4096}
        self._loaded = True

    def unload(self):
        self.unload_calls += 1
        self._loaded = False


def _degraded_beside(monkeypatch, *, a_busy=1, placements=(_PARTIAL, _FULL)):
    """model-a resident, active and serving *a_busy* requests; model-b loaded
    by a non-explicit switch that has to load beside it, partly on the CPU."""
    a = GatedEngine("model-a")
    b = _SizedEngine("model-b", placements)
    env = _install(monkeypatch, {"model-a": a, "model-b": b})
    monkeypatch.setattr(hs, "_engine_factory", env.factory)
    _seat("model-a", a, active=True, busy=a_busy)
    asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False, activate=False))
    return a, b, env


class TestBusyResidentModel:
    def test_a_non_explicit_load_waits_for_the_busy_model_then_evicts_it(
            self, monkeypatch, caplog):
        from localm.inference.protocol import (LOADING_MODEL_STATUS,
                                               WAITING_FOR_MODEL_STATUS)
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 5.0)
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)
        statuses = []

        async def scenario():
            asyncio.get_running_loop().call_later(
                0.1, lambda: setattr(a, "active_requests", 0))
            return await hs.switch_engine("model-b", env.factory, preempt=False,
                                          activate=False, on_status=statuses.append)

        with caplog.at_level("INFO", logger="localm"):
            res = asyncio.run(scenario())

        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b} and b.gpu_placement == _FULL
        assert env.cancelled == [], "the busy model's generation is never cancelled"
        assert statuses == [WAITING_FOR_MODEL_STATUS, LOADING_MODEL_STATUS]
        assert any("waiting up to 5s for 'model-a' to finish" in r.getMessage()
                   for r in caplog.records)
        assert res == {"status": "loaded", "model": "model-b", **_FULL}

    def test_a_model_still_busy_after_the_wait_is_kept_and_recorded_as_a_blocker(
            self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)

        assert a.loaded and a.unload_calls == 0 and env.cancelled == []
        assert hs._engines == {"model-a": a, "model-b": b}
        assert b.placement_heal == hs.PlacementHeal(frozenset({"model-a"}), False)

    def test_an_explicit_switch_still_cancels_instead_of_waiting(self, monkeypatch):
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 60.0)
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)
        statuses = []

        res = asyncio.run(hs.switch_engine("model-b", env.factory,
                                           on_status=statuses.append))

        assert res["status"] == "confirm_required"
        assert env.cancelled == ["model-a"] and statuses == []

    def test_two_loads_waiting_for_one_busy_model_both_finish(self, monkeypatch):
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 5.0)
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b")
        c = _SizedEngine("model-c")
        env = _install(monkeypatch, {"model-a": a, "model-b": b, "model-c": c})
        _seat("model-a", a, active=True, busy=1)

        async def scenario():
            asyncio.get_running_loop().call_later(
                0.1, lambda: setattr(a, "active_requests", 0))
            return await asyncio.gather(
                hs.switch_engine("model-b", env.factory, preempt=False, activate=False),
                hs.switch_engine("model-c", env.factory, preempt=False, activate=False),
                return_exceptions=True)

        results = asyncio.run(scenario())

        assert [r for r in results if isinstance(r, BaseException)] == []
        assert [r["status"] for r in results] == ["loaded", "loaded"]
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()

    def test_a_busy_model_another_request_starts_unloading_is_not_unloaded_twice(
            self, monkeypatch):
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 5.0)
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b", (_PARTIAL,))
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        _seat("model-a", a, active=True, busy=1)

        def _unload_begins():
            a.unloading = True
            a.active_requests = 0

        async def scenario():
            asyncio.get_running_loop().call_later(0.1, _unload_begins)
            return await hs.switch_engine("model-b", env.factory, preempt=False,
                                          activate=False)

        res = asyncio.run(scenario())

        assert a.unload_calls == 0, "only the unload already under way frees it"
        assert res["status"] == "loaded" and b.loaded

    def test_get_engine_passes_the_wait_status_to_the_load(self, monkeypatch):
        from localm.inference.protocol import (LOADING_MODEL_STATUS,
                                               WAITING_FOR_MODEL_STATUS)
        monkeypatch.setattr(hs, "_BUSY_VICTIM_IDLE_WAIT_S", 5.0)
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b")
        env = _install(monkeypatch, {"model-a": a, "model-b": b})
        monkeypatch.setattr(hs, "_engine_factory", env.factory)
        _seat("model-a", a, active=True, busy=1)
        statuses = []

        async def scenario():
            asyncio.get_running_loop().call_later(
                0.1, lambda: setattr(a, "active_requests", 0))
            return await hs.get_engine("model-b", activate=False,
                                       on_status=statuses.append)

        got = asyncio.run(scenario())

        assert got is b and b.gpu_placement == _FULL
        assert statuses == [WAITING_FOR_MODEL_STATUS, LOADING_MODEL_STATUS]

    @staticmethod
    def _worst_wait():
        """The longest a load can wait before it starts: a heal's own unload, one
        busy model's wait, one eviction, and the one longer release wait a load
        may take."""
        import inspect
        from localm.vram import wait_for_vram_release
        first_release_wait = inspect.signature(
            wait_for_vram_release).parameters["timeout_s"].default
        return _REAL_BUSY_WAIT_S + 2 * first_release_wait + _REAL_EXTRA_RELEASE_WAIT_S

    def test_the_waits_fit_inside_the_coder_http_request_timeout(self):
        import inspect
        from localm.plugins.coder.backends.http import HTTPBackend
        coder_timeout = inspect.signature(HTTPBackend.__init__).parameters["timeout"].default
        assert self._worst_wait() <= coder_timeout / 4, (
            "the waits must leave the coder's request most of its timeout for the load")

    def test_the_waits_leave_an_image_description_a_minute_for_load_and_reply(self):
        from localm.plugins.builtin.rag import plug as rag_plug
        assert rag_plug._DESCRIBE_TIMEOUT_S - self._worst_wait() >= 60


class TestPlacementHeal:
    def test_a_degraded_model_is_reloaded_once_its_blocker_is_idle(
            self, monkeypatch, caplog):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0

        with caplog.at_level("INFO", logger="localm"):
            got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b
        assert b.unload_calls == 1 and b.load_calls == 2 and b.gpu_placement == _FULL
        assert a.unload_calls == 1 and _registered_anywhere("model-a") == set()
        assert hs._engines == {"model-b": b} and hs._engines_lru == ["model-b"]
        assert b.placement_heal is None and b.unloading is False
        assert "model-b" in hs._inference_sems
        lines = [r.getMessage() for r in caplog.records]
        assert ("switch_engine: reloading 'model-b' (12/32 layers on the GPU) for a "
                "full GPU placement: a model that held the VRAM it lacked is idle or "
                "gone") in lines
        assert any(line.startswith("switch_engine: loaded 'model-b': 32/32 layers")
                   for line in lines)

    def test_a_degraded_model_is_reloaded_when_its_blocker_is_gone(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0
        hs._engines.pop("model-a")
        hs._engines_lru.remove("model-a")

        asyncio.run(hs.get_engine("model-b", activate=False))

        assert b.load_calls == 2 and b.gpu_placement == _FULL

    def test_no_reload_while_the_blocker_is_still_busy(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)

        got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b and b.load_calls == 1 and b.unload_calls == 0
        assert b.placement_heal is not None

    def test_no_reload_while_a_request_holds_the_degraded_model(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0
        b.active_requests = 1

        got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b and b.load_calls == 1 and b.unload_calls == 0

    def test_the_reload_happens_once_even_when_it_stays_degraded(self, monkeypatch):
        a1 = GatedEngine("model-a1")
        a2 = GatedEngine("model-a2")
        b = _SizedEngine("model-b", (_PARTIAL,))
        env = _install(monkeypatch, {"model-a1": a1, "model-a2": a2, "model-b": b},
                       total=10 * GB + NEED)
        monkeypatch.setattr(hs, "_engine_factory", env.factory)
        _seat("model-a1", a1, active=True, busy=1)
        _seat("model-a2", a2, busy=1)
        asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False, activate=False))
        assert b.placement_heal.blockers == frozenset({"model-a1", "model-a2"})

        a1.active_requests = 0
        asyncio.run(hs.get_engine("model-b", activate=False))
        assert b.load_calls == 2 and a1.unload_calls == 1 and a2.loaded
        assert b.gpu_placement == _PARTIAL and b.placement_heal is None

        a2.active_requests = 0
        asyncio.run(hs.get_engine("model-b", activate=False))
        assert b.load_calls == 2, "a reload that stayed degraded is not repeated"

    def test_an_explicit_switch_reuses_the_degraded_model_as_it_is(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0

        res = asyncio.run(hs.switch_engine("model-b", env.factory))

        assert res == {"status": "already_active", "model": "model-b", **_PARTIAL}
        assert b.load_calls == 1

    def test_a_load_without_a_blocker_records_nothing_to_heal(self, monkeypatch):
        b = _SizedEngine("model-b", (_PARTIAL,))
        env = _install(monkeypatch, {"model-b": b})

        asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert b.gpu_placement == _PARTIAL and b.placement_heal is None

    def test_a_pinned_models_entry_is_never_a_blocker(self, monkeypatch):
        _knobs(monkeypatch, pinned_models=["model-a"])
        a, b, env = _degraded_beside(monkeypatch, a_busy=0)

        assert a.loaded and a.unload_calls == 0
        assert b.gpu_placement == _PARTIAL and b.placement_heal is None

    def test_an_active_degraded_model_stays_active_after_its_reload(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        hs._active_model_name, hs._engine = "model-b", b
        a.active_requests = 0

        asyncio.run(hs.get_engine("model-b", activate=False))

        assert hs._active_model_name == "model-b" and hs._engine is b

    def test_an_unnamed_request_still_resolves_to_the_model_while_it_reloads(
            self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        monkeypatch.setattr(hs, "_default_model_name", "model-a")
        hs._active_model_name, hs._engine = "model-b", b
        hs._inference_sem = hs._inference_sems["model-b"]
        a.active_requests = 0
        resolved_during_reload = []
        unload = b.unload

        def _unload():
            resolved_during_reload.append(hs._resolve_unnamed_model_name())
            unload()

        b.unload = _unload

        asyncio.run(hs.get_engine("model-b"))

        assert b.load_calls == 2 and b.gpu_placement == _FULL
        assert resolved_during_reload == ["model-b"]
        assert hs._active_model_name == "model-b" and hs._last_active_model_name is None

    def test_no_reload_when_the_load_was_admitted_beside_an_idle_model(self, monkeypatch):
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b", (_PARTIAL, _FULL))
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, total=20 * GB)
        monkeypatch.setattr(hs, "_engine_factory", env.factory)
        _seat("model-a", a, active=True)
        asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False, activate=False))
        assert a.loaded and b.gpu_placement == _PARTIAL

        got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b and b.load_calls == 1 and b.unload_calls == 0
        assert b.placement_heal is None

    def test_no_reload_while_the_vram_probe_is_inconclusive(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0
        stalled = probe_double(lambda: {"total": 10 * GB}, status=GPU_PROBE_TIMEOUT)
        monkeypatch.setattr("localm.discover.vram_capacity", stalled)

        got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b and b.load_calls == 1 and b.unload_calls == 0
        assert a.loaded and a.unload_calls == 0
        assert hs._engines == {"model-a": a, "model-b": b}
        assert b.placement_heal is not None

    def test_a_model_whose_files_are_gone_is_used_as_loaded(self, monkeypatch):
        a, b, env = _degraded_beside(monkeypatch)
        a.active_requests = 0
        monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: None)

        got = asyncio.run(hs.get_engine("model-b", activate=False))

        assert got is b and b.load_calls == 1 and b.unload_calls == 0
        assert a.loaded and hs._engines == {"model-a": a, "model-b": b}


class TestVictimReleaseWait:
    def _wait_answers(self, monkeypatch, answers, *, estimate=8 * GB, after=None):
        """Each wait_for_vram_release call returns the next of *answers* (the
        last repeats), with free VRAM *after* (default: unchanged when the
        answer is False, 4 GB more when True). The evicted model is estimated to
        hold *estimate* bytes of VRAM."""
        calls = []

        def _wait(read_free, *, before_bytes, **kwargs):
            calls.append(kwargs.get("timeout_s"))
            released = answers[min(len(calls), len(answers)) - 1]
            if after is not None:
                return released, after
            return released, (before_bytes + 4 * GB if released else before_bytes)

        monkeypatch.setattr("localm.vram.wait_for_vram_release", _wait)
        monkeypatch.setattr(hs, "_victim_vram_estimate", lambda engine: estimate)
        return calls

    def _switch(self, monkeypatch, answers, *, estimate=8 * GB, reading=None,
                placements=(_FULL,), after=None):
        a = GatedEngine("model-a")
        b = _SizedEngine("model-b", placements)
        env = _install(monkeypatch, {"model-a": a, "model-b": b}, reading=reading)
        monkeypatch.setattr(hs, "_engine_factory", env.factory)
        monkeypatch.setattr(hs, "_VICTIM_RELEASE_EXTRA_WAIT_S", 7.0)
        calls = self._wait_answers(monkeypatch, answers, estimate=estimate, after=after)
        _seat("model-a", a)
        asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))
        return a, b, env, calls

    def test_an_unconfirmed_release_gets_one_longer_wait(self, monkeypatch, caplog):
        with caplog.at_level("INFO", logger="localm"):
            a, b, env, calls = self._switch(monkeypatch, [False, True])

        assert calls == [None, 7.0]
        assert any("evicting 'model-a': VRAM release confirmed" in r.getMessage()
                   and "after an extended wait" in r.getMessage()
                   and r.levelname == "INFO" for r in caplog.records)
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_release_still_unconfirmed_is_a_warning_and_a_heal_reason(
            self, monkeypatch, caplog):
        with caplog.at_level("INFO", logger="localm"):
            a, b, env, calls = self._switch(monkeypatch, [False],
                                            placements=(_PARTIAL, _FULL))

        assert calls == [None, 7.0]
        assert any(r.levelname == "WARNING" and "VRAM release NOT confirmed"
                   in r.getMessage() for r in caplog.records)
        assert b.placement_heal == hs.PlacementHeal(frozenset(), True)
        asyncio.run(hs.get_engine("model-b"))
        assert b.load_calls == 2 and b.gpu_placement == _FULL

    def test_no_longer_wait_when_free_vram_already_covers_the_load(self, monkeypatch):
        a, b, env, calls = self._switch(monkeypatch, [False], after=10 * GB)

        assert a.unload_calls == 1
        assert calls == [None]

    def test_no_longer_wait_on_a_process_scoped_reading(self, monkeypatch):
        a, b, env, calls = self._switch(
            monkeypatch, [False],
            reading=lambda: {"total": 10 * GB, "free": 1 * GB, "free_scope": "process"})

        assert calls == [None]

    def test_a_process_scoped_reading_reports_the_release_as_not_verifiable(
            self, monkeypatch, caplog):
        with caplog.at_level("INFO", logger="localm"):
            a, b, env, calls = self._switch(
                monkeypatch, [False], placements=(_PARTIAL, _FULL),
                reading=lambda: {"total": 10 * GB, "free": 1 * GB,
                                 "free_scope": "process"})

        assert a.unload_calls == 1
        assert any(r.levelname == "INFO" and r.getMessage().startswith(
            "switch_engine: evicting 'model-a': VRAM release could not be verified")
            for r in caplog.records)
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and r.getMessage().startswith("switch_engine: evicting")]
        assert b.placement_heal is None

    def test_only_one_longer_wait_per_load(self, monkeypatch):
        a1 = GatedEngine("model-a1")
        a2 = GatedEngine("model-a2")
        b = _SizedEngine("model-b")
        env = _install(monkeypatch, {"model-a1": a1, "model-a2": a2, "model-b": b})
        monkeypatch.setattr(hs, "_VICTIM_RELEASE_EXTRA_WAIT_S", 7.0)
        calls = self._wait_answers(monkeypatch, [False])
        _seat("model-a1", a1)
        _seat("model-a2", a2)

        asyncio.run(hs.switch_engine("model-b", env.factory, preempt=False))

        assert a1.unload_calls == 1 and a2.unload_calls == 1
        assert calls == [None, 7.0, None]

    def test_a_victim_too_small_to_show_a_release_is_neither_waited_for_nor_warned(
            self, monkeypatch, caplog):
        with caplog.at_level("DEBUG", logger="localm"):
            a, b, env, calls = self._switch(monkeypatch, [False], estimate=0,
                                            placements=(_PARTIAL,))

        assert calls == [None]
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and r.getMessage().startswith("switch_engine: evicting")]
        assert any("evicting 'model-a': no VRAM release seen, none expected"
                   in r.getMessage() for r in caplog.records)
        assert b.placement_heal is None


class TestVictimVramEstimate:
    def test_a_missing_model_path_is_zero(self):
        assert hs._victim_vram_estimate(GatedEngine("model-a")) == 0

    def test_a_real_file_is_its_required_vram_scaled_by_gpu_layers(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 1000)
        e = _SizedEngine("m", ({"gpu_layers_offloaded": 8, "gpu_layers_total": 32,
                                "degraded": True},))
        e.model_path = str(f)
        assert hs._victim_vram_estimate(e) == residency.required_vram_bytes(1000)
        e.load()
        assert hs._victim_vram_estimate(e) == residency.required_vram_bytes(250)

    def test_no_layers_on_the_gpu_is_zero(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 1000)
        e = _SizedEngine("m", ({"gpu_layers_offloaded": 0, "gpu_layers_total": 32,
                                "degraded": True},))
        e.model_path = str(f)
        e.load()
        assert hs._victim_vram_estimate(e) == 0
