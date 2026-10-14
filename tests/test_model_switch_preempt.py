# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preemptive model switching (http_server.switch_engine).

Selecting a new model while a previous selection is still loading must abort the
in-flight load and load the new model immediately, instead of finishing the
abandoned model first. These tests drive the coordinator with a fake Engine whose
load blocks until either a gate is opened or its cancel event fires, so the
preemption/coalescing logic is exercised deterministically without a real model.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from localm.inference import http_server as hs
from localm.inference.backends.base import ModelLoadCancelled


class FakeEngine:
    """Engine stand-in for switch_engine: its load() blocks until `load_gate` is
    set OR the installed cancel event fires (then it raises ModelLoadCancelled,
    exactly like the GGUF backend's aborted native load)."""

    def __init__(self, name, *, load_gate=None, loaded_log=None):
        self.display_name = name
        self._loaded = False
        self._cancel = None
        self._gate = load_gate
        self._log = loaded_log
        self.load_started = threading.Event()

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, event):
        self._cancel = event

    def load(self):
        self.load_started.set()
        # Poll for cancellation while waiting for the test to open the gate.
        while True:
            if self._cancel is not None and self._cancel.is_set():
                raise ModelLoadCancelled(f"aborted: {self.display_name}")
            if self._gate is None or self._gate.wait(0.01):
                break
        if self._cancel is not None and self._cancel.is_set():
            raise ModelLoadCancelled(f"aborted: {self.display_name}")
        self._loaded = True
        if self._log is not None:
            self._log.append(self.display_name)

    def unload(self):
        self._loaded = False


class _AlwaysCancelsEngine:
    """Engine stand-in whose load() always raises ModelLoadCancelled for a
    reason that has nothing to do with a newer selection - the runner torn
    down mid-load, or the test-only fault injector - never a supersession."""

    def __init__(self, name, reason):
        self.display_name = name
        self._loaded = False
        self._reason = reason

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, event):
        pass

    def load(self):
        raise ModelLoadCancelled(self._reason)

    def unload(self):
        self._loaded = False


def _reset_switch_state():
    hs._inference_sem = asyncio.Semaphore(1)
    hs._engine = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()


async def _await_started(engine, timeout=2.0):
    """Block until `engine.load()` has actually begun on the executor thread."""
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, engine.load_started.wait, timeout)
    assert ok, "load never started"


def test_newer_switch_preempts_in_flight_load():
    """A second selection aborts the first's in-flight load; only the second one
    actually loads, and the first reports 'superseded' (not an error)."""

    async def scenario():
        _reset_switch_state()
        log = []
        never = threading.Event()          # A never finishes on its own
        ready = threading.Event(); ready.set()   # B finishes as soon as it runs
        engines = {
            "A": FakeEngine("A", load_gate=never, loaded_log=log),
            "B": FakeEngine("B", load_gate=ready, loaded_log=log),
        }
        def make(n):
            return engines[n]

        task_a = asyncio.create_task(hs.switch_engine("A", make))
        await _await_started(engines["A"])
        task_b = asyncio.create_task(hs.switch_engine("B", make))
        res_a, res_b = await asyncio.gather(task_a, task_b)
        return res_a, res_b, log

    res_a, res_b, log = asyncio.run(scenario())
    assert res_a["status"] == "superseded"
    assert res_a["by"] == "B"
    assert res_b["status"] == "loaded"
    assert log == ["B"]                    # A never finished loading
    assert hs._engine.display_name == "B"
    assert hs._engine.loaded


def test_rapid_switches_coalesce_to_latest():
    """Three quick selections: the abandoned two never load, the last one wins.
    The queued-but-not-started middle switch is dropped without loading."""

    async def scenario():
        _reset_switch_state()
        log = []
        never = threading.Event()
        ready = threading.Event(); ready.set()
        engines = {
            "A": FakeEngine("A", load_gate=never, loaded_log=log),
            "B": FakeEngine("B", load_gate=never, loaded_log=log),
            "C": FakeEngine("C", load_gate=ready, loaded_log=log),
        }
        def make(n):
            return engines[n]

        task_a = asyncio.create_task(hs.switch_engine("A", make))
        await _await_started(engines["A"])
        task_b = asyncio.create_task(hs.switch_engine("B", make))
        task_c = asyncio.create_task(hs.switch_engine("C", make))
        res_a, res_b, res_c = await asyncio.gather(task_a, task_b, task_c)
        return res_a, res_b, res_c, log

    res_a, res_b, res_c, log = asyncio.run(scenario())
    assert res_a["status"] == "superseded"
    assert res_b["status"] == "superseded"   # coalesced away, never loaded
    assert res_c["status"] == "loaded"
    assert log == ["C"]                        # only the final model loaded
    assert hs._engine.display_name == "C"


# ---------------------------------------------------------------------------
#  A cancelled-but-not-superseded load must report the real reason
# ---------------------------------------------------------------------------

def test_non_supersession_cancellation_is_not_reported_as_superseded():
    """A load cancelled for a reason unrelated to a newer selection (an
    API-routed, preempt=False request has no supersession mechanism at all)
    must carry the real reason, not a fabricated 'superseded by' claim."""

    async def scenario():
        _reset_switch_state()
        engine = _AlwaysCancelsEngine(
            "A", reason="the model was unloaded while it was still loading")
        return await hs.switch_engine("A", lambda n: engine, preempt=False)

    res = asyncio.run(scenario())
    assert res["status"] == "cancelled"
    assert res["reason"] == "the model was unloaded while it was still loading"
    assert "by" not in res


def test_preempted_switch_with_nothing_newer_is_also_not_superseded():
    """Even under preempt=True, a cancellation that did NOT come from a newer
    selection (nothing else ever changed _switch_desired away from this call's
    own name) must not claim supersession either."""

    async def scenario():
        _reset_switch_state()
        engine = _AlwaysCancelsEngine("A", reason="forced cancellation (test-only)")
        return await hs.switch_engine("A", lambda n: engine, preempt=True)

    res = asyncio.run(scenario())
    assert res["status"] == "cancelled"
    assert res["reason"] == "forced cancellation (test-only)"


def test_get_engine_reports_the_real_cancellation_reason_not_none():
    """get_engine's 503 for a cancelled (not superseded) load must show the
    actual reason. Before the fix, get_engine always calls switch_engine with
    preempt=False, so _switch_desired is never this call's own concern and the
    503 fabricated 'superseded by a newer request: None'."""

    async def scenario():
        _reset_switch_state()
        hs._active_model_name = "A"
        hs._default_model_name = None
        engine = _AlwaysCancelsEngine(
            "A", reason="the model was unloaded while it was still loading")
        hs._engine_factory = lambda n: engine
        with pytest.raises(HTTPException) as exc_info:
            await hs.get_engine("A")
        return exc_info.value

    exc = asyncio.run(scenario())
    assert exc.status_code == 503
    assert "the model was unloaded while it was still loading" in exc.detail
    assert "superseded" not in exc.detail
    assert "None" not in exc.detail


def test_reselecting_the_loading_model_does_not_restart_it():
    """Re-selecting the SAME model that is still loading must let it finish (not
    cancel and reload it); the second request resolves to 'already_active'."""

    async def scenario():
        _reset_switch_state()
        log = []
        gate = threading.Event()           # opened by the test to finish A's load
        engine = FakeEngine("A", load_gate=gate, loaded_log=log)
        def make(n):
            return engine

        task1 = asyncio.create_task(hs.switch_engine("A", make))
        await _await_started(engine)
        task2 = asyncio.create_task(hs.switch_engine("A", make))
        await asyncio.sleep(0.05)          # let task2 register desired + queue
        assert not hs._switch_cancel.is_set(), "re-select must not cancel the load"
        gate.set()                         # now let A finish loading
        res1, res2 = await asyncio.gather(task1, task2)
        return res1, res2, log

    res1, res2, log = asyncio.run(scenario())
    assert res1["status"] == "loaded"
    assert res2["status"] == "already_active"
    assert log == ["A"]                    # loaded exactly once


def test_single_switch_loads_normally():
    """No contention: a lone switch just loads and reports 'loaded'."""

    async def scenario():
        _reset_switch_state()
        log = []
        ready = threading.Event(); ready.set()
        engine = FakeEngine("solo", load_gate=ready, loaded_log=log)
        res = await hs.switch_engine("solo", lambda n: engine)
        return res, log

    res, log = asyncio.run(scenario())
    assert res["status"] == "loaded"
    assert log == ["solo"]
    assert hs._engine.display_name == "solo"


# ---------------------------------------------------------------------------
#  Idle-unload activity seeding on registration
# ---------------------------------------------------------------------------

def test_switch_engine_seeds_per_model_activity_immediately():
    """A freshly loaded engine must get its OWN activity timestamp the instant
    it registers into _engines - not fall through, later, to whatever the
    GLOBAL _last_activity was left at by a PREVIOUS model."""

    async def scenario():
        _reset_switch_state()
        hs._last_activity_per_model.clear()
        # The state a real idle previous model leaves behind: the global timer
        # is old, and nothing has touched it since.
        hs._last_activity = time.monotonic() - 1000
        ready = threading.Event(); ready.set()
        engine = FakeEngine("fresh", load_gate=ready)
        before = time.monotonic()
        res = await hs.switch_engine("fresh", lambda n: engine)
        return res, before

    res, before = asyncio.run(scenario())
    assert res["status"] == "loaded"
    assert "fresh" in hs._last_activity_per_model
    assert hs._last_activity_per_model["fresh"] >= before


def test_switching_models_does_not_evict_the_new_one_via_the_old_ones_staleness():
    """Model A goes idle past the TTL, then the user switches to model B. If
    nothing seeds a per-model entry for B, it inherits A's stale GLOBAL
    _last_activity the instant it registers, and the very next idle sweep can
    evict B before it ever serves a single request."""

    async def scenario():
        _reset_switch_state()
        hs._last_activity_per_model.clear()
        ready = threading.Event(); ready.set()

        engine_a = FakeEngine("A", load_gate=ready)
        await hs.switch_engine("A", lambda n: engine_a)

        # Age A - and the global timer along with it, mirroring a real idle
        # stretch where nothing touched anything for a while - well past any
        # TTL the test below will use.
        stale = time.monotonic() - 1000
        hs._last_activity_per_model["A"] = stale
        hs._last_activity = stale

        engine_b = FakeEngine("B", load_gate=ready)
        res = await hs.switch_engine("B", lambda n: engine_b)
        assert res["status"] == "loaded"

        hs._inference_sem = asyncio.Semaphore(1)
        await hs._idle_unload_once(60)   # A's own staleness may evict A - fine
        return engine_b.loaded

    b_loaded = asyncio.run(scenario())
    assert b_loaded is True, "freshly switched-to model B must not be evicted"
