# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reused engine must never carry a stale, already-SET load-cancel event into
its next load.

switch_engine builds a fresh cancel Event per call but only ever INSTALLS it on
the engine when preempt=True. An engine object outlives a load: idle-unload
keeps it in _engines for lazy reload (see test_idle_unload_keeps_engine.py), so
a preempt=True switch that gets superseded leaves that engine holding a SET
event with nothing to clear it - set_load_cancel has exactly one call site, and
the GGUF backend clears _load_cancel only in its own __init__.

The next API-routed request for that model goes through
switch_engine(preempt=False), which reuses the engine and SKIPS the install, so
the stale SET event survives. The real backend honours it: gguf.py passes
_load_cancel into ModelRunner.spawn_and_load, which sends cancel_load
immediately when the event is already set, so the load aborts with
ModelLoadCancelled -> switch_engine returns "superseded" -> get_engine raises
503. Nothing clears the event, so EVERY later request for that model 503s
indefinitely.

The existing switch tests cannot catch this: they build a FRESH engine per test
and reset the _switch_* globals, so no engine ever survives a cancelled
preempt=True switch and is then loaded again via preempt=False.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localm.inference import http_server as hs
from localm.inference.backends.base import ModelLoadCancelled
from tests.conftest import probe_double


class CancelHonoringEngine:
    """Models the REAL backend contract that a stale event weaponizes: load()
    aborts when the installed cancel event is already set, exactly as the GGUF
    backend does via spawn_and_load's pre-set cancel_event check."""

    def __init__(self, name, *, load_gate=None):
        self.display_name = name
        self._loaded = False
        self._cancel = None
        self._gate = load_gate
        self.active_requests = 0
        self.load_started = threading.Event()

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, event):
        self._cancel = event

    def _cancelled(self):
        return self._cancel is not None and self._cancel.is_set()

    def load(self):
        self.load_started.set()
        if self._cancelled():                 # the pre-set case: abort at once
            raise ModelLoadCancelled(f"aborted: {self.display_name}")
        while self._gate is not None and not self._gate.wait(0.01):
            if self._cancelled():             # preempted mid-load
                raise ModelLoadCancelled(f"aborted: {self.display_name}")
        if self._cancelled():
            raise ModelLoadCancelled(f"aborted: {self.display_name}")
        self._loaded = True

    def unload(self):
        self._loaded = False


def _reset():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None


@pytest.fixture
def registered(monkeypatch):
    """M and N registered, with VRAM to spare so nothing is evicted. get_engine
    routes by NAME only against a populated registry; with an empty one it
    serves the active engine instead (single-model mode), which
    would sidestep the reuse path under test entirely."""
    reg = {"M": {"path": "models/M.gguf", "source": "local"},
           "N": {"path": "models/N.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "hint"))
    monkeypatch.setattr("localm.discover.vram_capacity",
                        probe_double({"free": 32 * 1024 ** 3,
                                      "total": 32 * 1024 ** 3}))
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)


async def _await_started(engine, timeout=5.0):
    loop = asyncio.get_running_loop()
    assert await loop.run_in_executor(None, engine.load_started.wait, timeout), \
        "load never started"


async def _strand_stale_cancel_on_reused_engine(engines, make):
    """Drive the REAL sequence that strands a SET cancel event on a REUSED
    engine: M is resident, idle-unloads (kept in _engines), an explicit switch
    back to M starts loading, and a newer explicit switch to N supersedes it.
    Returns once engine M is back in that state, still tracked and unloaded."""
    engM = engines["M"]
    # 1. M resident, then idle-unloaded: the engine object STAYS in _engines.
    engM._loaded = True
    hs._engines["M"] = engM
    hs._engines_lru.append("M")
    hs._inference_sems["M"] = asyncio.Semaphore(1)
    hs._active_model_name = "M"
    hs._engine = engM
    hs._last_activity_per_model["M"] = 0.0       # far idle -> ttl fires
    assert await hs._idle_unload_once(ttl=1) is True
    assert hs._engines.get("M") is engM, "idle-unload must keep the engine object"
    assert not engM.loaded

    # 2. Explicit switch back to M (preempt=True) installs a fresh cancel event.
    task_m = asyncio.create_task(hs.switch_engine("M", make, preempt=True))
    await _await_started(engM)
    # 3. A newer explicit switch to N sets M's cancel event.
    task_n = asyncio.create_task(hs.switch_engine("N", make, preempt=True))
    res_m, res_n = await asyncio.gather(task_m, task_n)
    assert res_m["status"] == "superseded", res_m
    assert res_n["status"] == "loaded", res_n
    # M is still tracked, unloaded, and now holds a SET cancel event.
    assert hs._engines.get("M") is engM
    assert engM._cancel is not None and engM._cancel.is_set()
    return engM


def test_api_load_of_reused_engine_after_preempted_switch_succeeds():
    """An API-routed (preempt=False) load of a model whose engine survived a
    superseded explicit switch must actually LOAD, not report 'superseded' off
    the previous switch's already-fired cancel event."""

    async def scenario():
        _reset()
        never = threading.Event()
        ready = threading.Event(); ready.set()
        engines = {
            "M": CancelHonoringEngine("M", load_gate=never),
            "N": CancelHonoringEngine("N", load_gate=ready),
        }
        make = engines.__getitem__
        engM = await _strand_stale_cancel_on_reused_engine(engines, make)

        # The API path: get_engine -> switch_engine(preempt=False), reusing engM.
        engM.load_started.clear()
        engM._gate = ready              # nothing should stop this load now
        return await hs.switch_engine("M", make, preempt=False)

    res = asyncio.run(scenario())
    assert res["status"] == "loaded", (
        "an API-routed load reused an engine still carrying the PREVIOUS "
        f"switch's fired cancel event and aborted: {res}")


def test_get_engine_does_not_503_after_preempted_switch(registered):
    """The user-visible symptom: every later /v1/chat/completions for that model
    503s 'superseded' indefinitely, on a load that should simply succeed."""

    async def scenario():
        _reset()
        never = threading.Event()
        ready = threading.Event(); ready.set()
        engines = {
            "M": CancelHonoringEngine("M", load_gate=never),
            "N": CancelHonoringEngine("N", load_gate=ready),
        }
        make = engines.__getitem__
        engM = await _strand_stale_cancel_on_reused_engine(engines, make)
        hs._engine_factory = make
        engM.load_started.clear()
        engM._gate = ready
        return await hs.get_engine("M")

    eng = asyncio.run(scenario())
    assert eng.display_name == "M"
    assert eng.loaded


def test_get_engine_stays_broken_forever_is_not_the_contract():
    """The negative case that makes this a real regression test rather than a
    one-shot: the stale event is never cleared by anything, so pre-fix EVERY
    subsequent request keeps failing, not just the first. Drives the API path
    twice and requires both to succeed."""

    async def scenario():
        _reset()
        never = threading.Event()
        ready = threading.Event(); ready.set()
        engines = {
            "M": CancelHonoringEngine("M", load_gate=never),
            "N": CancelHonoringEngine("N", load_gate=ready),
        }
        make = engines.__getitem__
        engM = await _strand_stale_cancel_on_reused_engine(engines, make)
        engM._gate = ready
        first = await hs.switch_engine("M", make, preempt=False)
        # Unload it (engine object kept, as idle-unload does) and load again.
        engM.unload()
        second = await hs.switch_engine("M", make, preempt=False)
        return first, second

    first, second = asyncio.run(scenario())
    assert first["status"] == "loaded", first
    assert second["status"] in ("loaded", "already_active"), second
