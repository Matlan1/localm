# SPDX-License-Identifier: AGPL-3.0-or-later
"""Concurrent loads of DIFFERENT models must not preempt each other.

The single-model preemption globals (_switch_desired / _switch_cancel) apply to
every /v1 request once switch_engine is the loader behind get_engine, so any
load-triggering request for model B sets _switch_desired="B" and fires
_switch_cancel for an in-flight load of model A, aborting it - and get_engine
turns that into HTTP 503 "superseded", so independent clients loading different
models cancel each other. Preemption must fire only for an explicit user
model-switch (GUI), never for API-routed loads.
"""

import asyncio
import threading

import pytest

from localm.inference import http_server as hs
from localm.inference.backends.base import ModelLoadCancelled
from tests.conftest import probe_double


class FakeEngine:
    def __init__(self, name, gate=None):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0
        self._cancel = None
        self._gate = gate  # threading.Event; when set, load may complete
        self.load_started = threading.Event()   # set the moment load() is entered

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, ev):
        self._cancel = ev

    def load(self):
        self.load_started.set()
        # Honour cancellation exactly like a real backend load does.
        if self._gate is not None:
            while not self._gate.wait(0.005):
                if self._cancel is not None and self._cancel.is_set():
                    raise ModelLoadCancelled()
        if self._cancel is not None and self._cancel.is_set():
            raise ModelLoadCancelled()
        self._loaded = True

    def unload(self):
        self._loaded = False


@pytest.fixture
def multi(monkeypatch):
    reg = {"model-a": {"path": "models/model-a.gguf", "source": "local"},
           "model-b": {"path": "models/model-b.gguf", "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda n: (f"models/{n}.gguf", "h"))
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return reg


def test_concurrent_different_model_loads_do_not_supersede(multi, monkeypatch):
    gate_a = threading.Event()
    engines = {"model-a": FakeEngine("model-a", gate_a),
               "model-b": FakeEngine("model-b")}
    monkeypatch.setattr(hs, "_engine_factory", lambda n: engines[n])

    async def scenario():
        loop = asyncio.get_running_loop()
        ta = asyncio.create_task(hs.get_engine("model-a"))
        try:
            # A is inside its gated load() before B is even requested.
            started = await loop.run_in_executor(
                None, engines["model-a"].load_started.wait, 2.0)
            assert started, "model-a's load never started"
            tb = asyncio.create_task(hs.get_engine("model-b"))
            # B runs to completion while A is still held in load().
            done, _ = await asyncio.wait({tb}, timeout=5.0)
            assert tb in done, "model-b's load did not finish while model-a was mid-load"
        finally:
            gate_a.set()                   # release A
        return await asyncio.gather(ta, tb, return_exceptions=True)

    results = asyncio.run(scenario())

    for r in results:
        assert not isinstance(r, BaseException), \
            f"a concurrent API load was falsely superseded/failed: {r!r}"
    assert engines["model-a"].loaded and engines["model-b"].loaded
