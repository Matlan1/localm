# SPDX-License-Identifier: AGPL-3.0-or-later
"""_gpu_registry_sync() must never run on the server event loop.

It does blocking work on every model load/unload: a registry temp-file
write + os.replace, a _model_file_size() stat/rglob walk, and - when a non-zero
main_gpu_index is configured - _current_gpu_index() -> resolve_main_gpu_index()
-> discover.list_gpus(), a real torch/nvidia-smi hardware probe bounded to a 4s
deadline. On the single event loop that stalls EVERY concurrent request and
stream for up to that deadline.

These tests assert the property directly - the sync work runs on a thread OTHER
than the event-loop thread - rather than trying to measure a stall.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localm.inference import http_server as hs


class _ThreadProbe:
    """Stands in for _gpu_registry_sync, recording which thread it ran on."""

    def __init__(self):
        self.threads = []

    def __call__(self):
        self.threads.append(threading.get_ident())


@pytest.fixture
def probe(monkeypatch):
    p = _ThreadProbe()
    monkeypatch.setattr(hs, "_gpu_registry_sync", p)
    # No real hardware probe / VRAM wait in a unit test.
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": 32 * 1024 ** 3,
                                             "total": 32 * 1024 ** 3})
    monkeypatch.setattr("localm.discover.gpu_split_shortfall",
                        lambda need, **k: ([], False)
                        if k.get("return_shares_adaptive") else [])
    monkeypatch.setattr("localm.discover.split_device_count", lambda: 1)
    monkeypatch.setattr("localm.vram.wait_for_vram_release",
                        lambda free_fn, before_bytes=None: (0, before_bytes))
    for d in (hs._engines, hs._engines_lru, hs._inference_sems,
              hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    return p


class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, cancel):
        pass


def _install_loaded(name):
    eng = FakeEngine(name)
    eng.load()
    hs._engines[name] = eng
    hs._engines_lru.append(name)
    hs._inference_sems[name] = asyncio.Semaphore(1)
    hs._active_model_name = name
    hs._engine = eng
    hs._inference_sem = hs._inference_sems[name]
    return eng


def _assert_off_loop(probe, loop_thread, where):
    assert probe.threads, f"_gpu_registry_sync never ran in {where}"
    assert all(t != loop_thread for t in probe.threads), (
        f"{where} ran the gpu-registry sync (registry file I/O + a GPU driver "
        f"probe) ON the event loop thread, stalling every concurrent request")


def test_switch_engine_syncs_registry_off_the_loop(probe):
    """Every successful load hits this path."""

    async def scenario():
        engines = {"A": FakeEngine("A")}
        await hs.switch_engine("A", engines.__getitem__)
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "switch_engine")


def test_unload_all_models_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_all_models()
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_all_models")


def test_unload_one_model_syncs_registry_off_the_loop(probe):
    async def scenario():
        _install_loaded("A")
        await hs.unload_one_model("A")
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "unload_one_model")


def test_unload_embedder_if_matches_syncs_registry_off_the_loop(probe, monkeypatch, tmp_path):
    """The targeted-unload counterpart: loaded_path(), the active_requests()
    precheck, reset_embedder(force=False) and the VRAM wait are already
    offloaded, and the registry sync must be too."""
    model = tmp_path / "emb.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: str(model))
    monkeypatch.setattr("localm.inference.embedder.active_requests", lambda: 0)
    monkeypatch.setattr("localm.inference.embedder.reset_embedder",
                        lambda force=True: True)
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"emb": {"path": str(model), "source": "local"}})
    monkeypatch.setattr("localm.model_manager._entry_path", lambda entry: str(model))

    async def scenario():
        loop = asyncio.get_running_loop()
        res = await hs._unload_embedder_if_matches("emb", loop)
        assert res is not None and res["status"] == "unloaded", res
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())
    _assert_off_loop(probe, loop_thread, "_unload_embedder_if_matches")


# --------------------------------------------------------------------------- #
#  The heartbeat's failure warning is throttled                                #
#                                                                              #
#  A heartbeat failure is usually PERSISTENT (an unwritable registry path, a   #
#  wedged driver probe). Warning unconditionally on a 20s tick emitted three   #
#  lines a minute, each with a full traceback, for the life of the server -    #
#  which is how the one line that mattered gets buried.                        #
# --------------------------------------------------------------------------- #

import logging


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _run_heartbeat_until(monkeypatch, sync_impl, target_calls):
    """Drive the real loop with a fast interval until *sync_impl* has been
    called *target_calls* times, then cancel it. Returns the captured records.

    Deterministic by construction rather than by clock: the stand-in counts its
    OWN calls and signals when it has been driven enough, so an assertion about
    "how many lines across N ticks" is a fact about the throttle rather than a
    race. The loop is the REAL one - only the tick period is overridden.
    """
    from localm.debuglog import logger as _dbg

    calls = {"n": 0}
    enough = threading.Event()

    def _counting():
        calls["n"] += 1
        if calls["n"] >= target_calls:
            enough.set()
        return sync_impl(calls["n"])

    monkeypatch.setattr(hs, "_gpu_registry_sync", _counting)

    handler = _Capture()
    prev_level = _dbg.level
    _dbg.addHandler(handler)
    _dbg.setLevel(logging.DEBUG)

    async def scenario():
        task = asyncio.create_task(hs._gpu_registry_heartbeat_loop(interval=0.01))
        try:
            await asyncio.get_running_loop().run_in_executor(None, enough.wait, 10)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(scenario())
    finally:
        _dbg.removeHandler(handler)
        _dbg.setLevel(prev_level)

    assert calls["n"] >= target_calls, (
        f"the heartbeat only ticked {calls['n']} times - this test never "
        "exercised the throttle")
    return handler.records


def _heartbeat_warnings(records):
    return [r for r in records
            if r[0] == "WARNING" and "gpu-registry heartbeat failed" in r[1]]


def test_a_persistently_failing_heartbeat_warns_once_then_throttles(monkeypatch):
    def _always_fails(n):
        raise OSError("registry path is unwritable")

    records = _run_heartbeat_until(monkeypatch, _always_fails, target_calls=5)

    warnings = _heartbeat_warnings(records)
    throttled = [r for r in records
                 if r[0] == "DEBUG" and "heartbeat still failing" in r[1]]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING across 5 failing ticks, got "
        f"{len(warnings)}: {warnings}")
    assert throttled, "the repeats vanished entirely instead of dropping to DEBUG"
    # The throttled line still has to identify the cause, or a CHANGE of cause
    # after the first warning would be invisible.
    assert "OSError" in throttled[0][1]


def test_a_heartbeat_that_recovers_warns_again_on_a_LATER_failure(monkeypatch):
    """A success must re-arm the warning.

    This is the half a plain "only ever warn once" flag gets wrong, and it is
    the difference between a throttle and a permanent silence: a second,
    unrelated outage hours later would otherwise never be reported at all.
    """
    def _fails_recovers_fails(n):
        if n == 3:
            return None          # one good tick in the middle
        raise OSError("registry path is unwritable")

    records = _run_heartbeat_until(monkeypatch, _fails_recovers_fails, target_calls=6)

    warnings = _heartbeat_warnings(records)
    assert len(warnings) == 2, (
        f"expected a second WARNING after the recovery, got {len(warnings)}")
