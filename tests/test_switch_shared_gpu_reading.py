# SPDX-License-Identifier: AGPL-3.0-or-later
"""A cold switch_engine GGUF load reads the GPUs once before the worker spawns.

The first admission pass and the load's sizing share one list_gpus reading; a
pass after an eviction reads again; the reading after the load runs on its own
thread and switch_engine returns without waiting for it. The real admission,
Engine and GgufBackend sizing run; only the GPU probe, the registry and the
worker spawn are stubbed."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from localm import discover
from localm.inference import engine as engine_mod
from localm.inference import http_server as hs
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader
from localm.inference.engine import Engine
from tests.test_eviction_victim_race import GatedEngine
from tests.test_moe_auto_placement import _dense_model
from tests.test_switch_engine_transition_matrix import _seat

GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
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
                        ("_INCONCLUSIVE_LOAD_RETRY_DELAY", 0)):
        monkeypatch.setattr(hs, name, value)
    yield
    _clear()
    discover.end_shared_gpu_reading()


class Box:
    """One 16 GiB GPU seen only through discover._list_gpus_probe.

    ``probes`` records ``(thread name, spawned so far)`` per probe. A probe
    taken after a spawn waits on ``release_after_load`` when ``hold_after_load``
    is set."""

    def __init__(self, free=12 * GIB):
        self.free = free
        self.probes = []
        self.spawns = 0
        self.hold_after_load = False
        self.release_after_load = threading.Event()
        self.after_load_started = threading.Event()

    def probe(self):
        free = self.free() if callable(self.free) else self.free
        self.probes.append((threading.current_thread().name, self.spawns, free))
        if self.spawns and self.hold_after_load:
            self.after_load_started.set()
            self.release_after_load.wait(30)
        return [{"index": 0, "name": "GPU", "total": 16 * GIB, "free": free,
                 "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}]

    def before_spawn(self):
        return [p for p in self.probes if p[1] == 0]


@pytest.fixture
def box(monkeypatch, tmp_path):
    path = _dense_model(tmp_path)
    b = Box()
    b.path = path
    monkeypatch.setattr(discover, "_list_gpus_probe", b.probe)
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: None)
    monkeypatch.setattr(discover, "load_device_memory_isolated", lambda *a, **k: None)
    monkeypatch.setattr(discover, "_runtime_device_registry", lambda: None)
    monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                        staticmethod(lambda: (None, None)))
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    registry = {"m": {"path": str(path), "source": "local"},
                "victim": {"path": str(path), "source": "local"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (str(path), "hint") if name in registry else None)
    monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: None)

    def _spawn(self, params, **kw):
        b.spawns += 1
        return {"n_layers": 5, "kv_bytes_per_token": 0, "supports_images": False}

    monkeypatch.setattr("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                        _spawn)
    yield b
    b.release_after_load.set()


def _factory(box):
    return lambda name: Engine(str(box.path), n_gpu_layers=99, display_name=name)


def _join_vram_report(engine):
    thread = getattr(engine._backend, "_vram_report_thread", None)
    if thread is not None:
        thread.join(30)
        assert not thread.is_alive()


def test_a_cold_load_reads_the_gpus_once_before_the_spawn(box):
    res = asyncio.run(hs.switch_engine("m", _factory(box), preempt=False))
    assert res["status"] == "loaded"
    assert box.spawns == 1
    assert len(box.before_spawn()) == 1, box.probes
    _join_vram_report(hs._engines["m"])
    assert len(box.probes) == 2
    assert getattr(discover._shared_reading, "state", None) is None


def test_switch_engine_returns_before_the_after_load_reading_lands(box, capsys):
    box.hold_after_load = True

    async def _go():
        return await asyncio.wait_for(
            hs.switch_engine("m", _factory(box), preempt=False), 20)

    res = asyncio.run(_go())
    assert res["status"] == "loaded"
    assert box.after_load_started.wait(20)
    assert "GB in use" not in capsys.readouterr().out
    box.release_after_load.set()
    _join_vram_report(hs._engines["m"])
    assert "GB in use / 16.00 GB total" in capsys.readouterr().out


def test_after_an_eviction_admission_and_load_read_the_gpus_again(box):
    victim = GatedEngine("victim")
    _seat("victim", victim)
    box.free = lambda: 16 * GIB // 100 if victim.loaded else 12 * GIB
    res = asyncio.run(hs.switch_engine("m", _factory(box), preempt=False))
    assert res["status"] == "loaded"
    assert victim.unload_calls == 1 and "victim" not in hs._engines
    before = box.before_spawn()
    assert before[0][2] == 16 * GIB // 100
    after_eviction = [p for p in before if p[2] == 12 * GIB]
    # The re-admission pass (vram_capacity, implicit_split_free) and the
    # load's sizing each read the GPUs.
    assert len(after_eviction) >= 3, box.probes
    _join_vram_report(hs._engines["m"])


def test_a_load_that_waited_for_another_load_reads_the_gpus_again(box, monkeypatch):
    """A load that starts after another load took the load lock sizes against
    a reading taken after that load, not the admission's."""
    other_holding = threading.Event()

    def _other_load():
        with engine_mod._LOAD_LOCK:
            other_holding.set()
            time.sleep(0.3)
            box.free = 6 * GIB

    real_fingerprint = hs._routing_latch.fingerprint

    def _fingerprint(name):
        threading.Thread(target=_other_load, daemon=True).start()
        assert other_holding.wait(10)
        return real_fingerprint(name)

    monkeypatch.setattr(hs._routing_latch, "fingerprint", _fingerprint)
    res = asyncio.run(hs.switch_engine("m", _factory(box), preempt=False))
    assert res["status"] == "loaded"
    before = box.before_spawn()
    assert len(before) == 2, box.probes
    assert before[-1][2] == 6 * GIB
    _join_vram_report(hs._engines["m"])


def test_a_load_lock_taken_and_released_after_the_admission_reading_voids_it(box, monkeypatch):
    real_fingerprint = hs._routing_latch.fingerprint

    def _fingerprint(name):
        def _other():
            with engine_mod._LOAD_LOCK:
                pass
        t = threading.Thread(target=_other)
        t.start()
        t.join(10)
        return real_fingerprint(name)

    monkeypatch.setattr(hs._routing_latch, "fingerprint", _fingerprint)
    res = asyncio.run(hs.switch_engine("m", _factory(box), preempt=False))
    assert res["status"] == "loaded"
    assert len(box.before_spawn()) == 2, box.probes
    _join_vram_report(hs._engines["m"])


def test_a_direct_engine_load_prints_the_vram_line_before_it_returns(box, capsys):
    eng = Engine(str(box.path), n_gpu_layers=99, display_name="m")
    eng.load()
    assert getattr(eng._backend, "_vram_report_thread", None) is None
    assert "GB in use / 16.00 GB total" in capsys.readouterr().out
    assert len(box.probes) == 2


def test_the_load_lock_generation_is_even_exactly_when_it_is_free():
    lock = engine_mod._LOAD_LOCK
    start = engine_mod.load_lock_mark()
    assert start is not None and start % 2 == 0
    with lock:
        assert engine_mod.load_lock_mark() is None
        with lock:
            assert engine_mod.load_lock_mark() is None
        assert engine_mod.load_lock_mark() is None
    assert engine_mod.load_lock_mark() == start + 2
    assert lock.acquire(blocking=False)
    lock.release()
    assert engine_mod.load_lock_mark() == start + 4
