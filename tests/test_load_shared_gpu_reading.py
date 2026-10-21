# SPDX-License-Identifier: AGPL-3.0-or-later
"""One model load's sizing reads the GPUs through one list_gpus probe.

Between discover.start_shared_gpu_reading() and end_shared_gpu_reading() a
thread's list_gpus calls after one that completed OK answer with that reading.
GgufBackend.load() opens it for its sizing and _load_native ends it before the
worker spawns, so the after-load reading is a new probe."""
from __future__ import annotations

import threading

import pytest

from localm import discover
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _loader

GIB = 1024 ** 3


def _reading(free=12 * GIB):
    return [{"index": 0, "name": "GPU", "total": 16 * GIB, "free": free,
             "free_scope": discover.FREE_SCOPE_DEVICE, "integrated": False}]


@pytest.fixture
def probes(monkeypatch):
    calls = []

    def _probe():
        calls.append(threading.current_thread().name)
        return _reading(free=(12 - len(calls)) * GIB)

    monkeypatch.setattr(discover, "_list_gpus_probe", _probe)
    yield calls
    discover.end_shared_gpu_reading()


def test_a_shared_reading_probes_once(probes):
    discover.start_shared_gpu_reading()
    first = discover.list_gpus()
    second, status = discover.list_gpus(return_status=True)
    discover.end_shared_gpu_reading()
    assert len(probes) == 1
    assert second == first and status == discover.GPU_PROBE_OK


def test_the_shared_reading_hands_out_copies(probes):
    discover.start_shared_gpu_reading()
    discover.list_gpus()[0]["free"] = 0
    assert discover.list_gpus()[0]["free"] == 11 * GIB


def test_ending_it_probes_again(probes):
    discover.start_shared_gpu_reading()
    discover.list_gpus()
    discover.end_shared_gpu_reading()
    assert discover.list_gpus()[0]["free"] == 10 * GIB
    assert len(probes) == 2


def test_without_a_shared_reading_every_call_probes(probes):
    discover.list_gpus()
    discover.list_gpus()
    assert len(probes) == 2


def test_a_probe_that_did_not_complete_ok_is_not_shared(monkeypatch):
    statuses = iter([([], discover.GPU_PROBE_TIMEOUT), (_reading(), discover.GPU_PROBE_OK)])
    calls = []

    def _driver(deadline, wait_for_inflight=False):
        calls.append(1)
        return next(statuses)

    monkeypatch.setattr(discover, "_list_gpus_with_status", _driver)
    discover.start_shared_gpu_reading()
    try:
        assert discover.list_gpus(return_status=True) == ([], discover.GPU_PROBE_TIMEOUT)
        assert discover.list_gpus(return_status=True)[1] == discover.GPU_PROBE_OK
        assert discover.list_gpus(return_status=True)[1] == discover.GPU_PROBE_OK
    finally:
        discover.end_shared_gpu_reading()
    assert len(calls) == 2


def test_another_thread_is_not_affected(probes):
    discover.start_shared_gpu_reading()
    discover.list_gpus()
    seen = []
    t = threading.Thread(target=lambda: seen.append(discover.list_gpus()))
    t.start()
    t.join(30)
    discover.list_gpus()
    assert len(probes) == 2
    assert seen and seen[0][0]["free"] == 10 * GIB


@pytest.fixture
def load_box(monkeypatch, tmp_path, probes):
    """A real GGUF, one GPU seen only through the patched list_gpus probe, the
    other VRAM readings unmeasurable, and a stubbed worker spawn that records
    how many probes ran before it."""
    from tests.test_moe_auto_placement import _dense_model
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: None)
    monkeypatch.setattr(discover, "load_device_memory_isolated", lambda *a, **k: None)
    monkeypatch.setattr(discover, "_runtime_device_registry", lambda: None)
    monkeypatch.setattr(GgufBackend, "_free_total_vram_bytes",
                        staticmethod(lambda: (None, None)))
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    at_spawn = []

    def _spawn(self, params, **kw):
        at_spawn.append(len(probes))
        return {"n_layers": 5, "kv_bytes_per_token": 0, "supports_images": False}

    monkeypatch.setattr("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                        _spawn)
    backend = GgufBackend(str(_dense_model(tmp_path)), n_ctx=512, n_gpu_layers=99,
                          n_gpu_layers_auto=True, ctx_auto=True)
    return backend, at_spawn


def test_a_load_sizes_against_one_probe_and_reads_again_after_the_spawn(load_box, probes):
    backend, at_spawn = load_box
    backend.load()
    assert at_spawn == [1]
    assert len(probes) == 2
    assert getattr(discover._shared_reading, "state", None) is None


def test_a_refused_load_leaves_no_shared_reading_behind(load_box, probes, monkeypatch):
    backend, at_spawn = load_box

    def _refuse(self):
        discover.list_gpus()
        raise RuntimeError("does not fit")

    monkeypatch.setattr(GgufBackend, "_check_vram", _refuse)
    with pytest.raises(RuntimeError, match="does not fit"):
        backend.load()
    assert at_spawn == []
    assert getattr(discover._shared_reading, "state", None) is None
    discover.list_gpus()
    assert len(probes) == 2
