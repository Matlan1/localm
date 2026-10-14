# SPDX-License-Identifier: AGPL-3.0-or-later
"""Free VRAM for GGUF sizing on a box whose runtime registers 2+ GPU devices.

The isolated native probe answers only for a runtime with exactly one GPU
device, and on Windows torch is not imported in the server, so the load
device's reading comes from the runtime registry's entry for that device."""
from __future__ import annotations

import sys

import pytest

from localm import discover, gpu_usage
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

GIB = 1024 ** 3
IGPU_TYPE = _loader.GGML_DEV_TYPE_GPU + 1

READINGS = [
    {"index": 0, "name": "AMD Radeon(TM) Graphics", "total": 2 * GIB, "free": 1 * GIB,
     "integrated": True},
    {"index": 1, "name": "AMD Radeon RX 7900 XTX", "total": 24 * GIB, "free": 20 * GIB,
     "integrated": False},
]
REGISTRY = [
    {"index": 0, "name": "ROCm0", "type": IGPU_TYPE, "total": 2 * GIB, "free": 1 * GIB},
    {"index": 1, "name": "ROCm1", "type": _loader.GGML_DEV_TYPE_GPU,
     "total": 24 * GIB, "free": 13 * GIB},
]


@pytest.fixture
def windows_two_gpus(monkeypatch):
    """Windows, torch not imported here, the isolated probe declining (2+ GPU
    devices), a completed list_gpus reading, and a 1-entry split on the
    discrete GPU."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(gpu_usage, "torch_fully_imported", lambda: False)
    monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: False)
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: None)
    monkeypatch.setattr(VramSizingMixin, "_torch_rocm_init_broken", False)
    monkeypatch.setattr(VramSizingMixin, "_torch_vram_read_wedged", False)
    monkeypatch.setattr(discover, "_native_gpu_index_space_is_opaque", lambda: False)
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: [dict(g) for g in READINGS])
    monkeypatch.setattr(discover, "_runtime_device_registry",
                        lambda: [dict(d) for d in REGISTRY])
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"gpu_split_indices": [1]})
    probes = []

    def _list_gpus(*, return_status=False, deadline=None, wait_for_inflight=False):
        probes.append(wait_for_inflight)
        return [dict(g) for g in READINGS], discover.GPU_PROBE_OK

    monkeypatch.setattr(discover, "list_gpus", _list_gpus)
    return probes


def test_free_vram_is_the_load_devices_registry_reading(windows_two_gpus):
    assert VramSizingMixin._free_vram_bytes() == 13 * GIB
    assert windows_two_gpus == [], "probed although a completed reading existed"


def test_total_vram_is_the_load_devices_registry_reading(windows_two_gpus):
    assert VramSizingMixin._total_vram_bytes() == 24 * GIB


def test_without_a_reading_one_probe_numbers_the_devices(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: None)
    assert discover.load_device_memory_isolated() == (13 * GIB, 24 * GIB)
    assert len(windows_two_gpus) == 1


def test_with_the_native_runtime_loaded_and_no_reading_nothing_is_probed(
        windows_two_gpus, monkeypatch):
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: None)
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
    assert discover.load_device_memory_isolated() is None
    assert windows_two_gpus == []


def test_an_integrated_load_device_is_not_read_from_another_device(
        windows_two_gpus, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"main_gpu_index": 0})
    assert discover.load_device_memory_isolated() is None


def test_an_unproven_numbering_answers_none(windows_two_gpus, monkeypatch):
    unflagged = [{k: v for k, v in g.items() if k != "integrated"} for g in READINGS]
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: unflagged)
    assert discover.load_device_memory_isolated() is None


def test_a_registry_whose_totals_differ_answers_none(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(discover, "_runtime_device_registry", lambda: [
        dict(REGISTRY[0]), dict(REGISTRY[1], total=8 * GIB)])
    assert discover.load_device_memory_isolated() is None


def test_an_unreadable_registry_answers_none(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(discover, "_runtime_device_registry", lambda: None)
    assert discover.load_device_memory_isolated() is None
    assert VramSizingMixin._free_vram_bytes() is None


def test_an_opaque_index_space_answers_none(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(discover, "_native_gpu_index_space_is_opaque", lambda: True)
    assert discover.load_device_memory_isolated() is None


def test_a_blind_reading_is_returned_corrected(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: True)
    monkeypatch.setattr(gpu_usage, "device_global_used_bytes",
                        lambda gpus: {g["index"]: 15 * GIB for g in gpus})
    assert VramSizingMixin._free_vram_bytes() == 9 * GIB


def test_a_blind_reading_that_cannot_be_corrected_is_unmeasurable(
        windows_two_gpus, monkeypatch):
    monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: True)
    monkeypatch.setattr(gpu_usage, "device_global_used_bytes", lambda gpus: {})
    assert VramSizingMixin._free_vram_bytes() is None


def test_a_single_gpu_runtime_still_uses_the_isolated_probe(windows_two_gpus, monkeypatch):
    monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: (5 * GIB, 8 * GIB))
    monkeypatch.setattr(discover, "load_device_memory_isolated",
                        lambda config=None: pytest.fail("read the registry needlessly"))
    assert VramSizingMixin._free_vram_bytes() == 5 * GIB
