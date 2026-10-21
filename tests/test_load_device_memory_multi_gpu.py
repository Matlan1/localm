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
        gpus = [dict(g) for g in READINGS]
        return (gpus, discover.GPU_PROBE_OK) if return_status else gpus

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


TWO_DISCRETE = [
    {"index": 0, "name": "AMD Radeon RX 7800 XT", "total": 16 * GIB, "free": 15 * GIB,
     "integrated": False},
    {"index": 1, "name": "AMD Radeon RX 7800 XT", "total": 16 * GIB, "free": 15 * GIB,
     "integrated": False},
]
TWO_DISCRETE_REGISTRY = [
    {"index": 0, "name": "ROCm0", "type": _loader.GGML_DEV_TYPE_GPU,
     "total": 16 * GIB, "free": 14 * GIB},
    {"index": 1, "name": "ROCm1", "type": _loader.GGML_DEV_TYPE_GPU,
     "total": 16 * GIB, "free": 12 * GIB},
]


@pytest.fixture
def windows_two_discrete(windows_two_gpus, monkeypatch):
    """Two discrete 16 GB GPUs in both the torch reading and the runtime
    registry, no gpu_split_indices configured."""
    monkeypatch.setattr(discover, "last_gpu_reading",
                        lambda: [dict(g) for g in TWO_DISCRETE])
    monkeypatch.setattr(discover, "_runtime_device_registry",
                        lambda: [dict(d) for d in TWO_DISCRETE_REGISTRY])
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    return windows_two_gpus


def test_an_implicit_split_over_two_gpus_is_not_read_as_one(windows_two_discrete):
    assert discover.load_device_memory_isolated() is None
    assert VramSizingMixin._free_vram_bytes() is None
    assert VramSizingMixin._total_vram_bytes() is None


def test_a_configured_two_gpu_split_is_not_read_as_one(windows_two_discrete, monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"gpu_split_indices": [0, 1], "main_gpu_index": 0})
    assert discover.load_device_memory_isolated() is None
    assert VramSizingMixin._total_vram_bytes() is None


def test_a_one_entry_split_reads_its_device_beside_another_discrete_gpu(
        windows_two_discrete, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"gpu_split_indices": [1]})
    assert discover.load_device_memory_isolated() == (12 * GIB, 16 * GIB)


def test_a_one_entry_split_naming_no_detected_gpu_is_not_read(
        windows_two_discrete, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"gpu_split_indices": [5]})
    assert discover.load_device_memory_isolated() is None


def test_a_runtime_keeping_the_integrated_gpu_is_not_read_as_one(
        windows_two_gpus, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    monkeypatch.setattr(discover, "_runtime_device_registry", lambda: [
        dict(REGISTRY[0], type=_loader.GGML_DEV_TYPE_GPU), dict(REGISTRY[1])])
    assert discover.load_device_memory_isolated() is None


def test_a_lone_discrete_gpu_beside_an_integrated_one_is_read_with_no_split(
        windows_two_gpus, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert discover.load_device_memory_isolated() == (13 * GIB, 24 * GIB)


def test_the_figures_come_from_the_registry_not_the_last_reading(
        windows_two_discrete, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"gpu_split_indices": [0]})
    stale = [dict(g, free=1 * GIB) for g in TWO_DISCRETE]
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: stale)
    assert discover.load_device_memory_isolated() == (14 * GIB, 16 * GIB)


def test_check_vram_does_not_refuse_a_model_that_fits_across_two_gpus(
        windows_two_discrete, monkeypatch, tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False, n_ctx=4096)
    b._model_bytes = lambda: 20 * GIB
    b.effective_gpu_layers = 99
    monkeypatch.setattr(GgufBackend, "_split_free_total_bytes",
                        classmethod(lambda cls: (None, None, 0)))
    b._check_vram()


class _FakeCuda:
    def __init__(self, count):
        self.count = count
        self.read = []

    def is_available(self):
        return True

    def device_count(self):
        return self.count

    def mem_get_info(self, idx):
        self.read.append(idx)
        return (10 + idx) * GIB, 16 * GIB


@pytest.fixture
def torch_resident(monkeypatch):
    """Linux with torch imported in this process (the torch read path), two
    discrete GPUs in the last reading, no split configured, and the isolated
    probe and the per-device registry reading both declining."""
    import types
    cuda = _FakeCuda(2)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: False)
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    monkeypatch.setattr(_loader, "gpu_memory_isolated", lambda: None)
    monkeypatch.setattr(discover, "load_device_memory_isolated", lambda *a, **k: None)
    monkeypatch.setattr(VramSizingMixin, "_torch_rocm_init_broken", False)
    monkeypatch.setattr(VramSizingMixin, "_torch_vram_read_wedged", False)
    monkeypatch.setattr(discover, "_native_gpu_index_space_is_opaque", lambda: False)
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: [dict(g) for g in TWO_DISCRETE])
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    return cuda


def test_the_torch_read_does_not_answer_for_an_implicit_split(torch_resident):
    assert VramSizingMixin._torch_free_total_uncapped() == (None, None)
    assert VramSizingMixin._free_vram_bytes() is None
    assert VramSizingMixin._total_vram_bytes() is None
    assert torch_resident.read == []


def test_the_torch_read_does_not_answer_for_a_configured_two_gpu_split(
        torch_resident, monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"gpu_split_indices": [0, 1], "main_gpu_index": 1})
    assert VramSizingMixin._torch_free_total_uncapped() == (None, None)
    assert torch_resident.read == []


def test_the_torch_read_answers_for_a_one_entry_split(torch_resident, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"gpu_split_indices": [1]})
    assert VramSizingMixin._torch_free_total_uncapped() == (11 * GIB, 16 * GIB)
    assert torch_resident.read[-1] == 1


def test_the_torch_read_answers_for_a_lone_discrete_gpu_beside_an_integrated_one(
        torch_resident, monkeypatch):
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: [dict(g) for g in READINGS])
    assert VramSizingMixin._torch_free_total_uncapped() == (11 * GIB, 16 * GIB)
    assert torch_resident.read == [1]


@pytest.mark.parametrize("count,expected", [(1, (10 * GIB, 16 * GIB)), (2, (None, None))])
def test_with_no_reading_the_torch_device_count_decides(
        torch_resident, monkeypatch, count, expected):
    monkeypatch.setattr(discover, "last_gpu_reading", lambda: None)
    torch_resident.count = count
    assert VramSizingMixin._torch_free_total_uncapped() == expected


def test_check_vram_with_torch_does_not_refuse_a_model_that_fits_across_two_gpus(
        torch_resident, monkeypatch, tmp_path):
    from localm.inference.backends.gguf import GgufBackend
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=99, n_gpu_layers_auto=False, n_ctx=4096)
    b._model_bytes = lambda: 20 * GIB
    b.effective_gpu_layers = 99
    monkeypatch.setattr(GgufBackend, "_split_free_total_bytes",
                        classmethod(lambda cls: (None, None, 0)))
    b._check_vram()
