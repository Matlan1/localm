# SPDX-License-Identifier: AGPL-3.0-or-later
"""_cuda_device_map: the HF (transformers) backend's device_map/max_memory builder, honouring gpu_split_indices / main_gpu_index the same way the GGUF backend's native params do (see localm.discover.apply_gpu_split / apply_main_gpu) - closing the gap where this backend used to hardcode device_map='auto..."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm.inference.backends._hf_worker import _cuda_device_map

_HEADROOM = int(0.5e9)   # must match the headroom baked into _cuda_device_map
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_unless_torch_and_accelerate_available() -> None:
    """Skip (without ever importing torch/accelerate IN THIS PROCESS) when either is not installed at all."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    if importlib.util.find_spec("accelerate") is None:
        pytest.skip("accelerate not installed")


def _fake_torch(mem_by_device=None):
    """A stand-in for the torch module: torch.cuda.mem_get_info(idx) returns mem_by_device[idx] (an (free, total) tuple), or raises when the mapped value is an Exception instance."""
    fake = MagicMock()

    def _mem_get_info(idx):
        val = (mem_by_device or {})[idx]
        if isinstance(val, Exception):
            raise val
        return val

    fake.cuda.mem_get_info.side_effect = _mem_get_info
    return fake


class TestNoSplitNoMainGpu:
    """Neither gpu_split_indices nor main_gpu_index configured -> today's unchanged default, device_map 'auto' across every visible device."""

    def test_defaults_to_auto_with_config_kwarg(self, monkeypatch):
        # resolve_gpu_split short-circuits on an empty gpu_split_indices before
        # ever calling list_gpus(), and main_gpu_index is unset so
        # resolve_main_gpu_index is never called either - list_gpus() must not
        # be consulted at all.
        calls = []
        monkeypatch.setattr("localm.discover.list_gpus", lambda: calls.append(1) or [])
        result = _cuda_device_map(
            _fake_torch(), config={"gpu_split_indices": None, "main_gpu_index": None})
        assert result == {"device_map": "auto"}
        assert calls == []

    def test_defaults_to_auto_via_load_config(self, monkeypatch):
        # config=None -> _cuda_device_map reads localm.config.load_config() itself.
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": None, "main_gpu_index": None})
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        assert _cuda_device_map(_fake_torch()) == {"device_map": "auto"}


class TestMainGpuIndexOnly:
    """main_gpu_index configured, no split -> confine the model to that ONE device, WITHOUT losing device_map='auto''s CPU-offload fallback (REG-528)."""

    def test_confines_to_resolved_device_but_keeps_cpu_offload(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {1: (8_000_000_000, 16_000_000_000)}
        result = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 1})

        assert result["device_map"] == "auto", (
            "a literal {'': idx} map disables accelerate's sharding/offload "
            "entirely, so an oversized model OOMs instead of loading")
        # Only the chosen GPU is offered: every other card is absent from
        # max_memory, which is what confines "auto" to it.
        assert [k for k in result["max_memory"] if isinstance(k, int)] == [1]
        assert result["max_memory"][1] == 8_000_000_000 - _HEADROOM
        assert result["max_memory"].get("cpu", 0) > 0, (
            "without a cpu budget accelerate sends the overflow to disk, which "
            "needs an offload_folder - the CPU fallback 'auto' gave is gone")

    def test_invalid_main_gpu_index_falls_back_to_resolved_zero(self, monkeypatch, caplog):
        # resolve_main_gpu_index is the REAL function: index 7 isn't among the
        # detected devices, so it drops to 0 with a warning - _cuda_device_map
        # must confine to whatever resolve_main_gpu_index actually RETURNS, not
        # the raw configured value. Pin non-Vulkan so that membership validation
        # actually runs, regardless of what native backend is provisioned in
        # the ambient environment (see test_discover.py::TestResolveMainGpuIndex).
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        mem = {0: (8_000_000_000, 16_000_000_000)}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 7})
        assert result["device_map"] == "auto"
        assert [k for k in result["max_memory"] if isinstance(k, int)] == [0]
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_unreadable_free_vram_still_honors_the_selection(self, monkeypatch, caplog):
        """If free VRAM cannot be read for the chosen device there is no budget to build, so fall back to pinning it (the pre-existing behavior) rather than silently ignoring the user's explicit GPU selection - and say why (rule 5), since that fallback is the one case with no CPU offload."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 1})
        assert result == {"device_map": {"": 1}}
        assert any("main_gpu_index" in r.message for r in caplog.records)


class TestGpuSplitSuccess:
    """gpu_split_indices configured with 2+ valid devices and mem_get_info succeeding for each -> 'auto' plus a max_memory dict with a headroom-adjusted entry per device."""

    def test_two_devices_produce_max_memory_with_headroom(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: (8_000_000_000, 16_000_000_000), 1: (4_000_000_000, 8_000_000_000)}
        result = _cuda_device_map(
            _fake_torch(mem), config={"gpu_split_indices": [0, 1]})

        assert result["device_map"] == "auto"
        assert {k for k in result["max_memory"] if isinstance(k, int)} == {0, 1}
        for idx, (free, _total) in mem.items():
            got = result["max_memory"][idx]
            assert isinstance(got, int)
            assert got == free - _HEADROOM
            assert got < free   # strictly less than the raw free bytes (headroom)

    def test_split_also_keeps_the_cpu_offload_fallback(self, monkeypatch):
        """The split path has the same gap as the pinned one (REG-528): a GPU-only max_memory removes the 'cpu' entry that plain 'auto' gets from get_max_memory(), so a model too big for the split lands on 'disk' and needs an offload_folder instead of spilling to CPU."""
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: (8_000_000_000, 16_000_000_000), 1: (4_000_000_000, 8_000_000_000)}
        result = _cuda_device_map(
            _fake_torch(mem), config={"gpu_split_indices": [0, 1]})
        assert result["max_memory"].get("cpu", 0) > 0

    def test_three_devices_all_readable(self, monkeypatch):
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}])
        mem = {0: (8_000_000_000, 16_000_000_000),
               1: (4_000_000_000, 8_000_000_000),
               2: (2_000_000_000, 4_000_000_000)}
        result = _cuda_device_map(
            _fake_torch(mem), config={"gpu_split_indices": [0, 1, 2]})
        assert result["device_map"] == "auto"
        assert {k: v for k, v in result["max_memory"].items() if isinstance(k, int)} == {
            0: 7_500_000_000, 1: 3_500_000_000, 2: 1_500_000_000}
        assert result["max_memory"].get("cpu", 0) > 0


def _run_isolated_script(script: str) -> "subprocess.CompletedProcess[str]":
    """Run *script* in a genuinely FRESH child ``python.exe`` process (the same interpreter/venv as this test run, via ``sys.executable``) and return the completed process so callers can assert on returncode/stdout/stderr."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=120,
    )


def _assert_subprocess_ok(result: "subprocess.CompletedProcess[str]") -> None:
    assert result.returncode == 0, (
        f"isolated accelerate check subprocess failed (returncode="
        f"{result.returncode}):\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")


# Self-contained (no imports from this test module - it runs in a separate
# interpreter, see _run_isolated_script above): builds the same over-budget
# model and _cuda_device_map-derived max_memory as the old in-process version,
# then drives the REAL accelerate.infer_auto_device_map against it and asserts
# the overflow lands on CPU, not disk.
_CPU_OFFLOAD_CHILD_SCRIPT = """
from unittest.mock import MagicMock

from localm.inference.backends._hf_worker import _cuda_device_map
import localm.discover as discover
from torch import nn
from accelerate.utils import infer_auto_device_map

_HEADROOM = int(0.5e9)   # must match the headroom baked into _cuda_device_map


def _fake_torch(mem_by_device):
    fake = MagicMock()

    def _mem_get_info(idx):
        val = mem_by_device[idx]
        if isinstance(val, Exception):
            raise val
        return val

    fake.cuda.mem_get_info.side_effect = _mem_get_info
    return fake


model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512),
                       nn.Linear(512, 512), nn.Linear(512, 512))
total = sum(p.numel() * p.element_size() for p in model.parameters())
gpu_budget = int(total // 4 * 1.5)   # room for roughly one block

discover.list_gpus = lambda: [{"index": 0}]
# Free VRAM chosen so the resulting budget only fits part of the model.
mem = {0: (gpu_budget + _HEADROOM, 16_000_000_000)}
kwargs = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 0})

device_map = infer_auto_device_map(model, max_memory=kwargs["max_memory"])
placements = set(device_map.values())
assert "cpu" in placements, f"overflow did not reach CPU: {dict(device_map)}"
assert "disk" not in placements, (
    "overflow went to disk, which needs an offload_folder; a cpu budget "
    f"should absorb it first: {dict(device_map)}")
print("CPU_OFFLOAD_OK", dict(device_map))
"""

# The negative control: the identical over-budget model, but WITHOUT the
# "cpu" max_memory entry - proves the entry above is what does the work (and
# that this check could fail).
_DISK_CONTROL_CHILD_SCRIPT = """
from torch import nn
from accelerate.utils import infer_auto_device_map

model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512),
                       nn.Linear(512, 512), nn.Linear(512, 512))
total = sum(p.numel() * p.element_size() for p in model.parameters())
gpu_budget = int(total // 4 * 1.5)

device_map = infer_auto_device_map(model, max_memory={0: gpu_budget})   # no "cpu" key
placements = set(device_map.values())
assert "disk" in placements, dict(device_map)
print("DISK_CONTROL_OK", dict(device_map))
"""


class TestRealAccelerateHonoursTheMapping:
    """The shape assertions above cannot see what accelerate DOES with the mapping - which is exactly why the suite missed REG-528 (it validated {'device_map': {'': 1}} as correct without ever observing that a literal map offloads nothing)."""

    def test_our_max_memory_actually_offloads_overflow_to_cpu(self):
        _skip_unless_torch_and_accelerate_available()

        result = _run_isolated_script(_CPU_OFFLOAD_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "CPU_OFFLOAD_OK" in result.stdout

    def test_a_cpu_less_budget_would_send_overflow_to_disk(self):
        """The negative control proving the 'cpu' entry is what does the work (and that this test could fail): the same budget WITHOUT it maps the overflow to disk instead."""
        _skip_unless_torch_and_accelerate_available()

        result = _run_isolated_script(_DISK_CONTROL_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "DISK_CONTROL_OK" in result.stdout


class TestSurvivesPriorLlamaCppNativeLoad:
    """Negative-oracle regression proof for issue #698: loading llama.cpp's bundled native runtime into THIS test process (the same load tests/test_gpu_split_wiring.py triggers, via a real, unmocked LlamaCpp construction) must not break the isolated accelerate check above."""

    def test_survives_prior_llamacpp_native_load(self):
        _skip_unless_torch_and_accelerate_available()

        from localm.inference.backends.llamacpp import _loader
        try:
            _loader.load_lib()
        except RuntimeError as e:
            pytest.skip(
                f"native llama.cpp runtime not provisioned on this box: {e}")

        # Also import LlamaCpp itself, matching exactly what
        # tests/test_gpu_split_wiring.py imports at module scope (harmless
        # here since the lib is already loaded above by the explicit
        # load_lib() call - this line documents that the same code path is
        # exercised, not just an equivalent one).
        from localm.inference.backends.llamacpp.llama import LlamaCpp  # noqa: F401

        result = _run_isolated_script(_CPU_OFFLOAD_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "CPU_OFFLOAD_OK" in result.stdout


class TestGpuSplitMemGetInfoFailures:
    """torch.cuda.mem_get_info raising for enough devices that fewer than 2 end up usable -> falls back to whichever of the device_map branches applies, with a warning logged."""

    def test_falls_back_to_auto_when_only_one_device_readable(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: (8_000_000_000, 16_000_000_000), 1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(mem), config={"gpu_split_indices": [0, 1]})
        assert result == {"device_map": "auto"}
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_falls_back_to_main_gpu_pin_when_also_configured(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {0: RuntimeError("no driver"), 1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(mem),
                config={"gpu_split_indices": [0, 1], "main_gpu_index": 1})
        assert result == {"device_map": {"": 1}}
        assert any("gpu_split_indices" in r.message for r in caplog.records)


class TestGpuSplitDropsUnknownDevice:
    """An index resolve_gpu_split() itself would drop (unmatched by list_gpus()) - confirms _cuda_device_map composes with the REAL resolve_gpu_split rather than trusting the raw config."""

    def test_unknown_device_dropped_leaves_single_gpu_default(self, monkeypatch, caplog):
        # list_gpus() only reports device 0; resolve_gpu_split drops the unknown
        # index 99, leaving a single valid device (< 2) -> no split at all.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(), config={"gpu_split_indices": [0, 99]})
        assert result == {"device_map": "auto"}
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_unknown_device_dropped_then_main_gpu_pin_applies(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(
                _fake_torch(),
                config={"gpu_split_indices": [0, 99], "main_gpu_index": 0})
        assert result == {"device_map": {"": 0}}
        assert any("gpu_split_indices" in r.message for r in caplog.records)
