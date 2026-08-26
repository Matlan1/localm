# SPDX-License-Identifier: AGPL-3.0-or-later
"""_cuda_device_map: the HF (transformers) backend's device_map/max_memory builder,
honouring gpu_split_indices / main_gpu_index the same way the GGUF backend's native
params do (see localm.discover.apply_gpu_split / apply_main_gpu).

resolve_gpu_split / resolve_main_gpu_index are the REAL functions here, not mocked;
only localm.discover.list_gpus (their shared device-detection dependency) and
torch.cuda.mem_get_info are faked, mirroring the style in tests/test_hf_device_xpu.py
and tests/test_discover.py."""

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
    """Skip (without ever importing torch/accelerate IN THIS PROCESS) when
    either is not installed at all.

    NOT ``pytest.importorskip("torch")``: that helper's try/except only catches
    ``ImportError``, so an ``import torch`` that instead raises
    ``OSError: [WinError 127] ...`` propagates as a test failure rather than a
    skip. Its ``__import__`` call also runs in this (possibly already
    contaminated) pytest worker process, which is the very in-process import
    this file avoids. ``importlib.util.find_spec`` with a bare (non-dotted)
    module name locates the module via the path finders without executing its
    code, so torch never lands in sys.modules from this call."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    if importlib.util.find_spec("accelerate") is None:
        pytest.skip("accelerate not installed")


def _fake_torch(mem_by_device=None):
    """A stand-in for the torch module: torch.cuda.mem_get_info(idx) returns
    mem_by_device[idx] (an (free, total) tuple), or raises when the mapped value
    is an Exception instance. A missing idx raises KeyError, same as a real bug
    in the test (every device _cuda_device_map queries must be accounted for)."""
    fake = MagicMock()

    def _mem_get_info(idx):
        val = (mem_by_device or {})[idx]
        if isinstance(val, Exception):
            raise val
        return val

    fake.cuda.mem_get_info.side_effect = _mem_get_info
    return fake


class TestNoSplitNoMainGpu:
    """Neither gpu_split_indices nor main_gpu_index configured -> today's
    unchanged default, device_map "auto" across every visible device."""

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
    """main_gpu_index configured, no split -> confine the model to that ONE
    device, WITHOUT losing device_map="auto"'s CPU-offload fallback.

    A literal map (``device_map={"": idx}``) puts the whole model on idx:
    accelerate infers nothing and offloads nothing, so an HF model bigger than
    that card's free VRAM raises CUDA out-of-memory at load. Confining "auto" to
    a device subset via max_memory is the same technique the split branch below
    uses, and a "cpu" entry is what keeps the overflow landing on CPU: against
    the real accelerate.infer_auto_device_map, ``{0: small}`` alone sends the
    overflow to "disk" (which then needs an offload_folder), while
    ``{0: small, "cpu": N}`` sends it to CPU, exactly as plain "auto" does
    (get_max_memory() populates a "cpu" entry itself, but ONLY when the caller
    passes no max_memory at all).
    """

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
        # resolve_main_gpu_index is the REAL function: index 7 is not among the
        # detected devices, so it drops to 0 with a warning. _cuda_device_map
        # confines to what resolve_main_gpu_index RETURNS, not the raw configured
        # value. Pin non-Vulkan so membership validation actually runs, whatever
        # native backend is provisioned in the ambient environment.
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        mem = {0: (8_000_000_000, 16_000_000_000)}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 7})
        assert result["device_map"] == "auto"
        assert [k for k in result["max_memory"] if isinstance(k, int)] == [0]
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_unreadable_free_vram_still_honors_the_selection(self, monkeypatch, caplog):
        """If free VRAM cannot be read for the chosen device there is no budget
        to build, so the device is pinned rather than the user's explicit GPU
        selection being ignored - and the fallback is announced, since it is the
        one case with no CPU offload."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}])
        mem = {1: RuntimeError("no driver")}
        with caplog.at_level("WARNING", logger="localm"):
            result = _cuda_device_map(_fake_torch(mem), config={"main_gpu_index": 1})
        assert result == {"device_map": {"": 1}}
        assert any("main_gpu_index" in r.message for r in caplog.records)


class TestGpuSplitSuccess:
    """gpu_split_indices configured with 2+ valid devices and mem_get_info
    succeeding for each -> "auto" plus a max_memory dict with a headroom-adjusted
    entry per device."""

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
        """The split path has the same gap as the pinned one: a GPU-only
        max_memory removes the "cpu" entry that plain "auto" gets from
        get_max_memory(), so a model too big for the split lands on "disk" and
        needs an offload_folder instead of spilling to CPU."""
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
    """Run *script* in a genuinely FRESH child ``python.exe`` process (the same
    interpreter/venv as this test run, via ``sys.executable``) and return the
    completed process so callers can assert on returncode/stdout/stderr.

    A subprocess, not an in-process call, and the same pattern as
    tests/test_mp_spawn_fix.py: a real, unmocked LlamaCpp construction
    (tests/test_gpu_split_wiring.py's TestLlamaCppGpuSplitWiring) loads
    llama.cpp's bundled HIP/ROCm native runtime into whatever process runs it,
    via localm.inference.backends.llamacpp._loader.load_lib(). Once that native
    runtime is resident, a LATER ``import torch`` in the SAME process faults with
    ``OSError: [WinError 127] The specified procedure could not be found``
    loading torch's own separate rocm_sdk package's hipsolver.dll - two
    incompatible native ROCm/HIP runtimes cannot coexist in one process. This is
    the same mechanism ``VramSizingMixin._free_total_vram_bytes``
    (localm/inference/backends/llamacpp/_sizing.py) guards against for a
    different call site (its ``_torch_rocm_init_broken`` cache). A fresh child
    interpreter sidesteps the collision, so the real accelerate behaviour is
    exercised regardless of what else has run in this pytest worker."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=120,
    )


def _assert_subprocess_ok(result: "subprocess.CompletedProcess[str]") -> None:
    assert result.returncode == 0, (
        f"isolated accelerate check subprocess failed (returncode="
        f"{result.returncode}):\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")


# Self-contained: this runs in a separate interpreter, so it imports nothing
# from this test module. It builds the same over-budget model and
# _cuda_device_map-derived max_memory, drives the REAL
# accelerate.infer_auto_device_map, and asserts the overflow lands on CPU.
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
# "cpu" max_memory entry.
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
    """The shape assertions above cannot see what accelerate DOES with the
    mapping: ``{'device_map': {'': 1}}`` validates as correct while offloading
    nothing. This drives the REAL accelerate.infer_auto_device_map - the function
    transformers calls for device_map="auto" - against a real module too big for
    the GPU budget, and asserts the overflow actually lands on CPU. No GPU
    needed: max_memory is supplied, so nothing is probed.

    Both bodies run the exercise in a FRESH CHILD python.exe process (see
    _run_isolated_script above), not in-process: this pytest worker may also run
    tests/test_gpu_split_wiring.py, which loads llama.cpp's own native ROCm/HIP
    runtime, after which an in-process ``import torch`` faults. See
    TestSurvivesPriorLlamaCppNativeLoad below.
    """

    def test_our_max_memory_actually_offloads_overflow_to_cpu(self):
        _skip_unless_torch_and_accelerate_available()

        result = _run_isolated_script(_CPU_OFFLOAD_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "CPU_OFFLOAD_OK" in result.stdout

    def test_a_cpu_less_budget_would_send_overflow_to_disk(self):
        """The negative control proving the 'cpu' entry is what does the work
        (and that this test could fail): the same budget WITHOUT it maps the
        overflow to disk instead."""
        _skip_unless_torch_and_accelerate_available()

        result = _run_isolated_script(_DISK_CONTROL_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "DISK_CONTROL_OK" in result.stdout


class TestSurvivesPriorLlamaCppNativeLoad:
    """Negative-oracle regression proof: loading llama.cpp's bundled native
    runtime into THIS test process (the same load tests/test_gpu_split_wiring.py
    triggers, via a real, unmocked LlamaCpp construction) must not break the
    isolated accelerate check above.

    With the accelerate exercise running IN-PROCESS, loading the native runtime
    first and then doing ``import torch`` in the same process raises
    ``OSError: [WinError 127] The specified procedure could not be found``. With
    the exercise isolated into a fresh child subprocess (via _run_isolated_script
    above), the same in-process contamination has no effect on the child, which
    never shares the parent's already-loaded native libraries. This test performs
    exactly that contamination step and then asserts the isolated check still
    passes."""

    def test_survives_prior_llamacpp_native_load(self):
        _skip_unless_torch_and_accelerate_available()

        from localm.inference.backends.llamacpp import _loader
        try:
            _loader.load_lib()
        except RuntimeError as e:
            pytest.skip(
                f"native llama.cpp runtime not provisioned on this box: {e}")

        # Also import LlamaCpp itself; the lib is already loaded by the explicit
        # load_lib() call above.
        from localm.inference.backends.llamacpp.llama import LlamaCpp  # noqa: F401

        result = _run_isolated_script(_CPU_OFFLOAD_CHILD_SCRIPT)
        _assert_subprocess_ok(result)
        assert "CPU_OFFLOAD_OK" in result.stdout


class TestGpuSplitMemGetInfoFailures:
    """torch.cuda.mem_get_info raising for enough devices that fewer than 2 end
    up usable -> falls back to whichever of the device_map branches applies,
    with a warning logged."""

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
    """An index resolve_gpu_split() itself would drop (unmatched by list_gpus())
    - confirms _cuda_device_map composes with the REAL resolve_gpu_split rather
    than trusting the raw config."""

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
