# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server process never cold-imports torch.

A cold ROCm ``import torch`` on Windows holds the OS loader lock for as long as
torch's native preload runs, and no thread can be created in the process
meanwhile: the asyncio loop cannot hand work to its executor, so every request
stalls. Each test plants a ``torch`` package whose ``__init__`` records an
import attempt in a marker file (an import that raises is evicted from
``sys.modules``, so the marker is the only reliable witness) and asserts the
marker never appears.
"""
import importlib
import importlib.machinery
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

from localm import _torch_gpu_probe, discover, gpu_usage

GB = 1024 ** 3


@pytest.fixture
def torch_pkg(tmp_path, monkeypatch):
    """Factory for a fake on-disk torch whose import is detectable.

    ``make(version_py)`` writes the package, puts it first on sys.path, clears
    any real torch from sys.modules and the build-flavour cache, and returns the
    marker path. ``marker.exists()`` afterwards means torch was imported."""
    marker = tmp_path / "torch-was-imported.marker"

    def make(version_py):
        pkg = tmp_path / "torch"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text(
            f"open({str(marker)!r}, 'w').write('imported')\n"
            "raise RuntimeError('torch must not be imported in the server')\n",
            encoding="utf-8")
        if version_py is not None:
            (pkg / "version.py").write_text(version_py, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(gpu_usage, "_torch_build_hip_cache", None)
        importlib.invalidate_caches()
        return marker

    return make


HIP_VERSION_PY = (
    "from typing import Optional\n"
    "__version__ = '2.11.0+rocm7.13.0'\n"
    "cuda: Optional[str] = None\n"
    "hip: Optional[str] = '7.13.99004'\n")
CUDA_VERSION_PY = (
    "from typing import Optional\n"
    "__version__ = '2.11.0+cu128'\n"
    "cuda: Optional[str] = '12.8'\n"
    "hip: Optional[str] = None\n")


class TestTorchBuildIsHip:
    def test_hip_build_is_read_from_version_py_without_importing(self, torch_pkg):
        marker = torch_pkg(HIP_VERSION_PY)
        assert gpu_usage.torch_build_is_hip() is True
        assert not marker.exists()
        assert "torch" not in sys.modules

    def test_cuda_build_is_not_hip(self, torch_pkg):
        marker = torch_pkg(CUDA_VERSION_PY)
        assert gpu_usage.torch_build_is_hip() is False
        assert not marker.exists()

    def test_unparseable_version_py_is_unknown_not_false(self, torch_pkg):
        marker = torch_pkg("__version__ = '1.0'\n")
        assert gpu_usage.torch_build_is_hip() is None
        assert not marker.exists()

    def test_missing_version_py_is_unknown(self, torch_pkg):
        torch_pkg(None)
        assert gpu_usage.torch_build_is_hip() is None

    def test_torch_not_installed_is_unknown(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(gpu_usage, "_torch_build_hip_cache", None)
        monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec",
                            classmethod(lambda cls, name, *a, **k: None))
        assert gpu_usage.torch_build_is_hip() is None

    def test_answer_is_cached(self, torch_pkg, monkeypatch):
        torch_pkg(HIP_VERSION_PY)
        assert gpu_usage.torch_build_is_hip() is True
        monkeypatch.setattr(
            importlib.machinery.PathFinder, "find_spec",
            classmethod(lambda cls, *a, **k: pytest.fail("a cached answer re-read the disk")))
        assert gpu_usage.torch_build_is_hip() is True


class TestTorchFullyImported:
    def test_absent(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        assert gpu_usage.torch_fully_imported() is False

    def test_mid_import_is_not_imported(self, monkeypatch):
        mod = types.ModuleType("torch")
        mod.__spec__ = types.SimpleNamespace(_initializing=True)
        monkeypatch.setitem(sys.modules, "torch", mod)
        assert gpu_usage.torch_fully_imported() is False
        assert discover._torch_is_resident() is False

    def test_finished_import_is_imported(self, monkeypatch):
        mod = types.ModuleType("torch")
        mod.__spec__ = types.SimpleNamespace(_initializing=False)
        monkeypatch.setitem(sys.modules, "torch", mod)
        assert gpu_usage.torch_fully_imported() is True
        assert discover._torch_is_resident() is True

    def test_a_module_without_a_spec_counts_as_imported(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", MagicMock())
        assert gpu_usage.torch_fully_imported() is True


class TestScopeGateNeverImports:
    @pytest.fixture(autouse=True)
    def _benign_guards(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(discover, "_gpu_probe_inflight", False)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)

    def test_hip_build_answers_true_and_torch_stays_unimported(self, torch_pkg):
        marker = torch_pkg(HIP_VERSION_PY)
        assert gpu_usage.raw_reading_is_process_scoped() is True
        assert not marker.exists()
        assert "torch" not in sys.modules

    def test_cuda_build_answers_false_and_torch_stays_unimported(self, torch_pkg):
        marker = torch_pkg(CUDA_VERSION_PY)
        assert gpu_usage.raw_reading_is_process_scoped() is False
        assert not marker.exists()

    def test_unreadable_build_falls_back_to_the_resident_runtime_signal(
            self, torch_pkg, monkeypatch):
        marker = torch_pkg(None)
        monkeypatch.setattr(discover, "native_hip_runtime_resident", lambda: True)
        assert gpu_usage.raw_reading_is_process_scoped() is True
        monkeypatch.setattr(discover, "native_hip_runtime_resident", lambda: False)
        assert gpu_usage.raw_reading_is_process_scoped() is False
        assert not marker.exists()

    def test_a_half_imported_torch_is_never_touched(self, torch_pkg, monkeypatch):
        """Another thread is mid-import: reading ``torch.version`` would block on
        the module lock. The answer must come from the file instead."""
        torch_pkg(HIP_VERSION_PY)
        mod = MagicMock()
        mod.__spec__ = types.SimpleNamespace(_initializing=True)
        type(mod).version = property(
            lambda self: pytest.fail("touched a module that is still importing"))
        monkeypatch.setitem(sys.modules, "torch", mod)
        assert gpu_usage.raw_reading_is_process_scoped() is True


class TestPciBusNeverImports:
    def test_unimported_torch_gives_no_bus_and_no_import(self, torch_pkg, monkeypatch):
        marker = torch_pkg(HIP_VERSION_PY)
        assert gpu_usage._torch_pci_bus(0) is None
        assert not marker.exists()

    def test_the_bus_id_on_the_device_entry_is_used_first(self, monkeypatch):
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {45: 123, 7: 5})
        asked = MagicMock(return_value=None)
        monkeypatch.setattr(gpu_usage, "_torch_pci_bus", asked)
        got = gpu_usage.device_global_used_bytes(
            [{"index": 0, "pci_bus_id": 45}, {"index": 1, "pci_bus_id": 7}])
        assert got == {0: 123, 1: 5}
        asked.assert_not_called()

    def test_an_entry_without_a_bus_id_still_asks_resident_torch(self, monkeypatch):
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {45: 123})
        monkeypatch.setattr(gpu_usage, "_torch_pci_bus", lambda idx: 45)
        assert gpu_usage.device_global_used_bytes([{"index": 0}]) == {0: 123}

    def test_a_bus_id_that_matches_no_adapter_is_not_paired_by_position(
            self, monkeypatch):
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {45: 123})
        monkeypatch.setattr(gpu_usage, "_pdh_adapter_used", lambda: [])
        assert gpu_usage.device_global_used_bytes(
            [{"index": 0, "pci_bus_id": 9}]) == {}


class _Props:
    def __init__(self, bus):
        self.pci_bus_id = bus


class TestChildReportsThePciBus:
    def _torch(self, bus):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda i: (1000, 4000),
            get_device_name=lambda i: "STUB",
            get_device_properties=lambda i: _Props(bus))
        return types.SimpleNamespace(cuda=cuda)

    def test_flag_reads_an_int_bus(self):
        assert _torch_gpu_probe.pci_bus_flag(self._torch(45), 0) == 45

    @pytest.mark.parametrize("bad", [None, "45", 4.5, True])
    def test_flag_ignores_a_non_int_bus(self, bad):
        assert _torch_gpu_probe.pci_bus_flag(self._torch(bad), 0) is None

    def test_flag_survives_a_properties_failure(self):
        torch = self._torch(45)

        def boom(i):
            raise RuntimeError("no props")
        torch.cuda.get_device_properties = boom
        assert _torch_gpu_probe.pci_bus_flag(torch, 0) is None

    def test_enumerate_carries_the_bus_id_only_when_torch_reports_it(
            self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self._torch(45))
        assert _torch_gpu_probe._enumerate()[0]["pci_bus_id"] == 45
        monkeypatch.setitem(sys.modules, "torch", self._torch(None))
        assert "pci_bus_id" not in _torch_gpu_probe._enumerate()[0]


class TestSizingReadNeverColdImports:
    @pytest.fixture(autouse=True)
    def _guards(self, monkeypatch):
        from localm.inference.backends.llamacpp import _loader
        from localm.inference.backends.llamacpp._sizing import VramSizingMixin
        monkeypatch.setattr(VramSizingMixin, "_torch_rocm_init_broken", False)
        monkeypatch.setattr(VramSizingMixin, "_torch_vram_read_wedged", False)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        self.mixin = VramSizingMixin

    def test_windows_without_torch_answers_unmeasurable_at_once(
            self, torch_pkg, monkeypatch):
        marker = torch_pkg(HIP_VERSION_PY)
        monkeypatch.setattr(sys, "platform", "win32")
        before = {t.name for t in threading.enumerate()}
        assert self.mixin._free_total_vram_bytes() == (None, None)
        started = {t.name for t in threading.enumerate()} - before
        assert "localm-torch-vram-read" not in started
        assert not marker.exists()

    def test_an_imported_torch_is_still_read(self, monkeypatch):
        mod = types.ModuleType("torch")
        mod.__spec__ = types.SimpleNamespace(_initializing=False)
        monkeypatch.setitem(sys.modules, "torch", mod)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self.mixin, "_torch_free_total_uncapped",
                            staticmethod(lambda: (5, 10)))
        assert self.mixin._free_total_vram_bytes() == (5, 10)

    def test_a_half_imported_torch_is_not_read(self, monkeypatch):
        mod = types.ModuleType("torch")
        mod.__spec__ = types.SimpleNamespace(_initializing=True)
        monkeypatch.setitem(sys.modules, "torch", mod)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self.mixin, "_torch_free_total_uncapped",
                            staticmethod(lambda: pytest.fail("read a half import")))
        assert self.mixin._free_total_vram_bytes() == (None, None)
