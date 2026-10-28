# SPDX-License-Identifier: AGPL-3.0-or-later
"""A free-VRAM reading taken through the isolated native probe is corrected
device-globally when the runtime the probe loads is HIP on Windows.

The parent process that sizes a GGUF load (server, coder, embedder before its
model loads) has no native runtime loaded and, on a torch-less install, no
torch to ask. Its free reading comes from the isolated VRAM probe daemon,
which loads the resolved runtime directory; on Windows plus HIP that reading
counts only the probe's own allocations. These tests drive the real chain
``_free_vram_bytes`` -> ``_device_global_free_bytes`` ->
``raw_reading_is_process_scoped`` -> ``device_global_used_bytes`` with only the
hardware edges replaced: the in-process torch read (unanswered), the probe's
reply, the ADL per-adapter usage, and the DLL names in a real runtime directory.
"""
import sys

import pytest

from localm import discover, gpu_usage
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

GB = 1024 ** 3
TOTAL = 16 * GB
PROBE_FREE = TOTAL - GB // 8
ADL_USED = 7 * GB


@pytest.fixture
def parent(monkeypatch, tmp_path):
    """A Windows parent process with no native runtime loaded and no torch,
    reading VRAM through the isolated probe. ``make(dll, torch_hip)`` writes a
    runtime directory shipping *dll* and sets the installed torch's build
    flavour (None for no torch)."""

    def make(dll, torch_hip=None, platform="win32"):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(gpu_usage, "_torch_build_hip_cache", (torch_hip,))
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        rt = tmp_path / "native-runtime"
        rt.mkdir()
        (rt / "llama.dll").write_bytes(b"")
        (rt / "ggml-base.dll").write_bytes(b"")
        (rt / dll).write_bytes(b"")
        monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: rt)
        monkeypatch.setattr(_loader, "gpu_memory_isolated",
                            lambda: (PROBE_FREE, TOTAL))
        monkeypatch.setattr(VramSizingMixin, "_free_total_vram_bytes",
                            staticmethod(lambda: (None, None)))
        monkeypatch.setattr(discover, "resolve_load_gpu_index", lambda *a, **k: 0)
        monkeypatch.setattr(discover, "last_known_gpus", lambda: [])
        monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {45: ADL_USED})
        monkeypatch.setattr(gpu_usage, "_pdh_adapter_used", lambda: [])

    return make


class TestIsolatedProbeReadingScope:
    def test_torchless_hip_runtime_reading_is_corrected(self, parent):
        parent("ggml-hip.dll")
        assert gpu_usage.raw_reading_is_process_scoped() is True
        assert VramSizingMixin._free_vram_bytes() == TOTAL - ADL_USED

    def test_non_hip_torch_with_a_hip_runtime_is_corrected(self, parent):
        parent("ggml-hip.dll", torch_hip=False)
        assert VramSizingMixin._free_vram_bytes() == TOTAL - ADL_USED

    def test_hip_torch_build_is_still_corrected(self, parent):
        parent("ggml-vulkan.dll", torch_hip=True)
        assert VramSizingMixin._free_vram_bytes() == TOTAL - ADL_USED

    @pytest.mark.parametrize("dll", ["ggml-vulkan.dll", "ggml-cuda.dll", "ggml-cpu.dll"])
    def test_non_hip_runtime_reading_is_left_alone(self, parent, dll):
        parent(dll)
        assert gpu_usage.raw_reading_is_process_scoped() is False
        assert VramSizingMixin._free_vram_bytes() == PROBE_FREE

    def test_cuda_torch_with_a_cuda_runtime_is_left_alone(self, parent):
        parent("ggml-cuda.dll", torch_hip=False)
        assert VramSizingMixin._free_vram_bytes() == PROBE_FREE

    def test_linux_hip_runtime_reading_is_left_alone(self, parent):
        parent("ggml-hip.dll", platform="linux")
        assert gpu_usage.raw_reading_is_process_scoped() is False
        assert VramSizingMixin._free_vram_bytes() == PROBE_FREE

    def test_a_loaded_non_hip_runtime_answers_from_the_resident_runtime(
            self, parent, monkeypatch):
        parent("ggml-hip.dll")
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(discover, "native_hip_runtime_resident", lambda: False)
        assert gpu_usage.raw_reading_is_process_scoped() is False

    def test_an_unresolved_runtime_is_not_process_scoped(self, parent, monkeypatch):
        parent("ggml-hip.dll")
        monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: None)
        assert gpu_usage.raw_reading_is_process_scoped() is False
