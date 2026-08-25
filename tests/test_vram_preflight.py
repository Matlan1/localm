# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the VRAM pre-flight warning in GgufBackend.load()."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=99, n_ctx=4096):
    # A tiny REAL file (so is_file()/stat work) with the "on disk" size FAKED via
    # _model_bytes, the same pattern test_auto_gpu_layers.py uses.
    #
    # This used to `truncate(size_bytes)` under the comment "Sparse-ish: just
    # truncate to size without writing real bytes". That comment was WRONG:
    # truncate() is NOT sparse on Windows/NTFS and allocates the full size for
    # real (memory: windows-truncate-not-sparse). Measured on this box: one
    # truncate(2GB) consumed 1.61 GB. This helper is called 17x per pass
    # including two 9 GB models (~18.9 GB per pass), each in its own tmp_path,
    # times the xdist workers, times pytest's retention of the last 3 basetemps.
    # That is what filled D: to 99.5% and crashed the box (2026-07-15).
    #
    # Nothing here needs the bytes to exist: every caller only exercises the
    # preflight DECISION, which reads the size back through _model_bytes().
    # (test_model_bytes_sums_split_parts builds its own real files and is
    # deliberately untouched - it tests the real stat-summing path at 1 MB.)
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    b._model_bytes = lambda: size_bytes
    return b


@pytest.fixture(autouse=True)
def _small_overhead(monkeypatch):
    """Scale the fixed KV/buffer overhead down to match the MB-scale files."""
    monkeypatch.setattr(GgufBackend, "_VRAM_OVERHEAD_BYTES", 15_000_000)


@pytest.fixture(autouse=True)
def _reset_torch_broken_flag():
    """_torch_rocm_init_broken is a deliberate process-lifetime cache (see its docstring in _sizing.py) - correct for a real server process, but poison for a test session where many unrelated tests share one process."""
    from localm.inference.backends.llamacpp._sizing import VramSizingMixin
    saved = VramSizingMixin._torch_rocm_init_broken
    VramSizingMixin._torch_rocm_init_broken = False
    yield
    VramSizingMixin._torch_rocm_init_broken = saved


@pytest.fixture(autouse=True)
def _neutralise_native_lib_loaded():
    """_loader.native_lib_loaded() (added by #754) is True for the rest of ANY xdist worker in which a real_gguf-gated test has RUN (conftest.py's lazy resource gate - or the test itself - calls load_lib() at that test's setup, and _loaded_lib is deliberately never reset)."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader.native_lib_loaded
    _loader.native_lib_loaded = lambda: False
    yield
    _loader.native_lib_loaded = saved


class TestVramPreflight:
    def test_warns_when_model_exceeds_free_vram(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "-g 0" in out          # actionable advice present

    def test_silent_when_model_fits(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=40_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=300_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()
        assert "Low VRAM" not in capsys.readouterr().out

    def test_silent_when_vram_not_measurable(self, tmp_path, capsys):
        b = _backend(tmp_path)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=None):
            b._check_vram()
        assert capsys.readouterr().out == ""

    def test_silent_for_cpu_only_run(self, tmp_path, capsys):
        b = _backend(tmp_path, n_gpu_layers=0)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=0):
            b._check_vram()
        assert capsys.readouterr().out == ""

    def test_low_vram_warning_notes_a_possibly_blind_reading(self, tmp_path, capsys):
        """AGENTS.md rule 5: this warning fires only when even a possibly over-stated free already reads as insufficient (sound - see discover.gpu_split_shortfall's docstring for the same asymmetry), but the quoted GB figure itself can be the raw, cross-process-blind reading when the device-global correction (..."""
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=True):
            b._check_vram()
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "may not see other processes" in out

    def test_low_vram_warning_omits_the_blind_note_on_a_trusted_platform(
        self, tmp_path, capsys
    ):
        """The complement: a platform gpu_usage does NOT flag as blind gets no caveat - the figure stands as reported, exactly as before this fix."""
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=False):
            b._check_vram()
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "may not see other processes" not in out

    def test_model_bytes_sums_split_parts(self, tmp_path):
        for i in (1, 2):
            f = tmp_path / f"m-0000{i}-of-00002.gguf"
            with open(f, "wb") as fh:
                fh.truncate(1_000_000)
        b = GgufBackend(str(tmp_path / "m-00001-of-00002.gguf"))
        assert b._model_bytes() == 2_000_000

    def test_load_failure_mentions_vram_when_low(self, tmp_path):
        # No subprocess fallback: a native load failure raises, and the message
        # includes the low-VRAM hint and points at setup-llama.
        # _total_vram_bytes is stubbed like the sibling tests above: without it,
        # load()'s preflight ran the REAL probe (a real `import torch`), which
        # is not this test's subject and, in a mixed multi-file run on the
        # ROCm box, faulted with a 0xc0000139 stderr trace per test (caught by
        # the product's own broken-import latch, so the tests still passed -
        # pure noise, reproduced identically on a pristine checkout).
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=20_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("alloc failed")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        msg = str(exc.value)
        assert "low on memory" in msg
        assert "setup-llama" in msg
        assert b.loaded is False           # never claims to be loaded

    def test_load_failure_no_vram_hint_when_plenty_free(self, tmp_path):
        b = _backend(tmp_path, size_bytes=20_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=120_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("bad dll")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        assert "low on memory" not in str(exc.value)


class TestKvCacheAwarePreflight:
    """CHK-KVCACHE-OVERFLOW: _check_vram()'s ``need`` estimate must include the KV cache for the context this load will actually create (n_ctx), not just model weights."""

    def test_raises_when_kv_cache_pushes_need_past_total_vram(self, tmp_path):
        # 80MB weights (bytes-per-token floors at 16_000 regardless of model
        # size) with a 2,000,000-token context needs ~32GB of KV cache alone -
        # far more than even a clean, empty 16GB card could ever hold. This must
        # be refused BEFORE the native context-creation call, not discovered by
        # letting the driver attempt it.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=2_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            with pytest.raises(RuntimeError) as exc:
                b._check_vram()
        msg = str(exc.value)
        assert "too large" in msg.lower()
        assert "2,000,000" in msg            # names the actual context size
        assert "-c " in msg                  # actionable: how to shrink it

    def test_no_regression_when_context_genuinely_fits(self, tmp_path, capsys):
        # A normal, modest context on the same small model must neither raise
        # nor warn - the new KV-cache term must not make a genuinely-fitting
        # load look oversized.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=10_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out

    def test_warns_but_does_not_raise_when_only_free_is_short(self, tmp_path, capsys):
        # need exceeds FREE (something else is using the GPU) but comfortably
        # fits under TOTAL (a clean card would hold it) - this is the
        # recoverable "free some VRAM" case and must stay a warning, not a
        # hard refusal.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=100_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" in capsys.readouterr().out

    def test_raise_check_skipped_when_total_not_measurable(self, tmp_path, capsys):
        # total unmeasurable (e.g. no torch / registry-fallback tier): the hard
        # refusal must be skipped rather than crash on a None comparison, even
        # for a context that would have tripped it - falls through to the
        # ordinary (here: silent, since free comfortably covers it) free-vram
        # check instead of raising.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=None):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out


class TestCheckContextFit:
    """GgufBackend._check_context_fit() is the vram_check callback wired into LlamaCpp: it decides WHERE a growing context's KV cache must live."""

    def test_uses_ram_when_kv_growth_does_not_fit_vram(self, tmp_path):
        # A grow whose NET KV growth cannot fit VRAM keeps the FULL window but puts
        # the KV cache in system RAM (return False) - never shrinks, never aborts.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000):
            assert b._check_context_fit(2_000_000, current_ctx=4096) is False

    def test_charges_only_net_growth_over_resident_context(self, tmp_path):
        # The old context's KV is reclaimed before the new is allocated, so only the
        # DELTA (target - current) is charged. Free here holds the delta but NOT the
        # whole target KV + overhead: the OLD hard check would have false-refused;
        # the delta check passes -> KV fits VRAM (return True).
        # per_token ~= 90_000; delta(8192 from 4096) ~= 368 MB; whole target KV
        # ~= 737 MB, + 1.5 GB overhead ~= 2.2 GB. Free = 500 MB.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=500_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is True

    def test_does_not_double_count_resident_weights(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4_000_000_000):
            assert b._check_context_fit(8192, current_ctx=4096) is True   # KV fits VRAM

    def test_returns_none_when_vram_not_measurable(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=None):
            assert b._check_context_fit(2_000_000) is None   # unmeasurable -> keep default (VRAM)

    def test_returns_none_for_cpu_only_run(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=0)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=0):
            assert b._check_context_fit(2_000_000) is None   # CPU-only -> KV already in RAM

    def test_ram_kv_hint_notes_a_possibly_blind_reading(self, tmp_path, caplog):
        """Same rule-5 caveat as _check_vram's warning: the KV-to-RAM hint fires only when even a possibly-inflated free already reads as insufficient (sound), but the quoted GB figure can still be the raw, cross-process-blind reading when the device-global correction silently declined - say so on a platform g..."""
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=True), \
             caplog.at_level(logging.WARNING, logger="localm"):
            assert b._check_context_fit(2_000_000, current_ctx=4096) is False
        assert any("may not see other processes" in r.getMessage()
                  for r in caplog.records)

    def test_ram_kv_hint_omits_the_blind_note_on_a_trusted_platform(
        self, tmp_path, caplog
    ):
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=False), \
             caplog.at_level(logging.WARNING, logger="localm"):
            assert b._check_context_fit(2_000_000, current_ctx=4096) is False
        assert not any("may not see other processes" in r.getMessage()
                      for r in caplog.records)

    # test_load_native_wires_check_context_fit_into_llamacpp moved to
    # tests/test_gguf_worker.py (TestLoad.test_load_wires_check_context_fit_and_returns_metadata):
    # the real LlamaCpp construction (and the vram_check wiring) now happens
    # in GgufWorker, inside the isolated worker process, not in
    # GgufBackend._load_native() - see llamacpp/_runner.py.


class TestVramReport:
    """The post-load VRAM line must use driver-level numbers (mem_get_info), never torch allocator counters - llama.dll allocates outside torch, so memory_allocated() reads 0.00 GB no matter what the model occupies."""

    def _fake_torch(self, free_total_per_device):
        import sys
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(free_total_per_device)
        fake.cuda.mem_get_info.side_effect = \
            lambda i: free_total_per_device[i]
        return patch.dict(sys.modules, {"torch": fake})

    def test_vram_levels_returns_driver_numbers(self):
        with self._fake_torch([(4_000_000_000, 16_000_000_000)]):
            levels = GgufBackend._vram_levels()
        assert levels == [(4_000_000_000, 16_000_000_000)]

    def test_vram_levels_empty_without_torch(self):
        import sys
        # Simulate torch being absent (import raises)
        with patch.dict(sys.modules, {"torch": None}):
            assert GgufBackend._vram_levels() == []

    def test_load_reports_usage_delta(self, tmp_path, capsys):
        """End to end through _load_native with a stubbed ModelRunner: the printed line shows in-use/total and the delta this load consumed, when discover.list_gpus reports a TRUSTED (fresh, device-global) reading - #960 replaced the raw torch _vram_levels() read here with discover.list_gpus() gated through sy..."""
        from localm.discover import FREE_SCOPE_DEVICE, GPU_PROBE_OK
        b = _backend(tmp_path, size_bytes=1_000_000)
        # 12 GiB free before the load, 4 GiB free after -> 8 GiB this load.
        # Sizes are displayed in binary units (GiB labelled "GB").
        GIB = 1024 ** 3
        readings = iter([
            [{"index": 0, "total": 16 * GIB, "free": 12 * GIB,
              "free_scope": FREE_SCOPE_DEVICE}],
            [{"index": 0, "total": 16 * GIB, "free": 4 * GIB,
              "free_scope": FREE_SCOPE_DEVICE}],
        ])

        def _fake_list_gpus(*, deadline=None, return_status=False,
                            wait_for_inflight=False):
            gpus = next(readings)
            return (gpus, GPU_PROBE_OK) if return_status else gpus

        with patch("localm.discover.list_gpus", side_effect=_fake_list_gpus), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": None, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()
        out = capsys.readouterr().out
        assert "12.00 GB in use / 16.00 GB total" in out
        assert "+8.00 GB this load" in out
        assert "0.00 GB allocated" not in out      # the old, wrong line

    def test_load_omits_used_free_when_reading_not_trusted(self, tmp_path, capsys):
        """#960: a process-scoped reading (Windows + AMD ROCm/HIP torch, blind to every OTHER process's VRAM - exactly the case for a GGUF load, which always runs in its own isolated worker process) must NOT print a used/free figure it cannot stand behind as current fact."""
        from localm.discover import FREE_SCOPE_PROCESS, GPU_PROBE_OK
        b = _backend(tmp_path, size_bytes=1_000_000)
        GIB = 1024 ** 3

        def _fake_list_gpus(*, deadline=None, return_status=False,
                            wait_for_inflight=False):
            # Blind to the model this process's own worker just loaded -
            # "free" reads almost the whole board free, same shape as the
            # filed bug.
            gpus = [{"index": 0, "total": 16 * GIB, "free": 15.86 * GIB,
                     "free_scope": FREE_SCOPE_PROCESS}]
            return (gpus, GPU_PROBE_OK) if return_status else gpus

        with patch("localm.discover.list_gpus", side_effect=_fake_list_gpus), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": None, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()
        out = capsys.readouterr().out
        assert "16.00 GB total" in out
        assert "in use" not in out
        assert "0.14 GB" not in out
        assert "not trusted" in out

    def test_load_skips_vram_probe_entirely_for_cpu_only_load(self, tmp_path):
        """A CPU-only load (n_gpu_layers=0, mirroring _check_vram's own 'CPU-only run, VRAM is irrelevant' early-return a few lines up the call chain) must never touch discover.list_gpus at all - not just skip printing the result."""
        b = _backend(tmp_path, size_bytes=1_000_000, n_gpu_layers=0)
        with patch("localm.discover.list_gpus") as fake_list_gpus, \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": None, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()
        fake_list_gpus.assert_not_called()


class TestFreeVramBytesDeviceSelection:
    """_free_vram_bytes() must read the CONFIGURED main GPU device, not always device 0, once a multi-GPU main_gpu_index is set."""

    @pytest.fixture(autouse=True)
    def _uncorrected_reading(self, monkeypatch):
        # WHAT THIS ISOLATES: #706 added a cross-process correction on top of the raw
        # torch reading (_sizing.py's _free_vram_bytes -> _device_global_free_bytes),
        # which engages ONLY where the raw reading is known process-scoped, i.e. Windows
        # + a ROCm/HIP torch. These tests are about WHICH DEVICE is read, not about that
        # correction, and their fake torch inadvertently switched it on: the correction
        # returns total-minus-all-process-used, and an idle box reports used=0, so the
        # assertion saw the device's TOTAL instead of its free value (2000 not 1000).
        # That is why they passed on ubuntu (correction early-returns off-win32) and
        # failed on windows-latest. Pin it off so the assertion measures the reading path
        # regardless of the ambient platform - same intent as the _native_backend_has_vulkan
        # pin below. The correction itself is covered on its own (test_auto_gpu_layers.py,
        # test_kv_bytes_offload.py, test_discover.py, and TestVramReport here).
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped",
                            lambda: False)

    def _fake_torch(self, per_device_free_total):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(per_device_free_total)
        fake.cuda.mem_get_info.side_effect = lambda i: per_device_free_total[i]
        return patch.dict(sys.modules, {"torch": fake})

    def test_reads_configured_device(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 5_000

    def test_defaults_to_device_zero_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000

    def test_invalid_configured_index_falls_back_to_device_zero(self, monkeypatch):
        # Pin non-Vulkan so membership validation actually runs, regardless of
        # what native backend is provisioned in the ambient environment (see
        # test_discover.py::TestResolveMainGpuIndex).
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 9})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        with self._fake_torch([(1_000, 2_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000


class TestFreeVramBytesUsesIsolatedNativeFallback:
    """_free_vram_bytes() must prefer torch.cuda.mem_get_info, and must NEVER call the DIRECT, abort-prone loader.gpu_memory() - only the crash-safe loader.gpu_memory_isolated() (subprocess-isolated) as a fallback when torch cannot answer."""

    @pytest.fixture(autouse=True)
    def _uncorrected_reading(self, monkeypatch):
        # Same isolation, same reason as TestFreeVramBytesDeviceSelection above: these
        # tests assert the SOURCE and ORDER of the reading (torch first, isolated probe
        # as fallback, direct native never), not #706's cross-process correction. Left
        # live, that correction overwrites the fake's free value with total-minus-used on
        # Windows, so the asserts saw 9000/5000 (totals) instead of 7000/3000 (frees) -
        # green on ubuntu, red on windows-latest, for a reason unrelated to what these
        # tests exist to guard.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped",
                            lambda: False)

    def _fake_torch_ok(self):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.cuda.mem_get_info.return_value = (7_000, 9_000)
        return patch.dict(sys.modules, {"torch": fake})

    def test_isolated_fallback_never_called_when_torch_answers(self, monkeypatch):
        sentinel = MagicMock(side_effect=AssertionError(
            "gpu_memory_isolated() must not be called when torch already answered"))
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated", sentinel)
        with self._fake_torch_ok():
            free = GgufBackend._free_vram_bytes()
        assert free == 7_000
        sentinel.assert_not_called()

    def test_direct_native_ggml_query_never_called(self, monkeypatch):
        """The DIRECT, abort-prone native call must never be reached from _free_vram_bytes under any condition - only the isolated wrapper."""
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        sentinel = MagicMock(side_effect=AssertionError(
            "loader.gpu_memory() (direct, abort-prone) must never be called "
            "from _free_vram_bytes - only gpu_memory_isolated()"))
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory", sentinel)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000
        sentinel.assert_not_called()

    def test_falls_back_to_isolated_probe_when_torch_unmeasurable(self, monkeypatch):
        """Vulkan/Metal/NVIDIA-without-torch/Linux/macOS (no CUDA/ROCm torch): the isolated probe must be consulted and its answer used - VRAM measurement must keep working, not just degrade to None."""
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000

    def test_none_when_neither_torch_nor_isolated_probe_can_answer(self, monkeypatch):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: None)
        with patch.dict(sys.modules, {"torch": fake}):
            free = GgufBackend._free_vram_bytes()
        assert free is None

    def test_none_when_torch_itself_raises(self, monkeypatch):
        """A torch import/query failure (e.g. the observed ROCm-init hiccup) must degrade via _free_total_vram_bytes' own try/except and still try the isolated fallback, not propagate."""
        class _BoomModule:
            def __getattr__(self, name):
                raise RuntimeError("simulated torch ROCm-init failure")
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": _BoomModule()}):
            free = GgufBackend._free_vram_bytes()
        assert free == 3_000

    def test_torch_import_failure_is_cached_and_never_retried(self, monkeypatch):
        """The real bug this guards (found live, reproduced on demand - see _sizing.py's VramSizingMixin._free_total_vram_bytes docstring): once llama.cpp's bundled HIP runtime is loaded in-process (as GgufWorker's isolated worker process does for every real load), a LATER `import torch` hits a genuine DLL ent..."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert GgufBackend._free_total_vram_bytes() == (None, None)
        from localm.inference.backends.llamacpp._sizing import VramSizingMixin
        assert VramSizingMixin._torch_rocm_init_broken is True

        # Second call must be answered from the cached flag WITHOUT
        # re-attempting the import - proven by making torch "recover" to a
        # working fake and confirming it is still never consulted.
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.mem_get_info.return_value = (1, 2)
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert GgufBackend._free_total_vram_bytes() == (None, None), (
            "a cached broken-import must not be retried, even once torch "
            "would now succeed")
        fake.cuda.is_available.assert_not_called()


class _FakeStream:
    """A minimal write()/flush() or readline() double for a fake daemon proc's stdin/stdout, scripted with a fixed queue of responses."""

    def __init__(self, lines=None):
        self._lines = list(lines or [])
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def readline(self):
        return self._lines.pop(0) if self._lines else ""     # "" == EOF


class _FakeDaemonProc:
    """A minimal subprocess.Popen double: scripted stdout lines, a `killed` flag that flips poll() from None (running) to an exit code (dead)."""

    def __init__(self, response_lines):
        self.stdin = _FakeStream()
        self.stdout = _FakeStream(response_lines)
        self._killed = False

    def poll(self):
        return -9 if self._killed else None

    def kill(self):
        self._killed = True


class TestGpuMemoryIsolated:
    """loader.gpu_memory_isolated() must degrade to None on every failure mode of the daemon subprocess, and never raise - this is what makes it safe to call unconditionally from _free_vram_bytes."""

    def _loader(self):
        from localm.inference.backends.llamacpp import _loader
        return _loader

    @pytest.fixture(autouse=True)
    def _isolate_probe_singleton(self):
        loader = self._loader()
        saved = loader._PROBE_PROC
        loader._PROBE_PROC = None
        yield
        loader._PROBE_PROC = saved

    def _patch_readline_passthrough(self, monkeypatch, loader):
        """Replace the real threaded timeout wrapper with a direct readline() call for these higher-level tests (the timeout mechanism itself has its own dedicated tests below) - keeps these fast and deterministic."""
        monkeypatch.setattr(
            loader, "_readline_with_timeout",
            lambda stream, timeout: (stream.readline() or None))

    def test_first_call_spawns_daemon_and_parses_response(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["123456 789012\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() == (123456, 789012)
        assert fake.stdin.written == ["q\n"]

    def test_second_call_reuses_running_daemon_without_respawning(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["100 200\n", "300 400\n"])
        spawn_calls = []
        monkeypatch.setattr(
            loader, "_spawn_probe_daemon",
            lambda: spawn_calls.append(1) or fake)
        assert loader.gpu_memory_isolated() == (100, 200)
        assert loader.gpu_memory_isolated() == (300, 400)
        assert len(spawn_calls) == 1, "the second call must reuse the daemon, not respawn"

    def test_daemon_err_reply_returns_none_but_keeps_daemon_alive(self, monkeypatch):
        """A daemon that is alive and answered 'unmeasurable' is not the same as a crashed daemon - it must not be killed/respawned for that alone."""
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["ERR\n", "50 100\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() is None
        assert not fake._killed
        assert loader.gpu_memory_isolated() == (50, 100)

    def test_err_reply_with_cause_is_unmeasurable_not_desync(
            self, monkeypatch, caplog):
        """An 'ERR <cause>' reply (the daemon naming its startup load_lib failure - see _vram_probe's protocol) is the ERR branch, not a protocol desync: the daemon stays alive and the cause reaches the caller's debug log."""
        import logging

        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(
            ["ERR load_lib failed: RuntimeError('Cannot find llama.dll')\n",
             "50 100\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            assert loader.gpu_memory_isolated() is None
        assert not fake._killed, (
            "a cause-carrying ERR reply was treated as desync and killed a "
            "healthy daemon")
        assert "load_lib failed" in caplog.text, (
            "the daemon's named cause must reach the caller's debug log")
        assert loader.gpu_memory_isolated() == (50, 100)

    def test_dead_daemon_triggers_respawn(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        dead = _FakeDaemonProc([])
        dead.kill()                          # already dead before any query
        loader._PROBE_PROC = dead            # simulate a daemon that died between calls
        fresh = _FakeDaemonProc(["10 20\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fresh)
        assert loader.gpu_memory_isolated() == (10, 20)

    def test_eof_on_read_kills_and_clears_daemon_for_next_respawn(self, monkeypatch):
        """The exact scenario this exists for: the daemon hard-aborts mid-query (native crash) - the OS closes its pipes, so the next read returns EOF, never an exception that could propagate into the parent."""
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        crashing = _FakeDaemonProc([])       # readline() -> "" == EOF immediately
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: crashing)
        assert loader.gpu_memory_isolated() is None
        assert loader._PROBE_PROC is None, "a crashed daemon must be cleared, not reused"

    def test_garbage_response_kills_and_clears_daemon(self, monkeypatch):
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["not a valid pair of ints\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() is None
        assert fake._killed
        assert loader._PROBE_PROC is None

    def test_spawn_failure_degrades_to_none(self, monkeypatch):
        loader = self._loader()

        def _raise():
            raise OSError("could not spawn python")

        monkeypatch.setattr(loader, "_spawn_probe_daemon", _raise)
        assert loader.gpu_memory_isolated() is None
        assert loader._PROBE_PROC is None


def test_spawn_probe_daemon_uses_localm_capable_interpreter(monkeypatch):
    """_spawn_probe_daemon must resolve the interpreter via _mp_spawn.interpreter_for_localm_children(), never bare sys.executable: inside an mp-spawn worker sys.executable is the BASE interpreter, whose children cannot import localm or resolve the runtime wheel - the daemon then fails on every query and t..."""
    import subprocess

    import localm._mp_spawn as mp_spawn
    from localm.inference.backends.llamacpp import _loader

    sentinel = r"Z:\resolved\venv\python.exe"
    monkeypatch.setattr(mp_spawn, "interpreter_for_localm_children",
                        lambda: sentinel)
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return _FakeDaemonProc([])

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    proc = _loader._spawn_probe_daemon()
    assert isinstance(proc, _FakeDaemonProc)
    assert captured["argv"][0] == sentinel
    assert captured["argv"][-1].endswith("_vram_probe")


def test_vram_probe_daemon_names_its_load_failure(monkeypatch, capsys):
    """The daemon's startup load_lib failure must ride along in its ERR replies ('ERR <cause>'): its stderr is discarded by the caller, so the protocol line is the only channel that can carry WHY into the caller's debug log (rule 5 - a bare ERR is indistinguishable from 'no GPU')."""
    import io

    from localm.inference.backends.llamacpp import _loader, _vram_probe

    def _boom():
        raise RuntimeError("Cannot find llama.dll - not provisioned")

    monkeypatch.setattr(_loader, "load_lib", _boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO("q\ndevices\n"))
    assert _vram_probe.main() == 0
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert len(out_lines) == 2
    for line in out_lines:
        assert line.startswith("ERR "), line
        assert "Cannot find llama.dll" in line


class TestReadlineWithTimeout:
    """The low-level timeout wrapper itself: a genuinely blocking stream must not hang the caller past `timeout`, and a normal read must pass through."""

    def _loader(self):
        from localm.inference.backends.llamacpp import _loader
        return _loader

    def test_normal_read_passes_through(self):
        loader = self._loader()
        stream = _FakeStream(["hello\n"])
        assert loader._readline_with_timeout(stream, timeout=2.0) == "hello\n"

    def test_eof_returns_none(self):
        loader = self._loader()
        stream = _FakeStream([])
        assert loader._readline_with_timeout(stream, timeout=2.0) is None

    def test_genuinely_blocking_read_times_out(self):
        """A stream whose readline() never returns must not hang this call past the timeout - uses a small timeout so the test itself stays fast."""
        import threading

        loader = self._loader()

        class _HangingStream:
            def readline(self):
                threading.Event().wait()      # blocks forever

        import time
        t0 = time.time()
        result = loader._readline_with_timeout(_HangingStream(), timeout=0.2)
        elapsed = time.time() - t0
        assert result is None
        assert elapsed < 2.0, f"timeout wrapper did not return promptly: {elapsed:.2f}s"


class TestFreeVramCrossProcessCorrection:
    """#697/#700 follow-up (V&V finding #1): the GGUF sizing DECISION path must consume a DEVICE-GLOBAL free reading, not the raw process-scoped one."""

    def test_correction_is_applied_when_a_source_answers(self):
        """raw says 15GB free (blind); device-global says 6GB genuinely free -> the decision path must use 6GB."""
        with patch.object(GgufBackend, "_free_total_vram_bytes",
                          return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes",
                          return_value=6_000_000_000):
            assert GgufBackend._free_vram_bytes() == 6_000_000_000

    def test_uncorrected_reading_kept_when_no_source(self):
        """Device-global platforms (Linux/NVIDIA) and torch-less builds return None from the corrector, so the raw reading - correct there - is kept unchanged."""
        with patch.object(GgufBackend, "_free_total_vram_bytes",
                          return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes", return_value=None):
            assert GgufBackend._free_vram_bytes() == 15_000_000_000

    def test_isolated_fallback_is_also_corrected(self):
        """When torch cannot answer, the isolated ggml probe's reading (equally blind on Win/AMD) is corrected too, not just the torch path."""
        with patch.object(GgufBackend, "_free_total_vram_bytes", return_value=(None, None)), \
             patch("localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
                   return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes",
                          return_value=6_000_000_000):
            assert GgufBackend._free_vram_bytes() == 6_000_000_000

    def test_none_when_nothing_is_measurable(self):
        with patch.object(GgufBackend, "_free_total_vram_bytes", return_value=(None, None)), \
             patch("localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
                   return_value=None):
            assert GgufBackend._free_vram_bytes() is None

    def test_corrector_returns_none_off_known_blind_platform(self):
        """The correction must only fire where the raw reading is KNOWN blind, or it would fabricate a number on a platform that never needed one."""
        with patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=False):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) is None

    def test_corrector_computes_total_minus_device_global_used(self):
        with patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=True), \
             patch("localm.gpu_usage.device_global_used_bytes",
                   return_value={0: 10_000_000_000}), \
             patch("localm.discover.resolve_main_gpu_index", return_value=0):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) == 6_000_000_000

    def test_corrector_never_raises_into_a_model_load(self):
        """A correction that cannot be made must degrade to 'use the uncorrected reading', never crash a load (rule 5)."""
        with patch("localm.gpu_usage.raw_reading_is_process_scoped",
                   side_effect=RuntimeError("driver boom")):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) is None

    def test_corrector_none_total_is_none(self):
        assert GgufBackend._device_global_free_bytes(None) is None
