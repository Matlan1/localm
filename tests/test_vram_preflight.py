# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the VRAM pre-flight warning in GgufBackend.load()."""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=99, n_ctx=4096):
    # A tiny REAL file (so is_file()/stat work) with the on-disk size FAKED via
    # _model_bytes(), which is what the preflight decision reads.
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
    """Save and restore ``VramSizingMixin._torch_rocm_init_broken`` around every
    test in this module.

    That flag is a process-lifetime cache: once latched True, every VRAM read
    short-circuits to None before `import torch` is reached, regardless of what a
    later test mocks. Resetting it here keeps one test's incidental unmocked call
    from leaking into another. The flag's production behavior is untouched."""
    from localm.inference.backends.llamacpp._sizing import VramSizingMixin
    saved = VramSizingMixin._torch_rocm_init_broken
    VramSizingMixin._torch_rocm_init_broken = False
    yield
    VramSizingMixin._torch_rocm_init_broken = saved


@pytest.fixture(autouse=True)
def _neutralise_native_lib_loaded():
    """Force ``_loader.native_lib_loaded()`` to False for every test in this
    module, restoring it afterwards.

    That function stays True for the rest of any xdist worker in which a
    real_gguf-gated test has run, and once True ``_free_total_vram_bytes()``
    skips the torch attempt entirely and returns (None, None) - which defeats
    every test here that mocks torch via patch.dict(sys.modules, ...).

    Scoped to THIS module only, not conftest.py: tests/test_native_dll_conflict_
    guard.py unit-tests native_lib_loaded() itself, and a global override would
    defeat ITS mock. Patches the FUNCTION; a test that wants the real value can
    still monkeypatch it back locally."""
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

    @pytest.mark.parametrize("model", ["x[/dim] y", "qwen[v2]", "[bold red]m"])
    def test_a_peer_model_name_is_printed_as_text(self, tmp_path, capsys, monkeypatch, model):
        monkeypatch.setattr("localm.gpu_registry.list_gpu_peers",
                            lambda **k: [{"port": 8650, "model": model, "gpu_index": 0}])
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000),              patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()
        out = " ".join(capsys.readouterr().out.split())
        assert f"is running '{model}'" in out

    def test_low_vram_warning_notes_a_possibly_blind_reading(self, tmp_path, capsys):
        """The warning fires only when even a possibly over-stated free already
        reads as insufficient, but the quoted GB figure itself can be the raw,
        cross-process-blind reading when the device-global correction
        (_device_global_free_bytes) declined. On a platform gpu_usage says is
        known process-scoped-blind, the warning must say so."""
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
        """The complement: a platform gpu_usage does NOT flag as blind gets no
        caveat - the figure stands as reported."""
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
        # _total_vram_bytes is stubbed like the sibling tests above, so load()'s
        # preflight does not run the REAL probe (a real import torch).
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=20_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("alloc failed")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        msg = str(exc.value)
        assert "needed about" in msg and "0.0 GB free" in msg
        assert "-g 24" not in msg
        assert "setup-llama" in msg
        assert b.loaded is False           # never claims to be loaded

    def test_load_failure_no_vram_hint_when_plenty_free(self, tmp_path):
        b = _backend(tmp_path, size_bytes=20_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=120_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("bad dll")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        assert "needed about" not in str(exc.value)


class TestKvCacheAwarePreflight:
    """_check_vram()'s ``need`` estimate must include the KV cache for the
    context this load will actually create (n_ctx), not just model weights. A
    large -c/n_ctx request can need many GB of KV cache on top of small weights,
    and llama.cpp allocates that KV cache as part of context construction, AFTER
    this preflight runs.
    """

    def test_raises_when_kv_cache_pushes_need_past_total_vram(self, tmp_path):
        # 80MB weights (bytes-per-token floors at 16_000 regardless of model
        # size) with a 2,000,000-token context needs ~32GB of KV cache alone.
        # Refused BEFORE the native context-creation call.
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
        # A normal, modest context on the same small model neither raises nor
        # warns.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=10_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out

    def test_warns_but_does_not_raise_when_only_free_is_short(self, tmp_path, capsys):
        # need exceeds FREE (something else is using the GPU) but fits under
        # TOTAL (a clean card would hold it): a warning, not a hard refusal.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=100_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16_000_000_000):
            b._check_vram()   # must not raise
        assert "Low VRAM" in capsys.readouterr().out

    def test_raise_check_skipped_when_total_not_measurable(self, tmp_path, capsys):
        # total unmeasurable (no torch / registry-fallback tier): the hard
        # refusal is skipped rather than crashing on a None comparison, falling
        # through to the ordinary free-vram check.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15_000_000_000), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=None):
            b._check_vram()   # must not raise
        assert "Low VRAM" not in capsys.readouterr().out


class TestCheckContextFit:
    """GgufBackend._check_context_fit() is the vram_check callback wired into
    LlamaCpp: it decides WHERE a growing context's KV cache must live. Returns True
    to keep the KV cache in VRAM (offload_kqv - fast), False to place it in system
    RAM (slower, but keeps the FULL window - a degrade, not a shrink or an abort),
    or None when VRAM is unmeasurable / irrelevant.

    It NEVER raises and NEVER charges weights (already resident). Because
    _prefill_fresh_context frees the live context (current_ctx KV) BEFORE allocating
    the bigger one, only the NET KV growth over the resident context is charged; the
    resident compute-buffer overhead is not re-charged."""

    def test_uses_ram_when_kv_growth_does_not_fit_vram(self, tmp_path):
        # A grow whose NET KV growth cannot fit VRAM keeps the FULL window but puts
        # the KV cache in system RAM (return False) - never shrinks, never aborts.
        b = _backend(tmp_path, size_bytes=80_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000):
            assert b._check_context_fit(2_000_000, current_ctx=4096) is False

    def test_charges_only_net_growth_over_resident_context(self, tmp_path):
        # The old context's KV is reclaimed before the new is allocated, so only the
        # DELTA (target - current) is charged. Free here holds the delta but NOT the
        # whole target KV + overhead, so the delta check passes and KV fits VRAM.
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
        """Same caveat as _check_vram's warning: the KV-to-RAM hint fires only
        when even a possibly-inflated free already reads as insufficient, but the
        quoted GB figure can still be the raw, cross-process-blind reading when
        the device-global correction declined, so it must say so on a platform
        gpu_usage says is known process-scoped-blind."""
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


class TestContextFitsBudget:
    """GgufBackend._context_fits_budget() answers whether a grow stays within the
    Windows per-process video memory budget (gpu_usage.process_vram_budget:
    budget - usage), charged like _check_context_fit: the net growth while the
    live KV is in VRAM, the whole target while it is in system RAM."""

    GB = 1024 ** 3

    def _fits(self, b, budget, n_ctx=8192, current=4096):
        with patch("localm.gpu_usage.process_vram_budget", return_value=budget):
            return b._context_fits_budget(n_ctx, current)

    def test_growth_past_the_budget_does_not_fit(self, tmp_path):
        # Net growth 4096 -> 8192 at 90 KB/token is ~0.37 GB; 0.1 GB of room.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        assert self._fits(b, (10 * self.GB, 10 * self.GB - 100_000_000)) is False

    def test_growth_within_the_budget_fits(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        assert self._fits(b, (10 * self.GB, 8 * self.GB)) is True

    def test_a_process_over_its_budget_fits_nothing_new(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        assert self._fits(b, (9 * self.GB, 10 * self.GB)) is False

    def test_a_shrink_fits_when_it_brings_usage_under_the_budget(self, tmp_path):
        # Shrinking 8192 -> 4096 frees ~0.37 GB of KV.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        assert self._fits(b, (9 * self.GB, 9 * self.GB + 200_000_000),
                          n_ctx=4096, current=8192) is True
        assert self._fits(b, (9 * self.GB, 10 * self.GB), n_ctx=4096, current=8192) is False

    def test_kv_in_ram_is_charged_in_full(self, tmp_path):
        # The whole 8192-token KV (~0.74 GB) is charged when the live KV is in
        # RAM: 0.5 GB of room holds the net growth but not the whole cache.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        room = (10 * self.GB, 10 * self.GB - 500_000_000)
        b._llm = SimpleNamespace(_offload_kqv=False)
        assert self._fits(b, room) is False
        b._llm = SimpleNamespace(_offload_kqv=True)
        assert self._fits(b, room) is True

    def test_no_budget_reading_is_unknown(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        assert self._fits(b, None) is None

    def test_cpu_only_run_is_unknown_without_reading_the_budget(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_gpu_layers=0)
        with patch("localm.gpu_usage.process_vram_budget") as read:
            assert b._context_fits_budget(8192, 4096) is None
        read.assert_not_called()

    def test_a_failing_charge_is_unknown(self, tmp_path):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_kv_bytes_per_token", side_effect=OSError("bad header")):
            assert self._fits(b, (10 * self.GB, 1 * self.GB)) is None

    def test_budget_overrun_keeps_the_kv_cache_in_vram(self, tmp_path):
        """_check_context_fit places the KV cache on free VRAM alone: a grow past
        the budget whose growth fits free VRAM stays in VRAM, where it decodes
        faster than with the KV cache in system RAM."""
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4 * self.GB), \
             patch("localm.gpu_usage.process_vram_budget",
                   return_value=(10 * self.GB, 10 * self.GB)):
            assert b._check_context_fit(8192, current_ctx=4096) is True

    def test_ctx_grow_debug_line_carries_the_budget(self, tmp_path, caplog):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4 * self.GB), \
             patch("localm.gpu_usage.process_vram_budget",
                   return_value=(10 * self.GB, 9 * self.GB)), \
             caplog.at_level(logging.DEBUG, logger="localm"):
            b._check_context_fit(8192, current_ctx=4096)
        line = next(r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("ctx-grow fit:"))
        assert f"free={4 * self.GB} budget={10 * self.GB} usage={9 * self.GB}" in line

    def test_ctx_grow_skips_the_budget_read_when_debug_is_off(self, tmp_path, caplog):
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4 * self.GB), \
             patch("localm.gpu_usage.process_vram_budget") as read, \
             caplog.at_level(logging.INFO, logger="localm"):
            assert b._check_context_fit(8192, current_ctx=4096) is True
        read.assert_not_called()


class TestProcessVramBudget:
    """gpu_usage.process_vram_budget picks the one hardware adapter on which this
    process holds VRAM and answers None everywhere else."""

    MB = 1024 ** 2

    def test_none_off_windows(self, monkeypatch):
        from localm import gpu_usage
        monkeypatch.setattr(gpu_usage.sys, "platform", "linux")
        with patch.object(gpu_usage, "_dxgi_local_memory") as read:
            assert gpu_usage.process_vram_budget() is None
        read.assert_not_called()

    @pytest.mark.parametrize("readings, expected", [
        ([(16_000 * MB, 9_000 * MB)], (16_000 * MB, 9_000 * MB)),
        ([(16_000 * MB, 100 * MB)], None),
        ([(16_000 * MB, 0), (8_000 * MB, 5_000 * MB)], (8_000 * MB, 5_000 * MB)),
        ([(16_000 * MB, 9_000 * MB), (8_000 * MB, 5_000 * MB)], None),
        ([], None),
    ])
    def test_adapter_selection(self, monkeypatch, readings, expected):
        from localm import gpu_usage
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32")
        with patch.object(gpu_usage, "_dxgi_local_memory", return_value=readings):
            assert gpu_usage.process_vram_budget() == expected

    def test_dxgi_failure_is_none(self, monkeypatch):
        from localm import gpu_usage
        monkeypatch.setattr(gpu_usage.sys, "platform", "win32")
        with patch.object(gpu_usage, "_dxgi_local_memory", side_effect=OSError("no dxgi")):
            assert gpu_usage.process_vram_budget() is None

    @pytest.mark.skipif(sys.platform != "win32", reason="DXGI is Windows-only")
    def test_real_dxgi_query(self):
        from localm import gpu_usage
        readings = gpu_usage._dxgi_local_memory()
        if not readings:
            pytest.skip("no hardware DXGI adapter")
        for budget, usage in readings:
            assert budget > 0
            assert usage >= 0


class TestVramReport:
    """The post-load VRAM line must use driver-level numbers (mem_get_info),
    never torch allocator counters - llama.dll allocates outside torch, so
    memory_allocated() reads 0.00 GB no matter what the model occupies."""

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
        """End to end through _load_native with a stubbed ModelRunner: the
        printed line shows in-use/total and the delta this load consumed, when
        discover.list_gpus reports a TRUSTED (fresh, device-global) reading."""
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
        assert b.vram_load_bytes == 8 * GIB

    def test_load_vram_is_not_measured_when_one_device_is_not_trusted(self, tmp_path):
        from localm.discover import FREE_SCOPE_DEVICE, FREE_SCOPE_PROCESS, GPU_PROBE_OK
        b = _backend(tmp_path, size_bytes=1_000_000)
        GIB = 1024 ** 3
        readings = iter([
            [{"index": 0, "total": 16 * GIB, "free": 12 * GIB, "free_scope": FREE_SCOPE_DEVICE},
             {"index": 1, "total": 8 * GIB, "free": 8 * GIB, "free_scope": FREE_SCOPE_PROCESS}],
            [{"index": 0, "total": 16 * GIB, "free": 10 * GIB, "free_scope": FREE_SCOPE_DEVICE},
             {"index": 1, "total": 8 * GIB, "free": 8 * GIB, "free_scope": FREE_SCOPE_PROCESS}],
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
        assert b.vram_load_bytes is None

    def test_load_omits_used_free_when_reading_not_trusted(self, tmp_path, capsys):
        """A process-scoped reading (Windows + AMD ROCm/HIP torch, blind to every
        OTHER process's VRAM - the case for a GGUF load, which always runs in its
        own isolated worker process) must NOT print a used/free figure it cannot
        stand behind. Total-only, with a stated reason."""
        from localm.discover import FREE_SCOPE_PROCESS, GPU_PROBE_OK
        b = _backend(tmp_path, size_bytes=1_000_000)
        GIB = 1024 ** 3

        def _fake_list_gpus(*, deadline=None, return_status=False,
                            wait_for_inflight=False):
            # Blind to the model this process's own worker just loaded: free
            # reads almost the whole board as free.
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
        assert b.vram_load_bytes is None

    def test_load_skips_vram_probe_entirely_for_cpu_only_load(self, tmp_path):
        """A CPU-only load (n_gpu_layers=0, mirroring _check_vram's own
        "CPU-only run, VRAM is irrelevant" early-return a few lines up the
        call chain) must never touch discover.list_gpus at all - not just skip
        printing the result. A CPU-only load places nothing on the GPU, and each
        list_gpus(wait_for_inflight=True) call goes through an isolated
        subprocess probe that is not cached across calls."""
        b = _backend(tmp_path, size_bytes=1_000_000, n_gpu_layers=0)
        with patch("localm.discover.list_gpus") as fake_list_gpus, \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": None, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()
        fake_list_gpus.assert_not_called()


class TestFreeVramBytesDeviceSelection:
    """_free_vram_bytes() reads the device a one-device load runs on: the one a
    1-entry gpu_split_indices names, else main_gpu_index on a box where the load
    uses one GPU. A load that llama.cpp spreads over 2+ GPUs is not read from
    one of them."""

    @pytest.fixture(autouse=True)
    def _uncorrected_reading(self, monkeypatch):
        # These tests are about WHICH DEVICE is read, so the cross-process
        # correction is pinned off and the non-torch readings answer nothing.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped",
                            lambda: False)
        monkeypatch.setattr("localm.discover.last_gpu_reading", lambda: None)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated", lambda: None)
        monkeypatch.setattr("localm.discover.load_device_memory_isolated",
                            lambda *a, **k: None)

    def _fake_torch(self, per_device_free_total):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(per_device_free_total)
        fake.cuda.mem_get_info.side_effect = lambda i: per_device_free_total[i]
        return patch.dict(sys.modules, {"torch": fake})

    def test_reads_the_device_a_one_entry_split_names(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [1], "main_gpu_index": None})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        monkeypatch.setattr("localm.discover._native_gpu_index_space_is_opaque",
                            lambda: False)
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 5_000

    def test_main_gpu_index_on_two_gpus_with_no_split_is_not_read_as_one(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        with self._fake_torch([(1_000, 2_000), (5_000, 8_000)]):
            assert GgufBackend._free_vram_bytes() is None

    def test_reads_device_zero_on_a_one_gpu_box(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        with self._fake_torch([(1_000, 2_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000

    def test_invalid_configured_index_falls_back_to_device_zero(self, monkeypatch):
        # Pin non-Vulkan so membership validation actually runs, regardless of
        # what native backend is provisioned in the ambient environment.
        monkeypatch.setattr("localm.discover._native_gpu_index_space_is_opaque", lambda: False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 9})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        with self._fake_torch([(1_000, 2_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000


class TestFreeVramBytesUsesIsolatedNativeFallback:
    """_free_vram_bytes() must prefer torch.cuda.mem_get_info, and must NEVER
    call the DIRECT, abort-prone loader.gpu_memory() - only the crash-safe
    loader.gpu_memory_isolated() (subprocess-isolated) as a fallback when torch
    cannot answer.

    loader.gpu_memory()'s mem_fn() ctypes call (ggml_backend_dev_memory) is not
    exception-safe: llama.cpp's own CUDA/HIP error macro aborts the WHOLE PROCESS
    on a transient driver failure. This project's own torch dependency is an
    AMD-ROCm, Windows-only wheel, so the fallback path is the common case for
    NVIDIA/Linux/macOS/Vulkan/Metal users and must actually work, not merely
    degrade to None."""

    @pytest.fixture(autouse=True)
    def _uncorrected_reading(self, monkeypatch):
        # These tests assert the SOURCE and ORDER of the reading (torch first,
        # isolated probe as fallback, direct native never), so the cross-process
        # correction is pinned off.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped",
                            lambda: False)
        # Windows never imports torch for this read (see
        # test_server_never_cold_imports_torch.py); the torch-first order
        # asserted here is every other platform's.
        monkeypatch.setattr(sys, "platform", "linux")

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
        """The DIRECT, abort-prone native call must never be reached from
        _free_vram_bytes under any condition - only the isolated wrapper."""
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
        """Vulkan/Metal/NVIDIA-without-torch/Linux/macOS (no CUDA/ROCm torch):
        the isolated probe must be consulted and its answer used - VRAM
        measurement must keep working, not just degrade to None."""
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
        """A torch import/query failure must degrade via _free_total_vram_bytes'
        own try/except and still try the isolated fallback, not propagate."""
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
        """Once llama.cpp's bundled HIP runtime is loaded in-process (as the
        isolated GGUF worker does for every real load), a LATER `import torch`
        hits a DLL entry-point conflict in torch's separate rocm_sdk ROCm
        runtime. A module that faults during import is evicted from sys.modules,
        so an uncached failure would retry - and re-fault - on EVERY subsequent
        VRAM check for the rest of the process's life.

        `sys.modules[name] = None` simulates `import torch` itself raising, as
        opposed to _BoomModule above, which simulates torch importing fine but a
        later attribute access failing (a different, NOT cached, failure mode)."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert GgufBackend._free_total_vram_bytes() == (None, None)
        from localm.inference.backends.llamacpp._sizing import VramSizingMixin
        assert VramSizingMixin._torch_rocm_init_broken is True

        # The second call is answered from the cached flag WITHOUT re-attempting
        # the import: torch recovers to a working fake and is still never
        # consulted.
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.mem_get_info.return_value = (1, 2)
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert GgufBackend._free_total_vram_bytes() == (None, None), (
            "a cached broken-import must not be retried, even once torch "
            "would now succeed")
        fake.cuda.is_available.assert_not_called()


class TestTotalVramBytesUsesIsolatedNativeFallback:
    """_total_vram_bytes() must fall back to the isolated native probe exactly
    like _free_vram_bytes() does (see TestFreeVramBytesUsesIsolatedNativeFallback
    above) - it used to stop at torch alone, so on hardware where only the
    isolated probe can answer (torch unavailable/broken/wedged), a caller
    holding a real, isolated-probe-sourced FREE reading was still handed a
    None TOTAL, purely because _total_vram_bytes() itself never tried the same
    fallback. That silently broke any decision needing both figures together
    (see _sizing.py's _auto_gpu_layers_cause, which needs total to tell "the
    model exceeds this GPU's capacity" apart from "something else is using
    it")."""

    @pytest.fixture(autouse=True)
    def _uncorrected_reading(self, monkeypatch):
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped",
                            lambda: False)

    def test_isolated_fallback_never_called_when_torch_answers(self, monkeypatch):
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.cuda.mem_get_info.return_value = (7_000, 9_000)
        sentinel = MagicMock(side_effect=AssertionError(
            "gpu_memory_isolated() must not be called when torch already answered"))
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated", sentinel)
        with patch.dict(sys.modules, {"torch": fake}):
            total = GgufBackend._total_vram_bytes()
        assert total == 9_000
        sentinel.assert_not_called()

    def test_falls_back_to_isolated_probe_when_torch_unmeasurable(self, monkeypatch):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: (3_000, 5_000))
        with patch.dict(sys.modules, {"torch": fake}):
            total = GgufBackend._total_vram_bytes()
        assert total == 5_000

    def test_none_when_neither_torch_nor_isolated_probe_can_answer(self, monkeypatch):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
            lambda: None)
        with patch.dict(sys.modules, {"torch": fake}):
            total = GgufBackend._total_vram_bytes()
        assert total is None


class _FakeStream:
    """A minimal write()/flush() or readline() double for a fake daemon proc's
    stdin/stdout, scripted with a fixed queue of responses."""

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
    """A minimal subprocess.Popen double: scripted stdout lines, a `killed`
    flag that flips poll() from None (running) to an exit code (dead)."""

    def __init__(self, response_lines):
        self.stdin = _FakeStream()
        self.stdout = _FakeStream(response_lines)
        self._killed = False

    def poll(self):
        return -9 if self._killed else None

    def kill(self):
        self._killed = True


class TestGpuMemoryIsolated:
    """loader.gpu_memory_isolated() must degrade to None on every failure mode
    of the daemon subprocess, and never raise - this is what makes it safe to
    call unconditionally from _free_vram_bytes. Every test resets the module-
    level daemon singleton (_PROBE_PROC) before and after, since it is shared
    global state across calls by design (that is the whole point of the
    daemon - reuse across calls within one process)."""

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
        """Replace the real threaded timeout wrapper with a direct readline()
        call for these higher-level tests (the timeout mechanism itself has its
        own dedicated tests below) - keeps these fast and deterministic."""
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
        """A daemon that is alive and answered "unmeasurable" is not the same
        as a crashed daemon - it must not be killed/respawned for that alone."""
        loader = self._loader()
        self._patch_readline_passthrough(monkeypatch, loader)
        fake = _FakeDaemonProc(["ERR\n", "50 100\n"])
        monkeypatch.setattr(loader, "_spawn_probe_daemon", lambda: fake)
        assert loader.gpu_memory_isolated() is None
        assert not fake._killed
        assert loader.gpu_memory_isolated() == (50, 100)

    def test_err_reply_with_cause_is_unmeasurable_not_desync(
            self, monkeypatch, caplog):
        """An "ERR <cause>" reply (the daemon naming its startup load_lib
        failure - see _vram_probe's protocol) is the ERR branch, not a protocol
        desync: the daemon stays alive and the cause reaches the caller's debug
        log."""
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
        """The exact scenario this exists for: the daemon hard-aborts mid-query
        (native crash) - the OS closes its pipes, so the next read returns EOF,
        never an exception that could propagate into the parent."""
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
    """_spawn_probe_daemon must resolve the interpreter via
    _mp_spawn.interpreter_for_localm_children(), never bare sys.executable:
    inside an mp-spawn worker sys.executable is the BASE interpreter, whose
    children cannot import localm or resolve the runtime wheel - the daemon
    then fails on every query and the worker never has a raw VRAM reading."""
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
    """The daemon's startup load_lib failure must ride along in its ERR replies
    ("ERR <cause>"): its stderr is discarded by the caller, so the protocol
    line is the only channel that can carry WHY into the caller's debug log; a
    bare ERR is indistinguishable from 'no GPU'."""
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
    """The low-level timeout wrapper itself: a genuinely blocking stream must
    not hang the caller past `timeout`, and a normal read must pass through."""

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
        """A stream whose readline() never returns must not hang this call past
        the timeout - uses a small timeout so the test itself stays fast."""
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
    """The GGUF sizing DECISION path must consume a DEVICE-GLOBAL free reading,
    not the raw process-scoped one. The blind reading only ever OVERSTATES free
    VRAM, so a decision made on it overcommits exactly when another process (a
    game, ComfyUI, another model's isolated worker) holds the card."""

    def test_correction_is_applied_when_a_source_answers(self):
        """raw says 15GB free (blind); device-global says 6GB genuinely free -> the
        decision path must use 6GB."""
        with patch.object(GgufBackend, "_free_total_vram_bytes",
                          return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes",
                          return_value=6_000_000_000):
            assert GgufBackend._free_vram_bytes() == 6_000_000_000

    def test_uncorrected_reading_kept_when_no_source(self):
        """Device-global platforms (Linux/NVIDIA) and torch-less builds return None
        from the corrector, so the raw reading - correct there - is kept unchanged."""
        with patch.object(GgufBackend, "_free_total_vram_bytes",
                          return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes", return_value=None):
            assert GgufBackend._free_vram_bytes() == 15_000_000_000

    def test_isolated_fallback_is_also_corrected(self):
        """When torch cannot answer, the isolated ggml probe's reading (equally blind
        on Win/AMD) is corrected too, not just the torch path."""
        with patch.object(GgufBackend, "_free_total_vram_bytes", return_value=(None, None)), \
             patch("localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
                   return_value=(15_000_000_000, 16_000_000_000)), \
             patch.object(GgufBackend, "_device_global_free_bytes",
                          return_value=6_000_000_000):
            assert GgufBackend._free_vram_bytes() == 6_000_000_000

    def test_none_when_nothing_is_measurable(self):
        with patch.object(GgufBackend, "_free_total_vram_bytes", return_value=(None, None)), \
             patch("localm.inference.backends.llamacpp._loader.gpu_memory_isolated",
                   return_value=None), \
             patch("localm.discover.load_device_memory_isolated", return_value=None):
            assert GgufBackend._free_vram_bytes() is None

    def test_corrector_returns_none_off_known_blind_platform(self):
        """The correction must only fire where the raw reading is KNOWN blind, or it
        would fabricate a number on a platform that never needed one."""
        with patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=False):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) is None

    def test_corrector_computes_total_minus_device_global_used(self):
        with patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=True), \
             patch("localm.gpu_usage.device_global_used_bytes",
                   return_value={0: 10_000_000_000}), \
             patch("localm.discover.resolve_main_gpu_index", return_value=0):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) == 6_000_000_000

    def test_corrector_never_raises_into_a_model_load(self):
        """A correction that cannot be made must degrade to 'use the uncorrected
        reading', never crash a load."""
        with patch("localm.gpu_usage.raw_reading_is_process_scoped",
                   side_effect=RuntimeError("driver boom")):
            assert GgufBackend._device_global_free_bytes(16_000_000_000) is None

    def test_corrector_none_total_is_none(self):
        assert GgufBackend._device_global_free_bytes(None) is None
