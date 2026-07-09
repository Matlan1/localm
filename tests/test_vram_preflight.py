# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the VRAM pre-flight warning in GgufBackend.load()."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=99, n_ctx=4096):
    f = tmp_path / "model.gguf"
    # Sparse-ish: just truncate to size without writing real bytes
    with open(f, "wb") as fh:
        fh.truncate(size_bytes)
    return GgufBackend(str(f), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)


@pytest.fixture(autouse=True)
def _small_overhead(monkeypatch):
    """Scale the fixed KV/buffer overhead down to match the MB-scale files."""
    monkeypatch.setattr(GgufBackend, "_VRAM_OVERHEAD_BYTES", 15_000_000)


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
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=20_000_000), \
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
             patch.object(b, "_load_native", side_effect=RuntimeError("bad dll")):
            with pytest.raises(RuntimeError) as exc:
                b.load()
        assert "low on memory" not in str(exc.value)


class TestKvCacheAwarePreflight:
    """CHK-KVCACHE-OVERFLOW: _check_vram()'s ``need`` estimate must include the
    KV cache for the context this load will actually create (n_ctx), not just
    model weights. A large -c/n_ctx request can need many GB of KV cache on
    top of small weights (e.g. a 10GB Q6_K model with -c 131072 needs a ~20GB
    KV cache - see dev-notes/ for the real-hardware repro), and llama.cpp
    allocates that KV cache as part of context construction, AFTER this
    preflight runs. A weights-only estimate stayed silent for exactly this
    case, letting the load reach the native context-creation call with no
    warning at all - which on ROCm has been observed to either silently spill
    into slow system memory or crash the GPU driver ("unspecified launch
    failure") with nothing surfaced to the user.
    """

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
    """CHK-KVCACHE-OVERFLOW (growth path): GgufBackend._check_context_fit() is
    the vram_check callback wired into LlamaCpp so context GROWTH (e.g. the
    first prompt, since default max_tokens=4096 already forces a grow past the
    default base n_ctx=4096 - see _prefill_fresh_context) gets the same
    preflight the initial load already has. Unlike _check_vram() (which runs
    before anything is resident, so must include model weights), weights are
    ALREADY resident by growth time - the check must compare only the NEW KV
    cache + overhead against currently free VRAM, never weights again (that
    would double-count them and false-refuse an ordinary-sized grow)."""

    def test_raises_when_new_kv_cache_exceeds_free_vram(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=50_000_000):
            with pytest.raises(RuntimeError) as exc:
                b._check_context_fit(2_000_000)   # kv alone ~32GB, no weights term
        msg = str(exc.value)
        assert "too large" in msg.lower()
        assert "2,000,000" in msg

    def test_does_not_double_count_resident_weights(self, tmp_path):
        # Free VRAM here is LESS than the model's weight size (as expected,
        # since the weights are already resident and thus already NOT free) -
        # a check that wrongly re-added weights to "need" would false-refuse
        # this ordinary small grow. Only the new KV cache + overhead may count.
        b = _backend(tmp_path, size_bytes=9_000_000_000, n_ctx=4096)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=4_000_000_000):
            b._check_context_fit(8192)   # must not raise

    def test_silent_when_vram_not_measurable(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=None):
            b._check_context_fit(2_000_000)   # must not raise - unmeasurable

    def test_silent_for_cpu_only_run(self, tmp_path):
        b = _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=0)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=0):
            b._check_context_fit(2_000_000)   # must not raise - CPU-only

    def test_load_native_wires_check_context_fit_into_llamacpp(self, tmp_path):
        """End to end: _load_native() must pass GgufBackend's OWN bound
        _check_context_fit as LlamaCpp's vram_check - not omit it (which would
        leave context growth unguarded again, a facade fix)."""
        b = _backend(tmp_path, size_bytes=1_000_000)
        fake_llamacpp = MagicMock()
        with patch.object(GgufBackend, "_vram_levels", return_value=[]), \
             patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.dict(sys.modules,
                        {"localm.inference.backends.llamacpp": fake_llamacpp}):
            b._load_native()
        _, kwargs = fake_llamacpp.LlamaCpp.call_args
        assert kwargs.get("vram_check") == b._check_context_fit


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
        """End to end through _load_native with a stubbed LlamaCpp: the
        printed line shows in-use/total and the delta this load consumed."""
        import sys
        from unittest.mock import MagicMock
        b = _backend(tmp_path, size_bytes=1_000_000)
        # 12 GiB free before the load, 4 GiB free after -> 8 GiB this load.
        # Sizes are displayed in binary units (GiB labelled "GB").
        GIB = 1024 ** 3
        levels = iter([[(12 * GIB, 16 * GIB)],
                       [(4 * GIB, 16 * GIB)]])
        fake_llamacpp = MagicMock()
        with patch.object(GgufBackend, "_vram_levels",
                          side_effect=lambda: next(levels)), \
             patch("localm.inference.backends.llamacpp._loader.load_lib"), \
             patch.dict(sys.modules,
                        {"localm.inference.backends.llamacpp": fake_llamacpp}):
            b._load_native()
        out = capsys.readouterr().out
        assert "12.00 GB in use / 16.00 GB total" in out
        assert "+8.00 GB this load" in out
        assert "0.00 GB allocated" not in out      # the old, wrong line


class TestFreeVramBytesDeviceSelection:
    """_free_vram_bytes() must read the CONFIGURED main GPU device, not always
    device 0, once a multi-GPU main_gpu_index is set."""

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
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 9})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        with self._fake_torch([(1_000, 2_000)]):
            free = GgufBackend._free_vram_bytes()
        assert free == 1_000
