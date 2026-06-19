# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the VRAM pre-flight warning in GgufBackend.load()."""

from unittest.mock import patch

import pytest

from localm.inference.backends.gguf import GgufBackend


def _backend(tmp_path, size_bytes=80_000_000, n_gpu_layers=99):
    f = tmp_path / "model.gguf"
    # Sparse-ish: just truncate to size without writing real bytes
    with open(f, "wb") as fh:
        fh.truncate(size_bytes)
    return GgufBackend(str(f), n_gpu_layers=n_gpu_layers)


@pytest.fixture(autouse=True)
def _small_overhead(monkeypatch):
    """Scale the fixed KV/buffer overhead down to match the MB-scale files."""
    monkeypatch.setattr(GgufBackend, "_VRAM_OVERHEAD_BYTES", 15_000_000)


class TestVramPreflight:
    def test_warns_when_model_exceeds_free_vram(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=40_000_000):
            b._check_vram()
        out = capsys.readouterr().out
        assert "Low VRAM" in out
        assert "-g 0" in out          # actionable advice present

    def test_silent_when_model_fits(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=40_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=120_000_000):
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
