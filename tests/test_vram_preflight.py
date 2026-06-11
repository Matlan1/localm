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

    def test_load_failure_mentions_vram_when_low(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=80_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=20_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("alloc failed")):
            b.load()
        out = capsys.readouterr().out
        assert "low on memory" in out
        assert b._use_subprocess is True   # fallback still engaged

    def test_load_failure_no_vram_hint_when_plenty_free(self, tmp_path, capsys):
        b = _backend(tmp_path, size_bytes=20_000_000)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=120_000_000), \
             patch.object(b, "_load_native", side_effect=RuntimeError("bad dll")):
            b.load()
        out = capsys.readouterr().out
        assert "low on memory" not in out
