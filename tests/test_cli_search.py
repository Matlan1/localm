# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm search <repo> --files` (localm/cli/models.py:search_cmd).

Covers the CLI mirror of the GUI's discover_files fit-badge computation."""

from unittest.mock import patch

from click.testing import CliRunner

from localm.cli import main


def test_search_files_shows_fit_badge(cli_runner):
    files = [{"file": "model.Q4_K_M.gguf", "quant": "Q4_K_M",
             "size_bytes": 2_000_000_000, "n_parts": 1}]
    with patch("localm.discover.hf_gguf_files", return_value=files), \
         patch("localm.discover.vram_info", return_value={"total": 8_000_000_000}):
        result = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
    assert result.exit_code == 0, result.output
    assert "fits" in result.output
    assert "Q4_K_M" in result.output
    # No split configured: the caption names the single main GPU's ceiling, not
    # a machine total.
    assert "main GPU" in result.output
    assert "total VRAM" not in result.output


def test_search_files_fit_reflects_combined_split_capacity(cli_runner):
    """`localm search <repo> --files` weighs fit against
    discover.vram_capacity()'s COMBINED split capacity, not vram_info()'s single
    main-GPU number: a file too big for one GPU alone but fitting across a
    configured 2-GPU split badges "fits", and the caption shows the COMBINED
    total."""
    # The CLI displays total/GiB (1024**3), so use GiB-based sizes throughout.
    # need ~= 15GiB*1.1 + 1.5e9 =~ 18 GiB: exceeds the 16 GiB main GPU alone
    # (too-big), but fits under 0.85 * the 24 GiB combined split.
    gib = 1024 ** 3
    files = [{"file": "model.Q8_0.gguf", "quant": "Q8_0",
             "size_bytes": 15 * gib, "n_parts": 1}]
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    with patch("localm.discover.hf_gguf_files", return_value=files), \
         patch("localm.discover.list_gpus", return_value=[
             {"index": 0, "name": "A", "total": 16 * gib, "free": 16 * gib},
             {"index": 1, "name": "B", "total": 8 * gib, "free": 8 * gib},
         ]), \
         patch("localm.config.load_config",
               return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
        result = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
    assert result.exit_code == 0, result.output
    # Names the basis (combined across the configured split) rather than calling
    # the single main GPU's 16 GB a machine total.
    assert "24 GB combined across your 2-GPU split" in result.output
    assert "total VRAM" not in result.output
    assert "fits" in result.output
    assert "too big" not in result.output


def test_search_files_stale_split_does_not_claim_combined(cli_runner):
    """A 2-entry split that resolves to only ONE detected device (a stale or
    typo'd index, or a GGUF-only box) makes vram_capacity() fall back to the
    single main GPU, so the caption says 'main GPU', not 'combined across your
    2-GPU split'. Gated on split_device_count, not the raw config length."""
    gib = 1024 ** 3
    files = [{"file": "model.Q4_K_M.gguf", "quant": "Q4_K_M",
             "size_bytes": 2 * gib, "n_parts": 1}]
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    with patch("localm.discover.hf_gguf_files", return_value=files), \
         patch("localm.discover.list_gpus", return_value=[
             {"index": 0, "name": "A", "total": 16 * gib, "free": 16 * gib},
         ]), \
         patch("localm.discover.vram_info",
               return_value={"total": 16 * gib, "free": 16 * gib}), \
         patch("localm.config.load_config",
               return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
        result = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
    assert result.exit_code == 0, result.output
    assert "main GPU" in result.output
    assert "combined" not in result.output   # a single-device fallback is NOT combined
