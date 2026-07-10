# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.sysstats._vram() - the GUI hardware-monitor's VRAM line.

No dedicated test file existed for this module before AUDIT-GPU-SPLIT-1: the
maintainer explicitly expected every relevant VRAM-reading function to be
multi/split-GPU aware, not just the ones that gate a load/refuse decision -
including this status-bar widget, which previously stayed on vram_info()'s
single main-GPU number as a deliberate (and, on reflection, unnecessary)
design choice."""

from unittest.mock import patch

from localm.sysstats import _vram


GB = 1024 ** 3


def test_reports_single_gpu_when_no_split_configured():
    with patch("localm.discover.vram_info",
               return_value={"total": 16 * GB, "free": 4 * GB}):
        out = _vram()
    assert out == {"vram": {"total": 16 * GB, "used": 12 * GB, "percent": 75.0}}


def test_reports_combined_capacity_with_a_configured_split():
    """AUDIT-GPU-SPLIT-1: with a configured 2-GPU split, the status bar must
    show the COMBINED total/used, not just the single main GPU's - it now
    goes through discover.vram_capacity(), not vram_info() directly."""
    from localm.config import load_config as real_load_config
    base_cfg = real_load_config()
    with patch("localm.discover.list_gpus", return_value=[
            {"index": 0, "name": "A", "total": 16 * GB, "free": 4 * GB},
            {"index": 1, "name": "B", "total": 8 * GB, "free": 8 * GB},
        ]), \
         patch("localm.config.load_config",
               return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
        out = _vram()
    assert out["vram"]["total"] == 24 * GB       # 16+8 combined, not 16 alone
    assert out["vram"]["used"] == 12 * GB        # (16-4)+(8-8) combined
    assert out["vram"]["percent"] == 50.0


def test_empty_when_unmeasurable():
    with patch("localm.discover.vram_info", return_value={}):
        assert _vram() == {}


def test_percent_omitted_when_free_unknown():
    """The registry-fallback tier reports total only (no per-process free
    reading available), so 'used'/'percent' must be omitted, not fabricated
    as 0% used."""
    with patch("localm.discover.vram_info", return_value={"total": 16 * GB}):
        out = _vram()
    assert out == {"vram": {"total": 16 * GB}}


def test_exception_from_discover_is_swallowed_not_raised():
    """_vram() must never raise - a probe failure just omits the section
    (matches the module's own documented 'NEVER raises' contract)."""
    with patch("localm.discover.vram_capacity", side_effect=RuntimeError("boom")):
        assert _vram() == {}
