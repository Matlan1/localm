# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for VRAM gating and engine reuse in scheduled jobs."""

import pytest
from unittest.mock import MagicMock, patch

from localm.plugins.builtin.jobs.runner import _load_engine


class TestJobsVramGating:
    @pytest.mark.parametrize("name", ["gemma-2b", None])
    def test_reuses_live_engine_when_same_model_or_unspecified(self, monkeypatch, name):
        # Mock the live engine in http_server
        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"

        monkeypatch.setattr("localm.inference.http_server._engine", live)

        # When calling _load_engine with the same name or None, it should return the live engine directly
        res = _load_engine(name)
        assert res is live
        assert live.unload.call_count == 0

    @patch("localm.model_manager.get_model_info")
    @patch("localm.inference.engine.Engine")
    def test_unloads_live_engine_when_different_model_and_low_vram(
        self, mock_engine_cls, mock_get_model_info, monkeypatch
    ):
        # Mock get_model_info so the next model is resolvable
        mock_get_model_info.return_value = ("/path/to/llama-3b", "llama-3b")
        
        # Mock the new Engine instance
        new_eng = MagicMock()
        mock_engine_cls.return_value = new_eng
        
        # Mock the live engine
        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        
        monkeypatch.setattr("localm.inference.http_server._engine", live)
        
        # Mock VRAM info and swap policy
        from localm import vram
        from localm import discover
        
        monkeypatch.setattr(discover, "vram_info", lambda: {"free": 1024})
        monkeypatch.setattr(vram, "should_swap_for_media", lambda free, est: True)
        
        wait_calls = []
        monkeypatch.setattr(vram, "wait_for_vram_release", lambda fn, before_bytes: wait_calls.append(before_bytes))
        
        # Calling _load_engine with a different name should unload the live engine first
        res = _load_engine("llama-3b")

        assert live.unload.call_count == 1
        assert len(wait_calls) == 1
        assert wait_calls[0] == 1024
        assert res is new_eng

    def _install_split_gpu(self, monkeypatch, gpu_split_indices):
        """Overlay just gpu_split_indices onto the REAL (test-isolated)
        config, same pattern used elsewhere in this fix - a stripped-down
        fake config dict would break unrelated load_config() readers."""
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {**base_cfg, "gpu_split_indices": gpu_split_indices})

    @patch("localm.model_manager.get_model_info")
    @patch("localm.inference.engine.Engine")
    def test_split_configured_combined_capacity_avoids_unnecessary_unload(
        self, mock_engine_cls, mock_get_model_info, monkeypatch
    ):
        """AUDIT-GPU-SPLIT-1: the VRAM gate must weigh the swap decision
        against discover.vram_capacity()'s COMBINED split capacity, not just
        the single main GPU - with a 2-GPU split where the single main GPU
        alone would trigger an unload but the combined free comfortably
        covers the media estimate, the live chat engine must NOT be unloaded.
        should_swap_for_media runs for REAL here (not mocked), so this proves
        the actual decision, not just that the gate was reached."""
        mock_get_model_info.return_value = ("/path/to/llama-3b", "llama-3b")
        new_eng = MagicMock()
        mock_engine_cls.return_value = new_eng

        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        monkeypatch.setattr("localm.inference.http_server._engine", live)

        GB = 1024 ** 3
        # media_estimate_bytes("chat") falls back to the 14 GB default (no
        # "chat" key in DEFAULT_MEDIA_VRAM_GB) -> threshold ~= 14 GB + 1 GB
        # headroom = 14.93 GB. 10 GB free on ONE GPU alone would swap; 20 GB
        # combined across two must not.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
            {"index": 1, "name": "B", "total": 16 * GB, "free": 10 * GB},
        ])
        self._install_split_gpu(monkeypatch, [0, 1])

        res = _load_engine("llama-3b")

        assert live.unload.call_count == 0, (
            "combined 20GB free should cover the ~14.93GB threshold without "
            "swapping the live engine, even though neither GPU alone would")
        assert res is new_eng   # a different model is being loaded regardless

    @patch("localm.model_manager.get_model_info")
    @patch("localm.inference.engine.Engine")
    def test_same_single_gpu_free_without_split_still_unloads(
        self, mock_engine_cls, mock_get_model_info, monkeypatch
    ):
        """Guard: the SAME 10 GB free, but with NO split configured (single
        GPU only), still correctly triggers the swap - proves the split
        configuration is genuinely what makes the difference above, not a
        coincidence of the threshold math."""
        mock_get_model_info.return_value = ("/path/to/llama-3b", "llama-3b")
        new_eng = MagicMock()
        mock_engine_cls.return_value = new_eng

        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        monkeypatch.setattr("localm.inference.http_server._engine", live)

        GB = 1024 ** 3
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
        ])
        self._install_split_gpu(monkeypatch, None)
        # Avoid a real 5s wait_for_vram_release timeout (the mocked unload()
        # never actually changes vram_capacity()'s return, so the real poll
        # would spin until its timeout) - same pattern as the sibling test above.
        from localm import vram
        monkeypatch.setattr(vram, "wait_for_vram_release",
                            lambda fn, before_bytes: (True, before_bytes))

        res = _load_engine("llama-3b")

        assert live.unload.call_count == 1
        assert res is new_eng
