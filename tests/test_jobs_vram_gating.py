# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for VRAM gating and engine reuse in scheduled jobs."""

from unittest.mock import MagicMock, patch

from localm.plugins.builtin.jobs.runner import _load_engine


class TestJobsVramGating:
    def test_reuses_live_engine_when_same_model(self, monkeypatch):
        # Mock the live engine in http_server
        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        
        monkeypatch.setattr("localm.inference.http_server._engine", live)
        
        # When calling _load_engine with the same name, it should return the live engine directly
        res = _load_engine("gemma-2b")
        assert res is live
        assert live.unload.call_count == 0

    def test_reuses_live_engine_when_no_model_specified(self, monkeypatch):
        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        
        monkeypatch.setattr("localm.inference.http_server._engine", live)
        
        # When calling _load_engine with None/empty, it should return the live engine
        res = _load_engine(None)
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
