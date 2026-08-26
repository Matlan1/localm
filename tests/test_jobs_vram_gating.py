# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for VRAM gating and engine reuse in scheduled jobs.

The gate's *decision* (reuse the live engine, or evict it to make room for the job
model) is what these tests pin down. The eviction MECHANISM runs through the
guarded server-loop helper ``_evict_shared_engine_for_media`` (which honors the
in-flight pin and serializes with get_engine) rather than a raw ``live.unload()``
on the worker thread, so these tests spy on that helper rather than on
``live.unload``. The guarded mechanism itself (pin honoured, freed via
unload_one_model on the loop) is exercised end to end in
``test_jobs_shared_engine_free.py``.
"""

import pytest
from unittest.mock import MagicMock, patch

import localm.plugins.builtin.jobs.runner as runner_mod
from localm.plugins.builtin.jobs.runner import _load_engine


class TestJobsVramGating:
    @pytest.mark.parametrize("name", ["gemma-2b", None])
    def test_reuses_live_engine_when_same_model_or_unspecified(self, monkeypatch, name):
        # Mock the live engine in http_server
        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"

        monkeypatch.setattr("localm.inference.http_server._engine", live)
        # If the gate ever tried to evict on the reuse path, this spy would record it.
        evicted = []
        monkeypatch.setattr(runner_mod, "_evict_shared_engine_for_media",
                            lambda live_eng: evicted.append(live_eng) or "unloaded")

        # When calling _load_engine with the same name or None, it should return the
        # live engine directly (reused=True) - no eviction, no fresh load.
        res, reused = _load_engine(name)
        assert res is live
        assert reused is True, "reusing the shared engine must be reported as reused"
        assert live.unload.call_count == 0
        assert evicted == [], "the reuse path must never evict the live engine"

    @patch("localm.model_manager.get_model_info")
    @patch("localm.inference.engine.Engine")
    def test_evicts_live_engine_when_different_model_and_low_vram(
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

        monkeypatch.setattr("localm.discover.vram_capacity",
                            lambda config=None: {"free": 1024})
        monkeypatch.setattr(vram, "should_swap_for_media", lambda free, est: True)

        # Spy on the guarded eviction helper: the gate must route the live-engine
        # unload through it (NOT a raw live.unload()), so the shared engine is freed
        # with the in-flight pin honoured and serialized with get_engine.
        evicted = []
        monkeypatch.setattr(runner_mod, "_evict_shared_engine_for_media",
                            lambda live_eng: evicted.append(live_eng) or "unloaded")

        # Loading a different model with tight VRAM should evict the live engine first.
        res, reused = _load_engine("llama-3b")

        assert evicted == [live], "the live engine must be evicted via the guarded helper"
        assert live.unload.call_count == 0, "must NOT raw-unload the shared engine"
        assert res is new_eng
        assert reused is False, "a freshly loaded engine is owned by the runner"

    def _install_split_gpu(self, monkeypatch, gpu_split_indices):
        """Overlay just gpu_split_indices onto the REAL (test-isolated)
        config, so unrelated load_config() readers still see a complete
        config."""
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
        """The VRAM gate must weigh the swap decision against
        discover.vram_capacity()'s COMBINED split capacity, not just the single
        main GPU: with a 2-GPU split where the single main GPU alone would
        trigger an unload but the combined free comfortably covers the media
        estimate, the live chat engine must NOT be evicted.
        should_swap_for_media runs for REAL here, not mocked."""
        mock_get_model_info.return_value = ("/path/to/llama-3b", "llama-3b")
        new_eng = MagicMock()
        mock_engine_cls.return_value = new_eng

        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        monkeypatch.setattr("localm.inference.http_server._engine", live)

        evicted = []
        monkeypatch.setattr(runner_mod, "_evict_shared_engine_for_media",
                            lambda live_eng: evicted.append(live_eng) or "unloaded")

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

        res, reused = _load_engine("llama-3b")

        assert evicted == [], (
            "combined 20GB free should cover the ~14.93GB threshold without "
            "evicting the live engine, even though neither GPU alone would")
        assert live.unload.call_count == 0
        assert res is new_eng   # a different model is being loaded regardless
        assert reused is False

    @patch("localm.model_manager.get_model_info")
    @patch("localm.inference.engine.Engine")
    def test_same_single_gpu_free_without_split_still_evicts(
        self, mock_engine_cls, mock_get_model_info, monkeypatch
    ):
        """The SAME 10 GB free, but with NO split configured (single GPU
        only), still triggers the eviction."""
        mock_get_model_info.return_value = ("/path/to/llama-3b", "llama-3b")
        new_eng = MagicMock()
        mock_engine_cls.return_value = new_eng

        live = MagicMock()
        live.loaded = True
        live.display_name = "gemma-2b"
        monkeypatch.setattr("localm.inference.http_server._engine", live)

        evicted = []
        monkeypatch.setattr(runner_mod, "_evict_shared_engine_for_media",
                            lambda live_eng: evicted.append(live_eng) or "unloaded")

        GB = 1024 ** 3
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [
            {"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
        ])
        self._install_split_gpu(monkeypatch, None)

        res, reused = _load_engine("llama-3b")

        assert evicted == [live]
        assert res is new_eng
        assert reused is False
