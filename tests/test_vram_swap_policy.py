# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 1: VRAM-aware media swap policy + the C4 driver-hang guard.

The decision: before a media generation, do we unload the chat LLM to free VRAM,
or does the media model fit alongside it (big card) so we keep chat hot?
"""
import importlib
from pathlib import Path

import pytest

from localm.vram import (
    should_swap_for_media,
    resolve_swap_policy,
    wait_for_vram_release,
    decide_media_swap,
    media_estimate_bytes,
)

GB = 1024 ** 3


class _Clock:
    """Deterministic fake clock: sleep advances time; monotonic reads it."""
    def __init__(self):
        self.t = 0.0
    def monotonic(self):
        return self.t
    def sleep(self, s):
        self.t += s


class TestShouldSwapForMedia:
    def test_policy_always_swaps_regardless(self):
        # plenty of free VRAM, but policy forces a swap
        assert should_swap_for_media(free_bytes=100 * GB, media_estimate_bytes=12 * GB,
                                     policy="always") is True

    def test_policy_never_keeps_chat_regardless(self):
        # almost no free VRAM, but policy refuses to swap
        assert should_swap_for_media(free_bytes=1 * GB, media_estimate_bytes=12 * GB,
                                     policy="never") is False

    def test_auto_keeps_chat_when_media_fits_alongside(self):
        # 128 GB-style card: 100 GB free, media needs 12 GB -> fits, keep chat
        assert should_swap_for_media(free_bytes=100 * GB, media_estimate_bytes=12 * GB,
                                     headroom_bytes=1 * GB, policy="auto") is False

    def test_auto_swaps_when_media_does_not_fit(self):
        # 16 GB box mid-chat: only 4 GB free, media needs 12 GB -> must swap
        assert should_swap_for_media(free_bytes=4 * GB, media_estimate_bytes=12 * GB,
                                     headroom_bytes=1 * GB, policy="auto") is True

    def test_auto_swaps_on_the_headroom_boundary(self):
        # free == media exactly, but headroom pushes it over -> swap
        assert should_swap_for_media(free_bytes=12 * GB, media_estimate_bytes=12 * GB,
                                     headroom_bytes=1 * GB, policy="auto") is True
        # free == media + headroom exactly -> just fits, keep
        assert should_swap_for_media(free_bytes=13 * GB, media_estimate_bytes=12 * GB,
                                     headroom_bytes=1 * GB, policy="auto") is False

    def test_auto_swaps_when_free_unmeasurable(self):
        # no torch / no GPU reading -> safe default is to swap (today's behaviour)
        assert should_swap_for_media(free_bytes=None, media_estimate_bytes=12 * GB,
                                     policy="auto") is True

    def test_auto_swaps_when_media_estimate_unknown(self):
        assert should_swap_for_media(free_bytes=100 * GB, media_estimate_bytes=None,
                                     policy="auto") is True


class TestResolveSwapPolicy:
    def test_default_is_auto(self):
        assert resolve_swap_policy(plugin_block={}, full_config={}) == "auto"

    def test_explicit_model_swap_policy_wins(self):
        assert resolve_swap_policy(plugin_block={}, full_config={"model_swap_policy": "never"}) == "never"

    def test_per_plugin_policy_overrides_global(self):
        assert resolve_swap_policy(plugin_block={"model_swap_policy": "never"},
                                   full_config={"model_swap_policy": "always"}) == "never"

    def test_legacy_reload_flag_does_not_affect_policy(self):
        # the legacy reload bool is a SEPARATE axis (reload-after, not swap-before);
        # it must NOT drag the policy to never - it stays the safe default auto.
        assert resolve_swap_policy(plugin_block={},
                                   full_config={"reload_llm_after_imagine": False}) == "auto"
        assert resolve_swap_policy(plugin_block={"reload_llm_after_generate": False},
                                   full_config={}) == "auto"

    def test_explicit_policy_still_wins_over_legacy(self):
        assert resolve_swap_policy(plugin_block={},
                                   full_config={"model_swap_policy": "always",
                                                "reload_llm_after_imagine": False}) == "always"

    def test_invalid_policy_falls_back_to_auto(self):
        assert resolve_swap_policy(plugin_block={}, full_config={"model_swap_policy": "bogus"}) == "auto"


class TestWaitForVramRelease:
    """The C4 driver-hang guard: unload must wait for VRAM to actually free."""

    def test_immediate_release_needs_no_sleep(self):
        clk = _Clock()
        released, final = wait_for_vram_release(
            lambda: 20 * GB, before_bytes=10 * GB, min_rise_bytes=1 * GB,
            sleep=clk.sleep, monotonic=clk.monotonic)
        assert released is True
        assert final == 20 * GB
        assert clk.t == 0.0  # freed on first read, never slept

    def test_release_detected_after_a_couple_polls(self):
        # the native free is deferred: free stays flat for two reads, then rises
        reads = iter([10 * GB, 10 * GB, 16 * GB, 16 * GB])
        clk = _Clock()
        released, final = wait_for_vram_release(
            lambda: next(reads), before_bytes=10 * GB, min_rise_bytes=1 * GB,
            timeout_s=5.0, poll_s=0.1, sleep=clk.sleep, monotonic=clk.monotonic)
        assert released is True
        assert final == 16 * GB
        assert clk.t > 0.0  # it had to wait

    def test_timeout_when_vram_never_frees(self):
        clk = _Clock()
        released, final = wait_for_vram_release(
            lambda: 10 * GB, before_bytes=10 * GB, min_rise_bytes=1 * GB,
            timeout_s=0.3, poll_s=0.1, sleep=clk.sleep, monotonic=clk.monotonic)
        assert released is False  # never rose -> the caller must proceed cautiously

    def test_unmeasurable_before_returns_none(self):
        released, final = wait_for_vram_release(lambda: None, before_bytes=None)
        assert released is None  # cannot verify -> behave as before, do not block

    def test_small_fluctuation_below_threshold_is_not_release(self):
        # a 100 MB jitter must not count as the model freeing (256 MB default floor)
        clk = _Clock()
        released, _ = wait_for_vram_release(
            lambda: 10 * GB + int(100e6), before_bytes=10 * GB,
            timeout_s=0.3, poll_s=0.1, sleep=clk.sleep, monotonic=clk.monotonic)
        assert released is False


class TestMediaEstimateBytes:
    def test_default_per_backend(self):
        assert media_estimate_bytes("image", {}) == int(14 * GB)
        assert media_estimate_bytes("music", {}) == int(12 * GB)
        assert media_estimate_bytes("video", {}) == int(16 * GB)

    def test_unknown_backend_falls_back(self):
        assert media_estimate_bytes("weird", {}) == int(14 * GB)

    def test_per_plugin_override_wins(self):
        assert media_estimate_bytes("image", {"vram_estimate_gb": 8}) == 8 * GB

    def test_bogus_override_ignored(self):
        assert media_estimate_bytes("image", {"vram_estimate_gb": 0}) == int(14 * GB)
        assert media_estimate_bytes("image", {"vram_estimate_gb": "huge"}) == int(14 * GB)
        assert media_estimate_bytes("image", None) == int(14 * GB)


class TestDecideMediaSwap:
    """The live decision: read free VRAM, apply the policy."""

    def _settings(self, policy="auto", est_gb=12):
        return {"swap_policy": policy, "vram_estimate_bytes": est_gb * GB}

    def test_auto_keeps_chat_when_media_fits(self):
        assert decide_media_swap(self._settings(), read_free=lambda: 100 * GB) is False

    def test_auto_swaps_when_tight(self):
        assert decide_media_swap(self._settings(), read_free=lambda: 4 * GB) is True

    def test_never_keeps_even_when_tight(self):
        assert decide_media_swap(self._settings("never"), read_free=lambda: 1 * GB) is False

    def test_always_swaps_even_when_roomy(self):
        assert decide_media_swap(self._settings("always"), read_free=lambda: 100 * GB) is True

    def test_unmeasurable_free_swaps(self):
        assert decide_media_swap(self._settings(), read_free=lambda: None) is True

    def test_missing_policy_defaults_to_auto(self):
        assert decide_media_swap({"vram_estimate_bytes": 12 * GB},
                                 read_free=lambda: 4 * GB) is True


class TestBackendSwapSettings:
    """The media backends surface swap_policy + vram_estimate_bytes (consumed by
    decide_media_swap) and keep the legacy reload toggle working."""

    def _settings(self, name, full_config):
        mod = importlib.import_module(f"localm.plugins.builtin.{name}.backend")
        return mod.settings(full_config)

    @pytest.mark.parametrize("name,gb", [("image", 14), ("music", 12), ("video", 16)])
    def test_defaults(self, name, gb):
        s = self._settings(name, {})
        assert s["swap_policy"] == "auto"
        assert s["vram_estimate_bytes"] == int(gb * GB)

    def test_legacy_reload_false_decouples_from_swap(self):
        # the legacy reload bool drives reload-after (lazy), NOT the swap policy:
        # swap_policy stays the safe auto and reload_after is preserved False.
        s = self._settings("image", {"reload_llm_after_imagine": False})
        assert s["swap_policy"] == "auto"
        assert s["reload_after"] is False
        # auto still swaps when VRAM is tight (no OOM) and keeps chat when it fits
        assert decide_media_swap(s, read_free=lambda: 1 * GB) is True
        assert decide_media_swap(s, read_free=lambda: 100 * GB) is False

    def test_explicit_policy_overrides_legacy(self):
        s = self._settings("video",
                            {"model_swap_policy": "always",
                             "reload_llm_after_imagine": False})
        assert s["swap_policy"] == "always"


@pytest.mark.parametrize("modpath, func, first_arg", [
    ("localm.image_gen.comfy", "generate_image", "a prompt"),
    ("localm.music_gen.comfy", "generate_music", "tags, mood"),
    ("localm.video_gen.comfy", "generate_video", "a prompt"),
])
class TestMediaTransportSwapGate:
    """The shared transports honour the swap flag: unload the chat LLM only when
    swap=True. (Mirrors the media flow: plug.py decides, the transport obeys.)"""

    def _arm(self, monkeypatch, mod, unload_calls):
        monkeypatch.setattr(mod, "ensure_comfy", lambda *a, **k: (True, "ok"))
        monkeypatch.setattr(mod, "_localm_unload",
                            lambda *a, **k: unload_calls.append(1))
        # Make the step right after the unload (workflow load) fail fast so the
        # transport returns without needing a live ComfyUI.
        monkeypatch.setattr(mod, "_workflow_path",
                            lambda: Path("____no_such_workflow____.json"))

    def test_swap_true_unloads(self, monkeypatch, modpath, func, first_arg):
        mod = importlib.import_module(modpath)
        calls = []
        self._arm(monkeypatch, mod, calls)
        ok, _msg = getattr(mod, func)(first_arg, Path("out.bin"), swap=True)
        assert ok is False        # bailed at the (missing) workflow load
        assert calls == [1]       # but the unload DID run first

    def test_swap_false_skips_unload(self, monkeypatch, modpath, func, first_arg):
        mod = importlib.import_module(modpath)
        calls = []
        self._arm(monkeypatch, mod, calls)
        ok, _msg = getattr(mod, func)(first_arg, Path("out.bin"), swap=False)
        assert ok is False
        assert calls == []        # the unload was skipped (media fits)
