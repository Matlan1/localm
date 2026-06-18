"""Phase 1: VRAM-aware media swap policy + the C4 driver-hang guard.

The decision: before a media generation, do we unload the chat LLM to free VRAM,
or does the media model fit alongside it (big card) so we keep chat hot?
"""
import pytest

from localm.vram import (
    should_swap_for_media,
    resolve_swap_policy,
    wait_for_vram_release,
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

    def test_legacy_reload_false_means_never(self):
        # back-compat: the old reload_llm_after_imagine=false (keep media loaded) -> never swap chat
        assert resolve_swap_policy(plugin_block={},
                                   full_config={"reload_llm_after_imagine": False}) == "never"

    def test_legacy_per_plugin_reload_false_means_never(self):
        assert resolve_swap_policy(plugin_block={"reload_llm_after_generate": False},
                                   full_config={}) == "never"

    def test_explicit_policy_overrides_legacy(self):
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
