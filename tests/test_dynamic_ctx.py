# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the dynamic context window: growth steps, ceiling, auto sizing."""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._fake_batch import fake_batch_init


def _llm(n_ctx=4096, n_ctx_max=16384, n_ctx_grow=4096):
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._n_ctx = n_ctx
    llm._n_ctx_max = max(n_ctx_max, n_ctx) if n_ctx_max else None
    llm._n_ctx_grow = max(256, n_ctx_grow)
    llm._cached_tokens = []
    llm._ctx_capacity = n_ctx
    llm._kv_supported = None
    llm._verbose = False
    llm._model_ptr = None
    llm._ctx_ptr = None
    llm._tokenizer = MagicMock()
    llm._vram_check = None
    return llm


class TestTargetCtx:
    def test_within_base_stays_at_base(self):
        assert _llm()._target_ctx(1000) == 4096

    def test_grows_in_steps(self):
        llm = _llm()
        assert llm._target_ctx(4097) == 8192
        assert llm._target_ctx(8192) == 8192
        assert llm._target_ctx(9000) == 12288

    def test_capped_at_max(self):
        llm = _llm(n_ctx_max=10000)
        assert llm._target_ctx(9999) == 10000

    def test_unlimited_when_no_max(self):
        llm = _llm(n_ctx_max=0)
        assert llm._target_ctx(50_000) == 53248   # next 4096 multiple

    def test_custom_grow_step(self):
        llm = _llm(n_ctx_grow=2048)
        assert llm._target_ctx(4097) == 6144

    def test_explicit_base_beats_smaller_cap(self):
        # User asked for -c 32768; a 16384 cap must not shrink it
        llm = _llm(n_ctx=32768, n_ctx_max=16384)
        assert llm._n_ctx_max == 32768
        assert llm._target_ctx(1000) == 32768


class TestGenerationBudget:
    def test_untouched_when_fits(self):
        assert _llm()._fit_generation_budget(1000, 1024) == 1024

    def test_clamped_near_ceiling(self):
        llm = _llm(n_ctx_max=8192)
        # room = 8192 - 7000 - 64 = 1128
        assert llm._fit_generation_budget(7000, 4096) == 1128

    def test_raises_when_conversation_outgrows_ceiling(self):
        llm = _llm(n_ctx_max=8192)
        with pytest.raises(RuntimeError, match="n_ctx_max"):
            llm._fit_generation_budget(8190, 1024)

    def test_unlimited_never_clamps(self):
        llm = _llm(n_ctx_max=0)
        assert llm._fit_generation_budget(100_000, 4096) == 4096


class TestFreshContextUsesPolicy:
    def test_rebuild_rounds_to_grow_step(self):
        llm = _llm()
        llm._ctx_ptr = 222
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_ctx == 8192          # rounded up, not exact-fit 5000
        assert llm._ctx_capacity == 8192

    def test_rebuild_respects_cap(self):
        llm = _llm(n_ctx_max=6000)
        llm._ctx_ptr = 222
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5500)
        assert cp.n_ctx == 6000


class TestFreshContextVramCheck:
    """_prefill_fresh_context() recreates a BIGGER context whenever a
    conversation outgrows the live one - which happens on the first prompt for
    anyone on default settings, since the default max_tokens (4096) already
    exceeds the default base n_ctx (4096), forcing a grow to 8192. That native
    (re)allocation needs a VRAM preflight, not just a NULL-pointer check after
    the fact; an optional vram_check callback (wired by
    GgufBackend._check_context_fit) provides it."""

    def test_vram_check_called_with_target_and_current_ctx_before_growing(self):
        llm = _llm()
        llm._ctx_ptr = 222
        llm._ctx_capacity = 4096
        calls = []
        # The hook receives (target, current_ctx) and returns where the KV cache
        # goes (True=VRAM, False=RAM, None=default), charging only the NET growth.
        llm._vram_check = lambda n_ctx, current=0: calls.append((n_ctx, current))
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert calls == [(8192, 4096)]   # rounded-up target + the live ctx size

    def test_vram_check_false_puts_kv_cache_in_system_ram(self):
        # When the hook returns False (KV cache does not fit VRAM), _prefill keeps
        # the FULL window (target unchanged) but creates the context with
        # offload_kqv False so the KV cache lives in system RAM - a degrade, not a
        # shrink, not an abort.
        llm = _llm()
        llm._ctx_ptr = 222
        llm._ctx_capacity = 4096
        llm._vram_check = lambda n_ctx, current=0: False
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_ctx == 8192          # FULL window kept (not shrunk)
        assert cp.offload_kqv is False   # KV cache placed in system RAM

    def test_vram_check_true_keeps_kv_cache_in_vram(self):
        llm = _llm()
        llm._ctx_ptr = 222
        llm._ctx_capacity = 4096
        llm._vram_check = lambda n_ctx, current=0: True
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_ctx == 8192
        assert cp.offload_kqv is True    # KV cache stays in VRAM (fast)

    def test_vram_check_exception_preserves_old_context(self):
        # The hook is consulted BEFORE the live context is freed, so an exception
        # in it leaves the OLD, still-working context intact.
        llm = _llm()
        llm._ctx_ptr = 222

        def _boom(n_ctx, current=0):
            raise RuntimeError("hook blew up")

        llm._vram_check = _boom
        mock_api = MagicMock()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            with pytest.raises(RuntimeError, match="blew up"):
                llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert llm._ctx_ptr == 222          # old context untouched

    def test_context_creation_failure_surfaces_clean_last_resort_error(self):
        # Last resort: if the native context cannot be created even with the KV
        # cache in system RAM (llama_init_from_model returns NULL), surface a clear
        # "not enough memory" message, not a bare failure or a crash.
        llm = _llm()
        llm._ctx_ptr = 222
        llm._ctx_capacity = 4096
        llm._vram_check = lambda n_ctx, current=0: False   # KV to RAM (VRAM was tight)
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 0    # NULL: allocation failed
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            with pytest.raises(RuntimeError, match="Not enough memory"):
                llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.offload_kqv is False     # it did try the system-RAM path first

    def test_no_vram_check_configured_is_a_noop(self):
        # A LlamaCpp built without a vram_check (the default) takes the plain path.
        llm = _llm()
        llm._ctx_ptr = 222
        assert llm._vram_check is None
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_ctx == 8192


class TestAutoCtxMax:
    def _backend(self, free_vram, model_bytes, n_ctx=4096):
        from localm.inference.backends.gguf import GgufBackend
        b = GgufBackend.__new__(GgufBackend)
        b.n_ctx = n_ctx
        b.n_ctx_max = None
        b.ctx_auto = True
        b._free_vram_bytes = lambda: free_vram
        b._model_bytes = lambda: model_bytes
        return b

    def test_no_gpu_falls_back(self):
        b = self._backend(free_vram=None, model_bytes=int(13e9))
        assert b._auto_ctx_max() == 16384

    def test_no_headroom_returns_minimum(self):
        b = self._backend(free_vram=int(14e9), model_bytes=int(13e9))
        assert b._auto_ctx_max() == 4096

    def test_plenty_of_vram_is_clamped_to_upper_bound(self):
        b = self._backend(free_vram=int(80e9), model_bytes=int(2e9))
        assert b._auto_ctx_max() == 65536

    def test_mid_range_scales_with_budget(self):
        # 16GB free, 13GB model, 1.5GB overhead → ~1.5GB KV budget
        # bytes/token = 13e9 / 100k = 130KB → ~11.5k tokens → 11264 rounded
        b = self._backend(free_vram=int(16e9), model_bytes=int(13e9))
        result = b._auto_ctx_max()
        assert 8192 <= result <= 13312
        assert result % 1024 == 0

    def test_result_always_in_sane_band(self):
        for free in (int(5e9), int(10e9), int(24e9), int(48e9)):
            for model in (int(1e9), int(8e9), int(13e9), int(30e9)):
                b = self._backend(free_vram=free, model_bytes=model)
                v = b._auto_ctx_max()
                assert 4096 <= v <= 65536


class TestUnlimitedCtxMax:
    """n_ctx_max=0 ('unlimited') must lift the _AUTO_CTX_MAX safety clamp when
    ctx_auto is on, so the window grows to the full VRAM-derived budget instead
    of being capped at 65536."""

    def _backend(self, free_vram, model_bytes, *, n_ctx_max, ctx_auto, n_ctx=4096):
        from localm.inference.backends.gguf import GgufBackend
        b = GgufBackend.__new__(GgufBackend)
        b.n_ctx = n_ctx
        b.n_ctx_max = n_ctx_max
        b.ctx_auto = ctx_auto
        b._free_vram_bytes = lambda: free_vram
        b._model_bytes = lambda: model_bytes
        return b

    def test_unlimited_lifts_the_clamp(self):
        # 80GB free, 2GB model: the capped auto would be 65536, but n_ctx_max=0 with
        # ctx_auto must return the full VRAM-derived budget (well above the clamp).
        b = self._backend(int(80e9), int(2e9), n_ctx_max=0, ctx_auto=True)
        eff = b._effective_ctx_max()
        assert eff > b._AUTO_CTX_MAX
        assert eff == b._auto_ctx_max(capped=False)

    def test_auto_with_positive_max_still_capped(self):
        # ctx_auto on plus a positive max: the 65536 safety clamp applies.
        b = self._backend(int(80e9), int(2e9), n_ctx_max=16384, ctx_auto=True)
        assert b._effective_ctx_max() == b._AUTO_CTX_MAX

    def test_unlimited_without_auto_passes_through(self):
        # ctx_auto off: n_ctx_max is used verbatim; 0 means unlimited downstream.
        b = self._backend(int(80e9), int(2e9), n_ctx_max=0, ctx_auto=False)
        assert b._effective_ctx_max() == 0

    def test_positive_max_without_auto_passes_through(self):
        b = self._backend(int(80e9), int(2e9), n_ctx_max=16384, ctx_auto=False)
        assert b._effective_ctx_max() == 16384
