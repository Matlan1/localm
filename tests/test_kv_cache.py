# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for persistent KV cache prefix reuse in the native llama.cpp wrapper.

The native DLL is never loaded - the api module is mocked throughout.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from localm.inference.backends.llamacpp.llama import (
    LlamaCpp,
    _build_sampler,
    _common_prefix_len,
)
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init


# ---------------------------------------------------------------------------
#  _common_prefix_len
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param([1, 2, 3], [1, 2, 3], 3, id="identical"),
        pytest.param([1, 2], [9, 8], 0, id="disjoint"),
        pytest.param([1, 2, 3, 4], [1, 2, 9], 2, id="partial"),
        pytest.param([1, 2], [1, 2, 3, 4], 2, id="one_is_prefix_of_other"),
        pytest.param([], [1, 2], 0, id="empty"),
    ],
)
def test_common_prefix_len(a, b, expected):
    assert _common_prefix_len(a, b) == expected


# ---------------------------------------------------------------------------
#  Wrapper-level KV behaviour (api module fully mocked)
# ---------------------------------------------------------------------------

def _bare_llama() -> LlamaCpp:
    """Construct a LlamaCpp without running __init__ (no DLL access).

    Delegates to the shared builder (tests/_bare_llama.py) so every
    __init__ invariant is set, not just the ones this file happens to
    read - see that module's docstring for why fourteen hand-maintained
    copies of this object were the actual defect.
    """
    return make_bare_llama(_model_ptr=111, _ctx_ptr=222)


@pytest.fixture(autouse=True)
def _no_native_mrope_probe():
    """_can_reuse_kv asks the model whether it uses M-RoPE, and that probe is a
    REAL native call. Handed this file's fake integer model pointer it loads the
    llama.cpp runtime and faults on a bad address, so answering it here is what
    keeps the module docstring's promise that the DLL is never touched. Tests
    that care about the M-RoPE branch patch it themselves."""
    with patch(
        "localm.inference.backends.llamacpp.llama.api.llama_model_has_mrope",
        return_value=False,
    ):
        yield


# Fake-pointer teardown is now handled globally by tests/conftest.py's
# autouse _neutralise_bare_llama_pointers fixture.


class TestCanReuseKv:
    def test_reuse_when_supported_and_fits(self):
        llm = _bare_llama()
        with patch(
            "localm.inference.backends.llamacpp.llama.api.has_memory_api",
            return_value=True,
        ):
            assert llm._can_reuse_kv(1000) is True

    def test_no_reuse_without_memory_api(self):
        llm = _bare_llama()
        with patch(
            "localm.inference.backends.llamacpp.llama.api.has_memory_api",
            return_value=False,
        ):
            assert llm._can_reuse_kv(1000) is False

    def test_no_reuse_when_request_exceeds_capacity(self):
        llm = _bare_llama()
        llm._kv_supported = True
        assert llm._can_reuse_kv(5000) is False

    def test_no_reuse_without_live_context(self):
        llm = _bare_llama()
        llm._ctx_ptr = None
        llm._kv_supported = True
        assert llm._can_reuse_kv(100) is False

    def test_probe_failure_treated_as_unsupported(self):
        llm = _bare_llama()
        with patch(
            "localm.inference.backends.llamacpp.llama.api.has_memory_api",
            side_effect=AttributeError("missing"),
        ):
            assert llm._can_reuse_kv(100) is False
        assert llm._kv_supported is False

    def test_no_reuse_for_mrope_models(self):
        """M-RoPE positions tokens on a multi-dimensional coordinate grid that
        sequence removal cannot rewind, so the context must start clean."""
        llm = _bare_llama()
        llm._kv_supported = True
        with patch(
            "localm.inference.backends.llamacpp.llama.api.llama_model_has_mrope",
            return_value=True,
        ):
            assert llm._can_reuse_kv(100) is False

    def test_probe_result_cached(self):
        llm = _bare_llama()
        with patch(
            "localm.inference.backends.llamacpp.llama.api.has_memory_api",
            return_value=True,
        ) as probe:
            llm._can_reuse_kv(100)
            llm._can_reuse_kv(100)
        probe.assert_called_once()


class TestPrefillWithReuse:
    def _patch_api(self, **overrides):
        mock_api = MagicMock()
        mock_api.llama_get_memory.return_value = 333
        mock_api.llama_memory_seq_rm.return_value = True
        mock_api.llama_decode.return_value = 0
        # Real ctypes-backed batch so _create_batch's native fill loop actually
        # runs (no mock-detection facade in production any more).
        mock_api.llama_batch_init.side_effect = fake_batch_init
        for k, v in overrides.items():
            setattr(mock_api, k, v)
        return patch(
            "localm.inference.backends.llamacpp.llama.api", mock_api
        ), mock_api

    def test_only_suffix_decoded_on_shared_prefix(self):
        llm = _bare_llama()
        llm._cached_tokens = [1, 2, 3, 4]
        ctx, mock_api = self._patch_api()
        with ctx:
            llm._prefill_with_reuse([1, 2, 3, 4, 5, 6])

        # Cached prefix [1,2,3,4] kept → seq_rm not needed (prefix == cache len)
        mock_api.llama_memory_seq_rm.assert_not_called()
        # Exactly one decode call for the 2-token suffix
        assert mock_api.llama_decode.call_count == 1
        args = mock_api.llama_batch_init.call_args[0]
        assert args[0] == 2  # n_tokens of the suffix batch
        assert llm._cached_tokens == [1, 2, 3, 4, 5, 6]

    def test_diverging_cache_trimmed(self):
        llm = _bare_llama()
        llm._cached_tokens = [1, 2, 9, 9]
        ctx, mock_api = self._patch_api()
        with ctx:
            llm._prefill_with_reuse([1, 2, 3, 4])

        # Diverges at index 2 → remove cached positions [2, end)
        mock_api.llama_memory_seq_rm.assert_called_once_with(333, 0, 2, -1)
        assert llm._cached_tokens == [1, 2, 3, 4]

    def test_identical_prompt_redecodes_last_token(self):
        """Same prompt again: last token re-decoded so logits are fresh."""
        llm = _bare_llama()
        llm._cached_tokens = [1, 2, 3]
        ctx, mock_api = self._patch_api()
        with ctx:
            llm._prefill_with_reuse([1, 2, 3])

        mock_api.llama_memory_seq_rm.assert_called_once_with(333, 0, 2, -1)
        args = mock_api.llama_batch_init.call_args[0]
        assert args[0] == 1  # only the final token decoded

    def test_seq_rm_failure_falls_back_to_full_clear(self):
        llm = _bare_llama()
        llm._cached_tokens = [1, 2, 9]
        ctx, mock_api = self._patch_api(
            llama_memory_seq_rm=MagicMock(return_value=False))
        with ctx:
            llm._prefill_with_reuse([1, 2, 3])

        mock_api.llama_memory_clear.assert_called_once()
        # Full prompt decoded from position 0
        args = mock_api.llama_batch_init.call_args[0]
        assert args[0] == 3
        assert llm._cached_tokens == [1, 2, 3]

    def test_empty_cache_clears_stale_native_kv_before_reuse(self):
        """U-1: an image turn (_generate_image never appends its tokens) and a
        mid-generate decode failure empty _cached_tokens but leave the NATIVE KV
        populated. Reusing the context must drop that residual KV first (prefix=0,
        so seq_rm(0, 0, -1)), else the new prompt decodes onto stale KV at shifted
        positions and the model attends to the previous turn's context."""
        llm = _bare_llama()
        llm._cached_tokens = []                 # bookkeeping says empty...
        ctx, mock_api = self._patch_api()       # ...native KV still holds a prior turn
        with ctx:
            llm._prefill_with_reuse([1, 2, 3])

        mock_api.llama_memory_clear.assert_called_once_with(333, True)
        mock_api.llama_memory_seq_rm.assert_not_called()
        # Full prompt decoded from position 0 after the wipe.
        args = mock_api.llama_batch_init.call_args[0]
        assert args[0] == 3
        assert llm._cached_tokens == [1, 2, 3]

    def test_zero_prefix_also_clears_the_draft_context_memory(self):
        """A model with MTP heads keeps a second KV cache for its draft context.
        Wiping only the main one leaves the draft still holding the previous
        turn, so the two disagree about what has been seen."""
        llm = _bare_llama()
        llm._cached_tokens = []
        llm._mtp_ctx_ptr = 444
        ctx, mock_api = self._patch_api()
        mock_api.llama_get_memory.side_effect = {222: 333, 444: 555}.__getitem__
        with ctx:
            llm._prefill_with_reuse([1, 2, 3])

        assert mock_api.llama_memory_clear.call_args_list == [
            call(333, True), call(555, True)]
        assert llm._cached_tokens == [1, 2, 3]

    def test_decode_failure_wipes_cache_state(self):
        llm = _bare_llama()
        llm._cached_tokens = [1, 2]
        ctx, mock_api = self._patch_api(
            llama_decode=MagicMock(return_value=-1))
        with ctx:
            with pytest.raises(RuntimeError, match="prefill"):
                llm._prefill_with_reuse([1, 2, 3, 4])

        assert llm._cached_tokens == []
        mock_api.llama_memory_clear.assert_called_once()

    def test_long_suffix_chunked(self):
        llm = _bare_llama()
        llm._cached_tokens = []
        llm._ctx_capacity = 10_000
        ctx, mock_api = self._patch_api()
        with ctx:
            llm._prefill_with_reuse(list(range(5000)))

        # 5000 tokens → chunks of 2048 → 3 decode calls
        assert mock_api.llama_decode.call_count == 3


class TestFreshContextPath:
    def test_fresh_context_resets_cache_and_capacity(self):
        llm = _bare_llama()
        llm._cached_tokens = [9, 9, 9]
        mock_api = MagicMock()
        mock_api.llama_context_default_params.return_value = MagicMock()
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context([1, 2, 3], needed=5000)

        mock_api.llama_free.assert_called_once_with(222)
        assert llm._ctx_ptr == 444
        assert llm._cached_tokens == [1, 2, 3]
        # Capacity grew to fit the request
        assert llm._ctx_capacity >= 5000

    def test_fresh_context_decode_failure_raises(self):
        llm = _bare_llama()
        mock_api = MagicMock()
        mock_api.llama_context_default_params.return_value = MagicMock()
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = -1
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            with pytest.raises(RuntimeError, match="prefill"):
                llm._prefill_fresh_context([1, 2, 3], needed=100)
        # No stale cache claim after failure
        assert llm._cached_tokens == []

    def test_long_prompt_prefill_is_chunked(self):
        """Regression: a fresh-context prefill larger than n_batch must be
        split into n_batch-sized decode calls. A single oversized batch
        aborts the process inside the native library (crash reported after
        long chat histories forced a context rebuild)."""
        llm = _bare_llama()
        mock_api = MagicMock()
        cp = MagicMock()
        mock_api.llama_context_default_params.return_value = cp
        mock_api.llama_init_from_model.return_value = 444
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm._prefill_fresh_context(list(range(5000)), needed=6000)

        # n_batch is capped at 2048 → 5000 tokens need 3 decode calls
        assert mock_api.llama_decode.call_count == 3
        batch_sizes = [c[0][0] for c in mock_api.llama_batch_init.call_args_list]
        assert batch_sizes == [2048, 2048, 904]
        assert all(size <= 2048 for size in batch_sizes)
        assert llm._cached_tokens == list(range(5000))

    def test_close_clears_cached_tokens(self):
        llm = _bare_llama()
        llm._cached_tokens = [1, 2, 3]
        mock_api = MagicMock()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm.close()
        assert llm._cached_tokens == []
        assert llm._ctx_ptr is None


# ---------------------------------------------------------------------------
#  Sampler chain - repetition penalty
# ---------------------------------------------------------------------------

class TestRepeatPenaltySampler:
    """Regression: repeat_penalty was accepted everywhere but never added to
    the sampler chain, so models prone to looping repeated marker lines
    until max_tokens."""

    def _mock_api(self, has_penalties=True, n_vocab=32000, needs_n_vocab=False):
        mock_api = MagicMock()
        mock_api.has_penalties_sampler.return_value = has_penalties
        # Newer llama builds take the vocabulary size as a leading argument
        # (upstream #26520); _api dispatches on the build but the caller must
        # supply the real value, so the chain builder reads it off the vocab.
        mock_api.llama_vocab_n_tokens.return_value = n_vocab
        mock_api.penalties_needs_n_vocab.return_value = needs_n_vocab
        return mock_api

    def test_penalty_added_when_set(self):
        mock_api = self._mock_api()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=1, repeat_penalty=1.1)
        mock_api.llama_sampler_init_penalties.assert_called_once_with(
            64, 1.1, 0.0, 0.0, n_vocab=32000)

    def test_n_vocab_is_read_from_the_real_vocab_not_hardcoded(self):
        """A 0 or wrong n_vocab under-allocates the native sampler's per-token
        frequency counters, so the value has to come from the model."""
        mock_api = self._mock_api(n_vocab=151936)
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=0xABCD, repeat_penalty=1.1)
        mock_api.llama_vocab_n_tokens.assert_called_once_with(0xABCD)
        assert mock_api.llama_sampler_init_penalties.call_args.kwargs[
            "n_vocab"] == 151936

    def test_skipped_when_build_needs_n_vocab_and_none_is_available(self):
        """Rather than call a 5-argument penalties sampler with n_vocab=0."""
        mock_api = self._mock_api(n_vocab=0, needs_n_vocab=True)
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=0, repeat_penalty=1.1)
        mock_api.llama_sampler_init_penalties.assert_not_called()

    def test_still_added_when_build_does_not_need_n_vocab(self):
        """The older 4-argument builds ignore n_vocab, so a missing vocab must
        NOT cost those users their repetition penalty."""
        mock_api = self._mock_api(n_vocab=0, needs_n_vocab=False)
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=0, repeat_penalty=1.1)
        mock_api.llama_sampler_init_penalties.assert_called_once()

    def test_no_penalty_at_neutral_value(self):
        mock_api = self._mock_api()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=1, repeat_penalty=1.0)
        mock_api.llama_sampler_init_penalties.assert_not_called()

    def test_skipped_when_dll_lacks_export(self):
        mock_api = self._mock_api(has_penalties=False)
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=1, repeat_penalty=1.3)
        mock_api.llama_sampler_init_penalties.assert_not_called()

    def test_penalty_applies_in_greedy_mode_too(self):
        mock_api = self._mock_api()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            _build_sampler(vocab=1, temperature=0.0, repeat_penalty=1.2)
        mock_api.llama_sampler_init_penalties.assert_called_once()


# ---------------------------------------------------------------------------
#  Inference Serialization Lock
# ---------------------------------------------------------------------------

class TestInferenceLock:
    def test_inference_lock_held_during_generate(self):
        llm = _bare_llama()
        
        # Patch the dependencies needed for _generate
        mock_api = MagicMock()
        mock_api.llama_sampler_sample.return_value = 42
        mock_api.llama_sampler_free = MagicMock()
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init

        # Make the tokenizer return False for is_eog so it generates some tokens
        llm._tokenizer.is_eog.return_value = False
        
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
            
            # Start the generator
            gen = llm._generate(prompt_tokens=[1, 2, 3], max_new_tokens=2,
                                temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1)
            
            # The lock is NOT acquired yet because the generator has not been iterated
            assert not llm._inference_lock.locked()
            
            # Pull the first token
            tok1 = next(gen)
            assert tok1 == 42
            
            # Now the lock IS acquired because we are inside the generator's execution
            assert llm._inference_lock.locked()
            
            # Pull the second token
            tok2 = next(gen)
            assert tok2 == 42
            
            # Complete the generator
            with pytest.raises(StopIteration):
                next(gen)
                
            # Now the lock is released
            assert not llm._inference_lock.locked()



# ---------------------------------------------------------------------------
#  Early-exit cleanup in _generate
# ---------------------------------------------------------------------------

class TestGenerateEarlyExitCleanup:
    """Regression: the cleanup block frees a draft sampler that is only bound
    after prefill, so every exit taken before that point raised
    UnboundLocalError - destroying the real reason the request stopped."""

    def _mock_api(self, **overrides):
        mock_api = MagicMock()
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        for k, v in overrides.items():
            setattr(mock_api, k, v)
        return mock_api

    def _run(self, llm, mock_api):
        return llm._generate(
            prompt_tokens=[1, 2, 3], max_new_tokens=4,
            temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1)

    def test_stop_before_prefill_ends_cleanly(self):
        """An unload or a user Stop landing before prefill: no tokens, no error."""
        llm = _bare_llama()
        llm._stop.set()
        mock_api = self._mock_api()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            produced = list(self._run(llm, mock_api))
        assert produced == []
        # Nothing was allocated this early, so cleanup must free nothing.
        mock_api.llama_sampler_free.assert_not_called()

    def test_context_creation_failure_reaches_the_caller_intact(self):
        """The out-of-memory diagnostic IS the value of this failure - it tells
        the user to start a new chat or lower n_ctx_max. Cleanup must not
        replace it, and must not swallow it either."""
        llm = _bare_llama()
        # A NULL context back from llama_init_from_model is how the native
        # library reports that the requested window does not fit.
        mock_api = self._mock_api(
            llama_init_from_model=MagicMock(return_value=0))
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            gen = self._run(llm, mock_api)
            with pytest.raises(RuntimeError) as excinfo:
                next(gen)
        message = str(excinfo.value)
        assert "Not enough memory to create a" in message
        assert "lower n_ctx_max" in message


# ---------------------------------------------------------------------------
#  MTP speculative drafting vs grammar-constrained sampling
# ---------------------------------------------------------------------------

class TestMtpDraftingRespectsGrammar:
    """Drafting picks tokens with a bare greedy sampler and then accepts them
    into the main chain. With a grammar in that chain the accepted token was
    never masked by the grammar, so a JSON-schema or tool-calling reply could
    emit text the schema forbids - and the out-of-step accept is the documented
    cause of a native abort. Constrained requests take the single-token path."""

    _MTP_CTX = 444

    def _mock_api(self):
        mock_api = MagicMock()
        mock_api.llama_sampler_sample.return_value = 42
        mock_api.llama_decode.return_value = 0
        mock_api.llama_batch_init.side_effect = fake_batch_init
        return mock_api

    def _drive(self, grammar):
        llm = _bare_llama()
        llm._mtp_ctx_ptr = self._MTP_CTX
        llm._tokenizer.is_eog.return_value = False
        mock_api = self._mock_api()
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=999):
            list(llm._generate(
                prompt_tokens=[1, 2, 3], max_new_tokens=2, temperature=0.8,
                top_k=40, top_p=0.95, repeat_penalty=1.1, grammar=grammar))
        # Sampling against the draft context is what drafting does and nothing
        # else does. Counting DECODES there would also catch the draft
        # context's own prefill, which happens either way.
        drafted = [c for c in mock_api.llama_sampler_sample.call_args_list
                   if c[0][1] == self._MTP_CTX]
        return mock_api, drafted

    def test_no_drafting_while_a_grammar_constrains_sampling(self):
        mock_api, drafted = self._drive(grammar='root ::= "a"')
        mock_api.llama_sampler_init_greedy.assert_not_called()
        assert drafted == []
        # A token chosen off-grammar must never be pushed into the real chain.
        mock_api.llama_sampler_accept.assert_not_called()

    def test_drafting_still_runs_for_unconstrained_requests(self):
        """The gate is narrow on purpose: MTP models keep their speedup on
        ordinary chat, which is what makes this a refusal and not a disable."""
        mock_api, drafted = self._drive(grammar=None)
        mock_api.llama_sampler_init_greedy.assert_called_once()
        assert drafted, "MTP drafting should still run without a grammar"
