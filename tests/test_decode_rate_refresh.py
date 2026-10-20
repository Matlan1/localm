# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reply that decoded far below the rate the model reached at its KV depth
and placement gets the context recreated, same size and placement, before the
next request."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init

API = "localm.inference.backends.llamacpp.llama.api"


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.cost = 0.02

    def __call__(self):
        return self.now


def _llm(capacity=4096, *, offload_kqv=True, n_ctx=4096, grow=4096):
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _n_ctx=n_ctx,
                          _n_ctx_max=65536, _n_ctx_grow=grow,
                          _ctx_capacity=capacity, _offload_kqv=offload_kqv,
                          _kv_supported=True)
    llm._tokenizer.is_eog.return_value = False
    llm._clock = _Clock()
    return llm


def _api(llm, init=444, sample_cost=0.0):
    api = MagicMock()
    cp = MagicMock()
    api.llama_context_default_params.return_value = cp
    if isinstance(init, list):
        api.llama_init_from_model.side_effect = init
    else:
        api.llama_init_from_model.return_value = init

    def decode(ctx, batch):
        llm._clock.now += llm._clock.cost
        return 0
    api.llama_decode.side_effect = decode
    api.llama_batch_init.side_effect = fake_batch_init
    def sample(sampler, ctx, row):
        llm._clock.now += sample_cost
        return 42
    api.llama_sampler_sample.side_effect = sample
    api.llama_memory_seq_rm.return_value = True
    return api, cp


def _generate(llm, api, prompt=(1, 2, 3), n=40, cost=None, take=None, **kw):
    if cost is not None:
        llm._clock.cost = cost
    with patch(API, api), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
        gen = llm._generate(prompt_tokens=list(prompt), max_new_tokens=n,
                            temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0, **kw)
        if take is None:
            return list(gen)
        out = [next(gen) for _ in range(take)]
        gen.close()
        return out


def _inits(api):
    return api.llama_init_from_model.call_count


class TestSingleSlot:
    def test_replies_at_a_steady_rate_never_refresh(self):
        llm = _llm()
        api, _cp = _api(llm)
        for _ in range(4):
            _generate(llm, api)
        assert llm._decode_refresh is None
        assert _inits(api) == 0

    def test_one_slow_reply_refreshes_once_before_the_next_request(self, caplog):
        llm = _llm()
        api, cp = _api(llm)
        _generate(llm, api, cost=0.02)
        _generate(llm, api, cost=0.2)
        assert llm._decode_refresh is not None
        assert _inits(api) == 0
        with caplog.at_level(logging.INFO, logger="localm"):
            assert _generate(llm, api, cost=0.02) == [42] * 40
        assert _inits(api) == 1
        assert cp.n_ctx == 4096 and cp.offload_kqv is True
        assert llm._ctx_ptr == 444 and llm._ctx_capacity == 4096
        assert llm._decode_refresh is None
        lines = [r.getMessage() for r in caplog.records if "recreating the" in r.getMessage()]
        assert len(lines) == 1
        assert "5.0 steps/s where 50.0 was expected" in lines[0]
        _generate(llm, api, cost=0.02)
        assert _inits(api) == 1

    def test_the_recreated_context_holds_the_new_prompt(self):
        llm = _llm()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        _generate(llm, api, cost=0.2)
        prompt = list(range(1, 300))
        _generate(llm, api, prompt=prompt, cost=0.02)
        assert llm._cached_tokens[:len(prompt)] == prompt

    def test_a_kv_cache_in_ram_is_refreshed_in_ram(self):
        llm = _llm(offload_kqv=False)
        api, cp = _api(llm)
        _generate(llm, api, cost=0.05)
        _generate(llm, api, cost=0.5)
        _generate(llm, api, cost=0.05)
        assert _inits(api) == 1
        assert cp.offload_kqv is False and llm._offload_kqv is False

    def test_a_refresh_that_cannot_get_vram_keeps_the_kv_cache_in_ram(self):
        llm = _llm()
        api, cp = _api(llm, init=[0, 555])
        _generate(llm, api, cost=0.02)
        _generate(llm, api, cost=0.2)
        assert _generate(llm, api, cost=0.02) == [42] * 40
        assert llm._ctx_ptr == 555 and llm._offload_kqv is False
        assert cp.offload_kqv is False

    def test_a_grammar_reply_is_not_judged(self):
        llm = _llm()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        _generate(llm, api, cost=0.2, grammar='root ::= "a"')
        assert llm._decode_refresh is None

    def test_a_logprobs_reply_is_not_judged(self):
        llm = _llm()
        api, _cp = _api(llm)
        scorer = MagicMock()
        scorer.score.side_effect = lambda ctx, row, tok: tok
        _generate(llm, api, cost=0.02)
        with patch.object(type(llm), "_logprob_scorer", return_value=scorer):
            _generate(llm, api, cost=0.2, logprobs=2)
        assert llm._decode_refresh is None

    def test_a_short_reply_is_not_judged(self):
        llm = _llm()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        _generate(llm, api, n=20, cost=0.2)
        assert llm._decode_refresh is None

    def test_a_reply_ended_by_its_consumer_is_judged(self):
        llm = _llm()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        assert _generate(llm, api, n=200, take=60, cost=0.2) == [42] * 60
        assert llm._decode_refresh is not None

    def test_a_reply_cut_by_an_unload_is_not_judged(self):
        llm = _llm()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        llm._clock.cost = 0.2
        with patch(API, api), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
            gen = llm._generate(prompt_tokens=[1, 2, 3], max_new_tokens=200,
                                temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0)
            for _ in range(60):
                next(gen)
            llm._stop.set()
            list(gen)
        assert llm._decode_refresh is None

    def test_a_reply_whose_context_grew_is_not_judged(self):
        llm = _llm(capacity=512, n_ctx=256, grow=256)
        api, _cp = _api(llm)
        _generate(llm, api, n=100, cost=0.02)
        calls = {"n": 0}

        def decode(ctx, batch):
            calls["n"] += 1
            llm._clock.now += llm._clock.cost
            return 1 if calls["n"] == 60 else 0
        api.llama_decode.side_effect = decode
        llm._clock.cost = 0.2
        prompt = list(range(1, 300))
        _generate(llm, api, prompt=prompt, n=200)
        assert llm._ctx_capacity > 512
        assert llm._decode_refresh is None

    def test_time_spent_waiting_in_the_next_sample_counts(self):
        llm = _llm()
        api, _cp = _api(llm, sample_cost=0.03)
        llm._clock.cost = 0.0
        llm._note_decode_rate = MagicMock()
        _generate(llm, api)
        steps, seconds, _depth = llm._note_decode_rate.call_args.args
        assert steps == 39
        assert seconds == pytest.approx(39 * 0.03)

    def test_a_load_not_fully_on_a_gpu_is_never_judged(self):
        llm = _llm()
        llm._decode_judged = False
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02)
        _generate(llm, api, cost=0.2)
        assert llm._decode_refresh is None
        assert llm._decode_samples is None

    def test_a_reply_without_prompt_cache_on_a_slotted_model_is_judged(self):
        import contextlib
        llm = _llm()
        llm._slots = MagicMock()
        llm._slots.exclusive.return_value = contextlib.nullcontext()
        api, _cp = _api(llm)
        _generate(llm, api, cost=0.02, cache_prompt=False)
        _generate(llm, api, cost=0.2, cache_prompt=False)
        assert llm._slots.exclusive.call_count == 2
        assert llm._decode_refresh is not None
        _generate(llm, api, cost=0.02, cache_prompt=False)
        assert api.llama_init_from_model.call_count == 1

    def test_a_speculating_reply_counts_main_decodes_not_tokens(self):
        llm = _llm()
        api, _cp = _api(llm, sample_cost=0.005)
        llm._note_decode_rate = MagicMock()
        source = MagicMock()
        source.begin_call.return_value = False
        source.drafting.return_value = True
        source.observes_steps = False
        source.ready.return_value = True
        source.budget.return_value = 3
        source.free_miss = False
        source.propose.side_effect = lambda tok, pos, n: (
            setattr(llm._clock, "now", llm._clock.now + 1.0) or [42, 42, 42])
        pacer = MagicMock()
        pacer.paused = False
        pacer.speculate.return_value = True
        llm._draft_pacer = pacer
        with patch.object(type(llm), "_draft_source", return_value=source):
            out = _generate(llm, api, n=161, cost=0.02)
        assert len(out) == 161
        steps, seconds, _depth = llm._note_decode_rate.call_args.args
        assert steps == 40
        assert seconds == pytest.approx(40 * (0.02 + 0.005 + 0.005))


class TestExpectation:
    def _note(self, llm, rate, depth, steps=100):
        llm._note_decode_rate(steps, steps / rate, depth)

    def test_a_deeper_reply_slower_by_depth_alone_is_not_slow(self):
        llm = _llm(offload_kqv=False)
        self._note(llm, 16.0, 300)
        self._note(llm, 5.65, 6000)
        assert llm._decode_refresh is None

    def test_a_slow_reply_at_the_same_depth_is_slow(self):
        llm = _llm()
        self._note(llm, 42.6, 3000)
        self._note(llm, 11.9, 3100)
        assert llm._decode_refresh == (pytest.approx(11.9), pytest.approx(42.6 * 5048 / 5148))

    def test_a_shallower_reply_is_held_to_the_deeper_rate(self):
        llm = _llm()
        self._note(llm, 37.9, 6000)
        self._note(llm, 12.0, 300)
        assert llm._decode_refresh is not None

    def test_placements_are_judged_apart(self):
        llm = _llm()
        self._note(llm, 43.0, 300)
        llm._offload_kqv = False
        self._note(llm, 16.0, 300)
        assert llm._decode_refresh is None

    def test_fewer_than_the_minimum_steps_are_ignored(self):
        llm = _llm()
        self._note(llm, 43.0, 300)
        self._note(llm, 4.0, 300, steps=31)
        assert llm._decode_refresh is None
        assert len(llm._decode_samples) == 1

    def test_no_live_context_records_nothing(self):
        llm = _llm()
        llm._ctx_ptr = None
        self._note(llm, 43.0, 300)
        assert llm._decode_samples is None

    def test_the_first_reply_sets_the_rate_and_is_never_slow(self):
        llm = _llm()
        self._note(llm, 1.0, 300)
        assert llm._decode_refresh is None


class TestBackoff:
    def _slow_then_refresh(self, llm):
        llm._note_decode_rate(100, 100 / 40.0, 300)
        llm._note_decode_rate(100, 100 / 4.0, 300)
        assert llm._decode_refresh is not None
        assert llm._rightsize_plan(100) == (4096, True)
        llm._consume_decode_refresh()

    def test_a_refresh_waits_out_the_backoff_and_the_backoff_doubles(self):
        llm = _llm()
        self._slow_then_refresh(llm)
        assert llm._decode_refresh_after > time.monotonic() + 50
        assert llm._decode_refresh_backoff == 60.0
        llm._note_decode_rate(100, 100 / 4.0, 300)
        assert llm._decode_refresh is None
        llm._decode_refresh_after = 0.0
        llm._note_decode_rate(100, 100 / 4.0, 300)
        assert llm._decode_refresh is not None
        llm._consume_decode_refresh()
        assert llm._decode_refresh_backoff == 120.0
        assert llm._decode_refresh_after > time.monotonic() + 110

    def test_the_backoff_stops_at_its_maximum(self):
        llm = _llm()
        for _ in range(8):
            llm._decode_refresh = (1.0, 10.0)
            llm._consume_decode_refresh()
        assert llm._decode_refresh_backoff == llm._DECODE_REFRESH_BACKOFF_MAX_S

    def test_a_reply_at_the_expected_rate_keeps_the_backoff(self):
        llm = _llm()
        self._slow_then_refresh(llm)
        llm._decode_refresh = (4.0, 40.0)
        llm._consume_decode_refresh()
        llm._note_decode_rate(100, 100 / 39.0, 300)
        llm._decode_refresh = (4.0, 40.0)
        llm._consume_decode_refresh()
        assert llm._decode_refresh_backoff == 240.0

    def test_a_refresh_long_after_the_last_one_starts_at_the_base_backoff(self):
        llm = _llm()
        self._slow_then_refresh(llm)
        llm._decode_refresh_backoff = 960.0
        llm._decode_refresh_at = time.monotonic() - 2000
        llm._decode_refresh = (4.0, 40.0)
        llm._consume_decode_refresh()
        assert llm._decode_refresh_backoff == 60.0

    def test_any_recreated_context_clears_a_pending_refresh(self):
        llm = _llm()
        api, _cp = _api(llm)
        llm._decode_refresh = (4.0, 40.0)
        with patch(API, api):
            llm._prefill_fresh_context([1, 2, 3], needed=5000)
        assert llm._decode_refresh is None
        llm._decode_refresh = (4.0, 40.0)
        with patch(API, api):
            llm._recreate_slots_context(4096, True)
        assert llm._decode_refresh is None


class TestPlan:
    def test_a_shrink_wins_over_a_refresh(self):
        llm = _llm(capacity=32768)
        llm._decode_refresh = (4.0, 40.0)
        assert llm._rightsize_plan(1000) == (4096, True)

    def test_a_promotion_wins_over_a_refresh(self):
        llm = _llm(offload_kqv=False)
        llm._vram_check = lambda t, c: True
        llm._decode_refresh = (4.0, 40.0)
        assert llm._rightsize_plan(1000) == (4096, True)

    def test_a_refused_promotion_still_refreshes_in_ram(self):
        llm = _llm(offload_kqv=False)
        llm._vram_check = lambda t, c: False
        llm._decode_refresh = (4.0, 40.0)
        assert llm._rightsize_plan(1000) == (4096, False)

    def test_nothing_due_keeps_the_context(self):
        llm = _llm()
        assert llm._rightsize_plan(1000) is None

    def test_the_slot_seam_passes_the_rate_through(self):
        from localm.inference.backends.llamacpp._slots import LlamaSlotOps
        llm = _llm()
        llm._note_decode_rate = MagicMock()
        LlamaSlotOps(llm).note_decode(40, 1.0, 12.5)
        llm._note_decode_rate.assert_called_once_with(40, 1.0, 12.5)
