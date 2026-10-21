# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live context is recreated smaller after a conversation shrinks, and a
KV cache kept in system RAM moves back to VRAM once it fits."""

from unittest.mock import MagicMock, patch

import pytest

from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init

API = "localm.inference.backends.llamacpp.llama.api"


def _llm(capacity, *, offload_kqv=True, vram_check=None, n_ctx=4096, cached=None):
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _n_ctx=n_ctx,
                          _n_ctx_max=65536, _n_ctx_grow=4096,
                          _ctx_capacity=capacity, _offload_kqv=offload_kqv,
                          _kv_supported=True)
    llm._vram_check = vram_check
    llm._cached_tokens = list(cached if cached is not None else range(1, 50))
    llm._tokenizer.is_eog.return_value = False
    return llm


def _api(init=444):
    api = MagicMock()
    cp = MagicMock()
    api.llama_context_default_params.return_value = cp
    if isinstance(init, list):
        api.llama_init_from_model.side_effect = init
    else:
        api.llama_init_from_model.return_value = init
    api.llama_decode.return_value = 0
    api.llama_batch_init.side_effect = fake_batch_init
    api.llama_sampler_sample.return_value = 42
    api.llama_memory_seq_rm.return_value = True
    return api, cp


def _generate(llm, api, prompt=(1, 2, 3), max_new_tokens=2):
    with patch(API, api), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999):
        return list(llm._generate(prompt_tokens=list(prompt), max_new_tokens=max_new_tokens,
                                  temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0))


class TestShrink:
    def test_a_conversation_that_shrank_gets_a_smaller_context(self):
        calls = []
        llm = _llm(32768, vram_check=lambda t, c: calls.append((t, c)) or True)
        api, cp = _api()
        assert _generate(llm, api) == [42, 42]
        assert cp.n_ctx == 4096
        assert llm._ctx_capacity == 4096
        assert cp.offload_kqv is True
        assert calls == [(4096, 32768)]
        api.llama_free.assert_any_call(222)

    def test_no_shrink_when_the_need_is_more_than_half_the_context(self):
        calls = []
        llm = _llm(12288, vram_check=lambda t, c: calls.append((t, c)) or True)
        api, _cp = _api()
        prompt = list(range(1, 6000))
        _generate(llm, api, prompt=prompt)
        assert llm._ctx_capacity == 12288
        api.llama_init_from_model.assert_not_called()
        assert calls == []

    def test_shrink_at_exactly_half(self):
        llm = _llm(8192, vram_check=lambda t, c: True)
        api, cp = _api()
        _generate(llm, api)
        assert cp.n_ctx == 4096

    def test_never_below_the_base_context(self):
        llm = _llm(8192, n_ctx=8192, vram_check=lambda t, c: True)
        api, _cp = _api()
        _generate(llm, api)
        assert llm._ctx_capacity == 8192
        api.llama_init_from_model.assert_not_called()

    def test_a_shrunk_context_keeps_the_kv_cache_in_ram_when_vram_says_no(self):
        llm = _llm(32768, offload_kqv=False, vram_check=lambda t, c: False)
        api, cp = _api()
        _generate(llm, api)
        assert cp.n_ctx == 4096
        assert cp.offload_kqv is False
        assert llm._offload_kqv is False

    def test_shrink_without_a_vram_check(self):
        llm = _llm(32768)
        api, cp = _api()
        _generate(llm, api)
        assert cp.n_ctx == 4096


class TestPromote:
    def test_a_kv_cache_in_ram_moves_to_vram_once_it_fits(self):
        calls = []
        llm = _llm(8192, offload_kqv=False,
                   vram_check=lambda t, c: calls.append((t, c)) or True)
        api, cp = _api()
        prompt = list(range(1, 5000))
        _generate(llm, api, prompt=prompt)
        assert calls == [(8192, 8192)]
        assert cp.n_ctx == 8192
        assert cp.offload_kqv is True
        assert llm._offload_kqv is True
        assert llm._cached_tokens[:len(prompt)] == prompt

    def test_a_refused_promotion_is_not_asked_again_right_away(self):
        calls = []
        llm = _llm(8192, offload_kqv=False,
                   vram_check=lambda t, c: calls.append((t, c)) or False)
        api, _cp = _api()
        prompt = list(range(1, 5000))
        _generate(llm, api, prompt=prompt)
        _generate(llm, api, prompt=prompt)
        assert calls == [(8192, 8192)]
        api.llama_init_from_model.assert_not_called()
        assert llm._offload_kqv is False
        llm._kv_promote_after = 0.0
        _generate(llm, api, prompt=prompt)
        assert len(calls) == 2

    def test_a_kv_cache_in_vram_is_not_probed(self):
        calls = []
        llm = _llm(8192, vram_check=lambda t, c: calls.append((t, c)) or True)
        api, _cp = _api()
        _generate(llm, api, prompt=list(range(1, 5000)))
        assert calls == []
        api.llama_init_from_model.assert_not_called()

    def test_a_failed_vram_context_falls_back_to_system_ram(self):
        llm = _llm(8192, offload_kqv=False, vram_check=lambda t, c: True)
        api, cp = _api(init=[0, 555])
        out = _generate(llm, api, prompt=list(range(1, 5000)))
        assert out == [42, 42]
        assert llm._ctx_ptr == 555
        assert llm._offload_kqv is False
        assert cp.offload_kqv is False
        assert llm._kv_promote_after > 0

    def test_a_raising_vram_check_keeps_the_live_context(self):
        def boom(t, c):
            raise RuntimeError("probe failed")
        llm = _llm(8192, offload_kqv=False, vram_check=boom)
        api, _cp = _api()
        out = _generate(llm, api, prompt=list(range(1, 5000)))
        assert out == [42, 42]
        assert llm._ctx_ptr == 222
        api.llama_init_from_model.assert_not_called()


class TestPlan:
    @pytest.mark.parametrize("capacity,needed,others,expected", [
        (32768, 1000, 0, (4096, True)),
        (32768, 1000, 9000, (12288, True)),
        (32768, 1000, 20000, None),
        (8192, 9000, 0, None),
    ])
    def test_plan(self, capacity, needed, others, expected):
        llm = _llm(capacity, vram_check=lambda t, c: True)
        assert llm._rightsize_plan(needed, others) == expected

    def test_no_live_context_no_plan(self):
        llm = _llm(32768, vram_check=lambda t, c: True)
        llm._ctx_ptr = None
        assert llm._rightsize_plan(100) is None


class TestThreadsOnARecreatedContext:
    def test_a_grown_context_keeps_the_configured_thread_count(self):
        llm = _llm(4096, vram_check=lambda t, c: True)
        llm._n_threads = 6
        api, cp = _api()
        with patch(API, api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_ctx == 8192
        assert cp.n_threads == 6 and cp.n_threads_batch == 6

    def test_no_thread_count_configured_leaves_the_runtime_default(self):
        llm = _llm(4096, vram_check=lambda t, c: True)
        api, cp = _api()
        cp.n_threads = 4
        cp.n_threads_batch = 4
        with patch(API, api):
            llm._prefill_fresh_context(list(range(100)), needed=5000)
        assert cp.n_threads == 4 and cp.n_threads_batch == 4


class TestRightsizeFailureKeepsSpeculation:
    def test_a_failed_vram_context_keeps_the_mtp_draft_context(self):
        llm = _llm(8192, offload_kqv=False, vram_check=lambda t, c: True)
        llm._mtp_ctx_ptr = 333
        llm._mtp_usable = True
        llm._free_backend_draft_sampler = MagicMock()
        llm._rebuild_mtp_context = MagicMock()
        llm._disable_mtp = MagicMock()
        api, _cp = _api(init=[0, 555])
        assert _generate(llm, api, prompt=list(range(1, 5000))) == [42, 42]
        llm._disable_mtp.assert_not_called()
        llm._rebuild_mtp_context.assert_called_once_with(8192, False)
        assert llm._ctx_ptr == 555 and llm._offload_kqv is False

    def test_a_promotion_is_not_asked_again_right_after_one_was_tried(self):
        calls = []
        llm = _llm(8192, offload_kqv=False,
                   vram_check=lambda t, c: calls.append((t, c)) or True)
        assert llm._rightsize_plan(5000) == (8192, True)
        assert llm._rightsize_plan(5000) is None
        assert calls == [(8192, 8192)]


class TestDraftContextFollowsTheTarget:
    def _source(self, llm, capacity, offload):
        import ctypes

        from localm.inference.backends.llamacpp._draftmodel import DraftModelSource
        src = DraftModelSource(llm, ctypes.c_void_p(5), 4)
        src._ctx = ctypes.c_void_p(6)
        src._ctx_capacity = capacity
        src._ctx_offload = offload
        return src

    def test_a_shrunk_target_context_puts_the_draft_context_out_of_step(self):
        llm = _llm(4096)
        assert self._source(llm, 8192, True)._context_out_of_step() is True

    def test_a_target_kv_cache_moved_to_vram_puts_it_out_of_step(self):
        llm = _llm(8192, offload_kqv=True)
        assert self._source(llm, 8192, False)._context_out_of_step() is True

    def test_a_matching_draft_context_is_kept(self):
        llm = _llm(8192, offload_kqv=True)
        assert self._source(llm, 8192, True)._context_out_of_step() is False
        assert self._source(llm, 8192, None)._context_out_of_step() is False


def _vram_upto(limit, calls):
    def check(target, current, hint=True):
        calls.append((target, current, hint))
        return target <= limit
    return check


def _capacity_api(llm):
    """A fake api whose decode fails once a position reaches the live context's
    capacity, and that records (n_ctx, offload_kqv) of every created context."""
    from tests._fake_batch import batch_backing
    api, cp = _api()
    created = []

    def init(_model, params):
        created.append((params.n_ctx, params.offload_kqv))
        return 444 + len(created)

    def decode(_ctx, batch):
        last = batch_backing(batch)["pos"][batch.n_tokens - 1]
        return 1 if last >= llm._ctx_capacity else 0

    api.llama_init_from_model.side_effect = init
    api.llama_decode.side_effect = decode
    return api, cp, created


class TestStepReserve:
    PROMPT = list(range(1, 301))

    def test_a_budget_that_would_push_the_kv_cache_to_ram_starts_with_one_step_in_vram(self):
        calls = []
        llm = _llm(4096, vram_check=_vram_upto(4096, calls), cached=[])
        api, _cp, created = _capacity_api(llm)
        out = _generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)
        assert len(out) == 4096
        assert llm.last_finish_reason == "length"
        assert calls[:2] == [(4096, 4096, False), (8192, 4096, False)]
        assert created == [(8192, False)]
        assert calls[2:] == [(8192, 4096, True)]
        assert llm._ctx_capacity == 8192 and llm._offload_kqv is False

    def test_a_budget_that_fits_vram_is_reserved_whole(self):
        calls = []
        llm = _llm(4096, vram_check=_vram_upto(8192, calls), cached=[])
        api, _cp, created = _capacity_api(llm)
        out = _generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)
        assert len(out) == 4096
        assert created == [(8192, True)]
        assert calls == [(4096, 4096, False), (8192, 4096, False), (8192, 4096, True)]

    def test_a_vram_context_that_holds_the_budget_is_not_probed(self):
        calls = []
        llm = _llm(8192, vram_check=_vram_upto(4096, calls), cached=[])
        api, _cp, created = _capacity_api(llm)
        assert len(_generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)) == 4096
        assert calls == [] and created == []

    def test_a_small_budget_is_reserved_whole_without_a_probe(self):
        calls = []
        llm = _llm(4096, vram_check=_vram_upto(4096, calls), cached=[])
        api, _cp, created = _capacity_api(llm)
        assert len(_generate(llm, api, prompt=self.PROMPT, max_new_tokens=400)) == 400
        assert calls == [] and created == []

    def test_a_ram_context_from_an_earlier_reply_gives_way_to_a_step_in_vram(self):
        calls = []
        llm = _llm(12288, offload_kqv=False, vram_check=_vram_upto(4096, calls), cached=[])
        api, _cp, created = _capacity_api(llm)
        out = _generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)
        assert len(out) == 4096
        assert created == [(4096, True), (8192, False)]

    def test_a_raising_check_reserves_the_whole_budget(self):
        def boom(target, current, hint=True):
            if not hint:
                raise RuntimeError("probe failed")
            return False
        llm = _llm(4096, vram_check=boom, cached=[])
        api, _cp, created = _capacity_api(llm)
        assert len(_generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)) == 4096
        assert created == [(8192, False)]

    @pytest.mark.parametrize("capacity,offload,throttled,expected", [
        (4096, True, False, 876),
        (8192, True, False, 4460),
        (12288, False, False, 876),
        (12288, False, True, 4460),
    ])
    def test_step_reserve(self, capacity, offload, throttled, expected):
        llm = _llm(capacity, offload_kqv=offload, vram_check=_vram_upto(4096, []))
        if throttled:
            llm._kv_promote_after = float("inf")
        assert llm._step_reserve(4460, 876) == expected

    def test_no_live_context_counts_as_a_growth(self):
        llm = _llm(4096, vram_check=_vram_upto(4096, []))
        llm._ctx_ptr = None
        assert llm._step_reserve(4460, 876) == 876


class TestStepReserveOsBudget:
    """_step_reserve also reserves one step when the whole budget would keep the
    KV cache in VRAM but past the OS video memory budget (``_budget_check``)."""

    PROMPT = list(range(1, 301))

    def _llm(self, budget_limit, calls, capacity=4096):
        llm = _llm(capacity, vram_check=_vram_upto(65536, []), cached=[])

        def budget(target, current):
            calls.append((target, current))
            return None if budget_limit is None else target <= budget_limit
        llm._budget_check = budget
        return llm

    def test_a_budget_past_the_os_budget_reserves_one_step(self):
        calls = []
        assert self._llm(4096, calls)._step_reserve(4460, 876) == 876
        assert calls == [(8192, 4096), (4096, 4096)]

    def test_a_budget_within_the_os_budget_is_reserved_whole(self):
        calls = []
        assert self._llm(8192, calls)._step_reserve(4460, 876) == 4460
        assert calls == [(8192, 4096)]

    def test_a_step_past_the_os_budget_reserves_the_whole_budget(self):
        calls = []
        assert self._llm(0, calls, capacity=0)._step_reserve(4460, 876) == 4460
        assert calls == [(8192, 0), (4096, 0)]

    def test_a_ram_placement_reserves_one_step_even_past_the_os_budget(self):
        calls = []
        llm = _llm(4096, vram_check=_vram_upto(4096, []), cached=[])
        llm._budget_check = lambda target, current: calls.append((target, current)) or False
        assert llm._step_reserve(4460, 876) == 876
        assert calls == []

    def test_an_unknown_budget_reserves_the_whole_budget(self):
        calls = []
        assert self._llm(None, calls)._step_reserve(4460, 876) == 4460

    def test_a_raising_budget_check_reserves_the_whole_budget(self):
        llm = self._llm(4096, [])

        def boom(target, current):
            raise RuntimeError("dxgi failed")
        llm._budget_check = boom
        assert llm._step_reserve(4460, 876) == 4460

    def test_a_ram_placement_reserves_one_step_without_a_budget_check(self):
        llm = _llm(4096, vram_check=_vram_upto(4096, []), cached=[])
        assert llm._step_reserve(4460, 876) == 876

    def test_the_reply_runs_in_the_live_context_until_it_fills(self):
        calls = []
        llm = self._llm(4096, calls)
        api, _cp, created = _capacity_api(llm)
        out = _generate(llm, api, prompt=self.PROMPT, max_new_tokens=4096)
        assert len(out) == 4096
        assert calls[:2] == [(8192, 4096), (4096, 4096)]
        assert created == [(8192, True)]
        assert llm._ctx_capacity == 8192 and llm._offload_kqv is True


class TestPromoteToASmallerContext:
    def test_a_refused_same_size_promotion_moves_to_vram_at_the_needed_size(self):
        calls = []
        llm = _llm(12288, offload_kqv=False, vram_check=_vram_upto(8192, calls))
        assert llm._rightsize_plan(5000) == (8192, True)
        assert calls == [(12288, 12288, False), (8192, 12288, False)]

    def test_nothing_smaller_fits_either(self):
        calls = []
        llm = _llm(12288, offload_kqv=False, vram_check=_vram_upto(4096, calls))
        assert llm._rightsize_plan(5000) is None
        assert len(calls) == 2


class TestStepGrowthFallsBackToRam:
    def test_a_step_growth_vram_cannot_create_continues_in_system_ram(self):
        calls = []
        llm = _llm(4096, vram_check=_vram_upto(8192, calls), cached=[])
        api, _cp, created = _capacity_api(llm)

        def init(_model, params):
            created.append((params.n_ctx, params.offload_kqv))
            return 0 if (params.n_ctx, params.offload_kqv) == (8192, True) else 444 + len(created)
        api.llama_init_from_model.side_effect = init
        out = _generate(llm, api, prompt=list(range(1, 301)), max_new_tokens=8192)
        assert len(out) == 8192
        assert llm.last_finish_reason == "length"
        assert created == [(8192, True), (8192, False), (12288, False)]
