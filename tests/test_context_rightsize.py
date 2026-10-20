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
