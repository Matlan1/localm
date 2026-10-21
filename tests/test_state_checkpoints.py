# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt prefix reuse on a model whose cache cannot drop a range of positions
(recurrent / hybrid layers), and the KV cache carried into a grown context.

The native library is replaced by a stateful fake with llama.cpp's recurrent
memory rules: ``llama_memory_seq_rm`` refuses a removal that starts inside the
recurrent state, and the PARTIAL_ONLY state save/restore round-trips only the
recurrent part. Decodes must arrive at the next free position, so a prefill
that skipped or repeated a token fails here.
"""

from __future__ import annotations

import ctypes
import struct
from unittest.mock import MagicMock, patch


from localm.inference.backends.llamacpp._api import LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._bare_llama import make_bare_llama

_MEM = 333


class FakeRecurrentNative:
    """Sequence 0 of a hybrid model: an attention cache holding positions
    0..n_attn-1 and a recurrent state that has consumed positions 0..rec_pos."""

    def __init__(self, recurrent: bool = True) -> None:
        self.recurrent = recurrent
        self.n_attn = 0
        self.rec_pos = -1
        self.decoded: list[list[int]] = []
        self.fail_set = False

    def api(self) -> MagicMock:
        api = MagicMock()
        api.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
        api.llama_get_memory.return_value = _MEM
        api.has_state_seq_ext_api.return_value = True
        api.llama_decode.side_effect = self.decode
        api.llama_memory_seq_rm.side_effect = self.seq_rm
        api.llama_memory_clear.side_effect = lambda mem, data=True: self.clear()
        api.llama_state_seq_get_size_ext.side_effect = self.get_size
        api.llama_state_seq_get_data_ext.side_effect = self.get_data
        api.llama_state_seq_set_data_ext.side_effect = self.set_data
        api.llama_context_default_params.return_value = MagicMock()
        api.llama_init_from_model.return_value = 444
        return api

    def decode(self, ctx, batch) -> int:
        tokens, pos0 = batch
        assert pos0 == self.n_attn, f"decode at {pos0}, cache ends at {self.n_attn}"
        assert pos0 == self.rec_pos + 1, f"decode at {pos0}, state at {self.rec_pos}"
        self.decoded.append(list(tokens))
        self.n_attn += len(tokens)
        self.rec_pos = self.n_attn - 1
        return 0

    def clear(self) -> None:
        self.n_attn = 0
        self.rec_pos = -1

    def seq_rm(self, mem, seq, p0, p1) -> bool:
        if p0 <= 0:
            self.clear()
            return True
        if self.recurrent and p0 <= self.rec_pos:
            return False
        self.n_attn = min(self.n_attn, p0)
        self.rec_pos = min(self.rec_pos, self.n_attn - 1)
        return True

    def get_size(self, ctx, seq, flags) -> int:
        return 8 if flags == LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY else 16

    def get_data(self, ctx, buf, size, seq, flags) -> int:
        data = struct.pack("<qq", self.rec_pos, self.n_attn)[:size]
        ctypes.memmove(buf, data, size)
        return size

    def set_data(self, ctx, buf, size, seq, flags) -> int:
        if self.fail_set:
            return 0
        raw = bytes(buf)[:size]
        if flags == LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY:
            (self.rec_pos,) = struct.unpack("<q", raw)
        else:
            self.rec_pos, self.n_attn = struct.unpack("<qq", raw)
        return size

    def generate(self, llm: LlamaCpp, tokens: list[int]) -> None:
        """What a reply leaves behind: its tokens decoded after the prompt."""
        self.decode(None, (tokens, self.n_attn))
        llm._cached_tokens = llm._cached_tokens + tokens


def _llm(native: FakeRecurrentNative, wanted: bool = True) -> LlamaCpp:
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
    llm._mtp_ctx_ptr = None
    llm._kv_supported = True
    llm._model_has_recurrent_layers = lambda: wanted
    llm._create_batch = lambda chunk, pos, logits_at_last_only=True: (list(chunk), pos)
    return llm


def _prefilled(native: FakeRecurrentNative) -> int:
    return sum(len(c) for c in native.decoded)


class TestRecurrentPrefixReuse:
    def test_a_follow_up_resumes_from_the_checkpoint_before_the_prompt_end(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        first = list(range(100))
        with patch("localm.inference.backends.llamacpp.llama.api", native.api()):
            llm._prefill_with_reuse(first)
            assert [len(c) for c in native.decoded] == [96, 4]
            native.generate(llm, [500, 501, 502])
            native.decoded.clear()

            # The next prompt re-renders the end of the last one differently.
            second = first[:99] + list(range(700, 730))
            llm._prefill_with_reuse(second)

        assert llm.prompt_reused == 100 - LlamaCpp.CHECKPOINT_TAIL
        assert _prefilled(native) == len(second) - 96
        assert native.n_attn == len(second) and native.rec_pos == len(second) - 1
        assert llm._cached_tokens == second

    def test_a_prompt_diverging_before_every_checkpoint_is_prefilled_whole(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        first = list(range(100))
        with patch("localm.inference.backends.llamacpp.llama.api", native.api()):
            llm._prefill_with_reuse(first)
            native.generate(llm, [500])
            native.decoded.clear()
            second = first[:50] + [900, 901]
            llm._prefill_with_reuse(second)

        assert llm.prompt_reused == 0
        assert _prefilled(native) == len(second)
        assert native.n_attn == len(second)

    def test_a_failed_restore_prefills_the_whole_prompt(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        first = list(range(100))
        with patch("localm.inference.backends.llamacpp.llama.api", native.api()):
            llm._prefill_with_reuse(first)
            native.generate(llm, [500])
            native.decoded.clear()
            native.fail_set = True
            second = first[:99] + [900]
            llm._prefill_with_reuse(second)

        assert llm.prompt_reused == 0
        assert _prefilled(native) == len(second)
        assert len(llm._checkpoints) == 1
        assert len(llm._checkpoints[0].tokens) == len(second) - LlamaCpp.CHECKPOINT_TAIL

    def test_checkpoints_past_the_shared_prefix_are_dropped_and_capped(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        prompt = list(range(40))
        with patch("localm.inference.backends.llamacpp.llama.api", native.api()):
            for turn in range(5):
                llm._prefill_with_reuse(prompt)
                native.generate(llm, [1000 + turn])
                prompt = prompt + [1000 + turn, 2000 + turn] + list(range(3000, 3010))
        lengths = [len(c.tokens) for c in llm._checkpoints]
        assert len(lengths) == LlamaCpp.CHECKPOINTS_MAX
        assert lengths == sorted(lengths)

    def test_a_model_without_recurrent_layers_takes_no_checkpoint(self):
        native = FakeRecurrentNative(recurrent=False)
        llm = _llm(native, wanted=False)
        api = native.api()
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_with_reuse(list(range(100)))
        assert [len(c) for c in native.decoded] == [100]
        api.llama_state_seq_get_size_ext.assert_not_called()
        assert list(llm._checkpoints) == []

    def test_a_runtime_without_the_state_api_takes_no_checkpoint(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        api = native.api()
        api.has_state_seq_ext_api.return_value = False
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_with_reuse(list(range(100)))
        assert [len(c) for c in native.decoded] == [100]
        api.llama_state_seq_get_size_ext.assert_not_called()


class TestGrowthCarriesTheCache:
    def _grow(self, native, llm, api, second):
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_fresh_context(second, needed=len(second) + 600)

    def test_a_grown_context_keeps_the_shared_prefix(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        llm._ctx_capacity = 256
        api = native.api()
        first = list(range(200))
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_with_reuse(first)
            native.generate(llm, [500, 501])
        native.decoded.clear()
        state_before = (native.rec_pos, native.n_attn)
        second = first[:199] + list(range(700, 760))

        # The new context starts empty: the carried copy is what fills it.
        api.llama_free.side_effect = lambda ctx: native.clear()
        with patch("localm.sysstats.system_ram", return_value=(1 << 34, 1 << 33)):
            self._grow(native, llm, api, second)

        assert state_before == (201, 202)
        assert llm._ctx_ptr == 444
        assert llm.prompt_reused == 200 - LlamaCpp.CHECKPOINT_TAIL
        assert _prefilled(native) == len(second) - 196
        assert native.n_attn == len(second)
        assert llm._cached_tokens == second

    def test_without_the_ram_for_a_copy_the_prompt_is_prefilled_whole(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        api = native.api()
        first = list(range(200))
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_with_reuse(first)
            native.generate(llm, [500])
        native.decoded.clear()
        api.llama_free.side_effect = lambda ctx: native.clear()
        second = first + [500, 900]
        with patch("localm.sysstats.system_ram", return_value=(1 << 34, 16)):
            self._grow(native, llm, api, second)

        assert llm.prompt_reused == 0
        assert _prefilled(native) == len(second)
        api.llama_state_seq_set_data_ext.assert_not_called()

    def test_a_carried_copy_that_does_not_load_is_not_trusted(self):
        native = FakeRecurrentNative()
        llm = _llm(native)
        api = native.api()
        first = list(range(200))
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_with_reuse(first)
            native.generate(llm, [500])
        native.decoded.clear()
        api.llama_free.side_effect = lambda ctx: native.clear()
        native.fail_set = True
        second = first + [500, 900]
        with patch("localm.sysstats.system_ram", return_value=(1 << 34, 1 << 33)):
            self._grow(native, llm, api, second)

        assert llm.prompt_reused == 0
        assert _prefilled(native) == len(second)
        assert native.n_attn == len(second)
