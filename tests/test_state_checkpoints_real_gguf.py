# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL prompt prefix reuse on a hybrid (recurrent + attention) model: a
follow-up turn resumes from the state checkpoint taken at the end of the
previous prompt, and a context that grows keeps its cache, instead of
prefilling the whole conversation again.

The reused reply must equal the reply decoded from a cleared cache in the same
batches (up to the checkpoint, then the rest): a restored state is exact. A
cleared-cache decode split at other positions can round differently and pick
another greedy token, so it is not the reference.

@integration: needs the native runtime (localm setup-llama) and the model on
disk or reachable; their absence is a skip, any later failure is real.
"""
from __future__ import annotations

import pytest

from tests._real_gguf import fetch_gguf, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "unsloth/Qwen3.5-0.8B-MTP-GGUF"
_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"

_SYSTEM = ("You are a terse assistant. " + "Answer in one short sentence. " * 20).strip()


@pytest.fixture(scope="module")
def model_path():
    require_native_runtime()
    return fetch_gguf(_REPO, _FILE)


def _load(path, **kw):
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    return LlamaCpp(path, n_gpu_layers=0, verbose=False, **kw)


def _reply(llm, messages) -> str:
    out = llm.create_chat_completion(messages=messages, max_tokens=24, temperature=0.0,
                                     seed=1, thinking=False)
    return out["choices"][0]["message"]["content"]


def _uncached_reply_split_at(llm, messages, cut: int) -> str:
    """The reply from a cleared cache, the prompt decoded in batches ending at
    *cut*, at CHECKPOINT_TAIL tokens before the end, and at the end."""
    from localm.inference.backends.llamacpp import llama as L

    def decode(prompt_tokens, start, step):
        n = len(prompt_tokens)
        bounds = [start, cut, n - L.LlamaCpp.CHECKPOINT_TAIL, n]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            chunk = prompt_tokens[a:b]
            batch = llm._create_batch(chunk, a, logits_at_last_only=True)
            ret = L.api.llama_decode(llm._ctx_ptr, batch)
            L.api.llama_batch_free(batch)
            if ret:
                return ret
            llm._after_prefill_chunk(chunk, a)
        return 0

    llm._decode_prompt = decode
    llm._cached_tokens = []
    try:
        return _reply(llm, messages)
    finally:
        del llm._decode_prompt


def _conversation(llm):
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "Name a colour."}]
    first = _reply(llm, messages)
    cut = len(llm._checkpoints[-1].tokens)
    messages += [{"role": "assistant", "content": first},
                 {"role": "user", "content": "Name a fruit of that colour."}]
    return messages, cut


def test_a_follow_up_turn_resumes_from_the_checkpoint(model_path):
    llm = _load(model_path, n_ctx=2048)
    try:
        assert llm._state_checkpoints_wanted() is True
        messages, cut = _conversation(llm)
        second = _reply(llm, messages)
        assert llm.prompt_reused == cut
        assert second == _uncached_reply_split_at(llm, messages, cut)
        assert llm.prompt_reused == 0
    finally:
        llm.close()


def test_a_grown_context_keeps_the_conversation_so_far(model_path):
    llm = _load(model_path, n_ctx=256, n_ctx_grow=256)
    try:
        messages, cut = _conversation(llm)
        capacity = llm._ctx_capacity
        messages[-1]["content"] += " " + "Think of the most common one. " * 40
        second = _reply(llm, messages)
        assert llm._ctx_capacity > capacity
        assert llm.prompt_reused == cut
        assert second == _uncached_reply_split_at(llm, messages, cut)
    finally:
        llm.close()
