# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL prompt prefix reuse on a hybrid (recurrent + attention) model: a
follow-up turn resumes from the state checkpoint taken at the end of the
previous prompt, and a context that grows keeps its cache, instead of
prefilling the whole conversation again. Each reused reply must match the reply
decoded from a cleared cache.

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
    out = llm.create_chat_completion(messages=messages, max_tokens=12, temperature=0.0,
                                     seed=1, thinking=False)
    return out["choices"][0]["message"]["content"]


def _uncached_reply(llm, messages) -> str:
    llm._cached_tokens = []
    return _reply(llm, messages)


def _conversation(llm):
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "Name a colour."}]
    first = _reply(llm, messages)
    first_prompt = llm.prompt_reused, len(llm._cached_tokens)
    messages += [{"role": "assistant", "content": first},
                 {"role": "user", "content": "Name a fruit of that colour."}]
    return messages, first_prompt


def test_a_follow_up_turn_reuses_the_conversation_so_far(model_path):
    llm = _load(model_path, n_ctx=2048)
    try:
        assert llm._state_checkpoints_wanted() is True
        messages, _ = _conversation(llm)
        cached_before = len(llm._cached_tokens)
        second = _reply(llm, messages)
        reused = llm.prompt_reused
        assert reused > cached_before // 2, (reused, cached_before)
        assert second == _uncached_reply(llm, messages)
        assert llm.prompt_reused == 0
    finally:
        llm.close()


def test_a_grown_context_keeps_the_conversation_so_far(model_path):
    llm = _load(model_path, n_ctx=256, n_ctx_grow=256)
    try:
        messages, _ = _conversation(llm)
        capacity = llm._ctx_capacity
        messages[-1]["content"] += " " + "Think of the most common one. " * 40
        second = _reply(llm, messages)
        assert llm._ctx_capacity > capacity
        assert llm.prompt_reused > 0
        assert second == _uncached_reply(llm, messages)
    finally:
        llm.close()
