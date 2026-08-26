# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real coverage for LlamaCpp._create_batch's native fill loop.

The fill loop writes token ids, per-token positions, seq ids and logits flags
into native memory through ctypes. These assert the real layout the native
decoder sees, against a real ctypes-backed batch.
"""

from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import api
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init, batch_backing
from tests.test_kv_cache import _bare_llama


def _bare():
    return make_bare_llama()


def test_fill_positions_tokens_logits_seqids():
    llm = _bare()
    with patch.object(api, "llama_batch_init", side_effect=fake_batch_init):
        batch = llm._create_batch([10, 20, 30], start_pos=5, logits_at_last_only=True)
    b = batch_backing(batch)
    assert b["n_tokens"] == 3
    assert b["token"] == [10, 20, 30]
    assert b["pos"] == [5, 6, 7]          # start_pos + idx (the off-by-one guard)
    assert b["n_seq_id"] == [1, 1, 1]
    assert b["seq_id0"] == [0, 0, 0]
    assert b["logits"] == [0, 0, 1]       # only the last token requests logits


def test_fill_logits_on_every_token_when_not_last_only():
    llm = _bare()
    with patch.object(api, "llama_batch_init", side_effect=fake_batch_init):
        batch = llm._create_batch([1, 2, 3, 4], start_pos=0, logits_at_last_only=False)
    assert batch_backing(batch)["logits"] == [1, 1, 1, 1]


def test_batch_freed_when_mid_generation_growth_raises():
    """llama_batch_init allocates native arrays; when the mid-generation
    context-growth re-prefill raises, the batch must still be freed."""
    llm = _bare_llama()
    llm._cached_tokens = [1, 2, 3]
    llm._ctx_capacity = 10

    mock_api = MagicMock()
    mock_api.llama_sampler_sample.return_value = 42
    mock_api.llama_decode.return_value = -1         # per-token decode fails -> growth
    mock_api.llama_batch_init.side_effect = fake_batch_init
    llm._tokenizer.is_eog.return_value = False

    with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=999), \
         patch.object(llm, "_fit_generation_budget", return_value=4), \
         patch.object(llm, "_can_reuse_kv", return_value=True), \
         patch.object(llm, "_prefill_with_reuse", return_value=None), \
         patch.object(llm, "_target_ctx", return_value=99999), \
         patch.object(llm, "_prefill_fresh_context", side_effect=RuntimeError("grow boom")):
        gen = llm._generate([1, 2, 3], max_new_tokens=4, temperature=0.8,
                            top_k=40, top_p=0.95, repeat_penalty=1.1)
        with pytest.raises(RuntimeError, match="grow boom"):
            list(gen)

    assert mock_api.llama_batch_free.call_count >= 1, (
        "the native batch must be freed even when mid-generation growth raises")
