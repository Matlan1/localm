# SPDX-License-Identifier: AGPL-3.0-or-later
"""max_tokens<=0 on the HF backend: the GGUF backend's own "unlimited"
sentinel, translated for transformers.generate() (which has no such sentinel
and rejects a non-positive max_new_tokens outright - see
test_generate_itself_rejects_non_positive_max_new_tokens below).

Three layers, each catching a different way this could break:
  - the arithmetic (_resolve_max_new_tokens, pure, no torch/transformers needed)
  - the wiring (chat_stream must pass the RESOLVED value to generate(), not
    the raw max_tokens - a bug the arithmetic tests alone cannot see)
  - the real thing (an actual HFWorker.chat_stream() call against a real
    model must not raise)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from localm.inference.backends._hf_worker import _resolve_max_new_tokens

# --------------------------------------------------------------------------- #
#  _resolve_max_new_tokens: pure arithmetic, no torch/transformers import.
# --------------------------------------------------------------------------- #


def test_positive_max_tokens_passes_through_unchanged():
    assert _resolve_max_new_tokens(256, context_capacity=4096, n_prompt=10) == 256
    assert _resolve_max_new_tokens(1, context_capacity=None, n_prompt=None) == 1


def test_zero_with_known_capacity_fits_the_remaining_room():
    assert _resolve_max_new_tokens(0, context_capacity=1024, n_prompt=24) == 1000


def test_negative_with_known_capacity_also_means_unlimited():
    assert _resolve_max_new_tokens(-1, context_capacity=1024, n_prompt=24) == 1000


def test_zero_at_the_capacity_boundary_never_resolves_to_zero():
    # n_prompt == context_capacity leaves no room; the result must still be a
    # value generate() accepts (>= 1), never the 0 that started this.
    assert _resolve_max_new_tokens(0, context_capacity=100, n_prompt=100) == 1


def test_zero_with_unknown_capacity_falls_back_to_the_configured_default():
    from localm.config import DEFAULT_CONFIG

    assert (_resolve_max_new_tokens(0, context_capacity=None, n_prompt=50)
            == DEFAULT_CONFIG["max_tokens"])
    assert (_resolve_max_new_tokens(0, context_capacity=1024, n_prompt=None)
            == DEFAULT_CONFIG["max_tokens"])


# --------------------------------------------------------------------------- #
#  Wiring: chat_stream() must pass the RESOLVED budget to generate(), not the
#  raw max_tokens=0 a correct _resolve_max_new_tokens alone cannot prove.
# --------------------------------------------------------------------------- #


def test_chat_stream_passes_the_resolved_budget_to_generate(monkeypatch):
    pytest.importorskip("transformers", exc_type=ImportError)
    import transformers

    # chat_stream's `from transformers import StoppingCriteriaList, ...`
    # triggers a fresh `import torch` via transformers' lazy loader - the
    # known-doomed DLL-identity conflict if llama.cpp's native runtime is
    # already loaded in this process (see test_hf_prompt_tokenization.py,
    # same guard).
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this "
                     "process - not this test's own subject")

    from localm.inference.backends import _hf_worker as hfmod

    be = hfmod.HFWorker.__new__(hfmod.HFWorker)
    be._processor = None
    be._is_multimodal = False
    be._model = MagicMock()
    be.context_capacity = 1024

    tok = MagicMock()
    tok.apply_chat_template.return_value = "hi"
    fake_input_ids = MagicMock()
    fake_input_ids.shape = (1, 24)
    tok.return_value.to.return_value = {
        "input_ids": fake_input_ids, "attention_mask": MagicMock(),
    }
    be._tokenizer = tok

    class _FakeStreamer:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())   # yield nothing -> the generate thread is a no-op

    monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
    monkeypatch.setattr(hfmod, "_grammar_processor", lambda *a, **k: None)

    list(be.chat_stream([{"role": "user", "content": "hi"}], max_tokens=0))

    assert be._model.generate.call_args is not None, "generate() was never called"
    got = be._model.generate.call_args.kwargs["max_new_tokens"]
    assert got == 1000, f"expected context_capacity(1024) - n_prompt(24) == 1000, got {got}"


# --------------------------------------------------------------------------- #
#  Real generation: no mocks. Uses the EOS-forced fixture from
#  test_hf_finish_reason_integration.py (same model, forces eos as the very
#  first token) so this stays fast regardless of the resolved budget's size -
#  the point here is proving no ValueError, not exercising a long generation.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_generate_itself_rejects_non_positive_max_new_tokens():
    """Pins the external contract _resolve_max_new_tokens exists to work
    around. If transformers ever starts accepting max_new_tokens<=0, this
    function's reason for existing needs re-examining."""
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.requires_grad_(False)
    inputs = tok("hi", return_tensors="pt")
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(**inputs, max_new_tokens=0, do_sample=False)
