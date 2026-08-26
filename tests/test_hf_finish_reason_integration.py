# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL finish_reason tests for the HuggingFace backend.

No mocks: a tiny ungated causal LM (sshleifer/tiny-gpt2, the same fixture
model as test_hf_grammar_integration.py) is loaded for real, in the real
isolated worker process (see backends/_hf_runner.py), and
HFBackend.chat_stream()'s reported last_finish_reason is checked against
real generation.

"stop" is also the answer whenever nothing overrides it, so a test asserting
only that would pass on a hardcoded value. The "length" case below is what
shows the value is computed: it requires the isolated worker to observe that
max_tokens was exhausted with no end-of-sequence token ever produced.

Marked @integration (downloads a small cached model on first run), so the
default `pytest -m "not integration"` skips it.
"""

from __future__ import annotations

import json
import shutil

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2 ships no chat template; this trivial one (concatenate message
# contents) lets chat_stream's apply_chat_template run.
_MINIMAL_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
_MESSAGES = [{"role": "user", "content": "Answer:"}]

# Multiplier applied to the model's own final hidden state when forcing its
# eos-token logit (see stop_model_dir below). Large enough to dominate
# tiny-gpt2's other, roughly [-1, 1]-scale logits across a save/reload
# round-trip.
_EOS_FORCE_SCALE = 1e6


def _fetch_and_prepare(tmp_path_factory, name):
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    # Into a COPY, never the shared HF hub cache snapshot_download returned,
    # which every other use of this cached model on the machine also reads.
    model_dir = tmp_path_factory.mktemp(name)
    shutil.copytree(local_dir, model_dir, dirs_exist_ok=True)
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = _MINIMAL_CHAT_TEMPLATE
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return model_dir


@pytest.fixture(scope="module")
def length_model_dir(tmp_path_factory):
    """sshleifer/tiny-gpt2 with UNTOUCHED weights (only the chat template is
    injected). Greedy decoding on this checkpoint does not produce its own eos
    token within 40 generated tokens, so a small max_tokens budget exhausts
    itself first and the result is "length".
    """
    return _fetch_and_prepare(tmp_path_factory, "tiny_gpt2_length")


@pytest.fixture(scope="module")
def stop_model_dir(tmp_path_factory):
    """A saved copy of sshleifer/tiny-gpt2 whose lm_head weight row for the
    real eos_token_id has been overwritten with a large positive multiple of
    the model's OWN final hidden state for this exact prompt/chat-template
    text: v . (k*v) = k*||v||^2 > 0 for any k > 0, so the eos logit dominates
    whatever the sign of each hidden dimension.

    That forces greedy decoding to emit eos as the very first generated token,
    deterministically, and survives a save/from_pretrained reload.
    GPT2LMHeadModel's lm_head is bias-free (tie_word_embeddings=True), so the
    override has to be a real WEIGHT edit; a bias would be dropped as an
    "unexpected key" on reload, and the isolated worker always loads fresh from
    disk in its own process rather than reusing this fixture's in-memory model.
    """
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = _fetch_and_prepare(tmp_path_factory, "tiny_gpt2_stop")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    # The exact tokenization HFWorker.chat_stream() produces for _MESSAGES (same
    # chat template, same add_special_tokens=False), so the hidden state used for
    # the surgery matches the position right after the prompt.
    text = tokenizer.apply_chat_template(
        _MESSAGES, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states[-1][0, -1, :]
        model.lm_head.weight.data[tokenizer.eos_token_id] = hidden * _EOS_FORCE_SCALE

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir


def test_max_tokens_truncation_reports_length(length_model_dir):
    """Real generation on the untouched model runs the full max_tokens budget
    (no natural eos - see length_model_dir), so last_finish_reason is "length",
    not the pre-generation default."""
    from localm.inference.backends.hf import HFBackend

    be = HFBackend(str(length_model_dir), device="cpu")
    be.load()
    try:
        list(be.chat_stream(_MESSAGES, temperature=0.0, max_tokens=8))
        assert be.last_finish_reason == "length", (
            f"expected a genuine max_tokens truncation to report 'length', "
            f"got {be.last_finish_reason!r}"
        )
    finally:
        be.unload()


def test_natural_eos_reports_stop(stop_model_dir):
    """The rigged model emits eos as its very first token, well inside a
    generous max_tokens budget, so "stop" here reflects an end-of-sequence
    signal the isolated worker actually observed."""
    from localm.inference.backends.hf import HFBackend

    be = HFBackend(str(stop_model_dir), device="cpu")
    be.load()
    try:
        chunks = list(be.chat_stream(_MESSAGES, temperature=0.0, max_tokens=20))
        assert be.last_finish_reason == "stop", (
            f"expected a genuine early eos to report 'stop', got "
            f"{be.last_finish_reason!r}"
        )
        # eos is a special token: TextIteratorStreamer(skip_special_tokens=True)
        # never surfaces it as visible text, so nothing should have streamed.
        assert "".join(chunks) == ""
    finally:
        be.unload()
