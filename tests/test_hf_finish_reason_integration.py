# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL finish_reason tests for the HuggingFace backend."""

from __future__ import annotations

import json
import shutil

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2 ships no chat template; this trivial one (concatenate message
# contents) lets chat_stream's apply_chat_template run, mirroring
# test_hf_grammar_integration.py's identical fixture technique.
_MINIMAL_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
_MESSAGES = [{"role": "user", "content": "Answer:"}]

# Multiplier applied to the model's own final hidden state when forcing its
# eos-token logit (see stop_model_dir below). Empirically confirmed to
# dominate tiny-gpt2's other (tiny, ~[-1, 1]-scale) logits by six orders of
# magnitude, comfortably enough to survive a save/reload round-trip.
_EOS_FORCE_SCALE = 1e6


def _fetch_and_prepare(tmp_path_factory, name):
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    # Into a COPY, never the shared HF hub cache snapshot_download returned -
    # mutating that would leak across every other test/use of this cached
    # model on the machine (same reasoning as test_hf_grammar_integration.py).
    model_dir = tmp_path_factory.mktemp(name)
    shutil.copytree(local_dir, model_dir, dirs_exist_ok=True)
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = _MINIMAL_CHAT_TEMPLATE
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return model_dir


@pytest.fixture(scope="module")
def length_model_dir(tmp_path_factory):
    """sshleifer/tiny-gpt2 with UNTOUCHED weights (only the chat template is injected)."""
    return _fetch_and_prepare(tmp_path_factory, "tiny_gpt2_length")


@pytest.fixture(scope="module")
def stop_model_dir(tmp_path_factory):
    """A saved copy of sshleifer/tiny-gpt2 whose lm_head weight row for the real eos_token_id has been overwritten with a large positive multiple of the model's OWN final hidden state for this exact prompt/chat-template text: v . (k*v) = k*||v||^2 > 0 for any k > 0, so the eos logit is guaranteed to domina..."""
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = _fetch_and_prepare(tmp_path_factory, "tiny_gpt2_stop")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    # The exact tokenization HFWorker.chat_stream() will produce for
    # _MESSAGES (same chat template, same add_special_tokens=False), so the
    # hidden state used for the surgery matches what generation will
    # actually see at the position right after the prompt.
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
    """Real generation on the untouched model must run the full max_tokens budget (no natural eos - see length_model_dir's docstring), so last_finish_reason must be 'length', not the pre-generation default."""
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
    """The rigged model emits eos as its very first token, well inside a generous max_tokens budget - proving 'stop' reflects a real end-of- sequence signal actually observed by the isolated worker, not just whatever last_finish_reason happened to default to before generation."""
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
