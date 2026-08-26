# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL grammar-constrained-decoding test for the HuggingFace backend.

No mocks: a tiny ungated causal LM (sshleifer/tiny-gpt2, full GPT-2 tokenizer)
is loaded for real under transformers 5, and a GBNF/EBNF grammar passed through
``HFBackend.chat_stream`` must actually constrain the generated tokens via
xgrammar. The negative assertion (same prompt, no grammar -> NOT a legal value)
is what proves the grammar caused the constraint, not the model.

Marked @integration so the default `pytest -m "not integration"` skips it (it
downloads a few MB on first run and needs the optional [grammar] extra). A mock
here would prove nothing: the property is that the masking runs for real.
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


@pytest.fixture(scope="module")
def hf_backend(tmp_path_factory):
    # exc_type=ImportError on all three: each can raise a plain ImportError
    # (not ModuleNotFoundError) for a reason other than "not installed", and
    # pytest 9.1 narrowed importorskip's default to ModuleNotFoundError only.
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    pytest.importorskip("xgrammar", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    from localm.inference.backends.hf import HFBackend

    # The chat_template cannot be poked onto a live tokenizer object after
    # load(): the real tokenizer lives inside HFBackend's isolated child
    # process. Inject it at the SOURCE instead - tokenizer_config.json's own
    # "chat_template" key, which AutoTokenizer.from_pretrained reads at load
    # time - into a COPY of the snapshot, never the shared HF hub cache
    # directory snapshot_download returned.
    model_dir = tmp_path_factory.mktemp("tiny_gpt2_with_chat_template")
    shutil.copytree(local_dir, model_dir, dirs_exist_ok=True)
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = _MINIMAL_CHAT_TEMPLATE
    config_path.write_text(json.dumps(config), encoding="utf-8")

    be = HFBackend(str(model_dir), device="cpu")
    be.load()
    yield be
    be.unload()


def _generate(be, grammar, max_tokens):
    msgs = [{"role": "user", "content": "Answer:"}]
    # temperature=0.0 -> greedy/deterministic, so the assertions are stable.
    return "".join(
        be.chat_stream(msgs, grammar=grammar, temperature=0.0, max_tokens=max_tokens)
    ).strip()


def test_grammar_constrains_output_to_choice(hf_backend):
    """A 2-alternative grammar forces the output to one of its literals; the
    unconstrained model emits neither (the negative test that proves causation)."""
    choice = 'root ::= "yes" | "no"'
    constrained = _generate(hf_backend, choice, max_tokens=5)
    unconstrained = _generate(hf_backend, None, max_tokens=5)

    assert constrained in ("yes", "no"), (
        f"grammar did not constrain output: {constrained!r}"
    )
    # Without the grammar the (gibberish) tiny model must NOT land on a legal
    # value.
    assert unconstrained not in ("yes", "no"), (
        f"unconstrained output coincidentally legal ({unconstrained!r}); "
        "negative test is inconclusive"
    )


def test_grammar_forces_parseable_json(hf_backend):
    """A fixed-shape JSON grammar yields output that json.loads actually parses,
    with the expected key - structural validity by construction."""
    fixed = r'root ::= "{" "\"ok\"" ":" ("true"|"false") "}"'
    out = _generate(hf_backend, fixed, max_tokens=16)
    obj = json.loads(out)               # raises -> test fails if masking is wrong
    assert "ok" in obj and isinstance(obj["ok"], bool)


def test_unconstrained_baseline_is_not_json(hf_backend):
    """Guards the negative side of the JSON test: the same tiny model, free-
    running, does NOT emit valid JSON - so test_grammar_forces_parseable_json
    is really measuring the grammar, not the model."""
    out = _generate(hf_backend, None, max_tokens=16)
    with pytest.raises(Exception):
        json.loads(out)
