"""REAL grammar-constrained-decoding test for the HuggingFace backend.

No mocks: a tiny ungated causal LM (sshleifer/tiny-gpt2, full GPT-2 tokenizer)
is loaded for real under transformers 5, and we assert that a GBNF/EBNF grammar
passed through ``HFBackend.chat_stream`` actually constrains the generated
tokens via xgrammar. The negative assertion (same prompt, no grammar -> NOT a
legal value) is what proves the grammar caused the constraint, not the model.

Marked @integration so the default `pytest -m "not integration"` skips it (it
downloads ~2.5 MB on first run and needs the optional [grammar] extra). A mock
here would be theater - the whole point is that the masking is exercised for real.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2 ships no chat template; this trivial one (concatenate message
# contents) lets chat_stream's apply_chat_template run. It is prompt plumbing
# only - it has nothing to do with the grammar masking under test.
_MINIMAL_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"


@pytest.fixture(scope="module")
def hf_backend():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("xgrammar")
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    from localm.inference.backends.hf import HFBackend

    be = HFBackend(local_dir, device="cpu")
    be.load()
    be._tokenizer.chat_template = _MINIMAL_CHAT_TEMPLATE
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
    # value - otherwise the test above would pass even with masking disabled.
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
