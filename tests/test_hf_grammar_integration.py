# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL grammar-constrained-decoding test for the HuggingFace backend."""

from __future__ import annotations

import json
import shutil

import pytest

pytestmark = pytest.mark.integration

_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2 ships no chat template; this trivial one (concatenate message
# contents) lets chat_stream's apply_chat_template run. It is prompt plumbing
# only - it has nothing to do with the grammar masking under test.
_MINIMAL_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"


@pytest.fixture(scope="module")
def hf_backend(tmp_path_factory):
    # exc_type=ImportError on all three: each can raise a plain ImportError
    # (not ModuleNotFoundError) for a reason other than "not installed" - e.g.
    # transformers' own internal tokenizers version-gate, or a version-clashed
    # xgrammar/torch build. pytest 9.1 narrowed importorskip's default to
    # ModuleNotFoundError only, so without this these hard-fail/error instead
    # of skipping on exactly the case this skip exists for.
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("transformers", exc_type=ImportError)
    pytest.importorskip("xgrammar", exc_type=ImportError)
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(_MODEL)
    except Exception as e:                       # offline / hub unreachable
        pytest.skip(f"could not fetch {_MODEL}: {e}")

    from localm.inference.backends.hf import HFBackend

    # The chat_template can no longer be poked onto a live tokenizer object
    # after load(): the real tokenizer now lives inside HFBackend's isolated
    # child process (see the thread-pool-exhaustion fix), not directly
    # reachable from this test's process. Inject it at the SOURCE instead -
    # tokenizer_config.json's own "chat_template" key, which
    # AutoTokenizer.from_pretrained reads at load time regardless of which
    # process calls it - into a COPY of the snapshot (never the shared HF hub
    # cache directory snapshot_download returned: mutating that would leak
    # across every other test/use of this cached model on the machine).
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
    """A 2-alternative grammar forces the output to one of its literals; the unconstrained model emits neither (the negative test that proves causation)."""
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
    """A fixed-shape JSON grammar yields output that json.loads actually parses, with the expected key - structural validity by construction."""
    fixed = r'root ::= "{" "\"ok\"" ":" ("true"|"false") "}"'
    out = _generate(hf_backend, fixed, max_tokens=16)
    obj = json.loads(out)               # raises -> test fails if masking is wrong
    assert "ok" in obj and isinstance(obj["ok"], bool)


def test_unconstrained_baseline_is_not_json(hf_backend):
    """Guards the negative side of the JSON test: the same tiny model, free- running, does NOT emit valid JSON - so test_grammar_forces_parseable_json is really measuring the grammar, not the model."""
    out = _generate(hf_backend, None, max_tokens=16)
    with pytest.raises(Exception):
        json.loads(out)
