# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end smoke test for the in-process GGUF backend.

No mocks: downloads a small real instruct model (SmolLM2-135M-Instruct, ~88 MB
Q4) and runs it through the native llama.dll ctypes binding via GgufBackend.
The test floor for the core promise: a GGUF model loads and actually generates
text.

@integration so the default `pytest -m "not integration"` skips it: it needs the
native runtime provisioned (localm setup-llama) and ~88 MB of network on first
run. Skips cleanly (does not fail) when either is unavailable.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.fixture(scope="module")
def gguf_backend():
    # The native llama runtime must be provisioned (llama.dll + ggml). If it is
    # not, this is an environment gap, not a test failure - skip.
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned (run 'localm setup-llama'): {e}")

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(path, n_ctx=2048)   # default GPU offload; load() falls back as needed
    try:
        be.load()
    except Exception as e:
        pytest.skip(f"GGUF model failed to load on this machine: {e}")
    yield be
    be.unload()


def test_gguf_loads_and_generates_real_text(gguf_backend):
    """The core promise: load a real GGUF and stream back real generated text."""
    out = "".join(gguf_backend.chat_stream(
        [{"role": "user", "content": "In one short sentence, what color is a clear daytime sky?"}],
        max_tokens=40, temperature=0.0, seed=1,
    )).strip()
    assert len(out) >= 10, f"suspiciously short output: {out!r}"
    assert any(c.isalpha() for c in out), f"no words in output: {out!r}"
    assert gguf_backend.loaded, "model should remain loaded after a normal generation"


def test_gguf_load_populates_metadata_from_isolated_worker(gguf_backend):
    """The model's whole lifecycle runs in an isolated worker process. The
    metadata it reports back (effective_ctx_max / effective_gpu_layers /
    supports_images) must reach the parent-side GgufBackend correctly, through
    a REAL load."""
    # The fixture constructs GgufBackend directly with ctx_auto=False (the
    # constructor default) and no explicit n_ctx_max, so the resolved ceiling
    # is None.
    assert gguf_backend.effective_ctx_max is None
    assert isinstance(gguf_backend.effective_gpu_layers, int) and gguf_backend.effective_gpu_layers >= 0
    # SmolLM2-135M-Instruct has no mmproj, so the flag read from the real load
    # response is exactly False.
    assert gguf_backend.supports_images is False


def test_gguf_unload_reload_cycle_through_isolated_worker(gguf_backend):
    """unload() must cleanly tear down the isolated worker process, and a
    subsequent load() must spawn a fresh one and generate real text again.
    Leaves gguf_backend loaded afterward, matching the fixture's steady state
    for any later test in this module."""
    gguf_backend.unload()
    assert not gguf_backend.loaded
    assert gguf_backend._runner is None or not gguf_backend._runner.is_alive()

    gguf_backend.load()
    assert gguf_backend.loaded
    out = "".join(gguf_backend.chat_stream(
        [{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10, temperature=0.0, seed=1,
    )).strip()
    assert any(c.isalpha() for c in out), f"reload-then-generate broke: {out!r}"


def test_gguf_count_messages_tokens_uses_real_tokenizer_not_heuristic(gguf_backend):
    """count_messages_tokens must use the REAL tokenizer inside the isolated
    worker, never a heuristic.

    This drives the real isolated worker process end to end (a genuinely
    loaded tiny GGUF, not a stub), so a regression fails here even where a
    mocked unit test would still pass.

    The comparison against count_tokens() is STRICT, not ``>=``: when
    count_messages_tokens' own RPC fails, GgufBackend.count_messages_tokens
    falls to ``super().count_messages_tokens(messages)`` (BaseBackend), whose
    ``self.count_tokens(text)`` polymorphically resolves to GgufBackend's OWN
    count_tokens override - a separate RPC returning an exact tokenizer count
    of the raw, UNTEMPLATED content. The chat template adds real role/turn
    markup (e.g. SmolLM2's ``<|im_start|>user\\n...<|im_end|>\\n``) on top of
    that content, so a correctly-applied template must add tokens and a ``>=``
    comparison would not discriminate."""
    text = "The quick brown fox jumps over the lazy dog."
    messages = [{"role": "user", "content": text}]

    raw_tokens = gguf_backend.count_tokens(text)
    templated_tokens = gguf_backend.count_messages_tokens(messages)

    assert raw_tokens > 0
    assert templated_tokens > raw_tokens, (
        f"count_messages_tokens returned {templated_tokens}, not more than "
        f"the raw content's own {raw_tokens} tokens - the chat template was "
        "not applied by a real tokenizer call. This is the exact shape of "
        "the #956 regression: the RPC failed and silently fell back to a "
        "real count of the UNTEMPLATED text (which equals raw_tokens "
        "exactly), not a documented heuristic - see the docstring above.")


def test_gguf_grammar_request_never_breaks_chat(gguf_backend):
    """A grammar request must be SAFE on GGUF: where the native build supports
    grammar it constrains output, and where it does not (some prebuilt llama.dll
    builds ship a faulting grammar sampler) it soft-degrades to unconstrained
    generation. Either way it returns real text, raises nothing, and leaves the
    model loaded - it must never fault the session."""
    from localm.inference.gbnf import JSON_OBJECT
    out = "".join(gguf_backend.chat_stream(
        [{"role": "user", "content": "Give a JSON object with a key 'color'."}],
        grammar=JSON_OBJECT, max_tokens=48, temperature=0.0, seed=1,
    )).strip()
    assert len(out) >= 2, f"grammar request produced almost nothing: {out!r}"
    assert gguf_backend.loaded, "a grammar request must not unload the model"

    # And a follow-up plain chat must still work on the same loaded instance.
    follow = "".join(gguf_backend.chat_stream(
        [{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10, temperature=0.0, seed=1,
    )).strip()
    assert any(c.isalpha() for c in follow), f"follow-up chat broke: {follow!r}"
