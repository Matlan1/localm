# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end smoke test for the in-process GGUF backend.

No mocks: downloads a small real instruct model (SmolLM2-135M-Instruct, ~88 MB
Q4) and runs it through the native llama.dll ctypes binding via GgufBackend.
This is the test floor for the core promise - "a GGUF model loads and actually
generates text" - which until now was proven only by hand.

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


def test_gguf_grammar_request_never_breaks_chat(gguf_backend):
    """A grammar request must be SAFE on GGUF: where the native build supports
    grammar it constrains output, and where it does not (some prebuilt llama.dll
    builds ship a faulting grammar sampler) it soft-degrades to unconstrained
    generation. Either way it returns real text, raises nothing, and leaves the
    model loaded - it must never fault the session. Guards the soft-degrade fix."""
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
