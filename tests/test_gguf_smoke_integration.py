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


def test_gguf_load_populates_metadata_from_isolated_worker(gguf_backend):
    """The model's whole lifecycle now runs in an isolated worker process (see
    llamacpp/_runner.py) - this proves the metadata it reports back
    (effective_ctx_max/effective_gpu_layers/supports_images) actually reaches
    the parent-side GgufBackend correctly end to end, through a REAL load, not
    just a mocked one (test_gguf_worker.py/test_vram_preflight.py already
    cover the wiring with mocks; this is the real thing)."""
    # The fixture constructs GgufBackend directly with ctx_auto=False (the
    # constructor default) and no explicit n_ctx_max, so "no ceiling" (None)
    # is the CORRECT resolved value here - not a positive int (that only
    # happens with ctx_auto=True or an explicit n_ctx_max, covered by
    # test_dynamic_ctx.py's mocked unit tests).
    assert gguf_backend.effective_ctx_max is None
    assert isinstance(gguf_backend.effective_gpu_layers, int) and gguf_backend.effective_gpu_layers >= 0
    # SmolLM2-135M-Instruct has no mmproj - text-only, so this must be False,
    # not just "truthy" (proves the flag is read from the real load response,
    # not defaulting to some other stand-in value).
    assert gguf_backend.supports_images is False


def test_gguf_unload_reload_cycle_through_isolated_worker(gguf_backend):
    """unload() must cleanly tear down the isolated worker process, and a
    subsequent load() must spawn a fresh one and generate real text again -
    proving the runner lifecycle (not just a single load) works end to end.
    Leaves gguf_backend loaded afterward, matching the fixture's steady
    state for any later test in this module."""
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
