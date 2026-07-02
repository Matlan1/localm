# SPDX-License-Identifier: AGPL-3.0-or-later
"""The grammar sampler works; the generation loop must never double-accept.

llama_sampler_sample() already ACCEPTS the sampled token into every stateful
sampler in the chain (upstream documents it as "sample and accept"). The loop
used to call llama_sampler_accept() again after it, which advanced the grammar
parser twice per token until its parse stacks emptied and it threw
std::runtime_error across the C ABI (WinError 0xe06d7363) - misdiagnosed for
months as "the bundled build's grammar sampler faults". It also double-counted
every token in the repetition-penalty window.

The unit tests pin the no-double-accept contract with the DLL never loaded;
the @integration test proves grammar-constrained generation end to end on a
real model (skipped unless the native runtime + network are available)."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import LlamaCpp

_FAKES: list = []


def _bare_llama() -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._n_ctx = 4096
    llm._n_ctx_max = None
    llm._n_ctx_grow = 4096
    llm._seed = 1234
    llm._verbose = False
    llm._model_ptr = 111
    llm._ctx_ptr = 222
    llm._tokenizer = MagicMock()
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm._kv_supported = None
    llm._gen_lock = threading.RLock()
    llm._stop = threading.Event()
    _FAKES.append(llm)
    return llm


@pytest.fixture(autouse=True)
def _null_fake_pointers():
    # Null the fake pointers before GC so __del__ -> close() does not pass a
    # bogus int to the real llama_free.
    yield
    for llm in _FAKES:
        llm._model_ptr = None
        llm._ctx_ptr = None
    _FAKES.clear()


def test_generate_never_calls_accept_after_sample():
    llm = _bare_llama()
    mock_api = MagicMock()
    mock_api.llama_sampler_sample.side_effect = [11, 12, 13]
    mock_api.llama_decode.return_value = 0
    llm._tokenizer.is_eog.side_effect = lambda t: t == 13

    with patch("localm.inference.backends.llamacpp.llama.api", mock_api), \
         patch("localm.inference.backends.llamacpp.llama._build_sampler",
               return_value=999), \
         patch.object(llm, "_fit_generation_budget", return_value=8), \
         patch.object(llm, "_can_reuse_kv", return_value=True), \
         patch.object(llm, "_prefill_with_reuse", return_value=None):
        tokens = list(llm._generate([1, 2, 3], max_new_tokens=8, temperature=0.8,
                                    top_k=40, top_p=0.95, repeat_penalty=1.1))

    assert tokens == [11, 12], "sampled tokens stream until the EOG token"
    # THE regression pin: sample() accepts internally; a second accept faults
    # the grammar sampler and double-counts the penalties window.
    mock_api.llama_sampler_accept.assert_not_called()
    mock_api.llama_sampler_free.assert_called_once_with(999)


# --------------------------------------------------------------------------- #
# Real-model proof (same gating pattern as test_gguf_smoke_integration.py).
# --------------------------------------------------------------------------- #

_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"


@pytest.mark.integration
@pytest.mark.real_gguf
def test_grammar_constrains_real_generation():
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_REPO, filename=_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_REPO}/{_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend(path, n_ctx=1024)
    backend.load()
    try:
        out = "".join(backend.chat_stream(
            [{"role": "user", "content": "Answer with one word, yes or no: "
                                         "is water wet?"}],
            max_tokens=8, temperature=0.0,
            grammar='root ::= "yes" | "no"',
        ))
        assert out in ("yes", "no"), f"grammar must constrain the output, got {out!r}"
        # The soft-degrade flag must NOT have been tripped: the constraint was
        # actually enforced, not silently dropped.
        assert not getattr(backend, "_grammar_unsupported", False)
    finally:
        backend.unload()
