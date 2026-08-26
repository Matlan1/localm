# SPDX-License-Identifier: AGPL-3.0-or-later
"""A per-request seed must reach the GGUF sampler.

A request `seed` that lands in **_ignored leaves the sampler built with the
instance default (LLAMA_DEFAULT_SEED), so generation at temperature>0 is
non-deterministic and the user's seed is silently ignored. These tests pin that
the seed threads create_chat_completion -> _generate -> _build_sampler
(llama_sampler_init_dist). The DLL is never loaded.
"""

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
    llm._inference_lock = threading.Lock()
    llm._stop = threading.Event()
    _FAKES.append(llm)
    return llm


@pytest.fixture(autouse=True)
def _null_fake_pointers():
    # Null the fake model/ctx pointers before GC so __del__ -> close() does not
    # pass a bogus int to the real llama_free.
    yield
    for llm in _FAKES:
        llm._model_ptr = None
        llm._ctx_ptr = None
    _FAKES.clear()


class _StopAfterSampler(Exception):
    pass


def _seed_passed_to_sampler(llm, **gen_kwargs):
    captured = {}

    def fake_build_sampler(**kwargs):
        captured["seed"] = kwargs["seed"]
        raise _StopAfterSampler()

    with patch("localm.inference.backends.llamacpp.llama._build_sampler",
               side_effect=fake_build_sampler), \
         patch.object(llm, "_fit_generation_budget", return_value=8), \
         patch.object(llm, "_can_reuse_kv", return_value=True), \
         patch.object(llm, "_prefill_with_reuse", return_value=None):
        with pytest.raises(_StopAfterSampler):
            list(llm._generate([1, 2, 3], max_new_tokens=8, temperature=0.8,
                               top_k=40, top_p=0.95, repeat_penalty=1.1,
                               **gen_kwargs))
    return captured["seed"]


def test_generate_threads_request_seed():
    # The request seed reaches the sampler.
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm, seed=42) == 42


def test_generate_seed_masked_to_uint32():
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm, seed=0x1_0000_0001) == 1


def test_generate_defaults_to_instance_seed():
    # No request seed: the instance default is used.
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm) == 1234


def test_create_chat_completion_forwards_seed():
    # The seed is forwarded to _generate, not absorbed into **_ignored.
    llm = _bare_llama()
    with patch("localm.inference.backends.llamacpp.llama._apply_model_template",
               return_value=("hi", None)), \
         patch.object(llm, "_generate", return_value=iter([])) as gen:
        llm.create_chat_completion(
            [{"role": "user", "content": "x"}], seed=7, stream=False)
    assert gen.call_args.kwargs.get("seed") == 7
