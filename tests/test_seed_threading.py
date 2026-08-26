# SPDX-License-Identifier: AGPL-3.0-or-later
"""B5: a per-request seed must reach the GGUF sampler.

The native llama.cpp wrapper dropped the request `seed` into **_ignored and
always built the sampler with the instance default (LLAMA_DEFAULT_SEED), so
generation at temperature>0 was non-deterministic and a user seed was silently
ignored. These tests pin that the seed threads create_chat_completion ->
_generate -> _build_sampler (llama_sampler_init_dist). The DLL is never loaded.
"""

from unittest.mock import patch

import pytest

from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests._bare_llama import make_bare_llama


def _bare_llama() -> LlamaCpp:
    return make_bare_llama(_model_ptr=111, _ctx_ptr=222)


# Fake-pointer teardown is handled globally by tests/conftest.py's autouse
# _neutralise_bare_llama_pointers fixture.


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
    # NEGATIVE: pre-fix _generate has no seed kwarg -> TypeError (not _StopAfterSampler).
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm, seed=42) == 42


def test_generate_seed_masked_to_uint32():
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm, seed=0x1_0000_0001) == 1


def test_generate_defaults_to_instance_seed():
    # Regression: no request seed -> the instance default is still used.
    llm = _bare_llama()
    assert _seed_passed_to_sampler(llm) == 1234


def test_create_chat_completion_forwards_seed():
    # NEGATIVE: pre-fix seed lands in **_ignored and is never forwarded.
    llm = _bare_llama()
    with patch("localm.inference.backends.llamacpp.llama._apply_model_template",
               return_value=("hi", None)), \
         patch.object(llm, "_generate", return_value=iter([])) as gen:
        llm.create_chat_completion(
            [{"role": "user", "content": "x"}], seed=7, stream=False)
    assert gen.call_args.kwargs.get("seed") == 7
