# SPDX-License-Identifier: AGPL-3.0-or-later
"""A context the runtime refuses frees the loaded model exactly once.

LlamaCpp frees the model when llama_init_from_model returns NULL and then
raises. The half-built instance is still collected afterwards and its
``__del__`` runs ``close()``, which frees whatever ``_model_ptr`` still holds,
so the pointer must be cleared at the first free: a second free of the same
native model is a double free in the worker process.
"""

import gc
from unittest.mock import patch

import pytest

from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp
from tests.test_kv_cache_type_setting import API, _load_api


@pytest.mark.parametrize("kind", [None, "q8_0"])
def test_a_refused_context_frees_the_model_once(kind):
    api, _seen = _load_api(init_results=[None])
    model = api.llama_load_model_from_file.return_value
    with patch(API, api), \
            patch.object(llama_mod.pretokenizer_guard, "read_pre_type", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to create llama context"):
            LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=0, verbose=True,
                     kv_cache_type=kind)
        gc.collect()
    assert api.llama_free_model.call_count == 1
    api.llama_free_model.assert_called_with(model)
