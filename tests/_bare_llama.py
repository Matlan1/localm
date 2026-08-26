# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared builder for a bare LlamaCpp instance (native __init__ bypassed).

Fourteen test files used to hand-build this object, each re-establishing
__init__'s invariants by copy-pasting a subset of them. That is what let
_mtp_ctx_ptr go missing from ten of the fourteen copies when __init__ grew
it: a fixture duplicated fourteen times rots at whichever copy nobody is
looking at.

make_bare_llama() sets every attribute LlamaCpp.__init__ establishes, so a
caller overriding only what it needs cannot end up missing an invariant a
different code path later reads. Pointer-shaped attributes (_model_ptr,
_ctx_ptr, _mtp_ctx_ptr) default to None - the same value __init__ holds
before any native call - so close()/__del__ is inert by default: nothing
here reaches the real native free unless a caller deliberately overrides a
pointer to a fake truthy value. tests/conftest.py's autouse
_neutralise_bare_llama_pointers fixture calls neutralise_fake_pointers()
after every test, so no test module needs its own teardown for this.
"""
import threading
from unittest.mock import MagicMock

from localm.inference.backends.llamacpp.llama import LlamaCpp

_LIVE: list = []


def make_bare_llama(**overrides) -> LlamaCpp:
    """Build a LlamaCpp with every __init__ invariant set to its
    before-any-native-call default, then apply *overrides* on top."""
    llm = LlamaCpp.__new__(LlamaCpp)
    llm._n_ctx = 4096
    llm._mtp_enabled = True
    llm._vram_check = None
    llm._n_ctx_max = None
    llm._n_ctx_grow = 4096
    llm._seed = 1234
    llm._verbose = False
    llm._model_ptr = None
    llm._ctx_ptr = None
    llm._mtp_ctx_ptr = None
    llm.supports_mtp = False
    llm._mmproj_path = None
    llm._mtmd = None
    llm._tokenizer = MagicMock()
    llm._gen_lock = threading.RLock()
    llm._stop = threading.Event()
    llm._inference_lock = threading.Lock()
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm._offload_kqv = True
    llm._kv_supported = None
    llm.chat_template_fallback_reason = None
    llm._main_gpu_index = 0
    llm._moe_override_keepalive = None
    llm.moe_skip_reason = None
    llm._cancel_event = None
    llm._load_progress_cb = None
    llm.weight_placement = []
    llm.n_layers = None
    llm.kv_bytes_per_token = 0
    for key, value in overrides.items():
        setattr(llm, key, value)
    _LIVE.append(llm)
    return llm


# Back-compat alias: several files already imported `_bare_llama` as the
# builder name (test_kv_cache.py's original, imported in turn by
# test_create_batch_fill.py). Keep the name importable so neither needs an
# unrelated rename.
_bare_llama = make_bare_llama


def neutralise_fake_pointers() -> None:
    """Null the native pointers on every instance make_bare_llama has built
    in this test run, so __del__ -> close() cannot dereference a fake
    address through the real native API. Call from an autouse teardown
    fixture in any module that overrides a pointer to a truthy fake."""
    for llm in _LIVE:
        llm._model_ptr = None
        llm._ctx_ptr = None
        llm._mtp_ctx_ptr = None
    _LIVE.clear()
