# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wiring tests for gpu_split_indices/gpu_split_ratios into the native
llama.cpp model params, for both native-load call sites: the chat backend
(LlamaCpp) and the embedder (GGUFEmbedder). Both route through
localm.discover.apply_gpu_split (called right after apply_main_gpu - see
llama.py and embedder.py), but each constructs its own ``mp`` via a mocked
ctypes API, so we verify each call site actually sets mp.split_mode /
mp.tensor_split / mp.main_gpu end to end.

Mirrors tests/test_main_gpu_wiring.py's structure and mocking style exactly:
only the ctypes ``api``/``_api`` module and localm.discover.list_gpus /
localm.config.load_config are mocked. apply_gpu_split (and apply_main_gpu)
themselves run for REAL - this proves the actual wiring, not a mock of the
function under test. apply_gpu_split's tensor_split-capacity probe
(discover._tensor_split_capacity) calls the real
localm.inference.backends.llamacpp._api.has_max_devices(), which is NOT
touched by patching the "api" name inside llama.py's module namespace (that
only rebinds the name llama.py itself uses); it either succeeds against a
provisioned native runtime or raises, which _tensor_split_capacity already
catches and falls back to its documented constant - so this test is safe on a
box with no native llama.cpp runtime provisioned too."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from localm.inference.backends.llamacpp.llama import LlamaCpp

# llama.cpp's LLAMA_SPLIT_MODE_LAYER - see discover.py's
# _LLAMA_SPLIT_MODE_LAYER and _structs.py's field-layout comment
# ("[20] i32 split_mode = 1 (LLAMA_SPLIT_MODE_LAYER)" - the real native
# default). apply_gpu_split() sets it explicitly whenever it activates a
# split, and leaves it alone otherwise, so it is also what we seed the mock's
# "before construction" value with to prove "untouched".
_LLAMA_SPLIT_MODE_LAYER = 1

# Sentinel standing in for the native default tensor_split pointer (a real
# build hands back a live pointer to a static default array, not NULL/0 - see
# _structs.py). This must be a REAL, safely-dereferenceable address rather
# than a bogus int like 0xDEADBEEF: test_split_indices_set_split_mode_and_
# tensor_split() dereferences mp.tensor_split via ctypes after construction,
# so if a future regression breaks the wiring and apply_gpu_split() never
# runs, mp.tensor_split stays at this "untouched" sentinel and gets
# dereferenced anyway - a fake address there segfaults the whole pytest
# process (verified via a manual mutation test: temporarily no-opping
# apply_gpu_split's call site crashed with a Windows access violation
# instead of a clean assertion failure). Backing it with a real, permanently-
# kept-alive ctypes buffer means that same regression instead reads back
# zeros and fails a normal, readable assertion.
_NATIVE_DEFAULT_TENSOR_SPLIT_BUF = (ctypes.c_float * 16)()
_NATIVE_DEFAULT_TENSOR_SPLIT = ctypes.cast(
    _NATIVE_DEFAULT_TENSOR_SPLIT_BUF, ctypes.c_void_p).value


def _seeded_mp() -> SimpleNamespace:
    """A model-params SimpleNamespace seeded with the real native defaults for
    every field these tests care about (main_gpu, split_mode, tensor_split),
    same rationale as test_main_gpu_wiring.py's _mock_llama_api(): an
    untouched attribute must read back as the true native default, not an
    auto-generated child MagicMock."""
    return SimpleNamespace(
        main_gpu=0,
        n_gpu_layers=0,
        use_mmap=True,
        split_mode=_LLAMA_SPLIT_MODE_LAYER,
        tensor_split=_NATIVE_DEFAULT_TENSOR_SPLIT,
    )


def _mock_llama_api():
    mock_api = MagicMock()
    mock_api.llama_model_default_params.return_value = _seeded_mp()
    return mock_api


def _mock_embed_api():
    mock_api = MagicMock()
    mock_api.llama_model_default_params.return_value = _seeded_mp()
    mock_api.llama_model_n_embd.return_value = 768   # int(...) would TypeError on a bare Mock
    mock_api.has_embeddings_api.return_value = True
    mock_api.has_memory_api.return_value = False
    return mock_api


def _tensor_split_values(mp, count: int):
    """Read *count* floats back out of mp.tensor_split the same way the task
    spec describes: cast the raw pointer to POINTER(c_float) and index in."""
    ptr = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
    return [ptr[i] for i in range(count)]


class TestLlamaCppGpuSplitWiring:
    """LlamaCpp.__init__ (localm/inference/backends/llamacpp/llama.py)."""

    def _build(self, mock_api):
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            # verbose=True skips the native-stderr-capture context managers
            # (irrelevant to this test and unnecessary fd juggling against a
            # fully mocked api).
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            # Close deterministically while `api` is still patched: LlamaCpp's
            # cleanup path (_free_native) reads the module-global `api` name,
            # not a stored instance attribute, so a GC-triggered __del__ after
            # this `with` exits would otherwise call the REAL native
            # llama_free_model with a MagicMock pointer.
            llm.close()
            return llm

    def test_split_indices_set_split_mode_and_tensor_split(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 2], "gpu_split_ratios": [0.25, 0.75]},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}],
        )
        mock_api = _mock_llama_api()
        self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        values = _tensor_split_values(mp, 3)
        assert values[0] == 0.25
        assert values[1] == 0.0   # not one of the configured indices
        assert values[2] == 0.75
        # index 0 (native default main_gpu) is already one of the split
        # devices, so no correction is needed here.
        assert mp.main_gpu == 0

    def test_unset_split_indices_leaves_native_defaults(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"main_gpu_index": 1, "gpu_split_indices": None},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}]
        )
        mock_api = _mock_llama_api()
        self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        # main_gpu still behaves exactly like the existing main_gpu wiring
        # tests (test_main_gpu_wiring.py) prove in isolation.
        assert mp.main_gpu == 1
        # apply_gpu_split() saw fewer than 2 valid split devices (none
        # configured) and returned without touching either field.
        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        assert mp.tensor_split == _NATIVE_DEFAULT_TENSOR_SPLIT

    def test_main_gpu_not_in_split_indices_is_corrected(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"main_gpu_index": 0, "gpu_split_indices": [1, 2]},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}],
        )
        mock_api = _mock_llama_api()
        with caplog.at_level("WARNING", logger="localm"):
            self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        # main_gpu_index=0 was explicitly configured and resolves cleanly on
        # its own (apply_main_gpu sets mp.main_gpu = 0), but it is not one of
        # the split devices [1, 2] - apply_gpu_split must correct it to the
        # first split device, not leave the originally configured value.
        assert mp.main_gpu == 1
        assert mp.main_gpu != 0
        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        values = _tensor_split_values(mp, 3)
        assert values[1] == 1.0   # no ratios given -> equal split
        assert values[2] == 1.0
        assert any(
            "gpu_split_indices" in r.message and "main_gpu_index" in r.message
            for r in caplog.records
        )


class TestGgufEmbedderGpuSplitWiring:
    """GGUFEmbedder.__init__ (localm/inference/embedder.py)."""

    def _build(self, mock_api):
        # embedder.py imports _api locally each call ("from ...llamacpp import
        # _api as api"), which resolves via getattr on the already-imported
        # llamacpp PACKAGE object - patch that package attribute, not sys.modules.
        with patch("localm.inference.backends.llamacpp._api", mock_api):
            from localm.inference.embedder import GGUFEmbedder
            return GGUFEmbedder("embed.gguf")

    def test_split_indices_set_split_mode_and_tensor_split(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 2], "gpu_split_ratios": [0.25, 0.75]},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}],
        )
        mock_api = _mock_embed_api()
        self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        values = _tensor_split_values(mp, 3)
        assert values[0] == 0.25
        assert values[1] == 0.0
        assert values[2] == 0.75
        assert mp.main_gpu == 0

    def test_unset_split_indices_leaves_native_defaults(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"main_gpu_index": 1, "gpu_split_indices": None},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus", lambda: [{"index": 0}, {"index": 1}]
        )
        mock_api = _mock_embed_api()
        self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        assert mp.main_gpu == 1
        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        assert mp.tensor_split == _NATIVE_DEFAULT_TENSOR_SPLIT

    def test_main_gpu_not_in_split_indices_is_corrected(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"main_gpu_index": 0, "gpu_split_indices": [1, 2]},
        )
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0}, {"index": 1}, {"index": 2}],
        )
        mock_api = _mock_embed_api()
        with caplog.at_level("WARNING", logger="localm"):
            self._build(mock_api)
        mp = mock_api.llama_model_default_params.return_value

        assert mp.main_gpu == 1
        assert mp.main_gpu != 0
        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        values = _tensor_split_values(mp, 3)
        assert values[1] == 1.0
        assert values[2] == 1.0
        assert any(
            "gpu_split_indices" in r.message and "main_gpu_index" in r.message
            for r in caplog.records
        )
