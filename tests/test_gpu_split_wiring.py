# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wiring tests for gpu_split_indices/gpu_split_ratios into the native
llama.cpp model params, for both native-load call sites: the chat backend
(LlamaCpp) and the embedder (GGUFEmbedder). Both route through
localm.discover.apply_gpu_split (called right after apply_main_gpu), but each
constructs its own ``mp`` via a mocked ctypes API, so each call site is checked
end to end for mp.split_mode / mp.tensor_split / mp.main_gpu.

Only the ctypes ``api``/``_api`` module and localm.discover.list_gpus /
localm.config.load_config are mocked; apply_gpu_split and apply_main_gpu
themselves run for REAL. apply_gpu_split's tensor_split-capacity probe
(discover._tensor_split_capacity) calls the real
localm.inference.backends.llamacpp._api.has_max_devices(), which patching the
"api" name inside llama.py's module namespace does NOT touch (that only rebinds
the name llama.py itself uses); it either succeeds against a provisioned native
runtime or raises, and _tensor_split_capacity catches that and falls back to its
constant, so these tests also run on a box with no native llama.cpp runtime
provisioned."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.llamacpp.llama import LlamaCpp

# llama.cpp's LLAMA_SPLIT_MODE_LAYER, the native default. apply_gpu_split() sets
# it explicitly whenever it activates a split and leaves it alone otherwise, so
# it is also the mock's before-construction value.
_LLAMA_SPLIT_MODE_LAYER = 1

# Sentinel standing in for the native default tensor_split pointer. It must be a
# real, safely-dereferenceable address: a test below dereferences
# mp.tensor_split via ctypes after construction, and an untouched sentinel is
# dereferenced too. The backing buffer is kept alive for the process lifetime.
_NATIVE_DEFAULT_TENSOR_SPLIT_BUF = (ctypes.c_float * 16)()
_NATIVE_DEFAULT_TENSOR_SPLIT = ctypes.cast(
    _NATIVE_DEFAULT_TENSOR_SPLIT_BUF, ctypes.c_void_p).value


def _seeded_mp() -> SimpleNamespace:
    """A model-params SimpleNamespace seeded with the real native defaults for
    every field these tests care about (main_gpu, split_mode, tensor_split), so
    an untouched attribute reads back as the true native default rather than an
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
    mock_api.llama_n_ctx_seq.return_value = 2048   # must be int-comparable, not a bare Mock
    return mock_api


def _tensor_split_values(mp, count: int):
    """Read *count* floats back out of mp.tensor_split: cast the raw pointer to
    POINTER(c_float) and index in."""
    ptr = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
    return [ptr[i] for i in range(count)]


class TestLlamaCppGpuSplitWiring:
    """LlamaCpp.__init__ (localm/inference/backends/llamacpp/llama.py)."""

    def _build(self, mock_api):
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            # verbose=True skips the native-stderr-capture context managers.
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            # Close deterministically while `api` is still patched: _free_native
            # reads the module-global `api` name, not a stored instance attribute.
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

        # main_gpu keeps its existing main_gpu wiring behaviour.
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
        # the split devices [1, 2], so apply_gpu_split corrects it to the first
        # split device.
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


class TestIsolatedEmbedderGpuSplitPreflight:
    """IsolatedEmbedder._preflight_vram (localm/inference/embedder.py): the
    parent-side gate, which runs BEFORE a child is ever spawned. GGUFEmbedder
    .__init__ is the RAW native loader and is constructed only inside the
    isolated child process. gpu_split_shortfall() and list_gpus() computation
    run for real; only EmbedderRunner is stubbed, so no subprocess spawns."""

    class _StubRunner:
        spawned = False

        def spawn_and_load(self, params, timeout=None):
            type(self).spawned = True
            return {"dim": 768, "n_ctx": params.get("n_ctx") or 512}

    def _build(self, monkeypatch, model_file):
        type(self)._StubRunner.spawned = False
        monkeypatch.setattr(
            "localm.inference._embedder_runner.EmbedderRunner", self._StubRunner)
        from localm.inference.embedder import IsolatedEmbedder
        return IsolatedEmbedder(str(model_file))

    def test_split_configured_but_one_device_short_refuses(self, monkeypatch, tmp_path):
        """The embedder is a second, independent GGUF/llama.cpp load path, and
        it refuses - without spawning a child that could hard-abort - when a
        configured split device's own proportional share is short, small as
        embedding models are. Ratios are PINNED equal here: left unset, the auto
        free-VRAM-proportional split gives the tight device a near-zero share
        and the load proceeds instead."""
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"\0" * (2 * 1024 * 1024))   # 2 MB, realistic size
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0, "name": "A", "total": 1024, "free": 512},
                     {"index": 1, "name": "B", "total": 16 * 1024 ** 3,
                      "free": 16 * 1024 ** 3}])
        with pytest.raises(RuntimeError, match="configured split"):
            self._build(monkeypatch, model_file)
        # Refused BEFORE a child process was ever spawned.
        assert self._StubRunner.spawned is False

    def test_split_configured_with_enough_room_loads_normally(self, monkeypatch, tmp_path):
        """A split-configured embedder load whose file fits each device's free
        VRAM still proceeds to spawn."""
        model_file = tmp_path / "embed.gguf"
        model_file.write_bytes(b"\0" * (2 * 1024 * 1024))
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda: [{"index": 0, "name": "A", "total": 16 * 1024 ** 3, "free": 16 * 1024 ** 3},
                     {"index": 1, "name": "B", "total": 16 * 1024 ** 3, "free": 16 * 1024 ** 3}])
        e = self._build(monkeypatch, model_file)
        assert self._StubRunner.spawned is True
        assert e.dim == 768
