# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wiring tests for main_gpu_index into the native llama.cpp model params, for
both native-load call sites: the chat backend (LlamaCpp) and the embedder
(GGUFEmbedder). Both route through localm.discover.apply_main_gpu, but each
constructs its own ``mp`` via a mocked ctypes API, so each call site is checked
end to end for actually SETTING mp.main_gpu."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from localm.inference.backends.llamacpp.llama import LlamaCpp


def _mock_llama_api():
    """A MagicMock standing in for the ctypes llama.cpp API. The model-params
    struct is a SimpleNamespace seeded with the real native default (main_gpu=0),
    so an untouched attribute reads back as that default and not as an
    auto-generated child mock."""
    mock_api = MagicMock()
    mp = SimpleNamespace(main_gpu=0, n_gpu_layers=0, use_mmap=True)
    mock_api.llama_model_default_params.return_value = mp
    return mock_api


class TestLlamaCppMainGpuWiring:
    """LlamaCpp.__init__ (localm/inference/backends/llamacpp/llama.py)."""

    def _build(self, mock_api):
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            # verbose=True skips the native-stderr-capture context managers,
            # which are unnecessary fd juggling against a fully mocked api.
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            # Close deterministically while `api` is still patched: LlamaCpp's
            # cleanup path (_free_native) reads the module-global `api` name,
            # not a stored instance attribute, so a GC-triggered __del__ after
            # this `with` exits would call the REAL native llama_free_model
            # with a MagicMock pointer.
            llm.close()
            return llm

    def test_configured_index_is_set_on_model_params(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mock_api = _mock_llama_api()
        self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 1

    def test_unconfigured_leaves_native_default_zero(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        mock_api = _mock_llama_api()
        self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 0

    def test_invalid_configured_index_falls_back_to_zero(self, monkeypatch, caplog):
        # Pin non-Vulkan so membership validation actually runs, whatever native
        # backend is provisioned in the ambient environment.
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 7})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        mock_api = _mock_llama_api()
        with caplog.at_level("WARNING", logger="localm"):
            self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)


class TestGgufEmbedderMainGpuWiring:
    """GGUFEmbedder.__init__ (localm/inference/embedder.py)."""

    def _mock_embed_api(self):
        mock_api = MagicMock()
        mp = SimpleNamespace(main_gpu=0, n_gpu_layers=0, use_mmap=True)
        mock_api.llama_model_default_params.return_value = mp
        mock_api.llama_model_n_embd.return_value = 768   # int(...) would TypeError on a bare Mock
        mock_api.has_embeddings_api.return_value = True
        mock_api.has_memory_api.return_value = False
        mock_api.llama_n_ctx_seq.return_value = 2048   # must be int-comparable, not a bare Mock
        return mock_api

    def _build(self, mock_api):
        # embedder.py imports _api locally each call ("from ...llamacpp import
        # _api as api"), which resolves via getattr on the already-imported
        # llamacpp PACKAGE object - patch that package attribute, not sys.modules.
        with patch("localm.inference.backends.llamacpp._api", mock_api):
            from localm.inference.embedder import GGUFEmbedder
            return GGUFEmbedder("embed.gguf")

    def test_configured_index_is_set_on_model_params(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mock_api = self._mock_embed_api()
        self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 1

    def test_unconfigured_leaves_native_default_zero(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        mock_api = self._mock_embed_api()
        self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 0

    def test_invalid_configured_index_falls_back_to_zero(self, monkeypatch, caplog):
        # Pin non-Vulkan so membership validation actually runs, whatever native
        # backend is provisioned in the ambient environment.
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 7})
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}])
        mock_api = self._mock_embed_api()
        with caplog.at_level("WARNING", logger="localm"):
            self._build(mock_api)
        assert mock_api.llama_model_default_params.return_value.main_gpu == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)
