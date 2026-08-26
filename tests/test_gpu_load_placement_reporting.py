# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /v1/models/load must not report unqualified success when the backend's
own sizing (_auto_gpu_layers, llamacpp/_sizing.py) settled on a partial or zero
GPU offload. A model too big to fully fit VRAM still loads, entirely on CPU or
split CPU/GPU, and the API caller has to be able to tell that apart from a full
GPU load.

GgufBackend records gpu_layers_offloaded/gpu_layers_total in _load_native, the
same way applied_gpu_split is recorded, so Engine.gpu_placement, switch_engine's
returned dict and the /v1/models/load response can all report it. This file
covers the backend-level recording: the arithmetic against the model's TRUE
layer count reported back by the native worker for this load.
"""

import asyncio
from unittest.mock import patch

from localm.inference.backends.gguf import GgufBackend
from localm.inference import http_server as hs


def _backend(tmp_path, *, n_gpu_layers=99, n_gpu_layers_auto=False):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return GgufBackend(str(f), n_gpu_layers=n_gpu_layers,
                        n_gpu_layers_auto=n_gpu_layers_auto, n_ctx=512)


def _load(backend, *, n_layers):
    """Drive _load_native() for real, stubbing only the two boundaries
    test_gpu_split_status_display.py already stubs for the same method: the
    VRAM-delta console print (no real GPU needed) and the isolated worker
    process (spawn_and_load), whose canned response reports the model's true
    layer count exactly as the real native worker would."""
    with patch("localm.discover.list_gpus", return_value=([], "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load",
               return_value={"n_layers": n_layers, "kv_bytes_per_token": 0,
                             "supports_images": False}):
        backend._load_native()


class TestBackendRecordsGpuPlacement:
    def test_full_offload_sentinel_resolves_to_true_total_not_degraded(self, tmp_path):
        # n_gpu_layers=99 ("everything") is the sentinel the loader hands
        # llama.cpp when the whole model fits; the recorded "offloaded" count is
        # the real total from the worker's response, not the raw sentinel.
        b = _backend(tmp_path, n_gpu_layers=99)
        b.effective_gpu_layers = 99
        _load(b, n_layers=32)
        assert b.gpu_layers_total == 32
        assert b.gpu_layers_offloaded == 32

    def test_auto_sized_partial_offload_recorded_against_true_total(self, tmp_path):
        # _auto_gpu_layers resolved a partial count (e.g. 12 of 32) before the
        # native call; that number, not the sentinel, is what llama.cpp was
        # handed.
        b = _backend(tmp_path, n_gpu_layers=99, n_gpu_layers_auto=True)
        b.effective_gpu_layers = 12
        _load(b, n_layers=32)
        assert b.gpu_layers_total == 32
        assert b.gpu_layers_offloaded == 12

    def test_zero_offload_recorded_as_zero_not_none(self, tmp_path):
        # The whole model ran on CPU: the concrete int 0, not None ("unknown").
        b = _backend(tmp_path, n_gpu_layers=99, n_gpu_layers_auto=True)
        b.effective_gpu_layers = 0
        _load(b, n_layers=32)
        assert b.gpu_layers_total == 32
        assert b.gpu_layers_offloaded == 0

    def test_explicit_partial_choice_also_recorded(self, tmp_path):
        # An explicit -g 24 (n_gpu_layers_auto off, or auto on with an explicit
        # non-99 value) is honoured verbatim by _effective_gpu_layers and is
        # reported the same way as the auto case.
        b = _backend(tmp_path, n_gpu_layers=24, n_gpu_layers_auto=False)
        b.effective_gpu_layers = 24
        _load(b, n_layers=32)
        assert b.gpu_layers_total == 32
        assert b.gpu_layers_offloaded == 24

    def test_offloaded_never_exceeds_true_total(self, tmp_path):
        # An effective_gpu_layers larger than the model's real layer count (e.g.
        # resolved against _ASSUMED_LAYERS) clamps to the true total.
        b = _backend(tmp_path, n_gpu_layers=99, n_gpu_layers_auto=True)
        b.effective_gpu_layers = 40   # larger than the true 32
        _load(b, n_layers=32)
        assert b.gpu_layers_total == 32
        assert b.gpu_layers_offloaded == 32

    def test_unknown_true_total_leaves_total_none(self, tmp_path, monkeypatch):
        # The worker's response carries no n_layers and nothing was cached from a
        # prior load, so gpu_layers_total stays None.
        monkeypatch.setattr(GgufBackend, "_cached_layer_count", lambda self: None)
        b = _backend(tmp_path, n_gpu_layers=99, n_gpu_layers_auto=True)
        b.effective_gpu_layers = 12
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load",
                   return_value={"kv_bytes_per_token": 0, "supports_images": False}):
            b._load_native()
        assert b.gpu_layers_total is None
        # Partial (below the "everything" sentinel) is an absolute count even
        # with no known total to compare it against.
        assert b.gpu_layers_offloaded == 12

    def test_unknown_true_total_and_full_sentinel_leaves_offloaded_none(
            self, tmp_path, monkeypatch):
        # "Everything" (99) was requested and the true count cannot be learned
        # this load, so there is nothing to report.
        monkeypatch.setattr(GgufBackend, "_cached_layer_count", lambda self: None)
        b = _backend(tmp_path, n_gpu_layers=99, n_gpu_layers_auto=False)
        b.effective_gpu_layers = 99
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load",
                   return_value={"kv_bytes_per_token": 0, "supports_images": False}):
            b._load_native()
        assert b.gpu_layers_total is None
        assert b.gpu_layers_offloaded is None


class _FakeEngine:
    """Minimal switch_engine-compatible Engine stand-in - the same shape as
    test_model_switch_preempt.py's FakeEngine, plus a settable gpu_placement
    (the real Engine exposes this as a property over its backend; a plain
    attribute here is equivalent for switch_engine's getattr-based read)."""

    def __init__(self, name, gpu_placement=None):
        self.display_name = name
        self._loaded = False
        self.gpu_placement = gpu_placement
        self.unloading = False

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, event):
        pass

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False


class TestSwitchEngineReportsGpuPlacement:
    """switch_engine (localm.inference.http_server) merges Engine.gpu_placement
    into its returned dict for both the "loaded" and "already_active" success
    statuses."""

    def setup_method(self):
        hs._engines = {}
        hs._engines_lru = []
        hs._active_model_name = None
        hs._engine = None
        hs._inference_sems = {}
        hs._switch_desired = None
        hs._switch_loading = None
        hs._switch_cancel = None

    def _make_engine_factory(self, gpu_placement):
        def factory(name):
            return _FakeEngine(name, gpu_placement=gpu_placement)
        return factory

    def test_fresh_load_carries_degraded_placement(self):
        placement = {"gpu_layers_offloaded": 12, "gpu_layers_total": 32,
                     "degraded": True}
        with patch("localm.config.load_registry", return_value={}):
            result = asyncio.run(hs.switch_engine(
                "model-a", self._make_engine_factory(placement)))
        assert result["status"] == "loaded"
        assert result["gpu_layers_offloaded"] == 12
        assert result["gpu_layers_total"] == 32
        assert result["degraded"] is True

    def test_fresh_load_omits_fields_when_placement_unknown(self):
        with patch("localm.config.load_registry", return_value={}):
            result = asyncio.run(hs.switch_engine(
                "model-a", self._make_engine_factory(None)))
        assert result["status"] == "loaded"
        assert "gpu_layers_offloaded" not in result
        assert "degraded" not in result

    def test_already_active_also_carries_placement(self):
        placement = {"gpu_layers_offloaded": 0, "gpu_layers_total": 32,
                      "degraded": True}
        factory = self._make_engine_factory(placement)
        with patch("localm.config.load_registry", return_value={}):
            asyncio.run(hs.switch_engine("model-a", factory))
            result = asyncio.run(hs.switch_engine("model-a", factory))
        assert result["status"] == "already_active"
        assert result["gpu_layers_offloaded"] == 0
        assert result["degraded"] is True
