# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.inference.engine.Engine - loaded property, auto-reload, idempotent load."""

import unittest
from unittest.mock import MagicMock, patch


def _make_mock_backend(loaded: bool = False):
    """Return a mock BaseBackend-alike with controllable loaded state."""
    backend = MagicMock()
    _state = {"loaded": loaded}

    def _load():
        _state["loaded"] = True

    def _unload():
        _state["loaded"] = False

    def _chat_stream(messages, **kwargs):
        yield "hello"

    backend.load.side_effect = _load
    backend.unload.side_effect = _unload
    backend.chat_stream.side_effect = _chat_stream
    type(backend).loaded = property(lambda self: _state["loaded"])
    return backend


class TestEngineLoadedProperty(unittest.TestCase):
    def _make_engine(self, loaded=False):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = _make_mock_backend(loaded=loaded)
        return engine

    def test_loaded_false_initially(self):
        engine = self._make_engine(loaded=False)
        self.assertFalse(engine.loaded)

    def test_loaded_true_after_load(self):
        engine = self._make_engine(loaded=False)
        engine.load()
        self.assertTrue(engine.loaded)

    def test_loaded_false_after_unload(self):
        engine = self._make_engine(loaded=True)
        engine.unload()
        self.assertFalse(engine.loaded)


class TestEngineLoadIdempotency(unittest.TestCase):
    def _make_engine(self, loaded=False):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = _make_mock_backend(loaded=loaded)
        return engine

    def test_load_twice_only_calls_backend_once(self):
        engine = self._make_engine(loaded=False)
        engine.load()
        engine.load()   # second call - already loaded, should be no-op
        engine._backend.load.assert_called_once()

    def test_load_when_already_loaded_is_noop(self):
        engine = self._make_engine(loaded=True)
        engine.load()
        engine._backend.load.assert_not_called()


class TestEngineAutoReload(unittest.TestCase):
    def _make_engine(self, loaded=False):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = _make_mock_backend(loaded=loaded)
        return engine

    @patch("localm.inference.engine.load_config", return_value={
        "max_tokens": 512, "temperature": 0.7, "top_p": 0.9,
        "top_k": 40, "repeat_penalty": 1.1,
    })
    def test_chat_stream_auto_reloads_when_unloaded(self, _cfg):
        engine = self._make_engine(loaded=False)
        tokens = list(engine.chat_stream([{"role": "user", "content": "hi"}]))
        engine._backend.load.assert_called_once()
        self.assertEqual(tokens, ["hello"])

    @patch("localm.inference.engine.load_config", return_value={
        "max_tokens": 512, "temperature": 0.7, "top_p": 0.9,
        "top_k": 40, "repeat_penalty": 1.1,
    })
    def test_chat_stream_does_not_reload_when_loaded(self, _cfg):
        engine = self._make_engine(loaded=True)
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))
        engine._backend.load.assert_not_called()

    @patch("localm.inference.engine.load_config", return_value={
        "max_tokens": 512, "temperature": 0.7, "top_p": 0.9,
        "top_k": 40, "repeat_penalty": 1.1,
    })
    def test_chat_stream_after_unload_auto_reloads(self, _cfg):
        """Unload then call chat_stream - should auto-reload transparently."""
        engine = self._make_engine(loaded=True)
        engine.unload()
        self.assertFalse(engine.loaded)
        list(engine.chat_stream([{"role": "user", "content": "hi"}]))
        engine._backend.load.assert_called_once()
        self.assertTrue(engine.loaded)


class TestEngineContextManager(unittest.TestCase):
    def _make_engine(self):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = _make_mock_backend(loaded=False)
        return engine

    @patch("localm.inference.engine.load_config", return_value={
        "max_tokens": 512, "temperature": 0.7, "top_p": 0.9,
        "top_k": 40, "repeat_penalty": 1.1,
    })
    def test_context_manager_loads_and_unloads(self, _cfg):
        engine = self._make_engine()
        with engine:
            self.assertTrue(engine.loaded)
        self.assertFalse(engine.loaded)


class TestEngineLoadSerialization(unittest.TestCase):
    """Model loads must serialise process-wide, even across separate Engine
    instances (server chat vs a background job), so two loads never compete for
    VRAM at once."""

    def _slow_backend(self, state, lock):
        """A backend whose load() takes ~40ms and records peak concurrency."""
        import time
        backend = MagicMock()
        _loaded = {"v": False}

        def _load():
            with lock:
                state["concurrent"] += 1
                state["max"] = max(state["max"], state["concurrent"])
            time.sleep(0.04)
            _loaded["v"] = True
            with lock:
                state["concurrent"] -= 1

        backend.load.side_effect = _load
        type(backend).loaded = property(lambda self: _loaded["v"])
        return backend

    def _make_engine(self, state, lock):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = self._slow_backend(state, lock)
        return engine

    def test_loads_serialise_across_engine_instances(self):
        import threading
        state = {"concurrent": 0, "max": 0}
        counter_lock = threading.Lock()   # guards the counter only
        engines = [self._make_engine(state, counter_lock) for _ in range(4)]
        threads = [threading.Thread(target=e.load) for e in engines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # The process-global _LOAD_LOCK must keep loads one-at-a-time.
        self.assertEqual(state["max"], 1,
                         "model loads must serialise process-wide")
        for e in engines:
            self.assertTrue(e.loaded)


class TestEngineGpuPlacement(unittest.TestCase):
    """Engine.gpu_placement: whether the last load's transformer layers ended
    up on GPU or CPU, so a caller can tell a full GPU load from a CPU
    fallback."""

    def _make_engine(self, *, gpu_layers_offloaded=None, gpu_layers_total=None):
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        backend = MagicMock()
        backend.gpu_layers_offloaded = gpu_layers_offloaded
        backend.gpu_layers_total = gpu_layers_total
        engine._backend = backend
        return engine

    def test_none_before_any_load(self):
        engine = self._make_engine()
        self.assertIsNone(engine.gpu_placement)

    def test_none_when_backend_has_no_layer_count_knob(self):
        # An HF backend (device_map="auto") has no gpu_layers_offloaded/total
        # attributes at all - getattr's default keeps this None rather than
        # fabricating a placement claim it cannot back up.
        from localm.inference.engine import Engine
        engine = object.__new__(Engine)
        engine.model_path = "/fake/model.gguf"
        engine.display_name = "fake-model"
        engine._backend = object()   # no gpu_layers_* attributes at all
        self.assertIsNone(engine.gpu_placement)

    def test_full_offload_is_not_degraded(self):
        engine = self._make_engine(gpu_layers_offloaded=32, gpu_layers_total=32)
        self.assertEqual(engine.gpu_placement, {
            "gpu_layers_offloaded": 32, "gpu_layers_total": 32,
            "degraded": False,
        })

    def test_partial_offload_is_degraded(self):
        engine = self._make_engine(gpu_layers_offloaded=12, gpu_layers_total=32)
        placement = engine.gpu_placement
        self.assertEqual(placement["gpu_layers_offloaded"], 12)
        self.assertEqual(placement["gpu_layers_total"], 32)
        self.assertTrue(placement["degraded"])

    def test_zero_offload_is_degraded(self):
        engine = self._make_engine(gpu_layers_offloaded=0, gpu_layers_total=32)
        self.assertTrue(engine.gpu_placement["degraded"])

    def test_zero_total_is_treated_as_unknown_not_degraded(self):
        # A total of 0 (or None) means the true layer count is unknowable,
        # not that the model has zero layers - never divide/compare against
        # a total we do not actually have.
        engine = self._make_engine(gpu_layers_offloaded=0, gpu_layers_total=0)
        self.assertIsNone(engine.gpu_placement)


if __name__ == "__main__":
    unittest.main()
