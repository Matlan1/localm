# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.inference.engine.Engine - loaded property, auto-reload, idempotent load."""

import unittest
from unittest.mock import MagicMock, call, patch
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
