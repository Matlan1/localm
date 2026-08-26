# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for pre-dispatch context capacity guard and worker survival on context ceiling.

Two guards:
1. routes/chat.py checks counted prompt tokens against engine.context_capacity()
   before dispatch, rejecting oversized requests with HTTP 413 Payload Too Large
   instead of passing them to the native backend worker.
2. llamacpp/_runner.py catches ContextCapacityExceededError (raised by
   _fit_generation_budget) and marshals it across IPC as a typed error rather than
   letting the worker process crash with exit code 1 and evicting the loaded model.
"""

from typing import Iterator, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    BaseBackend,
    ContextCapacityExceededError,
)
from localm.inference.engine import Engine
from localm.inference.http_server import create_app


class _MockBackend(BaseBackend):
    def __init__(self, capacity: Optional[int] = 4096) -> None:
        self.effective_ctx_max = capacity
        self.n_ctx_max = capacity
        self.display_name = "test-cap-model"
        self._loaded = True
        self.last_finish_reason = "stop"
        self.unloaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded and not self.unloaded

    @property
    def supports_images(self) -> bool:
        return False

    def load(self) -> None:
        self._loaded = True
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True
        self._loaded = False

    def count_tokens(self, text: str) -> int:
        # 1 char = 1 token for deterministic testing
        return len(text)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages)

    def chat_stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        prompt_len = self.count_messages_tokens(messages)
        if self.effective_ctx_max and prompt_len > self.effective_ctx_max:
            raise ContextCapacityExceededError(
                f"Conversation ({prompt_len} tokens) has outgrown the maximum "
                f"context window (n_ctx_max={self.effective_ctx_max}). Start a new "
                f"chat, or raise it:  localm config n_ctx_max 32768  "
                f"(or set ctx_auto true to size it from free VRAM)."
            )
        yield "Hello "
        yield "world!"


class _EngineWithBackend(Engine):
    def __init__(self, backend: BaseBackend, display_name: str = "test-cap-model") -> None:
        from localm.inference.engine import _LOAD_LOCK
        self.model_path = "test-model.gguf"
        self.display_name = display_name
        self._load_lock = _LOAD_LOCK
        self._backend = backend
        self.active_requests = 0
        self.unloading = False


class TestPreDispatchContextCapacityGuard:
    """Pre-dispatch check in routes/chat.py must reject oversized prompts with 413."""

    def test_chat_completions_non_streaming_rejects_oversized_prompt(self):
        backend = _MockBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        # 150 characters = 150 tokens > 100 capacity
        oversized_content = "x" * 150
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-cap-model",
                "messages": [{"role": "user", "content": oversized_content}],
                "stream": False,
            },
        )
        assert resp.status_code == 413
        assert "150 tokens" in resp.json()["detail"]
        assert "100 tokens" in resp.json()["detail"]
        assert "localm config n_ctx_max" in resp.json()["detail"]
        assert backend.loaded is True
        assert backend.unloaded is False

    def test_chat_completions_streaming_rejects_oversized_prompt_before_stream(self):
        backend = _MockBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        oversized_content = "x" * 150
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-cap-model",
                "messages": [{"role": "user", "content": oversized_content}],
                "stream": True,
            },
        )
        # Rejected up front with 413, not 200 SSE stream
        assert resp.status_code == 413
        assert "150 tokens" in resp.json()["detail"]
        assert "100 tokens" in resp.json()["detail"]
        assert backend.loaded is True

    def test_completions_non_streaming_rejects_oversized_prompt(self):
        backend = _MockBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        oversized_prompt = "x" * 150
        resp = client.post(
            "/v1/completions",
            json={
                "model": "test-cap-model",
                "prompt": oversized_prompt,
                "stream": False,
            },
        )
        assert resp.status_code == 413
        assert "150 tokens" in resp.json()["detail"]
        assert "100 tokens" in resp.json()["detail"]
        assert backend.loaded is True

    def test_completions_streaming_rejects_oversized_prompt_before_stream(self):
        backend = _MockBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        oversized_prompt = "x" * 150
        resp = client.post(
            "/v1/completions",
            json={
                "model": "test-cap-model",
                "prompt": oversized_prompt,
                "stream": True,
            },
        )
        assert resp.status_code == 413
        assert "150 tokens" in resp.json()["detail"]
        assert "100 tokens" in resp.json()["detail"]
        assert backend.loaded is True

    def test_normal_prompt_within_capacity_succeeds(self):
        backend = _MockBackend(capacity=100)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        # 50 tokens <= 100 capacity
        normal_content = "x" * 50
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-cap-model",
                "messages": [{"role": "user", "content": normal_content}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Hello world!"

    def test_unbounded_model_without_known_capacity_allows_request(self):
        # Backend with no known context ceiling (e.g. proxy or unbounded backend)
        backend = _MockBackend(capacity=None)
        engine = _EngineWithBackend(backend)
        app = create_app(engine)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-cap-model",
                "messages": [{"role": "user", "content": "x" * 5000}],
                "stream": False,
            },
        )
        assert resp.status_code == 200


class TestWorkerNonCrashingOnContextOverflow:
    """Worker error envelope handling must not unload model on ContextCapacityExceededError."""

    def test_llama_fit_generation_budget_raises_context_capacity_exceeded_error(self):
        from localm.inference.backends.llamacpp.llama import LlamaCpp

        llm = object.__new__(LlamaCpp)
        llm._n_ctx_max = 4096

        # When n_prompt = 4686 > n_ctx_max, room = 4096 - 4686 - 64 = -654 < 32
        with pytest.raises(ContextCapacityExceededError) as excinfo:
            llm._fit_generation_budget(n_prompt=4686, max_new_tokens=1024)

        err_msg = str(excinfo.value)
        assert "4686 tokens" in err_msg
        assert "n_ctx_max=4096" in err_msg
        assert "localm config n_ctx_max 32768" in err_msg

    def test_gguf_backend_does_not_unload_on_context_capacity_exceeded(self):
        from localm.inference.backends.gguf import GgufBackend

        backend = object.__new__(GgufBackend)
        backend.effective_ctx_max = 4096
        backend._n_ctx_max = 4096
        backend._loaded = True
        backend._unloaded = False
        backend._first_token_timeout_seconds = lambda: 30.0

        mock_runner = MagicMock()
        mock_runner.is_alive.return_value = True

        def _raising_stream(**kwargs):
            raise ContextCapacityExceededError("Conversation outgrown context window")
            yield

        mock_runner.chat_stream.side_effect = _raising_stream
        backend._runner = mock_runner

        # chat_stream should raise ContextCapacityExceededError without calling backend.unload()
        with pytest.raises(ContextCapacityExceededError):
            list(backend.chat_stream(messages=[{"role": "user", "content": "hi"}]))

        # Model instance was preserved
        assert backend._loaded is True
        assert backend._runner is mock_runner
