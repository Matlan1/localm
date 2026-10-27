# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GUI chat surface sends a system message with every request; an
encoder-decoder model (T5) reads every message as input text, so it would answer
that message instead of the user. The chat route leaves system messages out of a
chat-surface request for such a model, and leaves every other request alone."""

from typing import Iterator

from fastapi.testclient import TestClient

from localm.inference.backends.base import BaseBackend
from localm.inference.engine import Engine
from localm.inference.http_server import create_app

SYSTEM = {"role": "system", "content": "You have no internet access. Never refer to yourself in the third person."}
USER = {"role": "user", "content": "Translate to German: good morning"}


class _Backend(BaseBackend):
    def __init__(self, encoder_decoder: bool) -> None:
        self.effective_ctx_max = 4096
        self.n_ctx_max = 4096
        self.display_name = "test-t5"
        self._loaded = True
        self.last_finish_reason = "stop"
        self.encoder_decoder = encoder_decoder
        self.seen: list[list[dict]] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def supports_images(self) -> bool:
        return False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def count_tokens(self, text: str) -> int:
        return len(text)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages)

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        self.seen.append([dict(m) for m in messages])
        yield "Guten Morgen"


class _Engine(Engine):
    def __init__(self, backend: BaseBackend) -> None:
        from localm.inference.engine import _LOAD_LOCK
        self.model_path = "test-t5.gguf"
        self.display_name = "test-t5"
        self._load_lock = _LOAD_LOCK
        self._backend = backend
        self.active_requests = 0
        self.unloading = False


def _ask(encoder_decoder: bool, headers: dict):
    backend = _Backend(encoder_decoder)
    client = TestClient(create_app(_Engine(backend)))
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "test-t5", "messages": [SYSTEM, USER], "stream": False})
    assert resp.status_code == 200, resp.text
    assert len(backend.seen) == 1
    return backend.seen[0]


def _roles(messages):
    return [m["role"] for m in messages]


def test_the_chat_surface_sends_an_encoder_decoder_model_no_system_message():
    seen = _ask(True, {"X-Client-Id": "chat"})
    assert "system" not in _roles(seen)
    assert seen[-1]["content"] == USER["content"]


def test_another_caller_keeps_its_system_message_for_an_encoder_decoder_model():
    seen = _ask(True, {})
    assert "system" in _roles(seen)


def test_the_chat_surface_keeps_the_system_message_for_an_ordinary_model():
    seen = _ask(False, {"X-Client-Id": "chat"})
    assert "system" in _roles(seen)
