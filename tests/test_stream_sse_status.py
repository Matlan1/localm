# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for live inference status chunks in /v1/chat/completions SSE stream."""

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.backends.base import VISION_CPU_FALLBACK_STATUS
from localm.inference.http_server import create_app


def _make_status_mock_engine(statuses=None, supports_images=False):
    engine = MagicMock()
    _state = {"loaded": True}
    statuses = statuses or []

    def _chat_stream(messages, on_status=None, **kwargs):
        if on_status:
            for s in statuses:
                on_status(s)
        yield "Hello"
        yield " world"

    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = 4096
    engine.supports_images = supports_images
    engine.can_be_multimodal = supports_images
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


def test_stream_sse_emits_initial_and_live_status_chunks():
    engine = _make_status_mock_engine(
        statuses=["Encoding image (GPU)...", "Generating response..."],
    )
    app = create_app(engine)
    client = TestClient(app)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    statuses_received = []
    tokens_received = []

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "status" in delta and delta["status"]:
                statuses_received.append(delta["status"])
            if "content" in delta and delta["content"]:
                tokens_received.append(delta["content"])

    assert statuses_received == [
        "Processing prompt...",
        "Encoding image (GPU)...",
        "Generating response...",
    ]
    assert "".join(tokens_received) == "Hello world"


def test_stream_sse_image_initial_status():
    engine = _make_status_mock_engine(
        statuses=[VISION_CPU_FALLBACK_STATUS],
        supports_images=True,
    )
    app = create_app(engine)
    client = TestClient(app)

    payload = {
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        "stream": True,
    }

    statuses_received = []
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "status" in delta and delta["status"]:
                statuses_received.append(delta["status"])

    assert statuses_received == [
        "Encoding image...",
        VISION_CPU_FALLBACK_STATUS,
    ]


def test_stream_sse_status_chunks_carry_a_stable_code():
    engine = _make_status_mock_engine(
        statuses=[
            "Encoding image (GPU)...",
            VISION_CPU_FALLBACK_STATUS,
            "Generating response...",
        ],
        supports_images=True,
    )
    app = create_app(engine)
    client = TestClient(app)

    payload = {
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        "stream": True,
    }

    codes_by_status = {}
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("status"):
                codes_by_status[delta["status"]] = delta.get("status_code")

    assert codes_by_status == {
        "Encoding image...": "encoding_image",
        "Encoding image (GPU)...": "encoding_image_gpu",
        VISION_CPU_FALLBACK_STATUS: "vision_cpu_retry",
        "Generating response...": "generating",
    }
