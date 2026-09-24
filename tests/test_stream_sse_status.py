# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for live inference status chunks in the /v1/chat/completions and
/v1/completions SSE streams."""

import asyncio
import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.backends.base import VISION_CPU_FALLBACK_STATUS
from localm.inference.http_server import _stream_sse, create_app
from localm.inference.protocol import STATUS_CODE_BY_TEXT, WAITING_FOR_MODEL_STATUS


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


def _status_from_chunk(sse_line: str) -> str:
    payload = json.loads(sse_line[len("data: "):].strip())
    return payload["choices"][0]["delta"]["status"]


def test_queued_request_reports_waiting_not_processing_until_semaphore_frees():
    """A request that finds the per-model semaphore already held is queued,
    not processing: it gets a distinct waiting status, and the real
    "Processing prompt..." status must not appear until the semaphore is
    actually acquired."""
    engine = _make_status_mock_engine(statuses=["Generating response..."])
    sem = asyncio.Semaphore(1)

    async def _drive():
        await sem.acquire()
        gen = _stream_sse(engine, [{"role": "user", "content": "hi"}],
                          "test-model", sem, prompt_tokens=1)

        role_chunk = await gen.__anext__()
        assert '"role":"assistant"' in role_chunk.replace(" ", "")

        waiting_chunk = await gen.__anext__()
        assert _status_from_chunk(waiting_chunk) == WAITING_FOR_MODEL_STATUS

        next_task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        assert not next_task.done(), (
            "the generator produced a chunk before the held semaphore was "
            "released")

        sem.release()
        processing_chunk = await asyncio.wait_for(next_task, timeout=5)
        assert _status_from_chunk(processing_chunk) == "Processing prompt..."

        await gen.aclose()

    asyncio.run(_drive())


def test_unqueued_request_reports_processing_first_with_no_waiting_chunk():
    """The uncontended case (the common one): no waiting chunk at all, and
    the real initial status is still the second chunk overall."""
    engine = _make_status_mock_engine(statuses=["Generating response..."])
    sem = asyncio.Semaphore(1)

    async def _drive():
        gen = _stream_sse(engine, [{"role": "user", "content": "hi"}],
                          "test-model", sem, prompt_tokens=1)
        await gen.__anext__()   # role chunk
        status_chunk = await gen.__anext__()
        assert _status_from_chunk(status_chunk) == "Processing prompt..."
        await gen.aclose()

    asyncio.run(_drive())


def test_completions_stream_status_chunk_carries_status_and_code():
    """/v1/completions has no `delta`, so its status chunk lives directly on
    the choice, but it must still carry status_code the same way chat does -
    nothing currently reads this endpoint's stream, but the shape should not
    silently diverge from the one that is read."""
    engine = _make_status_mock_engine(statuses=["Generating response..."])
    app = create_app(engine)
    client = TestClient(app)

    payload = {"model": "test-model", "prompt": "hi", "stream": True}

    status_choices = []
    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            choice = chunk.get("choices", [{}])[0]
            if choice.get("status"):
                status_choices.append(choice)

    assert status_choices, "no status chunk was emitted at all"
    for choice in status_choices:
        assert "delta" not in choice, (
            "text completions has no delta field; status belongs directly "
            "on the choice")
        assert choice["status_code"] == STATUS_CODE_BY_TEXT.get(choice["status"]), (
            f"status_code must match the same table the chat endpoint uses: "
            f"{choice!r}")
