# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat stream never sends the same status twice in a row: not when the
preparation outlasts the grace period and the reply starts with the status the
preparation last showed, and not when the backend repeats the status the reply
just sent. Each case runs with the stream opened early (grace 0) and late
(grace far above any preparation), so no wall-clock timing decides it."""

import json

import pytest
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.inference.http_server import create_app
from tests.test_stream_sse_status import _make_status_mock_engine


def _statuses(engine) -> list:
    client = TestClient(create_app(engine))
    out = []
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "Hello"}]}) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            line = raw if isinstance(raw, str) else raw.decode("utf-8")
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:]).get("choices", [{}])[0].get("delta", {})
            if delta.get("status"):
                out.append(delta["status"])
    return out


@pytest.fixture(params=[0.0, 600.0], ids=["opened-early", "opened-late"])
def grace(request, monkeypatch):
    monkeypatch.setattr(hs, "PREP_STATUS_GRACE_S", request.param)
    return request.param


def _no_repeats(seq):
    return all(a != b for a, b in zip(seq, seq[1:], strict=False))


def test_text_turn_sends_processing_prompt_once(grace):
    got = _statuses(_make_status_mock_engine(
        statuses=["Processing prompt...", "Generating response..."]))
    assert got == ["Processing prompt...", "Generating response..."]


def test_image_turn_keeps_the_backend_order_without_a_repeat(grace):
    got = _statuses(_make_status_mock_engine(
        statuses=["Encoding image (GPU)...", "Processing prompt...",
                  "Generating response..."]))
    assert got == ["Processing prompt...", "Encoding image (GPU)...",
                   "Processing prompt...", "Generating response..."]
    assert _no_repeats(got)


def test_the_existing_order_holds_in_both_paths(grace):
    got = _statuses(_make_status_mock_engine(
        statuses=["Encoding image (GPU)...", "Generating response..."]))
    assert got == ["Processing prompt...", "Encoding image (GPU)...",
                   "Generating response..."]


def test_status_of_line_reads_only_status_chunks():
    assert hs._status_of_line(
        'data: {"choices":[{"delta":{"status":"X"}}]}\n\n') == "X"
    assert hs._status_of_line(
        'data: {"choices":[{"delta":{"content":"the \\"status\\" word"}}]}\n\n') is None
    assert hs._status_of_line('data: {"choices":[]}\n\n') is None
    assert hs._status_of_line(": keepalive\n\n") is None
    assert hs._status_of_line("data: {not json\n\n") is None
