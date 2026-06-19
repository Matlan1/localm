# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for TTFT / throughput metrics in the usage field of HTTP responses."""

import json
import os

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from localm.inference.http_server import _tokens_per_sec, _ttft_ms, create_app


class TestMetricHelpers:
    def test_ttft_none_when_no_token(self):
        assert _ttft_ms(10.0, None) is None

    def test_ttft_milliseconds(self):
        assert _ttft_ms(10.0, 10.25) == 250.0

    def test_tokens_per_sec(self):
        assert _tokens_per_sec(50, 2.0) == 25.0

    def test_tokens_per_sec_none_when_zero_tokens(self):
        assert _tokens_per_sec(0, 2.0) is None

    def test_tokens_per_sec_none_when_zero_elapsed(self):
        assert _tokens_per_sec(50, 0.0) is None


def _make_engine():
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.count_tokens.return_value = 5
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["hello", " world"])
    type(engine).loaded = property(lambda self: True)
    return engine


CHAT_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


class TestUsageMetricsInResponses:
    def setup_method(self):
        os.environ.pop("LOCALM_API_KEY", None)

    def test_non_streaming_chat_has_tokens_per_sec(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert r.status_code == 200
        usage = r.json()["usage"]
        assert usage["tokens_per_sec"] is not None
        assert usage["tokens_per_sec"] > 0
        # TTFT is not observable without streaming
        assert usage["ttft_ms"] is None

    def test_streaming_chat_final_chunk_has_ttft(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post(
                "/v1/chat/completions",
                json={**CHAT_PAYLOAD, "stream": True},
            )
        assert r.status_code == 200
        # Find the usage-bearing chunk (the "done" chunk before [DONE])
        usage = None
        for line in r.text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[len("data: "):])
            if data.get("usage"):
                usage = data["usage"]
        assert usage is not None
        assert usage["ttft_ms"] is not None
        assert usage["ttft_ms"] >= 0
        assert usage["tokens_per_sec"] is not None

    def test_streaming_completions_final_chunk_has_metrics(self):
        with TestClient(create_app(_make_engine())) as client:
            r = client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "hi", "stream": True},
            )
        assert r.status_code == 200
        usage = None
        for line in r.text.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[len("data: "):])
            if data.get("usage"):
                usage = data["usage"]
        assert usage is not None
        assert usage["ttft_ms"] is not None
        assert usage["tokens_per_sec"] is not None
