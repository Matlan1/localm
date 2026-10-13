# SPDX-License-Identifier: AGPL-3.0-or-later
"""A worker-side input decode failure on a streamed chat turn carries
``localm_error`` with the mapped HTTP status, so a client can drop the input."""

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.backends.base import AudioInputError
from localm.inference.http_server import create_app


def _engine(tokens, exc):
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        yield from tokens
        raise exc

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 1
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "stop"
    engine.context_capacity.return_value = 4096
    type(engine).loaded = property(lambda self: True)
    return engine


def _chunks(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
            out.append(json.loads(line[5:].strip()))
    return out


def _post(engine):
    with TestClient(create_app(engine)) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "test-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    return _chunks(r.text)


def test_audio_decode_failure_before_any_token_carries_400():
    chunks = _post(_engine((), AudioInputError("The audio could not be decoded")))
    errs = [c["localm_error"] for c in chunks if "localm_error" in c]
    assert errs == [{"status": 400, "detail": "The audio could not be decoded"}]
    assert any("[inference error" in (c["choices"][0]["delta"].get("content") or "")
               for c in chunks if c.get("choices"))


def test_failure_after_tokens_stays_in_stream_only():
    chunks = _post(_engine(("part",), AudioInputError("late")))
    assert not [c for c in chunks if "localm_error" in c]


def test_unmapped_failure_carries_no_status():
    chunks = _post(_engine((), RuntimeError("boom")))
    assert not [c for c in chunks if "localm_error" in c]
