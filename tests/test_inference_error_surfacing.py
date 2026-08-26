# SPDX-License-Identifier: AGPL-3.0-or-later
"""An inference failure in a streaming SSE generator must reach the client as a
clean '[inference error: ...]' chunk, NOT crash the daemon generation thread
(which fires a crash report and looks to the user like an empty reply).

The n_ctx-overflow RuntimeError (a conversation that outgrew the context window)
is the case this guards: /v1/completions' _generate() has a try/finally with no
except, so the exception escapes the thread, and the chat path's own catch called
traceback.print_exc(), the WinError-6 console crash source on Windows. Both paths
must surface the error to the client and keep the server alive.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

_OVERGROWN = (
    "Conversation (122000 tokens) has outgrown the maximum context window "
    "(n_ctx_max=65536). Start a new chat, or raise it."
)


def _raising_engine():
    """A loaded text-only engine whose chat_stream raises on first iteration."""
    engine = MagicMock()
    _state = {"loaded": True}
    engine.load.side_effect = lambda: _state.__setitem__("loaded", True)

    def _chat_stream(messages, **kwargs):
        raise RuntimeError(_OVERGROWN)
        yield  # pragma: no cover - makes this a generator function

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 7
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "error"
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


def test_chat_completions_surfaces_inference_error():
    engine = _raising_engine()
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "[inference error" in r.text
    assert "outgrown" in r.text


def test_raw_completions_surfaces_inference_error():
    # The /v1/completions streaming path surfaces the error too.
    engine = _raising_engine()
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "hi",
            "stream": True,
        })
    assert r.status_code == 200
    assert "[inference error" in r.text
    assert "outgrown" in r.text


def test_server_survives_error_and_serves_next_request():
    # After a generation error the server must still answer (no process/thread death).
    engine = _raising_engine()
    with TestClient(create_app(engine)) as client:
        bad = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": True})
        assert bad.status_code == 200
        # A plain liveness probe still works.
        health = client.get("/health")
        assert health.status_code == 200


# --------------------------------------------------------------------------- #
# The TERMINAL frame's finish_reason says "error", not "stop": a programmatic
# client keys off finish_reason and would otherwise treat the visible error
# chunk as a successful completion.
# --------------------------------------------------------------------------- #

import json   # noqa: E402


def _terminal_finish_reason(sse_text: str):
    """The last non-null choices[].finish_reason across all SSE data frames."""
    reasons = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        for ch in obj.get("choices", []):
            fr = ch.get("finish_reason")
            if fr is not None:
                reasons.append(fr)
    return reasons[-1] if reasons else None


def _yielding_engine(tokens=("hel", "lo"), reason="stop"):
    """A loaded engine whose chat_stream yields *tokens* then ends with *reason*."""
    engine = MagicMock()
    _state = {"loaded": True}

    def _chat_stream(messages, **kwargs):
        for t in tokens:
            yield t

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = len(tokens)
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = reason
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


def test_chat_stream_error_marks_finish_reason_error():
    # last_finish_reason is "stop", so a pass proves the HTTP layer sets "error"
    # from gen_error rather than echoing what the engine left behind.
    engine = _raising_engine()
    engine.last_finish_reason = "stop"
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert _terminal_finish_reason(r.text) == "error"


def test_completions_stream_error_marks_finish_reason_error():
    engine = _raising_engine()
    engine.last_finish_reason = "stop"
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": True,
        })
    assert r.status_code == 200
    assert _terminal_finish_reason(r.text) == "error"


def test_chat_stream_success_keeps_stop():
    # The happy path still reports "stop": the error override does not leak into
    # a clean generation.
    engine = _yielding_engine(reason="stop")
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
    assert r.status_code == 200
    assert "[inference error" not in r.text
    assert _terminal_finish_reason(r.text) == "stop"


def test_completions_stream_success_keeps_stop():
    engine = _yielding_engine(reason="stop")
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "hi", "stream": True,
        })
    assert r.status_code == 200
    assert "[inference error" not in r.text
    assert _terminal_finish_reason(r.text) == "stop"


# --------------------------------------------------------------------------- #
# The NON-STREAMING path (_generate_full): a generation failure surfaces as
# 200, the error as content, and finish_reason "error", matching the streaming
# path rather than raising a raw HTTP 500.
# --------------------------------------------------------------------------- #

def test_chat_completions_nonstream_surfaces_inference_error():
    engine = _raising_engine()
    engine.last_finish_reason = "stop"   # prove the HTTP layer sets "error", not the engine
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert r.status_code == 200          # NEVER a raw 500
    choice = r.json()["choices"][0]
    assert "[inference error" in choice["message"]["content"]
    assert "outgrown" in choice["message"]["content"]
    assert choice["finish_reason"] == "error"


def test_chat_completions_nonstream_success_keeps_stop():
    # The error override must not leak into a clean non-streaming generation.
    engine = _yielding_engine(reason="stop")
    with TestClient(create_app(engine)) as client:
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert "[inference error" not in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
