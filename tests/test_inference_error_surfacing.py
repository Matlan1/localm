# SPDX-License-Identifier: AGPL-3.0-or-later
"""An inference failure in a streaming SSE generator must reach the client as a
clean '[inference error: ...]' chunk, NOT crash the daemon generation thread
(which fires a crash report and looks to the user like an empty reply).

Regression guard for R16: the n_ctx-overflow RuntimeError (a conversation that
outgrew the context window) used to kill the /v1/completions generation thread -
its _generate() had a try/finally with no except, so the exception escaped the
thread. The chat path caught it but then called traceback.print_exc(), which was
the historical WinError-6 console crash source on Windows. Both paths must now
surface the error to the client and keep the server alive.
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
    # The /v1/completions streaming path had NO except: pre-fix the error was lost
    # (silent empty reply + a crashed daemon thread). It must now surface it too.
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
