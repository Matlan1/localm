# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grammar validation runs OFF the single event loop, on both chat routes.

``Engine.validate_grammar`` reaches the backend, and on the GGUF backend that is
an RPC to the isolated model worker: it waits on that worker's reply queue and
returns when the worker answers, or when the request times out. Called directly
from an async handler it holds the ONE event loop for that whole wait, so every
other in-flight request on the server stalls behind one caller's grammar check -
not just the request that asked for it.

The offload this pins is the same one the ``grammar_lazy`` trigger probe a few
lines above each call site already uses.

THE ASSERTION IS ON A THREAD IDENTITY, not on wall-clock time: a timing
assertion measures a proxy and is load-sensitive on a shared box. Thread
identity states the property directly - this call did not execute on the loop's
thread - and is deterministic.

THE ASSERTION IS MADE FROM OUTSIDE THE CALL: the route wraps
``validate_grammar`` in ``except GrammarUnsupportedError / InvalidGrammarError /
RuntimeError``, so a ``side_effect`` raising to signal "wrong thread" would be
an INPUT to the code under test and the RuntimeError arm would turn it into a
tidy 503, passing in both directions. The thread is RECORDED here and compared
after the response comes back.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm.inference.http_server import create_app


_GRAMMAR = 'root ::= "yes" | "no"'
_TEXT_MSG = [{"role": "user", "content": "hello"}]


def _engine_recording_validate_thread(seen: dict) -> MagicMock:
    """An engine whose ``validate_grammar`` records the thread it ran on.

    ``supports_grammar`` is True and the validator accepts: this test is about
    WHERE the check runs, not whether it refuses, so nothing here may take an
    early-return path that skips the call entirely.
    """
    engine = MagicMock()
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.supports_grammar = True
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = None
    type(engine).loaded = property(lambda self: True)
    engine.chat_stream.side_effect = lambda messages, **kw: iter(["ok"])

    def _record(grammar, lazy=False):
        seen["validate_thread"] = threading.get_ident()

    engine.validate_grammar.side_effect = _record
    return engine


def _app_recording_loop_thread(engine, seen: dict):
    """``create_app`` plus one middleware that records the event loop's thread.

    A middleware coroutine is awaited by the ASGI app ON the loop, so its
    ``get_ident()`` IS the loop thread - it does not depend on which engine
    methods the route happens to call directly, which a marker borrowed from
    (say) ``context_capacity`` would. If a later change offloaded that method
    too, this test would keep testing the right thing.
    """
    app = create_app(engine)

    @app.middleware("http")
    async def _capture(request, call_next):
        seen["loop_thread"] = threading.get_ident()
        return await call_next(request)

    return app


def _post(payload: dict, path: str) -> dict:
    seen: dict = {}
    engine = _engine_recording_validate_thread(seen)
    with TestClient(_app_recording_loop_thread(engine, seen)) as client:
        resp = client.post(path, json=payload)
    seen["status"] = resp.status_code
    return seen


class TestGrammarValidationIsOffloaded:
    def test_chat_completions_validates_grammar_off_the_event_loop(self):
        seen = _post(
            {"model": "test-model", "messages": _TEXT_MSG,
             "grammar": _GRAMMAR, "stream": False},
            "/v1/chat/completions")

        # The premise first: a test that never reached the validator would pass
        # the inequality below vacuously (both keys absent). Assert the call
        # HAPPENED before asserting anything about where.
        assert "validate_thread" in seen, (
            "validate_grammar was never called - this test cannot say anything "
            f"about where it runs (status {seen.get('status')})")
        assert "loop_thread" in seen, "the loop-thread marker never ran"
        assert seen["validate_thread"] != seen["loop_thread"], (
            "validate_grammar ran ON the event loop thread - its backend RPC "
            "would block every other request on this server for the length of "
            "the model worker's reply")

    def test_completions_validates_grammar_off_the_event_loop(self):
        # The SECOND call site. /v1/completions carries its own copy of this
        # block, so a fix applied only to /v1/chat/completions leaves the
        # server freezable through this route with every assertion above green.
        seen = _post(
            {"model": "test-model", "prompt": "hi",
             "grammar": _GRAMMAR, "stream": False},
            "/v1/completions")

        assert "validate_thread" in seen, (
            "validate_grammar was never called - this test cannot say anything "
            f"about where it runs (status {seen.get('status')})")
        assert "loop_thread" in seen, "the loop-thread marker never ran"
        assert seen["validate_thread"] != seen["loop_thread"], (
            "validate_grammar ran ON the event loop thread on /v1/completions - "
            "the same freeze as the chat route, through a different door")
