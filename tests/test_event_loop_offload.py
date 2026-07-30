# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blocking work must run in an executor, not on the single-threaded event loop.

Two break-it findings: running a full model generation (context compaction) or an
unbounded per-item tokenizer loop (embeddings usage) directly on the asyncio loop
freezes EVERY other request, the heartbeat, and the disconnect watchers for its
whole duration - the same event-loop-block class that PR #541 fixed for the GPU
probes. These tests pin that the work runs off the loop thread.
"""

from __future__ import annotations

import asyncio
import threading
import types

from localm.inference.http_server import _complete, _stream_sse


class _CompactEngine:
    """Engine near its context limit so the streaming/complete paths compact. It
    records the thread the COMPACTION generation (temperature=0.3) runs on."""

    display_name = "m"
    last_finish_reason = "stop"

    def __init__(self):
        self.compaction_thread = None

    def count_messages_tokens(self, messages):
        return 100

    def count_tokens(self, text):
        return 1

    def context_capacity(self):
        # buffer = max(2048, 12) = 2048; 128 - 100 = 28 < 2048 -> compaction fires.
        return 128

    def chat_stream(self, messages, **kw):
        if kw.get("temperature") == 0.3:      # the compaction summarization call
            self.compaction_thread = threading.current_thread()
            return iter(["a concise summary"])
        return iter(["t0 ", "t1 "])           # the real reply generation


# > KEEP_RECENT (4) non-system messages so compact_messages actually summarizes.
_MSGS = [{"role": "user", "content": f"turn {i}"} for i in range(12)]


def test_stream_sse_compaction_runs_off_the_event_loop():
    async def scenario():
        loop_thread = threading.current_thread()
        eng = _CompactEngine()
        sem = asyncio.Semaphore(1)
        async for _ in _stream_sse(eng, _MSGS, "m", sem):
            pass
        assert eng.compaction_thread is not None, "compaction never ran"
        assert eng.compaction_thread is not loop_thread, \
            "compaction generation ran ON the event loop (blocks every other request)"

    asyncio.run(scenario())


def test_complete_compaction_runs_off_the_event_loop():
    async def scenario():
        loop_thread = threading.current_thread()
        eng = _CompactEngine()
        sem = asyncio.Semaphore(1)
        await _complete(eng, _MSGS, "m", sem)
        assert eng.compaction_thread is not None, "compaction never ran"
        assert eng.compaction_thread is not loop_thread, \
            "compaction generation ran ON the event loop (complete path)"

    asyncio.run(scenario())


class _EmbEngine:
    """Embeddings engine that records the loop thread (via a _backend access the
    route makes on the loop) and the thread count_tokens runs on."""

    display_name = "m"
    active_requests = 0

    def __init__(self, seen):
        self._seen = seen

    @property
    def loaded(self):
        return True

    @property
    def _backend(self):
        # The route touches engine._backend (the can_embed check) synchronously on
        # the event loop, so this records the loop thread. can_embed=False keeps the
        # route from trying to reload a chat model.
        self._seen.setdefault("loop_thread", threading.current_thread())
        return types.SimpleNamespace(can_embed=False)

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def count_tokens(self, text):
        self._seen["count_thread"] = threading.current_thread()
        return 1


def test_embeddings_count_tokens_runs_off_the_event_loop():
    import os

    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app

    os.environ.pop("LOCALM_API_KEY", None)
    seen: dict = {}
    engine = _EmbEngine(seen)
    client = TestClient(create_app(engine), raise_server_exceptions=True)

    r = client.post("/v1/embeddings", json={"model": "m", "input": ["a", "b", "c"]})
    assert r.status_code == 200, r.text
    body = r.json()
    # Correctness preserved: one token per input.
    assert body["usage"]["total_tokens"] == 3, body
    # And the per-item tokenizer loop ran OFF the event loop.
    assert seen.get("loop_thread") is not None
    assert seen.get("count_thread") is not None
    assert seen["count_thread"] is not seen["loop_thread"], \
        "embeddings usage token-count ran ON the event loop (unbounded-input freeze)"


class _CompletionEngine:
    """Non-streaming /v1/completions engine that records the thread each
    count_tokens() call runs on. completions() calls count_tokens() twice per
    request (prompt_tokens at the top, completion_tokens after generation),
    both currently direct calls with no run_in_executor - unlike the
    embeddings path above and unlike grammar_triggers validation in the same
    function, which already do offload their own native/blocking calls.

    A plain-iterator chat_stream (iter([...])) is NOT enough here: _run() in
    http_server._generate_full calls gen.close() in a finally block, which a
    list_iterator does not have - it fails silently (caught and logged, not
    raised) but never lets the generation actually run to completion in the
    way a real generator would. Must be an actual generator function."""

    active_requests = 0

    def __init__(self, seen):
        self._seen = seen

    @property
    def display_name(self):
        # NOTE: do not use this to capture "the loop thread" - create_app()
        # itself reads display_name synchronously at APP CONSTRUCTION time
        # (registering the engine into _engines/_engines_lru), on whatever
        # thread calls create_app(), before any request exists. A
        # seen.setdefault(...) keyed off this property records THAT access,
        # not the per-request one - caught live while writing this test (the
        # first draft used exactly this and could not fail pre-fix).
        return "m"

    @property
    def loaded(self):
        return True

    def count_tokens(self, text):
        self._seen.setdefault("count_threads", []).append(threading.current_thread())
        return 1

    def chat_stream(self, messages, **kw):
        yield "hello"


def test_completions_count_tokens_runs_off_the_event_loop():
    import os
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import localm.inference.http_server as hs_mod

    os.environ.pop("LOCALM_API_KEY", None)
    seen: dict = {}
    engine = _CompletionEngine(seen)

    # _touch_activity(engine.display_name) is the first thing completions()
    # calls, synchronously, on the true per-request thread - before either of
    # the (possibly buggy) count_tokens calls. Recording via this call, not
    # via display_name itself, avoids the app-construction-time false read
    # documented on _CompletionEngine.display_name above.
    #
    # routes/chat.py's register() captures `_touch_activity = _hs._touch_activity`
    # as a LOCAL/closure variable at create_app() time - it is not a module
    # attribute of the chat routes module, so it can only be patched on
    # http_server (its real home) BEFORE create_app() runs register(), not
    # around the request afterward. Calls through to the real function so
    # activity tracking still happens.
    real_touch_activity = hs_mod._touch_activity

    def _recording_touch_activity(name=None):
        seen.setdefault("loop_thread", threading.current_thread())
        return real_touch_activity(name)

    with patch.object(hs_mod, "_touch_activity", side_effect=_recording_touch_activity):
        app = hs_mod.create_app(engine)
        client = TestClient(app, raise_server_exceptions=True)
        r = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": False})

    assert r.status_code == 200, r.text

    assert seen.get("loop_thread") is not None
    count_threads = seen.get("count_threads") or []
    # Both the prompt_tokens and completion_tokens calls must have happened.
    assert len(count_threads) == 2, (
        f"expected 2 count_tokens calls (prompt_tokens + completion_tokens), "
        f"got {len(count_threads)}")
    for t in count_threads:
        assert t is not seen["loop_thread"], (
            "completions token-count ran ON the event loop (blocks every other "
            "request for the duration of the native tokenizer call)")
