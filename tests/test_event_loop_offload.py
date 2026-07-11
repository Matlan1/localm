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
