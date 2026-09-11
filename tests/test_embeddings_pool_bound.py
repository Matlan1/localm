# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dedicated-embedder /v1/embeddings path must never park more than one
default-pool worker inside embed_texts, and an eviction abandoned by its
caller must not run later.

embed_texts -> get_embedder -> _maybe_swap_for_embedder ->
vram.evict_chat_for_embedder submits http_server.unload_all_models onto the
server loop and blocks the calling worker thread on the result. That coroutine
offloads its own steps onto the SAME default pool. With no bound, N concurrent
embedding requests on a cold embedder put N workers into that wait; once the
pool is exhausted nothing can run the unload and every worker sits the full
timeout, stalling all inference (chat generation shares the pool).

Two properties, one test each:
  (1) the route holds one default-pool worker at a time on this path;
  (2) evict_chat_for_embedder cancels the eviction it gave up waiting for.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.vram import evict_chat_for_embedder


@pytest.fixture
def app_client(monkeypatch):
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    app = hs.create_app(None)
    # Entered as a context manager so every request runs on the ONE portal
    # event loop, as on a real server. A bare TestClient.post spins up a loop
    # per call, and a loop-bound asyncio primitive cannot serialise across
    # loops.
    with TestClient(app) as client:
        yield client


def test_dedicated_embed_path_holds_one_pool_worker_at_a_time(app_client, monkeypatch):
    """Four concurrent requests on the dedicated path: at most ONE is inside
    embed_texts (on a default-pool worker) at any instant, and all four still
    succeed."""
    registry = {
        "embedding-bge-small-en-v1.5": {
            "path": "models/embeddings/bge-small-en-v1.5-q4_k_m.gguf",
            "source": "setup-embeddings",
            "model_type": "embedding",
        },
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": "bge-small-en-v1.5"})

    n_requests = 4
    issued = threading.Semaphore(0)     # one release per request that reached embed
    release = threading.Event()         # set only once all four have been issued
    state = {"inflight": 0, "peak": 0}
    guard = threading.Lock()

    def fake_embed_texts(texts):
        with guard:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        issued.release()
        try:
            assert release.wait(10.0), "the test never released the embed"
            return [[0.0] for _ in texts]
        finally:
            with guard:
                state["inflight"] -= 1

    monkeypatch.setattr("localm.inference.embedder.embed_texts", fake_embed_texts)

    results = [None] * n_requests

    def _post(i):
        results[i] = app_client.post(
            "/v1/embeddings", json={"model": "bge-small-en-v1.5", "input": f"text {i}"})

    threads = [threading.Thread(target=_post, args=(i,)) for i in range(n_requests)]
    for t in threads:
        t.start()
    # Wait until the FIRST request is inside embed, then give the other three
    # time to arrive at the route; only then let the embeds run.
    assert issued.acquire(timeout=10.0), "no request reached embed_texts"
    import time
    time.sleep(0.5)
    release.set()
    for t in threads:
        t.join(20.0)
    assert not any(t.is_alive() for t in threads), "a request never returned"

    assert state["peak"] == 1, (
        f"{state['peak']} default-pool workers were inside embed_texts at once; "
        "the dedicated path must be bounded to one")
    for i, r in enumerate(results):
        assert r is not None and r.status_code == 200, (i, getattr(r, "text", r))
        assert r.json()["data"][0]["embedding"] == [0.0]


@pytest.fixture
def hsclean():
    saved = hs.unload_all_models
    hs._server_loop = None
    yield
    hs.unload_all_models = saved
    hs._server_loop = None


def test_abandoned_eviction_is_cancelled_when_its_caller_gives_up(hsclean):
    """One caller on a 1-worker default pool waits for an unload that itself
    needs that worker: it times out with "error" (the documented mechanism).
    The unload it abandoned must NOT run once the worker frees up: the caller
    cancels it and waits for the loop to apply the cancellation before its
    worker is released."""
    ran: list = []

    async def _fake_unload():
        await asyncio.get_running_loop().run_in_executor(None, lambda: ran.append("step"))
        return {"status": "unloaded"}

    async def _main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        hs._server_loop = loop
        hs.unload_all_models = _fake_unload
        status = await loop.run_in_executor(
            None, lambda: evict_chat_for_embedder(timeout_s=0.5))
        await asyncio.sleep(0.3)   # let the freed worker drain anything still queued
        return status

    status = asyncio.run(_main())
    assert ran == [], (
        f"an eviction abandoned by its caller still ran afterwards: {ran}")
    assert status == "error", status

