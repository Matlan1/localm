# SPDX-License-Identifier: AGPL-3.0-or-later
"""_maybe_auto_consolidate/_maybe_sweep_backfill spawn raw daemon threads
(_auto_consolidate_bg / _backfill_sweep_bg) that resolve the shared embedder via
_embed_fn() -> get_embedder(), same as every _off_loop-routed memory route. But
they never go through _off_loop, so they never acquire
http_server._get_embedder_sem() - the semaphore that bounds every _off_loop
caller to one concurrent default-pool worker inside get_embedder() (see
test_embeddings_pool_bound.py).

_auto_lock/_sweep_lock only dedupe a background pass against ANOTHER instance of
the SAME background pass; they say nothing about a concurrent, unrelated
embedder caller (an _off_loop route, or the other background pass), so they
cannot serialize this.

Three properties, one test each:
  (1) _backfill_sweep_bg reaches get_embedder() while an _off_loop caller is
      still parked inside it holding the semaphore - the bypass is REACHABLE,
      not just structurally possible.
  (2) _auto_consolidate_bg does the same, via its real call into
      synthesize_memory -> _embed_fn().
  (3) Even with the one semaphore-bound _off_loop caller AND both background
      threads concurrently blocked inside evict_chat_for_embedder, on a pool
      with only ONE spare worker beyond the bound caller,
      unload_all_models() still gets a free worker and every caller completes -
      the background threads occupy no default-pool slot, so they cannot
      reproduce the N-pool-workers-starve-the-pool deadlock the semaphore
      exists to prevent.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import localm.inference.http_server as hs
import localm.inference.embedder as emb_mod
from localm.vram import evict_chat_for_embedder
from localm.plugins.builtin.memory import plug


@pytest.fixture
def semclean():
    """A fresh, unheld embedder semaphore for every test - it is lazily created
    once per process and must not carry state (or a waiter) across tests."""
    saved = hs._embedder_sem
    hs._embedder_sem = None
    yield
    hs._embedder_sem = saved


class _FakeEmb:
    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]


def _make_tracking_get_embedder():
    """A get_embedder() stand-in that counts concurrent callers and blocks each
    one until released, mirroring the fakes in test_embeddings_pool_bound.py."""
    issued = threading.Semaphore(0)   # one release per caller that has arrived
    release = threading.Event()
    state = {"inflight": 0, "peak": 0}
    guard = threading.Lock()

    def fake_get_embedder(*, on_progress=None):
        with guard:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        issued.release()
        try:
            assert release.wait(10.0), "test never released get_embedder"
            return _FakeEmb()
        finally:
            with guard:
                state["inflight"] -= 1

    return fake_get_embedder, issued, release, state


# --------------------------------------------------------------------------- #
#  (1) + (2) reachability: a background thread reaches get_embedder() while    #
#      an _off_loop caller is still inside it holding the semaphore            #
# --------------------------------------------------------------------------- #

def _run_reachability(monkeypatch, spawn_bg_thread):
    """Shared harness: start one _off_loop(embedder_bound=True) caller, block it
    inside get_embedder(), then start *spawn_bg_thread* (a callable taking no
    args that starts and returns the background daemon Thread) and confirm it
    ALSO reaches get_embedder() while the first caller is still parked there."""
    fake_get_embedder, issued, release, state = _make_tracking_get_embedder()
    monkeypatch.setattr(emb_mod, "get_embedder", fake_get_embedder)

    async def _main():
        loop = asyncio.get_running_loop()
        off_loop_task = asyncio.ensure_future(
            plug._off_loop(lambda: plug._embed_fn()))

        # A bare issued.acquire() here would block THIS coroutine's own
        # thread - the event loop's only thread - synchronously, so the loop
        # never gets a tick to actually start off_loop_task and it
        # self-deadlocks. The wait itself must run off-loop.
        first = await loop.run_in_executor(None, lambda: issued.acquire(timeout=10.0))
        assert first, "the _off_loop caller never reached get_embedder()"

        t = spawn_bg_thread()

        # The background thread must reach get_embedder() too, WHILE the
        # _off_loop caller is still parked inside it (inflight == 2 momentarily).
        reached = await loop.run_in_executor(None, lambda: issued.acquire(timeout=10.0))
        peak = state["peak"]

        release.set()
        await loop.run_in_executor(None, lambda: t.join(10.0))
        await off_loop_task
        return reached, peak, t.is_alive()

    reached, peak, still_alive = asyncio.run(_main())
    assert reached, (
        "the background thread never reached get_embedder(); the semaphore "
        "bypass is not reachable after all")
    assert not still_alive, "the background thread never finished"
    assert peak == 2, (
        f"peak concurrent get_embedder() callers was {peak}, expected 2: the "
        "_off_loop caller (holding _get_embedder_sem()) and the background "
        "thread (which never acquires it)")


def test_backfill_sweep_bg_bypasses_the_embedder_semaphore(monkeypatch, semclean):
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setattr(
        "localm.memory.backfill.backfill_all",
        lambda root, embed_fn: {"embedded": 0, "namespaces": 0,
                                "remaining": 0, "unreadable": 0})

    def spawn():
        t = threading.Thread(target=plug._backfill_sweep_bg, daemon=True)
        t.start()
        return t

    _run_reachability(monkeypatch, spawn)


def test_auto_consolidate_bg_bypasses_the_embedder_semaphore(monkeypatch, semclean):
    class _FakeEngine:
        display_name = "fake-chat-model"
        loaded = True
        active_requests = 0

    eng = _FakeEngine()
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)

    def fake_synthesize_memory(complete, *, principal=None, **kw):
        # The real synthesize_memory resolves the embedder via _embed_fn()
        # (plug.py, "embed_fn = _embed_fn()") before touching the LLM at all;
        # this stub keeps that exact link real and skips only the
        # distillation/LLM machinery, which is not what is under test.
        plug._embed_fn()
        return {"added": 0}

    monkeypatch.setattr(plug, "synthesize_memory", fake_synthesize_memory)

    def spawn():
        t = threading.Thread(target=plug._auto_consolidate_bg, daemon=True)
        t.start()
        return t

    _run_reachability(monkeypatch, spawn)


# --------------------------------------------------------------------------- #
#  (3) the bypass does not starve the pool unload_all_models needs             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def hsclean():
    saved = hs.unload_all_models
    hs._server_loop = None
    yield
    hs.unload_all_models = saved
    hs._server_loop = None


def test_bg_threads_do_not_starve_the_pool_serving_unload(hsclean):
    """One _off_loop-style caller occupies the ONE default-pool worker the
    semaphore permits; two more raw threads (mirroring _auto_consolidate_bg and
    _backfill_sweep_bg) ALSO block inside evict_chat_for_embedder at the same
    time. With only one spare pool worker beyond the bound caller,
    unload_all_models() must still get one and every caller must complete -
    proving the background threads, which never touch the default pool, cannot
    reproduce the worker-starvation deadlock _get_embedder_sem() exists to
    prevent (test_dedicated_embed_path_holds_one_pool_worker_at_a_time)."""
    ran = {"unload_calls": 0}
    calls_lock = threading.Lock()

    async def _fake_unload():
        with calls_lock:
            ran["unload_calls"] += 1
        # Mirrors the real unload_all_models: it offloads its OWN step onto the
        # SAME default executor every caller here is blocked on.
        await asyncio.get_running_loop().run_in_executor(None, lambda: None)
        return {"status": "unloaded"}

    async def _main():
        loop = asyncio.get_running_loop()
        # 1 slot for the one semaphore-bound caller + exactly 1 spare for
        # unload_all_models's own step - the minimum that must still work.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        hs._server_loop = loop
        hs.unload_all_models = _fake_unload

        off_loop_status = loop.run_in_executor(
            None, lambda: evict_chat_for_embedder(timeout_s=5.0))

        results: dict = {}

        def _bg(name):
            results[name] = evict_chat_for_embedder(timeout_s=5.0)

        t1 = threading.Thread(target=_bg, args=("auto_consolidate",), daemon=True)
        t2 = threading.Thread(target=_bg, args=("backfill_sweep",), daemon=True)
        t0 = time.monotonic()
        t1.start()
        t2.start()

        status = await off_loop_status
        # A bare t.join() here would block the loop's own thread, preventing
        # it from ever running the _tracked_unload() coroutines t1/t2 are
        # themselves waiting on - a self-inflicted deadlock in the TEST, not
        # in the code under test. Join off-loop.
        await loop.run_in_executor(None, lambda: t1.join(10.0))
        await loop.run_in_executor(None, lambda: t2.join(10.0))
        return status, dict(results), time.monotonic() - t0, dict(ran)

    status, results, dt, ran_snapshot = asyncio.run(_main())
    assert dt < 8.0, f"the three concurrent evictions took {dt:.2f}s (starved?)"
    assert status == "unloaded", (
        f"the semaphore-bound caller did not complete cleanly: {status!r}")
    for name, r in results.items():
        assert r == "unloaded", (
            f"background-thread eviction {name!r} did not complete: {r!r}")
    assert ran_snapshot["unload_calls"] >= 1
