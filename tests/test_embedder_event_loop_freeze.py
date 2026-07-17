# SPDX-License-Identifier: AGPL-3.0-or-later
"""A synchronous embedder.loaded_dim()/loaded_path() call on the event loop
freezes the WHOLE server, not just its own request - a distinct hazard from
the cross-thread deadlock covered by test_embedder_vram_swap.py.

Root cause: since #643 (subprocess-isolate the GGUF embedder), get_embedder()
can hold embedder._LOCK for the full duration of an IsolatedEmbedder native/
subprocess load (up to its load timeout - 300s in production), on WHATEVER
thread called it (a RAG-indexing executor thread, an embedding-setup job
thread, ...). loaded_dim()/loaded_path() both also acquire that same _LOCK.
Three call sites this PR added called one of them directly on an async
route's coroutine instead of via loop.run_in_executor(), unlike every other
blocking call in those same functions:
  - http_server.unload_all_models() -> loaded_dim()
  - http_server._unload_embedder_if_matches() -> loaded_path()
  - gui/routes/models.py's GET /api/models -> loaded_path()

A synchronous call blocked on _LOCK inside a coroutine blocks asyncio's
single-threaded event loop entirely - not just that one request, EVERY other
in-flight or incoming request on the same server, for as long as the lock is
held. Confirmed via adversarial review (2026-07-14) with a live reproduction
(a lock held for 2s blocked a concurrent loaded_dim() call for ~1.78s of that
window). Fixed by wrapping each call in loop.run_in_executor(), matching this
codebase's own established pattern for every other blocking call in these
same functions.

These tests reproduce the mechanism directly: hold embedder._LOCK on a
background thread (standing in for an in-progress IsolatedEmbedder load,
however long it takes) and assert a TRIVIAL, unrelated coroutine scheduled
concurrently on the SAME event loop still gets to run promptly - proving the
call under test yielded the loop instead of blocking it outright.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from localm.inference import embedder as emb
import localm.inference.http_server as hs


@pytest.fixture(autouse=True)
def _reset_embedder():
    emb.reset_embedder()
    yield
    emb.reset_embedder()


@pytest.fixture
def hsclean():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    yield
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None


async def _assert_event_loop_stays_responsive(make_awaitable, *, lock_hold_s=2.0):
    """Hold embedder._LOCK on a background thread for *lock_hold_s* seconds,
    then run ``await make_awaitable()`` concurrently with a trivial coroutine
    on the same loop. Fails loudly if the trivial coroutine does not get to
    run promptly - proof the call under test blocked the WHOLE event loop, not
    just its own task."""
    hold_started = threading.Event()
    release_lock = threading.Event()

    def _hold():
        with emb._LOCK:
            hold_started.set()
            release_lock.wait(timeout=lock_hold_s + 5)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert hold_started.wait(timeout=2), "background lock holder never started"

    trivial_ran_at = []

    async def _trivial():
        # A genuinely-blocked loop cannot advance even a no-op coroutine
        # through its own await points.
        for _ in range(3):
            await asyncio.sleep(0)
        trivial_ran_at.append(time.monotonic())

    t0 = time.monotonic()
    trivial_task = asyncio.ensure_future(_trivial())
    main_task = asyncio.ensure_future(make_awaitable())
    try:
        await asyncio.wait_for(trivial_task, timeout=lock_hold_s * 0.5)
    except asyncio.TimeoutError:
        release_lock.set()
        holder.join(timeout=5)
        pytest.fail(
            "a concurrent trivial coroutine never got to run while the call "
            "under test held embedder._LOCK-contended work on the event "
            "loop itself - the whole server would freeze, not just this "
            "request, for as long as an embedder load takes")
    trivial_elapsed = trivial_ran_at[0] - t0

    release_lock.set()
    result = await asyncio.wait_for(main_task, timeout=10)
    holder.join(timeout=5)

    assert trivial_elapsed < lock_hold_s * 0.5, (
        f"a concurrent trivial coroutine took {trivial_elapsed:.2f}s to run "
        f"(lock held for {lock_hold_s}s) - the event loop was blocked")
    return result


def test_unload_all_models_does_not_freeze_event_loop(hsclean, monkeypatch):
    monkeypatch.setattr("localm.discover.vram_capacity", lambda: {"free": None})
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)

    async def _drive():
        return await _assert_event_loop_stays_responsive(hs.unload_all_models)

    result = asyncio.run(_drive())
    assert result["status"] == "already_unloaded"  # nothing was actually loaded


def test_unload_embedder_if_matches_does_not_freeze_event_loop(hsclean, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry", lambda: {})

    async def _drive():
        loop = asyncio.get_running_loop()
        return await _assert_event_loop_stays_responsive(
            lambda: hs._unload_embedder_if_matches("some-model", loop))

    result = asyncio.run(_drive())
    assert result is None  # no embedder was resident, so no match either


def test_gui_models_route_does_not_freeze_event_loop(hsclean, monkeypatch):
    """Same hazard, the GET /api/models route (gui/routes/models.py)."""
    from localm.plugins.gui.web import attach_gui
    from fastapi import FastAPI

    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
              switch_model=lambda name: None, active_model=lambda: None)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/models")
    monkeypatch.setattr("localm.config.load_registry", lambda: {})

    async def _drive():
        return await _assert_event_loop_stays_responsive(lambda: endpoint(type=""))

    result = asyncio.run(_drive())
    assert result == {"models": [], "active": None}


# --------------------------------------------------------------------------- #
#  embedder.active_requests() (added by #650's pin fix) has the IDENTICAL     #
#  _LOCK-blocking hazard as loaded_dim()/loaded_path() above - it must also   #
#  be executor-offloaded at both call sites that gate an embedder release on  #
#  it. Structural check (does the call go through run_in_executor), not a    #
#  full timing simulation: the mechanism is identical to what the timing     #
#  tests above already prove for the same lock; this confirms the THIRD      #
#  accessor was not missed when #650's pin check was merged in.              #
# --------------------------------------------------------------------------- #

class _FakeLoadedEmbedder:
    dim = 384
    active_requests = 0
    model_path = "/fake/embedder.gguf"

    def close(self):
        pass


def _recording_run_in_executor(loop, calls):
    real = loop.run_in_executor

    def _wrapped(executor, func, *args):
        calls.append(func)
        return real(executor, func, *args)

    return _wrapped


def test_unload_all_models_offloads_active_requests_check(hsclean, monkeypatch):
    monkeypatch.setattr("localm.discover.vram_capacity", lambda: {"free": None})
    monkeypatch.setattr(hs, "_gpu_registry_sync", lambda: None)
    emb._EMBEDDER = _FakeLoadedEmbedder()

    async def _drive():
        loop = asyncio.get_running_loop()
        calls = []
        monkeypatch.setattr(loop, "run_in_executor", _recording_run_in_executor(loop, calls))
        result = await hs.unload_all_models()
        return result, calls

    result, calls = asyncio.run(_drive())
    assert result["embedder_unloaded"] is True
    assert emb.active_requests in calls, (
        "active_requests() must run via loop.run_in_executor, like loaded_dim() "
        "just above it - a direct call reintroduces the event-loop-freeze "
        "hazard this file's other tests already prove for the same lock")


def test_unload_embedder_if_matches_offloads_active_requests_check(hsclean, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"embed-model": {"path": "/fake/embedder.gguf"}})
    emb._EMBEDDER = _FakeLoadedEmbedder()

    async def _drive():
        loop = asyncio.get_running_loop()
        calls = []
        monkeypatch.setattr(loop, "run_in_executor", _recording_run_in_executor(loop, calls))
        result = await hs._unload_embedder_if_matches("embed-model", loop)
        return result, calls

    result, calls = asyncio.run(_drive())
    assert result["status"] == "unloaded"
    assert emb.active_requests in calls, (
        "active_requests() must run via loop.run_in_executor - same hazard as "
        "loaded_path() just above it in this same function")
