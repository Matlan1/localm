# SPDX-License-Identifier: AGPL-3.0-or-later
"""A synchronous embedder.loaded_dim()/loaded_path() call on the event loop
freezes the WHOLE server, not just its own request - a distinct hazard from
the cross-thread deadlock covered by test_embedder_vram_swap.py.

get_embedder() can hold embedder._LOCK for the full duration of an
IsolatedEmbedder native/subprocess load (up to its load timeout, 300s in
production), on WHATEVER thread called it - a RAG-indexing executor thread, an
embedding-setup job thread. loaded_dim() and loaded_path() acquire that same
_LOCK, so a synchronous call to either inside a coroutine blocks asyncio's
single-threaded event loop entirely: not just that request, EVERY other
in-flight or incoming request, for as long as the lock is held. Each such call
therefore goes through loop.run_in_executor().

These tests reproduce the mechanism directly: hold embedder._LOCK on a
background thread (standing in for an in-progress IsolatedEmbedder load) and
assert a TRIVIAL, unrelated coroutine scheduled concurrently on the SAME event
loop still gets to run promptly.
"""

from __future__ import annotations

import asyncio
import functools
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
    on the same loop. Fails when the trivial coroutine does not get to run
    promptly, which means the call under test blocked the WHOLE event loop
    rather than just its own task."""
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
#  embedder.active_requests() takes the same _LOCK as loaded_dim() and         #
#  loaded_path() above, so it is executor-offloaded at both call sites that    #
#  gate an embedder release on it. Structural check: the call goes through     #
#  run_in_executor.                                                            #
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


def _offloaded(calls, fn):
    """True if `fn` (or a functools.partial wrapping it) is among the funcs
    handed to loop.run_in_executor. reset_embedder(force=False) is invoked as
    ``functools.partial(_embedder_mod.reset_embedder, force=False)`` at its
    two production call sites, not bare - see http_server.py."""
    return any(
        c is fn or (isinstance(c, functools.partial) and c.func is fn)
        for c in calls)


def test_unload_all_models_offloads_active_requests_check(hsclean, monkeypatch):
    """reset_embedder(force=False) takes embedder._LOCK to decide the busy/idle
    question atomically, so it runs via loop.run_in_executor; a direct call
    would block the event loop on the same lock."""
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
    assert _offloaded(calls, emb.reset_embedder), (
        "reset_embedder() must run via loop.run_in_executor - a direct call "
        "reintroduces the event-loop-freeze hazard this file's other tests "
        "already prove for the same lock")


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
    assert _offloaded(calls, emb.reset_embedder), (
        "reset_embedder() must run via loop.run_in_executor - same hazard as "
        "loaded_path() just above it in this same function")


# --------------------------------------------------------------------------- #
#  The same hazard on more routes: a cheap-looking embedder reader taking      #
#  _LOCK on an `async def` handler. These checks hold the real _LOCK and       #
#  require an unrelated coroutine to still get its turn.                       #
# --------------------------------------------------------------------------- #

def _gui_endpoint(path: str, method: str = "POST"):
    from fastapi import FastAPI
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None, active_model=lambda: None)
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"no {method} {path} route was mounted")


def test_embedding_warmup_does_not_freeze_event_loop(hsclean, monkeypatch):
    """POST /api/embedding/warmup: its first statement is a loaded_dim()
    fast-path check ("already warm?"), which against a running load waits for
    that whole load."""
    endpoint = _gui_endpoint("/api/embedding/warmup")

    # The route's `jobs` comes from its register() closure, not a module global,
    # so the real JobManager runs. The handler re-imports from
    # localm.inference.embedder on every request, so that is stubbed to keep the
    # background job from attempting a genuine load.
    monkeypatch.setattr("localm.plugins.gui.routes.models.principal_id",
                        lambda request: None)
    monkeypatch.setattr(emb, "get_embedder", lambda **kw: _FakeLoadedEmbedder())

    async def _drive():
        return await _assert_event_loop_stays_responsive(
            lambda: endpoint(request=None))

    result = asyncio.run(_drive())
    # A job is still started and the route still answers with its id.
    assert isinstance(result.get("job_id"), str) and result["job_id"]


def test_rag_embedding_status_does_not_freeze_event_loop(monkeypatch):
    """GET /api/rag/embedding - the Knowledge page's own poll, so on a cold
    server it lands while the first embedder load is running. Three readers in
    one handler (loaded_dim, last_error, gpu_fallback_reason), all on the same
    lock."""
    from localm.plugins.builtin.rag import plug as ragplug

    endpoint = None
    for route in ragplug._router.routes:
        if getattr(route, "path", None) == "/api/rag/embedding" \
                and "GET" in getattr(route, "methods", ()):
            endpoint = route.endpoint
    assert endpoint is not None, "GET /api/rag/embedding is not mounted"

    monkeypatch.setattr("localm.inference.http_server.caller_scopes",
                        lambda request: None)
    monkeypatch.setattr("localm.config.load_config", lambda *a, **k: {})
    monkeypatch.setattr("localm.config.load_registry", lambda *a, **k: {})
    monkeypatch.setattr("localm.inference.embedder.resolve_embedding_model_path",
                        lambda **k: None)

    async def _drive():
        return await _assert_event_loop_stays_responsive(
            lambda: endpoint(request=None))

    result = asyncio.run(_drive())
    # Every field the client relies on is still present and unchanged.
    assert result["dim"] is None
    assert result["status"] == "not_installed"
    assert "gpu_fallback_reason" in result and "error" in result
