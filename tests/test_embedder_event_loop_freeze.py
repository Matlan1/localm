# SPDX-License-Identifier: AGPL-3.0-or-later
"""A synchronous embedder.loaded_dim()/loaded_path() call on the event loop freezes the WHOLE server, not just its own request - a distinct hazard from the cross-thread deadlock covered by test_embedder_vram_swap.py."""

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
    """Hold embedder._LOCK on a background thread for *lock_hold_s* seconds, then run ``await make_awaitable()`` concurrently with a trivial coroutine on the same loop."""
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


def _offloaded(calls, fn):
    """True if `fn` (or a functools.partial wrapping it) is among the funcs handed to loop.run_in_executor. reset_embedder(force=False) is invoked as ``functools.partial(_embedder_mod.reset_embedder, force=False)`` at its two production call sites, not bare - see http_server.py."""
    return any(
        c is fn or (isinstance(c, functools.partial) and c.func is fn)
        for c in calls)


def test_unload_all_models_offloads_active_requests_check(hsclean, monkeypatch):
    """reset_embedder(force=False) is what now takes embedder._LOCK to decide the busy/idle question atomically (see embedder.reset_embedder's docstring: the old separate, unlocked active_requests() call before an unconditional reset_embedder() left a real TOCTOU window, so the check moved INSIDE reset_emb..."""
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
#  QA 2026-08-20: the SAME hazard, three more routes, found four weeks after   #
#  the tests above were written. The mechanism is unchanged - a cheap-looking  #
#  embedder reader taking _LOCK on an `async def` handler - so these reuse     #
#  _assert_event_loop_stays_responsive rather than inventing a new instrument. #
#                                                                             #
#  What made them survive the earlier pass is worth stating, because it is     #
#  the same sentence every time: each reader's docstring says "Does NOT        #
#  trigger a load - safe for a cheap status probe", and each of these three    #
#  handlers repeats that reassurance in its OWN docstring. It is true about    #
#  WORK and silent about WAITING, and the second is what freezes the server.   #
#  POST /api/embedding/warmup was measured in the field at 47s and climbing,   #
#  by localm's own hang alarm, with loaded_dim on top of the stack.            #
#                                                                             #
#  These are behavioural, not structural: they hold the real _LOCK and require #
#  an unrelated coroutine to still get its turn. A route that stops offloading #
#  fails them regardless of HOW it was written to offload.                     #
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
    """POST /api/embedding/warmup - QA #5, NEW-EMBEDDING-WARMUP-FREEZES-THE-EVENT-LOOP."""
    endpoint = _gui_endpoint("/api/embedding/warmup")

    # The route's `jobs` comes from its register() closure, not a module global,
    # so it is NOT patchable from here and the real JobManager runs. That is
    # fine and better - the assertion below is on the real contract - but the
    # background job it starts must not go off and attempt a genuine 300s load
    # once the lock holder releases, so the one thing that IS bound at call time
    # (the handler re-imports from localm.inference.embedder on every request)
    # is stubbed.
    monkeypatch.setattr("localm.plugins.gui.routes.models.principal_id",
                        lambda request: None)
    monkeypatch.setattr(emb, "get_embedder", lambda **kw: _FakeLoadedEmbedder())

    async def _drive():
        return await _assert_event_loop_stays_responsive(
            lambda: endpoint(request=None))

    result = asyncio.run(_drive())
    # A job is still started, and the route still answers with its id: the
    # offload moves WHO waits, it does not change the contract.
    assert isinstance(result.get("job_id"), str) and result["job_id"]


def test_rag_embedding_status_does_not_freeze_event_loop(monkeypatch):
    """GET /api/rag/embedding - the Knowledge page's own poll, so on a cold server it lands exactly while the first embedder load is running."""
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
    # Every field the client already relies on is still present and unchanged -
    # the fix moves WHO waits, never what is answered.
    assert result["dim"] is None
    assert result["status"] == "not_installed"
    assert "gpu_fallback_reason" in result and "error" in result
