# SPDX-License-Identifier: AGPL-3.0-or-later
"""get_embedder()/evict_chat_for_embedder must never freeze the server event
loop.

The async memory routes call get_embedder() SYNCHRONOUSLY on the event-loop
thread; get_embedder() -> _maybe_swap_for_embedder -> evict_chat_for_embedder,
which does
`run_coroutine_threadsafe(unload_all_models(), _server_loop).result(timeout=300)`.
When the caller is ALREADY on _server_loop's thread, the loop is blocked in
.result() and can never run the scheduled coroutine, so the whole server freezes
for up to 300s and the TimeoutError is then swallowed into "error" (a degraded
load).

The existing swap suite (test_embedder_vram_swap.py) cannot see this: its
harness `_drive_evict` runs evict on an EXECUTOR thread (off-loop), where
run_coroutine_threadsafe works. These tests drive the ON-loop path the real
routes use.

Two parts, one test each:
  (1) an on-loop guard in evict_chat_for_embedder (never block-wait on the loop
      it runs on);
  (2) the memory write routes resolve/load the embedder OFF the loop
      (run_in_executor).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import localm.inference.http_server as hs
from localm.vram import evict_chat_for_embedder
from localm.plugins.builtin.memory import plug


@pytest.fixture
def hsclean():
    saved = hs.unload_all_models
    hs._server_loop = None
    yield
    hs.unload_all_models = saved
    hs._server_loop = None


# --------------------------------------------------------------------------- #
#  (1) the on-loop guard: evict must not deadlock the loop it runs on          #
# --------------------------------------------------------------------------- #

def test_evict_on_loop_does_not_freeze_the_server(hsclean):
    """Calling evict_chat_for_embedder ON the server-loop thread (as an async
    memory route does via a synchronous get_embedder()) must return promptly, NOT
    block the whole loop for the timeout. Unguarded it blocks the full timeout,
    the unload coroutine never runs, and it returns 'error'."""
    ran = {"unload": False}

    async def _fake_unload():
        ran["unload"] = True
        return {"status": "unloaded"}

    async def _main():
        hs._server_loop = asyncio.get_running_loop()
        hs.unload_all_models = _fake_unload
        t0 = time.monotonic()
        status = evict_chat_for_embedder(timeout_s=3.0)  # called ON the loop thread
        return status, time.monotonic() - t0

    status, dt = asyncio.run(_main())
    assert dt < 1.5, (
        f"on-loop evict froze the event loop for {dt:.2f}s "
        f"(deadlock regression #648; should short-circuit)")
    # The guard degrades to loading the embedder alongside the chat model
    # ('skipped') instead of scheduling and blocking.
    assert status == "skipped", f"expected on-loop guard to skip, got {status!r}"
    assert ran["unload"] is False, (
        "the guard must not have run the cross-thread unload from the loop thread")


def test_evict_off_loop_still_evicts(hsclean):
    """The working off-loop path (executor thread / media-job path) must be
    UNCHANGED: evict still routes the unload through the loop and completes."""
    ran = {"unload": False}

    async def _fake_unload():
        ran["unload"] = True
        return {"status": "unloaded"}

    async def _main():
        hs._server_loop = asyncio.get_running_loop()
        hs.unload_all_models = _fake_unload
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        status = await loop.run_in_executor(
            None, lambda: evict_chat_for_embedder(timeout_s=3.0))
        return status, time.monotonic() - t0

    status, dt = asyncio.run(_main())
    assert dt < 1.5, f"off-loop evict should be fast, took {dt:.2f}s"
    assert status == "unloaded", f"off-loop evict must complete, got {status!r}"
    assert ran["unload"] is True, "off-loop evict must actually run the unload coroutine"


# --------------------------------------------------------------------------- #
#  (2) memory write routes resolve the embedder OFF the loop                   #
# --------------------------------------------------------------------------- #

def test_memory_append_resolves_embedder_off_the_loop(tmp_path, monkeypatch):
    """POST /api/memory/append must not resolve/load the embedder on the event
    loop: get_embedder()/embed() has to run on an executor thread, not inline on
    the loop thread with the store write."""
    from localm.memory import MemoryStore
    import localm.inference.embedder as emb_mod

    store = MemoryStore("owner", "chat", root=tmp_path)
    monkeypatch.setattr(plug, "_request_store", lambda request: store)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)

    embed_thread: dict = {}

    class _FakeEmb:
        def embed(self, texts):
            embed_thread["id"] = threading.get_ident()
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(emb_mod, "get_embedder", lambda: _FakeEmb())

    loop_thread: dict = {}

    async def _main():
        loop_thread["id"] = threading.get_ident()
        return await plug.memory_append(
            plug.MemoryAppend(text="remember the alamo"), request=None)

    res = asyncio.run(_main())
    assert res["status"] == "appended"
    assert embed_thread.get("id") is not None, "embed_fn was never invoked by the write"
    assert embed_thread["id"] != loop_thread["id"], (
        "get_embedder()/embed ran ON the event-loop thread "
        "(offload regression #648; the write must be offloaded to an executor)")
