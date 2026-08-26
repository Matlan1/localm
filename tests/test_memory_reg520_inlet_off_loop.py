# SPDX-License-Identifier: AGPL-3.0-or-later
"""The recall inlet must not do its blocking work on the event loop.

`recall(reinforce=True)` runs on EVERY non-privacy chat turn that recalls at
least one memory: it takes the namespace lock, `_load()`s the whole records
JSONL AND the `.vec.json` embedding sidecar, then `_save()`s both back
(store.py). With embeddings enabled that sidecar is hundreds of records x
embedding-dim floats - multiple MB of JSON parsed and re-serialised per turn.

`chat.py` does `await pipeline.run_inlet(...)`, and `chat_pipeline.py` calls
`entry.fn(messages, ctx)` INLINE, so a sync hook body executes on the loop
thread.

The inlet also resolves the shared embedder (`_embed_fn` -> `get_embedder`). A
VRAM swap (vram.py) submits a coroutine onto the server loop and BLOCKS the
calling thread waiting for it, so calling it FROM the loop thread would
deadlock; a guard detects that, SKIPS the guarded chat-model eviction, warns
"This call should be offloaded to an executor" and loads the embedder alongside
the resident chat model.

So the registered hook is an async wrapper that runs the unchanged, still-sync,
still directly testable `_memory_inlet` body through the `_off_loop` helper.
`run_inlet` already awaits an awaitable hook.
"""

from __future__ import annotations

import asyncio
import threading

from localm.plugins.builtin.memory import plug


class _Host:
    """Captures what register() actually wires up."""

    def __init__(self):
        self.hooks: dict = {}

    def mount_router(self, r):
        pass

    def register_chat_hook(self, kind, fn):
        self.hooks.setdefault(kind, []).append(fn)

    def __getattr__(self, name):
        return lambda *a, **kw: None


def _registered_inlet():
    host = _Host()
    try:
        plug.register(host)
    except Exception:
        # register() may touch optional wiring; the hook list is what matters.
        pass
    inlets = host.hooks.get("inlet") or []
    assert inlets, "the memory plugin registered no inlet hook"
    return inlets[0]


class TestInletHookIsOffLoaded:
    def test_the_registered_inlet_is_awaitable(self):
        """run_inlet awaits an awaitable hook (chat_pipeline.py), so an async hook
        is the seam that lets the body leave the loop thread. A plain sync hook is
        executed INLINE on the loop."""
        import inspect
        fn = _registered_inlet()
        assert inspect.iscoroutinefunction(fn), (
            "the inlet hook is registered as a plain sync function, so its "
            "blocking store I/O runs on the event loop")

    def test_the_blocking_body_runs_off_the_event_loop_thread(self, monkeypatch):
        """THE assertion. Records which thread the inlet body executes on and
        compares it to the loop's own thread."""
        body_thread: dict = {}
        loop_thread: dict = {}

        def _fake_body(messages, ctx):
            body_thread["id"] = threading.get_ident()
            return None

        monkeypatch.setattr(plug, "_memory_inlet", _fake_body)
        fn = _registered_inlet()

        async def _drive():
            loop_thread["id"] = threading.get_ident()
            await fn([{"role": "user", "content": "hi"}], None)

        asyncio.run(_drive())
        assert body_thread.get("id") is not None, "the inlet body never ran"
        assert body_thread["id"] != loop_thread["id"], (
            "the inlet body ran ON the event-loop thread: its per-turn "
            "_load()/_save() of the records JSONL + embedding sidecar (and its "
            "get_embedder() VRAM swap) block the whole server")

    def test_the_hook_returns_the_bodys_result_unchanged(self, monkeypatch):
        """NEGATIVE CASE: offloading must not change the contract - the hook
        returns the body's result unchanged."""
        sentinel = [{"role": "system", "content": "MEMORY BLOCK"}]
        monkeypatch.setattr(plug, "_memory_inlet", lambda m, c: sentinel)
        fn = _registered_inlet()
        out = asyncio.run(fn([{"role": "user", "content": "hi"}], None))
        assert out is sentinel

    def test_a_none_result_is_preserved(self, monkeypatch):
        monkeypatch.setattr(plug, "_memory_inlet", lambda m, c: None)
        fn = _registered_inlet()
        assert asyncio.run(fn([{"role": "user", "content": "hi"}], None)) is None

    def test_the_body_receives_the_real_arguments(self, monkeypatch):
        seen: dict = {}

        def _fake_body(messages, ctx):
            seen["messages"] = messages
            seen["ctx"] = ctx
            return None

        monkeypatch.setattr(plug, "_memory_inlet", _fake_body)
        fn = _registered_inlet()
        msgs = [{"role": "user", "content": "hello"}]
        marker = object()
        asyncio.run(fn(msgs, marker))
        assert seen["messages"] is msgs
        assert seen["ctx"] is marker


class TestSyncBodyStaysDirectlyTestable:
    """The sync body must remain importable and callable, so tests can drive
    plug._memory_inlet(...) directly instead of a coroutine."""

    def test_memory_inlet_is_still_a_plain_sync_callable(self):
        import inspect
        assert not inspect.iscoroutinefunction(plug._memory_inlet)

    def test_calling_it_directly_returns_a_value_not_a_coroutine(self):
        # The coder client short-circuits to None with no I/O - cheap and stable.
        class _Ctx:
            state = {"client_id": "coder"}

        assert plug._memory_inlet([{"role": "user", "content": "hi"}], _Ctx()) is None
