# SPDX-License-Identifier: AGPL-3.0-or-later
"""#953: the server "appears frozen" during "Synthesize now" - not appearance,
the event loop actually stalls. Measured in the reporter's log: `POST
/api/memory/consolidate -> 200 (4853 ms, loop_lag=5.34s)` - loop_lag EXCEEDS the
request duration, so the 2.5s /api/stats heartbeat that drives every live GUI
indicator was starved before the request even finished.

Every other mutating route in plug.py wraps its body in _off_loop (see BUG #648:
a memory write can resolve the shared embedder, which can trigger a VRAM swap
lasting minutes). memory_consolidate was the one exception: it drove a full
blocking LLM generation (complete() -> engine.chat_stream) straight on the
`async def` route body, with no `await` until the very end - so once started,
nothing else on the event loop gets a turn until it returns.

Fix: wrap the body in the existing _off_loop helper, same as every sibling route.

Negative case: without the fix, a concurrent asyncio task standing in for the
/api/stats heartbeat never gets scheduled while the blocking chat_stream call is
in flight, because the coroutine has no await point to yield at.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from localm.plugins.builtin.memory import plug


def _facts_reply():
    return json.dumps({"facts": [{"fact": "User's name is Ada", "confidence": 0.9}]})


def _seed_session(home):
    sdir = home / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "user", "data": {"content": "my name is Ada"}},
        {"type": "llm", "data": {"content": "Noted, Ada!"}},
    ]
    (sdir / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setenv("LOCALM_MODE", "log")
    _seed_session(tmp_path)
    return tmp_path


class _StubEngine:
    loaded = True

    def __init__(self, reply, display_name="stub-model"):
        self._reply = reply
        self.calls = 0
        self.caller_thread = None
        self.display_name = display_name
        self.active_requests = 0
        self.active_requests_during_call = None

    def chat_stream(self, messages, **kw):
        self.calls += 1
        self.caller_thread = threading.get_ident()
        self.active_requests_during_call = self.active_requests
        yield self._reply


class _SlowEngine:
    """A stand-in for a real chat_stream() call: blocks the calling thread until
    released, so the test can prove WHICH thread it ran on and whether the event
    loop kept serving other work while it was blocked."""
    loaded = True

    def __init__(self, started: threading.Event, release: threading.Event, reply: str):
        self._started = started
        self._release = release
        self._reply = reply
        self.caller_thread = None

    def chat_stream(self, messages, **kw):
        self.caller_thread = threading.get_ident()
        self._started.set()
        self._release.wait(10)
        yield self._reply


def test_consolidate_calls_the_model_off_the_event_loop_thread(home, monkeypatch):
    eng = _StubEngine(_facts_reply())
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)

    loop_thread = {}

    async def _drive():
        loop_thread["id"] = threading.get_ident()
        return await plug.memory_consolidate(request=None)

    result = asyncio.run(_drive())

    assert result.get("status") not in (None,), result
    assert eng.calls > 0, "consolidation never called the model - broken test setup"
    assert eng.caller_thread is not None
    assert eng.caller_thread != loop_thread["id"], (
        "memory_consolidate called the model's chat_stream() ON the event-loop "
        "thread - a blocking generation there starves /api/stats and every other "
        "request for the whole run (#953)")


def test_consolidate_pins_the_engine_busy_for_the_whole_call(home, monkeypatch):
    """F9: memory_consolidate's synthesize_memory call must be wrapped in
    driving_engine, or idle-unload cannot tell this route apart from a bare
    engine() access and can unload the model mid-consolidation on a quiet
    server. Checked from OUTSIDE, at the actual chat_stream() call, so a
    forgotten/misplaced wrap fails this test regardless of implementation
    detail."""
    from localm.inference import http_server as hs
    eng = _StubEngine(_facts_reply())
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)
    hs._last_activity_per_model.pop(eng.display_name, None)

    async def _drive():
        return await plug.memory_consolidate(request=None)

    asyncio.run(_drive())

    assert eng.calls > 0, "consolidation never called the model - broken test setup"
    assert eng.active_requests_during_call == 1, (
        "the model was not pinned busy (active_requests) during its own "
        "chat_stream call - a quiet server running only this route has "
        "nothing marking the model as in-use for idle-unload")
    assert eng.active_requests == 0, "the pin must be released once the call returns"
    assert eng.display_name in hs._last_activity_per_model, (
        "the per-model activity clock was never touched")


def test_event_loop_stays_responsive_while_consolidate_is_in_flight(home, monkeypatch):
    """Not just "did the loop get a turn eventually" (too easy to satisfy by
    accident, e.g. if the blocking call happens to start before the heartbeat's
    first tick lands, on wall-clock timing that says nothing about the fix) - hold
    the blocking call open for a fixed, generous window and confirm the heartbeat
    keeps ticking roughly on schedule DURING it. Pre-fix, the whole window is
    spent stuck inside the inline chat_stream() call with no await point to
    yield at, so nothing else - not even this test's own driver - runs until it
    returns; post-fix the executor thread runs it in parallel and the loop stays
    free the whole time."""
    started = threading.Event()
    release = threading.Event()
    eng = _SlowEngine(started, release, _facts_reply())
    monkeypatch.setattr(plug, "_live_engine", lambda: eng)

    async def _wait_until(cond, timeout=5.0, interval=0.01):
        deadline = time.monotonic() + timeout
        while not cond() and time.monotonic() < deadline:
            await asyncio.sleep(interval)
        return cond()

    HOLD_SECONDS = 0.3
    TICK_SECONDS = 0.02
    EXPECTED_TICKS = HOLD_SECONDS / TICK_SECONDS       # ~15 if the loop stays free

    async def _drive():
        ticks: list = []

        async def _heartbeat():
            # Stands in for the 2.5s /api/stats poll: as long as the loop is
            # free, this keeps ticking regardless of what else is running.
            while True:
                await asyncio.sleep(TICK_SECONDS)
                ticks.append(1)

        consolidate_task = asyncio.create_task(plug.memory_consolidate(request=None))
        heartbeat_task = asyncio.create_task(_heartbeat())

        reached_blocking_call = await _wait_until(started.is_set)
        # Hold the blocking call open for a fixed real-time window and let the
        # heartbeat run freely during it: the tick COUNT over that window is the
        # measurement, not whether any tick happened at all.
        await asyncio.sleep(HOLD_SECONDS)
        ticks_during_hold = len(ticks)

        release.set()
        heartbeat_task.cancel()
        await consolidate_task
        return reached_blocking_call, ticks_during_hold

    reached_blocking_call, ticks_during_hold = asyncio.run(_drive())

    assert reached_blocking_call, "consolidate never reached the blocking model call"
    # Generous lower bound (well under the ~15 ticks a free loop manages in
    # 0.3s) to absorb scheduling jitter, still far above the 0 ticks a frozen
    # loop would produce.
    assert ticks_during_hold >= EXPECTED_TICKS / 3, (
        f"only {ticks_during_hold} heartbeat tick(s) landed in a {HOLD_SECONDS}s "
        f"window where a free loop should manage ~{EXPECTED_TICKS:.0f} - the event "
        "loop was blocked for (most of) the duration of consolidate's blocking "
        "model call (#953)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
