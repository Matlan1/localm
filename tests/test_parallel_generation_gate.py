# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-model admission for parallel slots: InferenceGate on its own, and the
real streaming / non-streaming chat coroutines running generations side by side
on one engine that reports parallel slots.

The engine stand-ins keep their finish reason per thread (PerThread), like the
GGUF backend: a reply's outcome read anywhere but on the thread that drove it
comes back as the default, so a response carrying the right finish reason
proves the outcome was read on the generating thread.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from localm.inference import residency
from localm.inference.backends.base import WAITING_FOR_MODEL_STATUS, PerThread
from localm.inference.http_server import _complete, _stream_sse
from localm.inference.inference_gate import InferenceGate, exclusively, set_capacity


@pytest.fixture(autouse=True)
def _clean_cancel_registry():
    residency._cancel_events.clear()
    yield
    residency._cancel_events.clear()


# ------------------------------------------------------------------ gate


def test_capacity_one_admits_one_holder_at_a_time():
    async def scenario():
        gate = InferenceGate()
        await gate.acquire()
        assert gate.locked()
        second = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        assert not second.done()
        gate.release()
        await asyncio.wait_for(second, 1)
        gate.release()
        assert not gate.locked()
    asyncio.run(scenario())


def test_shared_holders_up_to_capacity_then_wait():
    async def scenario():
        gate = InferenceGate(3)
        for _ in range(3):
            await asyncio.wait_for(gate.acquire(), 1)
        assert gate.active == 3 and gate.locked()
        fourth = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        assert not fourth.done()
        gate.release()
        await asyncio.wait_for(fourth, 1)
        assert gate.active == 3
    asyncio.run(scenario())


def test_exclusive_waits_for_every_holder_and_later_arrivals_wait_behind_it():
    async def scenario():
        gate = InferenceGate(2)
        await gate.acquire()
        await gate.acquire()
        order = []

        async def exclusive():
            async with gate.exclusive():
                order.append("exclusive")
                assert gate.active == 0
                await asyncio.sleep(0.02)
            order.append("exclusive-done")

        async def late():
            async with gate:
                order.append("late")

        ex = asyncio.ensure_future(exclusive())
        await asyncio.sleep(0.01)
        la = asyncio.ensure_future(late())
        await asyncio.sleep(0.01)
        assert order == []
        gate.release()
        await asyncio.sleep(0.01)
        assert order == []
        gate.release()
        await asyncio.wait_for(asyncio.gather(ex, la), 2)
        assert order == ["exclusive", "exclusive-done", "late"]
    asyncio.run(scenario())


def test_raising_capacity_admits_waiters():
    async def scenario():
        gate = InferenceGate(1)
        await gate.acquire()
        waiting = [asyncio.ensure_future(gate.acquire()) for _ in range(2)]
        await asyncio.sleep(0.01)
        assert not any(w.done() for w in waiting)
        set_capacity(gate, 3)
        await asyncio.wait_for(asyncio.gather(*waiting), 1)
        assert gate.active == 3
    asyncio.run(scenario())


def test_a_cancelled_waiter_leaves_no_trace():
    async def scenario():
        gate = InferenceGate(1)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        gate.release()
        assert not gate.locked() and gate.active == 0
        async with exclusively(gate):
            assert gate.locked()
        assert not gate.locked()
    asyncio.run(scenario())


def test_a_waiter_cancelled_after_its_grant_gives_the_place_back():
    async def scenario():
        gate = InferenceGate(1)
        await gate.acquire()
        waiter = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        gate.release()          # grants the waiter before it resumes
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert gate.active == 0 and not gate.locked()
    asyncio.run(scenario())


def test_a_plain_semaphore_is_used_as_it_is():
    sem = asyncio.Semaphore(1)
    assert exclusively(sem) is sem
    set_capacity(sem, 4)
    assert sem._value == 1


def test_release_without_acquire_raises():
    with pytest.raises(ValueError):
        InferenceGate().release()


# ------------------------------------------------------------------ chat paths


class _SlotEngine:
    """Engine stand-in with *slots* parallel slots. Each generation records how
    many ran at once, waits (up to 3 s) for *together* to be running, then
    yields its tokens and ends with *finish* (a per-thread value)."""

    last_finish_reason = PerThread("stop")

    def __init__(self, slots, together=1):
        self.display_name = "slot-model"
        self.parallel_slots = slots
        self.together = together
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0

    def count_messages_tokens(self, messages):
        return 3

    def count_tokens(self, text):
        return len(str(text).split())

    def context_capacity(self):
        return None

    def chat_stream(self, messages, **kwargs):
        content = messages[-1]["content"]
        finish = "length" if "long" in content else "stop"
        return self._stream(content, finish)

    def _stream(self, content, finish):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with self._lock:
                    if self.max_running >= self.together:
                        break
                time.sleep(0.005)
            for i in range(3):
                yield f"{content}{i} "
            self.last_finish_reason = finish
        finally:
            with self._lock:
                self.running -= 1


async def _sse(agen):
    chunks = []
    async for raw in agen:
        if raw.startswith("data: ") and raw.strip() != "data: [DONE]":
            chunks.append(json.loads(raw[len("data: "):]))
    return chunks


def _finish(chunks):
    reasons = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    return [r for r in reasons if r][-1]


def _statuses(chunks):
    return [c["choices"][0]["delta"].get("status") for c in chunks
            if c.get("choices") and c["choices"][0].get("delta", {}).get("status")]


def test_two_streams_generate_at_once_on_a_model_with_two_slots():
    async def scenario():
        eng = _SlotEngine(slots=2, together=2)
        gate = InferenceGate()
        a, b = await asyncio.gather(
            _sse(_stream_sse(eng, [{"role": "user", "content": "long"}], "slot-model", gate)),
            _sse(_stream_sse(eng, [{"role": "user", "content": "short"}], "slot-model", gate)))
        return eng, a, b
    eng, a, b = asyncio.run(scenario())
    assert eng.max_running == 2
    assert _finish(a) == "length"
    assert _finish(b) == "stop"


def test_one_slot_still_runs_streams_one_at_a_time_and_says_so():
    async def scenario():
        eng = _SlotEngine(slots=1, together=2)
        gate = InferenceGate()
        first = asyncio.ensure_future(
            _sse(_stream_sse(eng, [{"role": "user", "content": "long"}], "slot-model", gate)))
        await asyncio.sleep(0.05)
        second = await _sse(_stream_sse(eng, [{"role": "user", "content": "short"}],
                                        "slot-model", gate))
        return eng, await first, second
    eng, first, second = asyncio.run(scenario())
    assert eng.max_running == 1
    assert WAITING_FOR_MODEL_STATUS in _statuses(second)
    assert (_finish(first), _finish(second)) == ("length", "stop")


def test_non_streaming_replies_at_once_each_keep_their_finish_reason():
    async def scenario():
        eng = _SlotEngine(slots=2, together=2)
        gate = InferenceGate()
        return eng, await asyncio.gather(
            _complete(eng, [{"role": "user", "content": "long"}], "slot-model", gate),
            _complete(eng, [{"role": "user", "content": "short"}], "slot-model", gate))
    eng, (a, b) = asyncio.run(scenario())
    assert eng.max_running == 2

    def reason(resp):
        body = resp if isinstance(resp, dict) else json.loads(resp.body)
        return body["choices"][0]["finish_reason"]

    assert (reason(a), reason(b)) == ("length", "stop")
