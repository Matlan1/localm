# SPDX-License-Identifier: AGPL-3.0-or-later
"""SlotScheduler against a simulated llama.cpp context.

``_SimContext`` keeps real per-sequence KV cells, refuses a batch whose
positions do not continue a sequence or that overflows the cells, and its
model's next token depends only on the tokens of the sequence it is computed
for. So every reply decoded beside others must equal the same reply decoded
alone (``_reference``): a sequence id, position or row mixed up anywhere in the
scheduler changes the tokens.
"""
from __future__ import annotations

import contextlib
import threading
import time

import pytest

from localm.inference.backends.base import WAITING_FOR_MODEL_STATUS
from localm.inference.backends.llamacpp import _slots
from localm.inference.backends.llamacpp._slots import (
    RESERVE_PAD, UNLIMITED_STEP, UNLOADED_MESSAGE, SlotScheduler)

EOG = 0
VOCAB = 97


def _next_token(tokens):
    h = 0
    for t in tokens:
        h = (h * 131 + t + 7) % 1_000_003
    return 1 + h % (VOCAB - 1)


def _reference(prompt, budget, eog_at=None):
    tokens = list(prompt)
    out = []
    while budget <= 0 or len(out) < budget:
        if eog_at is not None and len(out) == eog_at:
            return out, "stop"
        tok = _next_token(tokens)
        out.append(tok)
        tokens.append(tok)
    return out, "length"


class _Sampler:
    """Picks the computed token; returns EOG once *eog_at* tokens were picked."""

    def __init__(self, eog_at=None, raise_at=None):
        self.eog_at = eog_at
        self.raise_at = raise_at
        self.picked = 0

    def pick(self, token):
        if self.raise_at is not None and self.picked == self.raise_at:
            raise OSError("sampler fault")
        if self.eog_at is not None and self.picked == self.eog_at:
            return EOG
        self.picked += 1
        return token


class _SimContext:
    def __init__(self, capacity=4096, n_ctx_max=None, base=None, grow=256,
                 vram_fits=None, partial_rm=True):
        self._lock = threading.RLock()
        self.cap = capacity
        self.base = base if base is not None else capacity
        self.max = n_ctx_max
        self.grow = grow
        self.vram_fits = vram_fits
        self.partial_rm = partial_rm
        self.cells = {}
        self.rows = {}
        self.calls = []
        self.freed = []
        self.recreated = []
        self.cleared = 0
        self.seq_rms = []
        self.stop = threading.Event()
        self.fail = None          # entries -> rc, or None to decode normally
        self.step_delay = 0.0

    # SlotScheduler's ops seam
    def lock(self):
        return self._lock

    def stopped(self):
        return self.stop.is_set()

    def capacity(self):
        return self.cap

    def max_capacity(self):
        return self.max

    def target_ctx(self, needed):
        t = -(-needed // self.grow) * self.grow
        t = max(self.base, t)
        return min(t, self.max) if self.max else t

    def vram_fit(self, target):
        return self.vram_fits

    def recreate(self, target, offload_kqv):
        self.recreated.append((target, offload_kqv))
        self.cap = target
        self.cells = {}

    def n_batch(self):
        return min(self.cap, 2048)

    def decode(self, entries):
        self.calls.append(list(entries))
        if self.step_delay:
            time.sleep(self.step_delay)
        if self.fail is not None:
            rc = self.fail(entries)
            if rc:
                return rc
        if sum(len(v) for v in self.cells.values()) + len(entries) > self.cap:
            return 1
        staged = {s: list(v) for s, v in self.cells.items()}
        rows = {}
        for i, (tok, pos, seq, want) in enumerate(entries):
            cur = staged.setdefault(seq, [])
            if pos != len(cur):
                return -1
            cur.append(tok)
            if want:
                rows[i] = _next_token(cur)
        self.cells = staged
        self.rows = rows
        return 0

    def sample(self, sampler, row):
        return sampler.pick(self.rows[row])

    def is_eog(self, token):
        return token == EOG

    def seq_rm(self, seq, p0, p1):
        self.seq_rms.append((seq, p0, p1))
        if p0 > 0 and not self.partial_rm:
            return False
        cur = self.cells.get(seq, [])
        self.cells[seq] = cur[:p0] if p0 > 0 else []
        return True

    def clear_memory(self):
        self.cleared += 1
        self.cells = {}

    def free_sampler(self, sampler):
        self.freed.append(sampler)

    def score(self, scorer, row, token):
        return scorer.score(row, token)

    def stderr_scope(self):
        return contextlib.nullcontext()


def _drain(stream):
    out = []
    for tok in stream:
        out.append(tok)
    return out, stream.finish_reason


def _collect(sched, prompt, budget, sampler=None, results=None, key=None, **kw):
    sampler = sampler or _Sampler()
    statuses = []
    stream = sched.submit(prompt, budget, sampler, on_status=statuses.append, **kw)
    out, reason = _drain(stream)
    rec = {"out": out, "reason": reason, "statuses": statuses, "sampler": sampler}
    if results is not None:
        results[key] = rec
    return rec


def _run_threads(targets, timeout=30):
    threads = [threading.Thread(target=t, daemon=True) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "a consumer did not finish"


PROMPTS = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3, 5, 8], [2, 7, 1, 8], [1, 6, 1, 8, 3, 3]]


@pytest.fixture
def made():
    scheds = []

    def make(ctx, n_slots):
        s = SlotScheduler(ctx, n_slots)
        scheds.append(s)
        return s
    yield make
    for s in scheds:
        s.close(timeout=5)


def test_concurrent_replies_each_equal_the_reply_decoded_alone(made):
    ctx = _SimContext()
    ctx.step_delay = 0.002
    sched = made(ctx, 4)
    with ctx.lock():
        streams = [sched.submit(PROMPTS[i], 40, _Sampler()) for i in range(4)]
    results = {}
    _run_threads([lambda i=i: results.__setitem__(i, _drain(streams[i]))
                  for i in range(4)])
    for i, prompt in enumerate(PROMPTS):
        assert results[i] == _reference(prompt, 40)
    assert any(len({e[2] for e in call}) > 1 for call in ctx.calls), \
        "no decode carried more than one sequence"


def test_more_requests_than_slots_never_decode_more_sequences_than_slots(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    results = {}
    prompts = PROMPTS + [[4, 4, 4], [5, 1]]
    _run_threads([lambda i=i: _collect(sched, prompts[i], 12, results=results, key=i)
                  for i in range(len(prompts))])
    for i, prompt in enumerate(prompts):
        assert results[i]["out"] == _reference(prompt, 12)[0]
    assert max(len({e[2] for e in call}) for call in ctx.calls) <= 2
    assert {e[2] for call in ctx.calls for e in call} <= {0, 1}


def test_end_of_generation_stops_without_yielding_the_end_token(made):
    sched = made(_SimContext(), 2)
    rec = _collect(sched, PROMPTS[0], 50, sampler=_Sampler(eog_at=5))
    assert (rec["out"], rec["reason"]) == _reference(PROMPTS[0], 50, eog_at=5)
    assert EOG not in rec["out"]
    assert rec["statuses"][:1] == [_slots.GENERATING_STATUS]


def test_every_sampler_is_freed_exactly_once(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    samplers = [_Sampler(eog_at=3), _Sampler(), _Sampler()]
    _run_threads([lambda s=s, i=i: _collect(sched, PROMPTS[i], 6, sampler=s)
                  for i, s in enumerate(samplers)])
    assert sorted(map(id, ctx.freed)) == sorted(map(id, samplers))


def test_a_request_that_does_not_fit_waits_for_the_running_reply(made):
    # 200 cells, no growth: each request reserves 5 + 100 + RESERVE_PAD cells.
    ctx = _SimContext(capacity=200, n_ctx_max=200)
    ctx.step_delay = 0.01
    sched = made(ctx, 2)
    first = sched.submit(PROMPTS[0], 100, _Sampler())
    tok = next(first)
    results = {}
    waiter = threading.Thread(
        target=lambda: _collect(sched, PROMPTS[2], 100, results=results, key="b"),
        daemon=True)
    waiter.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sched.counts()[1] == 0:
        time.sleep(0.01)
    assert sched.counts() == (1, 1)
    rest, reason = _drain(first)
    assert [tok] + rest == _reference(PROMPTS[0], 100)[0]
    waiter.join(10)
    assert results["b"]["out"] == _reference(PROMPTS[2], 100)[0]
    assert WAITING_FOR_MODEL_STATUS in results["b"]["statuses"]
    assert all(len({e[2] for e in call}) == 1 for call in ctx.calls)


def test_an_idle_cache_is_dropped_to_make_room(made):
    ctx = _SimContext(capacity=200, n_ctx_max=200)
    sched = made(ctx, 2)
    _collect(sched, PROMPTS[0], 100)
    assert sum(len(v) for v in ctx.cells.values()) > 0
    rec = _collect(sched, PROMPTS[2], 100)
    assert rec["out"] == _reference(PROMPTS[2], 100)[0]
    assert any(p0 == 0 for _seq, p0, _p1 in ctx.seq_rms)


def test_a_follow_up_turn_decodes_only_its_new_tokens(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    first = _collect(sched, PROMPTS[1], 6)
    follow = PROMPTS[1] + first["out"][:5] + [11, 12, 13]
    before = len(ctx.calls)
    rec = _collect(sched, follow, 4)
    assert rec["out"] == _reference(follow, 4)[0]
    prompt_entries = [e for call in ctx.calls[before:] for e in call if e[1] < len(follow)]
    assert [e[0] for e in prompt_entries] == [11, 12, 13]


def test_an_identical_prompt_re_decodes_only_its_last_token(made):
    ctx = _SimContext()
    sched = made(ctx, 1)
    _collect(sched, PROMPTS[1], 3)
    before = len(ctx.calls)
    rec = _collect(sched, PROMPTS[1], 3)
    assert rec["out"] == _reference(PROMPTS[1], 3)[0]
    prompt_entries = [e for call in ctx.calls[before:] for e in call
                      if e[1] < len(PROMPTS[1])]
    assert prompt_entries == [(PROMPTS[1][-1], len(PROMPTS[1]) - 1, 0, True)]


def test_a_cache_that_cannot_be_truncated_is_rebuilt_from_scratch(made):
    ctx = _SimContext(partial_rm=False)
    sched = made(ctx, 1)
    _collect(sched, PROMPTS[1], 3)
    rec = _collect(sched, PROMPTS[1] + [40, 41], 3)
    assert rec["out"] == _reference(PROMPTS[1] + [40, 41], 3)[0]


def test_growing_the_context_replays_the_running_replies(made):
    ctx = _SimContext(capacity=256, n_ctx_max=4096, grow=256)
    ctx.step_delay = 0.005
    sched = made(ctx, 2)
    first = sched.submit(PROMPTS[0], 150, _Sampler())
    got = [next(first) for _ in range(5)]
    rec = _collect(sched, PROMPTS[1], 150)
    rest, reason = _drain(first)
    assert got + rest == _reference(PROMPTS[0], 150)[0]
    assert rec["out"] == _reference(PROMPTS[1], 150)[0]
    assert ctx.recreated and ctx.recreated[0][0] >= 2 * (150 + RESERVE_PAD)


def test_no_growth_beside_running_replies_when_vram_says_it_does_not_fit(made):
    ctx = _SimContext(capacity=256, n_ctx_max=4096, grow=256, vram_fits=False)
    ctx.step_delay = 0.01
    sched = made(ctx, 2)
    first = sched.submit(PROMPTS[0], 150, _Sampler())
    next(first)
    results = {}
    waiter = threading.Thread(
        target=lambda: _collect(sched, PROMPTS[1], 150, results=results, key="b"),
        daemon=True)
    waiter.start()
    time.sleep(0.3)
    assert sched.counts() == (1, 1)
    assert ctx.recreated == []
    _drain(first)
    waiter.join(10)
    assert results["b"]["out"] == _reference(PROMPTS[1], 150)[0]
    assert ctx.recreated == [] or all(offload is False for _t, offload in ctx.recreated)


def test_a_reply_with_no_budget_ends_with_length_at_the_ceiling(made):
    ceiling = len(PROMPTS[0]) + UNLIMITED_STEP + RESERVE_PAD + 100
    ctx = _SimContext(capacity=ceiling, n_ctx_max=ceiling)
    sched = made(ctx, 1)
    rec = _collect(sched, PROMPTS[0], 0)
    assert rec["reason"] == "length"
    assert rec["out"] == _reference(PROMPTS[0], len(rec["out"]))[0]
    assert len(PROMPTS[0]) + len(rec["out"]) >= ceiling - 1


def test_cancelling_one_reply_leaves_the_others_intact(made):
    ctx = _SimContext()
    ctx.step_delay = 0.002
    sched = made(ctx, 3)
    victim = sched.submit(PROMPTS[3], 200, _Sampler())
    results = {}
    others = [threading.Thread(target=lambda i=i: _collect(sched, PROMPTS[i], 60,
                                                           results=results, key=i),
                               daemon=True) for i in (0, 1)]
    for t in others:
        t.start()
    for _ in range(3):
        next(victim)
    victim.close()
    rest, reason = _drain(victim)
    assert reason == "stop"
    for t in others:
        t.join(10)
    for i in (0, 1):
        assert results[i]["out"] == _reference(PROMPTS[i], 60)[0]


def test_a_cancelled_queued_request_never_decodes(made):
    ctx = _SimContext()
    ctx.step_delay = 0.002
    sched = made(ctx, 1)
    running = sched.submit(PROMPTS[0], 50, _Sampler())
    next(running)
    sampler = _Sampler()
    queued = sched.submit([77, 78, 79], 50, sampler)
    queued.close()
    out, reason = _drain(queued)
    assert (out, reason) == ([], "stop")
    _drain(running)
    assert all(e[0] not in (77, 78, 79) for call in ctx.calls for e in call)
    assert sampler in ctx.freed


def test_a_stop_check_cancels_a_waiting_reply(made):
    ctx = _SimContext(capacity=200, n_ctx_max=200)
    ctx.step_delay = 0.01
    sched = made(ctx, 2)
    running = sched.submit(PROMPTS[0], 100, _Sampler())
    next(running)
    stop = threading.Event()
    waiting = sched.submit(PROMPTS[1], 100, _Sampler(), stop_requested=stop.is_set)
    stop.set()
    t0 = time.monotonic()
    out, reason = _drain(waiting)
    assert (out, reason) == ([], "stop")
    assert time.monotonic() - t0 < 5
    _drain(running)


def test_exclusive_section_waits_for_active_slots_without_deadlock(made):
    ctx = _SimContext()
    ctx.step_delay = 0.002
    sched = made(ctx, 2)
    running = sched.submit(PROMPTS[0], 80, _Sampler())
    next(running)
    entered = threading.Event()
    waits = []
    order = []

    def hold():
        with sched.exclusive(on_wait=lambda: waits.append(1)):
            entered.set()
            order.append(("exclusive", sched.counts()[0]))
            time.sleep(0.1)
            order.append(("exclusive-done", len(ctx.calls)))

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    time.sleep(0.05)
    assert not entered.is_set()
    late = {}
    late_thread = threading.Thread(target=lambda: _collect(sched, PROMPTS[2], 5, results=late,
                                                       key="x"), daemon=True)
    late_thread.start()
    rest, _ = _drain(running)
    holder.join(10)
    late_thread.join(10)
    assert not holder.is_alive() and not late_thread.is_alive()
    assert order[0] == ("exclusive", 0)
    calls_at_exit = order[1][1]
    first_late = min(i for i, call in enumerate(ctx.calls)
                     if any(e[0] == PROMPTS[2][0] and e[1] == 0 for e in call))
    assert first_late >= calls_at_exit, "a new reply started inside the exclusive section"
    assert late["x"]["out"] == _reference(PROMPTS[2], 5)[0]
    assert waits, "on_wait was never called while waiting"
    assert ctx.cleared >= 1


def test_a_request_cancelled_behind_an_exclusive_section_ends_at_once(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with sched.exclusive():
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    sampler = _Sampler()
    queued = sched.submit(PROMPTS[0], 10, sampler)
    queued.close()
    t0 = time.monotonic()
    out, reason = _drain(queued)
    waited = time.monotonic() - t0
    release.set()
    holder.join(5)
    assert (out, reason) == ([], "stop")
    assert waited < 2, f"the cancel waited {waited:.1f}s for the exclusive section"
    assert sampler in ctx.freed


def test_close_ends_queued_and_active_replies_and_frees_their_samplers(made):
    ctx = _SimContext(capacity=200, n_ctx_max=200)
    ctx.step_delay = 0.005
    sched = made(ctx, 1)
    s1, s2 = _Sampler(), _Sampler()
    active = sched.submit(PROMPTS[0], 100, s1)
    next(active)
    queued = sched.submit(PROMPTS[1], 100, s2)
    ctx.stop.set()
    sched.close(timeout=5)
    for stream in (active, queued):
        with pytest.raises(RuntimeError) as err:
            _drain(stream)
        assert str(err.value) == UNLOADED_MESSAGE
    assert sorted(map(id, ctx.freed)) == sorted(map(id, [s1, s2]))
    with pytest.raises(RuntimeError):
        sched.submit(PROMPTS[2], 5, _Sampler())


def test_a_failing_sequence_does_not_end_its_neighbours(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    bad = 66
    ctx.fail = lambda entries: 1 if any(e[0] == bad for e in entries) else 0
    first = sched.submit(PROMPTS[0], 30, _Sampler())
    next(first)
    with pytest.raises(RuntimeError, match="prefill"):
        _drain(sched.submit([bad, 1, 2], 30, _Sampler()))
    rest, reason = _drain(first)
    assert reason == "length"
    assert len(rest) == 29


def test_a_sampler_fault_ends_only_its_own_reply(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    results = {}

    def faulty():
        try:
            _collect(sched, PROMPTS[1], 30, sampler=_Sampler(raise_at=2))
        except OSError as exc:
            results["fault"] = exc

    _run_threads([faulty,
                  lambda: _collect(sched, PROMPTS[0], 30, results=results, key="ok")])
    assert isinstance(results["fault"], OSError)
    assert results["ok"]["out"] == _reference(PROMPTS[0], 30)[0]


def test_growing_for_replies_with_no_budget_keeps_every_reply_correct(made):
    # Both replies run out of their reservation on the same step; the first one's
    # extension grows the context, and the second must still sample its own row.
    ctx = _SimContext(capacity=640, n_ctx_max=4096, base=640, grow=128)
    sched = made(ctx, 2)
    statuses = []
    with ctx.lock():
        streams = [sched.submit(PROMPTS[0], 0, _Sampler(), on_status=statuses.append),
                   sched.submit(PROMPTS[2], 0, _Sampler())]
    results = {}
    _run_threads([lambda i=i: results.__setitem__(i, _drain(streams[i])) for i in range(2)],
                 timeout=60)
    assert ctx.recreated, "the context never grew"
    for i, prompt in enumerate((PROMPTS[0], PROMPTS[2])):
        out, reason = results[i]
        assert reason == "length"
        assert len(out) > UNLIMITED_STEP
        assert out == _reference(prompt, len(out))[0]
    assert statuses.count(_slots.GENERATING_STATUS) >= 2, \
        "a running reply heard nothing while the cache was rebuilt"


def test_no_reply_starts_once_an_exclusive_section_got_in_during_admission(made):
    ctx = _SimContext()
    sched = made(ctx, 2)
    real_capacity = ctx.capacity
    fired = []

    def capacity_then_exclusive():
        if not fired:
            fired.append(1)
            with sched._cond:
                sched._exclusive_held = True
        return real_capacity()

    ctx.capacity = capacity_then_exclusive
    stream = sched.submit(PROMPTS[1], 5, _Sampler())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not fired:
        time.sleep(0.01)
    time.sleep(0.2)
    assert fired
    assert sched.counts() == (0, 1), "a reply was admitted inside the exclusive section"
    assert not any(e[0] == PROMPTS[1][0] and e[1] == 0 for call in ctx.calls for e in call)
    with sched._cond:
        sched._exclusive_held = False
        sched._dirty = True
        sched._cond.notify_all()
    out, reason = _drain(stream)
    assert (out, reason) == _reference(PROMPTS[1], 5)


def test_queued_requests_hear_they_wait_while_an_exclusive_section_drains(made, monkeypatch):
    monkeypatch.setattr(_slots, "WAIT_HEARTBEAT_S", 0.05)
    ctx = _SimContext()
    ctx.step_delay = 0.01
    sched = made(ctx, 1)
    running = sched.submit(PROMPTS[0], 150, _Sampler())
    next(running)
    statuses = []
    queued = sched.submit(PROMPTS[1], 3, _Sampler(), on_status=statuses.append)
    watcher = threading.Thread(target=lambda: [None for _ in queued], daemon=True)
    watcher.start()
    holder = threading.Thread(target=lambda: sched.exclusive().__enter__(), daemon=True)
    holder.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sched._exclusive_waiting == 0:
        time.sleep(0.01)
    assert sched._exclusive_waiting == 1
    time.sleep(0.3)
    before = statuses.count(WAITING_FOR_MODEL_STATUS)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and statuses.count(WAITING_FOR_MODEL_STATUS) <= before:
        time.sleep(0.02)
    after = statuses.count(WAITING_FOR_MODEL_STATUS)
    queued.close()
    _drain(running)
    assert after > before, "a queued request heard nothing while the exclusive section waited"


def test_a_failed_context_recreate_ends_the_running_and_the_waiting_reply(made):
    ctx = _SimContext(capacity=256, n_ctx_max=4096, grow=256)
    ctx.step_delay = 0.005
    sched = made(ctx, 2)

    real_recreate = ctx.recreate

    def refuse_once(target, offload_kqv):
        ctx.recreate = real_recreate
        ctx.cap = 0
        ctx.cells = {}
        raise RuntimeError("Not enough memory to create a context")

    ctx.recreate = refuse_once
    running = sched.submit(PROMPTS[0], 150, _Sampler())
    next(running)
    growing = sched.submit(PROMPTS[1], 150, _Sampler())
    for stream in (running, growing):
        with pytest.raises(RuntimeError, match="Not enough memory"):
            _drain(stream)
    rec = _collect(sched, PROMPTS[2], 5)
    assert rec["out"] == _reference(PROMPTS[2], 5)[0]


class _Scorer:
    """Scores a token as a ScoredToken holding its row; raises at *fail_at*."""

    def __init__(self, fail_at=None):
        self.scored = []
        self.closed = 0
        self.fail_at = fail_at

    def score(self, row, token):
        from localm.inference.backends.llamacpp._logprobs import ScoredToken
        if self.fail_at is not None and len(self.scored) == self.fail_at:
            raise RuntimeError("llama.cpp returned no logits for output row 0")
        self.scored.append(token)
        return ScoredToken(token, -float(len(self.scored)), ((token, -1.0),))

    def close(self):
        self.closed += 1


def test_a_scored_reply_yields_scored_tokens_and_closes_its_scorer_once(made):
    from localm.inference.backends.llamacpp._logprobs import ScoredToken
    sched = made(_SimContext(), 2)
    scorer = _Scorer()
    rec = _collect(sched, PROMPTS[0], 50, sampler=_Sampler(eog_at=6), scorer=scorer)
    assert (rec["out"], rec["reason"]) == _reference(PROMPTS[0], 50, eog_at=6)
    assert all(isinstance(t, ScoredToken) for t in rec["out"])
    assert [t.logprob for t in rec["out"]] == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    assert scorer.scored == rec["out"] and EOG not in scorer.scored
    assert scorer.closed == 1


def test_a_scoring_fault_ends_that_reply_only(made):
    sched = made(_SimContext(), 2)
    scorer = _Scorer(fail_at=2)
    results = {}

    def faulty():
        try:
            _collect(sched, PROMPTS[0], 20, scorer=scorer)
        except RuntimeError as e:
            results["error"] = str(e)
    _run_threads([faulty, lambda: _collect(sched, PROMPTS[1], 20, results=results, key="ok")])
    assert "no logits" in results["error"]
    assert results["ok"]["out"] == _reference(PROMPTS[1], 20)[0]
    assert scorer.closed == 1


def test_a_cancelled_reply_closes_its_scorer(made):
    ctx = _SimContext()
    sched = made(ctx, 1)
    scorer = _Scorer()
    with ctx.lock():
        running = sched.submit(PROMPTS[0], 30, _Sampler())
        queued = sched.submit(PROMPTS[1], 30, _Sampler(), scorer=scorer)
        queued.close()
    _drain(running)
    _drain(queued)
    assert scorer.closed == 1
