# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the global child-agent concurrency gate (localm.plugins.coder.child_limit).

The gate exists so that two independent child-spawning features cannot each admit
their own 2 children and jointly run 4 on a box whose ceiling is about 2 resident
models. The tests that matter here are therefore the ATOMICITY one (two spawns
racing for one free slot) and the IDEMPOTENT-RELEASE one (a double release must not
silently widen the cap) - both are invisible in ordinary single-threaded use.
"""

from __future__ import annotations

import threading

import pytest

from localm.plugins.coder import child_limit as cl


@pytest.fixture
def gate():
    """A private gate instance, so these tests never race the process-wide one."""
    return cl.ChildLimit(max_children=2)


def test_cap_is_two_and_third_acquire_is_refused(gate):
    a = gate.try_acquire("parallel", "child-a")
    b = gate.try_acquire("parallel", "child-b")
    assert a is not None and b is not None
    # The third must be refused, and refused by returning None rather than blocking.
    assert gate.try_acquire("background", "child-c") is None
    assert gate.available() == 0


def test_release_frees_exactly_one_slot(gate):
    a = gate.try_acquire("parallel", "child-a")
    gate.try_acquire("parallel", "child-b")
    assert gate.try_acquire("background", "child-c") is None
    gate.release(a)
    assert gate.available() == 1
    c = gate.try_acquire("background", "child-c")
    assert c is not None
    assert gate.try_acquire("background", "child-d") is None


def test_double_release_does_not_widen_the_cap(gate):
    """A decrement-based counter would drift and silently allow a 3rd child."""
    a = gate.try_acquire("parallel", "child-a")
    b = gate.try_acquire("parallel", "child-b")
    gate.release(a)
    gate.release(a)          # idempotent: same token again
    gate.release(a)          # and again
    assert gate.available() == 1, "double release widened the budget"
    assert gate.try_acquire("background", "c") is not None
    assert gate.try_acquire("background", "d") is None, "cap widened past 2"
    gate.release(b)


def test_release_tolerates_none_and_stale_tokens(gate):
    """Callers release in a finally, where the acquire may have failed (None)."""
    gate.release(None)
    stale = cl.Token(id=9999, kind="parallel", label="never-acquired")
    gate.release(stale)      # must not raise inside somebody's cleanup handler
    assert gate.available() == 2


def test_holders_names_the_running_children(gate):
    gate.try_acquire("parallel", "refactor-auth")
    gate.try_acquire("background", "run-tests")
    names = {(h.kind, h.label) for h in gate.holders()}
    assert names == {("parallel", "refactor-auth"), ("background", "run-tests")}


def test_concurrent_acquires_never_exceed_the_cap(gate):
    """THE RACE TEST. 24 threads start simultaneously against a 2-slot gate.

    A non-atomic check-then-insert lets several threads all observe a free slot
    and all admit; ordinary sequential use never reveals that.
    """
    n_threads = 24
    barrier = threading.Barrier(n_threads)
    won: list[cl.Token] = []
    won_lock = threading.Lock()
    peak = 0
    peak_lock = threading.Lock()

    def worker(i: int) -> None:
        nonlocal peak
        barrier.wait()                     # maximise real contention
        tok = gate.try_acquire("parallel", f"child-{i}")
        if tok is None:
            return
        with won_lock:
            won.append(tok)
        # Sample occupancy while holding, so an over-admit is caught in the act.
        live = len(gate.holders())
        with peak_lock:
            peak = max(peak, live)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(won) == 2, f"expected exactly 2 winners, got {len(won)}"
    assert peak <= 2, f"observed {peak} children holding a 2-slot budget"


def test_contended_slot_has_exactly_one_winner(gate):
    """One free slot, many contenders: exactly one may win, never two."""
    held = gate.try_acquire("parallel", "incumbent")
    assert held is not None
    assert gate.available() == 1

    n_threads = 16
    barrier = threading.Barrier(n_threads)
    winners: list[cl.Token] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        barrier.wait()
        tok = gate.try_acquire("background", f"contender-{i}")
        if tok is not None:
            with lock:
                winners.append(tok)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"one free slot admitted {len(winners)} children"


def test_no_blocking_acquire_is_exposed():
    """The silent queue must be impossible to reintroduce by accident.

    A blocking acquire would turn "budget full" into an invisible wait, defeating
    a background-spawn caller and hanging a caller that meant to reject. The
    module offers no such entry point.
    """
    assert not hasattr(cl, "acquire"), "a blocking acquire() was added"
    assert not hasattr(cl.ChildLimit, "acquire"), "a blocking acquire() was added"


def test_module_level_functions_share_one_process_wide_budget():
    """Both features must draw from the SAME budget, not per-caller instances."""
    cl._reset_for_tests()
    try:
        first = cl.try_acquire("parallel", "a")
        second = cl.try_acquire("parallel", "b")
        assert first is not None and second is not None
        # A different feature ("background") must see the budget already spent.
        assert cl.try_acquire("background", "c") is None
        assert cl.available() == 0
        assert "parallel 'a'" in cl.describe_holders()
        cl.release(first)
        assert cl.try_acquire("background", "c") is not None
    finally:
        cl._reset_for_tests()


def test_describe_holders_when_idle():
    cl._reset_for_tests()
    assert cl.describe_holders() == "no child agents are running"
