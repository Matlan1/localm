# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background consolidation must NOT hold the per-namespace RLock across its slow
per-candidate LLM calls, or every chat recall freezes the event loop.

recall(reinforce=True) is the chat inlet and runs ON the asyncio event-loop
thread, taking the namespace RLock; MemoryStore.__init__ takes it too. If
run_consolidation held that SAME lock across its whole read-decide-write,
including one blocking `_decide()` LLM call per candidate (seconds to minutes on
a local model), the next chat turn would block for the entire consolidation.

So consolidation runs the slow decide loop OFF the lock, against a snapshot, then
takes the lock only briefly to reload fresh and MERGE its decided deltas by id,
so a concurrent write during the decide window survives.
"""

from __future__ import annotations

import json
import threading

import pytest

from localm.memory import MemoryRecord, MemoryStore, run_consolidation


@pytest.fixture(autouse=True)
def _writes_on(monkeypatch):
    # run_consolidation is gated on writes_allowed(): without log/full mode it
    # returns "skipped" (privacy) and never calls the model, so _decide() would
    # never fire.
    monkeypatch.setenv("LOCALM_MODE", "log")


def _run_slow_consolidation(store):
    """Start a consolidation whose per-candidate _decide() LLM call blocks until
    released, holding wherever the lock is held. Returns (thread, decide_started,
    release, errors)."""
    decide_started = threading.Event()
    release = threading.Event()
    errors: list = []

    def slow_complete(prompt: str) -> str:
        if "EXISTING:" in prompt:                  # _decide(), NOT extract()
            decide_started.set()
            release.wait(30)                       # hold long enough to expose a stall
            return json.dumps({"decision": "ADD", "confidence": 0.9})
        return json.dumps({"facts": [
            {"fact": "User prefers dark mode in the terminal", "confidence": 0.9}]})

    def worker():
        try:
            run_consolidation(store, "User: I like dark mode in the terminal too",
                              slow_complete)
        except Exception as e:                     # pragma: no cover
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    return t, decide_started, release, errors


def test_recall_not_blocked_while_consolidation_decides(tmp_path):
    store = MemoryStore("owner", "chat", root=tmp_path)
    # Seed a record whose text difflib-near-matches the extracted candidate, so
    # _nearest() routes it to the SLOW _decide() LLM step rather than a
    # deterministic ADD/NO_OP shortcut.
    store.add(MemoryRecord(text="User prefers dark mode in the editor", source="synth"))

    t, decide_started, release, errors = _run_slow_consolidation(store)
    assert decide_started.wait(5), "consolidation's _decide() never started"

    # Consolidation is now mid-_decide(). A chat recall - a fresh store plus
    # recall(reinforce=True), what the inlet does on the loop thread - must not
    # block for the decide duration.
    done = threading.Event()
    out: dict = {}

    def recall_worker():
        try:
            s = MemoryStore("owner", "chat", root=tmp_path)   # __init__ locks (store.py:255)
            out["res"] = s.recall("editor", reinforce=True)   # recall locks (store.py:738)
        except Exception as e:                                # pragma: no cover
            errors.append(e)
        finally:
            done.set()

    rt = threading.Thread(target=recall_worker)
    rt.start()

    completed = done.wait(6)          # a recall blocked on the held lock times out
    release.set()
    t.join(timeout=10)
    rt.join(timeout=10)

    assert not errors, errors
    assert completed, ("recall blocked on the namespace lock held across "
                       "consolidation's slow LLM _decide() (event loop would freeze)")
    assert out.get("res"), "recall returned nothing"
    assert any("editor" in r.text for r in out["res"])


def test_consolidation_still_consolidates_after_decide_offloaded(tmp_path):
    """With _decide() off the lock, consolidation still applies its result: the
    ADD decision lands."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="User prefers dark mode in the editor", source="synth"))

    t, decide_started, release, errors = _run_slow_consolidation(store)
    assert decide_started.wait(5)
    release.set()
    t.join(timeout=10)

    assert not errors, errors
    texts = [r.text for r in MemoryStore("owner", "chat", root=tmp_path).all()]
    assert any("editor" in x for x in texts), "consolidation lost the existing fact"
    assert any("terminal" in x for x in texts), "consolidation did not apply its ADD"


def test_concurrent_add_during_offloaded_decide_survives(tmp_path):
    """A record a concurrent writer commits DURING the lock-free decide window
    survives consolidation's final apply rather than being clobbered by a
    stale-snapshot overwrite."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="User prefers dark mode in the editor", source="synth"))

    t, decide_started, release, errors = _run_slow_consolidation(store)
    assert decide_started.wait(5), "consolidation's _decide() never started"

    concurrent: dict = {}

    def add_worker():
        try:
            concurrent["rec"] = MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text="concurrent user fact added mid-decide", source="user"))
        except Exception as e:                     # pragma: no cover
            errors.append(e)

    at = threading.Thread(target=add_worker)
    at.start()
    at.join(timeout=10)                            # must NOT block on consolidation now
    add_done = "rec" in concurrent
    release.set()
    t.join(timeout=10)
    at.join(timeout=10)

    assert not errors, errors
    assert add_done, ("the concurrent add blocked on the consolidation lock "
                      "(decide loop still holds it)")
    final_ids = {r.id for r in MemoryStore("owner", "chat", root=tmp_path).all()}
    assert concurrent["rec"].id in final_ids, (
        "data loss: a concurrent add during the off-lock decide window was discarded")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
