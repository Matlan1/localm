# SPDX-License-Identifier: AGPL-3.0-or-later
"""CHK-MEM-LOCK regression suite (checkup 2026-07-09, item 13): MemoryStore had no
per-namespace write lock, so concurrent writers silently lost data.

Mirrors rag/store.py's test_concurrent_add_paths_no_data_loss (CHK-RAG-LOCK, the
identical bug already fixed there): each writer constructs its OWN MemoryStore
instance (exactly how every HTTP request/plugin call does it - see
localm/plugins/builtin/memory/plug.py's ``_chat_store``), so a per-instance lock
cannot help. Pre-fix, each instance ``_load()``s the same on-disk state, appends
its own record, and ``_save()``s - last writer wins and the rest are silently
dropped; ``_atomic_write`` also reused a fixed ``.tmp`` filename per namespace, so
concurrent saves could collide on the SAME temp path (PermissionError on Windows).

Also covers CHK-MEM-WINRESOLVE, a separate and narrower Windows-only race found
while testing the fix above: namespace_file()'s path-safety check compared
Path.resolve() output directly, which two DIFFERENT namespaces of the same agent
(distinct ns_hash -> distinct per-namespace lock, so they do not serialise
against each other) racing on their SHARED, not-yet-existing agent directory
(e.g. ``<home>/memory/chat/``) could make return mismatched \\?\\-prefixed vs.
unprefixed forms for the identical location, spuriously raising ValueError - loud,
not silent, and only on a true cold start (see test_cold_start_concurrent_
namespaces_no_path_safety_race below).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from localm.memory import MemoryRecord, MemoryStore

N_WRITERS = 8


def test_concurrent_add_no_data_loss(tmp_path):
    """N threads, each with its OWN MemoryStore instance, concurrently add() a
    distinct record to the SAME namespace. All N must survive and no writer may
    raise (in particular no PermissionError from a colliding temp file)."""
    # Warm the namespace (mkdir + first save) before racing: a cold concurrent
    # FIRST write into a directory that does not exist yet hits a separate,
    # unrelated Windows Path.resolve() quirk in namespace_file() (an existing vs.
    # not-yet-existing target resolves to a \\?\-prefixed path or not, so the
    # path-safety comparison can spuriously raise) - not the CHK-MEM-LOCK race
    # this test targets. Flagged separately; by the time real concurrent writers
    # hit a namespace it has almost always already been written to once.
    seed = MemoryStore("owner", "chat", root=tmp_path)
    seed.add(MemoryRecord(text="seed", source="user"))
    seed.delete(seed.all()[0].id)
    start = threading.Barrier(N_WRITERS)
    errors: list = []
    ids: list = []
    lock = threading.Lock()

    def worker(i):
        try:
            start.wait()                          # release every thread together
            rec = MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text=f"unique concurrent fact {i}", source="user"))
            with lock:
                ids.append(rec.id)
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    final = MemoryStore("owner", "chat", root=tmp_path)
    stored_ids = {r.id for r in final.all()}
    assert len(final) == N_WRITERS, (
        f"data loss: expected {N_WRITERS} records, got {len(final)}")
    assert set(ids) == stored_ids, "a writer's record id is missing from disk"


def test_cold_start_concurrent_namespaces_no_path_safety_race(tmp_path):
    """CHK-MEM-WINRESOLVE: the per-namespace lock MemoryStore.__init__ takes
    around namespace_file() (CHK-MEM-LOCK) only serialises writers to the SAME
    namespace - it cannot serialise DIFFERENT namespaces of the same agent
    (distinct scope_key -> distinct ns_hash -> distinct lock) against each
    other. Those namespaces share one parent directory (``<root>/chat/``), so
    when it does not exist yet, one namespace's first _save() (its mkdir) can
    race a sibling namespace's concurrent namespace_file() call resolving a
    file under that same directory. On Windows this can make Path.resolve()
    return a \\?\\-prefixed extended-length path for one and an unprefixed
    path for the other, even though both denote the identical location, so the
    path-safety comparison spuriously raised ValueError. Confirmed via an
    isolated repro (not just this pytest run): the old comparison hit this on
    the majority of rounds when racing 40 distinct cold namespaces; a
    same-single-namespace race (as in test_concurrent_add_no_data_loss above)
    does NOT reach this code path, because CHK-MEM-LOCK's lock already
    serialises same-namespace construction - hence a SEPARATE, DISTINCT-
    namespace test is required to exercise it, unlike the sibling tests above
    which warm the namespace first specifically to avoid this race."""
    n = 40
    start = threading.Barrier(n)
    errors: list = []

    def worker(i):
        try:
            start.wait()                          # release every thread together
            MemoryStore("owner", "chat", scope_key=f"scope-{i}", root=tmp_path).add(
                MemoryRecord(text=f"cold start fact {i}", source="user"))
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for i in range(n):
        ns_store = MemoryStore("owner", "chat", scope_key=f"scope-{i}", root=tmp_path)
        assert len(ns_store) == 1, f"data loss in namespace scope-{i}"


def test_concurrent_add_update_delete_no_data_loss(tmp_path):
    """Mixed concurrent add/update/delete against one namespace: every add must
    persist, and the store must never end up corrupt or partially written."""
    seed = MemoryStore("owner", "chat", root=tmp_path)
    to_update = seed.add(MemoryRecord(text="will be updated", source="user"))
    to_delete = seed.add(MemoryRecord(text="will be deleted", source="user"))

    start = threading.Barrier(6)
    errors: list = []

    def add_worker(i):
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text=f"added {i}", source="user"))
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    def update_worker():
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).update(
                to_update.id, text="updated text")
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    def delete_worker():
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).delete(to_delete.id)
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    threads = (
        [threading.Thread(target=add_worker, args=(i,)) for i in range(4)]
        + [threading.Thread(target=update_worker), threading.Thread(target=delete_worker)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    final = MemoryStore("owner", "chat", root=tmp_path)
    # 2 seeded + 4 added - 1 deleted = 5
    assert len(final) == 5, f"data loss: expected 5 records, got {len(final)}"
    updated = final.get(to_update.id)
    assert updated is not None and updated.text == "updated text"
    assert final.get(to_delete.id) is None


def test_atomic_write_unique_tmp_path_survives_concurrent_saves(tmp_path):
    """Two threads saving the SAME namespace concurrently must not collide on
    _atomic_write's temp file (pre-fix: both used ``<file>.tmp`` verbatim, so one
    thread's ``tmp.replace()`` could race the other's ``tmp.write_text()`` and
    raise PermissionError on Windows)."""
    s = MemoryStore("owner", "chat", root=tmp_path)
    s.add(MemoryRecord(text="seed", source="user"))   # warm the namespace dir
    s.delete(s.all()[0].id)
    start = threading.Barrier(2)
    errors: list = []

    def saver(text):
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text=text, source="user"))
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=saver, args=(f"race {i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(MemoryStore("owner", "chat", root=tmp_path)) == 2


def test_concurrent_consolidation_and_add_no_data_loss(tmp_path, monkeypatch):
    """CHK-MEM-LOCK (adversarial review finding, round 2): run_consolidation()
    snapshots store.all(), then for each candidate that near-but-not-exactly
    matches an existing record calls the SLOW per-candidate _decide() LLM step,
    before finally overwriting the whole namespace via replace(). A concurrent
    add() landing during THAT decide call must block on the namespace lock and
    survive, not be silently discarded by the stale snapshot's overwrite.

    The delay is deliberately placed inside _decide() (identified by the
    "EXISTING:" marker unique to _DECIDE_PROMPT), not extract(): a version of
    this test that only blocks during extract() (before store.all() is even
    read) stays green even with consolidate.py's outer store.lock() removed,
    because replace()'s own lock+reload already covers that narrower window - it
    would not catch a regression that re-acquires the lock only around the final
    replace() rather than across the whole decide loop. The seeded record's text
    is a difflib near-duplicate of the extracted candidate (ratio in
    (MATCH_THRESHOLD, NEAR_DUP_RATIO)), so _nearest() routes it to _decide()
    instead of a deterministic ADD/NO_OP shortcut that would skip the LLM call."""
    monkeypatch.setenv("LOCALM_MODE", "log")
    from localm.memory import run_consolidation
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="User prefers dark mode in the editor",
                           source="synth"))

    decide_started = threading.Event()
    release = threading.Event()

    def slow_complete(prompt: str) -> str:
        if "EXISTING:" in prompt:                 # _decide(), not extract()
            decide_started.set()
            release.wait(timeout=5)
            return json.dumps({"decision": "ADD", "confidence": 0.9})
        return json.dumps({"facts": [
            {"fact": "User prefers dark mode in the terminal", "confidence": 0.9}]})

    errors: list = []

    def consolidate_worker():
        try:
            run_consolidation(store, "User: I like dark mode in the terminal too",
                              slow_complete)
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    t1 = threading.Thread(target=consolidate_worker)
    t1.start()
    assert decide_started.wait(timeout=5), "consolidation's _decide() never started"

    concurrent = {}

    def add_worker():
        try:
            concurrent["rec"] = MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text="concurrent user fact", source="user"))
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    t2 = threading.Thread(target=add_worker)
    t2.start()
    time.sleep(0.2)          # let add_worker actually reach and block on the lock
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, errors
    final_ids = {r.id for r in MemoryStore("owner", "chat", root=tmp_path).all()}
    assert concurrent["rec"].id in final_ids, (
        "data loss: the concurrent add() was silently discarded by "
        "consolidation's replace()")


def test_reinforce_recall_does_not_resurrect_concurrent_delete(tmp_path):
    """CHK-MEM-LOCK (adversarial review finding): recall(reinforce=True) is
    called on every chat turn and used to save this instance's whole
    self._records unlocked - a concurrent delete() landing in that window would
    be silently reverted (the deleted record resurrected) by recall's save."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    rec = store.add(MemoryRecord(text="a fact about the user's editor preference",
                                 source="user", importance=0.8))

    start = threading.Barrier(2)
    errors: list = []

    def recall_worker():
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).recall(
                "editor preference", reinforce=True)
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    def delete_worker():
        try:
            start.wait()
            MemoryStore("owner", "chat", root=tmp_path).delete(rec.id)
        except Exception as e:                    # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=recall_worker),
               threading.Thread(target=delete_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    final = MemoryStore("owner", "chat", root=tmp_path)
    # Whichever ran first, the lock serialises them, so the delete must always
    # hold in the end - recall's reinforce-save must never resurrect it.
    assert final.get(rec.id) is None, (
        "reinforce's save silently resurrected a concurrently-deleted record")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
