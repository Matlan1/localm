# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory namespaces must serialise writers ACROSS PROCESSES.

``storekit.NamespaceLockRegistry`` serialises writers inside ONE process, which
its own docstring says. It cannot serialise `localm memory add|forget|edit|
accept|restore` - its own OS process, its own registry - against a running
server's consolidation pass: both ``_load()`` the same state, mutate their copy
and ``_save()``, so one update is gone. rag covers the same exposure with
``collection_lock.py``.

Without a cross-process lock, `localm memory add` prints
"Remembered <id>: <fact>" and exits 0 while the fact is ABSENT afterwards: a
FALSE SUCCESS, not merely a lost update.

The load-bearing tests here spawn REAL subprocesses, because a per-process lock
passes every same-process simulation of this bug by construction. The headline
test carries its own FIRES-CONTROL in the same run: the identical harness with
the cross-process lock neutralised IN THE CHILD (test-side, never a product
switch) must show the loss the locked version prevents.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from localm.memory import MemoryRecord, MemoryStore
from localm.memory.store import _namespace_lockfile
from localm.rag.collection_lock import CollectionLockedError


# heavy_slot (only ONE subprocess-heavy test at a time, box-wide) comes from
# tests/conftest.py and is shared with tests/test_rag_collection_lock.py: these
# spawn real interpreters, and two files each serialising only themselves would
# starve each other.


# The read-decide-write shape run_consolidation's apply block has: load the
# current state, spend time, then save. The delay is what makes the window real;
# in production it is the embedder resolution.
_WORKER = """
import contextlib, sys, time
root, text, delay, neuter = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
import localm.memory.store as st
from localm.memory import MemoryRecord, MemoryStore

if neuter == "1":
    # Test-side only: strip the CROSS-process half, keep the in-process half, so
    # the child is exactly the pre-fix code. Never a product flag - a shipped
    # "skip the lock" switch would be the unserialised write path this forbids.
    @contextlib.contextmanager
    def _plain(ns_hash, store_file, op, timeout=None):
        with st._namespace_lock(ns_hash):
            yield
    st._namespace_write_lock = _plain

s = MemoryStore("owner", "chat", root=root)
with s.lock():
    s._load()
    time.sleep(delay)
    s._records.append(MemoryRecord(text=text, kind="semantic", source="user",
                                   importance=0.8))
    s._save()
"""

# Long enough to swamp interpreter-startup skew, short enough not to hold two
# real interpreters on the box longer than the race needs.
_RACE_DELAY = 1.5


def _race_two_writers(root: Path, *, neuter: bool) -> set:
    """Two separate OS processes each add a different fact to the SAME namespace.

    Returns the set of fact texts that survived."""
    env = dict(os.environ)
    # Import the SAME localm this test runs (the worktree), not whatever is
    # installed elsewhere.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["LOCALM_MODE"] = "log"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(root), text, str(_RACE_DELAY),
             "1" if neuter else "0"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for text in ("fact from the server", "fact from the CLI")
    ]
    for i, p in enumerate(procs):
        _out, err = p.communicate(timeout=180)
        assert p.returncode == 0, (
            f"writer {i} exited {p.returncode}: "
            f"{err.decode('utf-8', 'replace')[-2000:]}")
    return {r.text for r in MemoryStore("owner", "chat", root=root).all()}


def test_two_processes_writing_one_namespace_lose_nothing(heavy_slot, tmp_path):
    """The headline case: `localm memory add` in its own process racing the
    server's consolidation on the SAME namespace. Both facts must survive."""
    survived = _race_two_writers(tmp_path, neuter=False)
    assert survived == {"fact from the server", "fact from the CLI"}, (
        f"a concurrent memory write from a SEPARATE OS process was lost: "
        f"{survived}")


def test_the_two_process_harness_does_catch_a_lost_update(heavy_slot, tmp_path):
    """FIRES-CONTROL for the test above.

    Same two processes, same timing, cross-process lock neutralised in the
    children: one write MUST be lost. If this passes with both facts present,
    the test above is not evidence of anything."""
    survived = _race_two_writers(tmp_path, neuter=True)
    assert survived != {"fact from the server", "fact from the CLI"}, (
        "with the cross-process lock removed, two overlapping memory writes "
        "still kept both facts - this harness cannot observe the race it is "
        "supposed to be proving, so the locked test above proves nothing")


# --------------------------------------------------------------------------- #
#  Refusal, and what a refusal must NOT do                                     #
# --------------------------------------------------------------------------- #

def _hold_from_another_process(root: Path, seconds: float):
    """Start a real second process holding the namespace, and wait until it
    actually has the lock (the lock FILE existing is the proof, not a sleep)."""
    src = ("import sys, time\n"
           "from localm.memory import MemoryStore\n"
           "s = MemoryStore('owner', 'chat', root=sys.argv[1])\n"
           "with s.lock():\n"
           "    print('HELD', flush=True)\n"
           f"    time.sleep({seconds})\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["LOCALM_MODE"] = "log"
    p = subprocess.Popen([sys.executable, "-c", src, str(root)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout.readline().strip() == b"HELD", "holder never took the lock"
    return p


def test_a_refused_write_changes_nothing_and_names_the_holder(heavy_slot,
                                                              tmp_path,
                                                              monkeypatch):
    """Fails CLOSED. There is no path that proceeds to write without the lock,
    because an unserialised write is the lost update this exists to prevent."""
    monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "1")
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="already here", source="user", importance=0.8))
    holder = _hold_from_another_process(tmp_path, 25)
    try:
        with pytest.raises(CollectionLockedError) as caught:
            MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text="must not land", source="user",
                             importance=0.8))
        assert "Memory namespace" in str(caught.value)
        assert "nothing was changed" in str(caught.value)
        # The store on disk is untouched, asserted before any message check.
        assert [r.text for r in MemoryStore("owner", "chat", root=tmp_path).all()] \
            == ["already here"]
    finally:
        holder.kill()
        holder.communicate(timeout=60)


def test_reads_stay_available_while_another_process_holds_the_lock(heavy_slot,
                                                                   tmp_path,
                                                                   monkeypatch):
    """Reads do NOT take the cross-process lock.

    _save() writes through storekit.atomic_write (tmp + os.replace), so a reader
    sees the old file or the new one, never a mix - the read side is already
    safe. Putting every _load() behind the file lock would park the chat recall
    inlet behind whatever a background consolidation is doing."""
    monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "1")
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="readable", source="user", importance=0.8))
    holder = _hold_from_another_process(tmp_path, 25)
    try:
        started = time.time()
        fresh = MemoryStore("owner", "chat", root=tmp_path)
        assert [r.text for r in fresh.all()] == ["readable"]
        assert fresh.forgotten() == []
        assert fresh.corrections() == []       # degrades, never raises
        assert time.time() - started < 15, (
            "a read waited on the cross-process write lock")
    finally:
        holder.kill()
        holder.communicate(timeout=60)


# --------------------------------------------------------------------------- #
#  Reentrancy: the nesting that already exists in the tree                     #
# --------------------------------------------------------------------------- #

def test_batching_under_store_lock_does_not_deadlock(tmp_path):
    """plug._migrate_legacy and memory_put both hold store.lock() across several
    save=False mutations. collection_write_lock turns a NESTED acquisition into
    an error, so the outermost acquisition has to be the only one that takes the
    file lock."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    with store.lock():
        store._load()
        for i in range(3):
            store.add(MemoryRecord(text=f"batched {i}", source="user",
                                   importance=0.8), save=False)
        store._save()
    assert len(MemoryStore("owner", "chat", root=tmp_path).all()) == 3


def test_prune_calling_replace_does_not_deadlock(tmp_path):
    """prune() calls replace(), and both are write paths."""
    store = MemoryStore("owner", "chat", root=tmp_path)
    for i in range(4):
        store.add(MemoryRecord(text=f"fact {i}", source="user", importance=0.8,
                               last_used=1_700_000_000.0 - i * 86400.0),
                  save=False)
    store._save()
    assert store.prune(now=1_700_000_000.0, n_max=2) == 2
    assert len(MemoryStore("owner", "chat", root=tmp_path).all()) == 2


def test_the_lock_file_cannot_be_mistaken_for_a_namespace(tmp_path):
    """backfill._namespaces globs ``*/*.jsonl``; the lock file is a sibling named
    ``<ns>.jsonl.lock``, so it is excluded by construction rather than by a deny
    list."""
    from localm.memory.backfill import _namespaces
    store = MemoryStore("owner", "chat", root=tmp_path)
    store.add(MemoryRecord(text="a fact", source="user", importance=0.8))
    lockfile = _namespace_lockfile(store.path)
    lockfile.write_text(json.dumps({"token": "x"}), encoding="utf-8")
    assert lockfile.exists() and lockfile.parent == store.path.parent
    assert list(_namespaces(tmp_path)) == [store.path]


def test_a_write_waiting_on_the_file_lock_does_not_block_reads_in_its_own_process(
        heavy_slot, tmp_path, monkeypatch):
    """The lock ORDER, pinned.

    The wait for the FILE lock happens OUTSIDE the namespace RLock. Taking the
    RLock first and waiting for the file lock inside it blocks every read in the
    writing process for the writer's whole budget, because
    ``MemoryStore.__init__`` takes that same RLock to ``_load()``.
    """
    monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "20")
    holder = _hold_from_another_process(tmp_path, 20)
    try:
        writer = threading.Thread(target=lambda: _swallow(
            lambda: MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text="waits", source="user", importance=0.8))))
        writer.start()
        time.sleep(2)                      # let the writer get into its wait
        started = time.time()
        MemoryStore("owner", "chat", root=tmp_path).all()
        elapsed = time.time() - started
        assert elapsed < 8, (
            f"a read in the writer's own process waited {elapsed:.1f}s: the "
            f"namespace RLock is being held across the cross-process wait")
    finally:
        holder.kill()
        holder.communicate(timeout=60)
        writer.join(timeout=120)


def _swallow(fn):
    """Run *fn*, ignoring a lock refusal: the test above is about the READ's
    latency, and whether the writer eventually wins the race is not its subject."""
    try:
        fn()
    except CollectionLockedError:
        pass


def test_a_bounded_caller_stays_bounded_behind_a_blocked_writer(heavy_slot,
                                                                tmp_path,
                                                                monkeypatch):
    """corrections() passes a SHORT timeout so a read path never stalls. That
    budget has to survive a sibling thread in the same process already waiting on
    the long one.

    With an unbounded in-process gate, a writer blocked for its full budget
    holds the gate the whole time and this call queues behind it regardless of
    the timeout it asked for. The gate shares the budget instead.
    """
    monkeypatch.setenv("LOCALM_RAG_LOCK_WAIT", "25")
    holder = _hold_from_another_process(tmp_path, 25)
    writer = None
    try:
        writer = threading.Thread(target=lambda: _swallow(
            lambda: MemoryStore("owner", "chat", root=tmp_path).add(
                MemoryRecord(text="waits", source="user", importance=0.8))))
        writer.start()
        time.sleep(2)                      # let the writer own the gate
        started = time.time()
        MemoryStore("owner", "chat", root=tmp_path).corrections()
        elapsed = time.time() - started
        assert elapsed < 10, (
            f"a bounded caller waited {elapsed:.1f}s behind a blocked writer: its "
            f"short budget is not being applied to the in-process gate")
    finally:
        holder.kill()
        holder.communicate(timeout=60)
        if writer is not None:
            writer.join(timeout=120)
