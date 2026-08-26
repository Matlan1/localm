# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-process write lock for a RAG collection.

``store._collection_lock`` is a per-process ``threading.RLock``, so `localm rag
add|resync` in its own OS process can interleave its read-modify-write with a
running server's scheduled re-sync of the SAME collection and lose one update.
``config._cross_process_lock`` cannot serve here because it reclaims any holder
older than 30 s as abandoned, which would reap a LIVE indexing run;
``localm/rag/collection_lock.py`` uses a heartbeat instead of a wall-clock hold
limit.

The load-bearing tests here spawn REAL subprocesses, because a per-process lock
passes every same-process simulation of this bug by construction. Each one
carries its own FIRES-CONTROL in the same run: the identical harness, with the
cross-process lock neutralised IN THE CHILD (test-side, never a product switch),
must show the interleave the locked version prevents.

The heartbeat is the lock file's MTIME (the module docstring explains why a
rewritten timestamp field could clobber a successor's record), so a "stale"
holder is staged here with ``os.utime``, not by editing a field.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import types

import pytest

import localm.rag.collection_lock as cl
from localm.rag import Collection, CollectionLockedError, collection_names
from localm.rag.collection_lock import collection_write_lock, lock_path_for


# heavy_slot (only ONE subprocess-heavy test at a time, box-wide) lives in
# tests/conftest.py, so this file and tests/test_memory_cross_process_lock.py
# share ONE slot rather than each serialising only itself.


@pytest.fixture
def base(tmp_path):
    """A collections root under tmp_path (never the user's real rag dir)."""
    d = tmp_path / "collections"
    d.mkdir()
    return d


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alpha.txt").write_text("alpha content about turbines", encoding="utf-8")
    (d / "beta.txt").write_text("beta content about gearboxes", encoding="utf-8")
    return d


def _record(**over) -> dict:
    """A well-formed lock record for a plausible FOREIGN holder."""
    rec = {"token": "feedface" * 4, "pid": 999_999, "pid_create_time": None,
           "machine": "another-machine-hash", "collection": "kb",
           "op": "a re-sync", "started": time.time() - 120}
    rec.update(over)
    return rec


def _hold(path, rec, *, silent_for: float = 0.0) -> None:
    """Stage a foreign lock file whose holder last reported *silent_for* ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf-8")
    if silent_for:
        when = time.time() - silent_for
        os.utime(path, (when, when))


def _flat(text: str) -> str:
    """Collapse whitespace so an assertion survives console line wrapping."""
    return re.sub(r"\s+", " ", text or "")


# --------------------------------------------------------------------------- #
#  1. Two REAL processes writing the same collection                          #
# --------------------------------------------------------------------------- #

# Each child indexes ONE file into the SAME collection. A file rendezvous makes
# both children reach the write at the same moment however long their
# interpreter took to start, and a delay INSIDE the locked read-modify-write
# (after _load(), before the save) widens the window a lost update needs.
#
# NEUTER=1 replaces the cross-process lock with a no-op IN THE CHILD ONLY: with
# it, an update must be lost.
_WORKER = r"""
import contextlib, sys, time
from pathlib import Path
import localm.rag.store as store
from localm.rag import Collection

base, name, doc, delay, neuter, rv = (sys.argv[1], sys.argv[2], sys.argv[3],
                                      float(sys.argv[4]), sys.argv[5] == "1",
                                      Path(sys.argv[6]))
if neuter:
    store.Collection._write_lock = (
        lambda self, op, on_progress=None: contextlib.nullcontext())

_orig = store.Collection._add_paths_locked
def _slow(self, *a, **kw):
    time.sleep(delay)          # widen the load-modify-save window, inside the lock
    return _orig(self, *a, **kw)
store.Collection._add_paths_locked = _slow

# Rendezvous: announce readiness, then wait for the sibling, so interpreter
# start-up skew cannot decide the race before it begins.
(rv / (Path(doc).name + ".ready")).write_text("1", encoding="utf-8")
_deadline = time.time() + 60
while len(list(rv.glob("*.ready"))) < 2 and time.time() < _deadline:
    time.sleep(0.01)

coll = Collection(name, base=Path(base))
coll.create()
coll.add_paths([doc])
"""

# Long enough to swamp the residual skew the rendezvous leaves (milliseconds),
# short enough that four real interpreters are not held on the box any longer
# than the race needs.
_RACE_DELAY = 0.5


def _race_two_writers(tmp_path, base, docs, name, *, neuter: bool):
    """Two separate OS processes each index a different file into *name*.

    Returns the set of document basenames that survived in the collection."""
    rv = tmp_path / ("rv-neuter" if neuter else "rv-locked")
    rv.mkdir()
    env = dict(os.environ)
    # Import the SAME localm this test runs (the worktree), not whatever is
    # installed elsewhere.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(base), name, str(doc),
             str(_RACE_DELAY), "1" if neuter else "0", str(rv)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for doc in (docs / "alpha.txt", docs / "beta.txt")
    ]
    for i, p in enumerate(procs):
        out, err = p.communicate(timeout=180)
        assert p.returncode == 0, (
            f"writer {i} crashed (rc={p.returncode}): "
            f"{err.decode('utf-8', 'replace')[-2000:]}")
    meta = json.loads((base / name / "meta.json").read_text(encoding="utf-8"))
    return {os.path.basename(k) for k in meta.get("docs", {})}


def test_two_processes_indexing_one_collection_lose_nothing(heavy_slot, tmp_path,
                                                            base, docs):
    """The headline case: `localm rag add` in its own process racing another
    localm process on the SAME collection. Both documents must survive."""
    survived = _race_two_writers(tmp_path, base, docs, "kb", neuter=False)
    assert survived == {"alpha.txt", "beta.txt"}, (
        f"a concurrent index from a SEPARATE OS process was lost: {survived}")


def test_the_two_process_harness_does_catch_a_lost_update(heavy_slot, tmp_path,
                                                          base, docs):
    """FIRES-CONTROL for the test above.

    Same two processes, same timing, with the cross-process lock neutralised in
    the children: one update MUST be lost."""
    survived = _race_two_writers(tmp_path, base, docs, "kb", neuter=True)
    assert survived != {"alpha.txt", "beta.txt"}, (
        "with the cross-process lock removed, two overlapping indexing runs "
        "still kept both documents - this harness cannot observe the race it "
        "is supposed to be proving, so the locked test above proves nothing")


# --------------------------------------------------------------------------- #
#  2. Staleness: heartbeat age, both directions                               #
# --------------------------------------------------------------------------- #

def test_a_dead_holders_stale_lock_is_reclaimed(base, capsys):
    """A lock left behind by a crashed holder must not wedge the collection
    forever: once its heartbeat goes stale it is reclaimed, and said out loud."""
    lp = lock_path_for(base / "kb")
    _hold(lp, _record(), silent_for=3600)

    with collection_write_lock(lp, collection="kb", op="a test", timeout=5.0):
        held = json.loads(lp.read_text(encoding="utf-8"))
    assert held["pid"] == os.getpid(), "the stale lock was not taken over"
    assert not lp.exists(), "the lock must be released afterwards"
    assert "reclaimed the write lock" in capsys.readouterr().err


def test_a_live_holder_with_a_fresh_heartbeat_is_never_reclaimed(base):
    """The other direction, and the whole reason for the heartbeat: a holder
    that has held the lock far LONGER than the staleness window but is still
    reporting must be left alone. config._cross_process_lock's fixed 30 s rule
    would have reaped this one - it is a normal indexing run."""
    lp = lock_path_for(base / "kb")
    _hold(lp, _record(started=time.time() - 7200))     # held 2h, reporting now

    with pytest.raises(CollectionLockedError) as e:
        with collection_write_lock(lp, collection="kb", op="a test",
                                   timeout=0.3, stale_after=1.0):
            pass
    assert "pid 999999" in _flat(str(e.value)), "the refusal must name the holder"
    assert json.loads(lp.read_text(encoding="utf-8"))["pid"] == 999_999, (
        "a live holder's lock was stolen")


def test_a_real_hold_outlasting_stale_after_survives_because_it_beats(
        heavy_slot, base, monkeypatch):
    """The same property with the REAL heartbeat thread running, which is what
    makes an hours-long indexing run safe: hold across several staleness
    windows, and a waiter must still refuse rather than steal it.

    Takes ``heavy_slot`` even though it spawns no subprocess: it is the only
    test here whose correctness depends on a THREAD being scheduled promptly,
    so it serialises against the siblings in this file that spawn real
    interpreters. The widened window below covers the rest."""
    # stale_after is TWELVE heartbeat intervals, the same ratio the shipped
    # defaults use (HEARTBEAT_INTERVAL 5 s, STALE_AFTER 60 s). Reclaim triggers
    # on the age of the LAST beat, so the window has to survive an ordinary
    # contiguous starvation of the heartbeat thread on a loaded runner. The
    # control below still requires a NON-beating record with this same
    # stale_after to be reclaimed.
    monkeypatch.setattr(cl, "HEARTBEAT_INTERVAL", 0.1)
    lp = lock_path_for(base / "kb")
    stale_after = 1.2

    with collection_write_lock(lp, collection="kb", op="a long index",
                               timeout=5.0):
        # TWO windows, not four, to bound how long this test occupies heavy_slot,
        # which is box-wide across xdist workers. Crossing the staleness
        # threshold twice proves the same property: a REFRESHED record survives
        # past the threshold.
        time.sleep(stale_after * 2)          # two whole staleness windows
        mine = json.loads(lp.read_text(encoding="utf-8"))["token"]
        with pytest.raises(CollectionLockedError):
            with collection_write_lock(lp, collection="kb", op="a waiter",
                                       timeout=0.2, stale_after=stale_after):
                pass
        assert json.loads(lp.read_text(encoding="utf-8"))["token"] == mine

    # The control, same run and same stale_after: an identical record that is
    # NOT being refreshed is reclaimed.
    _hold(lp, _record(), silent_for=60)
    with collection_write_lock(lp, collection="kb", op="a waiter",
                               timeout=5.0, stale_after=stale_after):
        assert json.loads(lp.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_recycled_pid_is_not_read_as_a_live_holder(base):
    """pids are reused. A record pinning THIS pid but a start time from before
    this process existed describes a process that is gone, whatever the number
    says - so it is reclaimable early rather than holding the collection for the
    full staleness window."""
    lp = lock_path_for(base / "kb")
    _hold(lp, _record(pid=os.getpid(), pid_create_time=1.0,   # 1970: not us
                      machine=cl._machine_id()),
          silent_for=cl.DEAD_HOLDER_GRACE + 5)

    with collection_write_lock(lp, collection="kb", op="a test", timeout=5.0,
                               stale_after=3600):
        assert json.loads(lp.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_the_real_pid_of_a_live_process_is_read_as_alive(base):
    """The control for the test above: the SAME pid with its REAL start time is
    a living holder, so the identical record is not reclaimed early."""
    lp = lock_path_for(base / "kb")
    if cl._create_time(os.getpid()) is None:
        pytest.skip("no psutil: the (pid, create_time) pin is inert by design")
    _hold(lp, _record(pid=os.getpid(), pid_create_time=cl._create_time(os.getpid()),
                      machine=cl._machine_id()),
          silent_for=cl.DEAD_HOLDER_GRACE + 5)

    with pytest.raises(CollectionLockedError):
        with collection_write_lock(lp, collection="kb", op="a test",
                                   timeout=0.3, stale_after=3600):
            pass


def test_a_pid_from_another_pid_space_is_never_judged_dead(base):
    """A pid only means something inside one pid table. A LOCALM_HOME shared
    across a boundary that has its own pids (a network share, or the Windows
    side of a WSL mount, where the Linux hostname defaults to the Windows
    machine name) would otherwise let each side look the other's pids up in its
    own process table, find nothing, and reclaim a perfectly live holder early."""
    foreign = _record(pid=os.getpid(), pid_create_time=1.0,
                      machine="a-different-pid-space")
    assert cl._holder_liveness(foreign) == "unknown"

    lp = lock_path_for(base / "kb")
    _hold(lp, foreign, silent_for=cl.DEAD_HOLDER_GRACE + 5)
    with pytest.raises(CollectionLockedError):
        with collection_write_lock(lp, collection="kb", op="a test",
                                   timeout=0.3, stale_after=3600):
            pass


def test_the_machine_id_separates_two_pid_spaces_on_one_host(monkeypatch):
    """The id must not be the hostname alone, or WSL and Windows on one box
    (same hostname, unrelated pid tables) would trust each other's pids."""
    monkeypatch.setattr(cl, "_machine_id_cache", None)
    monkeypatch.setattr(cl.sys, "platform", "win32")
    win = cl._machine_id()
    monkeypatch.setattr(cl, "_machine_id_cache", None)
    monkeypatch.setattr(cl.sys, "platform", "linux")
    lin = cl._machine_id()
    assert win != lin, "one hostname on two platforms produced one machine id"


# --------------------------------------------------------------------------- #
#  3. Fail safe: never fall through to an unserialised write                  #
# --------------------------------------------------------------------------- #

def test_a_corrupt_lock_record_is_held_not_free(base):
    """A lock file we cannot parse means we do not know who holds it or whether
    they are alive. Treating that as "free" would let a second writer in exactly
    when the on-disk state is already suspect, so it counts as HELD - until the
    file itself goes stale, which is what stops a corrupt file wedging it."""
    lp = lock_path_for(base / "kb")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(CollectionLockedError) as e:
        with collection_write_lock(lp, collection="kb", op="a test",
                                   timeout=0.3, stale_after=3600):
            pass
    assert str(lp) in str(e.value), (
        "a holder that cannot identify itself must at least name the lock file, "
        "or the user has no way to act on the refusal")

    when = time.time() - 3600
    os.utime(lp, (when, when))
    with collection_write_lock(lp, collection="kb", op="a test", timeout=5.0):
        pass          # stale by its own mtime, so recoverable rather than wedged


def test_a_refused_write_changes_nothing_on_disk(base, docs, monkeypatch):
    """The fail-safe contract: when the lock cannot be had, add_paths raises and
    the collection is untouched. It must never proceed unserialised."""
    monkeypatch.setattr(cl, "WAIT_TIMEOUT", 0.3)
    coll = Collection("kb", base=base)
    coll.create()
    coll.add_paths([docs / "alpha.txt"])
    before = (base / "kb" / "meta.json").read_text(encoding="utf-8")
    chunks_before = (base / "kb" / "chunks.jsonl").read_text(encoding="utf-8")

    _hold(lock_path_for(base / "kb"), _record())
    with pytest.raises(CollectionLockedError):
        coll.add_paths([docs / "beta.txt"])

    assert (base / "kb" / "meta.json").read_text(encoding="utf-8") == before
    assert (base / "kb" / "chunks.jsonl").read_text(encoding="utf-8") == chunks_before
    assert "beta.txt" not in before


def test_every_write_entry_point_is_covered(base, docs, monkeypatch):
    """Not just add_paths: resync, remove_doc, add_uploads, create and delete all
    take the lock, so the CLI, the API and the scheduler are covered by
    construction rather than by each call site remembering to."""
    from localm.rag import delete_collection
    monkeypatch.setattr(cl, "WAIT_TIMEOUT", 0.3)
    coll = Collection("kb", base=base)
    coll.create()
    coll.add_paths([docs])
    before = (base / "kb" / "meta.json").read_text(encoding="utf-8")
    source = next(iter(json.loads(before)["docs"]))

    _hold(lock_path_for(base / "kb"), _record())
    with pytest.raises(CollectionLockedError):
        coll.resync(policy=None)
    with pytest.raises(CollectionLockedError):
        coll.remove_doc(source)
    with pytest.raises(CollectionLockedError):
        coll.add_uploads([{"filename": "note.txt", "data": b"hello"}])
    with pytest.raises(CollectionLockedError):
        delete_collection("kb", base=base)

    assert (base / "kb" / "meta.json").read_text(encoding="utf-8") == before
    assert (base / "kb" / "meta.json").is_file(), "a refused delete removed it"


def test_creating_a_collection_takes_the_lock_too(base, monkeypatch):
    """Creating is a write. Without the lock a `rag add` that finds no
    collection can drop a fresh meta.json into a directory another process is
    part-way through deleting - resurrecting a deleted collection and breaking
    that delete's final rmdir."""
    monkeypatch.setattr(cl, "WAIT_TIMEOUT", 0.3)
    _hold(lock_path_for(base / "kb"), _record(op="a delete"))

    with pytest.raises(CollectionLockedError):
        Collection("kb", base=base).create()
    assert not (base / "kb" / "meta.json").exists()


def test_release_does_not_delete_a_lock_that_was_taken_over(base, capsys):
    """If our hold is reclaimed while we are still inside it, our own release
    must not delete the NEW holder's lock - that would let a third writer in
    while the second believes it has exclusive access, rebuilding the race
    through the recovery path (config.py learned this one the hard way)."""
    lp = lock_path_for(base / "kb")
    cm = collection_write_lock(lp, collection="kb", op="a test", timeout=5.0)
    cm.__enter__()
    successor = _record(token="succ" * 8)
    lp.write_text(json.dumps(successor), encoding="utf-8")

    cm.__exit__(None, None, None)

    assert lp.exists(), "our release deleted the successor's live lock"
    assert json.loads(lp.read_text(encoding="utf-8"))["token"] == successor["token"]
    assert "taken over" in capsys.readouterr().err
    lp.unlink()


def test_release_does_not_delete_a_lock_it_cannot_read(base, capsys):
    """The same fence, for the case that has no token to compare: a successor
    that has created its lock file but not yet written its record into it (two
    syscalls), or a transient read failure. "I cannot read it" is never taken
    as "it must be mine"."""
    lp = lock_path_for(base / "kb")
    cm = collection_write_lock(lp, collection="kb", op="a test", timeout=5.0)
    cm.__enter__()
    lp.write_text("", encoding="utf-8")          # a successor mid-create

    cm.__exit__(None, None, None)

    assert lp.exists(), (
        "release deleted a lock whose record it could not read - that can be a "
        "live successor's, and deleting it lets a third writer in")
    assert "no longer readable" in capsys.readouterr().err
    lp.unlink()


def test_the_lock_file_sits_beside_the_collection_not_inside_it(base):
    """It must survive a delete of the collection directory (an inside lock
    would be rmtree'd out from under its holder) and must never be mistaken for
    a collection."""
    coll = Collection("kb", base=base)
    coll.create()
    with collection_write_lock(lock_path_for(base / "kb"),
                               collection="kb", op="a test"):
        assert (base / "kb.lock").is_file()
        assert not any(p.name.endswith(".lock") for p in (base / "kb").iterdir())
        assert collection_names(base) == ["kb"]


def test_concurrent_threads_in_one_process_still_serialise(base, docs):
    """The cross-process lock is taken INSIDE the per-process one, so two
    threads of the same process queue on the in-process lock and never meet
    their own process's lock file (which a file lock can only read as a nested
    call, the way config's does). Both writes must land."""
    errors: list = []

    def _index(doc):
        try:
            c = Collection("kb", base=base)
            c.create()
            c.add_paths([doc])
        except BaseException as e:            # reported, never swallowed
            errors.append(e)

    threads = [threading.Thread(target=_index, args=(docs / n,))
               for n in ("alpha.txt", "beta.txt")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"a same-process concurrent write failed: {errors}"
    meta = json.loads((base / "kb" / "meta.json").read_text(encoding="utf-8"))
    assert {os.path.basename(k) for k in meta["docs"]} == {"alpha.txt", "beta.txt"}


# --------------------------------------------------------------------------- #
#  4. How each surface reports a refusal                                      #
# --------------------------------------------------------------------------- #

def _run_cli(args, home, *, wait: str = "0.5"):
    """Run a real `localm` CLI command in its own OS process."""
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env[cl.ENV_WAIT] = wait
    env["COLUMNS"] = "400"          # keep rich from wrapping the message we assert on
    return subprocess.run(
        [sys.executable, "-c", "from localm.cli import main; main()", *args],
        env=env, capture_output=True, text=True, timeout=180)


def test_cli_waits_then_refuses_naming_the_holder(heavy_slot, tmp_path, docs):
    """`localm rag resync` against a collection another process is writing must
    WAIT (and say so), then REFUSE with a message naming the holder and a
    non-zero exit - never block for ever, never interleave silently."""
    home = tmp_path / "home"
    home.mkdir()
    rag = home / "rag"
    coll = Collection("kb", base=rag)
    coll.create()
    coll.add_paths([docs])
    before = (rag / "kb" / "meta.json").read_text(encoding="utf-8")
    (docs / "gamma.txt").write_text("gamma content about bearings", encoding="utf-8")

    # THIS process holds the lock while a genuinely separate process runs the CLI.
    with collection_write_lock(lock_path_for(rag / "kb"),
                               collection="kb", op="a re-sync"):
        started = time.time()
        r = _run_cli(["rag", "resync", "kb"], home, wait="2")
        waited = time.time() - started

    out = _flat(r.stdout)
    assert r.returncode != 0, f"the CLI wrote anyway:\n{r.stdout}\n{r.stderr}"
    assert "is being written by" in out, r.stdout
    assert f"pid {os.getpid()}" in out, (
        f"the refusal must name who holds the collection:\n{r.stdout}")
    assert "waiting for the write lock" in out, (
        f"a bounded wait must say it is waiting:\n{r.stdout}")
    assert 1.0 < waited < 120, f"the CLI did not wait its budget ({waited:.1f}s)"
    assert (rag / "kb" / "meta.json").read_text(encoding="utf-8") == before, (
        "the refused CLI run still modified the collection")


def test_cli_succeeds_once_the_holder_releases(heavy_slot, tmp_path, docs):
    """The other half of the bounded wait: a CLI that waits out a SHORT hold
    goes on to do its work. A lock that only ever refuses would be useless."""
    home = tmp_path / "home"
    home.mkdir()
    rag = home / "rag"
    coll = Collection("kb", base=rag)
    coll.create()
    coll.add_paths([docs])
    (docs / "gamma.txt").write_text("gamma content about bearings", encoding="utf-8")

    released = threading.Event()

    def _hold_briefly():
        with collection_write_lock(lock_path_for(rag / "kb"),
                                   collection="kb", op="a re-sync"):
            time.sleep(4.0)         # comfortably longer than the child's start-up
        released.set()

    t = threading.Thread(target=_hold_briefly)
    t.start()
    time.sleep(0.2)                 # make sure the holder got there first
    try:
        r = _run_cli(["rag", "resync", "kb"], home, wait="120")
    finally:
        t.join(timeout=60)

    assert released.is_set()
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "waiting for the write lock" in _flat(r.stdout), r.stdout
    meta = json.loads((rag / "kb" / "meta.json").read_text(encoding="utf-8"))
    assert any(k.endswith("gamma.txt") for k in meta["docs"]), (
        "the CLI acquired the lock but did not do the re-sync")


def test_a_scheduled_job_reports_the_refusal_instead_of_writing(tmp_path, docs,
                                                                monkeypatch):
    """The unattended surface. A scheduled re-sync that meets a hand-run write
    records WHY it did nothing, rather than showing a clean run in the job
    history."""
    from localm.plugins.builtin.jobs import runner
    rag = tmp_path / "rag"
    monkeypatch.setattr("localm.rag.store.rag_dir", lambda: rag)
    monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)
    monkeypatch.setattr(cl, "WAIT_TIMEOUT", 0.3)
    coll = Collection("kb", base=rag)
    coll.create()
    coll.add_paths([docs])
    before = (rag / "kb" / "meta.json").read_text(encoding="utf-8")

    _hold(lock_path_for(rag / "kb"), _record(op="an index"))
    job = types.SimpleNamespace(id="j1", task_kind="rag", collection="kb",
                                model=None, prompt="")
    result = runner.run_job(job)

    assert result["status"] == "error", result
    assert "is being written by" in _flat(result["error"]), result["error"]
    assert "next scheduled run" in _flat(result["error"]), (
        "an unattended reader needs to know the folder is not abandoned")
    assert (rag / "kb" / "meta.json").read_text(encoding="utf-8") == before


def test_the_api_answers_409_rather_than_500(base):
    """A locked collection is a conflict, not a server fault: the HTTP surface
    must say so, with the message that names the holder."""
    from fastapi import HTTPException

    from localm.plugins.builtin.rag import plug

    def _locked():
        raise CollectionLockedError("kb", _record(), 30.0)

    async def _go():
        with pytest.raises(HTTPException) as e:
            await plug._write_off_loop(_locked)
        return e.value

    exc = asyncio.run(_go())
    assert exc.status_code == 409
    assert "is being written by" in _flat(str(exc.detail))


def test_env_override_reports_a_malformed_value(monkeypatch):
    """An operator override that cannot be read is reported, not silently
    ignored - a typo that quietly reverts to the default is how a setting looks
    like it worked."""
    monkeypatch.setenv(cl.ENV_WAIT, "soon please")
    warned: list = []
    monkeypatch.setattr(cl._log, "warning",
                        lambda *a, **k: warned.append(a[0] % a[1:]))
    assert cl._env_float(cl.ENV_WAIT, 30.0) == 30.0
    assert warned and "not a positive number" in warned[0]


# --------------------------------------------------------------------------
# A TRANSIENT ACCESS-DENIED ON THE LOCK FILE IS A WAIT, NOT A CRASH (Windows).
#
# On Windows a lock file that exists but is momentarily inaccessible (the
# holder's unlink in flight, a scanner holding a handle) reports
# ERROR_ACCESS_DENIED from the O_CREAT|O_EXCL create, not the
# ERROR_FILE_EXISTS that becomes FileExistsError.
#
# The fault is injected at os.open rather than by racing two real processes,
# whose window is microseconds wide. What is asserted is the CONTRACT: a
# PermissionError from the create is retried within the deadline.

def _permission_denied_then_real(lockpath, times):
    """Fail the create for *lockpath* `times` times with PermissionError, then
    behave normally. Every other path is untouched."""
    real = os.open
    state = {"left": times}

    def fake(path, flags, *a, **kw):
        if str(path) == str(lockpath) and (flags & os.O_CREAT) and state["left"]:
            state["left"] -= 1
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, flags, *a, **kw)

    return fake, state


@pytest.mark.skipif(os.name != "nt", reason="the retry is deliberately Windows-only")
def test_transient_access_denied_on_the_lock_file_is_waited_out(tmp_path, monkeypatch):
    rag = tmp_path / "rag"
    rag.mkdir()
    lp = lock_path_for(rag / "kb")
    fake, state = _permission_denied_then_real(lp, 3)
    monkeypatch.setattr(os, "open", fake)

    with collection_write_lock(lp, collection="kb", op="a re-sync", timeout=30):
        assert lp.exists(), "the lock was acquired, so its file must be present"
    assert state["left"] == 0, "the injected access-denied never fired"


@pytest.mark.skipif(os.name != "nt", reason="the retry is deliberately Windows-only")
def test_access_denied_that_never_clears_reports_a_lock_timeout_not_a_crash(
        tmp_path, monkeypatch):
    """Bounded: it must never wait longer than the caller's timeout, and what
    surfaces is the normal lock error, not a raw PermissionError."""
    rag = tmp_path / "rag"
    rag.mkdir()
    lp = lock_path_for(rag / "kb")
    fake, _ = _permission_denied_then_real(lp, 10_000)
    monkeypatch.setattr(os, "open", fake)

    started = time.time()
    with pytest.raises(CollectionLockedError):
        with collection_write_lock(lp, collection="kb", op="a re-sync", timeout=2):
            pass
    assert time.time() - started < 30, "the bounded wait was not bounded"
