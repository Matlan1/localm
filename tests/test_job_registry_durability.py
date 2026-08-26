# SPDX-License-Identifier: AGPL-3.0-or-later
"""The in-flight job registry survives a restart.

What these tests pin:

  * the record is written at the LIFECYCLE points and not on progress. A
    per-second progress tick must not rewrite the whole file.
  * a row that was still running is reported as "interrupted", NOT "failed".
    "I lost the connection" and "the work failed" are different claims: a pull
    that was 99% done may well have finished, and "failed" would assert
    something nobody measured.
  * a LIVE writer's file is left completely alone. Several localm servers may
    share one data dir, and adopting a running server's operations would make
    each of them report the other's work as interrupted.
  * the corrupt-file posture matches the sibling scheduled-jobs store: quarantine
    a copy, redact the owner digest to a NON-NULL sentinel, keep the newest few,
    warn loudly. Nulling the owner would make job_owner_ok treat every recovered
    row as unowned, i.e. streamable and cancellable by any caller.
  * a recovered job's event stream terminates. Without a seeded end frame a client
    that reattaches by id waits on keepalives forever, because the worker thread
    that would have pushed it died with the previous process.

TIMING: the terminal write happens on the job's worker thread and takes about
25 ms (restrict_file_perms shells out to icacls on Windows), so it is NOT
complete the instant job.status flips. os.replace is atomic, so a read inside
that window legitimately returns the PREVIOUS file. Every assertion about
persisted state therefore polls - see _wait_for_row - rather than sleeping a
fixed amount or asserting immediately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from localm.plugins.gui import jobs as J


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """An activity dir under tmp_path, pinned so nothing touches a real one."""
    root = tmp_path / "activity"
    monkeypatch.setattr(J, "activity_dir", lambda: root)
    return root


def _rows(path: Path) -> list:
    """The operations in a record file, retrying a transient Windows share lock.

    A file written microseconds ago can still be held by the ACL tool or a
    scanner; that is the documented transient case
    (config._is_transient_permission_error), not a defect in what is under
    test."""
    for _ in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8"))["operations"]
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))["operations"]


def _wait_for_row(store, job_id, *, status=None, timeout=10.0):
    """The persisted row for *job_id* once it reaches *status*, else fail.

    Polls rather than sleeping a fixed amount - see this module's timing note."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if store.path.is_file():
            rows = [r for r in _rows(store.path) if r.get("id") == job_id]
            if rows:
                last = rows[0]
                if status is None or last.get("status") == status:
                    return last
        time.sleep(0.02)
    pytest.fail(f"row {job_id} never reached status={status!r}; last seen {last!r}")


def _wait_done(job, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline and job.status == "running":
        time.sleep(0.01)
    return job.status


def _write_record(root: Path, *, pid: int, rows: list, run: str = "0123456789ab") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{pid}-{run}.json"
    path.write_text(json.dumps({"version": 1, "pid": pid, "operations": rows}),
                    encoding="utf-8")
    return path


def _row(**over) -> dict:
    base = {"id": "aaaaaaaaaaaa", "kind": "pull", "label": "Pull tiny/model.gguf",
            "status": "running", "created_at": time.time() - 60,
            "finished_at": None, "returncode": None, "result": None,
            "owner": None, "child_pid": None}
    base.update(over)
    return base


DEAD_PID = 999_997          # never a live process; asserted below


# --------------------------------------------------------------------------- #
#  the record is written, and written at the right moments
# --------------------------------------------------------------------------- #

class TestPersistence:

    def test_a_started_job_is_recorded_immediately(self, store_root):
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: True, label="Smoke", owner="a" * 64)
        row = _wait_for_row(m._store, job.id)
        assert row["kind"] == "smoke"
        assert row["label"] == "Smoke"
        assert row["owner"] == "a" * 64, (
            "the owner digest must be persisted: without it job_owner_ok treats a "
            "recovered row as unowned, i.e. reachable by any caller")

    def test_the_terminal_status_reaches_disk(self, store_root):
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: True)
        assert _wait_done(job) == "done"
        row = _wait_for_row(m._store, job.id, status="done")
        assert row["finished_at"] is not None, (
            "finished_at must be persisted; the TTL sweep keys on it")

    def test_a_failure_is_recorded_as_failed_not_interrupted(self, store_root):
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: False)
        assert _wait_done(job) == "failed"
        _wait_for_row(m._store, job.id, status="failed")

    def test_argv_is_never_persisted(self, store_root):
        """summary() withholds argv from clients because it carries the resolved
        model spec and any host path the caller passed. A durable file has no
        business holding what the API refuses to hand out."""
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: True)
        row = _wait_for_row(m._store, job.id)
        assert "argv" not in row
        text = m._store.path.read_text(encoding="utf-8")
        assert "argv" not in text

    def test_progress_does_not_rewrite_the_record(self, store_root):
        """A per-second tick must not rewrite the whole file. Counted, not
        eyeballed - a wall-clock or file-mtime check could not tell "no write"
        from "a write that happened to be fast"."""
        m = J.JobManager()
        job = m.start_fn("prog", lambda j: True)
        _wait_for_row(m._store, job.id, status="done")
        writes = []
        real = m._store.write
        m._store.write = lambda rows: (writes.append(1), real(rows))[1]
        for i in range(25):
            job.progress(phase="working", done=i, total=25)
        assert writes == [], (
            f"{len(writes)} store write(s) triggered by progress events; a "
            "download ticking every second would rewrite the whole record file "
            "each time")

    def test_an_empty_registry_removes_the_file(self, store_root):
        """Otherwise a data dir collects one abandoned file per process that ever
        started a job."""
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: True)
        _wait_for_row(m._store, job.id, status="done")
        with m._lock:
            m._jobs.clear()
            m._persist_locked()
        assert not m._store.path.exists()

    def test_the_record_file_is_owner_restricted(self, store_root):
        m = J.JobManager()
        job = m.start_fn("smoke", lambda j: True)
        _wait_for_row(m._store, job.id)
        if os.name == "posix":
            mode = m._store.path.stat().st_mode & 0o777
            assert mode == 0o600, oct(mode)
        else:
            # icacls output is the only reliable read of a Windows ACL; assert
            # the call path was taken rather than reading the ACL back.
            out = subprocess.run(["icacls", str(m._store.path)],
                                 capture_output=True, text=True).stdout
            user = os.environ.get("USERNAME", "")
            assert user and user.lower() in out.lower(), out
            assert "BUILTIN\\Users" not in out, (
                "inherited ACEs were not dropped; the owner key digest in this "
                f"file would be readable by another local account: {out}")


# --------------------------------------------------------------------------- #
#  startup reconciliation
# --------------------------------------------------------------------------- #

class TestReconciliation:

    def test_dead_pid_is_a_precondition_of_these_tests(self):
        """A guard, not a test of the product: if this pid were alive, every
        adoption test below would silently exercise the leave-it-alone path and
        pass for the wrong reason."""
        from localm.instances import pid_alive
        assert not pid_alive(DEAD_PID)

    def test_a_running_row_from_a_dead_writer_becomes_interrupted(self, store_root):
        _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        m = J.JobManager()
        snap = m.snapshot()
        assert [s["status"] for s in snap] == ["interrupted"], snap
        assert snap[0]["label"] == "Pull tiny/model.gguf"
        assert snap[0]["cancellable"] is False

    def test_interrupted_is_not_failed(self, store_root):
        """A row from a dead writer is reported as "interrupted", never as
        "failed"."""
        _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        m = J.JobManager()
        assert m.get("aaaaaaaaaaaa").status == "interrupted"
        assert m.get("aaaaaaaaaaaa").status != "failed"

    def test_a_finished_row_keeps_its_real_status(self, store_root):
        _write_record(store_root, pid=DEAD_PID, rows=[
            _row(id="bbbbbbbbbbbb", status="done", finished_at=time.time() - 5),
            _row(id="cccccccccccc", status="failed", finished_at=time.time() - 5),
            _row(id="dddddddddddd", status="cancelled", finished_at=time.time() - 5),
        ])
        m = J.JobManager()
        got = {s["id"]: s["status"] for s in m.snapshot()}
        assert got == {"bbbbbbbbbbbb": "done", "cccccccccccc": "failed",
                       "dddddddddddd": "cancelled"}, got

    def test_an_interrupted_row_is_stamped_at_detection_so_the_ttl_keeps_it(
            self, store_root):
        """_gc sweeps on finished_at, so stamping a recovered row with anything
        from before the crash would evict it the instant it was recovered. A
        two-hour pull interrupted seconds ago must still be visible after the
        restart."""
        old = time.time() - 3 * J.JobManager._TTL_S
        _write_record(store_root, pid=DEAD_PID, rows=[_row(created_at=old)])
        m = J.JobManager()
        snap = m.snapshot()
        assert len(snap) == 1, (
            "a long-running operation interrupted by a restart was swept before "
            f"anyone could see it: {snap}")
        assert snap[0]["finished_at"] >= old + J.JobManager._TTL_S
        assert snap[0]["created_at"] == pytest.approx(old), (
            "created_at must be preserved: a client renders the operation's real "
            "age from it")

    def test_an_expired_finished_row_is_still_swept(self, store_root):
        """The stamp above must not turn the TTL off: a row that finished hours
        ago is history, not activity."""
        long_ago = time.time() - 5 * J.JobManager._TTL_S
        _write_record(store_root, pid=DEAD_PID, rows=[
            _row(status="done", created_at=long_ago, finished_at=long_ago)])
        m = J.JobManager()
        assert m.snapshot() == []

    def test_the_adopted_file_is_removed(self, store_root):
        path = _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        m = J.JobManager()
        assert not path.exists(), (
            "an adopted file left behind lets a third process adopt the same "
            "rows again and report them twice")
        assert m._store.path.is_file(), "the adopted rows were not re-persisted"
        assert [r["id"] for r in _rows(m._store.path)] == ["aaaaaaaaaaaa"]

    def test_a_live_writers_file_is_left_completely_alone(self, store_root):
        """Two localm servers can share a data dir. Uses a REAL live process: pid
        1 does not exist on Windows, so hardcoding it would make this test
        silently exercise the DEAD path and pass for the wrong reason."""
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            from localm.instances import pid_alive
            assert pid_alive(sleeper.pid), "the helper process is not alive"
            path = _write_record(store_root, pid=sleeper.pid, rows=[_row()])
            before = path.read_text(encoding="utf-8")
            m = J.JobManager()
            assert path.exists(), "a live server's record file was deleted"
            assert path.read_text(encoding="utf-8") == before
            assert m.snapshot() == [], (
                "another live server's in-flight operations were adopted and "
                "would now be reported as interrupted")
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=10)

    def test_a_sibling_managers_file_is_left_alone(self, store_root):
        """pid liveness cannot answer the same-process question: this process is
        obviously alive. Two managers can coexist (gui/web.py's fallback creates
        one), and the second must not swallow the first's live operations."""
        first = J.JobManager()
        job = first.start_fn("smoke", lambda j: True)
        _wait_for_row(first._store, job.id)
        second = J.JobManager()
        assert first._store.path.exists()
        assert second.get(job.id) is None, (
            "a second manager adopted a live sibling's in-flight job")
        assert first._store.path != second._store.path

    def test_a_row_that_cannot_be_parsed_is_skipped_not_fatal(self, store_root):
        _write_record(store_root, pid=DEAD_PID, rows=[
            {"id": "eeeeeeeeeeee"},                      # no status/created_at
            _row(id="ffffffffffff", status="nonsense"),  # unknown status
            _row(id="111111111111", created_at="soon"),  # unusable timestamp
            _row(id="222222222222"),                     # the good one
        ])
        m = J.JobManager()
        assert [s["id"] for s in m.snapshot()] == ["222222222222"]

    def test_a_recovered_job_has_a_terminal_event_stream(self, store_root):
        """Without this a client reattaching by id waits on keepalives forever:
        the worker thread that would have pushed the end frame died with the
        previous process."""
        _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        m = J.JobManager()
        history = list(m.get("aaaaaaaaaaaa")._history)
        assert history[-1]["type"] == "end"
        assert history[-1]["status"] == "interrupted"
        assert any(e["type"] == "line" for e in history), (
            "the stream should say WHY it ends, not just that it does")

    def test_a_recovered_job_keeps_its_result_and_returncode(self, store_root):
        _write_record(store_root, pid=DEAD_PID, rows=[_row(
            status="done", finished_at=time.time(), returncode=0,
            result="C:/out/image.png")])
        m = J.JobManager()
        job = m.get("aaaaaaaaaaaa")
        assert job.returncode == 0
        assert job.result == "C:/out/image.png"
        assert list(job._history)[-1]["result"] == "C:/out/image.png"

    def test_a_still_alive_child_pid_is_reported(self, store_root, caplog):
        """Adopting or killing an orphan is out of scope, but a process still
        running with nothing tracking it must not be invisible."""
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            _write_record(store_root, pid=DEAD_PID,
                          rows=[_row(child_pid=sleeper.pid)])
            with caplog.at_level("WARNING", logger="localm"):
                J.JobManager()
            assert any(str(sleeper.pid) in r.getMessage()
                       for r in caplog.records), caplog.text
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=10)

    def test_reconcile_can_be_turned_off(self, store_root):
        _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        m = J.JobManager(reconcile=False)
        assert m.snapshot() == []


# --------------------------------------------------------------------------- #
#  corrupt / unreadable files
# --------------------------------------------------------------------------- #

class TestCorruptFiles:

    def test_a_corrupt_file_is_quarantined_and_warned_about(self, store_root, caplog):
        store_root.mkdir(parents=True, exist_ok=True)
        bad = store_root / f"{DEAD_PID}-abcabcabcabc.json"
        bad.write_text('{"operations": [ {"id": "x"} TRUNCATED', encoding="utf-8")
        with caplog.at_level("WARNING", logger="localm"):
            m = J.JobManager()
        copies = list(store_root.glob("*.json.corrupt-*"))
        assert len(copies) == 1, [c.name for c in copies]
        assert m.snapshot() == []
        assert "corrupt" in caplog.text.lower(), caplog.text

    def test_the_quarantine_copy_does_not_carry_the_owner_digest(self, store_root):
        """The copy is made from a file that failed to parse, so the redaction has
        to work on raw text. The replacement is a non-null SENTINEL: job_owner_ok
        treats owner=None as unowned and therefore unrestricted, so dropping the
        field would turn a recovery artefact into an open ACL."""
        digest = "d" * 64
        store_root.mkdir(parents=True, exist_ok=True)
        bad = store_root / f"{DEAD_PID}-abcabcabcabc.json"
        bad.write_text('{"operations": [{"owner": "%s"} BROKEN' % digest,
                       encoding="utf-8")
        J.JobManager()
        copy = next(iter(store_root.glob("*.json.corrupt-*")))
        text = copy.read_text(encoding="utf-8")
        assert digest not in text
        assert J._REDACTED_OWNER in text

    def test_quarantine_copies_are_capped(self, store_root):
        """Nothing else ever deletes these, and per-writer file names would give
        every pid its own unbounded series."""
        store_root.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            (store_root / f"{DEAD_PID}-run{i}.json.corrupt-{1000 + i}").write_text(
                "{}", encoding="utf-8")
        bad = store_root / f"{DEAD_PID}-abcabcabcabc.json"
        bad.write_text("not json at all", encoding="utf-8")
        J.JobManager()
        left = sorted(p.name for p in store_root.glob("*.json.corrupt-*"))
        assert len(left) == J._QUARANTINE_KEEP, left
        assert all("-1000" not in n for n in left), (
            f"pruning kept the OLDEST copies: {left}")

    def test_a_file_with_no_operations_list_is_quarantined(self, store_root):
        store_root.mkdir(parents=True, exist_ok=True)
        bad = store_root / f"{DEAD_PID}-abcabcabcabc.json"
        bad.write_text('{"version": 1}', encoding="utf-8")
        J.JobManager()
        assert list(store_root.glob("*.json.corrupt-*"))

    def test_an_unreadable_file_is_reported_and_left_in_place(self, store_root,
                                                             monkeypatch, caplog):
        """Not collapsed to empty and not raised: a process only ever writes its
        OWN file, so a file it could not read is never one it is about to
        overwrite. Reported, then left for a later run."""
        path = _write_record(store_root, pid=DEAD_PID, rows=[_row()])
        real = Path.read_text

        def _boom(self, *a, **kw):
            if self == path:
                raise OSError("locked by another process")
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", _boom)
        with caplog.at_level("WARNING", logger="localm"):
            m = J.JobManager()
        assert m.snapshot() == []
        assert path.exists(), "an unreadable file must not be deleted"
        assert "unreadable" in caplog.text.lower(), caplog.text

    def test_a_store_write_failure_never_breaks_the_job(self, store_root,
                                                        monkeypatch, caplog):
        """Persistence is a convenience layered on top of an operation. A full
        disk must not fail a model pull - but it must not be silent either."""
        m = J.JobManager()
        # Injected at the LEAF the store actually depends on, not at the store's
        # own _write: patching _write replaces the guarded body, so the test would
        # be measuring its own patch instead of the product's handling.
        import localm.config as cfg
        monkeypatch.setattr(cfg, "atomic_write_private",
                            lambda path, text: (_ for _ in ()).throw(
                                OSError("no space left on device")))
        with caplog.at_level("WARNING", logger="localm"):
            job = m.start_fn("smoke", lambda j: True)
            assert _wait_done(job) == "done"
        assert m.get(job.id) is not None, "the job was lost with the write"
        assert "could not persist" in caplog.text.lower(), caplog.text


# --------------------------------------------------------------------------- #
#  the record file itself
# --------------------------------------------------------------------------- #

class TestStoreFile:

    def test_the_path_is_unique_per_store_within_one_process(self, store_root):
        """Two managers sharing one path both write through
        config.atomic_write_private's FIXED "<name>.tmp", so one writer's
        os.replace moves the temp out from under the other and terminal writes
        are LOST, leaving the file claiming "running" for a finished job."""
        a, b = J._ActivityStore(store_root), J._ActivityStore(store_root)
        assert a.path != b.path
        assert a.path.parent == b.path.parent

    def test_the_filename_still_encodes_the_writer_pid(self, store_root):
        """The run nonce must not cost the liveness check its input."""
        s = J._ActivityStore(store_root, pid=4242)
        assert J._ActivityStore._pid_of(s.path) == 4242

    def test_a_filename_with_no_pid_is_read_rather_than_guessed_about(self, store_root):
        store_root.mkdir(parents=True, exist_ok=True)
        hand = store_root / "hand-dropped.json"
        hand.write_text(json.dumps({"version": 1, "operations": [
            _row(id="333333333333")]}), encoding="utf-8")
        assert J._ActivityStore._pid_of(hand) is None
        m = J.JobManager()
        assert [s["id"] for s in m.snapshot()] == ["333333333333"]

    def test_concurrent_writers_do_not_lose_the_last_write(self, store_root):
        """The lost-write race above, driven directly: two managers in one process
        writing as fast as they can. Asserts on the FILES (what a restart would
        read), not on a call count."""
        import threading
        managers = [J.JobManager() for _ in range(3)]
        stop = []

        def _hammer(mgr, n):
            for i in range(15):
                job = J.Job(id=f"{n}{i:011d}", kind="k", argv=[])
                with mgr._lock:
                    mgr._jobs[job.id] = job
                    mgr._persist_locked()
            stop.append(n)

        threads = [threading.Thread(target=_hammer, args=(m, n))
                   for n, m in enumerate(managers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert sorted(stop) == [0, 1, 2]
        for n, mgr in enumerate(managers):
            ids = {r["id"] for r in _rows(mgr._store.path)}
            assert len(ids) == 15, (
                f"manager {n} persisted {len(ids)} of its 15 operations; writes "
                "are being lost")
