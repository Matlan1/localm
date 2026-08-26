# SPDX-License-Identifier: AGPL-3.0-or-later
"""JobManager must be able to say what is running, and must keep a job that
just finished.

Two defects:

1. The registry records every job but exposes no way to ENUMERATE them. With
   only get(job_id) and has_running(kind) public, and a job id handed out
   exactly once - in the body of the POST that started the job - a second
   client, or the same browser tab after a reload, cannot learn that a model
   pull is in flight even though the server knows.

2. A TTL sweep keyed on created_at contradicts the class docstring's promise
   that "finished jobs stay queryable for an hour", for any job that RAN for
   longer than the TTL: a two-hour pull is already past the cutoff the moment it
   succeeds, so it is evicted by the very next started job instead of staying
   queryable. finished_at makes the code match the promise.
"""

from __future__ import annotations

import time

import pytest

from localm.plugins.gui.jobs import Job, JobManager


def _finished(mgr: JobManager, *, kind: str = "pull", age_s: float,
              ran_for_s: float = 0.0, status: str = "done") -> Job:
    """Insert a job that finished *age_s* ago after running *ran_for_s*."""
    now = time.time()
    job = Job(id=f"j{len(mgr._jobs)}", kind=kind, argv=[], status=status)
    job.finished_at = now - age_s
    job.created_at = job.finished_at - ran_for_s
    mgr._jobs[job.id] = job
    return job


# ---------------------------------------------------------------- finished_at

def test_finished_at_is_none_while_running():
    job = Job(id="a", kind="pull", argv=[])
    assert job.status == "running"
    assert job.finished_at is None, "a running job has not finished"


def test_finished_at_is_stamped_when_the_worker_leaves():
    mgr = JobManager()
    job = mgr.start_fn("test", lambda j: True)
    for _ in range(200):
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "done", "worker should have completed"
    assert job.finished_at is not None, "finished_at must be stamped on exit"
    assert job.finished_at >= job.created_at


def test_finished_at_is_stamped_even_when_the_job_fails():
    mgr = JobManager()

    def _boom(job):
        raise RuntimeError("nope")

    job = mgr.start_fn("test", _boom)
    for _ in range(200):
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "failed"
    assert job.finished_at is not None, "a failed job still stopped being in flight"


def test_mark_finished_is_idempotent():
    """A second call must not move the stamp forward, or a cancel racing the
    worker's own exit would hand the job a fresh lease on the TTL."""
    job = Job(id="a", kind="pull", argv=[])
    job.mark_finished()
    first = job.finished_at
    time.sleep(0.01)
    job.mark_finished()
    assert job.finished_at == first


# ------------------------------------------------------------------------ _gc

def test_gc_keeps_a_long_job_that_finished_a_moment_ago():
    """Keyed on created_at this job is two hours old and gets swept; keyed on
    finished_at it finished one second ago and must survive."""
    mgr = JobManager()
    job = _finished(mgr, age_s=1.0, ran_for_s=2 * 60 * 60)
    assert job.created_at < time.time() - mgr._TTL_S, "precondition: older than the TTL"
    mgr._gc()
    assert job.id in mgr._jobs, (
        "a job that finished a second ago must stay queryable, however long it ran")


def test_gc_evicts_a_job_that_finished_long_ago():
    mgr = JobManager()
    job = _finished(mgr, age_s=mgr._TTL_S + 60, ran_for_s=1.0)
    mgr._gc()
    assert job.id not in mgr._jobs


def test_gc_never_evicts_a_running_job_at_any_age():
    """Evicting a live job would strand its SSE subscribers and lose the record
    while the work carries on."""
    mgr = JobManager()
    job = Job(id="old", kind="pull", argv=[])
    job.created_at = time.time() - 10 * mgr._TTL_S
    mgr._jobs[job.id] = job
    mgr._gc()
    assert job.id in mgr._jobs
    assert job.finished_at is None


# ------------------------------------------------------------------- snapshot

def test_snapshot_finds_a_job_without_holding_its_id():
    """The whole point: a caller that never saw the POST response can still
    learn the operation exists."""
    mgr = JobManager()
    job = Job(id="secret-id", kind="pull", argv=[], label="Model pull foo/bar")
    mgr._jobs[job.id] = job
    rows = mgr.snapshot()
    assert [r["id"] for r in rows] == ["secret-id"]
    assert rows[0]["kind"] == "pull"
    assert rows[0]["label"] == "Model pull foo/bar"
    assert rows[0]["status"] == "running"
    assert rows[0]["cancellable"] is True


def test_snapshot_is_newest_first():
    mgr = JobManager()
    for i, age in enumerate([30.0, 10.0, 20.0]):
        job = Job(id=f"j{i}", kind="pull", argv=[])
        job.created_at = time.time() - age
        mgr._jobs[job.id] = job
    rows = mgr.snapshot()
    assert [r["id"] for r in rows] == ["j1", "j2", "j0"]


def test_snapshot_includes_finished_jobs_with_their_finish_time():
    mgr = JobManager()
    _finished(mgr, age_s=5.0, ran_for_s=60.0, status="failed")
    row = mgr.snapshot()[0]
    assert row["status"] == "failed"
    assert row["finished_at"] is not None
    assert row["cancellable"] is False, "a finished job cannot be cancelled"


def test_snapshot_never_leaks_argv_or_owner():
    """argv carries the resolved model spec and any host path the caller
    passed, and owner is a keystore hash. Neither belongs in a listing."""
    mgr = JobManager()
    job = Job(id="a", kind="pull", argv=["python", "-m", "localm", "pull",
                                         "D:/some/host/path/model.gguf"],
              owner="deadbeefcafe")
    mgr._jobs[job.id] = job
    row = mgr.snapshot()[0]
    assert "argv" not in row
    assert "owner" not in row
    assert "deadbeefcafe" not in repr(row)


# -------------------------------------------------------------------- summary

def test_summary_omits_pct_entirely_when_nothing_reported_progress():
    """A pull that has not yet read a byte count is at an UNKNOWN percentage,
    not at 0 percent. Absent, never a fabricated zero."""
    job = Job(id="a", kind="pull", argv=[])
    row = job.summary()
    assert "pct" not in row
    assert "phase" not in row


def test_summary_reports_the_latest_progress():
    job = Job(id="a", kind="pull", argv=[])
    job.push({"type": "progress", "pct": 12.5, "phase": "download"})
    job.push({"type": "line", "text": "noise"})
    job.push({"type": "progress", "pct": 61.0, "phase": "download"})
    row = job.summary()
    assert row["pct"] == 61.0
    assert row["phase"] == "download"


def test_summary_omits_a_non_numeric_pct():
    job = Job(id="a", kind="pull", argv=[])
    job.push({"type": "progress", "pct": None, "phase": "download"})
    row = job.summary()
    assert "pct" not in row, "an unknown percentage must not be rendered"
    assert row["phase"] == "download"


def test_label_is_carried_from_host_label():
    mgr = JobManager()
    job = mgr.start_fn("test", lambda j: True, label="Warming the embedder")
    assert job.label == "Warming the embedder"
    assert mgr.snapshot()[0]["label"] == "Warming the embedder"


@pytest.mark.parametrize("status", ["done", "failed", "cancelled"])
def test_only_a_running_job_is_cancellable(status):
    job = Job(id="a", kind="pull", argv=[], status=status)
    assert job.summary()["cancellable"] is False
