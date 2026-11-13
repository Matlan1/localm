# SPDX-License-Identifier: AGPL-3.0-or-later
"""The end of an in-thread (``start_fn``) job reaches the ``localm`` log: a
failed job as a WARNING with its id, kind and cause, any other end as a DEBUG
line."""

from __future__ import annotations

import logging
import time

from localm import jobmanager as gj


def _end_record(caplog, job, timeout=3.0):
    """The log record naming *job*'s end, waiting for the worker thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for r in caplog.records:
            if r.name == "localm" and r.getMessage().startswith(f"job {job.id} ("):
                return r
        time.sleep(0.01)
    raise AssertionError(f"no end line for job {job.id}: "
                         f"{[r.getMessage() for r in caplog.records]}")


def test_a_failed_job_logs_a_warning_with_its_outcome_line(caplog):
    def fn(job):
        job.push({"type": "line", "text": "Image backend: native."})
        job.push({"type": "line", "outcome": True, "text": "No native image model is set up."})
        job.push({"type": "line", "text": "a later line"})
        return False

    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("imagine", fn)
        rec = _end_record(caplog, job)
    assert job.status == "failed"
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == (
        f"job {job.id} (imagine) failed: No native image model is set up.")


def test_a_job_that_raises_logs_the_exception_as_its_cause(caplog):
    def fn(job):
        job.push({"type": "line", "outcome": True, "text": "not the cause"})
        raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("music", fn)
        rec = _end_record(caplog, job)
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == f"job {job.id} (music) failed: RuntimeError: boom"


def test_a_failed_job_without_an_outcome_line_logs_its_last_line(caplog):
    def fn(job):
        job.push({"type": "line", "text": "first"})
        job.push({"type": "line", "text": "Could not warm up the embedder:\n  none"})
        return False

    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("embedding-warmup", fn)
        rec = _end_record(caplog, job)
    assert rec.getMessage() == (
        f"job {job.id} (embedding-warmup) failed: Could not warm up the embedder: none")


def test_a_failed_job_with_no_lines_says_no_cause_was_reported(caplog):
    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("rag-index", lambda job: False)
        rec = _end_record(caplog, job)
    assert rec.getMessage() == f"job {job.id} (rag-index) failed: no cause reported"


def test_a_long_cause_is_clipped():
    job = gj.Job(id="x", kind="k", argv=[])
    job.push({"type": "line", "outcome": True, "text": "y" * 1000})
    cause = gj._job_failure_cause(job, None)
    assert len(cause) == gj._END_CAUSE_MAX
    assert cause.endswith("...")


def test_a_done_job_logs_debug_and_no_warning(caplog):
    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("model-scan", lambda job: True)
        rec = _end_record(caplog, job)
    assert rec.levelno == logging.DEBUG
    assert rec.getMessage() == f"job {job.id} (model-scan) done"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_job_marked_done_before_its_cleanup_raised_is_not_logged_as_failed(caplog):
    def fn(job):
        job.mark_outcome("done")
        raise RuntimeError("cleanup blew up")

    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn("video", fn)
        rec = _end_record(caplog, job)
    assert job.status == "done"
    assert rec.levelno == logging.DEBUG
    assert rec.getMessage() == f"job {job.id} (video) done"
