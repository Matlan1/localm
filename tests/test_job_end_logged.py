# SPDX-License-Identifier: AGPL-3.0-or-later
"""The end of an in-thread (``start_fn``) job reaches the ``localm`` log: a
failed job as a WARNING with its id and kind, any other end as a DEBUG line.
The cause of a failure (an exception message or a line the job pushed) can hold
content, so it is written only while ``debuglog.debug_content_enabled()`` is
True; otherwise the WARNING carries the exception type and says the cause was
withheld."""

from __future__ import annotations

import logging
import time

import pytest

from localm import debuglog
from localm import jobmanager as gj

SECRET = "a prompt about my neighbour"


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


def _run(caplog, kind, fn):
    with caplog.at_level(logging.DEBUG, logger="localm"):
        job = gj.JobManager().start_fn(kind, fn)
        return job, _end_record(caplog, job)


def _no_secret(caplog):
    leaked = [r.getMessage() for r in caplog.records if SECRET in r.getMessage()]
    assert leaked == [], leaked


@pytest.fixture
def content_on(monkeypatch):
    monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)


@pytest.fixture
def content_off(monkeypatch):
    monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: False)
    monkeypatch.setattr(debuglog, "debug_enabled", lambda: True)


def _fails_with_outcome_line(job):
    job.push({"type": "line", "text": "Image backend: native."})
    job.push({"type": "line", "outcome": True, "text": f"No image model for: {SECRET}"})
    job.push({"type": "line", "text": "a later line"})
    return False


def _raises(job):
    job.push({"type": "line", "outcome": True, "text": "not the cause"})
    raise RuntimeError(f"boom on {SECRET}")


# --------------------------------------------- content allowed: the cause is logged

def test_a_failed_job_logs_a_warning_with_its_outcome_line(caplog, content_on):
    job, rec = _run(caplog, "imagine", _fails_with_outcome_line)
    assert job.status == "failed"
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == f"job {job.id} (imagine) failed: No image model for: {SECRET}"


def test_a_job_that_raises_logs_the_exception_as_its_cause(caplog, content_on):
    job, rec = _run(caplog, "music", _raises)
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == f"job {job.id} (music) failed: RuntimeError: boom on {SECRET}"


def test_a_failed_job_without_an_outcome_line_logs_its_last_line(caplog, content_on):
    def fn(job):
        job.push({"type": "line", "text": "first"})
        job.push({"type": "line", "text": "Could not warm up the embedder:\n  none"})
        return False

    job, rec = _run(caplog, "embedding-warmup", fn)
    assert rec.getMessage() == (
        f"job {job.id} (embedding-warmup) failed: Could not warm up the embedder: none")


def test_a_failed_job_with_no_lines_says_no_cause_was_reported(caplog, content_on):
    job, rec = _run(caplog, "rag-index", lambda job: False)
    assert rec.getMessage() == f"job {job.id} (rag-index) failed: no cause reported"


def test_a_long_cause_is_clipped():
    job = gj.Job(id="x", kind="k", argv=[])
    job.push({"type": "line", "outcome": True, "text": "y" * 1000})
    cause = gj._job_failure_cause(job, None)
    assert len(cause) == gj._END_CAUSE_MAX
    assert cause.endswith("...")


# ------------------------------------------ content not allowed: the cause is withheld

def test_privacy_withholds_the_outcome_line(caplog, content_off):
    job, rec = _run(caplog, "imagine", _fails_with_outcome_line)
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == f"job {job.id} (imagine) failed (cause withheld in privacy mode)"
    _no_secret(caplog)


def test_privacy_logs_the_exception_type_without_its_message(caplog, content_off):
    job, rec = _run(caplog, "music", _raises)
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == (
        f"job {job.id} (music) failed: RuntimeError (cause withheld in privacy mode)")
    _no_secret(caplog)


def test_without_a_debug_log_the_cause_is_withheld_too(caplog, monkeypatch):
    monkeypatch.setattr(debuglog, "debug_enabled", lambda: False)
    job, rec = _run(caplog, "music", _raises)
    assert rec.getMessage() == (
        f"job {job.id} (music) failed: RuntimeError (cause withheld without a debug log)")
    _no_secret(caplog)


def test_a_privacy_session_mode_withholds_the_cause_through_the_real_gate(caplog, monkeypatch):
    monkeypatch.setattr(debuglog, "debug_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    assert debuglog.debug_content_enabled() is False
    job, rec = _run(caplog, "imagine", _fails_with_outcome_line)
    assert rec.getMessage() == f"job {job.id} (imagine) failed (cause withheld in privacy mode)"
    _no_secret(caplog)


def test_a_job_started_in_a_privacy_request_withholds_the_cause(caplog, monkeypatch):
    from localm.audit import request_privacy
    monkeypatch.setattr(debuglog, "debug_enabled", lambda: True)
    monkeypatch.setenv("LOCALM_MODE", "full")
    with caplog.at_level(logging.DEBUG, logger="localm"):
        with request_privacy(True):
            job = gj.JobManager().start_fn("imagine", _fails_with_outcome_line)
        rec = _end_record(caplog, job)
    assert job.private is True
    assert rec.getMessage() == f"job {job.id} (imagine) failed (cause withheld in privacy mode)"
    _no_secret(caplog)


# --------------------------------------------------------------- other ends

def test_a_done_job_logs_debug_and_no_warning(caplog):
    job, rec = _run(caplog, "model-scan", lambda job: True)
    assert rec.levelno == logging.DEBUG
    assert rec.getMessage() == f"job {job.id} (model-scan) done"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_job_marked_done_before_its_cleanup_raised_is_not_logged_as_failed(caplog):
    def fn(job):
        job.mark_outcome("done")
        raise RuntimeError("cleanup blew up")

    job, rec = _run(caplog, "video", fn)
    assert job.status == "done"
    assert rec.levelno == logging.DEBUG
    assert rec.getMessage() == f"job {job.id} (video) done"
