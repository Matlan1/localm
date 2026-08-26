# SPDX-License-Identifier: AGPL-3.0-or-later
"""jobs.py's ``start_fn`` must not report a completed operation as failed.

Deciding a job's status purely from ``fn(job)``'s return value and whether it
raised means any exception raised by an in-process callback AFTER its real work
already succeeded (image/music/video generation's VRAM handover, RAG indexing's
summary report, ...) reports success as failure. ``start_fn`` has no
subprocess/stdout boundary to carry a sentinel frame across, so the mechanism is
an explicit ``Job.mark_outcome(status)`` call a callback makes once its own real
work is verifiably done, before any risky tail cleanup.

``start_fn``'s worker treats a "done" mark as an override for what would
otherwise be a blind ``except -> failed``, in ONE direction only, and NEVER as
license to invent success from silence: absent a mark (the default, ``None``),
or with a "failed" mark, the exception path is untouched - a callback that never
adopts this keeps the existing rule exactly.
"""

from __future__ import annotations

import time

import pytest

from localm.plugins.gui import jobs as gj


def _wait_for_terminal(job, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status != "running":
            return
        time.sleep(0.01)
    raise AssertionError(f"job never left 'running': {job.status}")


class TestMarkOutcomeOverridesAnExceptionAfterRealWorkSucceeded:
    def test_marked_done_survives_an_exception_after_it(self):
        """Real work finished (fn already called mark_outcome), then tail cleanup
        raised. The job must still read done, and the tail failure must still be
        visible on the stream - only the terminal STATUS is corrected."""
        def fn(job):
            job.result = "the-real-deliverable"
            job.mark_outcome("done")
            raise RuntimeError("cleanup blew up")

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "done", (
            f"a done mark must survive a later exception in tail cleanup - "
            f"got {job.status!r}")
        assert job.result == "the-real-deliverable"
        lines = [e["text"] for e in job._history if e.get("type") == "line"]
        assert any("cleanup after success failed" in t for t in lines), lines
        assert any("cleanup blew up" in t for t in lines), lines

    def test_marked_failed_is_not_rescued_by_mark_outcome(self):
        """Symmetric case: a callback that explicitly records its own failure,
        then also raises in cleanup, must not be rescued into done."""
        def fn(job):
            job.mark_outcome("failed")
            raise RuntimeError("cleanup blew up too")

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "failed"


class TestAbsenceOfTheMarkerNeverInventsSuccess:
    """The fallback path is unchanged in EVERY direction: never a new way to
    claim done, and never a new way to claim failed for callbacks that never
    adopt mark_outcome."""

    def test_no_marker_and_exception_stays_failed(self):
        """A callback that genuinely dies mid-work - mark_outcome was never
        called, because nothing finished - must still report failed."""
        def fn(job):
            raise RuntimeError("still working when this blew up")

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "failed"
        lines = [e["text"] for e in job._history if e.get("type") == "line"]
        assert lines == ["job error: still working when this blew up"], lines

    def test_no_marker_and_clean_true_return_stays_done(self):
        """A callback that has not adopted mark_outcome at all - the common
        case - must be completely unaffected: a clean True still means done."""
        job = gj.JobManager().start_fn("test", lambda job: True)
        _wait_for_terminal(job)
        assert job.status == "done"

    def test_no_marker_and_clean_false_return_stays_failed(self):
        job = gj.JobManager().start_fn("test", lambda job: False)
        _wait_for_terminal(job)
        assert job.status == "failed"


class TestMarkOutcomeIsIgnoredOutsideTheExceptPath:
    def test_marked_done_then_a_clean_false_return_still_fails(self):
        """A stale or premature mark_outcome("done") must never override an
        explicit, CLEAN return value - the mark is consulted ONLY when an
        exception actually escapes fn, never on a normal return."""
        def fn(job):
            job.mark_outcome("done")
            return False   # genuinely failed afterward, no exception

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "failed", (
            "a stale done-mark must not survive a clean False return")

    def test_marked_failed_then_a_clean_true_return_still_succeeds(self):
        """A stale failed-mark must not survive an explicit, clean True return
        either: the marker is inert on the try branch in BOTH directions."""
        def fn(job):
            job.mark_outcome("failed")
            return True    # genuinely recovered/succeeded afterward, no exception

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "done", (
            "a stale failed-mark must not survive a clean True return")


class TestMarkOutcomeValidation:
    def test_invalid_status_raises_immediately(self):
        with pytest.raises(ValueError):
            gj.Job(id="x", kind="test", argv=[]).mark_outcome("bogus")

    def test_invalid_status_inside_fn_still_reports_failed(self):
        """A typo'd mark_outcome call raises before self._outcome is ever set,
        so the job correctly falls through to the unchanged failed path -
        never a silent no-op that leaves the mistake invisible."""
        def fn(job):
            job.mark_outcome("Done")   # wrong case - not "done"
            return True                # never reached

        job = gj.JobManager().start_fn("test", fn)
        _wait_for_terminal(job)
        assert job.status == "failed"
        lines = [e["text"] for e in job._history if e.get("type") == "line"]
        assert any("must be 'done' or 'failed'" in t for t in lines), lines
