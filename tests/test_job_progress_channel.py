# SPDX-License-Identifier: AGPL-3.0-or-later
"""An in-process job must be able to report structured progress.

`Job.push` is public and latches any dict whose type is "progress", so a
hand-rolled `job.push({"type": "progress", "pct": 12.5})` from an in-thread job
already reaches `Job.summary()`. What this unit adds is the shared constructor:
without one, each call site derives `pct` itself, and `done * 100 / total` with
no total either raises or - after the guard everyone reaches for - reports a
fabricated 0%. The honest derivation belongs in one place.

WHAT THE FIXTURES MUST BE ABLE TO EXPRESS: a fixture that always passes a total
can never produce the case worth pinning - an operation with no denominator must
report pct null, NEVER 0. Half the cases below therefore withhold the total, and
one withholds the numerator.
"""

from __future__ import annotations

import threading
import time

from localm.plugins.gui.jobs import Job, JobManager


def _job(**kw) -> Job:
    return Job(id="j1", kind="rag-index", argv=[], **kw)


def _progress_events(job: Job) -> list:
    return [e for e in job._history if e.get("type") == "progress"]


# --------------------------------------------- absence, never a fabricated 0

class TestAnUnknownPercentageIsNullNeverZero:
    def test_no_total_reports_null_not_zero(self):
        """An operation that has not established a denominator is at an UNKNOWN
        percentage. Zero is a claim, and it is the claim that reads as 'nothing
        is happening yet' on a job that is working hard."""
        job = _job()
        job.progress(phase="indexing", done=37, unit="files")
        ev = _progress_events(job)[-1]
        assert ev["pct"] is None, f"fabricated a percentage with no total: {ev}"
        assert ev["done"] == 37, "an honest count was dropped along with it"

    def test_a_zero_total_is_also_null(self):
        """total=0 is 'we could not size this', not 'the total is zero'."""
        job = _job()
        job.progress(phase="download", done=5, total=0)
        assert _progress_events(job)[-1]["pct"] is None

    def test_no_numerator_reports_null_not_zero(self):
        """A total without a measurement is still an unknown percentage."""
        job = _job()
        job.progress(phase="indexing", total=128, unit="files")
        ev = _progress_events(job)[-1]
        assert ev["pct"] is None, f"invented a numerator: {ev}"
        assert "done" not in ev

    def test_a_measured_zero_is_still_reported_as_zero(self):
        """The mirror case, so the suppression above does not overshoot: a caller
        reporting an observed 0 of 128 is making a true statement and must not be
        converted to unknown. Fabricating a zero is the defect; reporting an
        observed one is not."""
        job = _job()
        job.progress(phase="indexing", done=0, total=128, unit="files")
        assert _progress_events(job)[-1]["pct"] == 0.0


# ------------------------------------------------- the arithmetic and the fields

class TestThePayload:
    def test_pct_matches_the_cli_producer(self):
        """Derived here rather than accepted from the caller so the two
        producers cannot drift; same rounding as _emit_progress."""
        job = _job()
        job.progress(done=412, total=1900, unit="bytes")
        assert _progress_events(job)[-1]["pct"] == 21.7

    def test_unknown_fields_are_omitted_not_zeroed(self):
        job = _job()
        job.progress(phase="enumerating")
        ev = _progress_events(job)[-1]
        assert "done" not in ev and "total" not in ev and "unit" not in ev
        assert ev["phase"] == "enumerating"
        assert ev["pct"] is None

    def test_the_unit_travels_with_the_pair(self):
        """A bare percentage cannot be rendered as '37 of 128 files', and a
        unit-bearing pair degrades better: when a total is lost the numerator
        is still true."""
        job = _job()
        job.progress(done=37, total=128, unit="files")
        ev = _progress_events(job)[-1]
        assert (ev["done"], ev["total"], ev["unit"]) == (37, 128, "files")

    def test_a_numerator_past_its_denominator_is_not_clamped(self):
        """Clamping to 100% would hide a caller bug AND make a false completion
        claim at the same time. The over-100 value stays visible."""
        job = _job()
        job.progress(done=150, total=100)
        assert _progress_events(job)[-1]["pct"] == 150.0


# ------------------------------------------------- a LISTING can read it

class TestTheListingCanRenderIt:
    def test_progress_reaches_summary(self):
        job = _job()
        job.progress(phase="re-embedding", done=512, total=4096, unit="chunks")
        s = job.summary()
        assert s["pct"] == 12.5 and s["phase"] == "re-embedding"

    def test_summary_stays_silent_when_the_percentage_is_unknown(self):
        """summary() must not surface a null pct as a number. An operation
        with no denominator shows no percentage at all, rather than 0%."""
        job = _job()
        job.progress(phase="enumerating")
        s = job.summary()
        assert "pct" not in s, f"leaked an unknown percentage into a listing: {s}"
        assert s["phase"] == "enumerating"

    def test_the_latest_progress_wins(self):
        job = _job()
        job.progress(done=1, total=10)
        job.progress(done=9, total=10)
        assert job.summary()["pct"] == 90.0


# ------------------------------- an in-process job can do this

class TestAnInProcessJobCanReportAPercentage:
    def test_a_start_fn_job_puts_a_percentage_where_a_listing_reads_it(self):
        """End to end through the real worker thread, because the value is the
        whole path and not the dict: a job function calls one method and a
        listing row carries the percentage. Pins that the affordance reaches the
        listing through `start_fn`'s thread."""
        mgr = JobManager()
        done = threading.Event()

        def _work(job):
            job.progress(phase="re-embedding", done=512, total=4096,
                         unit="chunks")
            done.set()
            return True

        job = mgr.start_fn("rag-reembed", _work, label="Re-embedding")
        assert done.wait(5), "job function never ran"
        for _ in range(50):                      # let the worker finish
            if job.status != "running":
                break
            time.sleep(0.02)

        row = next(r for r in mgr.snapshot() if r["id"] == job.id)
        assert row["pct"] == 12.5, (
            f"an in-process job's percentage never reached the listing: {row}")
        assert row["phase"] == "re-embedding"

    def test_it_also_streams_to_a_subscriber(self):
        """The same event must reach an attached viewer, not only the listing -
        that is what a reattaching browser tab reads."""
        job = _job()
        job.progress(phase="indexing", done=3, total=9, unit="files")
        evs = [e for e in job._history if e.get("type") == "progress"]
        assert evs and evs[-1]["pct"] == 33.3
