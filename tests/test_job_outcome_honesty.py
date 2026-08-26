# SPDX-License-Identifier: AGPL-3.0-or-later
"""jobs.py's start_cli must not decide a job's status purely from the subprocess
exit code: an exception raised by a CLI command AFTER its real work succeeded
would report a completed operation as failed.

A terminal {"type": "outcome", "status": ...} sentinel frame
(_shared._emit_outcome) is sent by a CLI command BEFORE any risky display step,
once real work is verifiably done. start_cli's stdout reader treats it as an
override for the exit-code guess, in EITHER direction, and never as licence to
invent success from silence: absent the frame the exit-code rule is untouched,
so a job that dies mid-work, or a job kind that does not emit the frame, still
reports failed/done.
"""

from __future__ import annotations

import json
import time

from localm.plugins.gui import jobs as gj


def _wait_for_terminal(job, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status != "running":
            return
        time.sleep(0.01)
    raise AssertionError(f"job never left 'running': {job.status}")


class _FakeProc:
    """Mirrors test_host_announce.py's _FakeProc: a stdout line iterator plus
    a fixed returncode, standing in for subprocess.Popen."""

    def __init__(self, lines, returncode):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def _outcome_line(status: str) -> str:
    return gj.PROGRESS_SENTINEL + json.dumps({"type": "outcome", "status": status}) + "\n"


class TestExplicitOutcomeOverridesAMisleadingExitCode:
    def test_outcome_done_survives_a_crash_after_it(self, monkeypatch):
        """The exact bug: real work finished (the CLI already sent its
        outcome frame), then something else raised and the process exited
        non-zero. The job must still read done."""
        proc = _FakeProc([
            _outcome_line("done"),
            "Traceback (most recent call last):\n",
            "ModuleNotFoundError: No module named 'rich._unicode_data...'\n",
        ], returncode=1)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("comfy-setup", ["comfy", "setup"])
        _wait_for_terminal(job)
        assert job.status == "done", (
            "an explicit done-outcome frame must survive a later crash and "
            f"non-zero exit - got {job.status!r}")
        assert job.returncode == 1, "the real exit code is still recorded"

    def test_outcome_failed_overrides_a_zero_exit_too(self, monkeypatch):
        """Symmetric case: a command that explicitly reports failure must not
        be rescued by an accidentally-zero exit code."""
        proc = _FakeProc([_outcome_line("failed")], returncode=0)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("comfy-setup", ["comfy", "setup"])
        _wait_for_terminal(job)
        assert job.status == "failed"


class TestAbsenceOfTheFrameNeverInventsSuccess:
    """The trap the dispatch calls out explicitly: the fallback path must be
    byte-identical to today's exit-code behavior in BOTH directions - never a
    new way to claim done, and (just as important) not a regression into a
    new way to claim failed for jobs that never asked for this."""

    def test_no_frame_and_nonzero_exit_stays_failed(self, monkeypatch):
        """A job that genuinely dies mid-work - no outcome frame was ever
        sent, because nothing finished - must still report failed."""
        proc = _FakeProc(["still working...\n"], returncode=1)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("comfy-setup", ["comfy", "setup"])
        _wait_for_terminal(job)
        assert job.status == "failed"

    def test_no_frame_and_zero_exit_stays_done(self, monkeypatch):
        """A job kind that has not adopted the new frame at all (e.g.
        "remove") - or an older CLI build mid-rollout - must be completely
        unaffected: exit 0 still means done."""
        proc = _FakeProc(["ok\n"], returncode=0)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("remove", ["rm", "foo", "--yes"])
        _wait_for_terminal(job)
        assert job.status == "done"


class TestTheFrameIsInternalOnly:
    def test_outcome_event_never_reaches_history_or_subscribers(self, monkeypatch):
        proc = _FakeProc([
            gj.PROGRESS_SENTINEL + json.dumps(
                {"downloaded": 5, "total": 100, "pct": 5.0}) + "\n",
            _outcome_line("done"),
        ], returncode=0)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("pull", ["pull", "foo"])
        _wait_for_terminal(job)
        assert job.status == "done"
        types = [e.get("type") for e in job._history]
        assert "outcome" not in types, (
            f"the internal outcome frame leaked into the SSE history: {types}")
        # The ordinary progress event on the SAME channel is unaffected.
        assert "progress" in types


class TestExistingProgressParsingIsUnchanged:
    def test_a_progress_payload_with_no_type_key_still_becomes_a_progress_event(
            self, monkeypatch):
        """Regression guard for the data.pop("type", "progress") change:
        every existing _emit_progress payload carries no "type" key at all,
        and must keep resolving to a "progress" event exactly as before."""
        proc = _FakeProc([
            gj.PROGRESS_SENTINEL + json.dumps(
                {"downloaded": 60, "total": 100, "pct": 60.0}) + "\n",
        ], returncode=0)
        monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: proc)
        job = gj.JobManager().start_cli("pull", ["pull", "foo"])
        _wait_for_terminal(job)
        progress = [e for e in job._history if e.get("type") == "progress"]
        assert len(progress) == 1
        assert progress[0]["pct"] == 60.0
        assert progress[0]["total"] == 100
