# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side visibility for client-initiated GUI jobs (G2).

A model pull started from a phone or PWA sends its output to the per-job event
queue the browser reads, which the person running ``localm gui`` does not see.
_HostAnnouncer mirrors the job's start, throttled progress and end to the host
stdout and debug log. line() is pure (the throttle), so it is unit-testable; one
integration test drives start_cli with a mocked subprocess for the full path.

record_line/announce_failure_detail put a job's actual failure reason (the real
git/pip/native error text) somewhere a bug report can read, rather than only on
the ephemeral per-job SSE stream."""

import json
import logging
import time

from localm.plugins.gui import jobs as gj
from localm.plugins.gui.jobs import _HostAnnouncer


def test_progress_throttled_to_ten_percent_steps():
    a = _HostAnnouncer("Model pull foo")
    assert a.line({"type": "progress", "pct": 3}) == "Model pull foo: 0%"
    assert a.line({"type": "progress", "pct": 7}) is None       # same 0% bucket
    assert a.line({"type": "progress", "pct": 11}) == "Model pull foo: 10%"
    assert a.line({"type": "progress", "pct": 19}) is None      # same 10% bucket
    assert a.line({"type": "progress", "pct": 55}) == "Model pull foo: 50%"


def test_non_progress_lines_are_silent():
    a = _HostAnnouncer("Model pull foo")
    assert a.line({"type": "line", "text": "blah"}) is None
    assert a.line({"type": "progress"}) is None                 # no pct
    assert a.line({"type": "progress", "pct": None}) is None


def test_end_reports_status():
    assert _HostAnnouncer("p").line({"type": "end", "status": "done"}) == "p done"
    assert _HostAnnouncer("p").line({"type": "end", "status": "failed"}) == "p failed"


def test_announce_start_prints_to_host_stdout(capsys):
    # The host channel is stdout (the GUI server's terminal) - the user's whole ask.
    _HostAnnouncer("Model pull foo").announce_start()
    assert "Model pull foo started" in capsys.readouterr().out


def test_start_cli_mirrors_a_pull_to_the_host(capsys, monkeypatch):
    """Full path: start_cli with a host_label surfaces start/progress/end on the
    host stdout, driven from a mocked subprocess's progress lines."""
    s = gj.PROGRESS_SENTINEL

    class _FakeProc:
        returncode = 0

        def __init__(self):
            self.stdout = iter([
                s + json.dumps({"downloaded": 5, "total": 100, "pct": 5.0}) + "\n",
                s + json.dumps({"downloaded": 60, "total": 100, "pct": 60.0}) + "\n",
                "an ordinary log line\n",
            ])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(gj.subprocess, "Popen", lambda *a, **k: _FakeProc())
    mgr = gj.JobManager()
    job = mgr.start_cli("pull", ["pull", "foo"], host_label="Model pull foo")
    # Wait for the ANNOUNCEMENT, not for job.status, and accumulate rather than
    # draining once. The worker sets job.status = "done" BEFORE it emits the
    # closing announce (localm/plugins/gui/jobs.py: the status assignment, then
    # the announcer's end emit several lines later), so a loop that breaks on the
    # status flip can read capsys inside that window and miss the final line.
    # readouterr() drains the buffer, so each poll appends to what was already
    # seen instead of discarding the earlier lines.
    out = ""
    for _ in range(300):                       # await the daemon thread (instant proc)
        out += capsys.readouterr().out
        if "Model pull foo done" in out:
            break
        time.sleep(0.01)
    assert job.status == "done"
    assert "Model pull foo started" in out
    assert "Model pull foo: 0%" in out         # 5% -> 0% bucket
    assert "Model pull foo: 60%" in out
    assert "Model pull foo done" in out


def test_announce_failure_detail_is_noop_when_nothing_recorded(caplog):
    with caplog.at_level(logging.ERROR, logger="localm"):
        _HostAnnouncer("p").announce_failure_detail()
    assert not [r for r in caplog.records if r.name == "localm"]


def test_announce_failure_detail_logs_the_recorded_tail(caplog):
    a = _HostAnnouncer("ComfyUI setup")
    a.record_line("$ git clone --quiet https://example/repo dest")
    a.record_line("fatal: could not resolve host: example")
    with caplog.at_level(logging.ERROR, logger="localm"):
        a.announce_failure_detail()
    records = [r for r in caplog.records if r.name == "localm"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    msg = records[0].getMessage()
    assert "ComfyUI setup failed" in msg
    assert "could not resolve host" in msg


def test_record_line_tail_is_bounded():
    a = _HostAnnouncer("p")
    for i in range(a._TAIL_LINES + 10):
        a.record_line(f"line {i}")
    assert len(a._recent_lines) == a._TAIL_LINES
    # Oldest lines are evicted first - only the most recent tail survives.
    assert a._recent_lines[0] == f"line {10}"
    assert a._recent_lines[-1] == f"line {a._TAIL_LINES + 9}"


def test_start_cli_logs_failure_detail_from_real_output(caplog):
    """A job that FAILS must leave its actual output, not just "<label>
    failed", in the debug log via the localm logger, not only on the browser's
    SSE stream."""
    class _FakeFailingProc:
        returncode = 1

        def __init__(self):
            self.stdout = iter([
                "$ git clone --quiet https://example/repo dest\n",
                "fatal: could not resolve host: example\n",
            ])

        def wait(self):
            return 1

    import localm.plugins.gui.jobs as gj_mod
    monkeypatch_target = gj_mod.subprocess
    original_popen = monkeypatch_target.Popen
    monkeypatch_target.Popen = lambda *a, **k: _FakeFailingProc()
    try:
        mgr = gj_mod.JobManager()
        with caplog.at_level(logging.ERROR, logger="localm"):
            job = mgr.start_cli("comfy-setup", ["comfy", "setup"],
                                host_label="ComfyUI setup")
            for _ in range(300):
                if job.status in ("done", "failed", "cancelled"):
                    break
                time.sleep(0.01)
        assert job.status == "failed"
        records = [r for r in caplog.records
                  if r.name == "localm" and "ComfyUI setup failed" in r.getMessage()]
        assert len(records) == 1
        assert "could not resolve host" in records[0].getMessage()
    finally:
        monkeypatch_target.Popen = original_popen
