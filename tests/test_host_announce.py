# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side visibility for client-initiated GUI jobs (G2).

A model pull started from a phone/PWA used to be invisible on the host (the person
running ``localm gui``): its output went only to the per-job event queue the browser
reads. _HostAnnouncer mirrors the job's start / throttled progress / end to the host
stdout + debug log. line() is pure (the throttle), so it is unit-testable; one
integration test drives start_cli with a mocked subprocess for the full path."""

import json
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
    for _ in range(300):                       # await the daemon thread (instant proc)
        if job.status in ("done", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert job.status == "done"
    out = capsys.readouterr().out
    assert "Model pull foo started" in out
    assert "Model pull foo: 0%" in out         # 5% -> 0% bucket
    assert "Model pull foo: 60%" in out
    assert "Model pull foo done" in out
