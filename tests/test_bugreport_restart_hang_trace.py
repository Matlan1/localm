# SPDX-License-Identifier: AGPL-3.0-or-later
"""A freeze captured before a restart re-executed the server (a new process on
Windows) is still attached to a bug report filed from the restarted server."""

import os

from localm.bugreport import logs


def _trace(home, pid, text):
    d = home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"hang_20261113-080000_{pid}.log").write_text(text, encoding="utf-8")


def test_a_restart_records_the_pid_it_replaces(monkeypatch):
    monkeypatch.setenv(logs.RESTARTED_FROM_ENV, "11,12")
    from localm.inference import http_server as hs
    monkeypatch.setattr(hs, "_restart_ui", None, raising=False)
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)
    hs._set_restart_env()
    assert os.environ[logs.RESTARTED_FROM_ENV] == f"11,12,{os.getpid()}"


def test_run_pids_are_this_process_then_the_earlier_ones_newest_first(monkeypatch):
    monkeypatch.setenv(logs.RESTARTED_FROM_ENV, "11,x,12")
    assert logs.run_pids() == [os.getpid(), 12, 11]


def test_note_restart_keeps_the_newest_pids_only(monkeypatch):
    monkeypatch.setenv(logs.RESTARTED_FROM_ENV, ",".join(str(n) for n in range(1, 20)))
    logs.note_restart()
    kept = os.environ[logs.RESTARTED_FROM_ENV].split(",")
    assert len(kept) == logs._MAX_EARLIER_PIDS
    assert kept[-1] == str(os.getpid())


def test_a_freeze_of_an_earlier_process_is_attached(monkeypatch, tmp_path):
    monkeypatch.setenv(logs.RESTARTED_FROM_ENV, "4242")
    _trace(tmp_path, 4242, "Thread 0x1 (most recent call first):\n  File loop.py, line 1")
    _trace(tmp_path, 9999, "another server's freeze")

    found = logs.run_hang_traces(tmp_path)

    assert "loop.py" in found
    assert "another server's freeze" not in found
