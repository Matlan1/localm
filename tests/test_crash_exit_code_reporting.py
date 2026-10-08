# SPDX-License-Identifier: AGPL-3.0-or-later
"""A recovered-crash report states how the dead process ended.

The crash-recovery watchdog reads the dead server's exit code and records it
beside the crash marker; the next start's report names it, keeps the end of the
fault trace, and classifies from it. Without the exit code a process that died
inside the fault handler's own dump was reported as "most likely an OS kill".
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from localm import bugreport, instances

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCHDOG_SCRIPT = REPO_ROOT / "scripts" / "crash_recovery_watchdog.py"

_SURVIVED_COM_TRACE = (
    "Windows fatal exception: code 0x8001010d\n"
    "\n"
    "Thread 0x00005c08 (most recent call first):\n"
    '  File "threading.py", line 355 in wait\n'
)
_ACCESS_VIOLATION = 0xC0000005


def _classify(*, trace="", exit_code=None):
    return bugreport._classify_prior_death(
        native_trace=trace, hang_trace="", raw_tail_truncated=False,
        raw_tail_last_line="", exit_code=exit_code)


class TestClassifyWithExitCode:
    def test_a_fatal_exit_code_names_the_fault_when_the_trace_holds_none(self):
        summary, reason = _classify(trace=_SURVIVED_COM_TRACE,
                                    exit_code=_ACCESS_VIOLATION)
        assert "access violation" in summary
        assert "0xC0000005" in summary
        assert "OS kill" not in reason
        assert "0x8001010d" in reason
        assert "no fatal fault line" in reason

    def test_a_fatal_exit_code_with_no_trace_at_all_still_names_the_fault(self):
        summary, reason = _classify(exit_code=_ACCESS_VIOLATION)
        assert "access violation" in summary
        assert "recorded no fatal fault line" in reason

    def test_the_exit_code_is_added_to_a_captured_fatal_fault(self):
        summary, reason = _classify(
            trace="Windows fatal exception: access violation\n",
            exit_code=_ACCESS_VIOLATION)
        assert "native fault captured" in summary
        assert "0xC0000005" in reason

    def test_an_ordinary_exit_code_is_not_called_a_fault(self):
        """Fires-control: only a crash status is named as a fault."""
        summary, reason = _classify(trace=_SURVIVED_COM_TRACE, exit_code=1)
        assert "exited with" not in summary
        assert "OS kill" in reason

    def test_no_exit_code_keeps_the_honest_unknown(self):
        summary, reason = _classify(trace=_SURVIVED_COM_TRACE)
        assert "exited with" not in summary
        assert "OS kill" in reason


class TestTraceTrimming:
    def test_a_short_trace_is_kept_whole(self):
        assert bugreport.trim_trace_for_report("short") == "short"

    def test_a_long_trace_keeps_its_first_header_and_its_end(self, monkeypatch):
        monkeypatch.delenv("LOCALM_DEBUG", raising=False)
        text = "FIRST-HEADER\n" + ("x" * 9000) + "\nLAST-LINE"
        out = bugreport.trim_trace_for_report(text)
        assert out.startswith("FIRST-HEADER")
        assert out.endswith("LAST-LINE")
        assert "characters omitted" in out
        assert len(out) < 4200

    def test_debug_mode_keeps_the_whole_trace(self, monkeypatch):
        monkeypatch.setenv("LOCALM_DEBUG", "1")
        text = "FIRST-HEADER\n" + ("x" * 9000) + "\nLAST-LINE"
        assert bugreport.trim_trace_for_report(text) == text


def _write_marker(run_dir, instance_id, pid):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"server-crash.{instance_id}.marker").write_text(
        json.dumps({"pid": pid, "context": {}}), encoding="utf-8")


class TestExitRecordReachesTheReport:
    def _file_report(self, tmp_path, monkeypatch, exit_record):
        monkeypatch.setenv("LOCALM_MODE", "log")
        monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
        run = tmp_path / "run"
        _write_marker(run, "inst-av", 4242)
        if exit_record is not None:
            (run / "server-crash-exit.inst-av.json").write_text(
                json.dumps(exit_record), encoding="utf-8")
        captured = {}
        monkeypatch.setattr(bugreport, "report_failure",
                            lambda **k: captured.update(k) or str(tmp_path / "r.md"))
        bugreport.check_and_report_prior_crash(home=str(tmp_path))
        return captured, run

    def test_the_recorded_exit_code_classifies_and_is_consumed(self, tmp_path, monkeypatch):
        captured, run = self._file_report(
            tmp_path, monkeypatch,
            {"pid": 4242, "exit_code": _ACCESS_VIOLATION, "watched_for_s": 301.5})
        assert "access violation" in captured["summary"]
        assert captured["context"]["prior_exit"]["exit_code"] == _ACCESS_VIOLATION
        assert not (run / "server-crash-exit.inst-av.json").exists()

    def test_no_exit_record_changes_nothing(self, tmp_path, monkeypatch):
        captured, _run = self._file_report(tmp_path, monkeypatch, None)
        assert "prior_exit" not in captured["context"]
        assert "OS kill" in captured["reason"]

    def test_a_garbled_exit_record_is_ignored_and_removed(self, tmp_path, monkeypatch):
        captured, run = self._file_report(
            tmp_path, monkeypatch, {"exit_code": "not a number"})
        assert "prior_exit" not in captured["context"]
        assert not (run / "server-crash-exit.inst-av.json").exists()

    def test_the_report_has_a_process_exit_section(self):
        text = bugreport.build_report(
            "s", "r", context={"prior_exit": {"exit_code": _ACCESS_VIOLATION,
                                              "watched_for_s": 12.5}})
        assert "## Process exit" in text
        assert "access violation" in text
        assert "Watched for: 12.5s" in text


def _load_wd():
    spec = importlib.util.spec_from_file_location("crash_recovery_watchdog", WATCHDOG_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exit_record_path_matches_bugreport(tmp_path):
    marker = tmp_path / "server-crash.abc123.marker"
    assert (_load_wd().exit_record_path(tmp_path, "abc123")
            == bugreport._exit_path_for_marker(tmp_path, marker))


@pytest.mark.skipif(sys.platform != "win32", reason="reads a Windows exit code")
def test_the_watchdog_records_the_real_exit_code_of_a_process_that_crashed(tmp_path):
    wd = _load_wd()
    wd._MIN_UPTIME_TO_RELAUNCH_S = 0.0
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import ctypes, time; time.sleep(0.3); "
        f"ctypes.windll.kernel32.ExitProcess({_ACCESS_VIOLATION})"])
    _write_marker(tmp_path, "real-av", proc.pid)
    sentinel = tmp_path / "relaunched.txt"
    wd.run(pid=proc.pid, host="127.0.0.1", port=9, scheme="http",
           instance_id="real-av", crash_dir=tmp_path,
           relaunch_argv=[sys.executable, "-c",
                          f"open(r'{sentinel}', 'w').write('x')"],
           restart_history=[], poll_interval=0.02, grace_s=0.1,
           request_timeout=0.2, log_path=None)
    proc.wait(timeout=5)
    record = json.loads((tmp_path / "server-crash-exit.real-av.json")
                        .read_text(encoding="utf-8"))
    assert record["exit_code"] == _ACCESS_VIOLATION
    assert record["pid"] == proc.pid


@pytest.mark.skipif(sys.platform != "win32", reason="reads a Windows exit code")
def test_a_clean_stop_leaves_no_exit_record(tmp_path):
    wd = _load_wd()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    wd.run(pid=proc.pid, host="127.0.0.1", port=9, scheme="http",
           instance_id="clean", crash_dir=tmp_path,
           relaunch_argv=[sys.executable, "-c", "pass"], restart_history=[],
           poll_interval=0.02, grace_s=0.1, request_timeout=0.2, log_path=None)
    proc.wait(timeout=5)
    assert not (tmp_path / "server-crash-exit.clean.json").exists()
