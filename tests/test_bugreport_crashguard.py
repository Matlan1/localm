# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bug reporter must fire for the crashes the excepthooks miss - an
uncaught asyncio task exception, and a NATIVE/hard process death (caught on the
next start via a crash marker)."""

import asyncio
import faulthandler
import json
import logging
import os
import sys

import pytest

from localm import bugreport, instances


def test_crash_marker_arm_check_disarm(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))
    # arm_crash_guard() records this (live) test process's own pid, which the
    # pid-liveness check would treat as a sibling still running - so make the
    # marker look like it belongs to a dead process.
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    home = str(tmp_path)
    marker = tmp_path / "run" / "server-crash.marker"

    # No prior run -> nothing to report.
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert calls == []

    # Arming a run writes the marker.
    assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True
    assert marker.exists()

    # A fresh start that still sees the marker and finds its pid dead = the
    # prior run died hard -> report it once and clear the marker.
    assert bugreport.check_and_report_prior_crash(home=home) is not None
    assert len(calls) == 1
    assert "crash" in calls[0]["summary"].lower()
    assert not marker.exists()

    # A clean shutdown (disarm) leaves no marker, so the next start reports nothing.
    bugreport.arm_crash_guard(home=home)
    bugreport.disarm_crash_guard(home=home)
    assert not marker.exists()
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
#  A faulthandler attach failure is surfaced, never silently swallowed.        #
# --------------------------------------------------------------------------- #

def test_faulthandler_enable_exception_is_logged_not_silent(tmp_path, monkeypatch, caplog):
    import faulthandler

    def _boom(*a, **k):
        raise OSError("fd is not a real file on this platform")

    monkeypatch.setattr(faulthandler, "enable", _boom)
    monkeypatch.setenv("LOCALM_MODE", "log")
    home = str(tmp_path)

    with caplog.at_level(logging.WARNING, logger="localm"):
        # Arming still succeeds even though the trace mechanism failed to attach.
        assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True

    assert (tmp_path / "run" / "server-crash.marker").exists()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("faulthandler" in r.getMessage() for r in warnings), (
        "a faulthandler.enable() failure must be logged, not swallowed silently")


def test_faulthandler_silently_not_enabled_is_also_logged(tmp_path, monkeypatch, caplog):
    """enable() can return WITHOUT raising and still not actually be armed on
    some platforms/file shapes - "no exception" is not proof of success.
    is_enabled() is the one call that tells the truth."""
    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)
    monkeypatch.setenv("LOCALM_MODE", "log")
    home = str(tmp_path)

    with caplog.at_level(logging.WARNING, logger="localm"):
        assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("faulthandler" in r.getMessage() for r in warnings)


def test_faulthandler_successful_attach_logs_no_warning(tmp_path, monkeypatch, caplog):
    """The negative case: a genuinely successful attach on this real box (no
    mocking of faulthandler itself) must NOT spam a warning - only a real
    attach failure should."""
    monkeypatch.setenv("LOCALM_MODE", "log")
    home = str(tmp_path)
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True
    bugreport.disarm_crash_guard(home=home)   # tidy up: detach + close the fh

    warnings = [r for r in caplog.records
               if r.levelno >= logging.WARNING and "faulthandler" in r.getMessage()]
    assert warnings == []


# --------------------------------------------------------------------------- #
#  Per-instance scoping: several localm servers can run against the SAME       #
#  LOCALM_HOME, so each keeps its own marker.                                  #
# --------------------------------------------------------------------------- #

def _write_marker(run_dir, instance_id, pid):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"server-crash.{instance_id}.marker").write_text(
        json.dumps({"pid": pid, "context": {}}), encoding="utf-8")


def test_check_skips_a_marker_whose_recorded_pid_is_genuinely_still_alive(
        tmp_path, monkeypatch):
    """NO liveness mock at all: arm_crash_guard() records THIS test process's
    own real, still-running pid. A marker whose pid is confirmed alive must be
    skipped, never reported as a crash."""
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))
    home = str(tmp_path)
    marker = tmp_path / "run" / "server-crash.marker"

    assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True
    assert marker.exists()

    # This process's own pid is recorded and still running: not reported, not
    # deleted.
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert calls == []
    assert marker.exists()

    bugreport.disarm_crash_guard(home=home)
    assert not marker.exists()


def test_second_instance_does_not_report_a_live_first_instance_as_crashed(
        tmp_path, monkeypatch):
    home = str(tmp_path)
    run = tmp_path / "run"
    first_pid = 11111
    _write_marker(run, "instance-a", first_pid)

    # instance-a's recorded pid is still alive.
    monkeypatch.setattr(instances, "pid_alive", lambda pid: pid == first_pid)
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    # A second instance must not report the still-running first instance as
    # crashed.
    assert bugreport.check_and_report_prior_crash(home=home) is None
    assert calls == []
    # ...and must leave its marker completely untouched - still armed.
    assert (run / "server-crash.instance-a.marker").exists()


def test_disarm_only_clears_its_own_marker_never_a_siblings(tmp_path):
    home = str(tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "instance-a", 11111)
    _write_marker(run, "instance-b", 22222)

    # instance-b shuts down cleanly.
    bugreport.disarm_crash_guard(home=home, instance_id="instance-b")

    assert not (run / "server-crash.instance-b.marker").exists()
    # instance-a's marker (a different, still-running instance) survives.
    assert (run / "server-crash.instance-a.marker").exists()


def test_genuine_crash_still_detected_while_a_sibling_stays_alive(
        tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    home = str(tmp_path)
    run = tmp_path / "run"
    alive_pid, dead_pid = 11111, 22222
    _write_marker(run, "instance-a", alive_pid)
    _write_marker(run, "instance-b", dead_pid)

    monkeypatch.setattr(instances, "pid_alive", lambda pid: pid == alive_pid)
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    result = bugreport.check_and_report_prior_crash(home=home)

    # instance-b genuinely crashed (dead pid) and is reported and cleared...
    assert result is not None
    assert len(calls) == 1
    assert not (run / "server-crash.instance-b.marker").exists()
    # ...while instance-a (alive) is left completely alone.
    assert (run / "server-crash.instance-a.marker").exists()


# --------------------------------------------------------------------------- #
#  The companion server-crash-trace file is unlinked with its marker.          #
# --------------------------------------------------------------------------- #

def test_report_one_crash_marker_deletes_the_trace_file_too(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    home = str(tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "inst-x", 4242)
    trace = run / "server-crash-trace.inst-x.txt"
    trace.write_text("Current thread 0x1: SIGSEGV in ggml\n", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: captured.update(k) or str(tmp_path / "r.md"))

    result = bugreport.check_and_report_prior_crash(home=home)

    assert result is not None
    # The trace's content reached the report before the file was deleted.
    assert "SIGSEGV in ggml" in captured["context"].get("native_trace", "")
    assert not (run / "server-crash.inst-x.marker").exists()
    assert not trace.exists(), "the trace file must be deleted with its marker"


def test_report_one_crash_marker_survives_a_missing_trace_file(tmp_path, monkeypatch):
    """No trace at all (window-close/OS-kill leave none) must not be treated
    as a cleanup failure - the report still files normally."""
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    home = str(tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "inst-y", 5353)
    # No trace file written at all.

    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    result = bugreport.check_and_report_prior_crash(home=home)

    assert result is not None
    assert len(calls) == 1
    assert not (run / "server-crash.inst-y.marker").exists()


def test_a_live_siblings_trace_file_is_left_untouched(tmp_path, monkeypatch):
    """A marker whose pid is genuinely still alive is skipped entirely, and its
    trace file - still in active use by that live process's faulthandler - must
    not be touched either."""
    home = str(tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "inst-z", 6464)
    trace = run / "server-crash-trace.inst-z.txt"
    trace.write_text("", encoding="utf-8")

    monkeypatch.setattr(instances, "pid_alive", lambda pid: True)
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    result = bugreport.check_and_report_prior_crash(home=home)

    assert result is None
    assert calls == []
    assert (run / "server-crash.inst-z.marker").exists()
    assert trace.exists(), "a live sibling's trace file must not be deleted"


# --------------------------------------------------------------------------- #
#  Privacy mode: the marker (watchdog liveness) is always written, the trace   #
#  file and the crash report only when diagnostics are allowed.                #
# --------------------------------------------------------------------------- #

def _pin_mode(monkeypatch, mode, keep=False):
    """Resolve the session mode from config (never the LOCALM_MODE env), the
    same shape tests/test_hang_watchdog.py uses for _diagnostics_allowed."""
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"mode": mode, "keep_diagnostics": keep})


def _crash_files(run, instance_id):
    return (run / f"server-crash.{instance_id}.marker",
            run / f"server-crash-trace.{instance_id}.txt")


def test_privacy_arm_writes_the_marker_but_no_trace_file(tmp_path, monkeypatch):
    _pin_mode(monkeypatch, "privacy")
    run = tmp_path / "run"
    marker, trace = _crash_files(run, "inst-p")

    assert bugreport.arm_crash_guard(context={"port": 1}, home=str(tmp_path),
                                     instance_id="inst-p") is True
    try:
        assert marker.exists(), "the marker is the watchdog's liveness record"
        assert not trace.exists(), "privacy mode must not write a native-fault trace"
        assert sorted(p.name for p in run.glob("server-crash-trace.*")) == []
        info = json.loads(marker.read_text(encoding="utf-8"))
        assert info["diagnostics"] is False
        assert info["pid"]
    finally:
        bugreport.disarm_crash_guard(home=str(tmp_path), instance_id="inst-p")
    assert not marker.exists()


def test_privacy_prior_crash_is_cleared_but_not_reported(tmp_path, monkeypatch, caplog):
    _pin_mode(monkeypatch, "privacy")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    run = tmp_path / "run"
    _write_marker(run, "inst-q", 4242)
    marker, trace = _crash_files(run, "inst-q")
    trace.write_text("Windows fatal exception: access violation\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    with caplog.at_level(logging.INFO, logger="localm"):
        result = bugreport.check_and_report_prior_crash(home=str(tmp_path))

    assert calls == [], "privacy mode must not file a crash report"
    assert result is None
    assert not marker.exists()
    assert not trace.exists()
    assert list((tmp_path / "bug-reports").glob("bug-*.md")) == []
    assert any("prior hard crash detected" in r.getMessage()
               and "not reported" in r.getMessage() for r in caplog.records)


def test_privacy_prior_crash_files_no_report_through_the_real_reporter(
        tmp_path, monkeypatch):
    """No recorder on report_failure: the real save path must leave
    <home>/bug-reports empty."""
    _pin_mode(monkeypatch, "privacy")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "inst-r", 4242)

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is None
    assert list((tmp_path / "bug-reports").glob("bug-*.md")) == []
    assert not (run / "server-crash.inst-r.marker").exists()


def test_log_mode_prior_crash_is_reported_once(tmp_path, monkeypatch):
    """The control for the gate: the same marker IS reported outside privacy."""
    _pin_mode(monkeypatch, "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    run = tmp_path / "run"
    _write_marker(run, "inst-q", 4242)
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is not None
    assert len(calls) == 1
    assert not (run / "server-crash.inst-q.marker").exists()


def test_keep_diagnostics_toggle_reports_in_privacy_mode(tmp_path, monkeypatch):
    _pin_mode(monkeypatch, "privacy", keep=True)
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    run = tmp_path / "run"
    _write_marker(run, "inst-k", 4242)
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is not None
    assert len(calls) == 1


def test_a_marker_armed_in_privacy_mode_is_not_reported_later_in_log_mode(
        tmp_path, monkeypatch):
    """The run that died was a privacy run (its marker says diagnostics were
    off), so switching to log mode before the next start files nothing."""
    _pin_mode(monkeypatch, "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "server-crash.inst-v.marker"
    marker.write_text(json.dumps({"pid": 4242, "context": {}, "diagnostics": False}),
                      encoding="utf-8")
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is None
    assert calls == []
    assert not marker.exists()


def test_disarm_removes_this_instances_trace_file(tmp_path, monkeypatch):
    _pin_mode(monkeypatch, "log")
    run = tmp_path / "run"
    marker, trace = _crash_files(run, "inst-d")

    assert bugreport.arm_crash_guard(home=str(tmp_path), instance_id="inst-d") is True
    assert marker.exists() and trace.exists()
    bugreport.disarm_crash_guard(home=str(tmp_path), instance_id="inst-d")

    assert not marker.exists()
    assert not trace.exists(), "a clean exit must not leave the trace file behind"
    assert sorted(p.name for p in run.glob("server-crash*")) == []


def test_disarm_of_a_sibling_does_not_release_this_instances_faulthandler(
        tmp_path, monkeypatch):
    """Stopping a SIBLING instance (disarm_crash_guard called for a different
    instance_id than the one armed in this process) must remove only the
    sibling's own marker - never this process's own faulthandler attachment
    or its still-live trace file."""
    import faulthandler

    _pin_mode(monkeypatch, "log")
    home = str(tmp_path)
    run = tmp_path / "run"
    my_trace = run / "server-crash-trace.me.txt"

    assert bugreport.arm_crash_guard(home=home, instance_id="me") is True
    assert faulthandler.is_enabled()
    assert bugreport._crash_trace_fh is not None
    assert my_trace.exists()

    _write_marker(run, "sibling", 99999)
    sibling_marker = run / "server-crash.sibling.marker"
    assert sibling_marker.exists()

    bugreport.disarm_crash_guard(home=home, instance_id="sibling")

    assert not sibling_marker.exists(), "the sibling's own marker is removed"
    assert faulthandler.is_enabled(), (
        "disarming a sibling must not release this instance's own faulthandler")
    assert bugreport._crash_trace_fh is not None
    assert my_trace.exists(), "this instance's own trace file must survive"

    # This instance's own clean shutdown still releases everything.
    bugreport.disarm_crash_guard(home=home, instance_id="me")
    assert not faulthandler.is_enabled()
    assert bugreport._crash_trace_fh is None
    assert not my_trace.exists()


# --------------------------------------------------------------------------- #
#  The marker and the trace are released separately. Clearing the marker      #
#  leaves a stopping record; a run that ends before releasing its trace is     #
#  handled from that record by the next start.                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def _restore_faulthandler():
    """Arming attaches faulthandler to a trace file; put pytest's stderr
    attachment back afterwards when it was enabled before."""
    was_enabled = faulthandler.is_enabled()
    yield
    if was_enabled and not faulthandler.is_enabled():
        faulthandler.enable(file=sys.__stderr__, all_threads=True)


def test_clear_crash_marker_leaves_the_trace_attached_and_a_stopping_record(
        tmp_path, monkeypatch, _restore_faulthandler):
    _pin_mode(monkeypatch, "log")
    home = str(tmp_path)
    run = tmp_path / "run"
    marker, trace = _crash_files(run, "inst-split")
    record = run / "server-crash.inst-split.stopping"
    assert bugreport.arm_crash_guard(home=home, instance_id="inst-split") is True
    armed = json.loads(marker.read_text(encoding="utf-8"))

    bugreport.clear_crash_marker(home=home, instance_id="inst-split")

    assert not marker.exists()
    assert json.loads(record.read_text(encoding="utf-8")) == armed, (
        "the stopping record must carry the marker's own content (its pid)")
    assert trace.exists(), "clearing the marker must not delete the trace file"
    assert faulthandler.is_enabled(), "clearing the marker must not detach faulthandler"
    assert bugreport._crash_trace_instance_id == "inst-split"
    assert bugreport.armed_instance_id() is None

    bugreport.release_crash_trace(home=home, instance_id="inst-split")

    assert not trace.exists()
    assert not record.exists()
    assert not faulthandler.is_enabled()
    assert bugreport._crash_trace_fh is None


def test_release_crash_trace_of_another_instance_keeps_this_ones_attached(
        tmp_path, monkeypatch, _restore_faulthandler):
    _pin_mode(monkeypatch, "log")
    home = str(tmp_path)
    run = tmp_path / "run"
    assert bugreport.arm_crash_guard(home=home, instance_id="mine") is True
    other_trace = run / "server-crash-trace.other.txt"
    other_trace.write_text("", encoding="utf-8")

    bugreport.release_crash_trace(home=home, instance_id="other")

    assert not other_trace.exists()
    assert faulthandler.is_enabled()
    assert (run / "server-crash-trace.mine.txt").exists()
    bugreport.disarm_crash_guard(home=home, instance_id="mine")


_FATAL_TRACE = ("Windows fatal exception: access violation\n\n"
                "Thread 0x00001234 (most recent call first):\n"
                '  File "engine.py", line 10 in unload\n')
_FIRST_CHANCE_TRACE = "Windows fatal exception: code 0x8001010d\n"


def _write_stopping(run, instance_id, pid, trace_text, diagnostics=True):
    run.mkdir(parents=True, exist_ok=True)
    record = run / f"server-crash.{instance_id}.stopping"
    record.write_text(json.dumps({"pid": pid, "context": {"port": 1},
                                  "diagnostics": diagnostics}), encoding="utf-8")
    trace = run / f"server-crash-trace.{instance_id}.txt"
    trace.write_text(trace_text, encoding="utf-8")
    return record, trace


def _record_reports(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))
    return calls


def test_a_fault_after_a_clean_stop_began_is_reported_once(tmp_path, monkeypatch):
    """The run cleared its marker for a clean stop, then faulted before it
    released its trace: reported once, then both files are gone."""
    _pin_mode(monkeypatch, "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    calls = _record_reports(monkeypatch, tmp_path)
    record, trace = _write_stopping(tmp_path / "run", "stopped-then-faulted",
                                    424242, _FATAL_TRACE)

    result = bugreport.check_and_report_prior_crash(home=str(tmp_path))

    assert not record.exists() and not trace.exists()
    assert len(calls) == 1, "a fault captured after the marker was cleared was not reported"
    assert result is not None
    assert "while stopping" in calls[0]["summary"]
    assert "access violation" in calls[0]["summary"]
    assert "access violation" in calls[0]["context"]["native_trace"]
    assert calls[0]["context"]["prior_run"]["pid"] == 424242
    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is None
    assert len(calls) == 1, "the same stop was reported twice"


@pytest.mark.parametrize("text", ["", _FIRST_CHANCE_TRACE])
def test_a_clean_stop_that_ended_before_releasing_its_trace_is_only_cleaned_up(
        tmp_path, monkeypatch, text):
    _pin_mode(monkeypatch, "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    calls = _record_reports(monkeypatch, tmp_path)
    record, trace = _write_stopping(tmp_path / "run", "clean-stop", 424242, text)

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is None

    assert not record.exists(), "a leftover stopping record was never cleaned up"
    assert not trace.exists(), "a leftover trace file was never cleaned up"
    assert calls == []


@pytest.mark.parametrize("mode, armed_diagnostics", [("privacy", True), ("log", False)])
def test_a_fault_after_a_clean_stop_in_privacy_mode_is_cleaned_up_not_reported(
        tmp_path, monkeypatch, mode, armed_diagnostics):
    _pin_mode(monkeypatch, mode)
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    calls = _record_reports(monkeypatch, tmp_path)
    record, trace = _write_stopping(tmp_path / "run", "private-run", 424242,
                                    _FATAL_TRACE, diagnostics=armed_diagnostics)

    assert bugreport.check_and_report_prior_crash(home=str(tmp_path)) is None

    assert not record.exists() and not trace.exists()
    assert calls == []


def test_a_stopping_record_of_a_live_instance_is_left_alone(tmp_path, monkeypatch):
    """Another instance in the same LOCALM_HOME can be between clearing its
    marker and releasing its trace while this one starts: its recorded pid is
    alive, so neither file is touched even when the trace already names a
    fault."""
    _pin_mode(monkeypatch, "log")
    calls = _record_reports(monkeypatch, tmp_path)
    record, trace = _write_stopping(tmp_path / "run", "still-stopping",
                                    os.getpid(), _FATAL_TRACE)

    result = bugreport.check_and_report_prior_crash(home=str(tmp_path))

    assert record.exists() and trace.exists(), "a live instance's stopping files were deleted"
    assert calls == [], "a live instance's stop was reported as a crash"
    assert result is None


def test_a_crash_whose_trace_could_not_be_deleted_is_not_reported_again(
        tmp_path, monkeypatch):
    """A hard crash (marker still armed) is reported from its marker. When its
    trace file then cannot be deleted, the leftover trace is never reported a
    second time, as a stop or otherwise."""
    _pin_mode(monkeypatch, "log")
    monkeypatch.setattr(instances, "pid_alive", lambda pid: False)
    calls = _record_reports(monkeypatch, tmp_path)
    run = tmp_path / "run"
    _write_marker(run, "hard-crash", 424242)
    trace = run / "server-crash-trace.hard-crash.txt"
    trace.write_text(_FATAL_TRACE, encoding="utf-8")
    real_unlink = type(trace).unlink
    failed = []

    def unlink(self, *a, **k):
        if self.name == trace.name and not failed:
            failed.append(self.name)
            raise PermissionError("briefly held open elsewhere")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(type(trace), "unlink", unlink)

    bugreport.check_and_report_prior_crash(home=str(tmp_path))
    bugreport.check_and_report_prior_crash(home=str(tmp_path))

    assert failed, "fixture bug: the trace deletion never failed"
    assert len(calls) == 1, [c["summary"] for c in calls]
    assert "while stopping" not in calls[0]["summary"]


def test_disarm_leaves_neither_a_stopping_record_nor_a_trace(tmp_path, monkeypatch):
    _pin_mode(monkeypatch, "log")
    run = tmp_path / "run"
    assert bugreport.arm_crash_guard(home=str(tmp_path), instance_id="full") is True
    bugreport.disarm_crash_guard(home=str(tmp_path), instance_id="full")
    assert sorted(p.name for p in run.glob("server-crash*")) == []


def test_asyncio_handler_reports_task_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(bugreport, "report_failure", lambda **k: calls.append(k))
    loop = asyncio.new_event_loop()
    try:
        assert bugreport.install_asyncio_handler(loop) is True
        loop.call_exception_handler(
            {"message": "boom", "exception": RuntimeError("x")})
    finally:
        loop.close()
    assert len(calls) == 1
    assert "async" in calls[0]["summary"].lower()


def test_asyncio_handler_ignores_cancellation(monkeypatch):
    calls = []
    monkeypatch.setattr(bugreport, "report_failure", lambda **k: calls.append(k))
    loop = asyncio.new_event_loop()
    try:
        bugreport.install_asyncio_handler(loop)
        loop.call_exception_handler(
            {"message": "cancelled", "exception": asyncio.CancelledError()})
    finally:
        loop.close()
    assert calls == []
