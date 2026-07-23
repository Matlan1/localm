# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-3: the bug reporter must fire for the crashes the excepthooks miss - an
uncaught asyncio task exception, and a NATIVE/hard process death (caught on the
next start via a crash marker)."""

import asyncio
import json

from localm import bugreport, instances


def test_crash_marker_arm_check_disarm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))
    # arm_crash_guard() always records THIS (very much alive) test process's
    # own real pid, so the pid-liveness check would otherwise correctly treat
    # it as "a sibling still running" and skip reporting it - simulate the
    # marker belonging to a now-dead process instead, the same way
    # test_bugreport_hang_trace.py mocks instances.pid_alive rather than
    # relying on real pid values.
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
#  Per-instance scoping: running more than one localm server against the SAME #
#  LOCALM_HOME is a first-class scenario (`localm ps`, `serve --project`, the #
#  coder plugin self-starting its own backing server). Confirmed live: a      #
#  second instance found the first, still-healthy instance's marker, filed a  #
#  spurious "crashed" bug report, then its own clean-exit disarm deleted      #
#  whatever marker existed - masking a later genuine crash of a third,        #
#  still-live instance.                                                       #
# --------------------------------------------------------------------------- #

def _write_marker(run_dir, instance_id, pid):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"server-crash.{instance_id}.marker").write_text(
        json.dumps({"pid": pid, "context": {}}), encoding="utf-8")


def test_check_skips_a_marker_whose_recorded_pid_is_genuinely_still_alive(
        tmp_path, monkeypatch):
    """Direct reproduction of the reported bug with NO liveness mock at all:
    arm_crash_guard() records THIS test process's own real, still-running
    pid. The OLD code had no liveness check, so ANY marker present on the next
    start was reported as a crash - exactly what let a second instance
    misreport a first, healthy instance. The fix must skip a marker whose
    pid is confirmed alive."""
    calls = []
    monkeypatch.setattr(bugreport, "report_failure",
                        lambda **k: calls.append(k) or str(tmp_path / "r.md"))
    home = str(tmp_path)
    marker = tmp_path / "run" / "server-crash.marker"

    assert bugreport.arm_crash_guard(context={"port": 1}, home=home) is True
    assert marker.exists()

    # This process's own pid is recorded and genuinely still running - not a
    # crash. Must be left completely alone: not reported, not deleted.
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

    # A second instance starting against the same home must NOT report the
    # still-running first instance as crashed (the negative case).
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
    # instance-a's marker (a DIFFERENT, still-running instance) must survive -
    # this is the exact bug: a shared marker meant this disarm would have
    # deleted whatever marker existed, silencing a later genuine crash of A.
    assert (run / "server-crash.instance-a.marker").exists()


def test_genuine_crash_still_detected_while_a_sibling_stays_alive(
        tmp_path, monkeypatch):
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

    # The fires-control: instance-b genuinely crashed (dead pid) and must
    # still be reported and cleared...
    assert result is not None
    assert len(calls) == 1
    assert not (run / "server-crash.instance-b.marker").exists()
    # ...while instance-a (alive) is left completely alone.
    assert (run / "server-crash.instance-a.marker").exists()


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
