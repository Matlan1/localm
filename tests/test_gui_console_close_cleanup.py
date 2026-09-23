# SPDX-License-Identifier: AGPL-3.0-or-later
"""_console_close_cleanup is the callable wired into
winconsole.register_console_handler so a closed console window still kills
any coder background OS subprocess (see localm/plugins/coder/background.py's
JobRegistry - model workers already self-terminate on their own via
localm._mp_spawn.install_parent_death_watchdog and are not this function's
job), AND clears this instance's own crash marker (see
test_console_close_disarms_crash_guard below) so a deliberate close is never
mistaken for a crash by the recovery watchdog, while native-fault capture stays
armed through the kill. It must never block its caller for longer than its
budget, whatever the background kill or the marker removal does.
"""

import faulthandler
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

from localm import bugreport
from localm.plugins.gui import cli as gui_cli

# _console_close_cleanup does its real work on a background thread it never
# joins with a propagating result, so an exception escaping that thread would
# NOT surface as one of these tests raising - only as a
# PytestUnhandledThreadExceptionWarning. Elevate that one warning to an error
# so the exception-swallowing tests below can actually fail when the
# try/except is missing, instead of merely printing a warning nobody reads.
pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnhandledThreadExceptionWarning")


@pytest.fixture(autouse=True)
def _no_armed_instance(monkeypatch):
    """These tests are about the coder-registry kill, not the crash guard -
    force armed_instance_id() to None so _console_close_cleanup's disarm call
    is a guaranteed no-op regardless of what another test in this same
    process last armed (a real disarm otherwise touches the real default
    LOCALM_HOME via bugreport._crash_dir(home=None))."""
    monkeypatch.setattr(bugreport, "armed_instance_id", lambda: None)


def test_kills_the_background_job_registry(monkeypatch):
    registry = MagicMock()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()

    registry.shutdown_all.assert_called_once_with()


def test_swallows_an_exception_from_shutdown_all(monkeypatch):
    registry = MagicMock()
    registry.shutdown_all.side_effect = RuntimeError("kill FAILED")
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()   # must not raise, in this thread or the worker's


def test_swallows_the_coder_plugin_being_unavailable(monkeypatch):
    def _missing():
        raise ImportError("no module named localm.plugins.coder.background")
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", _missing)

    gui_cli._console_close_cleanup()   # must not raise, in this thread or the worker's


def test_returns_within_budget_even_if_shutdown_all_never_returns(monkeypatch):
    monkeypatch.setattr(gui_cli, "_CONSOLE_CLOSE_CLEANUP_BUDGET_S", 0.2)
    never = threading.Event()   # never .set() - shutdown_all blocks forever
    registry = MagicMock()
    registry.shutdown_all.side_effect = lambda: never.wait()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    started = time.monotonic()
    gui_cli._console_close_cleanup()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"blocked {elapsed:.2f}s waiting on a shutdown_all() call that "
        "never returns - the budget was not enforced")


def test_console_close_disarms_crash_guard(tmp_path, monkeypatch):
    """The actual regression this file was missing: closing the console
    window (CTRL_CLOSE_EVENT) must be treated as a clean shutdown, exactly
    like Ctrl+C or the GUI Stop button - it must disarm THIS instance's own
    crash marker so scripts/crash_recovery_watchdog.py sees a clean exit and
    does not relaunch.

    Every other test in this file mocks the coder registry only and never
    touches bugreport at all, so none of them could have caught a regression
    here - the crash-guard side of _console_close_cleanup had NO test.
    """
    registry = MagicMock()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    home = tmp_path / "home"
    home.mkdir()
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)

    assert bugreport.arm_crash_guard(
        context={"port": 1}, home=str(home), instance_id="console-close-it") is True
    marker = home / "run" / "server-crash.console-close-it.marker"
    assert marker.exists(), "arm_crash_guard did not write its own marker"

    # _console_close_cleanup clears the marker with home=None (it has no other
    # way to know the home dir), which resolves via the same
    # localm.config.HOME_DIR patched above - matching how it runs for real.
    monkeypatch.setattr(bugreport, "armed_instance_id",
                        lambda: "console-close-it")

    gui_cli._console_close_cleanup()

    assert not marker.exists(), (
        "closing the console window did not disarm the crash guard - the "
        "marker survived, so the watchdog will relaunch this instance as "
        "though it had crashed")


def test_console_close_disarms_crash_guard_before_coder_shutdown_finishes(tmp_path, monkeypatch):
    """The marker must be cleared BEFORE waiting on coder job shutdown, so Windows
    terminating the process during the shutdown wait does not leave the marker on disk."""
    home = tmp_path / "home"
    home.mkdir()
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)

    assert bugreport.arm_crash_guard(
        context={"port": 1}, home=str(home), instance_id="fast-disarm") is True
    marker = home / "run" / "server-crash.fast-disarm.marker"
    assert marker.exists()

    monkeypatch.setattr(bugreport, "armed_instance_id", lambda: "fast-disarm")

    marker_existed_during_shutdown = []

    def _slow_shutdown():
        marker_existed_during_shutdown.append(marker.exists())

    registry = MagicMock()
    registry.shutdown_all.side_effect = _slow_shutdown
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()

    assert marker_existed_during_shutdown == [False], (
        "crash guard was not disarmed before coder shutdown ran")


@pytest.fixture
def _diag_home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME as localm.config.HOME_DIR with diagnostics
    allowed, so arming writes a trace file and attaches faulthandler. Releases
    whatever trace is still attached afterwards and restores faulthandler to
    stderr when it was enabled before."""
    import localm.config as cfg
    was_enabled = faulthandler.is_enabled()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(bugreport, "_diagnostics_allowed", lambda: True)
    yield home
    bugreport.release_crash_trace(home=str(home),
                                  instance_id=bugreport._crash_trace_instance_id)
    if was_enabled and not faulthandler.is_enabled():
        faulthandler.enable(file=sys.__stderr__, all_threads=True)


def _arm(home, instance_id, monkeypatch):
    assert bugreport.arm_crash_guard(context={"port": 1}, home=str(home),
                                     instance_id=instance_id) is True
    monkeypatch.setattr(bugreport, "armed_instance_id", lambda: instance_id)
    run = home / "run"
    return (run / f"server-crash.{instance_id}.marker",
            run / f"server-crash-trace.{instance_id}.txt")


def test_console_close_keeps_native_fault_capture_through_coder_shutdown(
        _diag_home, monkeypatch):
    """Freeing native contexts during the coder shutdown is where a close can
    fault, so the trace file and faulthandler must still be attached while
    shutdown_all() runs, with only the marker already gone. Once shutdown_all()
    returns inside the budget the trace is released."""
    marker, trace = _arm(_diag_home, "cc-trace", monkeypatch)
    assert faulthandler.is_enabled()

    seen = {}

    def _shutdown():
        seen["trace"] = trace.exists()
        seen["faulthandler"] = faulthandler.is_enabled()
        seen["handle"] = bugreport._crash_trace_fh is not None
        seen["marker"] = marker.exists()

    registry = MagicMock()
    registry.shutdown_all.side_effect = _shutdown
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()

    assert seen["trace"] is True, (
        "the native-fault trace file was already deleted while the coder "
        "shutdown was still running, so a fault there leaves no trace")
    assert seen["faulthandler"] is True, (
        "faulthandler was already disabled while the coder shutdown was running")
    assert seen["handle"] is True
    assert seen["marker"] is False, "the marker must be cleared before the shutdown"
    assert not marker.exists()
    assert not trace.exists(), "a shutdown that finished in budget releases the trace"
    assert not (_diag_home / "run" / "server-crash.cc-trace.stopping").exists()
    assert bugreport._crash_trace_fh is None


def test_console_close_leaves_the_trace_armed_when_the_shutdown_overruns(
        _diag_home, monkeypatch):
    """A shutdown still running at the deadline may be the one that faults while
    the OS ends the process, so the trace stays attached; only the marker is
    gone. The next start deals with the leftover trace file."""
    monkeypatch.setattr(gui_cli, "_CONSOLE_CLOSE_CLEANUP_BUDGET_S", 0.3)
    marker, trace = _arm(_diag_home, "cc-overrun", monkeypatch)
    never = threading.Event()
    registry = MagicMock()
    registry.shutdown_all.side_effect = lambda: never.wait()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()

    assert not marker.exists()
    assert (_diag_home / "run" / "server-crash.cc-overrun.stopping").exists(), (
        "no stopping record was left for the next start to pair with the trace")
    assert trace.exists(), (
        "the trace was released while the coder shutdown was still running")
    assert faulthandler.is_enabled(), (
        "faulthandler was detached while the coder shutdown was still running")
    assert bugreport._crash_trace_instance_id == "cc-overrun"
    never.set()


def test_console_close_budget_includes_the_marker_removal(monkeypatch):
    """The whole handler, marker removal included, must return within
    _CONSOLE_CLOSE_CLEANUP_BUDGET_S: the OS allows only a few seconds after a
    console close before it kills the process."""
    monkeypatch.setattr(gui_cli, "_CONSOLE_CLOSE_CLEANUP_BUDGET_S", 1.0)
    monkeypatch.setattr(bugreport, "armed_instance_id", lambda: "slow-disk")

    def _slow_marker_removal(*a, **k):
        time.sleep(0.6)
    monkeypatch.setattr(bugreport, "clear_crash_marker", _slow_marker_removal)
    monkeypatch.setattr(bugreport, "disarm_crash_guard", _slow_marker_removal)
    never = threading.Event()
    registry = MagicMock()
    registry.shutdown_all.side_effect = lambda: never.wait()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    started = time.monotonic()
    gui_cli._console_close_cleanup()
    elapsed = time.monotonic() - started
    never.set()

    assert elapsed < 1.0 + 0.4, (
        f"_console_close_cleanup took {elapsed:.2f}s against a 1.0s budget - the "
        "marker removal ran outside the budget")


def test_console_close_with_no_guard_armed_leaves_legacy_crash_files_alone(
        tmp_path, monkeypatch):
    """armed_instance_id() is None (this file's autouse fixture) until this
    process arms. The legacy unscoped marker and trace in the same LOCALM_HOME
    belong to another instance and must survive a console close."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    legacy_marker = run / "server-crash.marker"
    legacy_trace = run / "server-crash-trace.txt"
    legacy_marker.write_text('{"pid": 1, "context": {}}', encoding="utf-8")
    legacy_trace.write_text("", encoding="utf-8")
    registry = MagicMock()
    monkeypatch.setattr(
        "localm.plugins.coder.background.get_registry", lambda: registry)

    gui_cli._console_close_cleanup()

    assert legacy_marker.exists(), "another instance's legacy crash marker was deleted"
    assert legacy_trace.exists(), "another instance's legacy crash trace was deleted"
    registry.shutdown_all.assert_called_once_with()
