# SPDX-License-Identifier: AGPL-3.0-or-later
"""_console_close_cleanup is the callable wired into
winconsole.register_console_handler so a closed console window still kills
any coder background OS subprocess (see localm/plugins/coder/background.py's
JobRegistry - model workers already self-terminate on their own via
localm._mp_spawn.install_parent_death_watchdog and are not this function's
job), AND disarms this instance's own crash guard (see
test_console_close_disarms_crash_guard below) so a deliberate close is never
mistaken for a crash by the recovery watchdog. It must never block its caller
for longer than its budget, whatever the background kill itself does.
"""

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

    # _console_close_cleanup calls disarm_crash_guard with home=None (it has
    # no other way to know the home dir), which resolves via the same
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
