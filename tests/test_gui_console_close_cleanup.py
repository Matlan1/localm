# SPDX-License-Identifier: AGPL-3.0-or-later
"""_console_close_cleanup is the callable wired into
winconsole.register_console_handler so a closed console window still kills
any coder background OS subprocess (see localm/plugins/coder/background.py's
JobRegistry - model workers already self-terminate on their own via
localm._mp_spawn.install_parent_death_watchdog and are not this function's
job). It must never block its caller for longer than its budget, whatever
the background kill itself does.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from localm.plugins.gui import cli as gui_cli

# _console_close_cleanup does its real work on a background thread it never
# joins with a propagating result, so an exception escaping that thread would
# NOT surface as one of these tests raising - only as a
# PytestUnhandledThreadExceptionWarning. Elevate that one warning to an error
# so the exception-swallowing tests below can actually fail when the
# try/except is missing, instead of merely printing a warning nobody reads.
pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnhandledThreadExceptionWarning")


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
