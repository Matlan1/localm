# SPDX-License-Identifier: AGPL-3.0-or-later
"""A server restart re-execs `python -m localm gui ...` with the ORIGINAL
argv (_restart_argv), which has no --no-browser unless the user's own launch
command did. Without a guard, the freshly re-exec'd process's normal startup
would auto-open a SECOND browser tab on every restart - even though the tab
the user is already looking at reconnects in place once the server is back up
(tests-js/server-restart.test.mjs; models.js's server-restart handler ->
init.js's onServerUnreachable, which polls and does a plain location.reload()
in the SAME tab).

http_server._do_restart sets LOCALM_RESTART_IN_PROGRESS right before
os.execv (see test_server_restart.py's
test_do_restart_sets_restart_in_progress_flag_before_relaunch);
_should_auto_open_browser is the consumer on the re-exec'd side."""

import os

from localm.plugins.gui.cli import (
    _RESTART_PORT_GRACE_WINDOW_S, _restart_port_grace_window,
    _should_auto_open_browser,
)


def test_fresh_launch_opens_the_browser():
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    assert _should_auto_open_browser(no_browser=False) is True


def test_explicit_no_browser_flag_is_still_honored():
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    assert _should_auto_open_browser(no_browser=True) is False


def test_restart_reexec_suppresses_the_browser_open():
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        assert _should_auto_open_browser(no_browser=False) is False
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def test_flag_is_consumed_not_merely_read():
    """A later, genuinely fresh launch inheriting this process's environment
    (e.g. a second restart down the line) must not see a stale flag from an
    earlier restart - the check must POP the flag, not just read it."""
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        _should_auto_open_browser(no_browser=False)   # consumes it
        assert "LOCALM_RESTART_IN_PROGRESS" not in os.environ
        assert _should_auto_open_browser(no_browser=False) is True
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def test_fresh_launch_gets_no_port_grace_window():
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    assert _restart_port_grace_window() == 0.0


def test_restart_reexec_gets_the_port_grace_window():
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        assert _restart_port_grace_window() == _RESTART_PORT_GRACE_WINDOW_S
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def test_port_grace_window_read_does_not_consume_the_flag():
    """Unlike _should_auto_open_browser, this check must NOT pop the flag:
    it runs before the server has bound a port, and _should_auto_open_browser
    still needs to see LOCALM_RESTART_IN_PROGRESS afterwards."""
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        _restart_port_grace_window()
        assert os.environ.get("LOCALM_RESTART_IN_PROGRESS") == "1"
        assert _should_auto_open_browser(no_browser=False) is False
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
