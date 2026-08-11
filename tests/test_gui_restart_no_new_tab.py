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

from localm.plugins.gui.cli import _should_auto_open_browser


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
