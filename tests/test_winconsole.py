# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-4: a console-close (or logoff/shutdown) must run cleanup (unload the
native model context) so the process does not segfault freeing it during
interpreter teardown. Ctrl+C / Ctrl+Break stay on Python's default path."""

from localm import winconsole


def test_terminating_event_routing():
    assert winconsole._is_terminating_event(winconsole.CTRL_CLOSE_EVENT)
    assert winconsole._is_terminating_event(winconsole.CTRL_LOGOFF_EVENT)
    assert winconsole._is_terminating_event(winconsole.CTRL_SHUTDOWN_EVENT)
    # Ctrl+C / Ctrl+Break must NOT trigger handler cleanup - they raise
    # KeyboardInterrupt so the normal finally-based shutdown runs instead.
    assert not winconsole._is_terminating_event(winconsole.CTRL_C_EVENT)
    assert not winconsole._is_terminating_event(winconsole.CTRL_BREAK_EVENT)


def test_dispatch_runs_cleanup_only_on_terminating_events():
    calls = []

    def cleanup():
        calls.append(1)

    assert winconsole._dispatch(winconsole.CTRL_CLOSE_EVENT, cleanup) is False
    assert len(calls) == 1
    assert winconsole._dispatch(winconsole.CTRL_C_EVENT, cleanup) is False
    assert len(calls) == 1  # unchanged - Ctrl+C did not run cleanup


def test_dispatch_swallows_cleanup_errors():
    def _boom():
        raise RuntimeError("cleanup blew up")
    # Must not propagate - the OS handler cannot raise.
    assert winconsole._dispatch(winconsole.CTRL_CLOSE_EVENT, _boom) is False


def test_register_console_handler_returns_bool():
    # No-op off Windows; on Windows it registers without raising.
    assert isinstance(winconsole.register_console_handler(lambda: None), bool)
