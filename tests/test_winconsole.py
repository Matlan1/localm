# SPDX-License-Identifier: AGPL-3.0-or-later
"""A console-close (or logoff/shutdown) runs cleanup, unloading the native model
context rather than freeing it during interpreter teardown. Ctrl+C / Ctrl+Break
stay on Python's default path."""

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


def test_sanitize_title_strips_control_and_caps():
    assert winconsole._sanitize_title("") == "LocaLM"
    assert winconsole._sanitize_title(None) == "LocaLM"
    assert winconsole._sanitize_title("  LocaLM - :8642  ") == "LocaLM - :8642"
    # control / escape chars are dropped, so a model name cannot inject a terminal
    # escape sequence into the title bar.
    assert "\x1b" not in winconsole._sanitize_title("a\x1b]0;evil\x07b")
    assert "\x07" not in winconsole._sanitize_title("x\x07y")
    assert len(winconsole._sanitize_title("z" * 500)) <= 200


def test_set_console_title_returns_bool():
    # Best-effort + fully guarded: must never raise, always returns a bool.
    assert isinstance(winconsole.set_console_title("LocaLM test"), bool)
