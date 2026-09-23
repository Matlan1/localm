# SPDX-License-Identifier: AGPL-3.0-or-later
"""A console-close (or logoff/shutdown) runs cleanup, unloading the native model
context rather than freeing it during interpreter teardown. Ctrl+C / Ctrl+Break
stay on Python's default path."""

import faulthandler
import logging
import sys

import pytest

from localm import bugreport, winconsole


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


@pytest.fixture
def _crash_home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME as localm.config.HOME_DIR, so a crash-guard call
    made with home=None lands there. Restores faulthandler to stderr afterwards
    when it was enabled before (arming attaches it to a trace file)."""
    import localm.config as cfg
    was_enabled = faulthandler.is_enabled()
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    yield tmp_path
    bugreport.release_crash_trace(home=str(tmp_path),
                                  instance_id=bugreport._crash_trace_instance_id)
    if was_enabled and not faulthandler.is_enabled():
        faulthandler.enable(file=sys.__stderr__, all_threads=True)


@pytest.mark.parametrize("event", [winconsole.CTRL_C_EVENT, winconsole.CTRL_BREAK_EVENT])
def test_console_interrupt_leaves_a_running_servers_crash_guard_armed(
        _crash_home, monkeypatch, event):
    """Ctrl+C / Ctrl+Break do not stop a server whose serving loop runs off the
    main thread (the app-window mode), so the handler must not tear its crash
    guard down: the marker, the trace file and faulthandler all stay armed."""
    monkeypatch.setattr(bugreport, "_diagnostics_allowed", lambda: True)
    run = _crash_home / "run"
    marker = run / "server-crash.live-inst.marker"
    trace = run / "server-crash-trace.live-inst.txt"
    assert bugreport.arm_crash_guard(context={"port": 1}, home=str(_crash_home),
                                     instance_id="live-inst") is True
    assert marker.exists() and trace.exists() and faulthandler.is_enabled()

    calls = []
    assert winconsole._dispatch(event, lambda: calls.append(1)) is False

    assert marker.exists(), (
        "a console interrupt removed the crash marker of a server that is still "
        "running, so its later hard crash would never be recovered or reported")
    assert trace.exists()
    assert faulthandler.is_enabled()
    assert bugreport.armed_instance_id() == "live-inst"
    assert calls == [], "a console interrupt must not run the console-close cleanup"
    bugreport.disarm_crash_guard(home=str(_crash_home), instance_id="live-inst")


@pytest.mark.parametrize("event", [winconsole.CTRL_C_EVENT, winconsole.CTRL_BREAK_EVENT])
def test_console_interrupt_with_no_guard_armed_leaves_legacy_crash_files_alone(
        _crash_home, monkeypatch, event):
    """Before this process arms its own guard, armed_instance_id() is None. The
    legacy unscoped marker and trace in the same LOCALM_HOME belong to another
    instance and must survive the interrupt."""
    monkeypatch.setattr(bugreport, "_armed_instance_id", None)
    run = _crash_home / "run"
    run.mkdir(parents=True)
    legacy_marker = run / "server-crash.marker"
    legacy_trace = run / "server-crash-trace.txt"
    legacy_marker.write_text('{"pid": 1, "context": {}}', encoding="utf-8")
    legacy_trace.write_text("", encoding="utf-8")

    assert winconsole._dispatch(event, lambda: None) is False

    assert legacy_marker.exists(), "another instance's legacy crash marker was deleted"
    assert legacy_trace.exists(), "another instance's legacy crash trace was deleted"


def test_dispatch_swallows_cleanup_errors(caplog):
    def _boom():
        raise RuntimeError("cleanup blew up")
    # Must not propagate - the OS handler cannot raise.
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert winconsole._dispatch(winconsole.CTRL_CLOSE_EVENT, _boom) is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, (
        f"the swallowed cleanup error must be logged, not silent: {caplog.records}")
    assert "console-close cleanup raised" in warnings[0].getMessage()
    assert warnings[0].exc_info is not None, "the exception must be logged, not just noted"


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
