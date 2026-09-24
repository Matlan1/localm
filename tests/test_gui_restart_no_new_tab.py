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
_should_auto_open_browser is the consumer on the re-exec'd side.

LOCALM_RESTART_UI carries the GUI surface the previous run showed ("window" or
"browser", recorded by `localm gui` through http_server.set_restart_ui). A
restart from the native app window into browser mode opens a tab; a restart
from a browser tab does not."""

import os

import pytest

from localm.plugins.gui.cli import (
    _RESTART_PORT_GRACE_WINDOW_S, _resolve_gui_launch_mode,
    _restart_port_grace_window, _should_auto_open_browser,
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


def test_resolve_gui_launch_mode_restart_with_native_window(monkeypatch):
    """When restarting and pywebview is available, the standalone window must
    reopen on the main thread (want_native=True) and not open a browser tab."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        want_native, should_open_browser = _resolve_gui_launch_mode(no_browser=False)
        assert want_native is True
        assert should_open_browser is False
        assert "LOCALM_RESTART_IN_PROGRESS" not in os.environ
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def test_resolve_gui_launch_mode_restart_browser_mode(monkeypatch):
    """When restarting in browser mode, no duplicate browser tab is opened;
    the existing browser tab reconnects in place."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: False)
    os.environ["LOCALM_RESTART_IN_PROGRESS"] = "1"
    try:
        want_native, should_open_browser = _resolve_gui_launch_mode(no_browser=False)
        assert want_native is False
        assert should_open_browser is False
        assert "LOCALM_RESTART_IN_PROGRESS" not in os.environ
    finally:
        os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)


def _restart_env(monkeypatch, previous_ui):
    """Set LOCALM_RESTART_IN_PROGRESS and LOCALM_RESTART_UI as a restart's
    re-exec'd process sees them; both are restored at teardown."""
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.setenv("LOCALM_RESTART_UI", previous_ui)


def test_resolve_gui_launch_mode_restart_window_to_browser(monkeypatch):
    """A restart from the native app window into browser mode opens a browser
    tab."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: False)
    _restart_env(monkeypatch, "window")
    assert _resolve_gui_launch_mode(False) == (False, True)
    assert "LOCALM_RESTART_IN_PROGRESS" not in os.environ
    assert "LOCALM_RESTART_UI" not in os.environ


def test_resolve_gui_launch_mode_restart_window_to_window(monkeypatch):
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    _restart_env(monkeypatch, "window")
    assert _resolve_gui_launch_mode(False) == (True, False)


@pytest.mark.parametrize("previous_ui", ["browser", "unknown"])
def test_resolve_gui_launch_mode_restart_from_a_browser_tab_opens_no_tab(
        monkeypatch, previous_ui):
    """A restart whose previous run showed a browser tab (or an unrecognised
    surface) opens no new tab."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: False)
    _restart_env(monkeypatch, previous_ui)
    assert _resolve_gui_launch_mode(False) == (False, False)
    assert "LOCALM_RESTART_UI" not in os.environ


def test_resolve_gui_launch_mode_restart_from_the_window_with_no_browser(monkeypatch):
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    _restart_env(monkeypatch, "window")
    assert _resolve_gui_launch_mode(True) == (False, False)


def test_restart_ui_without_the_restart_flag_is_a_fresh_launch(monkeypatch):
    """LOCALM_RESTART_UI without LOCALM_RESTART_IN_PROGRESS (a crash-recovery
    relaunch) is a fresh launch, and the variable is removed."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: False)
    monkeypatch.setenv("LOCALM_RESTART_IN_PROGRESS", "1")
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS")
    monkeypatch.setenv("LOCALM_RESTART_UI", "browser")
    assert _resolve_gui_launch_mode(False) == (False, True)
    assert "LOCALM_RESTART_UI" not in os.environ


def _run_gui_startup(monkeypatch, *, native, window_loads=True, args=()):
    """Run `localm gui --no-model --isolated` through its real startup, with the
    server, the app window and the browser replaced by recorders. Returns the
    surfaces passed to http_server.set_restart_ui and the URLs opened in a
    browser."""
    import contextlib
    import socket
    import threading

    from click.testing import CliRunner

    from localm.plugins.gui import cli as guicli

    recorded, opened = [], []
    monkeypatch.setattr("localm.appface.native_window_available", lambda: native)
    monkeypatch.setattr("localm.appface.run_native_window",
                        lambda url, *a, **k: window_loads)
    monkeypatch.setattr("webbrowser.open", lambda url, *a, **k: opened.append(url))
    monkeypatch.setattr("localm.inference.http_server.run_advertised",
                        lambda *a, **k: None)
    monkeypatch.setattr("localm.inference.http_server.set_restart_ui",
                        recorded.append)
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.winconsole.register_console_handler",
                        lambda *a, **k: None)
    monkeypatch.setattr("localm.winconsole.set_console_title", lambda *a, **k: None)
    monkeypatch.setattr("localm.applaunch.apply_window_identity",
                        lambda *a, **k: None)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: contextlib.nullcontext())
    for name in ("LOCALM_RESTART_IN_PROGRESS", "LOCALM_RESTART_UI"):
        monkeypatch.setenv(name, "1")
        monkeypatch.delenv(name)

    result = CliRunner().invoke(guicli.main, ["--no-model", "--isolated", *args])
    for t in threading.enumerate():
        if t.name == "open-browser":
            t.join(10.0)
            assert not t.is_alive(), "the browser-open thread did not finish"
    assert result.exit_code == 0, result.output
    return recorded, opened


def test_gui_startup_records_the_app_window(monkeypatch):
    recorded, opened = _run_gui_startup(monkeypatch, native=True)
    assert recorded == ["window"]
    assert opened == []


def test_gui_startup_records_a_browser_tab(monkeypatch):
    recorded, opened = _run_gui_startup(monkeypatch, native=False)
    assert recorded == ["browser"]
    assert len(opened) == 1


def test_gui_startup_records_the_fallback_browser_tab(monkeypatch):
    """An app window that fails to load falls back to a browser tab, which is
    then the recorded surface."""
    recorded, opened = _run_gui_startup(monkeypatch, native=True,
                                        window_loads=False)
    assert recorded == ["window", "browser"]
    assert len(opened) == 1


def test_gui_startup_with_no_browser_records_no_surface(monkeypatch):
    recorded, opened = _run_gui_startup(monkeypatch, native=True,
                                        args=("--no-browser",))
    assert recorded == []
    assert opened == []


def test_resolve_gui_launch_mode_cold_start_with_native_window(monkeypatch):
    """On cold start with pywebview available, the standalone window opens
    and browser auto-opening is suppressed."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    want_native, should_open_browser = _resolve_gui_launch_mode(no_browser=False)
    assert want_native is True
    assert should_open_browser is False


def test_resolve_gui_launch_mode_cold_start_browser_mode(monkeypatch):
    """On cold start in browser mode, the browser tab is auto-opened."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: False)
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    want_native, should_open_browser = _resolve_gui_launch_mode(no_browser=False)
    assert want_native is False
    assert should_open_browser is True


def test_resolve_gui_launch_mode_explicit_no_browser(monkeypatch):
    """When --no-browser is passed, neither the native window nor the browser
    is opened, even if pywebview is available."""
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    want_native, should_open_browser = _resolve_gui_launch_mode(no_browser=True)
    assert want_native is False
    assert should_open_browser is False
