# SPDX-License-Identifier: AGPL-3.0-or-later
"""appface.native_window_available / run_native_window (ADR-0011): `localm gui` prefers a native OS webview window over a browser tab when the optional `localm[desktop]` extra (pywebview) is installed, falling back to webbrowser.open otherwise."""

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm import appface
from localm.plugins.gui import cli as guicli


def test_native_window_available_false_under_pytest_guard():
    fake = MagicMock()
    sys.modules["webview"] = fake
    try:
        assert appface.native_window_available() is False
    finally:
        del sys.modules["webview"]


def test_native_window_available_false_when_not_installed(monkeypatch):
    """Real-path test, no import mocking: pywebview genuinely is not installed in this venv - exercises the actual 'not installed' branch."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    assert "webview" not in sys.modules  # sanity: not accidentally present
    assert appface.native_window_available() is False


def test_native_window_available_true_when_importable(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setitem(sys.modules, "webview", MagicMock())
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_mode": "auto"})
    assert appface.native_window_available() is True


def test_native_window_available_false_when_mode_is_browser(monkeypatch):
    """The explicit opt-out: even with pywebview genuinely importable, the user's desktop_window_mode='browser' preference must win."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setitem(sys.modules, "webview", MagicMock())
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_mode": "browser"})
    assert appface.native_window_available() is False


def test_native_window_allowed_defaults_true_when_config_read_fails(monkeypatch):
    """A config problem must never silently disable a feature the user did not ask to disable - same posture as the quit_on_close read."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setitem(sys.modules, "webview", MagicMock())

    def _boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("localm.config.load_config", _boom)
    assert appface.native_window_available() is True


def test_run_native_window_returns_false_when_mode_is_browser(monkeypatch):
    """Covers the attach-path call site directly: it calls run_native_window without ever going through native_window_available(), so the preference check has to live here too, not only in the other function."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_mode": "browser"})

    assert appface.run_native_window("http://127.0.0.1:8642/") is False
    fake.create_window.assert_not_called()


def test_run_native_window_returns_false_under_pytest_guard_even_if_webview_would_succeed():
    fake = MagicMock()
    sys.modules["webview"] = fake
    try:
        assert appface.run_native_window("http://127.0.0.1:8642/") is False
        fake.create_window.assert_not_called()
    finally:
        del sys.modules["webview"]


def test_run_native_window_returns_false_when_pywebview_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    assert "webview" not in sys.modules
    assert appface.run_native_window("http://127.0.0.1:8642/") is False


class _ClosingEvent:
    """Fake for window.events.closing: captures whatever handler run_native_window registers via `+=` so a test can invoke it directly, the same shape the real pywebview Event class supports (confirmed against the installed source: __iadd__ appends to a callback list)."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


def _fake_webview(*, loaded=True, create_raises=False, start_sleep=0.2):
    """A fake `webview` module. `start`'s side_effect sleeps briefly before returning - the REAL webview.start() blocks for the whole window lifetime, which is what gives run_native_window's watcher thread (reading window.events.loaded from a separate thread, since the calling thread is busy inside the blo..."""
    fake = MagicMock()
    window = SimpleNamespace(
        events=SimpleNamespace(
            loaded=SimpleNamespace(wait=MagicMock(return_value=loaded)),
            closing=_ClosingEvent()),
        hide=MagicMock(), destroy=MagicMock(), show=MagicMock())
    if create_raises:
        fake.create_window.side_effect = RuntimeError("boom")
    else:
        fake.create_window.return_value = window
    fake.start.side_effect = lambda *a, **k: time.sleep(start_sleep)
    return fake, window


def test_run_native_window_returns_true_when_the_window_actually_loads(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, window = _fake_webview(loaded=True)
    monkeypatch.setitem(sys.modules, "webview", fake)

    assert appface.run_native_window("http://127.0.0.1:8642/") is True
    fake.create_window.assert_called_once()
    fake.start.assert_called_once()
    window.events.loaded.wait.assert_called_once()
    # The foreground-focus fix (window.show() calls the real .Activate() -
    # a plain first-creation .Show() does not reliably grab OS focus for a
    # background-launched process, confirmed live): must actually fire once
    # the page loads, not merely be attempted and silently swallowed.
    window.show.assert_called_once()


def test_run_native_window_forces_qt_backend_on_linux(monkeypatch):
    """pywebview's default Linux order tries GTK first (webview/guilib.py), which this project deliberately never installs (see pyproject.toml's desktop extra comment) - forcing gui='qt' skips a guaranteed GTK-import failure and goes straight to the backend the extra actually installs there (verified live:..."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, _ = _fake_webview(loaded=True)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(appface.sys, "platform", "linux")

    assert appface.run_native_window("http://127.0.0.1:8642/") is True
    fake.start.assert_called_once()
    assert fake.start.call_args.kwargs.get("gui") == "qt"


def test_run_native_window_does_not_force_gui_on_windows(monkeypatch):
    """Windows keeps pywebview's own default (WinForms) - forcing anything here would be solving a problem this platform does not have."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, _ = _fake_webview(loaded=True)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(appface.sys, "platform", "win32")

    assert appface.run_native_window("http://127.0.0.1:8642/") is True
    fake.start.assert_called_once()
    assert "gui" not in fake.start.call_args.kwargs


def test_run_native_window_returns_false_when_the_window_never_reports_loaded(monkeypatch):
    """Fires-control for the success test above: a window pywebview happily 'creates' and 'starts' but that never actually finishes loading (events.loaded.wait times out - e.g. WebView2 present but broken) must fall back, not be reported as a success just because start() ran."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, _ = _fake_webview(loaded=False)
    monkeypatch.setitem(sys.modules, "webview", fake)

    assert appface.run_native_window("http://127.0.0.1:8642/") is False


def test_run_native_window_hides_and_vetoes_close_when_quit_setting_is_off(monkeypatch):
    """Default behavior (config key desktop_window_quit_on_close = False): the window's own close button hides it and vetoes the real close, matching how closing a browser tab has always left the server running."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, window = _fake_webview(loaded=True, start_sleep=0.05)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_quit_on_close": False})

    appface.run_native_window("http://127.0.0.1:8642/")

    assert len(window.events.closing.handlers) == 1
    result = window.events.closing.handlers[0]()
    assert result is False, "must veto the close (hide instead)"
    window.hide.assert_called_once()


def test_run_native_window_allows_close_and_calls_on_quit_when_setting_is_on(monkeypatch):
    """The opt-in preference: closing the window quits the app for real and triggers on_quit (the same callable the tray's Stop button uses)."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, window = _fake_webview(loaded=True, start_sleep=0.05)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_quit_on_close": True})
    quit_called = threading.Event()

    appface.run_native_window("http://127.0.0.1:8642/", on_quit=quit_called.set)

    result = window.events.closing.handlers[0]()
    assert result is True, "must allow the real close"
    assert quit_called.wait(timeout=2.0), "on_quit should have been called"
    window.hide.assert_not_called()


def test_run_native_window_hide_on_close_false_skips_the_veto_entirely(monkeypatch):
    """The attach-path shape: this process owns no server to keep alive, so its window must just close normally - no hide, no config lookup, no on_quit registration at all."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, window = _fake_webview(loaded=True, start_sleep=0.05)
    monkeypatch.setitem(sys.modules, "webview", fake)

    appface.run_native_window("http://127.0.0.1:8642/", hide_on_close=False)

    assert window.events.closing.handlers == []


def test_run_native_window_returns_false_and_never_raises_when_create_window_fails(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, _ = _fake_webview(create_raises=True)
    monkeypatch.setitem(sys.modules, "webview", fake)

    assert appface.run_native_window("http://127.0.0.1:8642/") is False


def test_run_native_window_returns_false_and_never_raises_when_start_fails(monkeypatch):
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    fake, _ = _fake_webview(loaded=False)
    fake.start.side_effect = RuntimeError("boom")
    monkeypatch.setitem(sys.modules, "webview", fake)

    assert appface.run_native_window("http://127.0.0.1:8642/") is False


def test_show_native_window_returns_false_when_no_window_active():
    assert appface._native_window is None, "no run_native_window call should be active here"
    assert appface.show_native_window() is False


def test_close_native_window_is_a_noop_when_no_window_active():
    assert appface._native_window is None
    appface.close_native_window()   # must not raise


def test_show_and_close_native_window_reach_the_active_window(monkeypatch):
    """show_native_window/close_native_window are called from a DIFFERENT thread than run_native_window's own caller (the tray/status thread vs. whichever thread is blocked inside webview.start()) - drive that for real with a controllable fake start() instead of a fixed sleep, so the test is deterministic..."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    # loaded=False: keeps window.show() call-count unambiguous - the
    # automatic foreground-focus call (tested separately above) never fires,
    # so the only show() call left is this test's own explicit one.
    fake, window = _fake_webview(loaded=False)
    release_start = threading.Event()
    fake.start.side_effect = lambda *a, **k: release_start.wait(timeout=5.0)
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"desktop_window_quit_on_close": False})

    runner = threading.Thread(
        target=appface.run_native_window, args=("http://127.0.0.1:8642/",))
    runner.start()
    try:
        deadline = time.monotonic() + 2.0
        while appface._native_window is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert appface._native_window is window, "window should be the active one"

        assert appface.show_native_window() is True
        window.show.assert_called_once()

        appface.close_native_window()
        window.destroy.assert_called_once()
    finally:
        release_start.set()
        runner.join(timeout=5.0)


# --------------------------------------------------------------------------- #
#  gui/cli.py wiring: the attach-to-existing-instance path                    #
# --------------------------------------------------------------------------- #
# The simplest of the two call sites - main() is already the process's real
# main thread here (nothing else competes for it, no server gets started on
# an attach), so run_native_window can be called inline with no threading of
# its own, unlike the fresh-launch path (see the module docstring above).

@pytest.fixture
def running(monkeypatch):
    """Pretend a full-mode localm is already serving this directory - the same shape test_gui_attach_flag_conflict.py's own `running` fixture uses, kept local here so this file stays self-contained."""
    entry = {
        "pid": 26164, "port": 8793, "host": "127.0.0.1", "scheme": "http",
        "mode": "full", "token": "tok", "root_dir": "/proj",
    }
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
    monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: entry)
    monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                        lambda *a, **k: ("unknown", None))
    return entry


def test_gui_attach_prefers_native_window_and_skips_the_browser(running, monkeypatch):
    calls = {"native": [], "browser": []}
    monkeypatch.setattr(
        "localm.appface.run_native_window",
        lambda url, *a, **k: (calls["native"].append(url), True)[1])
    monkeypatch.setattr(
        "webbrowser.open", lambda url, *a, **k: calls["browser"].append(url))

    result = CliRunner().invoke(guicli.main, [])   # no --no-browser: exercise the open step

    assert result.exit_code == 0, result.output
    assert calls["native"], "the native window should have been tried"
    assert not calls["browser"], "must not ALSO open a browser tab once native succeeded"


def test_gui_attach_falls_back_to_the_browser_when_native_unavailable(running, monkeypatch):
    calls = {"browser": []}
    monkeypatch.setattr("localm.appface.run_native_window", lambda url, *a, **k: False)
    monkeypatch.setattr(
        "webbrowser.open", lambda url, *a, **k: calls["browser"].append(url))

    result = CliRunner().invoke(guicli.main, [])

    assert result.exit_code == 0, result.output
    assert calls["browser"], "must fall back to the browser when no native window opened"


def test_gui_attach_no_browser_flag_skips_both(running, monkeypatch):
    calls = {"native": [], "browser": []}
    monkeypatch.setattr(
        "localm.appface.run_native_window",
        lambda url, *a, **k: (calls["native"].append(url), True)[1])
    monkeypatch.setattr(
        "webbrowser.open", lambda url, *a, **k: calls["browser"].append(url))

    result = CliRunner().invoke(guicli.main, ["--no-browser"])

    assert result.exit_code == 0, result.output
    assert not calls["native"]
    assert not calls["browser"]
