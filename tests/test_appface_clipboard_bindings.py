# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app window must let you select and copy text.

WebView2 gates Ctrl+C, Ctrl+A and the right-click menu behind two settings,
and pywebview ties both to its own debug flag - which also switches devtools
on. localm sets the two directly so the shortcuts work in a normal, non-debug
window and devtools stays off.

CoreWebView2 may only be touched from the UI thread, so the write is marshalled
through the form's Invoke. These tests stand in for the .NET objects; the live
behaviour was measured against a real WebView2 window, where the two settings
went False -> True with AreDevToolsEnabled left False.
"""

from __future__ import annotations

import sys
import types

import pytest

from localm import appface


class _Settings:
    def __init__(self):
        self.AreBrowserAcceleratorKeysEnabled = False
        self.AreDefaultContextMenusEnabled = False
        self.AreDevToolsEnabled = False


class _Core:
    def __init__(self, settings):
        self.Settings = settings


class _Widget:
    def __init__(self, settings, raises=None):
        self._settings = settings
        self._raises = raises

    @property
    def CoreWebView2(self):
        if self._raises is not None:
            raise self._raises
        return _Core(self._settings)


class _Form:
    """A WinForms form. Invoke runs the callable, standing in for the marshal
    onto the UI thread."""

    def __init__(self, widget, invoke_raises=None):
        self.browser = types.SimpleNamespace(webview=widget)
        self.invoked = 0
        self._invoke_raises = invoke_raises

    def Invoke(self, action):
        self.invoked += 1
        if self._invoke_raises is not None:
            raise self._invoke_raises
        action()


class _Window:
    uid = "master"


@pytest.fixture()
def wired(monkeypatch):
    """A fake winforms backend and a fake System.Action, wired into imports."""
    settings = _Settings()

    def _install(form):
        wf = types.ModuleType("webview.platforms.winforms")
        wf.BrowserView = types.SimpleNamespace(instances={"master": form})
        webview_mod = types.ModuleType("webview")
        platforms = types.ModuleType("webview.platforms")
        monkeypatch.setitem(sys.modules, "webview", webview_mod)
        monkeypatch.setitem(sys.modules, "webview.platforms", platforms)
        monkeypatch.setitem(sys.modules, "webview.platforms.winforms", wf)
        system = types.ModuleType("System")
        system.Action = lambda fn: fn
        monkeypatch.setitem(sys.modules, "System", system)
        monkeypatch.setattr(appface.sys, "platform", "win32")

    return settings, _install


def test_the_copy_shortcuts_and_menu_are_turned_on(wired):
    settings, install = wired
    form = _Form(_Widget(settings))
    install(form)
    assert appface._enable_clipboard_bindings(_Window()) == ""
    assert settings.AreBrowserAcceleratorKeysEnabled is True
    assert settings.AreDefaultContextMenusEnabled is True


def test_devtools_stays_off(wired):
    """Asking pywebview for debug would turn these on too, and devtools with
    them. That is the trade this exists to avoid."""
    settings, install = wired
    install(_Form(_Widget(settings)))
    appface._enable_clipboard_bindings(_Window())
    assert settings.AreDevToolsEnabled is False


def test_the_write_is_marshalled_onto_the_ui_thread(wired):
    """CoreWebView2 raises if touched from anywhere else."""
    settings, install = wired
    form = _Form(_Widget(settings))
    install(form)
    appface._enable_clipboard_bindings(_Window())
    assert form.invoked == 1


def test_a_failure_inside_the_ui_thread_is_reported(wired):
    settings, install = wired
    install(_Form(_Widget(settings, raises=RuntimeError("no interface"))))
    problem = appface._enable_clipboard_bindings(_Window())
    assert problem
    assert "no interface" in problem
    assert settings.AreBrowserAcceleratorKeysEnabled is False


def test_an_unreachable_ui_thread_is_reported(wired):
    settings, install = wired
    install(_Form(_Widget(settings), invoke_raises=RuntimeError("handle gone")))
    problem = appface._enable_clipboard_bindings(_Window())
    assert "handle gone" in problem


def test_a_missing_widget_is_reported(wired, monkeypatch):
    settings, install = wired
    form = _Form(None)
    form.browser = types.SimpleNamespace()
    install(form)
    assert appface._enable_clipboard_bindings(_Window())


def test_nothing_is_attempted_off_windows(monkeypatch):
    """The macOS backend leaves these alone and the Linux one only drops the
    context menu, so there is nothing here to set."""
    monkeypatch.setattr(appface.sys, "platform", "linux")
    assert appface._enable_clipboard_bindings(_Window()) == "not windows"
