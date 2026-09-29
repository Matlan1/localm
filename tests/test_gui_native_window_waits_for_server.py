# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm gui` with the native app window: the server runs on a background
thread while the window holds the main thread, and the command returns only
after that server thread has finished, even when the window closes first."""

import contextlib
import socket
import threading

from click.testing import CliRunner

from localm.plugins.gui import cli as guicli


def test_native_window_close_waits_for_the_server_to_stop(monkeypatch):
    serving = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    seen = {}

    def run_advertised(*a, **k):
        serving.set()
        release.wait(10.0)
        stopped.set()

    def run_native_window(url, *a, **k):
        seen["window_on_main_thread"] = (
            threading.current_thread() is threading.main_thread())
        seen["serving_when_window_opened"] = serving.wait(10.0)
        seen["stopped_when_window_closed"] = stopped.is_set()
        timer = threading.Timer(0.3, release.set)
        timer.daemon = True
        timer.start()
        return True

    monkeypatch.setattr("localm.inference.http_server.run_advertised", run_advertised)
    monkeypatch.setattr("localm.inference.http_server.set_restart_ui", lambda ui: None)
    monkeypatch.setattr("localm.appface.native_window_available", lambda: True)
    monkeypatch.setattr("localm.appface.run_native_window", run_native_window)
    monkeypatch.setattr("localm.appface.close_native_window", lambda: None)
    monkeypatch.setattr("localm.appface.start_app_face", lambda **k: None)
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.winconsole.register_console_handler", lambda *a, **k: None)
    monkeypatch.setattr("localm.winconsole.set_console_title", lambda *a, **k: None)
    monkeypatch.setattr("localm.applaunch.apply_window_identity", lambda *a, **k: None)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.delenv("LOCALM_RESTART_IN_PROGRESS", raising=False)

    result = CliRunner().invoke(guicli.main, ["--no-model", "--isolated"])
    returned_after_server_stopped = stopped.is_set()
    release.set()

    assert result.exit_code == 0, result.output
    assert seen == {"window_on_main_thread": True,
                    "serving_when_window_opened": True,
                    "stopped_when_window_closed": False}
    assert returned_after_server_stopped, (
        "the command returned while its server thread was still serving")
