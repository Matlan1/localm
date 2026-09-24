# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper subprocess for test_stop_signal_crash_guard: run the real `localm gui`
command with no status window or tray icon (appface.start_app_face returns
None, as it does under the test suite). NOT a test module (underscore prefix ->
pytest does not collect it).

Usage: python _gui_stop_signal.py <port> [--app-window] [--raise <signal name>]
                                   [--last-resort-bind]

Without --app-window it runs `localm gui --no-model --no-browser --isolated`,
serving on the main thread. With --app-window it runs the app-window mode (the
server on a background thread, the main thread held by the window loop) with
the native window replaced by a wait that runs no Python code until the server
closes it, the way a real webview loop can. On POSIX the stop signals are
blocked on the main thread for that wait, so they are handled on another
thread. It prints whether the server_stopped event was handed to the window
and whether it was already set when the window was closed. With --raise, the
process raises that signal on itself once the window is open. With
--last-resort-bind, the server is served through uvicorn's own bind (see
_portmux_stop_signal_server.force_last_resort_bind).
"""
import argparse
import signal
import sys
import threading
from pathlib import Path

# Import localm from this worktree rather than the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _NativeGate:
    """A one-shot gate whose wait() blocks in native code until open()."""

    def __init__(self):
        if sys.platform == "win32":
            import ctypes
            self._kernel32 = ctypes.windll.kernel32
            self._handle = self._kernel32.CreateEventW(None, True, False, None)
        else:
            import os
            self._r, self._w = os.pipe()

    def block_stop_signals(self):
        if sys.platform != "win32":
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP, signal.SIGTERM})

    def wait(self):
        if sys.platform == "win32":
            self._kernel32.WaitForSingleObject(self._handle, 0xFFFFFFFF)
            return
        import select
        try:
            select.select([self._r], [], [])
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGHUP, signal.SIGTERM})

    def open(self):
        if sys.platform == "win32":
            self._kernel32.SetEvent(self._handle)
        else:
            import os
            os.write(self._w, b"x")


def _use_stub_window(raise_name):
    from localm import appface

    gate = _NativeGate()
    handed = {}

    def run_native_window(url, on_quit=None, **kwargs):
        handed["server_stopped"] = kwargs.get("server_stopped")
        print(f"server_stopped handed over: {handed['server_stopped'] is not None}",
              flush=True)
        gate.block_stop_signals()
        print("native window open", flush=True)
        if raise_name:
            threading.Timer(0.5, signal.raise_signal,
                            (getattr(signal, raise_name),)).start()
        gate.wait()
        print("native window closed", flush=True)
        return True

    def close_native_window():
        stopped = handed.get("server_stopped")
        print(f"server_stopped set before close: {bool(stopped and stopped.is_set())}",
              flush=True)
        gate.open()

    appface.native_window_available = lambda: True
    appface.run_native_window = run_native_window
    appface.close_native_window = close_native_window


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--app-window", action="store_true")
    ap.add_argument("--raise", dest="raise_name", default=None)
    ap.add_argument("--last-resort-bind", action="store_true")
    opts = ap.parse_args()
    if opts.last_resort_bind:
        from _portmux_stop_signal_server import force_last_resort_bind
        force_last_resort_bind()

    from localm import appface
    from localm.cli import main

    def _refuse_real_ui(*args, **kwargs):
        print("real UI requested", flush=True)
        raise RuntimeError("the test helper must not show a real window")

    appface._StatusWindow = _refuse_real_ui
    appface._WinTray = _refuse_real_ui
    appface.start_app_face = lambda **kwargs: None
    gui_args = ["gui", "--no-model", "--isolated", "-p", opts.port]
    if opts.app_window:
        _use_stub_window(opts.raise_name)
    else:
        gui_args.insert(2, "--no-browser")
    main.main(args=gui_args, prog_name="localm")
