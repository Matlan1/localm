# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper subprocess for test_stop_signal_crash_guard: run `localm gui` in its
app-window mode (the server on a background thread, the main thread held by the
window loop) with the native window replaced by a wait that runs no Python code
until the server closes it, the way a real webview loop can. On POSIX the stop
signals are blocked on the main thread for that wait, so they are handled on
another thread. NOT a test module (underscore prefix -> pytest does not collect
it).

Usage: python _gui_native_stop_signal.py <port> [<signal name>]

With a signal name, the process raises that signal on itself once the window
is open. Prints whether the server_stopped event was handed to the window and
whether it was already set when the window was closed.
"""
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


if __name__ == "__main__":
    from localm import appface
    from localm.cli import main

    _gate = _NativeGate()
    _self_signal = sys.argv[2] if len(sys.argv) > 2 else None
    _handed = {}

    def _run_native_window(url, on_quit=None, **kwargs):
        _handed["server_stopped"] = kwargs.get("server_stopped")
        print(f"server_stopped handed over: {_handed['server_stopped'] is not None}",
              flush=True)
        _gate.block_stop_signals()
        print("native window open", flush=True)
        if _self_signal:
            threading.Timer(0.5, signal.raise_signal,
                            (getattr(signal, _self_signal),)).start()
        _gate.wait()
        print("native window closed", flush=True)
        return True

    def _close_native_window():
        stopped = _handed.get("server_stopped")
        print(f"server_stopped set before close: {bool(stopped and stopped.is_set())}",
              flush=True)
        _gate.open()

    appface.native_window_available = lambda: True
    appface.run_native_window = _run_native_window
    appface.close_native_window = _close_native_window
    main.main(args=["gui", "--no-model", "--isolated", "-p", sys.argv[1]],
              prog_name="localm")
