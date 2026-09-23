# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper subprocess for test_stop_signal_crash_guard: run `localm gui` in its
app-window mode (the server on a background thread, the main thread held by the
window loop) with the native window replaced by a plain wait that ends when the
server closes it. NOT a test module (underscore prefix -> pytest does not
collect it).

Usage: python _gui_native_stop_signal.py <port>
"""
import sys
import threading
from pathlib import Path

# Import localm from this worktree rather than the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from localm import appface
    from localm.cli import main

    _closed = threading.Event()

    def _run_native_window(url, on_quit=None, **kwargs):
        print("native window open", flush=True)
        while not _closed.wait(0.1):
            pass
        print("native window closed", flush=True)
        return True

    appface.native_window_available = lambda: True
    appface.run_native_window = _run_native_window
    appface.close_native_window = _closed.set
    main.main(args=["gui", "--no-model", "--isolated", "-p", sys.argv[1]],
              prog_name="localm")
