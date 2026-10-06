# SPDX-License-Identifier: AGPL-3.0-or-later
"""The launcher's background tasks hand their results back to the window safely.

The launcher runs the models-folder scan and model imports on worker threads.
The worker never calls into Tk: the Tk thread polls for the result with
``after``. A scan that finishes before ``mainloop()`` starts still reaches the
window, a scan failure is shown in the status line, and closing the window
mid-scan raises nothing from the worker thread.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

from tests._tk_root import build_tk_root

ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    """Import launcher.pyw.

    ``.pyw`` is in ``importlib.machinery.SOURCE_SUFFIXES`` only on Windows, so
    ``spec_from_file_location`` returns None elsewhere and ``spec.loader``
    raises AttributeError. Naming the loader keeps the import working on every
    platform. See test_launcher_pyw_loads_without_the_pyw_suffix.
    """
    from importlib.machinery import SourceFileLoader

    name = "localm_launcher_scan_probe"
    path = str(ROOT / "launcher.pyw")
    spec = importlib.util.spec_from_file_location(
        name, path, loader=SourceFileLoader(name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_launcher_pyw_loads_without_the_pyw_suffix(monkeypatch):
    """Both launcher helpers must import launcher.pyw on a platform where
    ``.pyw`` is not a source suffix, which is every platform except Windows.

    SOURCE_SUFFIXES is mutated IN PLACE rather than rebound: the import
    machinery holds the original list object, so rebinding the name leaves the
    real suffix set untouched and the simulation silently does nothing.
    """
    import importlib.machinery as machinery

    for name in ("localm_launcher_scan_probe", "localm_launcher_pyw"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    original = list(machinery.SOURCE_SUFFIXES)
    try:
        machinery.SOURCE_SUFFIXES[:] = [s for s in original if s != ".pyw"]
        assert importlib.util.spec_from_file_location(
            "probe", str(ROOT / "launcher.pyw")) is None, (
            "without an explicit loader the spec must be None here, or this "
            "test is not exercising the failure it exists for")

        from tests.test_launcher_console_hold import _load_launcher_pyw

        assert _load_launcher() is not None
        assert _load_launcher_pyw("localm_launcher_pyw") is not None
    finally:
        machinery.SOURCE_SUFFIXES[:] = original


class _FakeWindow:
    """Stands in for the Tk window: records every after() call and the thread
    it came from, and runs scheduled callbacks only when pump() is called."""

    def __init__(self):
        self.pending = []
        self.after_threads = []
        self.status = []
        self.busy = []
        self.ticker_stopped = 0

    def after(self, delay, fn, *args):
        self.after_threads.append(threading.current_thread())
        self.pending.append((fn, args))

    def pump(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self.pending and time.monotonic() < deadline:
            fn, args = self.pending.pop(0)
            fn(*args)
            time.sleep(0.01)

    def _stop_ticker(self):
        self.ticker_stopped += 1

    def _set_busy(self, busy):
        self.busy.append(busy)

    def status_msg(self, text, error=False):
        self.status.append((text, error))


def _bind(mod, window):
    """Give *window* the launcher's real background-task methods."""
    window._run_in_background = mod.Launcher._run_in_background.__get__(window)
    window._poll_background = mod.Launcher._poll_background.__get__(window)
    return window


class TestBackgroundTaskHandoff:
    def test_the_worker_thread_never_calls_into_tk(self):
        mod = _load_launcher()
        w = _bind(mod, _FakeWindow())
        go = threading.Event()

        def work():
            go.wait(5)
            return "scan-result"

        got = []
        w._run_in_background(work, got.append)
        go.set()
        w.pump()
        assert got == ["scan-result"]
        main = threading.main_thread()
        assert w.after_threads, "the result was never polled for"
        assert all(t is main for t in w.after_threads), (
            "after() was called from a worker thread: "
            f"{[t.name for t in w.after_threads if t is not main]}")

    def test_a_result_ready_before_polling_starts_is_delivered(self):
        mod = _load_launcher()
        w = _bind(mod, _FakeWindow())
        done = threading.Event()

        def work():
            done.set()
            return 7

        got = []
        w._run_in_background(work, got.append)
        assert done.wait(5)
        time.sleep(0.05)
        w.pump()
        assert got == [7]

    def test_a_failing_task_clears_busy_and_reports_the_error(self):
        mod = _load_launcher()
        w = _bind(mod, _FakeWindow())

        def work():
            raise OSError("disk went away")

        got = []
        w._run_in_background(work, got.append)
        w.pump()
        assert got == []
        assert w.ticker_stopped == 1
        assert w.busy == [False]
        assert w.status and w.status[-1][1] is True
        assert "disk went away" in w.status[-1][0]


class TestNoThreadEscapesTheSuite:
    def test_an_exception_in_a_worker_thread_would_be_seen(self):
        """Fires-control for the harness itself: pytest reports an exception that
        escapes a thread, which is why a leaked scan could fail other tests."""
        seen = {}

        def boom():
            try:
                raise RuntimeError("main thread is not in main loop")
            except RuntimeError as e:
                seen["err"] = str(e)

        t = threading.Thread(target=boom)
        t.start()
        t.join()
        assert "main thread is not in main loop" in seen["err"]


def _make_launcher(mod):
    """Construct the real launcher window."""
    return build_tk_root(mod.Launcher)


def _real_launcher(mod, monkeypatch, *, sync=None, models=("probe-model",)):
    """Build the real launcher window with the models-folder scan stubbed.

    *sync* is what ``sync_models_dir_safe`` returns."""
    monkeypatch.setattr(mod, "sync_models_dir_safe",
                        lambda: sync if sync is not None else (None, None))
    monkeypatch.setattr(mod, "load_models", lambda: list(models))
    app = _make_launcher(mod)
    app.withdraw()
    return app


def _run_until(app, done, timeout=5.0):
    """Run the real Tk main loop until ``done()`` is true or *timeout* passes."""
    deadline = time.monotonic() + timeout

    def check():
        if done() or time.monotonic() > deadline:
            app.quit()
            return
        app.after(50, check)

    app.after(50, check)
    app.mainloop()


class TestScanResultReachesTheWindow:
    def test_a_scan_that_finishes_before_the_main_loop_still_completes(
            self, monkeypatch):
        """An empty models folder scans fast enough to finish while the window
        is still being built. The result must still reach the window: the
        ticker stops and Launch is usable again."""
        mod = _load_launcher()
        thread_errors = []
        monkeypatch.setattr(threading, "excepthook",
                            lambda args: thread_errors.append(args.exc_value))
        app = _real_launcher(mod, monkeypatch)
        try:
            # Startup work between the scan starting and mainloop() running.
            time.sleep(1.5)
            _run_until(app, lambda: not app._ticker_active)
            assert not app._ticker_active, (
                "the scan finished but its result never reached the window; "
                f"status stuck at: {app.status.cget('text')!r}")
            assert str(app.launch_btn.cget("state")) == "normal"
            assert "probe-model" in app.model_box["values"]
            assert thread_errors == []
        finally:
            app.destroy()

    def test_a_scan_failure_is_reported_not_shown_as_up_to_date(
            self, monkeypatch):
        mod = _load_launcher()
        app = _real_launcher(
            mod, monkeypatch,
            sync=(None, "PermissionError: [Errno 13] models folder"))
        try:
            _run_until(app, lambda: not app._ticker_active)
            text = app.status.cget("text")
            assert "Could not check the models folder" in text, text
            assert str(app.launch_btn.cget("state")) == "normal"
        finally:
            app.destroy()

    def test_closing_the_window_during_a_scan_raises_nothing(self, monkeypatch):
        mod = _load_launcher()
        thread_errors = []
        monkeypatch.setattr(threading, "excepthook",
                            lambda args: thread_errors.append(args.exc_value))
        release = threading.Event()
        started = threading.Event()
        workers = []

        def slow_scan():
            workers.append(threading.current_thread())
            started.set()
            release.wait(5)
            return None, None

        monkeypatch.setattr(mod, "sync_models_dir_safe", slow_scan)
        monkeypatch.setattr(mod, "load_models", lambda: [])
        app = _make_launcher(mod)
        app.withdraw()
        assert started.wait(5)
        app.destroy()
        release.set()
        workers[0].join(5)
        assert not workers[0].is_alive()
        assert thread_errors == []
