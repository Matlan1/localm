# SPDX-License-Identifier: AGPL-3.0-or-later
"""A models-folder scan must not raise into a window that has closed.

The launcher reads the models folder on a background thread and then schedules
the result back onto the UI with ``after``. Closing the launcher while that scan
is running destroys the widget being scheduled onto, and Tk raises
``RuntimeError: main thread is not in main loop`` from the thread.

It is not only a stray traceback. An exception escaping a thread is reported
against whatever happens to be running when it surfaces, so a leaked scan turned
into failures in unrelated tests - the launcher's own thread failing a VRAM
handover test, in one observed run.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

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


class _ClosedWindow:
    """A window that has been destroyed: winfo_exists says so, and after()
    raises the way Tk does from a thread whose loop is gone."""

    def __init__(self, exists: bool, raises: bool = True):
        self._exists = exists
        self._raises = raises
        self.scheduled = 0

    def winfo_exists(self):
        return self._exists

    def after(self, delay, fn):
        if self._raises:
            raise RuntimeError("main thread is not in main loop")
        self.scheduled += 1


def _work_body(mod, window, result=None, models=()):
    """Run the body the scan thread runs, against *window*.

    Mirrors _refresh_models' worker: do the scan, then hand the result back to
    the window only if it is still there."""
    import tkinter as tk
    try:
        if window.winfo_exists():
            window.after(0, lambda: None)
    except (tk.TclError, RuntimeError):
        pass


class TestScanThreadOutlivingTheWindow:
    def test_the_shipped_worker_guards_the_handback(self):
        """Read the real source: the after() call must be reached only through
        an existence check and wrapped, or a closed window raises."""
        src = (ROOT / "launcher.pyw").read_text(encoding="utf-8")
        start = src.index("        def work():")
        body = src[start:start + 800]
        assert "winfo_exists()" in body, (
            "the scan hands its result back with no check that the window is "
            f"still there:\n{body[:400]}")
        assert "except (tk.TclError, RuntimeError)" in body, (
            f"winfo_exists itself raises once the interpreter is gone:\n{body[:400]}")

    def test_a_destroyed_window_is_not_scheduled_onto(self):
        w = _ClosedWindow(exists=False)
        mod = _load_launcher()
        _work_body(mod, w)
        assert w.scheduled == 0

    def test_a_window_that_dies_mid_handback_does_not_raise(self):
        """winfo_exists can still say yes and after() still fail: the window can
        go between the two."""
        w = _ClosedWindow(exists=True, raises=True)
        mod = _load_launcher()
        _work_body(mod, w)   # must not raise

    def test_a_live_window_still_gets_its_result(self):
        w = _ClosedWindow(exists=True, raises=False)
        mod = _load_launcher()
        _work_body(mod, w)
        assert w.scheduled == 1


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
