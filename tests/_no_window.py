# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse, in the process that runs the tests on Windows, every request that
would put a window on the desktop.

``arm()`` patches:

* ``_winapi.CreateProcess``, which ``subprocess``, ``asyncio`` and
  ``multiprocessing`` all start a Windows process through. ``DETACHED_PROCESS``
  is refused unless the call is inside ``allow_detached()``.
  ``CREATE_NEW_CONSOLE`` is refused unless the STARTUPINFO hides the window,
  and then becomes ``CREATE_NO_WINDOW``. Every call that goes ahead carries a
  STARTUPINFO with ``STARTF_USESHOWWINDOW`` and ``SW_HIDE``.
* ``os.startfile`` and ``webbrowser.open``, which are refused.
* ``tkinter.Tk`` and ``tkinter.Toplevel``, which start withdrawn, and
  ``deiconify``, which leaves a window withdrawn.

A refusal fails the test at the call with ``pytest.fail``, whose exception
derives from ``BaseException``, so an ``except Exception`` in the code under
test does not swallow it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import traceback
import webbrowser

import pytest

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008
STARTF_USESHOWWINDOW = 0x00000001
SW_HIDE = 0

ADVICE = (
    "  No test may put a window on the desktop of the person running the suite.\n"
    "  Fake the spawn (monkeypatch subprocess.Popen, os.startfile or\n"
    "  webbrowser.open), or start the child hidden: CREATE_NO_WINDOW, or\n"
    "  CREATE_NEW_CONSOLE with a STARTUPINFO whose wShowWindow is SW_HIDE.\n"
    "  A test that needs a console-less child starts it inside\n"
    "  tests._no_window.allow_detached(), and that child starts nothing with a\n"
    "  console of its own.")

_ARMED = False
_detached_allowed = threading.local()


@contextlib.contextmanager
def allow_detached():
    """Let ``DETACHED_PROCESS`` through on this thread while the block runs."""
    previous = getattr(_detached_allowed, "on", False)
    _detached_allowed.on = True
    try:
        yield
    finally:
        _detached_allowed.on = previous


def _hides(startupinfo) -> bool:
    """True when *startupinfo* asks for the new window to start hidden."""
    if startupinfo is None:
        return False
    return bool(getattr(startupinfo, "dwFlags", 0) & STARTF_USESHOWWINDOW
                and getattr(startupinfo, "wShowWindow", None) == SW_HIDE)


def hidden(flags: int, startupinfo, *, detached_allowed: bool = False):
    """Return ``(flags, startupinfo, refusal)`` for one CreateProcess call.

    *refusal* is ``None`` when the call may go ahead with the returned *flags*
    and *startupinfo*, else a sentence naming what would have shown a window
    (then the other two are returned unchanged). ``CREATE_NEW_CONSOLE`` with a
    hiding STARTUPINFO becomes ``CREATE_NO_WINDOW``; other flags are returned
    as given. The returned STARTUPINFO always carries ``STARTF_USESHOWWINDOW``
    and ``SW_HIDE``; *startupinfo* itself is never modified (a copy is
    returned, or a new one when it is ``None``).
    """
    if flags & DETACHED_PROCESS and not detached_allowed:
        return flags, startupinfo, (
            "DETACHED_PROCESS leaves the child without a console, so every "
            "console program it starts opens a visible console window")
    if flags & CREATE_NEW_CONSOLE:
        if not _hides(startupinfo):
            return flags, startupinfo, (
                "CREATE_NEW_CONSOLE without a STARTUPINFO that hides it opens a "
                "visible console window")
        flags = (flags & ~CREATE_NEW_CONSOLE) | CREATE_NO_WINDOW
    if startupinfo is None:
        startupinfo = subprocess.STARTUPINFO() if sys.platform == "win32" else _Info()
    else:
        startupinfo = startupinfo.copy()
    startupinfo.dwFlags = getattr(startupinfo, "dwFlags", 0) | STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE
    return flags, startupinfo, None


class _Info:
    """Stand-in for ``subprocess.STARTUPINFO`` where it does not exist."""

    def __init__(self, dwFlags: int = 0, wShowWindow: int = 0):
        self.dwFlags = dwFlags
        self.wShowWindow = wShowWindow

    def copy(self) -> _Info:
        return _Info(self.dwFlags, self.wShowWindow)


def _origin() -> str:
    """The innermost tests/ or localm/ frames that led here, newest first."""
    frames = []
    for frame in reversed(traceback.extract_stack()[:-2]):
        f = frame.filename.replace("\\", "/")
        if ("/tests/" in f or "/localm/" in f) and not f.endswith("/tests/_no_window.py"):
            frames.append(f"{os.path.basename(frame.filename)}:{frame.lineno}")
            if len(frames) >= 5:
                break
    origin = " <- ".join(frames) if frames else "unknown"
    thread = threading.current_thread()
    if thread is not threading.main_thread():
        origin += f" (thread {thread.name!r})"
    return origin


def refuse(reason: str, what: str) -> None:
    """Fail the current test with ``pytest.fail``, naming *reason*, *what* and
    the call site."""
    pytest.fail("BLOCKED by tests/_no_window.py: " + reason + "\n    " + what
                + "\n    from " + _origin() + "\n" + ADVICE, pytrace=False)


_real_create = None


def create_process(application, command, proc_attrs, thread_attrs, inherit,
                   flags, env, cwd, startupinfo):
    """Stand-in for ``_winapi.CreateProcess``: applies ``hidden()``, then calls
    ``_real_create`` with the result, or refuses."""
    flags, startupinfo, refusal = hidden(
        flags, startupinfo, detached_allowed=getattr(_detached_allowed, "on", False))
    if refusal:
        refuse(refusal, str(command)[:200])
    return _real_create(application, command, proc_attrs, thread_attrs, inherit,
                        flags, env, cwd, startupinfo)


def startfile(path, *args, **kwargs):
    refuse("os.startfile opens a window", str(path)[:200])


def browser_open(url, *args, **kwargs):
    refuse("webbrowser.open opens a browser window", str(url)[:200])


_real_tk_init = None
_real_toplevel_init = None


def tk_init(self, *args, **kwargs):
    """``tkinter.Tk.__init__``, then ``withdraw()`` before any idle task runs."""
    _real_tk_init(self, *args, **kwargs)
    self.withdraw()


def toplevel_init(self, *args, **kwargs):
    """``tkinter.Toplevel.__init__``, then ``withdraw()`` before any idle task runs."""
    _real_toplevel_init(self, *args, **kwargs)
    self.withdraw()


def deiconify(self):
    """Stand-in for ``Wm.deiconify``: the window stays withdrawn."""
    return None


def _arm_tk() -> None:
    global _real_tk_init, _real_toplevel_init
    try:
        import tkinter
    except ImportError:
        return
    _real_tk_init = tkinter.Tk.__init__
    _real_toplevel_init = tkinter.Toplevel.__init__
    tkinter.Tk.__init__ = tk_init
    tkinter.Toplevel.__init__ = toplevel_init
    tkinter.Wm.deiconify = deiconify
    tkinter.Wm.wm_deiconify = deiconify


def armed() -> bool:
    return _ARMED


def arm() -> bool:
    """Install the patches once, on Windows only. Returns True when it did."""
    global _ARMED, _real_create
    if _ARMED or sys.platform != "win32":
        return False
    import _winapi

    _real_create = _winapi.CreateProcess
    _winapi.CreateProcess = create_process
    os.startfile = startfile
    webbrowser.open = browser_open
    _arm_tk()
    _ARMED = True
    return True
