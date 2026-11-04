# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refuse, in the process that runs the tests on Windows, every request that
would put a window on the desktop, unless the test asked for windows with
``allow_window()``.

``arm()`` does nothing unless pytest is already imported. It patches:

* ``_winapi.CreateProcess``, which ``subprocess``, ``asyncio`` and
  ``multiprocessing`` all start a Windows process through. ``DETACHED_PROCESS``
  is refused. ``CREATE_NEW_CONSOLE`` is refused unless the STARTUPINFO hides
  the window, and then becomes ``CREATE_NO_WINDOW``. Every call that goes ahead
  carries a STARTUPINFO with ``STARTF_USESHOWWINDOW`` and ``SW_HIDE``.
* ``os.startfile`` and ``webbrowser.open``, which are refused.
* ``tkinter.Tk`` (with Tk loaded) and ``tkinter.Toplevel``, which start
  withdrawn. ``deiconify``, ``iconify`` and ``state("normal")`` leave a window
  withdrawn. Native dialogs (``tkinter.commondialog.Dialog.show``) and
  ``wait_visibility`` are refused.

Inside ``allow_window()`` every patch passes the call through unchanged, on
every thread of the process, and ``tests._no_console_window`` adds no flag.

A refusal fails the test at the call with ``pytest.fail``, whose exception
derives from ``BaseException``, so an ``except Exception`` in the code under
test does not swallow it. ``pytest.fail`` is looked up once, by ``arm()``, so a
refusal imports nothing. On a background thread the refusal ends that thread
and pytest reports it as a warning; the window is refused all the same.

Not covered: ``os.spawn*``, ``os.exec*``, ctypes calls, ``tk.call("wm", ...)``,
and programs a child process starts on its own.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import traceback
import webbrowser

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008
STARTF_USESHOWWINDOW = 0x00000001
SW_HIDE = 0

ADVICE = (
    "  No test may put a window on the desktop by accident.\n"
    "  Fake the spawn (monkeypatch subprocess.Popen, os.startfile or\n"
    "  webbrowser.open), or start the child hidden: CREATE_NO_WINDOW, or\n"
    "  CREATE_NEW_CONSOLE with a STARTUPINFO whose wShowWindow is SW_HIDE.\n"
    "  A test that needs a window runs that part inside\n"
    "  tests._no_window.allow_window().")

_ARMED = False
_allowed = 0
_allowed_lock = threading.Lock()


@contextlib.contextmanager
def allow_window():
    """Pass every window request through unchanged, on every thread, while the
    block runs. Blocks may nest."""
    global _allowed
    with _allowed_lock:
        _allowed += 1
    try:
        yield
    finally:
        with _allowed_lock:
            _allowed -= 1


def window_allowed() -> bool:
    """True while any ``allow_window()`` block is running."""
    return _allowed > 0


def _hides(startupinfo) -> bool:
    """True when *startupinfo* asks for the new window to start hidden."""
    if startupinfo is None:
        return False
    return bool(getattr(startupinfo, "dwFlags", 0) & STARTF_USESHOWWINDOW
                and getattr(startupinfo, "wShowWindow", None) == SW_HIDE)


def hidden(flags: int, startupinfo):
    """Return ``(flags, startupinfo, refusal)`` for one CreateProcess call.

    *refusal* is ``None`` when the call may go ahead with the returned *flags*
    and *startupinfo*, else a sentence naming what would have shown a window
    (then the other two are returned unchanged). ``CREATE_NEW_CONSOLE`` with a
    hiding STARTUPINFO becomes ``CREATE_NO_WINDOW``; other flags are returned
    as given. The returned STARTUPINFO always carries ``STARTF_USESHOWWINDOW``
    and ``SW_HIDE``; *startupinfo* itself is never modified (a copy is
    returned, or a new one when it is ``None``).
    """
    if flags & DETACHED_PROCESS:
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
    _fail("BLOCKED by tests/_no_window.py: " + reason + "\n    " + what
          + "\n    from " + _origin() + "\n" + ADVICE, pytrace=False)


_real_create = None
_fail = None


def create_process(application, command, proc_attrs, thread_attrs, inherit,
                   flags, env, cwd, startupinfo):
    """Stand-in for ``_winapi.CreateProcess``: applies ``hidden()``, then calls
    ``_real_create`` with the result, or refuses. Inside ``allow_window()`` it
    calls ``_real_create`` with the arguments as given."""
    if not window_allowed():
        flags, startupinfo, refusal = hidden(flags, startupinfo)
        if refusal:
            refuse(refusal, str(command)[:200])
    return _real_create(application, command, proc_attrs, thread_attrs, inherit,
                        flags, env, cwd, startupinfo)


_real_startfile = None
_real_browser_open = None


def startfile(path, *args, **kwargs):
    """Stand-in for ``os.startfile``: refuses, or calls it inside ``allow_window()``."""
    if window_allowed():
        return _real_startfile(path, *args, **kwargs)
    refuse("os.startfile opens a window", str(path)[:200])


def browser_open(url, *args, **kwargs):
    """Stand-in for ``webbrowser.open``: refuses, or calls it inside ``allow_window()``."""
    if window_allowed():
        return _real_browser_open(url, *args, **kwargs)
    refuse("webbrowser.open opens a browser window", str(url)[:200])


_real_tk_init = None
_real_toplevel_init = None
_real_deiconify = None
_real_iconify = None
_real_state = None
_real_dialog_show = None
_real_wait_visibility = None


def _loads_tk(args, kwargs) -> bool:
    """Whether ``tkinter.Tk.__init__`` called with *args* loads Tk; ``tkinter.Tcl``
    passes ``useTk=False`` as the fourth positional argument."""
    return bool(args[3] if len(args) > 3 else kwargs.get("useTk", True))


def tk_init(self, *args, **kwargs):
    """``tkinter.Tk.__init__``, then ``withdraw()`` before any idle task runs,
    unless Tk is not loaded or inside ``allow_window()``."""
    _real_tk_init(self, *args, **kwargs)
    if _loads_tk(args, kwargs) and not window_allowed():
        self.withdraw()


def toplevel_init(self, *args, **kwargs):
    """``tkinter.Toplevel.__init__``, then ``withdraw()`` before any idle task
    runs, except inside ``allow_window()``."""
    _real_toplevel_init(self, *args, **kwargs)
    if not window_allowed():
        self.withdraw()


def deiconify(self):
    """Stand-in for ``Wm.deiconify``: the window stays withdrawn, except inside
    ``allow_window()``."""
    if window_allowed():
        return _real_deiconify(self)
    return None


def iconify(self):
    """Stand-in for ``Wm.iconify``: the window stays withdrawn, except inside
    ``allow_window()``."""
    if window_allowed():
        return _real_iconify(self)
    return None


def state(self, newstate=None):
    """Stand-in for ``Wm.state``: reads the state, or sets it to ``withdrawn``;
    any other state is ignored, except inside ``allow_window()``."""
    if newstate is None or newstate == "withdrawn" or window_allowed():
        return _real_state(self, newstate)
    return None


def dialog_show(self, **options):
    """Stand-in for ``tkinter.commondialog.Dialog.show``: refuses, or calls it
    inside ``allow_window()``."""
    if window_allowed():
        return _real_dialog_show(self, **options)
    refuse("a native Tk dialog opens a window", type(self).__name__)


def wait_visibility(self, window=None):
    """Stand-in for ``Misc.wait_visibility``: refuses, since a withdrawn window
    never becomes visible, or calls it inside ``allow_window()``."""
    if window_allowed():
        return _real_wait_visibility(self, window)
    refuse("wait_visibility waits for a window on the desktop", str(window or self))


def _arm_tk() -> None:
    global _real_tk_init, _real_toplevel_init, _real_deiconify, _real_iconify
    global _real_state, _real_dialog_show, _real_wait_visibility
    try:
        import tkinter
        import tkinter.commondialog
    except ImportError:
        return
    _real_tk_init = tkinter.Tk.__init__
    _real_toplevel_init = tkinter.Toplevel.__init__
    _real_deiconify = tkinter.Wm.wm_deiconify
    _real_iconify = tkinter.Wm.wm_iconify
    _real_state = tkinter.Wm.wm_state
    _real_dialog_show = tkinter.commondialog.Dialog.show
    _real_wait_visibility = tkinter.Misc.wait_visibility
    tkinter.Tk.__init__ = tk_init
    tkinter.Toplevel.__init__ = toplevel_init
    tkinter.Wm.deiconify = deiconify
    tkinter.Wm.wm_deiconify = deiconify
    tkinter.Wm.iconify = iconify
    tkinter.Wm.wm_iconify = iconify
    tkinter.Wm.state = state
    tkinter.Wm.wm_state = state
    tkinter.commondialog.Dialog.show = dialog_show
    tkinter.Misc.wait_visibility = wait_visibility


def armed() -> bool:
    return _ARMED


def arm() -> bool:
    """Install the patches once, on Windows, in a process that has imported
    pytest. Returns True when it did."""
    global _ARMED, _real_create, _real_startfile, _real_browser_open, _fail
    if _ARMED or sys.platform != "win32" or "pytest" not in sys.modules:
        return False
    import _winapi

    _fail = sys.modules["pytest"].fail

    _real_create = _winapi.CreateProcess
    _winapi.CreateProcess = create_process
    _real_startfile = os.startfile
    _real_browser_open = webbrowser.open
    os.startfile = startfile
    webbrowser.open = browser_open
    _arm_tk()
    _ARMED = True
    return True
