# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep every process the test suite starts on Windows off the desktop.

``arm()`` patches, in the process that runs the tests:

* ``_winapi.CreateProcess``, the one call ``subprocess``, ``asyncio`` and
  ``multiprocessing`` all start a Windows process through. Every child gets
  ``CREATE_NO_WINDOW`` (its own console, with no window) and a STARTUPINFO
  that asks for ``SW_HIDE``. A request for ``DETACHED_PROCESS``, or for
  ``CREATE_NEW_CONSOLE`` without a STARTUPINFO that hides it, is refused.
* ``subprocess.Popen._get_handles``, so a child started with no redirection
  still writes to this process's standard handles rather than to its own
  hidden console.
* ``os.startfile`` and ``webbrowser.open``, which are refused, and
  ``os.system``, which runs through ``subprocess`` and so through the first patch.

A refusal raises ``RuntimeError`` at the call and is recorded in ``HITS``,
keyed by ``(reason, origin)``, so a caller that swallows the error still
fails the test (see the fixture in ``tests/conftest.py``).
"""

from __future__ import annotations

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

HITS: dict[tuple[str, str], int] = {}

ADVICE = (
    "  No test may put a window on the desktop of the person running the suite.\n"
    "  Fake the spawn (monkeypatch subprocess.Popen, os.startfile or\n"
    "  webbrowser.open), or start the child hidden: CREATE_NO_WINDOW, or\n"
    "  CREATE_NEW_CONSOLE with a STARTUPINFO whose wShowWindow is SW_HIDE.")

_ARMED = False


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
    hiding STARTUPINFO becomes ``CREATE_NO_WINDOW``; any other allowed call
    gains ``CREATE_NO_WINDOW``. The returned STARTUPINFO always carries
    ``STARTF_USESHOWWINDOW`` and ``SW_HIDE``; *startupinfo* itself is never
    modified (a copy is returned, or a new one when it is ``None``).
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
        flags &= ~CREATE_NEW_CONSOLE
    flags |= CREATE_NO_WINDOW
    if startupinfo is None:
        startupinfo = subprocess.STARTUPINFO() if sys.platform == "win32" else _Info()
    elif hasattr(startupinfo, "copy"):
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
    """Record a refusal in ``HITS`` and raise ``RuntimeError`` naming it."""
    key = (f"{reason}: {what}"[:300], _origin())
    HITS[key] = HITS.get(key, 0) + 1
    raise RuntimeError("BLOCKED by tests/_no_window.py: " + reason + "\n    " + what
                       + "\n" + ADVICE)


def format_hits(hits) -> str:
    return "\n".join(f"    {reason}\n        from {origin}   x{count}"
                     for (reason, origin), count in sorted(hits.items()))


def _base_popen() -> type:
    for cls in subprocess.Popen.__mro__:
        if cls.__module__ == "subprocess" and cls.__name__ == "Popen":
            return cls
    return subprocess.Popen


_real_create = None
_real_get_handles = None


def create_process(application, command, proc_attrs, thread_attrs, inherit,
                   flags, env, cwd, startupinfo):
    """Stand-in for ``_winapi.CreateProcess``: applies ``hidden()``, then calls
    ``_real_create`` with the result, or refuses."""
    flags, startupinfo, refusal = hidden(flags, startupinfo)
    if refusal:
        refuse(refusal, str(command)[:200])
    return _real_create(application, command, proc_attrs, thread_attrs, inherit,
                        flags, env, cwd, startupinfo)


def get_handles(self, stdin, stdout, stderr):
    """Stand-in for ``Popen._get_handles``: with no redirection at all, hand the
    child this process's stdin, stdout and stderr explicitly. Falls back to no
    handles when fd 2 is not open."""
    handles = _real_get_handles(self, stdin, stdout, stderr)
    if handles == (-1, -1, -1, -1, -1, -1):
        try:
            handles = _real_get_handles(self, None, None, 2)
        except OSError:
            pass
    return handles


def startfile(path, *args, **kwargs):
    refuse("os.startfile opens a window", str(path)[:200])


def browser_open(url, *args, **kwargs):
    refuse("webbrowser.open opens a browser window", str(url)[:200])


def system(command):
    """``os.system`` run through ``subprocess`` as ``%COMSPEC% /c <command>``;
    returns the exit code."""
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    return subprocess.run(f'"{comspec}" /c {command}').returncode


def armed() -> bool:
    return _ARMED


def arm() -> bool:
    """Install the patches once, on Windows only. Returns True when it did."""
    global _ARMED, _real_create, _real_get_handles
    if _ARMED or sys.platform != "win32":
        return False
    import _winapi

    _real_create = _winapi.CreateProcess
    _winapi.CreateProcess = create_process
    popen = _base_popen()
    _real_get_handles = popen._get_handles
    popen._get_handles = get_handles
    os.startfile = startfile
    webbrowser.open = browser_open
    os.system = system
    _ARMED = True
    return True
