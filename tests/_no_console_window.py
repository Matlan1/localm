"""Keeps a process that a test starts from opening its own console window on Windows.

A console-subsystem child (git, a .bat, powershell, python) started by a process that has no console
gets a new console, which is a visible window on the desktop. ``arm()`` wraps the two places a
Windows process is created from Python, ``_winapi.CreateProcess`` (subprocess, asyncio and
multiprocessing all end there) and ``os.system``, so a spawn from a console-less process carries
CREATE_NO_WINDOW. A process that has a console (a terminal, a hidden console from CREATE_NO_WINDOW)
is left alone: its children share that console and Ctrl+C still reaches them.
"""

from __future__ import annotations

import os
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_CONSOLE = 0x00000010

_KEEP_AS_IS = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_CONSOLE

_armed = False


def has_console() -> bool:
    """True when this process is attached to a console, visible or hidden. Windows only."""
    import ctypes

    return ctypes.windll.kernel32.GetConsoleProcessList((ctypes.c_uint * 1)(), 1) > 0


def creation_flags_for(flags: int, console_attached: bool) -> int:
    """The creation flags a child should be started with.

    CREATE_NO_WINDOW is added when the creating process has no console and *flags* does not already
    choose a console behaviour (CREATE_NO_WINDOW, DETACHED_PROCESS or CREATE_NEW_CONSOLE)."""
    if console_attached or flags & _KEEP_AS_IS:
        return flags
    return flags | CREATE_NO_WINDOW


def is_armed() -> bool:
    return _armed


def arm() -> bool:
    """Install the guard in this process. Idempotent; returns False off Windows."""
    global _armed
    if sys.platform != "win32":
        return False
    if _armed:
        return True
    import _winapi

    real_create = _winapi.CreateProcess
    real_system = os.system

    def create_process(app, cmd, proc_attrs, thread_attrs, inherit, flags, *rest):
        return real_create(app, cmd, proc_attrs, thread_attrs, inherit,
                           creation_flags_for(flags, has_console()), *rest)

    def system(command):
        if has_console():
            return real_system(command)
        return subprocess.call(command, shell=True)

    _winapi.CreateProcess = create_process
    os.system = system
    _armed = True
    return True
