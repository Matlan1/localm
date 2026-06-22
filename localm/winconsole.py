# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Windows console hardening for long-running server processes.

Windows consoles default to QuickEdit mode: a single stray click into the
window starts a text selection, which suspends every write to the console -
the next ``print()`` blocks, and with it whatever the server was doing
(model loading, inference, downloads) until someone presses a key. That is
fine for interactive shells and disastrous for servers.

``disable_quickedit()`` clears QuickEdit for the current console. Text can
still be copied: right-click the title bar → Edit → Mark (or re-enable
QuickEdit in the console properties for that window).
"""

from __future__ import annotations

import sys

# Windows console control event codes (wincon.h).
CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

# Events that TERMINATE the process without raising KeyboardInterrupt, so Python
# finally/atexit never run. We must clean up (free native GPU/model contexts) in
# the control handler itself, or they get freed during interpreter teardown -
# the segfault-on-window-close the user reported. Ctrl+C / Ctrl+Break are NOT in
# this set: they are passed through to Python's default handler so the existing
# KeyboardInterrupt shutdown path keeps working.
_TERMINATING_EVENTS = frozenset(
    {CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT})

# Keep registered ctypes callbacks alive for the process lifetime (Windows holds
# only a raw pointer; if the Python object is GC'd the callback faults).
_console_handlers: list = []


def _is_terminating_event(ctrl_type: int) -> bool:
    """True for a console event that hard-terminates the process (close / logoff
    / shutdown), so cleanup must run inside the handler."""
    return ctrl_type in _TERMINATING_EVENTS


def _dispatch(ctrl_type: int, cleanup) -> bool:
    """Run *cleanup* for a terminating console event; pass everything else
    through. Always returns False so the OS default handler still proceeds.
    Pure routing, separated from the ctypes plumbing so it can be tested."""
    if _is_terminating_event(ctrl_type):
        try:
            cleanup()
        except Exception:
            pass
    return False


def register_console_handler(cleanup) -> bool:
    """On Windows, run *cleanup* when the console window is closed (or the user
    logs off / shuts down) - events that terminate the process WITHOUT running
    Python finally/atexit, which otherwise leaves native model/GPU contexts to be
    freed during interpreter teardown (a segfault on exit). *cleanup* must be
    quick and not raise; Windows allows only a few seconds before killing the
    process. Ctrl+C / Ctrl+Break fall through to Python's default handler so the
    normal KeyboardInterrupt stop still runs. Returns True if installed; no-op
    (False) off Windows or when no console is attached."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handler_proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(ctrl_type):
            return _dispatch(ctrl_type, cleanup)

        cb = handler_proto(_handler)
        _console_handlers.append(cb)   # prevent GC of the live callback
        if not kernel32.SetConsoleCtrlHandler(cb, True):
            _console_handlers.remove(cb)
            return False
        return True
    except Exception:
        return False


def disable_quickedit() -> None:
    """Stop a stray click from freezing this process's console output.

    No-op on non-Windows platforms and when there is no console attached
    (pythonw, redirected stdio, services).
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return  # no console attached
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080  # required for QuickEdit changes to stick
        kernel32.SetConsoleMode(
            handle,
            (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE,
        )
    except Exception:
        pass  # cosmetic hardening must never block startup
