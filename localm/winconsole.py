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
