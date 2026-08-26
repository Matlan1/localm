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

# Console events that terminate the process without running Python
# finally/atexit, so cleanup must happen inside the handler itself. Ctrl+C and
# Ctrl+Break are excluded and fall through to Python's default handler.
_TERMINATING_EVENTS = frozenset(
    {CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT})

# Keeps registered ctypes callbacks alive for the process lifetime; Windows
# holds only a raw pointer.
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


def _sanitize_title(title: str) -> str:
    """A safe console-title string: printable characters only (so a model name or
    path can never inject a control / escape sequence into the terminal), trimmed,
    capped, and never empty."""
    text = "".join(ch for ch in (title or "") if ch.isprintable())
    text = text.strip()[:200]
    return text or "LocaLM"


def set_console_title(title: str) -> bool:
    """Set the console / terminal window title, so the server window reads
    'LocaLM ...' instead of a python.exe path.

    Windows uses SetConsoleTitleW; other platforms emit an OSC title escape, but
    ONLY when stdout is a real terminal (never into a redirected file, a pipe, or
    a service log). Best-effort and fully guarded: a failure never raises and
    never blocks startup. Returns True if a title was set.
    """
    text = _sanitize_title(title)
    try:
        if sys.platform == "win32":
            import ctypes
            return bool(ctypes.windll.kernel32.SetConsoleTitleW(text))
        out = sys.stdout
        if out is not None and hasattr(out, "isatty") and out.isatty():
            out.write(f"\033]0;{text}\007")
            out.flush()
            return True
        return False
    except Exception:
        return False  # cosmetic hardening must never block startup


def hide_console() -> bool:
    """Hide THIS process's own console window (Windows), so a launcher-spawned
    server runs like a background app - just its tray + status window, no raw
    console. Only the caller knows it is safe (the console is ours, not a user's
    interactive terminal), so this is opt-in via that caller. No-op off Windows or
    when there is no console. Best-effort; returns True if a window was hidden."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return False
        SW_HIDE = 0
        ctypes.windll.user32.ShowWindow(hwnd, SW_HIDE)
        return True
    except Exception:
        return False
