# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "running app" control surface: a small set of actions - Open, Copy
address, View logs, Restart, Stop - presented natively per OS so the localm
server reads as a real background app instead of a python.exe console.

Now: a Windows tray icon via the Win32 API through ctypes (NO dependency - the
llama.cpp-binding ethos: bind thinly to a shipped system API, do not import an
unknown wrapper). Later: a small styled Tk control window on Linux (bundled
stdlib Tk), so Linux gets a nice server app too.

Everything here is best-effort and fully guarded: the control surface is a
convenience layered on the server, so a failure to show it must NEVER stop the
server from running.
"""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path
from typing import Callable, Optional


def icon_path() -> Optional[str]:
    """The bundled LocaLM .ico, or None if missing."""
    p = Path(__file__).resolve().parents[1] / "assets" / "localm.ico"
    return str(p) if p.is_file() else None


def copy_to_clipboard(text: str) -> bool:
    """Put *text* on the clipboard with no dependency (Win32 clipboard via
    ctypes). Best-effort; returns True on success."""
    if sys.platform == "win32":
        try:
            import ctypes

            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002
            k32 = ctypes.windll.kernel32
            u32 = ctypes.windll.user32
            data = text.encode("utf-16-le") + b"\x00\x00"
            k32.GlobalAlloc.restype = ctypes.c_void_p
            k32.GlobalLock.restype = ctypes.c_void_p
            k32.GlobalLock.argtypes = [ctypes.c_void_p]
            k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            handle = k32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            ptr = k32.GlobalLock(handle)
            ctypes.memmove(ptr, data, len(data))
            k32.GlobalUnlock(handle)
            if u32.OpenClipboard(None):
                u32.EmptyClipboard()
                u32.SetClipboardData(CF_UNICODETEXT, ctypes.c_void_p(handle))
                u32.CloseClipboard()
                return True
        except Exception:
            pass
    return False


def open_logs(logfile) -> None:
    """Open the server logfile in the OS default viewer (best-effort)."""
    try:
        if not logfile:
            return
        if sys.platform == "win32":
            import os
            os.startfile(str(logfile))   # noqa: type-ignore (Windows-only)
        else:
            webbrowser.open("file://" + str(logfile))
    except Exception:
        pass


class AppFace:
    """Handle for a running control surface; .close() tears it down. The base is a
    no-op used off Windows (until the Linux Tk window lands)."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def start_app_face(*, name: str = "LocaLM", url: str, logfile=None,
                   on_restart: Optional[Callable] = None,
                   on_stop: Optional[Callable] = None) -> Optional[AppFace]:
    """Start the control surface for the running server. Returns a handle with
    .close(), or None if unavailable (unsupported platform, no session). NEVER
    raises - a control-surface failure must not take down the server."""
    try:
        if sys.platform == "win32":
            tray = _WinTray(name=name, url=url, logfile=logfile,
                            on_restart=on_restart, on_stop=on_stop)
            if tray.start():
                return tray
            return None
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
#  Windows tray icon (Win32 via ctypes, no dependency)                         #
# --------------------------------------------------------------------------- #

class _WinTray(AppFace):
    # menu command ids
    _ID_OPEN = 1001
    _ID_COPY = 1002
    _ID_LOGS = 1003
    _ID_RESTART = 1004
    _ID_STOP = 1005

    def __init__(self, *, name, url, logfile, on_restart, on_stop):
        self.name = name
        self.url = url
        self.logfile = logfile
        self.on_restart = on_restart
        self.on_stop = on_stop
        self._hwnd = None
        self._nid = None
        self._wndproc_ref = None   # keep the WINFUNCTYPE alive (Windows holds a raw ptr)
        self._ready = threading.Event()
        self._ok = False
        self._thread = None

    # ---- public ----
    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="localm-tray",
                                        daemon=True)
        self._thread.start()
        # Wait briefly for the icon to register so start_app_face can report success.
        self._ready.wait(timeout=3.0)
        return self._ok

    def close(self) -> None:
        try:
            import ctypes
            if self._hwnd:
                WM_CLOSE = 0x0010
                ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        except Exception:
            pass

    # ---- actions ----
    def _open(self):
        try:
            webbrowser.open(self.url)
        except Exception:
            pass

    def _copy(self):
        copy_to_clipboard(self.url)

    def _logs(self):
        open_logs(self.logfile)

    def _restart(self):
        if self.on_restart:
            # Run off the message-loop thread so the menu closes first.
            threading.Thread(target=self.on_restart, daemon=True).start()

    def _stop(self):
        if self.on_stop:
            threading.Thread(target=self.on_stop, daemon=True).start()

    def _dispatch(self, cmd_id: int):
        {
            self._ID_OPEN: self._open,
            self._ID_COPY: self._copy,
            self._ID_LOGS: self._logs,
            self._ID_RESTART: self._restart,
            self._ID_STOP: self._stop,
        }.get(cmd_id, lambda: None)()

    # ---- the Win32 plumbing (its own thread owns the window + message loop) ----
    def _run(self):
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        kernel32 = ctypes.windll.kernel32

        LRESULT = ctypes.c_ssize_t
        WM_APP = 0x8000
        WM_TRAY = WM_APP + 1
        WM_COMMAND = 0x0111
        WM_RBUTTONUP = 0x0205
        WM_LBUTTONDBLCLK = 0x0203
        WM_CONTEXTMENU = 0x007B
        WM_DESTROY = 0x0002
        WM_CLOSE = 0x0010
        NIM_ADD, NIM_DELETE = 0, 2
        NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x01, 0x02, 0x04
        IMAGE_ICON = 1
        LR_LOADFROMFILE, LR_DEFAULTSIZE = 0x0010, 0x0040
        MF_STRING, MF_SEPARATOR = 0x0000, 0x0800
        TPM_RIGHTBUTTON = 0x0002

        class NOTIFYICONDATAW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("hWnd", wintypes.HWND),
                ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT),
                ("uCallbackMessage", wintypes.UINT),
                ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128),
            ]

        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                                     wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HANDLE),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        # Correct 64-bit prototypes so pointers are not truncated.
        user32.DefWindowProcW.restype = LRESULT
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                          wintypes.WPARAM, wintypes.LPARAM]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR,
                                      wintypes.UINT, ctypes.c_int, ctypes.c_int,
                                      wintypes.UINT]
        shell32.Shell_NotifyIconW.restype = wintypes.BOOL

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_TRAY:
                if lparam in (WM_RBUTTONUP, WM_CONTEXTMENU):
                    self._show_menu(hwnd, user32, MF_STRING, MF_SEPARATOR,
                                    TPM_RIGHTBUTTON)
                elif lparam == WM_LBUTTONDBLCLK:
                    self._open()
                return 0
            if msg == WM_COMMAND:
                self._dispatch(int(wparam) & 0xFFFF)
                return 0
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                if self._nid is not None:
                    shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._nid))
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc_ref = WNDPROC(wndproc)
        hinst = kernel32.GetModuleHandleW(None)
        cls = WNDCLASSW()
        cls.lpfnWndProc = self._wndproc_ref
        cls.hInstance = hinst
        cls.lpszClassName = "LocaLMTrayWindow"
        atom = user32.RegisterClassW(ctypes.byref(cls))
        if not atom:
            # Class may already exist from a prior run in-process; proceed anyway.
            pass

        hwnd = user32.CreateWindowExW(0, "LocaLMTrayWindow", self.name, 0,
                                      0, 0, 0, 0, None, None, hinst, None)
        if not hwnd:
            self._ready.set()
            return
        self._hwnd = hwnd

        hicon = 0
        ipath = icon_path()
        if ipath:
            hicon = user32.LoadImageW(None, ipath, IMAGE_ICON, 0, 0,
                                      LR_LOADFROMFILE | LR_DEFAULTSIZE)

        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = hicon
        nid.szTip = f"{self.name} - {self.url}"[:127]
        self._nid = nid
        self._ok = bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid)))
        self._ready.set()

        # Message loop (owns this thread until WM_CLOSE / WM_DESTROY).
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _show_menu(self, hwnd, user32, MF_STRING, MF_SEPARATOR, TPM_RIGHTBUTTON):
        import ctypes
        from ctypes import wintypes

        hmenu = user32.CreatePopupMenu()
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_OPEN, "Open LocaLM")
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_COPY, "Copy address")
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_LOGS, "View logs")
        user32.AppendMenuW(hmenu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_RESTART, "Restart")
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_STOP, "Stop LocaLM")
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        # Required so the menu dismisses correctly when clicking elsewhere.
        user32.SetForegroundWindow(hwnd)
        user32.TrackPopupMenu(hmenu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, None)
        user32.PostMessageW(hwnd, 0x0000, 0, 0)  # WM_NULL: flush per Win32 docs
        user32.DestroyMenu(hmenu)
