# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "running app" control surface: a small set of actions - Open, Copy
address, View logs, Restart, Stop - presented natively per OS so the localm
server reads as a real background app instead of a python.exe console.

Now: a Windows tray icon via the Win32 API through ctypes, with no dependency.
Later: a small styled Tk control window on Linux (bundled stdlib Tk).

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

from localm.debuglog import logger


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


def _native_window_allowed_by_preference() -> bool:
    """Has the user explicitly turned the app window off (config key
    desktop_window_mode == "browser")? Defaults to True (allowed) on any read
    failure, so a config problem never silently disables a feature the user did
    not ask to disable - the same posture run_native_window's own
    desktop_window_quit_on_close read uses. Never raises."""
    try:
        from localm.config import load_config
        return load_config().get("desktop_window_mode", "auto") != "browser"
    except Exception:
        logger.debug("appface: reading desktop_window_mode failed, "
                     "defaulting to allowed", exc_info=True)
        return True


def native_window_available() -> bool:
    """Cheap, side-effect-free probe: is the optional ``localm[desktop]``
    extra (pywebview) installed AND has the user not turned it off via the
    desktop_window_mode setting?

    Used ONLY to decide, before anything else starts, whether the caller
    should hand this process's actual main thread to the native window loop
    instead of the server - see run_native_window's docstring for why that
    matters. The real question of whether a window actually opens and loads
    is answered by run_native_window's own return value, not this. Never
    raises.
    """
    if "pytest" in sys.modules:
        return False
    try:
        import webview  # noqa: F401
    except ImportError:
        return False
    return _native_window_allowed_by_preference()


# The single active native window, if any. Module state because the tray's
# "Open" and the server's shutdown call in from other threads, and pywebview
# supports only one window loop per process.
_native_window = None
_native_window_may_really_close = threading.Event()


def run_native_window(url: str, name: str = "LocaLM", *,
                      hide_on_close: bool = True,
                      on_quit: Optional[Callable] = None) -> bool:
    """Open *url* in a native OS webview window, BLOCKING the calling thread
    for the lifetime of the app.

    MUST be called from the process's actual main thread: pywebview raises
    ``WebViewException`` unconditionally otherwise
    (``if threading.current_thread().name != 'MainThread'`` in
    webview/__init__.py, with no override flag). A caller that wants this AND has
    something else already occupying the main thread (the server, in gui/cli.py's
    fresh-launch path) must move that something else to its own background thread
    first.

    *hide_on_close* (default True - the fresh-launch, this-process-owns-the-
    server case): the window's own close button HIDES it instead of destroying
    it, matching this module's _StatusWindow/_WinTray behaviour, so a later
    show_native_window() (the tray/status "Open" action) can bring the SAME
    window back rather than needing a second webview.start() call, which has no
    free main thread to run on (the caller of show_native_window is the tray/Tk
    thread, never the main thread). The user's own "quit when the app window is
    closed" preference (config key desktop_window_quit_on_close, off by default)
    overrides this per close: when on, the window closes for real and *on_quit*
    (if given - the same action the tray's Stop button triggers) is invoked. Pass
    False for a window that owns no server of its own to keep alive (gui/cli.py's
    attach-to-an-already-running-instance path); it then closes normally on its
    own close button, with no hiding, no setting lookup and no on_quit. This
    function returns only once the window is genuinely destroyed - via
    close_native_window() (normally the server's own shutdown path, once it has
    genuinely stopped), the user's own close with the quit preference on, or
    hide_on_close=False's plain close - or the window fails to load at all.

    Returns True only once the window actually LOADED the page, via pywebview's
    ``window.events.loaded`` (a plain threading.Event with .wait(timeout)),
    watched from a short-lived helper thread since the calling thread is busy
    inside the blocking webview.start() call by then. Returns False whenever a
    real, loaded window cannot be confirmed (extra absent, WebView2/WebKitGTK
    missing or broken, window never loaded) so the caller can fall back to
    webbrowser.open. NEVER raises.
    """
    global _native_window
    if "pytest" in sys.modules:
        return False
    if not _native_window_allowed_by_preference():
        # desktop_window_mode="browser". Also covers the attach-to-a-running-
        # instance call site, which never calls native_window_available().
        return False
    try:
        import webview
    except ImportError:
        # Extra not installed - the expected, common case, not a failure.
        return False
    try:
        window = webview.create_window(name, url, width=1280, height=860,
                                       min_size=(760, 500))
    except Exception:
        logger.debug("appface: native window creation failed", exc_info=True)
        return False
    if window is None:
        return False

    if hide_on_close:
        _native_window_may_really_close.clear()

        def _on_closing():
            # False vetoes the close (args.Cancel = True), matching the
            # WM_DELETE_WINDOW override used elsewhere in this module.
            if _native_window_may_really_close.is_set():
                return True
            try:
                from localm.config import load_config
                quit_on_close = bool(
                    load_config().get("desktop_window_quit_on_close", False))
            except Exception:
                logger.debug("appface: reading desktop_window_quit_on_close "
                             "failed, defaulting to hide", exc_info=True)
                quit_on_close = False
            if quit_on_close:
                if on_quit is not None:
                    # Off the closing-event's own thread, so the close is not
                    # blocked.
                    threading.Thread(target=on_quit, daemon=True).start()
                return True   # allow the real close - the app is stopping
            window.hide()
            return False

        window.events.closing += _on_closing

    loaded = {"v": False}

    def _watch_loaded():
        # window.show() also calls .Activate(); a plain WinForms .Show() on first
        # creation does not reliably take OS foreground focus for a window created
        # by a background-launched process. window.on_top is NOT toggled here: the
        # winforms backend sets TopMost with no thread marshaling, and calling it
        # from this background thread hangs the process.
        if window.events.loaded.wait(timeout=8.0):
            loaded["v"] = True
            try:
                window.show()
            except Exception:
                logger.debug("appface: native window foreground-activate "
                             "failed (non-fatal)", exc_info=True)

    threading.Thread(target=_watch_loaded, name="localm-webview-confirm",
                     daemon=True).start()
    _native_window = window
    try:
        # private_mode=False keeps the login cookie across restarts, like the
        # browser tab this replaces. Blocks until the window is destroyed.
        # gui="qt" on Linux: pywebview tries GTK first, and this project never
        # installs the GTK extra, so qt is the backend the `desktop` extra
        # actually provides there. Windows and macOS keep pywebview's default.
        start_kwargs = {"icon": icon_path(), "private_mode": False}
        if sys.platform.startswith("linux"):
            start_kwargs["gui"] = "qt"
        webview.start(**start_kwargs)
    except Exception:
        logger.debug("appface: native window loop failed", exc_info=True)
        return False
    finally:
        _native_window = None
    return loaded["v"]


def show_native_window() -> bool:
    """Thread-safe: re-show (and foreground-activate) the currently open
    native window, for the tray/status "Open" action. Returns False - the
    caller should fall back to webbrowser.open - when no native window is
    active this run (this platform/launch used a browser tab instead, or the
    window failed to load in the first place). NEVER raises."""
    window = _native_window
    if window is None:
        return False
    try:
        window.show()
        return True
    except Exception:
        logger.debug("appface: show_native_window failed", exc_info=True)
        return False


def close_native_window() -> None:
    """Thread-safe: let the native window ACTUALLY close now (not just
    hide), so run_native_window's blocking webview.start() call finally
    returns and the process can exit. Call this once the server has
    genuinely stopped - never merely because the window was hidden. A no-op
    when no native window is active. NEVER raises."""
    window = _native_window
    if window is None:
        return
    _native_window_may_really_close.set()
    try:
        window.destroy()
    except Exception:
        logger.debug("appface: close_native_window failed", exc_info=True)


def open_logs(logfile) -> None:
    """Open the server logfile in the OS default viewer (best-effort)."""
    try:
        if not logfile:
            return
        if sys.platform == "win32":
            import os
            os.startfile(str(logfile))   # os.startfile is Windows-only (guarded above)
        else:
            webbrowser.open("file://" + str(logfile))
    except Exception:
        pass


class AppFace:
    """Handle for a running control surface; .close() tears it down. The base is a
    no-op used off Windows (until the Linux Tk window lands)."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# Dark palette shared with the web GUI + launcher, so the window matches the app.
_BG = "#0f1115"
_BG_RAISED = "#161a21"
_BG_INPUT = "#1c212b"
_BORDER = "#262c38"
_TEXT = "#d7dde7"
_TEXT_DIM = "#8b94a5"
_ACCENT = "#4f9cf9"
_GREEN = "#3fb68b"
_RED = "#e25d5d"


class _StatusWindow(AppFace):
    """A small styled Tk status/control window (bundled stdlib Tk, no dependency).

    It shows ACCURATE startup progress ("Starting..." -> "Running") and, once
    ready, minimizes to the tray on Windows (hide_on_ready) or stays as a
    console-like status window with a live log tail on Linux. On an error it stays
    and shows the message in red instead of vanishing. Runs its own Tk mainloop in
    a dedicated thread; other threads update it through a queue polled via after().
    Fully guarded - a window failure must never take down the server.
    """

    def __init__(self, *, name, url, logfile=None, get_log_lines=None,
                 on_restart=None, on_stop=None, hide_on_ready=False):
        import queue as _queue
        self.name = name
        self.url = url
        self.logfile = logfile
        self.get_log_lines = get_log_lines
        self.on_restart = on_restart
        self.on_stop = on_stop
        self.hide_on_ready = hide_on_ready
        self._q = _queue.Queue()
        self._up = threading.Event()
        self._ok = False
        self._root = None
        self._thread = None
        self._autoscroll_on = True   # plain bool, NOT a tk.Variable (see _run)

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, name="localm-status",
                                        daemon=True)
        self._thread.start()
        self._up.wait(timeout=4.0)
        return self._ok

    # thread-safe API (called from the server / preload threads)
    def set_status(self, text):
        self._q.put(("status", text))

    def set_ready(self):
        self._q.put(("ready", None))

    def set_error(self, text):
        self._q.put(("error", text))

    def show(self):
        """Re-show the window (from the tray). Thread-safe."""
        self._q.put(("show", None))

    def close(self):
        self._q.put(("close", None))

    # actions (run in the Tk thread)
    def _open(self):
        try:
            # Re-show the existing native window rather than opening a browser
            # tab; a second webview.start() has no free main thread. Falls back to
            # the browser when this run has no native window.
            if not show_native_window():
                webbrowser.open(self.url)
        except Exception:
            pass

    def _copy(self):
        copy_to_clipboard(self.url)

    def _logs(self):
        try:
            lf = Path(self.logfile) if self.logfile else None
            if lf is not None and self.get_log_lines is not None:
                lines = self.get_log_lines() or []
                lf.parent.mkdir(parents=True, exist_ok=True)
                body = "\n".join(lines) if lines else "(no activity logged yet)"
                lf.write_text(f"LocaLM server log  -  {self.url}\n" + ("=" * 48)
                              + "\n" + body + "\n", encoding="utf-8")
            open_logs(lf)
        except Exception:
            pass

    def _restart(self):
        if self.on_restart:
            threading.Thread(target=self.on_restart, daemon=True).start()

    def _stop(self):
        if self.on_stop:
            threading.Thread(target=self.on_stop, daemon=True).start()

    def _toggle_autoscroll(self):
        self._autoscroll_on = not self._autoscroll_on
        try:
            self._as_btn.configure(
                text=f"autoscroll: {'on' if self._autoscroll_on else 'off'}")
        except Exception:
            pass

    def _run(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception:
            # No Tk / no display: signal "up" so start() returns, as a no-op window.
            self._up.set()
            return
        self._root = root
        try:
            root.title(self.name)
            root.configure(bg=_BG)
            root.geometry("640x480")
            root.minsize(520, 380)
            if sys.platform == "win32":
                ip = icon_path()
                if ip:
                    try:
                        root.iconbitmap(ip)
                    except Exception:
                        pass

            header = tk.Frame(root, bg=_BG)
            header.pack(fill="x", padx=16, pady=(14, 4))
            tk.Label(header, text="LocaL", bg=_BG, fg=_TEXT, padx=0, bd=0,
                     font=("Segoe UI", 18, "bold")).pack(side="left")
            tk.Label(header, text="M", bg=_BG, fg=_ACCENT, padx=0, bd=0,
                     font=("Segoe UI", 18, "bold")).pack(side="left")

            self._status_lbl = tk.Label(root, text="Starting LocaLM...", bg=_BG,
                                        fg=_TEXT_DIM, anchor="w", justify="left",
                                        font=("Segoe UI", 11))
            self._status_lbl.pack(fill="x", padx=16)
            # Wrap the status text to the live label width rather than truncating.
            # wraplength follows resizes and does not change the packed width.
            self._status_lbl.bind(
                "<Configure>",
                lambda e: self._status_lbl.configure(
                    wraplength=max(200, e.width - 8)))
            self._addr_lbl = tk.Label(root, text="", bg=_BG, fg=_ACCENT, anchor="w",
                                      font=("Consolas", 10))
            self._addr_lbl.pack(fill="x", padx=16, pady=(2, 8))

            btns = tk.Frame(root, bg=_BG)
            btns.pack(fill="x", padx=16, pady=(0, 6))
            for label, cmd in (("Open", self._open), ("Copy", self._copy),
                               ("View logs", self._logs), ("Restart", self._restart),
                               ("Stop", self._stop)):
                tk.Button(btns, text=label, command=cmd, bd=0, bg=_BG_INPUT,
                          fg=_TEXT, activebackground=_BORDER, activeforeground=_TEXT,
                          padx=10, pady=4, font=("Segoe UI", 9)).pack(side="left",
                                                                      padx=(0, 6))

            wrap = tk.Frame(root, bg=_BORDER)
            wrap.pack(fill="both", expand=True, padx=16, pady=(2, 6))
            # wrap="word": long log lines break onto new lines and stay visible at
            # any window size. With wrap="none" the autoscroll also scrolls
            # horizontally to each line's tail.
            self._log = tk.Text(wrap, bg="#0b0d11", fg=_TEXT_DIM, bd=0,
                                font=("Consolas", 9), wrap="word", height=9,
                                state="disabled")
            self._log.pack(fill="both", expand=True, padx=1, pady=1)

            foot = tk.Frame(root, bg=_BG)
            foot.pack(fill="x", padx=16, pady=(0, 10))
            # A plain toggle button, not a tk.Variable: a Tk Variable held on the
            # instance is finalized by GC in the wrong thread at process exit.
            self._as_btn = tk.Button(foot, text="autoscroll: on",
                                     command=self._toggle_autoscroll, bd=0, bg=_BG,
                                     fg=_TEXT_DIM, activebackground=_BG,
                                     activeforeground=_TEXT, font=("Segoe UI", 8))
            self._as_btn.pack(side="left")

            # Closing the window never stops the server: hide to tray on Windows,
            # iconify (minimize) on Linux. Stop is the deliberate quit.
            root.protocol("WM_DELETE_WINDOW", self._on_x)
            self._ok = True
            self._up.set()
            root.after(150, self._poll)
            root.mainloop()
        except Exception:
            # The control window is layered on the server and must never stop it,
            # but the failure is logged rather than discarded.
            logger.debug("appface: control window failed to start", exc_info=True)
            self._up.set()

    def _on_x(self):
        try:
            if sys.platform == "win32":
                self._root.withdraw()
            else:
                self._root.iconify()
        except Exception:
            pass

    def _poll(self):
        root = self._root
        if root is None:
            return
        import queue as _queue
        try:
            while True:
                try:
                    kind, val = self._q.get_nowait()
                except _queue.Empty:
                    break
                if kind == "status":
                    self._status_lbl.configure(text=val, fg=_TEXT_DIM)
                elif kind == "ready":
                    self._status_lbl.configure(text="Running", fg=_GREEN)
                    self._addr_lbl.configure(text=self.url)
                    if self.hide_on_ready:
                        root.withdraw()   # Windows: minimize to the tray
                elif kind == "error":
                    self._status_lbl.configure(text=val, fg=_RED)
                    try:
                        root.deiconify(); root.lift()
                    except Exception:
                        pass
                elif kind == "show":
                    try:
                        root.deiconify(); root.lift(); root.focus_force()
                    except Exception:
                        pass
                elif kind == "close":
                    try:
                        root.destroy()
                    finally:
                        # Drop every Tk reference so nothing is finalized by GC in
                        # the wrong thread at process exit.
                        self._root = self._log = self._status_lbl = None
                        self._addr_lbl = self._as_btn = None
                        return
            if self.get_log_lines is not None and self._log is not None:
                lines = self.get_log_lines() or []
                self._log.configure(state="normal")
                self._log.delete("1.0", "end")
                self._log.insert("1.0", "\n".join(lines[-250:]))
                self._log.configure(state="disabled")
                if self._autoscroll_on:
                    self._log.see("end")
        except Exception:
            pass
        if self._root is not None:
            self._root.after(300, self._poll)


class _AppFaceHandle(AppFace):
    """Composite control surface: a status window (all platforms) plus a tray
    (Windows). set_status/set_ready/set_error drive the window; close() tears both
    down. All methods are guarded no-ops when a piece is absent."""

    def __init__(self, window, tray):
        self._window = window
        self._tray = tray

    def set_status(self, text):
        if self._window is not None:
            self._window.set_status(text)

    def set_ready(self):
        if self._window is not None:
            self._window.set_ready()

    def set_error(self, text):
        if self._window is not None:
            self._window.set_error(text)

    def close(self):
        for part in (self._tray, self._window):
            try:
                if part is not None:
                    part.close()
            except Exception:
                pass


def start_app_face(*, name: str = "LocaLM", url: str, logfile=None,
                   get_log_lines: Optional[Callable] = None,
                   on_restart: Optional[Callable] = None,
                   on_stop: Optional[Callable] = None) -> Optional[AppFace]:
    """Start the control surface for the running server: a styled status window
    (accurate startup progress; on Windows it hides to the tray when ready, on
    Linux it stays as a console-like status window), plus a native tray on Windows.

    Returns a handle with .close() + set_status/set_ready/set_error, or None if
    nothing could be shown. NEVER raises - a control-surface failure must not take
    down the server. *get_log_lines* feeds the live log tail and "View logs".
    """
    # Never spin up real UI (tray icon / Tk window) inside the test suite.
    if "pytest" in sys.modules:
        return None
    try:
        # Window FIRST, so the tray can re-open it (double-click / "Show LocaLM").
        window = _StatusWindow(name=name, url=url, logfile=logfile,
                               get_log_lines=get_log_lines, on_restart=on_restart,
                               on_stop=on_stop, hide_on_ready=False)
        win_ok = window.start()
        tray = None
        tray_ok = False
        if sys.platform == "win32":
            tray = _WinTray(name=name, url=url, logfile=logfile,
                            get_log_lines=get_log_lines, on_restart=on_restart,
                            on_stop=on_stop,
                            on_show=(window.show if win_ok else None))
            tray_ok = tray.start()
        # Hide to the tray on ready only when the tray is there to bring it back;
        # otherwise it stays visible.
        if win_ok and tray_ok:
            window.hide_on_ready = True
        if not win_ok and not tray_ok:
            return None
        return _AppFaceHandle(window if win_ok else None, tray if tray_ok else None)
    except Exception:
        # Best-effort: no control surface just means the server runs headless. Log
        # the failure so a missing tray/window is diagnosable, not silent.
        logger.debug("appface: control surface unavailable (start failed)",
                     exc_info=True)
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
    _ID_SHOW = 1006

    _COINIT_APARTMENTTHREADED = 0x2

    def __init__(self, *, name, url, logfile, on_restart, on_stop, get_log_lines=None,
                 on_show=None):
        self.name = name
        self.url = url
        self.logfile = logfile
        self.get_log_lines = get_log_lines
        self.on_restart = on_restart
        self.on_stop = on_stop
        self.on_show = on_show
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
            # Re-show the existing native window rather than opening a browser
            # tab; a second webview.start() has no free main thread. Falls back to
            # the browser when this run has no native window.
            if not show_native_window():
                webbrowser.open(self.url)
        except Exception:
            pass

    def _copy(self):
        copy_to_clipboard(self.url)

    def _logs(self):
        # Refresh the readable log from the current activity buffer (INFO+, no
        # chat content), then open it. Best-effort; never raises into the menu.
        try:
            lf = Path(self.logfile) if self.logfile else None
            if lf is not None and self.get_log_lines is not None:
                lines = self.get_log_lines() or []
                lf.parent.mkdir(parents=True, exist_ok=True)
                header = f"LocaLM server log  -  {self.url}\n" + ("=" * 48) + "\n"
                body = "\n".join(lines) if lines else "(no activity logged yet)"
                lf.write_text(header + body + "\n", encoding="utf-8")
            open_logs(lf)
        except Exception:
            pass

    def _restart(self):
        if self.on_restart:
            # Run off the message-loop thread so the menu closes first.
            threading.Thread(target=self.on_restart, daemon=True).start()

    def _stop(self):
        if self.on_stop:
            threading.Thread(target=self.on_stop, daemon=True).start()

    def _show(self):
        if self.on_show:
            self.on_show()

    def _dispatch(self, cmd_id: int):
        {
            self._ID_OPEN: self._open,
            self._ID_COPY: self._copy,
            self._ID_LOGS: self._logs,
            self._ID_RESTART: self._restart,
            self._ID_STOP: self._stop,
            self._ID_SHOW: self._show,
        }.get(cmd_id, lambda: None)()

    # ---- COM apartment (see _run's call site for why) ----
    def _com_init(self):
        """Establish an STA COM apartment on this thread. Returns the ole32
        handle to pass to _com_uninit, or None if unavailable or the call
        failed. Best-effort: guarded so a failure here never blocks the tray
        from starting, matching every other call in this class."""
        import ctypes
        try:
            ole32 = ctypes.windll.ole32
            ole32.CoInitializeEx.restype = ctypes.c_long
            hr = ole32.CoInitializeEx(None, self._COINIT_APARTMENTTHREADED)
            if hr < 0:
                logger.debug("appface: CoInitializeEx returned 0x%08X on "
                             "the tray thread; continuing without it",
                             hr & 0xFFFFFFFF)
                return None
            return ole32
        except Exception:
            logger.debug("appface: CoInitializeEx failed on the tray thread",
                         exc_info=True)
            return None

    def _com_uninit(self, ole32) -> None:
        if ole32 is None:
            return
        try:
            ole32.CoUninitialize()
        except Exception:
            pass

    # ---- the Win32 plumbing (its own thread owns the window + message loop) ----
    def _run(self):
        import ctypes

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        kernel32 = ctypes.windll.kernel32
        # TrackPopupMenu's internal message pump can trigger outgoing COM
        # calls (UI Automation/accessibility notifications for the popup),
        # and a thread with no COM apartment established raises
        # RPC_E_CANTCALLOUT_ININPUTSYNCCALL when that happens. STA (not MTA)
        # because this thread runs a real GetMessage/DispatchMessage loop.
        ole32 = self._com_init()
        try:
            self._run_message_loop(user32, shell32, kernel32)
        finally:
            self._com_uninit(ole32)

    def _run_message_loop(self, user32, shell32, kernel32):
        from ctypes import wintypes
        import ctypes

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
                    self._show()   # double-click brings the dashboard window back
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
            # CreateWindowExW returned NULL - no tray this run. Not fatal (the
            # server runs fine without a tray icon), but log so it is not silent.
            logger.debug("appface: tray window not created (CreateWindowExW "
                         "returned NULL); no tray icon this run")
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
        if not self._ok:
            # The window exists but the shell rejected the notify-icon add; log so
            # a missing tray icon is diagnosable rather than a silent no-op.
            logger.debug("appface: Shell_NotifyIconW(NIM_ADD) failed; tray icon "
                         "not shown this run")
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
        user32.AppendMenuW(hmenu, MF_STRING, self._ID_SHOW, "Show LocaLM")
        user32.AppendMenuW(hmenu, MF_SEPARATOR, 0, None)
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
