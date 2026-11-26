# SPDX-License-Identifier: AGPL-3.0-or-later
"""The small window an update shows while it runs for a server, whose console nobody
sees: the progress of the update and, when a file is held or a process does not end,
the question "Try again" or "Cancel". A program, run by :class:`.window.Window` as
``<python> -I -S window_tk.py``; standard library only.

Each line on its standard input is one JSON object: ``{"say": text}`` adds a line of
progress, ``{"ask": text}`` shows *text* with the buttons Try again and Cancel and prints
``again`` or ``cancel`` on its standard output, ``{"end": text}`` shows the outcome and
closes the window after :data:`END_S` seconds. The end of its standard input before the
outcome closes it at once. The window cannot be closed while the update runs; closing it
while it asks answers Cancel. Its first line on standard output is ``shown``, or
``{"error": reason}`` when no window can be shown here.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable


def _words():
    """``window.py`` beside this file, loaded by its path: the words both sides use."""
    spec = importlib.util.spec_from_file_location(
        "_localm_update_window", Path(__file__).with_name("window.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_w = _words()
END_S, AGAIN, CANCEL, SHOWN = _w.END_S, _w.AGAIN, _w.CANCEL, _w.SHOWN
#: The lines of progress the window keeps.
LINES = 200
TITLE = "LocaLM update"

#: The design tokens of the GUI (``localm/plugins/gui/static/style.css``: its CSS
#: variables of the dark ``:root`` and the ``[data-theme="light"]`` theme, the sizes of
#: ``#logo``, ``html, body``, ``.btn-primary`` and ``.btn-secondary``) that the window
#: uses; a test keeps them equal to that file.
THEMES = {
    "dark": {"bg": "#0e1014", "bg-raised": "#171c26", "border": "#2c3341",
             "text": "#dce2ec", "text-dim": "#98a2b4", "accent": "#5aa2fb",
             "yellow": "#d9a741"},
    "light": {"bg": "#f5f6f8", "bg-raised": "#ffffff", "border": "#d8dce4",
              "text": "#23272f", "text-dim": "#6b7280", "accent": "#2563eb",
              "yellow": "#a16207"},
}
#: ``--sans`` of the GUI, the families Tk can be given.
SANS = ("Inter", "Segoe UI")
#: Pixel sizes: ``#logo`` (weight 600), ``body``, ``.btn-primary``, ``.btn-secondary``.
SIZES = {"logo": 18, "body": 14.5, "primary": 14, "secondary": 13}
#: ``padding`` (vertical, horizontal) of ``.btn-primary`` and ``.btn-secondary``.
PADDING = {"primary": (10, 22), "secondary": (9, 16)}
#: The text of ``.btn-primary``.
ON_ACCENT = "#fff"
#: The key of Windows' own setting for the theme of apps.
_PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
_HIDDEN = 0x08000000  # CREATE_NO_WINDOW


# ------------------------------------------------------------------ the theme

def _windows_light() -> int:
    """``AppsUseLightTheme`` of the user (1 light, 0 dark), read only. Raises OSError."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE, 0, winreg.KEY_READ) as key:
        return int(winreg.QueryValueEx(key, "AppsUseLightTheme")[0])


def _command(argv: list) -> str:
    """What *argv* prints; empty when it cannot run. Raises OSError."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=5,
                          stdin=subprocess.DEVNULL,
                          creationflags=_HIDDEN if sys.platform == "win32" else 0)
    return done.stdout.strip()


def _desktop_scheme() -> str:
    """The dark preference of the desktop on macOS (``AppleInterfaceStyle``) and on
    Linux (``GTK_THEME``, then GNOME's ``color-scheme``). Raises OSError."""
    import os
    if sys.platform == "darwin":
        return _command(["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"])
    gtk = os.environ.get("GTK_THEME", "")
    if gtk:
        return gtk
    return _command(["/usr/bin/gsettings", "get", "org.gnome.desktop.interface",
                     "color-scheme"])


def theme(platform: str = sys.platform, windows_light: Callable[[], int] = _windows_light,
          desktop: Callable[[], str] = _desktop_scheme) -> str:
    """``dark`` or ``light``: the theme the user chose for apps (Windows) or the desktop's
    dark preference (macOS, Linux); ``light`` where none can be read."""
    try:
        if platform == "win32":
            return "dark" if windows_light() == 0 else "light"
        return "dark" if "dark" in desktop().lower() else "light"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "light"


def _family(tk_root) -> str:
    """The first family of :data:`SANS` this system has, else Tk's own."""
    from tkinter import font
    have = set(font.families(tk_root))
    return next((f for f in SANS if f in have), "TkDefaultFont")


def _px(size: float) -> int:
    """A pixel size for Tk, which takes pixels as a negative size."""
    return -round(size)


def _dark_title_bar(tk_root) -> None:
    """Ask Windows for the dark title bar (``DWMWA_USE_IMMERSIVE_DARK_MODE``). Only the
    look of the bar depends on it: where Windows refuses, the bar stays light."""
    if sys.platform != "win32":
        return
    import ctypes
    tk_root.update_idletasks()
    hwnd = ctypes.windll.user32.GetParent(tk_root.winfo_id())
    on = ctypes.c_int(1)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))


# ------------------------------------------------------------------ the window

class UpdateWindow:
    """The widgets of the window in the Tk root *tk_root*, in the GUI's colours of
    *which* theme (:func:`theme` when None). *answer* gets ``again`` or ``cancel`` for
    each question; *closed* is called once the window is gone."""

    def __init__(self, tk_root, answer: Callable[[str], None],
                 closed: Callable[[], None] = lambda: None, which=None) -> None:
        import tkinter as tk
        self.root, self._answer, self._closed = tk_root, answer, closed
        self.asking = False
        self.ended = False
        self.theme = which or theme()
        c = self.colours = THEMES[self.theme]
        family = _family(tk_root)
        tk_root.title(TITLE)
        tk_root.configure(bg=c["bg"])
        tk_root.minsize(560, 320)
        tk_root.protocol("WM_DELETE_WINDOW", self._close_pressed)
        if self.theme == "dark":
            try:
                _dark_title_bar(tk_root)
            except (OSError, AttributeError):   # an older Windows without the attribute
                pass
        bar = tk.Frame(tk_root, bg=c["bg-raised"], highlightthickness=0)
        bar.pack(fill="x")
        logo = tk.Frame(bar, bg=c["bg-raised"])
        logo.pack(side="left", padx=16, pady=(18, 14))
        weight = (family, _px(SIZES["logo"]), "bold")
        tk.Label(logo, text="LocaL", font=weight, fg=c["text"], bg=c["bg-raised"],
                 bd=0, padx=0).pack(side="left")
        tk.Label(logo, text="M", font=weight, fg=c["accent"], bg=c["bg-raised"],
                 bd=0, padx=0).pack(side="left")
        tk.Frame(tk_root, height=1, bg=c["border"]).pack(fill="x")
        body = (family, _px(SIZES["body"]))
        self.heading = tk.Label(tk_root, text="LocaLM is being updated.", anchor="w",
                                justify="left", font=(family, _px(SIZES["body"]), "bold"),
                                fg=c["text"], bg=c["bg"], wraplength=540)
        self.heading.pack(fill="x", padx=16, pady=(14, 6))
        self.text = tk.Text(tk_root, width=72, height=12, wrap="word", state="disabled",
                            relief="flat", borderwidth=0, highlightthickness=0,
                            font=(family, _px(SIZES["secondary"])), fg=c["text-dim"],
                            bg=c["bg"], padx=0, pady=0)
        self.text.pack(fill="both", expand=True, padx=16)
        self.question = tk.Label(tk_root, text="", anchor="w", justify="left",
                                 wraplength=540, font=body, fg=c["yellow"], bg=c["bg"])
        self.buttons = tk.Frame(tk_root, bg=c["bg"])
        (py, px), (sy, sx) = PADDING["primary"], PADDING["secondary"]
        self.again = tk.Button(
            self.buttons, text="Try again", command=lambda: self._pressed(AGAIN),
            font=(family, _px(SIZES["primary"])), fg=ON_ACCENT, bg=c["accent"],
            activeforeground=ON_ACCENT, activebackground=c["accent"], relief="flat",
            bd=0, highlightthickness=0, padx=px, pady=py, cursor="hand2")
        # The 1 px border of .btn-secondary: the button inside a frame of that colour.
        edge = tk.Frame(self.buttons, bg=c["border"])
        self.cancel = tk.Button(
            edge, text="Cancel", command=lambda: self._pressed(CANCEL),
            font=(family, _px(SIZES["secondary"])), fg=c["text-dim"], bg=c["bg"],
            activeforeground=c["text"], activebackground=c["bg"], relief="flat", bd=0,
            highlightthickness=0, padx=sx, pady=sy, cursor="hand2")
        self.cancel.pack(padx=1, pady=1)
        edge.pack(side="right", padx=(8, 0))
        self.again.pack(side="right")

    def lines(self) -> list:
        """The lines of progress the window shows."""
        return [line for line in self.text.get("1.0", "end").split("\n") if line]

    def say(self, text: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", text.rstrip("\n") + "\n")
        extra = int(self.text.index("end-1c").split(".")[0]) - 1 - LINES
        if extra > 0:
            self.text.delete("1.0", f"{extra + 1}.0")
        self.text.configure(state="disabled")
        self.text.see("end")

    def ask(self, text: str) -> None:
        """Show *text* with the buttons; the answer goes to *answer*. A minimized window
        comes back for it."""
        self.asking = True
        self.question.configure(text=text)
        self.question.pack(fill="x", padx=16, pady=(10, 8))
        self.buttons.pack(fill="x", padx=16, pady=(0, 16))
        if self.root.state() == "iconic":
            self.root.deiconify()
        self.root.lift()

    def _pressed(self, which: str) -> None:
        if not self.asking:
            return
        self.asking = False
        self.question.pack_forget()
        self.buttons.pack_forget()
        self._answer(which)

    def end(self, text: str, close_after_s: float = END_S) -> None:
        """Show the outcome *text* and close the window after *close_after_s* seconds."""
        if self.asking:
            self._pressed(CANCEL)
        self.ended = True
        self.heading.configure(text=text)
        self.say(text)
        self.root.after(int(close_after_s * 1000), self.close)

    def _close_pressed(self) -> None:
        if self.asking:
            self._pressed(CANCEL)
        elif self.ended:
            self.close()

    def close(self) -> None:
        import tkinter as tk
        try:
            self.root.destroy()
        except tk.TclError:   # it was destroyed already
            pass
        self._closed()


def _icon(tk_root) -> None:
    """The icon of the GUI (``icon-192.png``) on the window, from the install this program
    updates (``<root>/.localm-update/<id>/applier/update_mechanism``); Tk's own icon stays
    where that file cannot be read."""
    import tkinter as tk
    try:
        root = Path(__file__).resolve().parents[4]
        image = tk.PhotoImage(master=tk_root, file=str(
            root / "localm" / "plugins" / "gui" / "static" / "icon-192.png"))
    except (IndexError, tk.TclError):
        return
    tk_root.iconphoto(True, image)
    tk_root._localm_icon = image   # Tk drops an image nothing in Python holds


def main() -> int:
    """The program: see the module."""
    def out(text: str) -> None:
        try:
            print(text, flush=True)
        except (OSError, ValueError):
            pass

    try:
        import tkinter as tk
        tk_root = tk.Tk()
    except Exception as e:   # noqa: BLE001  (no tkinter, no display, a broken Tcl)
        out(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    messages: queue.SimpleQueue = queue.SimpleQueue()

    def read() -> None:
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except ValueError:
                continue
        messages.put(None)

    window = UpdateWindow(tk_root, out)
    _icon(tk_root)
    threading.Thread(target=read, daemon=True).start()
    out(SHOWN)

    def poll() -> None:
        while True:
            try:
                message = messages.get_nowait()
            except queue.Empty:
                break
            if message is None:
                # After the outcome the window closes itself (UpdateWindow.end).
                if not window.ended:
                    window.close()
                return
            if not isinstance(message, dict):
                continue
            if isinstance(message.get("say"), str):
                window.say(message["say"])
            elif isinstance(message.get("ask"), str):
                window.ask(message["ask"])
            elif isinstance(message.get("end"), str):
                window.end(message["end"])
        tk_root.after(100, poll)

    tk_root.after(100, poll)
    tk_root.mainloop()
    return 0



if __name__ == "__main__":
    sys.exit(main())
