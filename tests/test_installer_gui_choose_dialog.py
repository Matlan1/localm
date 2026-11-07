# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installer window's modal choice dialog (Wizard._choose_dialog).

The dialog is built on a real Tk root and a real Toplevel, both kept withdrawn by
tests._no_window. ``grab_set`` needs a viewable window, so it is replaced by a
recorder; ``wait_window`` is the real one and returns when the dialog is
destroyed, which a callback scheduled on the root does the way a click would.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_GUI_PATH = Path(__file__).resolve().parents[1] / "installer" / "gui.py"
_MOD_NAME = "localm_installer_gui_choose_dialog"
_CHOICES = [["a", "First"], ["b", "Second"], ["c", "Third"]]


def _load():
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _GUI_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def wizard(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    gui = _load()
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    grabbed = []
    monkeypatch.setattr(tk.Toplevel, "grab_set", lambda self: grabbed.append(self))
    root = build_tk_root(tk.Tk)
    root.withdraw()
    try:
        w = gui.Wizard(root, tk, ttk, filedialog, messagebox=None)
        w.grabbed = grabbed
        yield w
    finally:
        root.destroy()
        sys.modules.pop(_MOD_NAME, None)


def _dialog(root):
    tk = pytest.importorskip("tkinter")
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(tops) == 1
    return tops[0]


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


def _button(top, text):
    found = [w for w in _descendants(top) if str(w.cget("text")) == text]
    assert len(found) == 1, text
    return found[0]


def _close(top):
    top.tk.call(top.protocol("WM_DELETE_WINDOW"))


def test_closing_the_dialog_returns_the_default(wizard):
    wizard.root.after(50, lambda: _close(_dialog(wizard.root)))
    assert wizard._choose_dialog("Pick one", _CHOICES, "b") == "b"
    assert len(wizard.grabbed) == 1


def test_ok_returns_the_picked_answer(wizard):
    def pick():
        top = _dialog(wizard.root)
        _button(top, "Third").invoke()
        _button(top, "OK").invoke()

    wizard.root.after(50, pick)
    assert wizard._choose_dialog("Pick one", _CHOICES, "a") == "c"


def test_ok_without_a_pick_returns_the_default(wizard):
    wizard.root.after(50, lambda: _button(_dialog(wizard.root), "OK").invoke())
    assert wizard._choose_dialog("Pick one", _CHOICES, "b") == "b"


def test_closing_after_a_pick_still_returns_the_default(wizard):
    def pick_then_close():
        top = _dialog(wizard.root)
        _button(top, "Third").invoke()
        _close(top)

    wizard.root.after(50, pick_then_close)
    assert wizard._choose_dialog("Pick one", _CHOICES, "a") == "a"


def test_the_dialog_offers_every_choice_and_the_text(wizard):
    seen = {}

    def look():
        top = _dialog(wizard.root)
        seen["labels"] = sorted(str(w.cget("text")) for w in _descendants(top))
        top.destroy()

    wizard.root.after(50, look)
    wizard._choose_dialog("Pick one", _CHOICES, "a")
    assert seen["labels"] == sorted(["Pick one", "First", "Second", "Third", "OK"])
