# SPDX-License-Identifier: AGPL-3.0-or-later
"""The small update window (``localm.update_mechanism.window`` and ``window_tk``): it
shows the progress of an update a server hands over, asks "Try again" or "Cancel" about
what is in the way, closes itself at the end, and where no window can be shown the
update goes on without it and says so."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from localm.update_mechanism import applier, window, window_tk
from tests import _update_mechanism as um
from tests._no_window import allow_window
from tests.test_update_applier import Update, _end, _free_port
from tests.test_update_gate import HIDDEN, ID, REPO


@pytest.fixture
def update(tmp_path):
    return Update(tmp_path / "inst")


def _tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk window can be made here: {e}")
    root.withdraw()
    return root


def _pump(root, until, limit_s: float = 10.0) -> None:
    deadline = time.monotonic() + limit_s
    while not until() and time.monotonic() < deadline:
        try:
            root.update()
        except Exception:   # noqa: BLE001  (TclError once the window closed itself)
            return
        time.sleep(0.02)


def test_the_window_shows_the_progress_asks_and_closes_itself():
    """The widgets for real: lines of progress, a question whose buttons answer Try
    again and Cancel, closing the window while it asks is Cancel and is refused while the
    update runs, and the outcome closes the window by itself."""
    with allow_window():
        root = _tk_root()
        answers: list = []
        closed: list = []
        shown = window_tk.UpdateWindow(root, answers.append, lambda: closed.append(True))
        shown.say("Replacing the files of localm 1 by those of 2...")
        shown.say("Ending chat (pid 7)...")
        assert shown.lines() == ["Replacing the files of localm 1 by those of 2...",
                                 "Ending chat (pid 7)..."]
        shown._close_pressed()
        assert not closed, "the window closed while the update ran"
        shown.ask("x.dll is held by editor (pid 9). Close that program, then try again.")
        root.update()
        assert shown.buttons.winfo_manager() == "pack"
        assert "editor (pid 9)" in shown.question.cget("text")
        shown.again.invoke()
        assert answers == [window_tk.AGAIN]
        assert shown.buttons.winfo_manager() == ""
        shown.ask("Still held.")
        shown.cancel.invoke()
        shown.ask("Still held.")
        shown._close_pressed()
        assert answers == [window_tk.AGAIN, window_tk.CANCEL, window_tk.CANCEL]
        assert not closed
        shown.end("Updated to localm 2.", close_after_s=0.2)
        assert shown.heading.cget("text") == "Updated to localm 2."
        _pump(root, lambda: bool(closed))
        assert closed == [True]


def test_the_window_keeps_only_its_last_lines():
    with allow_window():
        root = _tk_root()
        shown = window_tk.UpdateWindow(root, lambda _a: None)
        for i in range(window_tk.LINES + 30):
            shown.say(f"line {i}")
        assert shown.lines()[0] == "line 30" and len(shown.lines()) == window_tk.LINES
        shown.close()


#: The real window program, run as the applier runs it, with every window it makes
#: withdrawn, so that nothing shows on the desktop.
_WITHDRAWN = """\
import importlib.util, sys, tkinter
_Tk = tkinter.Tk


class Withdrawn(_Tk):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.withdraw()
        self.deiconify = lambda: None


tkinter.Tk = Withdrawn
spec = importlib.util.spec_from_file_location("window_tk", PROGRAM)
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)
sys.exit(program.main())
"""


def test_the_window_program_opens_and_closes_itself_after_the_outcome(tmp_path):
    """The real program in its own process, as the applier starts it (withdrawn): it
    says it is shown, takes lines, and ends by itself once it showed the outcome."""
    stand_in = tmp_path / "withdrawn.py"
    stand_in.write_text(_WITHDRAWN.replace("PROGRAM", repr(str(window.PROGRAM))),
                        encoding="utf-8")
    said: list = []
    shown = window.Window.start(said.append, stand_in)
    if shown is None:
        pytest.skip(f"no window can be made here: {said}")
    shown.say("Starting localm 2...")
    begun = time.monotonic()
    shown.end("Updated to localm 2.")
    took = time.monotonic() - begun
    assert shown.proc.returncode == 0 and said == []
    assert window_tk.END_S - 1 <= took < window_tk.END_S + 10


@pytest.mark.parametrize("light, want", [(1, "light"), (0, "dark")])
def test_the_window_follows_the_theme_windows_gives_apps(light, want):
    """``AppsUseLightTheme`` 1 is light and 0 dark; a setting that cannot be read is
    light. The colours are the GUI's of that theme."""
    assert window_tk.theme("win32", windows_light=lambda: light) == want

    def unreadable() -> int:
        raise FileNotFoundError("no such value")
    assert window_tk.theme("win32", windows_light=unreadable) == "light"
    with allow_window():
        root = _tk_root()
        shown = window_tk.UpdateWindow(root, lambda _a: None, which=want)
        c = window_tk.THEMES[want]
        assert root.cget("bg") == c["bg"] and shown.text.cget("fg") == c["text-dim"]
        assert shown.heading.cget("fg") == c["text"]
        assert shown.again.cget("bg") == c["accent"] and shown.cancel.cget("fg") == c["text-dim"]
        shown.close()


@pytest.mark.parametrize("said, want", [
    ("Dark", "dark"), ("", "light"), ("'prefer-dark'", "dark"), ("'default'", "light"),
    ("Adwaita:dark", "dark")])
def test_the_window_follows_the_dark_preference_of_the_desktop(said, want):
    assert window_tk.theme("darwin", desktop=lambda: said) == want
    assert window_tk.theme("linux", desktop=lambda: said) == want

    def missing() -> str:
        raise FileNotFoundError("gsettings")
    assert window_tk.theme("linux", desktop=missing) == "light"


def _css_blocks() -> dict:
    """The CSS variables of the GUI's dark (``:root``) and light theme, and the rules
    the window takes its sizes from."""
    import re
    css = (REPO / "localm" / "plugins" / "gui" / "static" / "style.css").read_text(
        encoding="utf-8")

    def block(selector: str) -> str:
        found = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert found, selector
        return found.group(1)

    def props(text: str) -> dict:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return {k.strip(): v.strip() for k, v in re.findall(r"([-\w]+)\s*:\s*([^;]+);", text)}
    return {name: props(block(sel)) for name, sel in (
        ("dark", ":root"), ("light", '[data-theme="light"]'), ("logo", "#logo"),
        ("body", "html, body"), ("primary", ".btn-primary"), ("secondary", ".btn-secondary"))}


def test_the_window_uses_the_design_of_the_gui():
    """The colours, the type family, the sizes and the button padding of the window are
    the GUI's own (``style.css``), for both themes."""
    css = _css_blocks()
    for which in ("dark", "light"):
        for name, value in window_tk.THEMES[which].items():
            assert css[which][f"--{name}"].lower() == value, (which, name)
    families = [f.strip().strip('"') for f in css["dark"]["--sans"].split(",")]
    assert list(window_tk.SANS) == families[:len(window_tk.SANS)]
    assert css["logo"]["font-size"] == f"{window_tk.SIZES['logo']}px"
    assert css["logo"]["font-weight"] == "600"
    for name in ("body", "primary", "secondary"):
        assert float(css[name]["font-size"].removesuffix("px")) == window_tk.SIZES[name]
    for name in ("primary", "secondary"):
        assert css[name]["padding"] == "{}px {}px".format(*window_tk.PADDING[name])
    assert css["primary"]["color"] == window_tk.ON_ACCENT
    assert css["primary"]["background"] == "var(--accent)"


def test_without_a_window_the_update_is_told_why_and_goes_on(monkeypatch, tmp_path):
    """Tk that cannot start (here a Tcl library that is not there, as on a desktop
    without Tk) is said with its reason, and nothing waits for the window."""
    monkeypatch.setenv("TCL_LIBRARY", str(tmp_path / "no-tcl"))
    said: list = []
    begun = time.monotonic()
    assert window.Window.start(said.append) is None
    assert time.monotonic() - begun < window.SHOW_S
    assert len(said) == 1 and said[0].startswith("The update window cannot be shown (")
    assert said[0].endswith("; the update goes on without it.")
    assert "TclError" in said[0]


#: A window program that shows nothing: it writes what it was sent to the file
#: ``window.log`` beside it and answers the questions with ``ANSWERS`` in order.
_FAKE = """\
import json, os, sys
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'window.log'), 'a',
           encoding='utf-8')
answers = iter(ANSWERS)
print('shown', flush=True)
for line in sys.stdin:
    log.write(line)
    log.flush()
    if 'ask' in json.loads(line):
        print(next(answers, 'cancel'), flush=True)
"""


def test_a_server_update_asks_in_its_window_about_a_process_in_the_way(update):
    """Server mode, whose console nobody sees: a process nobody agreed to end is named in
    the window with Try again and Cancel; Try again looks again, Cancel leaves the old
    version, and the window gets every line and the outcome."""
    run = update.root / ".localm-update" / ID / "applier" / "update_mechanism"
    (run / "window_tk.py").write_text(_FAKE.replace("ANSWERS", repr(["again", "cancel"])),
                                      encoding="utf-8")
    other = subprocess.Popen([sys.executable, "-m", "localm", "stay"], cwd=str(update.root),
                             env=update.env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, **HIDDEN)
    assert other.stdout is not None and other.stdin is not None
    assert other.stdout.readline().startswith("STARTED ")
    try:
        proc = update.apply(mode=applier.SERVER,
                            watch={"host": "127.0.0.1", "port": _free_port(), "scheme": "http"})
        waiting = update.waiter()
        (_, err), _ = _end(proc, waiting)
        assert proc.returncode == 1, err
        assert update.is_old()
        sent = [json.loads(line) for line in
                (run / "window.log").read_text(encoding="utf-8").splitlines()]
        asks = [m["ask"] for m in sent if "ask" in m]
        assert len(asks) == 2 and all("nobody agreed to end" in a for a in asks), sent
        assert sent[-1] == {"end": update.result()["note"]}
        assert any("was not started" in m.get("say", "") for m in sent)
    finally:
        other.stdin.close()
        _end(other)


def test_the_tests_run_the_applier_without_a_window():
    """The applier copies of the tests carry the window program that shows nothing."""
    out = subprocess.run([sys.executable, "-I", "-S", "-c", um.WINDOWLESS],
                         capture_output=True, text=True, **HIDDEN)
    assert json.loads(out.stdout)["error"] and os.path.basename(window.PROGRAM) == "window_tk.py"
