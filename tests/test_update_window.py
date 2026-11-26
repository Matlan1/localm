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
from pathlib import Path

import pytest

from localm.update_mechanism import applier, window, window_tk
from tests import _update_mechanism as um
from tests._no_window import allow_window
from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401
from tests.test_update_applier import Update, _end, _free_port
from tests.test_update_gate import HIDDEN, ID, REPO


@pytest.fixture
def update(tmp_path):
    return Update(tmp_path / "inst")


def _tk_root():
    import tkinter as tk
    root = build_tk_root(tk.Tk)
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
        shown.say("Starting localm 2...")
        before = shown.lines()
        shown.end("Updated to localm 2.", close_after_s=0.2)
        assert shown.heading.cget("text") == "Updated to localm 2."
        assert shown.lines() == before, "the outcome is the heading, not another line"
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


def test_the_icon_of_the_gui_is_found_above_the_program_wherever_it_runs(tmp_path):
    """The icon is the GUI's, found in the nearest folder above the program that has it:
    the source tree, or the install whose update copy runs it."""
    icon = REPO / "localm" / "plugins" / "gui" / "static" / "icon-192.png"
    assert window_tk.icon_file(window.PROGRAM) == icon
    install = tmp_path / "inst"
    um.write(install, {"localm/plugins/gui/static/icon-192.png": ""})
    run = install / ".localm-update" / ID / "applier" / "update_mechanism" / "window_tk.py"
    assert window_tk.icon_file(run) == install / "localm/plugins/gui/static/icon-192.png"
    assert window_tk.icon_file(tmp_path / "elsewhere" / "window_tk.py") is None


def test_a_window_without_the_icon_says_so_in_the_log(tmp_path):
    """The real program, copied where no icon is above it: it shows Tk's icon and the
    update's log says why."""
    for name in ("window.py", "window_tk.py"):
        (tmp_path / name).write_bytes((window.PROGRAM.parent / name).read_bytes())
    stand_in = tmp_path / "withdrawn.py"
    stand_in.write_text(_WITHDRAWN.replace("PROGRAM", repr(str(tmp_path / "window_tk.py"))),
                        encoding="utf-8")
    said: list = []
    shown = window.Window.start(said.append, stand_in)
    if shown is None:
        pytest.skip(f"no window can be made here: {said}")
    shown.end("Done.", wait_s=window.END_S + 20)
    assert len(said) == 1 and said[0].startswith("The update window shows Tk's icon: no "
                                                 "localm/plugins/gui/static/icon-192.png "
                                                 "above "), said


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
        assert shown.cancel.master.cget("bg") == c[window_tk.EDGE[want]]
        assert window_tk.EDGE == {"dark": "text-dim", "light": "border"}
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


#: A window program that shows nothing: it writes what it was sent to the file ``LOG``
#: (outside the update's folder, which the update deletes) and answers the questions
#: with ``ANSWERS`` in order: ``hang`` answers nothing, ``exit`` ends the program.
_FAKE = """\
import json, sys
log = open(LOG, 'a', encoding='utf-8')
answers = iter(ANSWERS)
print('shown', flush=True)
for line in sys.stdin:
    log.write(line)
    log.flush()
    if 'ask' in json.loads(line):
        answer = next(answers, 'cancel')
        if answer == 'exit':
            sys.exit(4)
        if answer != 'hang':
            print(answer, flush=True)
"""


def _fake(path, log, answers) -> Path:
    """The program :data:`_FAKE` at *path*, logging to *log*, answering *answers*."""
    path.write_text(_FAKE.replace("LOG", repr(str(log))).replace("ANSWERS", repr(answers)),
                    encoding="utf-8")
    return path


def _sent(log) -> list:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_a_question_nobody_answers_is_cancelled_at_its_deadline(tmp_path):
    """A window that never answers (nobody at the desktop, a stalled window): the
    question is answered Cancel once its time is up, taken back in the window, and the
    log says so; the window still gets the outcome."""
    log = tmp_path / "window.log"
    said: list = []
    shown = window.Window.start(said.append, _fake(tmp_path / "w.py", log, ["hang"]))
    assert shown is not None, said
    begun = time.monotonic()
    assert shown.ask("x.dll is held by editor (pid 9).", limit_s=1.0) is False
    assert 1.0 <= time.monotonic() - begun < 10
    assert said == ["The update window asked: x.dll is held by editor (pid 9). "
                    "Nobody answered within 1 s: Cancel."]
    shown.end("The update to 2 was not started.")
    sent = _sent(log)
    assert sent[0] == {"ask": "x.dll is held by editor (pid 9).", "closing": "cancel"}
    assert sent[1] == {"unask": "Nobody answered within 1 s: Cancel."}
    assert sent[-1] == {"end": "The update to 2 was not started."}


def test_a_window_that_ends_while_it_asks_is_cancel_and_is_said(tmp_path):
    """The window program ends while it asks: Cancel at once, and the log says that the
    window is gone; later lines and questions go nowhere and wait for nothing."""
    log = tmp_path / "window.log"
    said: list = []
    shown = window.Window.start(said.append, _fake(tmp_path / "w.py", log, ["exit"]))
    assert shown is not None, said
    begun = time.monotonic()
    assert shown.ask("Still held.") is False
    assert time.monotonic() - begun < 10
    assert said == ["The update window closed while it asked (exit status 4); the update "
                    "goes on without it."]
    shown.say("Putting localm 1 back...")
    assert shown.ask("Again?") is False and shown.gone
    shown.end("Done.")
    assert len(said) == 1


def test_a_late_answer_does_not_answer_the_next_question(tmp_path):
    """An answer that comes after its question was taken back is not taken for the
    answer to the next one."""
    log = tmp_path / "window.log"
    shown = window.Window.start(lambda _t: None,
                                _fake(tmp_path / "w.py", log, ["again", "hang"]))
    assert shown is not None
    shown._out.put(window.AGAIN)   # as if it came in after its question timed out
    assert shown.ask("Held.", limit_s=1.0) is True
    shown._out.put(window.AGAIN)
    assert shown.ask("Held again.", limit_s=1.0) is False
    shown.end("Done.")


def test_the_put_back_question_cannot_be_answered_by_closing_the_window(tmp_path):
    """While the old version is put back, Cancel leaves a mix: closing the window is no
    answer then (``closing`` null), the window stays and asks on; any other question is
    answered Cancel by closing it."""
    log = tmp_path / "window.log"
    update = applier.Applier.__new__(applier.Applier)
    update.lines, update.window, update.ask_back = ["Replacing..."], None, None
    update.root, update.h = tmp_path, {"old": "1"}
    update.show_window(_fake(tmp_path / "w.py", log, ["cancel", "again"]))
    assert update.window is not None
    assert update._held("x.dll is held") is False
    assert update._held_back("x.dll is held") is True
    update.close_window("Done.")
    asks = [m for m in _sent(log) if "ask" in m]
    assert [m["closing"] for m in asks] == ["cancel", None], asks
    assert "Cancel leaves localm 1 not put back completely" in asks[1]["ask"]
    with allow_window():
        root = _tk_root()
        answers: list = []
        closed: list = []
        real = window_tk.UpdateWindow(root, answers.append, lambda: closed.append(True))
        real.ask("Held while putting back.", closing=None)
        real._close_pressed()
        assert answers == [] and real.asking and not closed
        real.cancel.invoke()
        assert answers == [window_tk.CANCEL]
        real.close()


def test_a_server_update_asks_in_its_window_about_a_process_in_the_way(update, tmp_path):
    """Server mode, whose console nobody sees: a process nobody agreed to end is named in
    the window with Try again and Cancel; Try again looks again, Cancel leaves the old
    version, and the window gets every line and the outcome."""
    run = update.root / ".localm-update" / ID / "applier" / "update_mechanism"
    log = tmp_path / "window.log"
    _fake(run / "window_tk.py", log, ["again", "cancel"])
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
        assert log.is_file(), err
        sent = _sent(log)
        asks = [m["ask"] for m in sent if "ask" in m]
        assert len(asks) == 2 and all("nobody agreed to end" in a for a in asks), sent
        # The outcome once, as the window's heading; the lines say what led to it.
        note = update.result()["note"]
        assert sent[-1] == {"end": note} and "was not started" in note
        assert note not in [m.get("say") for m in sent], sent
    finally:
        other.stdin.close()
        _end(other)


def test_the_tests_run_the_applier_without_a_window():
    """The applier copies of the tests carry the window program that shows nothing."""
    out = subprocess.run([sys.executable, "-I", "-S", "-c", um.WINDOWLESS],
                         capture_output=True, text=True, **HIDDEN)
    assert json.loads(out.stdout)["error"] and os.path.basename(window.PROGRAM) == "window_tk.py"
