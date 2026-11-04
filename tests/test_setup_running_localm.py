# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup stops, with consent, the LocaLM programs running from the install folder.

Real spawned processes whose executable lies inside a folder are listed by
setup.bat's ``:find_running`` and by the window's first step; a program from
another folder is never listed or touched; declining leaves everything running
and ends the run with nothing changed."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"
_GUI_PATH = ROOT / "installer" / "gui.py"

_PYVENV_CFG = (
    f"home = {sys.base_prefix}\n"
    "implementation = CPython\n"
    "include-system-site-packages = false\n"
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe/WMI only")


def _spawn_from(folder: Path) -> subprocess.Popen:
    """A real long-lived process whose executable is folder/.venv/Scripts/python.exe."""
    venv = folder / ".venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    exe = scripts / "python.exe"
    shutil.copy(sys.executable, exe)
    (venv / "pyvenv.cfg").write_text(_PYVENV_CFG, encoding="utf-8")
    proc = subprocess.Popen([str(exe), "-c", "import time; time.sleep(120)"],
                            creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.time() + 10
    while time.time() < deadline and proc.poll() is not None:
        time.sleep(0.05)
    assert proc.poll() is None
    return proc


def _find_running_block() -> str:
    text = BAT.read_text(encoding="utf-8", errors="replace")
    start = text.index("\n:find_running\n")
    end = text.index("\nexit /b 0", start) + len("\nexit /b 0")
    return text[start:end]


def _probe(folder: Path, arg: str = "") -> subprocess.CompletedProcess:
    probe = folder / "probe.bat"
    probe.write_text(
        "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
        f'cd /d "{folder}"\r\nset "LOCALM_SETUP_ROOT=%CD%"\r\n'
        f"call :find_running {arg}\r\n"
        "echo RUNPIDS=[!RUNPIDS!]\r\nexit /b 0\r\n"
        + _find_running_block().replace("\n", "\r\n") + "\r\n",
        encoding="utf-8")
    return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                          cwd=str(folder), timeout=60)


@pytest.fixture
def two_folders(tmp_path):
    here, other = tmp_path / "here", tmp_path / "other"
    here.mkdir()
    other.mkdir()
    procs = []
    yield here, other, procs
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)


def test_setup_bat_lists_this_folders_program_and_not_another_folders(two_folders):
    here, other, procs = two_folders
    mine, theirs = _spawn_from(here), _spawn_from(other)
    procs += [mine, theirs]
    quiet = _probe(here)
    assert f"RUNPIDS=[ {mine.pid}]" in quiet.stdout, (quiet.stdout, quiet.stderr)
    shown = _probe(here, "show")
    assert f"python.exe (PID {mine.pid})" in shown.stdout, (shown.stdout, shown.stderr)
    assert str(theirs.pid) not in shown.stdout


def test_setup_bat_finds_nothing_when_nothing_runs_from_the_folder(two_folders):
    here, _, _ = two_folders
    out = _probe(here)
    assert "RUNPIDS=[]" in out.stdout, (out.stdout, out.stderr)


def test_setup_bat_checks_before_it_changes_anything_and_asks_before_stopping():
    text = BAT.read_text(encoding="utf-8", errors="replace")
    assert text.index("call :find_running") < text.index('set "STMP=.localm-setup-tmp"')
    assert text.index("SETUP-Q: running-localm") < text.index("taskkill /PID %%p /T /F")


@pytest.fixture
def gui(tmp_path, monkeypatch):
    name = "localm_gui_running_probe"
    spec = importlib.util.spec_from_file_location(name, _GUI_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path / "here")
    return mod


def _answer(reply, asked):
    def ask(question):
        asked.append(question)
        return reply
    return ask


def test_the_window_stops_this_folders_program_when_asked_and_spares_the_other(
        gui, two_folders, monkeypatch):
    here, other, procs = two_folders
    mine, theirs = _spawn_from(here), _spawn_from(other)
    procs += [mine, theirs]
    asked, log = [], []
    monkeypatch.setattr(gui, "ASK", _answer("y", asked))
    gui.running_programs_step().run(log.append)
    assert [q["id"] for q in asked] == ["running-localm"]
    assert any(str(here) in line for line in log)
    mine.wait(timeout=20)
    assert theirs.poll() is None


def test_declining_leaves_the_program_running_and_ends_the_run(gui, two_folders, monkeypatch):
    here, _, procs = two_folders
    mine = _spawn_from(here)
    procs.append(mine)
    asked = []
    monkeypatch.setattr(gui, "ASK", _answer("n", asked))
    with pytest.raises(gui.StepFailed, match="nothing was changed"):
        gui.running_programs_step().run(lambda s: None)
    assert mine.poll() is None


def test_nothing_running_asks_nothing(gui, two_folders, monkeypatch):
    asked = []
    monkeypatch.setattr(gui, "ASK", _answer("y", asked))
    gui.running_programs_step().run(lambda s: None)
    assert asked == []


def test_the_running_check_is_the_first_step(gui):
    labels = [s.label for s in gui.all_steps(gui.Plan())]
    assert labels[0].startswith("Checking for running")
