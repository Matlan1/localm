# SPDX-License-Identifier: AGPL-3.0-or-later
"""A PATH conflict must be REAL, and the user decides what happens about it.

At the "Make 'localm' runnable from any terminal?" step, setup could report:

    [!] A 'localm' command already exists at .\\localm.BAT. This clone's command
        was still added (lower precedence); the existing one takes priority
        until you reorder PATH.
    'localm' is available (its directory was already on PATH). ...

Two faults, and the second is worse than the first.

FALSE POSITIVE. There is no other localm. ``shutil.which`` on Windows searches
the CURRENT DIRECTORY before PATH, setup runs with the cwd set to the clone, and
the clone SHIPS its own localm.bat - so it finds OUR OWN file: with the cwd
inside a clone, ``shutil.which("localm")`` returns ``.\\localm.BAT``. Passing
``path=`` does NOT suppress that search (the current-directory insert happens
whether or not a search path was supplied), so the fix has to filter the RESULT.

DECIDED FOR THE USER. On the back of that phantom it demoted this install to
lower precedence and reported it as done - while the remedy it named (reorder
PATH) is exactly what the user had just asked it to do. Nobody was asked.
Reordering PATH has real consequences for whatever else is installed, so it is a
question; the DEFAULT answer stays "change nothing that already works".
"""
from __future__ import annotations

import os
import sys

import pytest

from localm import globalcmd as gc


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """A clone that ships its own localm launcher, with its bin dir on PATH."""
    root = tmp_path / "clone"
    (root / "bin").mkdir(parents=True)
    (root / "localm.bat").write_text("@echo off", encoding="utf-8")
    monkeypatch.setenv("PATH", str(root / "bin"))
    return root


# ------------------------- the false positive ----------------------------- #

def test_our_own_shipped_launcher_is_not_a_conflict(clone, monkeypatch):
    """The exact reported case: a cwd-relative hit on the clone's own file."""
    monkeypatch.chdir(clone)
    monkeypatch.setattr(gc.shutil, "which",
                        lambda name: os.path.join(".", "localm.BAT"))
    assert gc.existing_localm(gc.shim_path(clone), clone) is None


def test_anything_inside_the_clone_is_not_a_conflict(clone, monkeypatch):
    """Absolute form of the same thing - our own files are never a rival."""
    monkeypatch.setattr(gc.shutil, "which", lambda name: str(clone / "localm.bat"))
    assert gc.existing_localm(gc.shim_path(clone), clone) is None


def test_a_hit_that_is_not_on_path_is_not_a_conflict(clone, tmp_path, monkeypatch):
    """A conflict means reachable FROM PATH. A current-directory hit is not."""
    stray = tmp_path / "somewhere_else"
    stray.mkdir()
    (stray / "localm").write_text("x", encoding="utf-8")
    monkeypatch.setattr(gc.shutil, "which", lambda name: str(stray / "localm"))
    assert gc.existing_localm(gc.shim_path(clone), clone) is None


def test_a_real_path_conflict_is_still_reported(clone, tmp_path, monkeypatch):
    """The control: the fix must not blind the check to a genuine rival."""
    other = tmp_path / "otherbin"
    other.mkdir()
    (other / "localm").write_text("x", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join([str(other), str(clone / "bin")]))
    monkeypatch.setattr(gc.shutil, "which", lambda name: str(other / "localm"))
    assert gc.existing_localm(gc.shim_path(clone), clone) == str(other / "localm")


def test_detection_stays_lexical_and_touches_no_real_directory(monkeypatch):
    """resolve() hits the filesystem, so the PATH comparison must not use it.

    The two fixture entries must be valid PATH entries on the platform under
    test - a drive-letter path joined with POSIX's colon separator (os.pathsep)
    is unparseable on POSIX (the drive letter's own colon is indistinguishable
    from the separator), which would split one entry into two and fail the
    len() assertion for a reason unrelated to what this test checks."""
    a, b = (["C:/Windows/System32", "D:/tools/bin"] if os.name == "nt"
            else ["/usr/local/bin", "/usr/bin"])
    monkeypatch.setenv("PATH", os.pathsep.join([a, b]))
    dirs = gc.path_dirs()
    assert len(dirs) == 2
    assert all(isinstance(d, str) for d in dirs), \
        "path_dirs must return normalised STRINGS, never resolved Paths"


# ------------------------------ the prompt -------------------------------- #

def test_default_answer_keeps_the_existing_command(clone, capsys):
    """Pressing Enter must change nothing about a command that already works."""
    assert gc.ask_conflict("/usr/bin/localm", clone, ask=lambda p: "") == "keep"
    assert "already exists" in capsys.readouterr().out


@pytest.mark.parametrize("answer,expected", [
    ("1", "priority"), ("2", "keep"), ("3", "skip"), ("nonsense", "keep"),
])
def test_the_user_decides(clone, answer, expected):
    assert gc.ask_conflict("/usr/bin/localm", clone, ask=lambda p: answer) == expected


def test_the_prompt_offers_all_three_and_names_this_install(clone, capsys):
    gc.ask_conflict("/usr/bin/localm", clone, ask=lambda p: "")
    out = capsys.readouterr().out
    assert "[1] This install" in out and str(clone.resolve()) in out
    assert "[2] Keep the existing one" in out
    assert "[3] Neither" in out
    assert "until you reorder PATH" not in out, \
        "the old text told the user to do the very thing it had just refused to do"


def test_non_interactive_keeps_the_existing_command_and_says_so(clone, capsys):
    """No tty: never hang on a prompt, and never silently pick 'priority'."""
    assert gc.ask_conflict("/usr/bin/localm", clone, ask=None) == "keep"
    assert "not an interactive terminal" in capsys.readouterr().out


def test_eof_falls_back_to_keeping_the_existing_command(clone):
    def _eof(_prompt):
        raise EOFError
    assert gc.ask_conflict("/usr/bin/localm", clone, ask=_eof) == "keep"


# ------------------------- main() acts on the answer ---------------------- #

def _capture_install(monkeypatch, seen):
    def _fake(root, precedence="append"):
        seen.append(precedence)
        return {"path_dir": "d", "shim": "s", "path_modified": True,
                "conflict": "/usr/bin/localm", "precedence": precedence}
    monkeypatch.setattr(gc, "install", _fake)
    monkeypatch.setattr(gc, "existing_localm",
                        lambda shim, root=None: "/usr/bin/localm")


def test_skip_installs_nothing_at_all(monkeypatch, capsys):
    seen = []
    _capture_install(monkeypatch, seen)
    monkeypatch.setattr(gc, "ask_conflict", lambda *a, **k: "skip")
    assert gc.main(["install", "--root", "."]) == 30
    assert seen == [], "install() must not run when the user declined"
    assert "untouched" in capsys.readouterr().out


def test_priority_prepends_and_keep_appends(monkeypatch):
    for choice, expected in (("priority", "prepend"), ("keep", "append")):
        seen = []
        _capture_install(monkeypatch, seen)
        monkeypatch.setattr(gc, "ask_conflict", lambda *a, **k: choice)
        gc.main(["install", "--root", "."])
        assert seen == [expected], f"{choice} should install with {expected}"


def test_keeping_the_existing_one_is_reported_as_the_choice_it_was(monkeypatch, capsys):
    seen = []
    _capture_install(monkeypatch, seen)
    monkeypatch.setattr(gc, "ask_conflict", lambda *a, **k: "keep")
    gc.main(["install", "--root", "."])
    out = capsys.readouterr().out
    assert "as you chose" in out
    assert "lower precedence" not in out, \
        "a decision the user made is not a warning to hand back to them"


def test_yes_flag_never_prompts(monkeypatch):
    seen = []
    _capture_install(monkeypatch, seen)

    def _guard(conflict, root, assume_yes=False, ask=None):
        if not assume_yes:
            pytest.fail("prompted despite --yes")
        return "keep"

    monkeypatch.setattr(gc, "ask_conflict", _guard)
    assert gc.main(["install", "--root", ".", "--yes"]) == 0
    assert seen == ["append"]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the machine PATH is a Windows concept")
def test_a_system_path_rival_cannot_be_outranked(monkeypatch):
    """We only ever write the USER PATH, and Windows searches the machine PATH
    first - so we must not promise a priority we cannot deliver."""
    monkeypatch.setattr(gc, "_win_read_system_path", lambda: "C:/Program Files/other")
    assert gc.conflict_outranks_user_path("C:/Program Files/other/localm.exe") is True
    assert gc.conflict_outranks_user_path("C:/Users/me/bin/localm.exe") is False
