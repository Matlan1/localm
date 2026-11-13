# SPDX-License-Identifier: AGPL-3.0-or-later
"""The global command step says "as you chose" only for a choice the person
made, takes a choice the graphical installer asked for, and reports the other
localm on PATH; the uninstall plan names what Astral's uv installer added for a
shared uv; a home folder that cannot be found is named as the problem."""

from __future__ import annotations

from pathlib import Path

import pytest

from localm import globalcmd as gc
from localm import install_manifest as im


def _fake_install(monkeypatch, seen):
    def _install(root, precedence="append"):
        seen.append(precedence)
        return {"path_dir": "d", "shim": "s", "path_modified": False,
                "conflict": "/usr/bin/localm", "precedence": precedence,
                "path_note": None, "shim_created": True}
    monkeypatch.setattr(gc, "install", _install)
    monkeypatch.setattr(gc, "existing_localm", lambda shim, root=None: "/usr/bin/localm")


def test_a_given_choice_is_taken_without_asking(capsys):
    def never(prompt):
        pytest.fail("asked although the caller gave the answer")
    report = {}
    assert gc.ask_conflict("/usr/bin/localm", ".", ask=never, choice="priority",
                           report=report) == "priority"
    assert report == {"asked": True}


def test_the_default_without_a_terminal_is_not_reported_as_asked():
    report = {}
    assert gc.ask_conflict("/usr/bin/localm", ".", ask=None, report=report) == "keep"
    assert report == {"asked": False}


def test_an_answer_at_the_prompt_is_reported_as_asked():
    report = {}
    assert gc.ask_conflict("/usr/bin/localm", ".", ask=lambda p: "2",
                           report=report) == "keep"
    assert report == {"asked": True}


def test_yes_keeps_the_other_command_and_says_nothing_was_asked(monkeypatch, capsys):
    seen = []
    _fake_install(monkeypatch, seen)
    assert gc.main(["install", "--root", ".", "--yes"]) == 20
    out = capsys.readouterr().out
    assert seen == ["append"]
    assert "since nothing was asked here" in out
    assert "as you chose" not in out


@pytest.mark.parametrize("choice, precedence", [("priority", "prepend"), ("keep", "append")])
def test_a_choice_from_the_window_is_applied_and_reported(monkeypatch, capsys, choice,
                                                          precedence):
    seen = []
    _fake_install(monkeypatch, seen)
    gc.main(["install", "--root", ".", "--choice", choice])
    out = capsys.readouterr().out
    assert seen == [precedence]
    if choice == "keep":
        assert "as you chose" in out


def test_skip_from_the_window_installs_nothing(monkeypatch):
    seen = []
    _fake_install(monkeypatch, seen)
    assert gc.main(["install", "--root", ".", "--choice", "skip"]) == 30
    assert seen == []


def test_the_conflict_query(monkeypatch, capsys):
    monkeypatch.setattr(gc, "existing_localm", lambda shim, root=None: None)
    assert gc.main(["conflict", "--root", "."]) == 1
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(gc, "existing_localm", lambda shim, root=None: "/opt/x/localm")
    monkeypatch.setattr(gc, "conflict_outranks_user_path", lambda c: False)
    assert gc.main(["conflict", "--root", "."]) == 0
    assert capsys.readouterr().out.splitlines() == ["/opt/x/localm"]


# --------------------------------------------------------------------------- #
#  Uninstall plan: what Astral's uv installer added                            #
# --------------------------------------------------------------------------- #

def test_shared_uv_shell_lines_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(im.sys, "platform", "linux")
    monkeypatch.setattr(im.Path, "home", classmethod(lambda cls: tmp_path))
    rc = tmp_path / ".bashrc"
    rc.write_text('export X=1\n. "$HOME/.local/bin/env"\n', encoding="utf-8")
    other = tmp_path / ".zshrc"
    other.write_text("alias ll=ls\n", encoding="utf-8")
    monkeypatch.setattr(im, "_rc_files", lambda: [rc, other])
    items = im._shared_uv_pointers()
    assert [Path(str(i.path)) for i in items] == [rc]
    assert all(i.status == "keep" and i.kind == "report" for i in items)


def test_shared_uv_user_path_entry_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(im.sys, "platform", "win32")
    monkeypatch.setattr(im.Path, "home", classmethod(lambda cls: tmp_path))
    entry = str(tmp_path / ".local" / "bin")
    monkeypatch.setattr(gc, "_win_read_user_path",
                        lambda: (f"C:\\Other\\bin;{entry};", 2))
    items = im._shared_uv_pointers()
    assert len(items) == 1
    assert entry in str(items[0].path) and "user PATH" in str(items[0].path)


def test_no_pointers_when_nothing_was_added(tmp_path, monkeypatch):
    monkeypatch.setattr(im.sys, "platform", "linux")
    monkeypatch.setattr(im.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(im, "_rc_files", lambda: [])
    assert im._shared_uv_pointers() == []


# --------------------------------------------------------------------------- #
#  Data folder: a home folder that cannot be found                              #
# --------------------------------------------------------------------------- #

def test_an_unknown_home_folder_is_named_as_the_problem(tmp_path, monkeypatch):
    def no_home(cls):
        raise RuntimeError("Could not determine home directory.")
    monkeypatch.setattr(im.Path, "home", classmethod(no_home))
    root = tmp_path / "clone"
    root.mkdir()
    target = tmp_path / "data"
    with pytest.raises(ValueError) as exc:
        im.prepare_data(root, data_dir=str(target))
    text = str(exc.value)
    assert "home folder could not be found" in text
    assert "HOME" in text
    assert "choose a folder of its own" not in text
    assert str(target) not in text
    assert not target.exists()


def test_a_safe_folder_still_works(tmp_path):
    root = tmp_path / "clone"
    root.mkdir()
    target = tmp_path / "data"
    im.prepare_data(root, data_dir=str(target))
    assert target.is_dir()
