# SPDX-License-Identifier: AGPL-3.0-or-later
"""The setup journal: a setup that is killed leaves a record of how far it got.

``.localm-setup-journal`` (in the clone) gets one tab-separated line per event. The
next setup run reads it to say where the last one stopped, and uninstall treats a
shortcut or command the journal says setup was about to create as if it had been
recorded, through the same safety checks, so a kill between "create" and "record"
leaves nothing behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localm import install_manifest as im

# The isolating fixture (home, app data, PATH, process list) every manifest test uses.
from tests.test_install_manifest import _shortcut_to, isolated  # noqa: F401


def _lines(root) -> list:
    return im.journal_path(root).read_text(encoding="utf-8").splitlines()


class TestJournalFile:
    def test_events_are_one_tab_separated_line_each(self, tmp_path):
        im.journal_event(tmp_path, "begin", "venv")
        im.journal_event(tmp_path, "done", "venv")
        im.journal_event(tmp_path, "intend", "shortcut", str(tmp_path / "x" / "LocaLM.lnk"))
        im.journal_event(tmp_path, "complete")
        lines = _lines(tmp_path)
        assert lines[0] == "begin\tvenv"
        assert lines[1] == "done\tvenv"
        assert lines[2].startswith("intend\tshortcut\t") and lines[2].endswith("LocaLM.lnk")
        assert lines[3] == "complete"

    @pytest.mark.parametrize("args", [
        ("explode", ""), ("begin", ""), ("begin", "a\tb"), ("begin", "a\nb"),
        ("intend", "nonsense", "/x"), ("intend", "shortcut", ""),
        ("intend", "shortcut", "a\tb"),
    ])
    def test_a_malformed_event_is_refused_and_writes_nothing(self, tmp_path, args):
        with pytest.raises(ValueError):
            im.journal_event(tmp_path, *args)
        assert not im.journal_path(tmp_path).exists()

    def test_reset_removes_it_and_is_harmless_when_absent(self, tmp_path):
        im.journal_reset(tmp_path)
        im.journal_event(tmp_path, "begin", "venv")
        im.journal_reset(tmp_path)
        assert not im.journal_path(tmp_path).exists()


class TestJournalState:
    def test_no_journal(self, tmp_path):
        st = im.journal_state(tmp_path)
        assert st["exists"] is False and st["complete"] is False
        assert im.describe_journal(st) == "no setup has started here"

    def test_a_finished_setup(self, tmp_path):
        for ev, name in (("begin", "venv"), ("done", "venv"), ("complete", "")):
            im.journal_event(tmp_path, ev, name)
        st = im.journal_state(tmp_path)
        assert st["complete"] is True and st["done"] == ["venv"] and st["started"] == []
        assert im.describe_journal(st) == "the last setup finished"

    def test_a_setup_killed_between_steps(self, tmp_path):
        for step in ("uv", "venv"):
            im.journal_event(tmp_path, "begin", step)
            im.journal_event(tmp_path, "done", step)
        st = im.journal_state(tmp_path)
        assert st["complete"] is False
        assert st["done"] == ["uv", "venv"] and st["last_done"] == "venv"
        assert st["started"] == []
        assert "stopped after 'venv'" in im.describe_journal(st)

    def test_a_setup_killed_inside_a_step(self, tmp_path):
        im.journal_event(tmp_path, "begin", "uv")
        im.journal_event(tmp_path, "done", "uv")
        im.journal_event(tmp_path, "begin", "native-runtime")
        st = im.journal_state(tmp_path)
        assert st["last_done"] == "uv" and st["started"] == ["native-runtime"]
        text = im.describe_journal(st)
        assert "stopped after 'uv'" in text and "while running 'native-runtime'" in text

    def test_a_setup_killed_before_its_first_step_finished(self, tmp_path):
        im.journal_event(tmp_path, "begin", "uv")
        assert "before finishing its first step" in im.describe_journal(
            im.journal_state(tmp_path))

    def test_a_new_begin_after_complete_means_it_is_not_complete_again(self, tmp_path):
        im.journal_event(tmp_path, "complete")
        im.journal_event(tmp_path, "begin", "venv")
        assert im.journal_state(tmp_path)["complete"] is False

    def test_complete_closes_any_step_left_open(self, tmp_path):
        im.journal_event(tmp_path, "begin", "x")
        im.journal_event(tmp_path, "complete")
        assert im.journal_state(tmp_path)["started"] == []

    def test_resumes_are_counted(self, tmp_path):
        im.journal_event(tmp_path, "begin", "venv")
        im.journal_event(tmp_path, "resume")
        im.journal_event(tmp_path, "resume")
        assert im.journal_state(tmp_path)["resumed"] == 2

    def test_a_torn_last_line_and_unknown_lines_are_ignored(self, tmp_path):
        im.journal_path(tmp_path).write_bytes(
            b"begin\tvenv\ndone\tvenv\ngarbage line\nbegin\tnative-runt")
        st = im.journal_state(tmp_path)
        assert st["done"] == ["venv"] and st["started"] == []

    def test_intents_are_listed_in_order(self, tmp_path):
        im.journal_event(tmp_path, "intend", "shortcut", str(tmp_path / "a" / "LocaLM.lnk"))
        im.journal_event(tmp_path, "intend", "command", str(tmp_path / "b" / "localm"))
        kinds = [k for k, _p in im.journal_state(tmp_path)["intents"]]
        assert kinds == ["shortcut", "command"]


class TestJournalCli:
    def test_status_is_zero_for_none_and_finished_and_five_for_unfinished(self, tmp_path,
                                                                           capsys):
        assert im.main(["journal", "--root", str(tmp_path), "status"]) == 0
        im.main(["journal", "--root", str(tmp_path), "begin", "venv"])
        assert im.main(["journal", "--root", str(tmp_path), "status"]) == im.EXIT_INCOMPLETE
        assert "before finishing its first step" in capsys.readouterr().out
        im.main(["journal", "--root", str(tmp_path), "complete"])
        assert im.main(["journal", "--root", str(tmp_path), "status"]) == 0

    def test_a_bad_event_exits_one_and_says_why(self, tmp_path, capsys):
        assert im.main(["journal", "--root", str(tmp_path), "begin"]) == 1
        assert "setup journal" in capsys.readouterr().err

    def test_reset(self, tmp_path):
        im.main(["journal", "--root", str(tmp_path), "begin", "venv"])
        assert im.main(["journal", "--root", str(tmp_path), "reset"]) == 0
        assert not im.journal_path(tmp_path).exists()


class TestUninstallUsesTheJournal:
    def test_a_shortcut_created_but_never_recorded_is_removed(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        desk = _shortcut_to(tmp_path / "Desktop" / "LocaLM.lnk", clone)
        im.journal_event(clone, "intend", "shortcut", str(desk))
        assert im.load(clone) is None            # setup died before recording it

        rep = im.uninstall(clone, force=True)

        assert not desk.exists()
        assert rep["no_manifest"] is True        # and the report still says so

    def test_an_intended_shortcut_that_was_never_created_is_fine(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        im.journal_event(clone, "intend", "shortcut", str(tmp_path / "Desktop" / "LocaLM.lnk"))
        rep = im.uninstall(clone, force=True)
        assert rep["exit"] == im.EXIT_OK and not rep["failed"]

    def test_an_intended_shortcut_that_opens_another_folder_is_kept(self, tmp_path):
        clone, sibling = tmp_path / "LocaLM", tmp_path / "LocaLM2"
        clone.mkdir()
        sibling.mkdir()
        desk = _shortcut_to(tmp_path / "Desktop" / "LocaLM.lnk", sibling)
        im.journal_event(clone, "intend", "shortcut", str(desk))
        im.uninstall(clone, force=True)
        assert desk.is_file()

    def test_an_intended_shortcut_with_a_foreign_name_is_never_removed(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        note = tmp_path / "Desktop" / "notes.txt"
        note.parent.mkdir()
        note.write_text("mine", encoding="utf-8")
        im.journal_event(clone, "intend", "shortcut", str(note))
        rep = im.uninstall(clone, force=True)
        assert note.read_text(encoding="utf-8") == "mine"
        assert any("not a LocaLM shortcut" in why for _p, why in rep["refused"])

    @pytest.mark.skipif(os.name == "nt", reason="needs a symlink")
    def test_a_command_symlink_created_but_never_recorded_is_removed(self, tmp_path):
        clone = tmp_path / "LocaLM"
        (clone / ".venv" / "bin").mkdir(parents=True)
        target = clone / ".venv" / "bin" / "localm"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        shim = tmp_path / "bin" / "localm"
        shim.parent.mkdir()
        shim.symlink_to(target)
        im.journal_event(clone, "intend", "command", str(shim))
        im.uninstall(clone, force=True)
        assert not shim.is_symlink() and not shim.exists()

    @pytest.mark.skipif(os.name == "nt", reason="needs a symlink")
    def test_a_command_that_points_at_another_install_is_kept(self, tmp_path):
        clone, other = tmp_path / "LocaLM", tmp_path / "Other"
        clone.mkdir()
        other.mkdir()
        target = other / "localm"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        shim = tmp_path / "bin" / "localm"
        shim.parent.mkdir()
        shim.symlink_to(target)
        im.journal_event(clone, "intend", "command", str(shim))
        im.uninstall(clone, force=True)
        assert shim.is_symlink()

    def test_a_recorded_shortcut_is_not_handled_twice(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        desk = _shortcut_to(tmp_path / "Desktop" / "LocaLM.lnk", clone)
        im.record(clone, shortcut=str(desk))
        im.journal_event(clone, "intend", "shortcut", str(desk))
        rep = im.uninstall(clone, force=True)
        assert not desk.exists()
        assert not rep["failed"]

    def test_an_unfinished_profile_step_is_reported_as_a_note(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        im.journal_event(clone, "begin", "global-command")
        rep = im.uninstall(clone, force=True)
        assert any(step == "global-command" and "stopped" in why
                   for step, why in rep["notes"])

    def test_a_finished_profile_step_is_not_reported(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        im.journal_event(clone, "begin", "global-command")
        im.journal_event(clone, "done", "global-command")
        rep = im.uninstall(clone, force=True)
        assert not any(step == "global-command" for step, _w in rep["notes"])

    def test_the_journal_goes_with_the_manifest(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        im.record(clone)
        im.journal_event(clone, "begin", "venv")
        im.uninstall(clone, force=True)
        assert not im.journal_path(clone).exists()
        assert not im.manifest_path(clone).exists()

    def test_a_dry_run_keeps_the_journal(self, tmp_path):
        clone = tmp_path / "LocaLM"
        clone.mkdir()
        im.journal_event(clone, "begin", "venv")
        im.uninstall(clone, dry_run=True)
        assert im.journal_path(clone).exists()
