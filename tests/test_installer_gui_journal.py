# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graphical installer keeps the same setup journal as setup.bat and setup.sh.

Each step is journaled by name as it begins and finishes, ``complete`` is written only
when the run reaches the end, an interrupted earlier run is reported and picked up, and
the shortcut and the global command are journaled before they are created.
"""

from __future__ import annotations

import pytest

from localm import install_manifest as im
from tests.test_installer_gui import _load


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


def _step(gui, key, fn, *, fatal=True):
    return gui.Step(f"label {key}", fn, fatal=fatal, key=key)


def _ok(emit):
    return None


class TestRunSteps:
    @staticmethod
    def _boom(gui):
        def boom(emit):
            raise gui.StepFailed("it broke")
        return boom

    def test_each_step_is_journaled_and_the_run_is_completed(self, gui):
        failures, fatal = gui.run_steps(
            [_step(gui, "venv", _ok), _step(gui, "torch", _ok)], lambda s: None)
        assert (failures, fatal) == ([], None)
        st = im.journal_state(gui.ROOT)
        assert st["done"] == ["venv", "torch"] and st["complete"] and st["started"] == []

    def test_a_required_step_that_fails_stops_the_run_and_stays_open(self, gui):
        ran = []
        failures, fatal = gui.run_steps(
            [_step(gui, "venv", _ok), _step(gui, "install-localm", self._boom(gui)),
             _step(gui, "torch", lambda e: ran.append("torch"))], lambda s: None)
        assert fatal == "label install-localm: it broke" and ran == []
        st = im.journal_state(gui.ROOT)
        assert st["done"] == ["venv"] and st["started"] == ["install-localm"]
        assert not st["complete"]

    def test_an_optional_step_that_fails_does_not_stop_the_run(self, gui):
        lines = []
        failures, fatal = gui.run_steps(
            [_step(gui, "launcher", self._boom(gui), fatal=False),
             _step(gui, "record", _ok)], lines.append)
        assert fatal is None and failures == ["label launcher: it broke"]
        assert any("did not finish" in line for line in lines)
        st = im.journal_state(gui.ROOT)
        assert st["done"] == ["record"] and st["complete"]

    def test_an_unexpected_error_in_a_required_step_is_reported_not_raised(self, gui):
        def explode(emit):
            raise RuntimeError("kaboom")
        failures, fatal = gui.run_steps([_step(gui, "venv", explode)], lambda s: None)
        assert fatal == "label venv: kaboom"

    def test_the_progress_callback_sees_every_step(self, gui):
        seen = []
        gui.run_steps([_step(gui, "a", _ok), _step(gui, "b", _ok)], lambda s: None,
                      lambda i, n, label: seen.append((i, n, label)))
        assert seen == [(0, 2, "label a"), (1, 2, "label b")]

    def test_a_journal_that_cannot_be_written_warns_once_and_the_install_goes_on(self, gui):
        im.journal_path(gui.ROOT).mkdir()
        lines = []
        gui.begin_journal(lines.append)
        failures, fatal = gui.run_steps(
            [_step(gui, "venv", _ok), _step(gui, "torch", _ok)], lines.append)
        assert (failures, fatal) == ([], None)
        assert sum("Could not write the setup journal" in line for line in lines) == 1


class TestAJournalThatCannotBeReplaced:
    def test_a_finished_journal_that_cannot_be_deleted_warns_and_the_install_goes_on(
            self, gui, monkeypatch):
        im.journal_event(gui.ROOT, "begin", "venv")
        im.journal_event(gui.ROOT, "done", "venv")
        im.journal_event(gui.ROOT, "complete")

        def locked(root):
            raise PermissionError("locked")
        monkeypatch.setattr(im, "journal_reset", locked)
        lines = []
        state = gui.begin_journal(lines.append)
        assert not state["exists"]
        assert sum("Could not write the setup journal" in line for line in lines) == 1


class TestPickingUp:
    def test_a_first_run_has_nothing_to_report(self, gui):
        lines = []
        state = gui.begin_journal(lines.append)
        assert lines == [] and not state["exists"]

    def test_an_interrupted_run_is_reported_and_marked_resumed(self, gui):
        im.journal_event(gui.ROOT, "begin", "venv")
        im.journal_event(gui.ROOT, "done", "venv")
        im.journal_event(gui.ROOT, "begin", "native-runtime")
        lines = []
        state = gui.begin_journal(lines.append)
        assert "stopped after 'venv', while running 'native-runtime'" in " ".join(lines)
        assert state["started"] == ["native-runtime"]
        assert im.journal_state(gui.ROOT)["resumed"] == 1

    def test_a_finished_journal_is_a_fresh_start(self, gui):
        im.journal_event(gui.ROOT, "begin", "venv")
        im.journal_event(gui.ROOT, "done", "venv")
        im.journal_event(gui.ROOT, "complete")
        lines = []
        state = gui.begin_journal(lines.append)
        assert lines == [] and not state["exists"]
        assert not im.journal_path(gui.ROOT).exists()

    def test_a_cut_short_runtime_download_is_redone_with_force(self, gui, monkeypatch):
        seen = []

        def fake_run(cmd, emit, plan, **kw):
            seen.append([str(c) for c in cmd])
            return 0
        monkeypatch.setattr(gui, "_run", fake_run)
        for resume, forced in (({"started": ["native-runtime"]}, True),
                               ({"started": ["venv"]}, False), (None, False)):
            seen.clear()
            steps = gui.build_steps(gui.Plan(backend="cpu"), resume)
            next(s for s in steps if s.key == "native-runtime").run(lambda s: None)
            assert ("--force" in seen[0]) is forced, (resume, seen)


class TestStepNames:
    def test_every_step_has_a_journal_key_and_they_are_unique(self, gui):
        plan = gui.Plan(backend="cpu", add_to_path=True, shortcut="launcher",
                        plugins=("x",))
        keys = [s.key for s in gui.build_steps(plan)]
        assert all(keys) and len(keys) == len(set(keys)), keys

    def test_the_profile_steps_use_the_names_uninstall_reports_on(self, gui):
        plan = gui.Plan(backend="cpu", add_to_path=True, shortcut="launcher")
        keys = {s.key for s in gui.build_steps(plan)}
        assert {"menu-entry", "global-command"} <= keys
        assert {"menu-entry", "global-command"} <= set(im._PROFILE_STEPS)


class TestIntentsBeforeCreation:
    def test_the_shortcut_is_journaled_before_it_is_made(self, gui, monkeypatch):
        target = gui.ROOT / "Desktop" / "LocaLM.lnk"
        monkeypatch.setattr(gui, "intended_shortcut_path", lambda: str(target))
        seen = []

        def fake_make(plan, emit):
            seen.append(im.journal_state(gui.ROOT)["intents"])
            return str(target)
        monkeypatch.setattr(gui, "make_shortcut", fake_make)
        step = next(s for s in gui.build_steps(gui.Plan(shortcut="launcher"))
                    if s.key == "menu-entry")
        step.run(lambda s: None)
        assert seen == [[("shortcut", im._plain_abs(str(target)))]]

    def test_the_global_command_is_journaled_before_it_is_made(self, gui, monkeypatch):
        seen = []

        def fake_run(cmd, emit, plan, **kw):
            if "localm.globalcmd" in cmd and "install" in cmd:
                seen.append(list(im.journal_state(gui.ROOT)["intents"]))
            return 0
        monkeypatch.setattr(gui, "_run", fake_run)
        monkeypatch.setattr(gui, "_query", lambda args: "")
        step = next(s for s in gui.build_steps(gui.Plan(add_to_path=True, backend="own"))
                    if s.key == "global-command")
        step.run(lambda s: None)
        assert seen == [[("command", im._plain_abs(str(gui.intended_command_path())))]]
