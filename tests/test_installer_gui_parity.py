# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graphical installer asks and does what setup.bat / setup.sh do.

Covers the window's half of each setup-parity decision: an existing .venv is
kept unless replaced, the certificate fallback and the locked-.venv recovery,
the own-build folder, a failed runtime that asks whether to continue,
setup-llama questions answered in the window instead of --yes, the plugin deps
answer saved, the launcher's uv answer used and shown, and the Linux menu
entry name.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_GUI_PATH = Path(__file__).resolve().parents[1] / "installer" / "gui.py"
_MOD_NAME = "localm_installer_gui_parity"


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _GUI_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for name in (mod.STORE_ENV, mod.UV_ENV, mod.UV_INSTALLED_ENV):
        monkeypatch.delenv(name, raising=False)
    return mod


def _step(gui, plan, prefix, resume=None):
    matches = [s for s in gui.build_steps(plan, resume) if s.label.startswith(prefix)]
    assert len(matches) == 1, [s.label for s in gui.build_steps(plan, resume)]
    return matches[0]


def _recorder(gui, root, outputs=None):
    """A _run stand-in: records (argv, kwargs, env), creates .venv on `uv venv`,
    and returns the next (code, lines) from *outputs* (default success)."""
    calls = []
    queue = list(outputs or [])

    def fake_run(cmd, emit, plan, **kw):
        argv = [str(c) for c in cmd]
        calls.append((argv, kw, gui._env_for(plan)))
        code, lines = queue.pop(0) if queue else (0, [])
        if kw.get("collect") is not None:
            kw["collect"].extend(lines)
        if code == 0 and argv[1:2] == ["venv"]:
            (root / ".venv").mkdir(exist_ok=True)
        return code
    return fake_run, calls


def _answers(*replies):
    """An ASK that returns *replies* in order and records the questions."""
    asked = []
    it = iter(replies)

    def ask(question):
        asked.append(question)
        return next(it)
    return ask, asked


# --------------------------------------------------------------------------- #
#  An existing .venv (G5)                                                     #
# --------------------------------------------------------------------------- #

class TestExistingVenv:
    def test_it_is_kept_by_default(self, gui, tmp_path, monkeypatch):
        (tmp_path / ".venv").mkdir()
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        logged = []
        _step(gui, gui.Plan(), "Creating the Python environment").run(logged.append)
        assert calls == []
        assert "Keeping the existing .venv." in logged
        assert not (tmp_path / ".venv" / ".localm-venv").exists()

    def test_replace_recreates_it_with_clear(self, gui, tmp_path, monkeypatch):
        (tmp_path / ".venv").mkdir()
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        _step(gui, gui.Plan(replace_venv=True), "Creating the Python environment").run(
            lambda s: None)
        assert calls[0][0][1:2] == ["venv"] and "--clear" in calls[0][0]
        assert (tmp_path / ".venv" / ".localm-venv").is_file()

    def test_a_foreign_venv_a_cut_short_run_left_is_recreated(
            self, gui, tmp_path, monkeypatch):
        (tmp_path / ".venv").mkdir()
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        _step(gui, gui.Plan(), "Creating the Python environment",
              {"started": ["venv"]}).run(lambda s: None)
        assert calls and calls[0][0][1:2] == ["venv"]

    def test_the_question_is_shown_only_when_a_venv_exists(self, gui, tmp_path):
        tk = pytest.importorskip("tkinter")
        from tkinter import filedialog, ttk
        from tests._tk_root import build_tk_root
        for exists in (False, True):
            if exists:
                (tmp_path / ".venv").mkdir()
            root = build_tk_root(tk.Tk)
            root.withdraw()
            try:
                w = gui.Wizard(root, tk, ttk, filedialog)
                texts = _texts(w.pages[1][1])
                assert any(".venv found" in t for t in texts) is exists
                assert w.current_plan().replace_venv is False
            finally:
                root.destroy()


def _texts(widget):
    """The text option of every widget under *widget* that has one."""
    out = []
    for w in _widgets(widget):
        try:
            out.append(str(w.cget("text")))
        except Exception:
            continue
    return out


def _widgets(widget):
    out = [widget]
    for child in widget.winfo_children():
        out.extend(_widgets(child))
    return out


# --------------------------------------------------------------------------- #
#  Certificate fallback and a locked .venv (G9)                               #
# --------------------------------------------------------------------------- #

class TestVenvRecovery:
    def test_a_certificate_error_retries_once_with_the_bundled_roots(
            self, gui, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        fake, calls = _recorder(gui, tmp_path, [
            (1, ["error: invalid peer certificate: UnknownIssuer"]), (0, [])])
        monkeypatch.setattr(gui, "_run", fake)
        logged = []
        gui.create_venv(gui.Plan(), logged.append)
        assert len(calls) == 2
        assert calls[0][2].get("UV_SYSTEM_CERTS") == "1"
        assert "UV_SYSTEM_CERTS" not in calls[1][2]
        assert any("certificate bundle" in line for line in logged)
        assert "UV_SYSTEM_CERTS" not in gui._env_for(gui.Plan()), (
            "the rest of the run keeps the fallback")

    def test_a_second_certificate_error_is_not_retried_again(
            self, gui, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        fake, calls = _recorder(gui, tmp_path, [
            (1, ["certificate"]), (1, ["certificate"])])
        monkeypatch.setattr(gui, "_run", fake)
        with pytest.raises(gui.StepFailed):
            gui.create_venv(gui.Plan(), lambda s: None)
        assert len(calls) == 2

    def _locked(self, gui, tmp_path, monkeypatch, lockers):
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        monkeypatch.setattr(gui.time, "sleep", lambda s: None)
        monkeypatch.setattr(gui, "find_venv_lockers", lambda root: lockers)
        ran = []
        monkeypatch.setattr(gui.subprocess, "run",
                            lambda argv, **kw: ran.append([str(a) for a in argv]))
        return ran

    def test_a_locked_venv_resets_attributes_then_stops_its_lockers_when_asked(
            self, gui, tmp_path, monkeypatch):
        ran = self._locked(gui, tmp_path, monkeypatch, [(4242, "LocaLM.exe")])
        fake, calls = _recorder(gui, tmp_path, [
            (1, ["Access is denied. (os error 5)"]),
            (1, ["Access is denied. (os error 5)"]), (0, [])])
        monkeypatch.setattr(gui, "_run", fake)
        ask, asked = _answers("y")
        monkeypatch.setattr(gui, "ASK", ask)
        gui.create_venv(gui.Plan(), lambda s: None)
        assert ran[0][0] == "attrib"
        assert ran[1][:3] == ["taskkill", "/PID", "4242"]
        assert [q["id"] for q in asked] == ["venv-lockers"]
        assert len(calls) == 3

    def test_lockers_are_never_stopped_without_someone_to_ask(
            self, gui, tmp_path, monkeypatch):
        ran = self._locked(gui, tmp_path, monkeypatch, [(4242, "LocaLM.exe")])
        fake, calls = _recorder(gui, tmp_path, [(1, ["os error 32"])] * 3)
        monkeypatch.setattr(gui, "_run", fake)
        with pytest.raises(gui.StepFailed):
            gui.create_venv(gui.Plan(), lambda s: None)
        assert not any(r[0] == "taskkill" for r in ran)

    def test_try_again_is_asked_and_no_stops(self, gui, tmp_path, monkeypatch):
        self._locked(gui, tmp_path, monkeypatch, [])
        monkeypatch.setenv("OneDrive", str(tmp_path.parent))
        fake, calls = _recorder(gui, tmp_path, [(1, ["os error 5"])] * 3)
        monkeypatch.setattr(gui, "_run", fake)
        ask, asked = _answers("n")
        monkeypatch.setattr(gui, "ASK", ask)
        logged = []
        with pytest.raises(gui.StepFailed):
            gui.create_venv(gui.Plan(), logged.append)
        assert [q["id"] for q in asked] == ["venv-retry"]
        assert any("OneDrive" in line for line in logged)


# --------------------------------------------------------------------------- #
#  The own build, a failed runtime, setup-llama questions (G4, G11, G12)      #
# --------------------------------------------------------------------------- #

class TestRuntime:
    def test_the_own_build_folder_is_copied_in(self, gui, tmp_path, monkeypatch):
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        _step(gui, gui.Plan(backend="own", own_build_dir="D:/llama/build"),
              "Copying in your own llama.cpp build").run(lambda s: None)
        argv, kw, _ = calls[0]
        assert argv[1:] == ["setup-llama", "--from", "D:/llama/build"]
        assert kw.get("answers") is True

    def test_no_build_folder_skips_and_says_how_to_provision_later(self, gui):
        logged = []
        _step(gui, gui.Plan(backend="own"),
              "Skipping the inference runtime download").run(logged.append)
        assert "setup-llama --backend <" in logged[0]
        assert not any("setup-llama" in s.label for s in gui.build_steps(gui.Plan(backend="own")))

    def test_setup_llama_runs_without_yes_and_answers_in_the_window(
            self, gui, tmp_path, monkeypatch):
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        _step(gui, gui.Plan(backend="cuda"), "Provisioning the cuda").run(lambda s: None)
        argv, kw, _ = calls[0]
        assert "--yes" not in argv and "-y" not in argv
        assert kw.get("answers") is True

    @pytest.mark.parametrize("reply,fatal", [("y", False), ("n", True)])
    def test_a_failed_runtime_asks_whether_to_continue(
            self, gui, tmp_path, monkeypatch, reply, fatal):
        fake, calls = _recorder(gui, tmp_path, [(1, [])])
        monkeypatch.setattr(gui, "_run", fake)
        ask, asked = _answers(reply)
        monkeypatch.setattr(gui, "ASK", ask)
        step = _step(gui, gui.Plan(backend="vulkan"), "Provisioning the vulkan")
        logged = []
        failures, stopped = gui.run_steps([step], logged.append)
        assert [q["id"] for q in asked] == ["runtime-failed"]
        assert any("setup-llama --backend vulkan --force" in line for line in logged)
        if fatal:
            assert stopped and "setup-llama --backend vulkan --force" in stopped
        else:
            assert stopped is None
            assert failures and "setup-llama --backend vulkan --force" in failures[0]

    def test_unanswered_the_default_is_to_continue(self, gui, tmp_path, monkeypatch):
        fake, _ = _recorder(gui, tmp_path, [(1, [])])
        monkeypatch.setattr(gui, "_run", fake)
        step = _step(gui, gui.Plan(backend="vulkan"), "Provisioning the vulkan")
        failures, stopped = gui.run_steps([step], lambda s: None)
        assert stopped is None and failures


class TestAskProtocol:
    """A real child process asks over stdio; the window's _run answers it."""

    CHILD = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, {src!r})
        from localm.setup_llama import _common
        print("env", os.environ.get(_common.ASK_ENV))
        print("confirm", _common._confirm("Install Vulkan?", True))
        print("choose", _common._choose("Pick", [("1", "a"), ("2", "b"), ("3", "c")], "2"))
        print("interactive", _common._interactive())
    """)

    def test_questions_reach_ask_and_the_answers_reach_the_child(
            self, gui, tmp_path, monkeypatch):
        script = tmp_path / "child.py"
        script.write_text(self.CHILD.format(src=str(_GUI_PATH.parents[1])),
                          encoding="utf-8")
        ask, asked = _answers("n", "3")
        monkeypatch.setattr(gui, "ASK", ask)
        logged = []
        code = gui._run([sys.executable, str(script)], logged.append, gui.Plan(),
                        answers=True)
        assert code == 0, logged
        assert [q["kind"] for q in asked] == ["confirm", "choose"]
        assert asked[0]["default"] == "y" and asked[1]["default"] == "2"
        assert asked[1]["choices"] == [["1", "a"], ["2", "b"], ["3", "c"]]
        assert "env stdio" in logged
        assert "confirm False" in logged and "choose 3" in logged
        assert "interactive True" in logged
        assert not any(line.startswith("@@localm-ask") for line in logged)

    def test_without_answers_the_child_sees_no_protocol_and_no_input(
            self, gui, tmp_path, monkeypatch):
        script = tmp_path / "child.py"
        script.write_text("import os, sys\n"
                          "print('env', os.environ.get('LOCALM_SETUP_ASK'))\n"
                          "print('stdin', repr(sys.stdin.read()))\n", encoding="utf-8")
        monkeypatch.delenv("LOCALM_SETUP_ASK", raising=False)
        logged = []
        gui._run([sys.executable, str(script)], logged.append, gui.Plan())
        assert "env None" in logged and "stdin ''" in logged

    def test_an_unknown_answer_is_the_default(self, tmp_path, monkeypatch):
        env = {**os.environ, "LOCALM_SETUP_ASK": "stdio",
               "PYTHONPATH": str(_GUI_PATH.parents[1])}
        code = ("from localm.setup_llama import _common;"
                "print(_common._confirm('q', True));"
                "print(_common._choose('p', [('1','a'),('2','b')], '2'))")
        out = subprocess.run([sys.executable, "-c", code], input="maybe\n9\n",
                             capture_output=True, text=True, env=env, timeout=60)
        lines = [ln for ln in out.stdout.splitlines() if not ln.startswith("@@")]
        assert lines == ["True", "2"], out.stdout + out.stderr


# --------------------------------------------------------------------------- #
#  Plugins, the launcher's uv answer, the Linux menu entry (G1, G7, G8, G10)  #
# --------------------------------------------------------------------------- #

def test_the_plugin_deps_answer_is_saved(gui, tmp_path, monkeypatch):
    fake, calls = _recorder(gui, tmp_path)
    monkeypatch.setattr(gui, "_run", fake)
    _step(gui, gui.Plan(plugins=("coder",), backend="own"),
          "Installing the optional features").run(lambda s: None)
    assert "--remember-deps" in calls[0][0]


class TestLauncherUv:
    def test_the_launchers_uv_is_used_even_with_a_contained_one(
            self, gui, tmp_path, monkeypatch):
        contained = tmp_path / ".uv"
        contained.mkdir()
        (contained / ("uv.exe" if gui.IS_WINDOWS else "uv")).write_bytes(b"")
        shared = tmp_path / "elsewhere" / "uv.exe"
        shared.parent.mkdir()
        shared.write_bytes(b"")
        monkeypatch.setenv(gui.UV_ENV, str(shared))
        assert gui.find_uv(tmp_path) == str(shared)

    def test_a_launcher_uv_that_does_not_exist_falls_back(self, gui, tmp_path, monkeypatch):
        monkeypatch.setenv(gui.UV_ENV, str(tmp_path / "missing" / "uv"))
        monkeypatch.setattr(gui.uv_tool(), "find_uv", lambda root: "found")
        assert gui.find_uv(tmp_path) == "found"

    @pytest.mark.parametrize("value,expected", [("shared", "shared"),
                                                 ("portable", "portable"),
                                                 ("", "portable"), ("junk", "portable")])
    def test_the_launchers_answer_decides_the_store(self, gui, monkeypatch, value, expected):
        monkeypatch.setenv(gui.STORE_ENV, value)
        assert gui.launcher_store() == expected

    def test_a_shared_uv_the_launcher_installed_is_recorded(
            self, gui, tmp_path, monkeypatch):
        shared = tmp_path.parent / f"{tmp_path.name}-uv" / "uv.exe"
        shared.parent.mkdir()
        shared.write_bytes(b"")
        monkeypatch.setenv(gui.UV_ENV, str(shared))
        monkeypatch.setenv(gui.UV_INSTALLED_ENV, "1")
        fake, calls = _recorder(gui, tmp_path)
        monkeypatch.setattr(gui, "_run", fake)
        _step(gui, gui.Plan(backend="own", portable_store=False),
              "Writing the install record").run(lambda s: None)
        rec = calls[0][0]
        assert rec[rec.index("--uv-exe") + 1] == str(shared)
        assert "--uv-shared-installed" in rec

    def test_the_window_shows_the_answer_and_never_asks(self, gui, monkeypatch):
        tk = pytest.importorskip("tkinter")
        from tkinter import filedialog, ttk
        from tests._tk_root import build_tk_root
        monkeypatch.setenv(gui.STORE_ENV, "shared")
        root = build_tk_root(tk.Tk)
        root.withdraw()
        try:
            w = gui.Wizard(root, tk, ttk, filedialog)
            page = w.pages[1][1]
            radios = [c for c in _widgets(page) if c.winfo_class() == "TRadiobutton"]
            assert all(str(r.cget("variable")) != str(w.store_var) for r in radios)
            texts = _texts(page)
            assert any(t.startswith("Shared") for t in texts)
            assert w.current_plan().portable_store is False
        finally:
            root.destroy()


class TestLinuxMenuEntry:
    @pytest.fixture()
    def posix(self, gui, tmp_path, monkeypatch):
        home = tmp_path / "userhome"
        home.mkdir()
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        return home / ".local" / "share" / "applications"

    def test_the_file_name_is_the_one_setup_sh_writes(self, gui, posix):
        written = gui.make_shortcut(gui.Plan(shortcut="launcher"), lambda s: None)
        assert Path(written) == posix / "localm.desktop"
        assert gui.intended_shortcut_path().endswith("localm.desktop")

    def test_straight_to_the_gui_uses_the_branded_binary(self, gui, tmp_path, posix,
                                                          monkeypatch):
        bindir = gui.venv_bin(tmp_path)
        bindir.mkdir(parents=True)
        (bindir / "LocaLM").write_bytes(b"")
        monkeypatch.setattr(gui.os, "access", lambda p, mode: Path(p).name == "LocaLM")
        text = Path(gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)).read_text()
        assert f"Exec={bindir / 'LocaLM'} -m localm gui" in text

    def test_without_the_branded_binary_it_runs_localm_gui(self, gui, tmp_path, posix):
        text = Path(gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)).read_text()
        assert f"Exec={gui.venv_bin(tmp_path) / 'localm'} gui" in text

    def test_the_older_entry_for_this_folder_is_migrated(self, gui, tmp_path, posix):
        posix.mkdir(parents=True)
        (posix / "Probe").write_bytes(b"")
        if (posix / "probe").exists():
            pytest.skip("the two names are one file on a case-insensitive filesystem")
        (posix / "Probe").unlink()
        old = posix / "LocaLM.desktop"
        old.write_text(f"[Desktop Entry]\nPath={tmp_path}\n", encoding="utf-8")
        gui.make_shortcut(gui.Plan(), lambda s: None)
        assert not old.exists()
        assert (posix / "localm.desktop").is_file()

    def test_another_folders_entry_is_left_alone(self, gui, tmp_path, posix):
        posix.mkdir(parents=True)
        old = posix / "LocaLM.desktop"
        old.write_text("[Desktop Entry]\nPath=/somewhere/else\n", encoding="utf-8")
        gui.make_shortcut(gui.Plan(), lambda s: None)
        assert old.exists()
