# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graphical installer's decision logic (installer/gui.py).

The installer runs BEFORE localm is installed, on uv's managed CPython, so it
is not an importable part of the package: these load it by path. Its tkinter
import lives inside main(), so everything below runs headlessly.

What is pinned here is the part that decides what gets DONE to a machine - the
step list, and the data-directory write - not the widgets.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GUI_PATH = Path(__file__).resolve().parents[1] / "installer" / "gui.py"
_MOD_NAME = "localm_installer_gui"


def _load():
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _GUI_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for a module that was only
    # ever exec'd, and every dataclass in the installer then fails to build.
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    """The installer module with its ROOT pointed at a throwaway directory, so
    no test can write into the real clone."""
    mod = _load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


def _step(gui, plan, prefix):
    matches = [s for s in gui.build_steps(plan) if s.label.startswith(prefix)]
    assert len(matches) == 1, f"expected one {prefix!r} step, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------- #
#  What gets installed                                                         #
# --------------------------------------------------------------------------- #

def test_desktop_extra_only_when_an_app_window_was_asked_for(gui):
    """pythonnet arrives with the desktop extra, so a default install must not
    take it on unasked."""
    assert "desktop" not in gui.Plan().extras
    assert "desktop" in gui.Plan(app_window=True).extras


def test_own_backend_skips_provisioning(gui):
    """'I will provide my own build' must not then download one."""
    labels = [s.label for s in gui.build_steps(gui.Plan(backend="own"))]
    assert not any("Provisioning" in l for l in labels)
    labels = [s.label for s in gui.build_steps(gui.Plan(backend="cpu"))]
    assert any("Provisioning" in l for l in labels)


def test_shortcut_and_path_steps_are_opt_in(gui):
    """Neither the Desktop nor the user's PATH is touched unless asked."""
    labels = [s.label for s in gui.build_steps(
        gui.Plan(shortcut="none", add_to_path=False))]
    assert not any("shortcut" in l for l in labels)
    assert not any("PATH" in l for l in labels)

    labels = [s.label for s in gui.build_steps(
        gui.Plan(shortcut="gui", add_to_path=True))]
    assert any("shortcut" in l for l in labels)
    assert any("PATH" in l for l in labels)


def test_optional_steps_cannot_fail_the_install(gui):
    """A machine with no working torch wheel still gets a usable GGUF install,
    so those steps are marked non-fatal; the ones that define the install are
    not."""
    steps = {s.label: s for s in gui.build_steps(
        gui.Plan(backend="cpu", shortcut="gui", add_to_path=True))}
    assert not [s for l, s in steps.items() if "PyTorch" in l][0].fatal
    assert not [s for l, s in steps.items() if "shortcut" in l][0].fatal
    assert not [s for l, s in steps.items() if "PATH" in l][0].fatal
    assert [s for l, s in steps.items() if "Creating the Python" in l][0].fatal
    assert [s for l, s in steps.items() if l.startswith("Recording")][0].fatal


# --------------------------------------------------------------------------- #
#  Where the data goes                                                         #
# --------------------------------------------------------------------------- #

def test_portable_creates_home_and_clears_a_stale_marker(gui, tmp_path):
    (tmp_path / "localm-home.cfg").write_text("/somewhere/old", encoding="utf-8")
    _step(gui, gui.Plan(portable_data=True), "Recording").run(lambda _l: None)
    assert (tmp_path / "home").is_dir()
    assert not (tmp_path / "localm-home.cfg").exists(), \
        "a leftover marker would still win over ./home at the next start"


def test_custom_path_is_created_and_recorded(gui, tmp_path):
    target = tmp_path / "elsewhere" / "data"
    _step(gui, gui.Plan(portable_data=False, data_path=str(target)),
          "Recording").run(lambda _l: None)
    assert target.is_dir()
    assert (tmp_path / "localm-home.cfg").read_text(encoding="utf-8") == str(target)


def test_a_relative_path_is_refused_and_writes_no_marker(gui, tmp_path):
    """A relative path would resolve against whatever directory localm is
    started from. Refusing it is not enough - the marker must not be written
    either, or the next start reads a path that was rejected."""
    with pytest.raises(gui.StepFailed):
        _step(gui, gui.Plan(portable_data=False, data_path="relative/dir"),
              "Recording").run(lambda _l: None)
    assert not (tmp_path / "localm-home.cfg").exists()


def test_an_uncreatable_directory_writes_no_marker(gui, tmp_path):
    """The directory is made BEFORE the marker for this reason: a marker must
    never point at something that could not be created."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    with pytest.raises(gui.StepFailed):
        _step(gui, gui.Plan(portable_data=False, data_path=str(blocker / "sub")),
              "Recording").run(lambda _l: None)
    assert not (tmp_path / "localm-home.cfg").exists()


# --------------------------------------------------------------------------- #
#  Running commands                                                            #
# --------------------------------------------------------------------------- #

def test_a_failing_command_raises_rather_than_reporting_success(gui):
    lines = []
    with pytest.raises(gui.StepFailed):
        gui._run([gui.sys.executable, "-c", "import sys; sys.exit(3)"],
                 lines.append, gui.Plan())


def test_allow_fail_reports_the_failure_instead_of_hiding_it(gui):
    """An optional step that fails must still SAY so in the log."""
    lines = []
    code = gui._run([gui.sys.executable, "-c", "import sys; sys.exit(3)"],
                    lines.append, gui.Plan(), allow_fail=True)
    assert code == 3
    assert any("exited 3" in l for l in lines)


def test_command_output_is_streamed_into_the_log(gui):
    lines = []
    gui._run([gui.sys.executable, "-c", "print('hello from the step')"],
             lines.append, gui.Plan())
    assert any("hello from the step" in l for l in lines)


def test_portable_store_contains_uv_inside_the_install(gui, tmp_path):
    """Portable means nothing is written to the user profile: uv's managed
    interpreter and its wheel cache both land inside the install.

    Non-portable does not CLEAR an inherited UV_* setting (setup.bat does not
    either - a machine-wide choice is the user's), it simply does not point
    them into the install, so the assertion is about the install-local paths
    rather than the keys being absent."""
    env = gui._env_for(gui.Plan(portable_store=True))
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / ".python")
    assert env["UV_CACHE_DIR"] == str(tmp_path / ".cache")

    plain = gui._env_for(gui.Plan(portable_store=False))
    assert plain.get("UV_PYTHON_INSTALL_DIR") != str(tmp_path / ".python")
    assert plain.get("UV_CACHE_DIR") != str(tmp_path / ".cache")


# --------------------------------------------------------------------------- #
#  Finding uv                                                                  #
# --------------------------------------------------------------------------- #

class TestUvResolution:
    """The installer runs before anything is installed, so uv is the one tool
    it cannot assume. setup-gui.bat bootstraps a portable copy into ./.uv when
    the machine has none, and Astral's installer updates the PERSISTENT PATH
    rather than the shell that is already running - so on a fresh clone uv is
    routinely present in the folder and absent from PATH.

    What is pinned here is that the entry check and the steps resolve the SAME
    uv. When they disagreed, the window opened and died on its first command
    with "could not start uv: [WinError 2]"."""

    @pytest.fixture()
    def machine(self, gui, tmp_path, monkeypatch):
        """A machine with no uv at all: none on PATH, none in the clone, and a
        home directory that cannot contain one either."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        return gui

    @staticmethod
    def _portable_uv(gui, root):
        d = root / ".uv"
        d.mkdir(parents=True, exist_ok=True)
        exe = d / ("uv.exe" if gui.IS_WINDOWS else "uv")
        exe.write_bytes(b"")
        exe.chmod(0o755)
        return exe

    @staticmethod
    def _commands(gui, monkeypatch, plan=None):
        """Every command the install would run, without running any of them."""
        seen = []

        def fake_run(cmd, emit, plan, **kw):
            seen.append([str(c) for c in cmd])
            return 0

        (gui.ROOT / ".venv").mkdir(exist_ok=True)
        monkeypatch.setattr(gui, "_run", fake_run)
        monkeypatch.setattr(gui, "torch_spec_for", lambda backend: ("torch", ""))
        for step in gui.build_steps(plan or gui.Plan()):
            try:
                step.run(lambda s: None)
            except gui.StepFailed:
                raise
            except Exception:
                pass
        return seen

    def test_the_steps_run_the_portable_uv_when_it_is_not_on_path(
            self, machine, tmp_path, monkeypatch):
        """The reported failure: a fresh clone whose uv lives in ./.uv."""
        exe = self._portable_uv(machine, tmp_path)
        venv_cmd = self._commands(machine, monkeypatch)[0]
        assert venv_cmd[0] == str(exe), (
            f"the install ran {venv_cmd[0]!r}, which is not the uv that exists")
        assert venv_cmd[1:3] == ["venv", "--python"]

    def test_no_step_invokes_a_bare_uv(self, machine, tmp_path, monkeypatch):
        """Swept across every step, not only the one that was reported: a bare
        'uv' resolves through PATH, which is exactly where it is not."""
        self._portable_uv(machine, tmp_path)
        for cmd in self._commands(
                machine, monkeypatch,
                machine.Plan(app_window=True, add_to_path=True, shortcut="gui")):
            assert cmd[0] != "uv", f"{cmd} searches PATH for uv"

    def test_the_entry_check_and_the_steps_agree(
            self, machine, tmp_path, monkeypatch):
        """One resolver, so a uv good enough to OPEN the window is always a uv
        the steps can run."""
        exe = self._portable_uv(machine, tmp_path)
        assert machine.find_uv(tmp_path) == str(exe)
        assert self._commands(machine, monkeypatch)[0][0] == machine.find_uv(tmp_path)



    def test_the_portable_uv_wins_over_one_on_path(
            self, machine, tmp_path, monkeypatch):
        """Portable means the install uses its own copy, not the machine's."""
        exe = self._portable_uv(machine, tmp_path)
        onpath = tmp_path / "sysbin"
        onpath.mkdir()
        (onpath / ("uv.exe" if machine.IS_WINDOWS else "uv")).write_bytes(b"")
        monkeypatch.setenv("PATH", str(onpath))
        assert machine.find_uv(tmp_path) == str(exe)


    def test_a_machine_with_no_uv_reports_it_instead_of_running_nothing(
            self, machine, tmp_path, monkeypatch):
        """No uv anywhere is a step that FAILS, never a command handed to the
        OS that cannot start (AGENTS.md rule 5)."""
        assert machine.find_uv(tmp_path) is None
        with pytest.raises(machine.StepFailed) as excinfo:
            self._commands(machine, monkeypatch)
        assert "uv was not found" in str(excinfo.value)

    def test_a_uv_only_on_path_is_still_found(
            self, machine, tmp_path, monkeypatch):
        """The launcher puts the uv it used on PATH for this process, so one
        that lives nowhere the search looks is still reachable."""
        onpath = tmp_path / "sysbin"
        onpath.mkdir()
        exe = onpath / ("uv.exe" if machine.IS_WINDOWS else "uv")
        exe.write_bytes(b"")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", str(onpath))
        found = machine.find_uv(tmp_path)
        assert found is not None
        assert self._commands(machine, monkeypatch)[0][0] == found


# --------------------------------------------------------------------------- #
#  The same install the console performs                                       #
# --------------------------------------------------------------------------- #

def _commands_for(gui, monkeypatch, plan, spec="torch"):
    """Every command the install would run, without running any of them."""
    seen = []

    def fake_run(cmd, emit, plan, **kw):
        seen.append([str(c) for c in cmd])
        return 0

    (gui.ROOT / ".venv").mkdir(exist_ok=True)
    monkeypatch.setattr(gui, "_run", fake_run)
    monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
    monkeypatch.setattr(gui, "torch_spec_for", lambda b: (spec, ""))
    monkeypatch.setattr(gui, "_query", lambda args: "queried")
    for step in gui.build_steps(plan):
        try:
            step.run(lambda s: None)
        except gui.StepFailed:
            pass
    return seen


def _labels(gui, plan):
    return [s.label for s in gui.build_steps(plan)]


class TestInstallParity:
    """setup.bat and setup.sh do these; the window claims to do the same."""

    def test_the_venv_marker_is_written(self, gui, tmp_path, monkeypatch):
        """uninstall removes .venv only when this marker says setup made it."""
        (tmp_path / ".venv").mkdir()
        monkeypatch.setattr(gui, "_run", lambda *a, **k: 0)
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        step = _step(gui, gui.Plan(), "Creating the Python environment")
        step.run(lambda s: None)
        assert (tmp_path / ".venv" / ".localm-venv").is_file()

    def test_the_chosen_plugins_are_installed(self, gui, tmp_path, monkeypatch):
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(plugins=("coder", "rag"), backend="own"))
        setup = [c for c in cmds if "plugin" in c and "setup" in c]
        assert len(setup) == 1, "the chosen features are never installed"
        assert "--plugins" in setup[0]
        assert setup[0][setup[0].index("--plugins") + 1] == "coder,rag"

    def test_no_plugin_step_when_none_were_chosen(self, gui):
        assert not any("optional features" in lb
                       for lb in _labels(gui, gui.Plan(plugins=())))

    def test_the_deps_answer_is_always_explicit(self, gui, tmp_path, monkeypatch):
        """The flag's default is to ASK, and a window has no console to answer."""
        for deps, flag in ((True, "--with-deps"), (False, "--no-deps")):
            cmds = _commands_for(gui, monkeypatch,
                                 gui.Plan(plugins=("coder",), plugin_deps=deps,
                                          backend="own"))
            setup = [c for c in cmds if "plugin" in c and "setup" in c][0]
            assert flag in setup, f"{setup} lets the deps prompt decide"

    def test_the_global_command_never_waits_for_an_answer(
            self, gui, tmp_path, monkeypatch):
        """globalcmd install prompts on a PATH conflict; nothing can reply."""
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(add_to_path=True, backend="own"))
        gc = [c for c in cmds if "localm.globalcmd" in c]
        assert gc and "--yes" in gc[0], f"{gc} can block on stdin"

    def test_the_install_is_recorded(self, gui, tmp_path, monkeypatch):
        """Without this, uninstall cannot remove what setup created."""
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", shortcut="none"))
        rec = [c for c in cmds if "localm.install_manifest" in c]
        assert len(rec) == 1, "nothing records what was installed"
        assert "record" in rec[0]
        assert str(tmp_path / ".venv") in rec[0]
        assert rec[0][rec[0].index("--data-dir") + 1] == str(tmp_path / "home")

    def test_the_recorded_shortcut_is_the_one_created(
            self, gui, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "make_shortcut",
                            lambda plan, emit: str(tmp_path / "LocaLM.lnk"))
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", shortcut="launcher"))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert rec[rec.index("--shortcut") + 1] == str(tmp_path / "LocaLM.lnk")

    def test_a_portable_install_records_its_tooling_dirs(
            self, gui, tmp_path, monkeypatch):
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=True))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" in rec
        assert rec[rec.index("--python-dir") + 1] == str(tmp_path / ".python")

    def test_a_shared_install_does_not_claim_the_shared_dirs(
            self, gui, tmp_path, monkeypatch):
        """Uninstall must never delete a runtime other installs share."""
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=False))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" not in rec


class TestHonestFailures:
    """AGENTS.md rule 5: a step that failed is never reported as done."""

    def test_an_unanswerable_torch_probe_is_not_no_torch_needed(
            self, gui, monkeypatch):
        """The probe exiting non-zero says nothing about needing PyTorch."""
        class Failed:
            returncode = 3
            stdout = ""
            stderr = "ModuleNotFoundError: No module named 'localm'"
        monkeypatch.setattr(gui.subprocess, "run", lambda *a, **k: Failed())
        spec, problem = gui.torch_spec_for("cuda")
        assert spec is None
        assert "could not ask" in problem
        step = _step(gui, gui.Plan(backend="cuda"), "Installing PyTorch")
        with pytest.raises(gui.StepFailed) as e:
            step.run(lambda s: None)
        assert "could not ask" in str(e.value)

    def test_a_genuinely_torch_free_backend_still_says_so(
            self, gui, monkeypatch):
        monkeypatch.setattr(gui, "torch_spec_for", lambda b: (None, ""))
        lines = []
        _step(gui, gui.Plan(backend="vulkan"), "Installing PyTorch").run(lines.append)
        assert any("No PyTorch stack needed" in line for line in lines)

    def test_a_failed_torch_install_reaches_the_summary(self, gui, monkeypatch):
        """It used to scroll past in the log and end on an unqualified success."""
        monkeypatch.setattr(gui, "torch_spec_for", lambda b: ("torch", ""))
        monkeypatch.setattr(gui, "_run", lambda *a, **k: 1)
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        with pytest.raises(gui.StepFailed) as e:
            _step(gui, gui.Plan(backend="cuda"), "Installing PyTorch").run(
                lambda s: None)
        assert "PyTorch" in str(e.value)

    def test_a_failed_path_step_reaches_the_summary(self, gui, monkeypatch):
        monkeypatch.setattr(gui, "_run", lambda *a, **k: 1)
        monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
        with pytest.raises(gui.StepFailed):
            _step(gui, gui.Plan(add_to_path=True, backend="own"),
                  "Adding 'localm'").run(lambda s: None)


class TestBackendMenu:
    """hwdetect can recommend metal and hip, and setup_llama accepts both."""

    def test_every_offered_backend_is_one_the_provisioner_accepts(self, gui):
        from localm import setup_llama
        for key, _ in gui._BACKEND_CHOICES:
            if key == "own":
                continue
            assert key in setup_llama.BACKENDS, f"{key} is not a real backend"

    def test_metal_is_offered_on_macos_and_nowhere_else(self, gui, monkeypatch):
        monkeypatch.setattr(gui.sys, "platform", "darwin")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        assert "metal" in [k for k, _ in gui.backend_choices()]
        monkeypatch.setattr(gui.sys, "platform", "win32")
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        assert "metal" not in [k for k, _ in gui.backend_choices()]

    def test_hip_is_offered(self, gui):
        """An AMD box with a ROCm toolkit is recommended hip by hwdetect."""
        assert "hip" in [k for k, _ in gui.backend_choices()]

    def test_the_menu_can_show_whatever_hwdetect_recommends(self, gui, monkeypatch):
        """A recommendation with no matching row leaves the group unselected."""
        monkeypatch.setattr(gui.sys, "platform", "darwin")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        mac = {k for k, _ in gui.backend_choices()}
        monkeypatch.setattr(gui.sys, "platform", "win32")
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        win = {k for k, _ in gui.backend_choices()}
        from localm import hwdetect
        import inspect
        import re
        src = inspect.getsource(hwdetect.recommended_install_backend)
        for rec in set(re.findall(r'return "([a-z-]+)"', src)):
            assert rec in mac or rec in win, f"nothing offers {rec}"


class TestPluginChoices:
    def test_the_picker_reads_the_real_catalog(self, gui):
        from localm.plugins import catalog
        offered = [n for n, _ in gui.plugin_choices()]
        assert offered, "no optional features are offered"
        assert set(offered) == set(catalog.names()) - set(catalog.preinstalled())

    def test_chat_is_never_offered_because_it_is_always_installed(self, gui):
        assert "chat" not in [n for n, _ in gui.plugin_choices()]

    def test_the_recommended_set_matches_the_console_installer(self, gui):
        from localm.cli import plugins as cli_plugins
        assert set(gui.RECOMMENDED_PLUGINS) == set(cli_plugins._SETUP_DEFAULTS)


class TestPosixShortcut:
    def test_the_launcher_choice_is_honoured(self, gui, tmp_path, monkeypatch):
        """It used to write the same GUI entry whatever the user picked."""
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        gui.make_shortcut(gui.Plan(shortcut="launcher"), lambda s: None)
        launcher = (home / "Desktop" / "LocaLM.desktop").read_text(encoding="utf-8")
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        straight = (home / "Desktop" / "LocaLM.desktop").read_text(encoding="utf-8")
        assert launcher != straight, "both choices wrote the same entry"
        assert "localm-launcher.sh" in launcher
        assert launcher.count("Exec=") == 1

    def test_the_written_path_is_returned_for_the_manifest(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        written = gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        assert written and Path(written).is_file()


# --------------------------------------------------------------------------- #
#  The dialogue                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def wizard(gui):
    """A real Tk wizard, skipped where there is no display to build one on."""
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    try:
        yield gui.Wizard(root, tk, ttk, filedialog)
    finally:
        root.destroy()


class TestWizard:
    """It is a multi-page dialogue, so every question actually gets asked."""

    def test_it_has_a_page_per_group_of_questions(self, wizard):
        titles = [t for t, _ in wizard.pages]
        assert titles == ["Inference runtime", "Where things live",
                          "Optional features", "Options", "Installing"]

    def test_next_and_back_walk_the_pages(self, wizard):
        assert wizard.index == 0
        wizard.next_page()
        assert wizard.index == 1
        wizard.next_page()
        assert wizard.index == 2
        wizard.prev_page()
        assert wizard.index == 1
        wizard.prev_page()
        assert wizard.index == 0

    def test_back_does_nothing_on_the_first_page(self, wizard):
        wizard.prev_page()
        assert wizard.index == 0

    def test_the_last_question_page_installs_rather_than_advancing(
            self, wizard, monkeypatch):
        started = []
        monkeypatch.setattr(wizard, "start_install", lambda: started.append(True))
        for _ in range(len(wizard.pages)):
            wizard.next_page()
        assert started, "the dialogue never reaches the install"

    def test_an_empty_custom_data_folder_is_refused_without_advancing(
            self, wizard):
        wizard.next_page()
        assert wizard.pages[wizard.index][0] == "Where things live"
        wizard.portable_var.set(False)
        wizard.path_var.set("")
        wizard.next_page()
        assert wizard.index == 1, "advanced with no data folder chosen"
        assert "folder" in wizard.status.cget("text")

    def test_an_answer_survives_leaving_its_page(self, wizard):
        wizard.backend_var.set("cpu")
        wizard.next_page()
        wizard.prev_page()
        assert wizard.backend_var.get() == "cpu"
        assert wizard.current_plan().backend == "cpu"

    def test_every_page_feeds_the_plan(self, wizard):
        wizard.backend_var.set("cpu")
        wizard.store_var.set(False)
        wizard.appwin_var.set(True)
        wizard.path_cmd_var.set(True)
        wizard.shortcut_var.set("gui")
        wizard.deps_var.set(False)
        for name in wizard.plugin_vars:
            wizard.plugin_vars[name].set(False)
        if "coder" in wizard.plugin_vars:
            wizard.plugin_vars["coder"].set(True)
        plan = wizard.current_plan()
        assert plan.backend == "cpu"
        assert plan.portable_store is False
        assert plan.app_window is True
        assert plan.add_to_path is True
        assert plan.shortcut == "gui"
        assert plan.plugin_deps is False
        assert plan.plugins == ("coder",)

    def test_the_default_plan_matches_the_console_recommendation(self, wizard, gui):
        assert set(wizard.current_plan().plugins) == set(gui.RECOMMENDED_PLUGINS)
