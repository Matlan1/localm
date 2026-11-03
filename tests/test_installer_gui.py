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
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

_GUI_PATH = Path(__file__).resolve().parents[1] / "installer" / "gui.py"
_REPO_ROOT = _GUI_PATH.parents[1]
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
    """The installer module with its ROOT pointed at a throwaway directory, and
    the Shared store (LOCALAPPDATA, XDG_DATA_HOME) at another, so no test can
    write into the real clone or the real user profile."""
    mod = _load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    profile = tmp_path.parent / f"{tmp_path.name}-profile"
    monkeypatch.setenv("LOCALAPPDATA", str(profile))
    monkeypatch.setenv("XDG_DATA_HOME", str(profile))
    _fake_system32(mod, tmp_path, monkeypatch)
    for name in (mod.STORE_ENV, mod.UV_ENV, mod.UV_INSTALLED_ENV):
        monkeypatch.delenv(name, raising=False)
    return mod


def _shared_store(gui):
    return gui.shared_store_root()


def _record(root):
    import json
    return json.loads((root / ".localm-install.json").read_text(encoding="utf-8"))


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


@pytest.mark.parametrize("plat, expected", [("darwin", False), ("linux", True)])
def test_macos_gets_no_launcher_or_shortcut(gui, monkeypatch, plat, expected):
    monkeypatch.setattr(gui.sys, "platform", plat)
    labels = [s.label for s in gui.build_steps(gui.Plan(shortcut="gui"))]
    assert ("Building the launcher" in labels) is expected
    assert any("shortcut" in l for l in labels) is expected


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
    assert (tmp_path / "localm-home.cfg").read_text(encoding="utf-8").strip() == str(target)
    rec = _record(tmp_path)
    assert Path(rec["data_dir"]) == target and rec["data_created"] is True


def test_an_existing_custom_folder_is_recorded_as_existing(gui, tmp_path):
    """A folder that was already there is not the install's to delete; only
    the entries LocaLM adds to it are."""
    shared = tmp_path / "Shared models"
    (shared / "models").mkdir(parents=True)
    _step(gui, gui.Plan(portable_data=False, data_path=str(shared)),
          "Recording").run(lambda _l: None)
    rec = _record(tmp_path)
    assert rec["data_created"] is False
    assert rec["data_preexisting"] == ["models"]


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


def test_the_uv_being_used_is_on_the_path_the_steps_inherit(gui, tmp_path,
                                                            monkeypatch):
    """localm's own plugin dependency installer shells out to a bare uv, and a
    portable copy is not on PATH. Its fallback cannot help: an environment uv
    created has no pip."""
    d = tmp_path / ".uv"
    d.mkdir()
    exe = d / ("uv.exe" if gui.IS_WINDOWS else "uv")
    exe.write_bytes(b"")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    env = gui._env_for(gui.Plan())
    first = env["PATH"].split(os.pathsep)[0]
    assert first == str(d), env["PATH"]


def test_no_uv_leaves_the_path_alone(gui, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    env = gui._env_for(gui.Plan())
    assert env["PATH"] == str(tmp_path / "nothing")


@pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
def test_a_poisoned_inherited_environment_never_reaches_a_step(gui, tmp_path,
                                                              monkeypatch, portable):
    """Whatever uv, pip and Python settings the PC has, a step sees only
    localm's own: the inherited ones are dropped, D2/D3/D4 values are set."""
    poison = {
        "UV_INDEX_URL": "https://evil.example/simple",
        "PIP_INDEX_URL": "https://evil.example/simple",
        "PIP_EXTRA_INDEX_URL": "https://evil.example/extra",
        "PYTHONPATH": str(tmp_path / "evil"),
        "PYTHONHOME": str(tmp_path / "evilhome"),
        "PYTHONSTARTUP": str(tmp_path / "evil.py"),
        "PYTHONUSERBASE": str(tmp_path / "eviluser"),
        "VIRTUAL_ENV": str(tmp_path / "othervenv"),
        "CONDA_PREFIX": str(tmp_path / "conda"),
        "UV_CACHE_DIR": str(tmp_path / "usercache"),
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "userpythons"),
        "UV_PYTHON": str(tmp_path / "python.exe"),
        "UV_PYTHON_PREFERENCE": "only-system",
        "UV_NO_CACHE": "1",
        "UV_CONFIG_FILE": str(tmp_path / "uv.toml"),
        "UV_PYTHON_INSTALL_MIRROR": "https://evil.example/python",
        "UV_PYTHON_DOWNLOADS_JSON_URL": "https://evil.example/pythons.json",
        "UV_CREDENTIALS_DIR": str(tmp_path / "usercreds"),
        "LOCALM_HOME": str(tmp_path / "otherhome"),
        "TEMP": str(tmp_path / "systemp"), "TMP": str(tmp_path / "systemp"),
        "TMPDIR": str(tmp_path / "systemp"),
    }
    for key, value in poison.items():
        monkeypatch.setenv(key, value)
    env = gui._env_for(gui.Plan(portable_store=portable))
    store = gui.store_dirs(portable)
    for key in ("UV_INDEX_URL", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PYTHONPATH",
                "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV",
                "CONDA_PREFIX", "UV_PYTHON", "UV_NO_CACHE", "UV_CONFIG_FILE",
                "UV_PYTHON_INSTALL_MIRROR", "UV_PYTHON_DOWNLOADS_JSON_URL", "LOCALM_HOME"):
        assert key not in env, f"{key} from the PC reached a step"
    assert env["UV_CACHE_DIR"] == str(store["cache"])
    assert env["UV_PYTHON_INSTALL_DIR"] == str(store["python"])
    assert env["UV_CREDENTIALS_DIR"] == str(store["uv"] / "credentials")
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert env["UV_PYTHON_INSTALL_BIN"] == env["UV_PYTHON_INSTALL_REGISTRY"] == "0"
    assert env["UV_NO_CONFIG"] == "1"
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PYTHONNOUSERSITE"] == "1"
    tmp = str(tmp_path / ".localm-setup-tmp")
    assert env["TEMP"] == env["TMP"] == env["TMPDIR"] == tmp
    if portable:
        assert store["cache"] == tmp_path / ".cache"
    else:
        assert store["cache"] == _shared_store(gui) / "cache"
        assert _shared_store(gui) == Path(os.environ["LOCALAPPDATA" if gui.IS_WINDOWS
                                                     else "XDG_DATA_HOME"]) / "localm" / "shared"


def test_inherited_names_are_dropped_whatever_their_case(gui):
    env = gui.isolated_environ({"uv_index_url": "x", "Pip_Index_Url": "y",
                                "PythonPath": "z", "Path": "keep", "LOCALM_GUI_STORE": "shared"})
    assert set(env) == {"Path", "LOCALM_GUI_STORE", "UV_NO_CONFIG", "PIP_CONFIG_FILE",
                        "PYTHONNOUSERSITE"}


def test_shared_without_a_profile_folder_fails_the_step(gui, monkeypatch):
    """No LOCALAPPDATA on Windows: the Shared store is not guessed."""
    monkeypatch.setattr(gui, "IS_WINDOWS", True)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    with pytest.raises(gui.StepFailed, match="LOCALAPPDATA"):
        gui._env_for(gui.Plan(portable_store=False))


def test_portable_store_contains_uv_inside_the_install(gui, tmp_path):
    """Portable: uv's managed interpreter and its wheel cache land inside the
    install. Shared: in localm's own folder in the profile, never the folder."""
    env = gui._env_for(gui.Plan(portable_store=True))
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / ".python")
    assert env["UV_CACHE_DIR"] == str(tmp_path / ".cache")

    plain = gui._env_for(gui.Plan(portable_store=False))
    assert plain["UV_PYTHON_INSTALL_DIR"] == str(_shared_store(gui) / "python")
    assert plain["UV_CACHE_DIR"] == str(_shared_store(gui) / "cache")


# --------------------------------------------------------------------------- #
#  The native certificate store                                                #
# --------------------------------------------------------------------------- #

def test_the_native_certificate_store_is_used_like_the_console_installers(
        gui, monkeypatch):
    monkeypatch.delenv("UV_SYSTEM_CERTS", raising=False)
    env = gui._env_for(gui.Plan())
    assert env.get("UV_SYSTEM_CERTS") == "1"


def test_the_wrappers_set_the_native_certificate_store_before_opening_the_window():
    bat = (_REPO_ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    sh = (_REPO_ROOT / "setup-gui.sh").read_text(encoding="utf-8")
    bat_run = bat.index('"%UVEXE%" run')
    bat_set = bat.index('set "UV_SYSTEM_CERTS=1"')
    assert bat_set < bat_run, "setup-gui.bat opens the window before the store is set"
    sh_run = sh.index('"$UVEXE" run')
    sh_set = sh.index("export UV_SYSTEM_CERTS=1")
    assert sh_set < sh_run, "setup-gui.sh opens the window before the store is set"


def test_setup_bat_still_sets_the_native_certificate_store():
    text = (_REPO_ROOT / "setup.bat").read_text(encoding="utf-8")
    assert re.search(r'(?m)^set "UV_SYSTEM_CERTS=1"\s*$', text)


# --------------------------------------------------------------------------- #
#  Finding uv                                                                  #
# --------------------------------------------------------------------------- #

class TestUvResolution:
    """Every uv command is localm's own uv in the store the tooling answer
    picked; a uv on PATH or in the user profile is never run."""

    @pytest.fixture()
    def machine(self, gui, tmp_path, monkeypatch):
        """A machine with a uv on PATH and in the user profile only."""
        home = tmp_path / "home"
        for d in (home / ".local" / "bin", home / ".cargo" / "bin", tmp_path / "sysbin"):
            self._uv_in(gui, d)
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("PATH", str(tmp_path / "sysbin"))
        return gui

    @staticmethod
    def _uv_in(gui, d):
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
            if [str(c) for c in cmd][1:2] == ["venv"]:
                (gui.ROOT / ".venv").mkdir(exist_ok=True)
            return 0

        monkeypatch.setattr(gui, "_run", fake_run)
        monkeypatch.setattr(gui, "make_shortcut",
                            lambda plan, emit: str(gui.ROOT / "LocaLM.lnk"))
        monkeypatch.setattr(gui, "torch_spec_for", lambda backend: ("torch", ""))
        for step in gui.build_steps(plan or gui.Plan()):
            try:
                step.run(lambda s: None)
            except gui.StepFailed:
                raise
            except Exception:
                pass
        return seen

    def test_the_steps_run_the_portable_uv(self, machine, tmp_path, monkeypatch):
        exe = self._uv_in(machine, tmp_path / ".uv")
        venv_cmd = self._commands(machine, monkeypatch)[0]
        assert venv_cmd[0] == str(exe)
        assert venv_cmd[1:3] == ["venv", "--python"]

    def test_the_steps_run_the_shared_store_uv(self, machine, tmp_path, monkeypatch):
        self._uv_in(machine, tmp_path / ".uv")
        exe = self._uv_in(machine, _shared_store(machine) / "uv")
        for cmd in self._commands(machine, monkeypatch, machine.Plan(portable_store=False)):
            if cmd[1:2] in (["venv"], ["pip"]):
                assert cmd[0] == str(exe), cmd

    def test_no_step_invokes_a_bare_uv(self, machine, tmp_path, monkeypatch):
        self._uv_in(machine, tmp_path / ".uv")
        for cmd in self._commands(
                machine, monkeypatch,
                machine.Plan(app_window=True, add_to_path=True, shortcut="gui")):
            assert os.path.isabs(cmd[0]), f"{cmd} searches PATH for {cmd[0]}"

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_a_uv_on_path_or_in_the_profile_is_never_found(
            self, machine, tmp_path, monkeypatch, portable):
        assert machine.find_uv(tmp_path, portable) is None
        with pytest.raises(machine.StepFailed) as excinfo:
            self._commands(machine, monkeypatch, machine.Plan(portable_store=portable))
        assert "localm's own uv was not found" in str(excinfo.value)
        assert str(machine.store_dirs(portable)["uv"]) in str(excinfo.value)

    def test_the_launchers_uv_is_never_taken_on_its_word(
            self, machine, tmp_path, monkeypatch):
        monkeypatch.setenv(machine.UV_ENV, str(tmp_path / "sysbin" / "uv.exe"))
        assert machine.find_uv(tmp_path) is None
        exe = self._uv_in(machine, tmp_path / ".uv")
        assert machine.find_uv(tmp_path) == str(exe)


# --------------------------------------------------------------------------- #
#  The same install the console performs                                       #
# --------------------------------------------------------------------------- #

def _commands_for(gui, monkeypatch, plan, spec="torch", code_for=None):
    """Every command the install would run, without running any of them.

    code_for maps a substring of the command to the exit code it returns, so a
    test can drive one step's outcome without affecting the others."""
    seen = []

    def fake_run(cmd, emit, plan, **kw):
        flat = [str(c) for c in cmd]
        seen.append(flat)
        if flat[1:2] == ["venv"]:
            (gui.ROOT / ".venv").mkdir(exist_ok=True)
        for needle, code in (code_for or {}).items():
            if any(needle in part for part in flat):
                return code
        return 0

    monkeypatch.setattr(gui, "_run", fake_run)
    monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
    # make_shortcut does NOT go through _run: it shells out to PowerShell on
    # Windows and writes under Path.home() elsewhere, so stubbing _run alone
    # lets it touch the real machine.
    monkeypatch.setattr(gui, "make_shortcut",
                        lambda plan, emit: str(gui.ROOT / "LocaLM.lnk"))
    monkeypatch.setattr(gui, "torch_spec_for", lambda b: (spec, ""))
    monkeypatch.setattr(gui, "_query", lambda args: "queried")
    for step in gui.build_steps(plan):
        try:
            step.run(lambda s: None)
        except gui.StepFailed:
            pass
    return seen


def _creates_venv(root):
    """A _run stand-in for which every `uv venv` creates root/.venv."""
    def fake_run(cmd, emit, plan, **kw):
        if [str(c) for c in cmd][1:2] == ["venv"]:
            (root / ".venv").mkdir(exist_ok=True)
        return 0
    return fake_run


def _labels(gui, plan):
    return [s.label for s in gui.build_steps(plan)]


class TestInstallParity:
    """setup.bat and setup.sh do these; the window claims to do the same."""

    def test_the_venv_marker_is_written(self, gui, tmp_path, monkeypatch):
        """uninstall removes .venv only when this marker says setup made it."""
        monkeypatch.setattr(gui, "_run", _creates_venv(tmp_path))
        monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
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
        # The data step recorded the folder itself; the final record must not
        # overwrite what it found.
        assert "--data-dir" not in rec[0] and "--data-created" not in rec[0]
        assert Path(_record(tmp_path)["data_dir"]) == tmp_path / "home"

    def test_the_environment_is_recorded_as_soon_as_it_exists(
            self, gui, tmp_path, monkeypatch):
        """An install that stops later can then still be uninstalled."""
        (tmp_path / ".python").mkdir()
        monkeypatch.setattr(gui, "_run", _creates_venv(tmp_path))
        monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
        _step(gui, gui.Plan(), "Creating the Python environment").run(lambda s: None)
        rec = _record(tmp_path)
        assert Path(rec["venv"]) == (tmp_path / ".venv").resolve()
        assert rec["runtime_contained"] is True
        assert Path(rec["python_dir"]) == (tmp_path / ".python").resolve()

    def test_the_linux_launcher_entry_is_recorded(self, gui, tmp_path, monkeypatch):
        """make-launcher writes LocaLM.desktop into the folder on Linux."""
        (tmp_path / "LocaLM.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
        cmds = _commands_for(gui, monkeypatch, gui.Plan(backend="own"))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert rec[rec.index("--file") + 1] == str(tmp_path / "LocaLM.desktop")

    def test_the_recorded_shortcut_is_the_one_created(
            self, gui, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "make_shortcut",
                            lambda plan, emit: str(tmp_path / "LocaLM.lnk"))
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", shortcut="launcher"))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert rec[rec.index("--shortcut") + 1] == str(tmp_path / "LocaLM.lnk")

    def test_the_portable_home_is_recorded_even_when_it_already_existed(
            self, gui, tmp_path, monkeypatch):
        """An earlier step can create ./home first; ./home is the install's own
        folder either way."""
        (tmp_path / "home").mkdir()
        _commands_for(gui, monkeypatch, gui.Plan(backend="own"))
        assert Path(_record(tmp_path)["data_dir"]) == tmp_path / "home"
        assert (tmp_path / "home" / ".localm-data").is_file()

    def test_the_tooling_inside_the_folder_is_recorded(
            self, gui, tmp_path, monkeypatch):
        """On Portable the tooling lives inside the folder, so uninstall has to
        know about it."""
        (tmp_path / ".python").mkdir()
        (tmp_path / ".uv").mkdir()
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=True))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" in rec
        assert rec[rec.index("--python-dir") + 1] == str(tmp_path / ".python")
        assert rec[rec.index("--uv-dir") + 1] == str(tmp_path / ".uv")
        assert "--cache-dir" not in rec, "a directory that does not exist"

    def test_a_shared_install_never_records_tooling_left_in_the_folder(
            self, gui, tmp_path, monkeypatch):
        """A .python or .uv left by an earlier Portable install is not claimed by
        a Shared one."""
        (tmp_path / ".python").mkdir()
        (tmp_path / ".uv").mkdir()
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=False))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" not in rec
        assert str(tmp_path / ".python") not in rec and str(tmp_path / ".uv") not in rec

    def test_a_shared_install_records_the_shared_store_as_shared(
            self, gui, tmp_path, monkeypatch):
        """The store folders are recorded (D7) without --runtime-contained, which
        uninstall keeps as used by other installs."""
        store = gui.store_dirs(False)
        for d in store.values():
            d.mkdir(parents=True)
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=False))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" not in rec
        assert rec[rec.index("--python-dir") + 1] == str(store["python"])
        assert rec[rec.index("--cache-dir") + 1] == str(store["cache"])
        assert rec[rec.index("--uv-dir") + 1] == str(store["uv"])

    def test_nothing_outside_the_folder_is_ever_claimed(
            self, gui, tmp_path, monkeypatch):
        """Uninstall must never delete a runtime other installs share: a Shared
        install never records --runtime-contained."""
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(backend="own", portable_store=False))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--runtime-contained" not in rec

    def test_the_hf_stack_comes_from_the_hf_extra_not_an_inline_pin(
            self, gui, monkeypatch):
        """A non-gfx103x backend's torch install must resolve the HF stack
        from pyproject's [hf] extra, matching setup.sh/setup.bat, never from
        its own inline transformers[ specifier."""
        cmds = _commands_for(gui, monkeypatch, gui.Plan(backend="cuda"),
                             spec="torch torchvision --index-url https://x")
        hf_install = [c for c in cmds if any(".[hf" in part for part in c)]
        assert len(hf_install) == 1, f"expected one [hf] install, found {hf_install}"
        assert ".[hf,audio]" in hf_install[0]
        assert not any(part.startswith("transformers[") for part in hf_install[0]), (
            f"an inline transformers[ specifier is back: {hf_install[0]}")


class TestGlobalCommandExitCodes:
    """globalcmd exit 20 means the command WAS created and its directory was
    already on PATH. Both console installers record the shim on 20 and add
    --path-modified only on 0."""

    def test_the_command_is_recorded_when_path_already_had_its_directory(
            self, gui, tmp_path, monkeypatch):
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(add_to_path=True, backend="own"),
                             code_for={"localm.globalcmd": 20})
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert rec[rec.index("--command-shim") + 1] == "queried", (
            "the shim was created and is not recorded, so uninstall leaves it")
        assert "--path-modified" not in rec, (
            "exit 20 did not change PATH, so nothing should claim it did")

    def test_a_path_change_is_recorded_on_zero(self, gui, tmp_path, monkeypatch):
        cmds = _commands_for(gui, monkeypatch,
                             gui.Plan(add_to_path=True, backend="own"),
                             code_for={"localm.globalcmd": 0})
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert "--path-modified" in rec

    def test_a_real_failure_still_reaches_the_summary(
            self, gui, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "_run", lambda *a, **k: 1)
        monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
        with pytest.raises(gui.StepFailed):
            _step(gui, gui.Plan(add_to_path=True, backend="own"),
                  "Adding 'localm'").run(lambda s: None)


class TestToolingLocation:
    """Both choices build the environment on uv's managed Python only."""

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_the_environment_is_built_on_a_managed_python(
            self, gui, tmp_path, monkeypatch, portable):
        venv_cmd = _commands_for(gui, monkeypatch,
                                 gui.Plan(backend="own", portable_store=portable))[0]
        assert venv_cmd[1] == "venv"
        assert venv_cmd[venv_cmd.index("--python-preference") + 1] == "only-managed"


class TestHonestFailures:
    """A step that failed is never reported as done."""

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
        monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
        with pytest.raises(gui.StepFailed) as e:
            _step(gui, gui.Plan(backend="cuda"), "Installing PyTorch").run(
                lambda s: None)
        assert "PyTorch" in str(e.value)

    def test_a_failed_path_step_reaches_the_summary(self, gui, monkeypatch):
        monkeypatch.setattr(gui, "_run", lambda *a, **k: 1)
        monkeypatch.setattr(gui, "find_uv", lambda root, portable=None: "uv")
        with pytest.raises(gui.StepFailed):
            _step(gui, gui.Plan(add_to_path=True, backend="own"),
                  "Adding 'localm'").run(lambda s: None)


class TestBackendMenu:
    """hwdetect can recommend metal and hip, and setup_llama accepts both."""

    def test_every_offered_backend_is_one_the_provisioner_accepts(self, gui):
        from localm import setup_llama
        for key, _ in gui.backend_choices():
            if key == "own":
                continue
            assert key in setup_llama.BACKENDS, f"{key} is not a real backend"

    def test_metal_is_offered_on_macos_and_nowhere_else(self, gui, monkeypatch):
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "arm64")
        monkeypatch.setattr(gui.sys, "platform", "darwin")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        assert "metal" in [k for k, _ in gui.backend_choices()]
        monkeypatch.setattr(gui.sys, "platform", "win32")
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        assert "metal" not in [k for k, _ in gui.backend_choices()]

    @pytest.mark.parametrize("arch, expected", [
        ("arm64", ["metal", "cpu", "own"]),
        ("x64", ["cpu", "own"]),
    ])
    def test_macos_offers_only_backends_with_a_macos_build(self, gui, monkeypatch,
                                                           arch, expected):
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: arch)
        monkeypatch.setattr(gui.sys, "platform", "darwin")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        assert [k for k, _ in gui.backend_choices()] == expected

    def test_arm64_linux_offers_only_backends_with_an_arm64_build(self, gui, monkeypatch):
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "arm64")
        monkeypatch.setattr(gui.sys, "platform", "linux")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        assert [k for k, _ in gui.backend_choices()] == ["vulkan", "cuda", "cpu", "own"]

    def test_x64_menus_are_unchanged(self, gui, monkeypatch):
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "x64")
        monkeypatch.setattr(gui.sys, "platform", "linux")
        assert [k for k, _ in gui.backend_choices()] == [
            "vulkan", "cuda", "hip", "sycl", "cpu", "own"]
        monkeypatch.setattr(gui.sys, "platform", "win32")
        assert [k for k, _ in gui.backend_choices()] == [
            "vulkan", "cuda", "hip", "amd-rocm", "sycl", "cpu", "own"]

    def test_hip_is_offered(self, gui):
        """An AMD box with a ROCm toolkit is recommended hip by hwdetect."""
        assert "hip" in [k for k, _ in gui.backend_choices()]

    def test_sycl_is_offered_on_x64_windows_and_linux(self, gui, monkeypatch):
        # sycl has an upstream x64 asset for both win32 and linux, so it is
        # offered on both; it is not auto-recommended on Linux.
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "x64")
        monkeypatch.setattr(gui.sys, "platform", "linux")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        assert "sycl" in [k for k, _ in gui.backend_choices()]
        monkeypatch.setattr(gui.sys, "platform", "win32")
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        assert "sycl" in [k for k, _ in gui.backend_choices()]

    def test_the_menu_can_show_whatever_hwdetect_recommends(self, gui, monkeypatch):
        """A recommendation with no matching row leaves the group unselected."""
        from localm import hwdetect
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "arm64")
        monkeypatch.setattr(gui.sys, "platform", "darwin")
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        mac = {k for k, _ in gui.backend_choices()}
        monkeypatch.setattr(hwdetect, "machine_arch", lambda: "x64")
        monkeypatch.setattr(gui.sys, "platform", "win32")
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        win = {k for k, _ in gui.backend_choices()}
        import inspect
        import re
        src = (inspect.getsource(hwdetect.recommended_install_backend)
               + inspect.getsource(hwdetect._hardware_install_backend))
        for rec in set(re.findall(r'return "([a-z-]+)"', src)):
            assert rec in mac or rec in win, f"nothing offers {rec}"


class TestPluginChoices:
    def test_the_picker_reads_the_real_catalog(self, gui):
        from localm.plugins import catalog
        offered = [n for n, _ in gui.plugin_choices()]
        assert offered, "no optional features are offered"
        assert offered == [n for n in catalog.names()
                           if n not in catalog.preinstalled() and catalog.offered(n)]
        assert "exllama" not in offered

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
        entry = home / ".local/share/applications" / gui.MENU_ENTRY
        gui.make_shortcut(gui.Plan(shortcut="launcher"), lambda s: None)
        launcher = entry.read_text(encoding="utf-8")
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        straight = entry.read_text(encoding="utf-8")
        assert launcher != straight, "both choices wrote the same entry"
        assert "localm-launcher.sh" in launcher
        assert launcher.count("Exec=") == 1

    def test_a_quote_in_the_path_is_escaped_for_powershell(
            self, gui, tmp_path, monkeypatch):
        """An unescaped quote ends the PowerShell string early and the command
        fails to parse. A folder under a name like O'Brien reaches this."""
        if not gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", True)
        _fake_system32(gui, tmp_path, monkeypatch)
        root = tmp_path / "O'Brien" / "localm"
        root.mkdir(parents=True)
        monkeypatch.setattr(gui, "ROOT", root)
        seen = {}

        class Done:
            stdout = "C:\\Users\\x\\Desktop\\LocaLM.lnk"

        def fake(cmd, **kw):
            seen["ps"] = cmd[-1]
            return Done()

        monkeypatch.setattr(gui.subprocess, "run", fake)
        gui.make_shortcut(gui.Plan(shortcut="launcher"), lambda s: None)
        ps = seen["ps"]
        assert "O''Brien" in ps, ps
        assert "O'Brien'" not in ps.replace("O''Brien", "")

    def test_the_written_path_is_returned_for_the_manifest(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        written = gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        assert written and Path(written).is_file()
        # One entry only: the manifest carries a single shortcut path, so a
        # second file written here could never be removed on uninstall.
        assert not (home / "Desktop" / "LocaLM.desktop").exists()

    def test_categories_matches_the_console_installer(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        entry = home / ".local/share/applications" / gui.MENU_ENTRY
        text = entry.read_text(encoding="utf-8")
        assert "Categories=Utility;Development;Science;\n" in text

    def test_icon_prefers_the_svg_when_both_assets_exist(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "localm.svg").write_bytes(b"")
        (assets / "localm.ico").write_bytes(b"")
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        entry = home / ".local/share/applications" / gui.MENU_ENTRY
        text = entry.read_text(encoding="utf-8")
        assert f"Icon={assets / 'localm.svg'}\n" in text

    def test_icon_falls_back_to_the_ico_when_only_that_exists(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "localm.ico").write_bytes(b"")
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        entry = home / ".local/share/applications" / gui.MENU_ENTRY
        text = entry.read_text(encoding="utf-8")
        assert f"Icon={assets / 'localm.ico'}\n" in text

    def test_no_icon_line_when_neither_asset_exists(
            self, gui, tmp_path, monkeypatch):
        if gui.IS_WINDOWS:
            monkeypatch.setattr(gui, "IS_WINDOWS", False)
        home = tmp_path / "h"
        monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: home))
        gui.make_shortcut(gui.Plan(shortcut="gui"), lambda s: None)
        entry = home / ".local/share/applications" / gui.MENU_ENTRY
        text = entry.read_text(encoding="utf-8")
        assert "Icon=" not in text


# --------------------------------------------------------------------------- #
#  The data-folder text box                                                    #
# --------------------------------------------------------------------------- #

class TestCustomFolderText:
    """The typing-side predicate the path_var write-trace consults."""

    def test_nonempty_text_means_the_custom_option(self, gui):
        assert gui._is_custom_folder_text("D:/Models") is True

    def test_empty_text_leaves_the_choice_alone(self, gui):
        assert gui._is_custom_folder_text("") is False
        assert gui._is_custom_folder_text("   ") is False


# --------------------------------------------------------------------------- #
#  The dialogue                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def wizard(gui):
    """A real Tk wizard, skipped where there is no display to build one on."""
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    root = build_tk_root(tk.Tk)
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

    def test_typing_a_data_folder_selects_the_custom_option(self, wizard):
        wizard.next_page()
        assert wizard.pages[wizard.index][0] == "Where things live"
        wizard.path_var.set("D:/Models")
        assert wizard.portable_var.get() is False
        assert wizard.current_plan().portable_data is False
        assert wizard.current_plan().data_path == "D:/Models"

    def test_choosing_portable_after_typing_is_not_overridden(self, wizard):
        wizard.next_page()
        wizard.path_var.set("D:/Models")
        assert wizard.portable_var.get() is False
        wizard.portable_var.set(True)
        assert wizard.current_plan().portable_data is True

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

    def test_a_fresh_folder_opens_on_the_install_questions(self, wizard):
        assert wizard.mode == "install" and wizard.index == 0


# --------------------------------------------------------------------------- #
#  Uninstalling from the window                                                #
# --------------------------------------------------------------------------- #

class _Answer:
    """A messagebox stand-in that records the question and answers it."""

    def __init__(self, yes=True):
        self.yes = yes
        self.asked = []
        self.told = []

    def askyesno(self, title, message, **kw):
        self.asked.append(message)
        return self.yes

    def showinfo(self, title, message):
        self.told.append(message)


@pytest.fixture()
def installed(gui, tmp_path, monkeypatch):
    """A recorded Portable install in the throwaway ROOT, with the machine
    (home folder, user PATH, process list) kept out of reach."""
    from localm import globalcmd
    from localm import install_manifest as im
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.setattr(globalcmd, "_win_read_user_path", lambda: ("", 2))
    monkeypatch.setattr(globalcmd, "_win_write_user_path", lambda v, t: None)
    monkeypatch.setattr(im, "list_processes", lambda: [])
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / ".localm-venv").write_text("", encoding="utf-8")
    for name in (".python", ".cache", ".uv"):
        (tmp_path / name).mkdir()
    lib = tmp_path / "runtime" / "localm_llama_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "llama.dll").write_bytes(b"x")
    im.record(tmp_path, venv=str(tmp_path / ".venv"), lib_dir=str(lib),
              runtime_contained=True, python_dir=str(tmp_path / ".python"),
              cache_dir=str(tmp_path / ".cache"), uv_dir=str(tmp_path / ".uv"))
    data = im.prepare_data(tmp_path, portable=True)
    (data / "chats").mkdir()
    return {"lib": lib, "data": data}


@pytest.fixture()
def uninstall_wizard(gui, installed):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    root = build_tk_root(tk.Tk)
    root.withdraw()
    answer = _Answer()
    try:
        yield gui.Wizard(root, tk, ttk, filedialog, messagebox=answer), answer, root
    finally:
        root.destroy()


def _wait(wizard, root, timeout=60):
    import time
    deadline = time.monotonic() + timeout
    while wizard.installing and time.monotonic() < deadline:
        root.update()
        time.sleep(0.05)
    root.update()
    assert not wizard.installing, "the uninstall never finished"


def test_a_repair_offers_the_data_folder_in_use(gui, installed, tmp_path):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    from localm import install_manifest as im
    custom = tmp_path / "my data"
    im.prepare_data(tmp_path, data_dir=str(custom))
    root = build_tk_root(tk.Tk)
    root.withdraw()
    try:
        wizard = gui.Wizard(root, tk, ttk, filedialog, messagebox=_Answer())
        assert wizard.portable_var.get() is False
        assert wizard.current_plan().data_path == str(custom)
    finally:
        root.destroy()


def test_a_repair_of_a_portable_install_stays_portable(uninstall_wizard, tmp_path):
    wizard, _, _ = uninstall_wizard
    assert wizard.portable_var.get() is True
    assert wizard.current_plan().portable_data is True


class TestUninstallFromTheWindow:
    def test_an_existing_install_opens_on_repair_or_uninstall(self, uninstall_wizard):
        wizard, _, _ = uninstall_wizard
        assert wizard.mode == "choose"
        assert wizard.choice_var.get() == "repair"

    def test_repair_goes_to_the_install_questions_and_back_returns(self, uninstall_wizard):
        wizard, _, _ = uninstall_wizard
        wizard.next_page()
        assert wizard.mode == "install" and wizard.index == 0
        wizard.prev_page()
        assert wizard.mode == "choose"

    def test_the_uninstall_page_shows_what_will_go(self, uninstall_wizard, installed, tmp_path):
        wizard, _, _ = uninstall_wizard
        wizard.choice_var.set("uninstall")
        wizard.next_page()
        assert wizard.mode == "uninstall"
        plan = wizard.plan_text.get("1.0", "end")
        assert "Will be removed:" in plan
        assert str(installed["lib"] / "llama.dll") in plan
        assert "KEPT:" in plan
        wizard.purge_var.set(True)
        wizard._refresh_plan()
        assert "WILL BE DELETED:" in wizard.plan_text.get("1.0", "end")
        wizard.prev_page()
        assert wizard.mode == "choose"

    def test_uninstall_removes_the_install_and_leaves_the_runtime_for_the_script(
            self, uninstall_wizard, installed, tmp_path, gui):
        wizard, answer, root = uninstall_wizard
        wizard.choice_var.set("uninstall")
        wizard.next_page()
        wizard.next_page()                              # the Uninstall button
        _wait(wizard, root)
        assert answer.asked and "saved data is kept" in answer.asked[0]
        assert wizard.exit_code == gui.EXIT_FINISH_UNINSTALL
        assert not (installed["lib"] / "llama.dll").exists()
        assert (installed["data"] / "chats").is_dir()
        assert (tmp_path / ".python").is_dir()          # still running the window
        names = (tmp_path / ".localm-uninstall-pending").read_text(encoding="ascii").split()
        assert sorted(names) == [".cache", ".python", ".uv", ".venv"]

    def test_deleting_the_saved_data_too(self, uninstall_wizard, installed):
        wizard, answer, root = uninstall_wizard
        wizard.choice_var.set("uninstall")
        wizard.next_page()
        wizard.purge_var.set(True)
        wizard.next_page()
        _wait(wizard, root)
        assert "DELETED" in answer.asked[0]
        assert not installed["data"].exists()

    def test_saying_no_changes_nothing(self, uninstall_wizard, installed):
        wizard, answer, _ = uninstall_wizard
        answer.yes = False
        wizard.choice_var.set("uninstall")
        wizard.next_page()
        wizard.next_page()
        assert not wizard.installing and wizard.mode == "uninstall"
        assert (installed["lib"] / "llama.dll").exists()
        assert wizard.exit_code == 0

    @pytest.mark.parametrize("report, pending, expected", [
        ({"exit": 0}, True, "EXIT_FINISH_UNINSTALL"),
        ({"exit": 2}, True, "EXIT_FINISH_UNINSTALL_PARTIAL"),
        ({"exit": 2}, False, "EXIT_UNINSTALL_PARTIAL"),
        ({"exit": 0}, False, None),
        ({"exit": 1}, True, "EXIT_UNINSTALL_FAILED"),
        ({"exit": 3}, False, "EXIT_UNINSTALL_FAILED"),
        ({"exit": 4}, False, "EXIT_UNINSTALL_FAILED"),
        ({"exit": 1, "error": "boom"}, False, "EXIT_UNINSTALL_FAILED"),
    ])
    def test_each_uninstall_result_has_its_own_exit_code(
            self, uninstall_wizard, gui, tmp_path, report, pending, expected):
        wizard, _, _ = uninstall_wizard
        if pending:
            (tmp_path / ".localm-uninstall-pending").write_text(".venv\n", encoding="ascii")
        wizard.installing = True
        wizard._finish_uninstall(report)
        assert wizard.exit_code == (getattr(gui, expected) if expected else 0)
        assert not wizard.installing
        said = wizard.step_label.cget("text")
        if expected == "EXIT_UNINSTALL_FAILED":
            assert said == "Uninstall could not finish."
        else:
            assert (report["exit"] == 2) == ("were not deleted" in said), said

    def test_an_uninstall_that_keeps_saved_data_it_cannot_delete_says_so(
            self, gui, installed, tmp_path):
        """A data folder reached through a link is never deleted, so asking
        for the saved data to go leaves the window with the partial result."""
        tk = pytest.importorskip("tkinter")
        from tkinter import filedialog, ttk
        from localm import install_manifest as im
        real = tmp_path / "real-data"
        real.mkdir()
        link = tmp_path / "data-link"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(real), str(link))
        else:
            link.symlink_to(real, target_is_directory=True)
        im.prepare_data(tmp_path, data_dir=str(link))
        (link / "chats").mkdir()
        root = build_tk_root(tk.Tk)
        root.withdraw()
        try:
            wizard = gui.Wizard(root, tk, ttk, filedialog, messagebox=_Answer())
            wizard.choice_var.set("uninstall")
            wizard.next_page()
            wizard.purge_var.set(True)
            wizard.next_page()
            _wait(wizard, root)
            assert (real / "chats").is_dir()
            assert wizard.exit_code == gui.EXIT_FINISH_UNINSTALL_PARTIAL
            assert "were not deleted" in wizard.step_label.cget("text")
        finally:
            root.destroy()

    def test_the_window_cannot_be_closed_while_the_uninstall_runs(self, uninstall_wizard):
        """Runs the handler the window manager calls when the window is closed."""
        wizard, answer, root = uninstall_wizard
        wizard.installing = True
        wizard.mode = "uninstalling"
        root.tk.call(root.protocol("WM_DELETE_WINDOW"))
        assert root.winfo_exists()
        assert len(answer.told) == 1 and "still running" in answer.told[0]

    def test_the_window_closes_when_no_uninstall_is_running(self, gui, installed):
        tk = pytest.importorskip("tkinter")
        from tkinter import filedialog, ttk
        root = build_tk_root(tk.Tk)
        root.withdraw()
        answer = _Answer()
        wizard = gui.Wizard(root, tk, ttk, filedialog, messagebox=answer)
        wizard.installing = True                    # an install, not an uninstall
        root.tk.call(root.protocol("WM_DELETE_WINDOW"))
        assert answer.told == []
        with pytest.raises(tk.TclError):
            root.winfo_exists()


def test_a_repair_says_when_the_data_folder_in_use_is_unavailable(gui, installed, tmp_path):
    """The folder the install uses is kept as the choice, and the location page
    says it is not available instead of silently choosing another one."""
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    from localm import install_manifest as im
    custom = tmp_path / "offline drive" / "data"
    im.prepare_data(tmp_path, data_dir=str(custom))
    shutil.rmtree(custom.parent)
    root = build_tk_root(tk.Tk)
    root.withdraw()
    try:
        wizard = gui.Wizard(root, tk, ttk, filedialog, messagebox=_Answer())
        assert wizard.missing_data == str(custom)
        assert wizard.portable_var.get() is False
        assert wizard.current_plan().data_path == str(custom)
        location = dict(wizard.pages)["Where things live"]
        texts = [w.cget("text") for w in location.winfo_children()
                 if w.winfo_class() == "TLabel"]
        warning = [t for t in texts if "is not available right now" in t]
        assert len(warning) == 1 and warning[0].startswith("[!]")
        assert str(custom) in warning[0]
        styles = [str(w.cget("style")) for w in location.winfo_children()
                  if w.winfo_class() == "TLabel"
                  and "is not available right now" in w.cget("text")]
        assert styles == ["Warn.TLabel"]
    finally:
        root.destroy()


def test_a_repair_with_its_data_folder_present_shows_no_warning(uninstall_wizard):
    wizard, _, _ = uninstall_wizard
    assert wizard.missing_data == ""
    location = dict(wizard.pages)["Where things live"]
    texts = [w.cget("text") for w in location.winfo_children()
             if w.winfo_class() == "TLabel"]
    assert not any("is not available right now" in t for t in texts)


# --------------------------------------------------------------------------- #
#  Following the system's dark or light mode                                   #
# --------------------------------------------------------------------------- #

_MAC_STYLE = ("defaults", "read", "-g", "AppleInterfaceStyle")
_GNOME_SCHEME = ("gsettings", "get", "org.gnome.desktop.interface", "color-scheme")
_GNOME_GTK = ("gsettings", "get", "org.gnome.desktop.interface", "gtk-theme")


def _runner(outputs, calls=None):
    """A command runner answering from outputs (argv tuple -> stdout); None
    for any other command."""
    def run(argv):
        if calls is not None:
            calls.append(tuple(argv))
        return outputs.get(tuple(argv))
    return run


class TestThemeDetection:
    def test_windows_dark_apps_mean_dark(self, gui):
        assert gui.detect_theme(env={}, platform="win32",
                                read_windows=lambda: 0) == "dark"

    def test_windows_light_apps_mean_light(self, gui):
        assert gui.detect_theme(env={}, platform="win32",
                                read_windows=lambda: 1) == "light"

    def test_windows_without_the_setting_means_light(self, gui):
        def missing():
            raise FileNotFoundError(2, "The system cannot find the file specified")
        assert gui.detect_theme(env={}, platform="win32",
                                read_windows=missing) == "light"

    def test_macos_dark_mode_means_dark(self, gui):
        calls = []
        run = _runner({_MAC_STYLE: "Dark\n"}, calls)
        assert gui.detect_theme(env={}, platform="darwin", run=run) == "dark"
        assert calls == [_MAC_STYLE]

    def test_macos_without_the_setting_means_light(self, gui):
        assert gui.detect_theme(env={}, platform="darwin",
                                run=_runner({})) == "light"

    def test_gnome_prefer_dark_means_dark(self, gui):
        run = _runner({_GNOME_SCHEME: "'prefer-dark'\n"})
        assert gui.detect_theme(env={}, platform="linux", run=run) == "dark"

    def test_gnome_prefer_light_means_light_whatever_the_gtk_theme(self, gui):
        run = _runner({_GNOME_SCHEME: "'prefer-light'\n",
                       _GNOME_GTK: "'Adwaita-dark'\n"})
        assert gui.detect_theme(env={}, platform="linux", run=run) == "light"

    @pytest.mark.parametrize("scheme", ["'default'\n", None])
    def test_a_dark_gtk_theme_means_dark_without_a_colour_scheme(self, gui, scheme):
        outputs = {_GNOME_GTK: "'Adwaita-Dark'\n"}
        if scheme is not None:
            outputs[_GNOME_SCHEME] = scheme
        assert gui.detect_theme(env={}, platform="linux",
                                run=_runner(outputs)) == "dark"

    def test_no_dark_setting_means_light(self, gui):
        run = _runner({_GNOME_SCHEME: "'default'\n", _GNOME_GTK: "'Adwaita'\n"})
        assert gui.detect_theme(env={}, platform="linux", run=run) == "light"
        assert gui.detect_theme(env={}, platform="linux",
                                run=_runner({})) == "light"

    @pytest.mark.parametrize("value,expected", [("dark", "dark"),
                                                ("LIGHT", "light"),
                                                (" Dark ", "dark")])
    def test_localm_theme_wins_without_asking_the_system(self, gui, value, expected):
        asked = []

        def read():
            asked.append(True)
            return 0 if expected == "light" else 1

        assert gui.detect_theme(env={gui.THEME_ENV: value}, platform="win32",
                                read_windows=read) == expected
        assert asked == []

    @pytest.mark.parametrize("value", ["", "blue", "auto"])
    def test_any_other_localm_theme_value_is_ignored(self, gui, value):
        assert gui.detect_theme(env={gui.THEME_ENV: value}, platform="win32",
                                read_windows=lambda: 0) == "dark"

    def test_the_real_environment_is_read(self, gui, monkeypatch):
        monkeypatch.setenv("LOCALM_THEME", "dark")
        assert gui.detect_theme(platform="win32", read_windows=lambda: 1) == "dark"
        monkeypatch.setenv("LOCALM_THEME", "light")
        assert gui.detect_theme(platform="win32", read_windows=lambda: 0) == "light"

    @pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
    def test_a_reader_that_fails_means_light(self, gui, platform):
        def broken(*_args):
            raise RuntimeError("reader broke")
        assert gui.detect_theme(env={}, platform=platform, read_windows=broken,
                                run=broken) == "light"


class TestCommandOutput:
    def test_the_output_of_a_command_that_succeeds(self, gui):
        out = gui._command_output([sys.executable, "-c", "print('Dark')"])
        assert out is not None and out.strip() == "Dark"

    def test_a_command_that_fails_gives_none(self, gui):
        assert gui._command_output(
            [sys.executable, "-c", "import sys; print('Dark'); sys.exit(1)"]) is None

    def test_a_missing_program_gives_none(self, gui, tmp_path):
        assert gui._command_output([str(tmp_path / "no-such-program")]) is None

    def test_a_command_that_hangs_gives_none_at_the_timeout(self, gui):
        import time
        started = time.monotonic()
        assert gui._command_output(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5) is None
        assert time.monotonic() - started < 30


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows registry")
def test_the_windows_reader_matches_the_registry(gui):
    import subprocess
    proc = subprocess.run(
        ["reg", "query",
         r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
         "/v", "AppsUseLightTheme"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        with pytest.raises(OSError):
            gui._apps_use_light_theme()
        assert gui.detect_theme(env={}) == "light"
        return
    expected = int(proc.stdout.split()[-1], 16)
    assert gui._apps_use_light_theme() == expected
    assert gui.detect_theme(env={}) == ("dark" if expected == 0 else "light")


def _luminance(colour):
    """WCAG relative luminance of a #rrggbb colour."""
    channels = []
    for i in (1, 3, 5):
        c = int(colour[i:i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    """WCAG contrast ratio between two #rrggbb colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


_TEXT_PAIRS = [("text", "bg"), ("dim", "bg"), ("warn", "bg"),
               ("text", "surface"), ("text", "field"), ("on_accent", "accent")]
_CONTROL_PAIRS = [("outline", "bg"), ("accent", "field"), ("accent", "bg")]


class TestPalettes:
    def test_the_contrast_helper_gives_known_ratios(self):
        assert _contrast("#000000", "#ffffff") == pytest.approx(21.0)
        assert _contrast("#ffffff", "#777777") == pytest.approx(4.48, abs=0.01)
        assert _contrast("#123456", "#123456") == pytest.approx(1.0)

    def test_both_themes_name_the_same_colours(self, gui):
        assert set(gui.PALETTES) == {"dark", "light"}
        assert set(gui.PALETTES["dark"]) == set(gui.PALETTES["light"])

    def test_the_dark_theme_is_dark_and_the_light_theme_is_light(self, gui):
        dark, light = gui.PALETTES["dark"], gui.PALETTES["light"]
        assert _luminance(dark["bg"]) < 0.05 < _luminance(dark["text"])
        assert _luminance(light["bg"]) > 0.8 > _luminance(light["text"])

    @pytest.mark.parametrize("theme", ["dark", "light"])
    @pytest.mark.parametrize("fg,bg", _TEXT_PAIRS)
    def test_text_is_readable(self, gui, theme, fg, bg):
        p = gui.PALETTES[theme]
        ratio = _contrast(p[fg], p[bg])
        assert ratio >= 4.5, f"{theme}: {fg} on {bg} is {ratio:.2f}:1"

    @pytest.mark.parametrize("theme", ["dark", "light"])
    @pytest.mark.parametrize("fg,bg", _CONTROL_PAIRS)
    def test_control_edges_and_marks_are_visible(self, gui, theme, fg, bg):
        p = gui.PALETTES[theme]
        ratio = _contrast(p[fg], p[bg])
        assert ratio >= 3.0, f"{theme}: {fg} on {bg} is {ratio:.2f}:1"


def _tk_root():
    tk = pytest.importorskip("tkinter")
    root = build_tk_root(tk.Tk)
    root.withdraw()
    return root


def _all_widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _all_widgets(child)


@pytest.fixture(params=["dark", "light"])
def themed(request, gui):
    """A real Tk wizard built with each theme forced: (wizard, style, palette)."""
    from tkinter import filedialog, ttk
    import tkinter as tk
    root = _tk_root()
    try:
        wizard = gui.Wizard(root, tk, ttk, filedialog, theme=request.param)
        yield wizard, ttk.Style(root), gui.PALETTES[request.param]
    finally:
        root.destroy()


class TestThemedWindow:
    def test_the_window_uses_clam(self, themed):
        _, style, _ = themed
        assert style.theme_use() == "clam"

    def test_the_window_and_its_frames_take_the_background(self, themed):
        wizard, style, p = themed
        assert str(wizard.root.cget("background")) == p["bg"]
        assert style.lookup("TFrame", "background") == p["bg"]
        assert style.lookup("TLabel", "background") == p["bg"]
        assert style.lookup("TLabel", "foreground") == p["text"]

    def test_the_dim_and_warning_label_styles(self, themed):
        _, style, p = themed
        assert style.lookup("Dim.TLabel", "foreground") == p["dim"]
        assert style.lookup("Dim.TLabel", "background") == p["bg"]
        assert style.lookup("Warn.TLabel", "foreground") == p["warn"]
        assert style.lookup("Warn.TLabel", "background") == p["bg"]

    def test_no_label_keeps_a_fixed_colour(self, themed):
        wizard, _, _ = themed
        labels = [w for w in _all_widgets(wizard.root) if w.winfo_class() == "TLabel"]
        assert labels
        fixed = [(w.cget("text"), str(w.cget("foreground"))) for w in labels
                 if str(w.cget("foreground"))]
        assert fixed == []

    def test_subheadings_and_the_folder_path_use_the_dim_style(self, themed):
        wizard, _, _ = themed
        dim = [w.cget("text") for w in _all_widgets(wizard.root)
               if w.winfo_class() == "TLabel" and str(w.cget("style")) == "Dim.TLabel"]
        assert any(t.startswith("Both of these can stay inside this folder") for t in dim)
        assert any(t.endswith("home") for t in dim)

    def test_the_log_and_the_uninstall_list_use_the_palette(self, themed):
        wizard, _, p = themed
        for text in (wizard.log, wizard.plan_text):
            assert str(text.cget("background")) == p["surface"]
            assert str(text.cget("foreground")) == p["text"]
            assert str(text.cget("insertbackground")) == p["text"]
            assert str(text.cget("selectbackground")) == p["accent"]
            assert str(text.cget("selectforeground")) == p["on_accent"]

    def test_controls_use_the_palette(self, themed):
        _, style, p = themed
        assert style.lookup("TButton", "background") == p["surface"]
        assert style.lookup("TButton", "foreground") == p["text"]
        assert style.lookup("TEntry", "fieldbackground") == p["field"]
        assert style.lookup("TEntry", "foreground") == p["text"]
        assert style.lookup("TProgressbar", "background") == p["accent"]
        assert style.lookup("TProgressbar", "troughcolor") == p["field"]
        assert style.lookup("TScrollbar", "troughcolor") == p["bg"]
        for cls in ("TCheckbutton", "TRadiobutton"):
            assert style.lookup(cls, "foreground") == p["text"]
            assert style.lookup(cls, "indicatorbackground") == p["field"]
            assert style.lookup(cls, "indicatorforeground") == p["accent"]

    def test_hovered_pressed_and_disabled_controls_stay_in_the_palette(self, themed):
        _, style, p = themed
        for cls in ("TCheckbutton", "TRadiobutton", "TLabel", "TFrame"):
            assert style.lookup(cls, "background", ["active"]) == p["bg"]
        assert style.lookup("TButton", "background", ["active"]) == p["field"]
        assert style.lookup("TButton", "background", ["pressed"]) == p["border"]
        assert style.lookup("TButton", "background", ["disabled"]) == p["bg"]
        assert style.lookup("TButton", "foreground", ["disabled"]) == p["outline"]
        assert style.lookup("TScrollbar", "background", ["active"]) == p["field"]
        assert style.lookup("TEntry", "bordercolor", ["focus"]) == p["accent"]

    def test_every_widget_class_in_the_window_is_one_the_theme_styles(self, themed):
        wizard, _, _ = themed
        styled = {"TFrame", "TLabel", "TButton", "TCheckbutton", "TRadiobutton",
                  "TEntry", "TProgressbar", "TScrollbar", "Text"}
        classes = {w.winfo_class() for w in _all_widgets(wizard.root)}
        assert classes <= styled, f"not themed: {sorted(classes - styled)}"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_feature_list_warning_uses_the_warning_style(gui, monkeypatch, theme):
    import tkinter as tk
    from tkinter import filedialog, ttk
    monkeypatch.setattr(gui, "plugin_choices", lambda: [])
    root = _tk_root()
    try:
        wizard = gui.Wizard(root, tk, ttk, filedialog, theme=theme)
        features = dict(wizard.pages)["Optional features"]
        warnings = [w for w in features.winfo_children()
                    if w.winfo_class() == "TLabel"
                    and "could not be read" in w.cget("text")]
        assert len(warnings) == 1
        assert str(warnings[0].cget("style")) == "Warn.TLabel"
    finally:
        root.destroy()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_window_follows_localm_theme_when_no_theme_is_given(
        gui, monkeypatch, theme):
    import tkinter as tk
    from tkinter import filedialog, ttk
    monkeypatch.setenv("LOCALM_THEME", theme)
    root = _tk_root()
    try:
        wizard = gui.Wizard(root, tk, ttk, filedialog)
        assert wizard.theme == theme
        assert ttk.Style(root).lookup("TFrame", "background") == \
            gui.PALETTES[theme]["bg"]
    finally:
        root.destroy()


@pytest.mark.parametrize("theme,dark", [("dark", True), ("light", False)])
def test_the_window_asks_for_a_matching_title_bar(gui, monkeypatch, theme, dark):
    import tkinter as tk
    from tkinter import filedialog, ttk
    asked = []
    monkeypatch.setattr(gui, "set_title_bar_theme",
                        lambda root, want: asked.append((root, want)) or True)
    root = _tk_root()
    try:
        gui.Wizard(root, tk, ttk, filedialog, theme=theme)
        assert asked == [(root, dark)]
    finally:
        root.destroy()


class _FakeRoot:
    def __init__(self, idle_error=None):
        self.idle_error = idle_error
        self.idle = 0

    def update_idletasks(self):
        self.idle += 1
        if self.idle_error is not None:
            raise self.idle_error

    def winfo_id(self):
        return 4242


class _FakeWindll:
    """ctypes.windll stand-in recording DwmSetWindowAttribute calls as
    (hwnd, attribute, value, size)."""

    def __init__(self, parent=777, results=(0,), parent_error=None):
        self.calls = []
        self.parent_of = None
        fake = self

        class User32:
            def GetParent(self, window_id):
                if parent_error is not None:
                    raise parent_error
                fake.parent_of = window_id
                return parent

        class Dwmapi:
            def DwmSetWindowAttribute(self, hwnd, attribute, value, size):
                fake.calls.append((hwnd.value, attribute, value._obj.value, size))
                return results[min(len(fake.calls), len(results)) - 1]

        self.user32 = User32()
        self.dwmapi = Dwmapi()


_E_INVALIDARG = -2147024809


class TestTitleBar:
    @pytest.mark.parametrize("dark,value", [(True, 1), (False, 0)])
    def test_the_frame_window_is_asked_for_the_theme(self, gui, dark, value):
        root, windll = _FakeRoot(), _FakeWindll(parent=777)
        assert gui.set_title_bar_theme(root, dark, windll=windll) is True
        assert root.idle == 1
        assert windll.parent_of == 4242
        assert windll.calls == [(777, 20, value, 4)]

    def test_the_older_attribute_is_tried_when_the_current_one_is_refused(self, gui):
        windll = _FakeWindll(results=(_E_INVALIDARG, 0))
        assert gui.set_title_bar_theme(_FakeRoot(), True, windll=windll) is True
        assert [c[1] for c in windll.calls] == [20, 19]

    def test_a_title_bar_that_cannot_be_set_is_reported_not_raised(self, gui):
        refused = _FakeWindll(results=(_E_INVALIDARG,))
        assert gui.set_title_bar_theme(_FakeRoot(), True, windll=refused) is False
        assert [c[1] for c in refused.calls] == [20, 19]

        no_parent = _FakeWindll(parent_error=OSError("GetParent failed"))
        assert gui.set_title_bar_theme(_FakeRoot(), True, windll=no_parent) is False

        assert gui.set_title_bar_theme(
            _FakeRoot(idle_error=RuntimeError("window gone")), True,
            windll=_FakeWindll()) is False

    def test_no_frame_window_means_nothing_is_set(self, gui):
        windll = _FakeWindll(parent=0)
        assert gui.set_title_bar_theme(_FakeRoot(), True, windll=windll) is False
        assert windll.calls == []

    def test_other_systems_do_not_try(self, gui, monkeypatch):
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        root = _FakeRoot()
        assert gui.set_title_bar_theme(root, True) is False
        assert root.idle == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows title bars only")
@pytest.mark.parametrize("dark", [True, False])
def test_the_title_bar_setting_reaches_the_real_window(gui, dark):
    import ctypes
    root = _tk_root()
    try:
        assert gui.set_title_bar_theme(root, dark) is True
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(-1)
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.c_void_p(hwnd), 20, ctypes.byref(value), ctypes.sizeof(value))
        assert hr == 0
        assert value.value == int(dark)
    finally:
        root.destroy()


class TestStoreUv:
    """The first step puts localm's own pinned uv in the store the tooling
    answer picked, whatever uv the launcher used to open the window."""

    @staticmethod
    def _uv(directory, gui):
        directory.mkdir(parents=True, exist_ok=True)
        exe = directory / ("uv.exe" if gui.IS_WINDOWS else "uv")
        exe.write_bytes(b"")
        exe.chmod(0o755)
        return exe

    @staticmethod
    def _pin(gui, tmp_path, installer=b"installer"):
        import hashlib
        sha = hashlib.sha256(installer).hexdigest()
        for name in ("setup.bat", "setup-gui.bat", "setup.sh", "setup-gui.sh"):
            (tmp_path / name).write_text(
                f'set "UV_INSTALLER_VERSION=0.13.6"\nset "UV_INSTALLER_SHA256={sha}"\n',
                encoding="utf-8")
        return installer

    @pytest.mark.parametrize("portable,key", [(True, "uv-portable"), (False, "uv-bootstrap")])
    def test_every_install_starts_with_the_uv_step(self, gui, portable, key):
        steps = gui.all_steps(gui.Plan(backend="own", portable_store=portable))
        assert steps[0].key == key and steps[0].fatal
        assert [s.label for s in steps[1:]] == [s.label for s in gui.build_steps(
            gui.Plan(backend="own", portable_store=portable))]

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_a_pinned_uv_already_in_the_store_is_kept(self, gui, tmp_path, monkeypatch,
                                                      portable):
        self._pin(gui, tmp_path)
        exe = self._uv(gui.store_dirs(portable)["uv"], gui)
        monkeypatch.setattr(gui, "uv_version", lambda e: "0.13.6")
        monkeypatch.setattr(gui, "install_store_uv",
                            lambda *a, **k: pytest.fail("reinstalled a pinned uv"))
        logged = []
        gui.uv_store_step(gui.Plan(portable_store=portable)).run(logged.append)
        assert any(str(exe) in line for line in logged)

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_a_wrong_version_or_missing_uv_is_installed_into_the_store(
            self, gui, tmp_path, monkeypatch, portable):
        self._pin(gui, tmp_path)
        self._uv(tmp_path / "sysbin", gui)
        monkeypatch.setenv("PATH", str(tmp_path / "sysbin"))
        installed = []
        monkeypatch.setattr(gui, "install_store_uv",
                            lambda plan, emit: installed.append(plan.portable_store) or "x")
        gui.uv_store_step(gui.Plan(portable_store=portable)).run(lambda s: None)
        self._uv(gui.store_dirs(portable)["uv"], gui)
        monkeypatch.setattr(gui, "uv_version", lambda e: "0.12.0")
        gui.uv_store_step(gui.Plan(portable_store=portable)).run(lambda s: None)
        assert installed == [portable, portable]

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_the_installer_puts_uv_in_the_store_with_nothing_outside(
            self, gui, tmp_path, monkeypatch, portable):
        installer = self._pin(gui, tmp_path)
        target = gui.store_dirs(portable)["uv"]
        seen = {}

        def run(script, env, emit):
            seen.update(env)
            seen["script"] = script
            self._uv(Path(env["UV_INSTALL_DIR"]), gui)
            return 0

        monkeypatch.setattr(gui, "uv_version", lambda e: "0.13.6")
        exe = gui.install_store_uv(gui.Plan(portable_store=portable), lambda s: None,
                                   download=lambda u, d: d.write_bytes(installer),
                                   run_installer=run)
        assert exe == target / ("uv.exe" if gui.IS_WINDOWS else "uv")
        assert seen["UV_INSTALL_DIR"] == seen["UV_UNMANAGED_INSTALL"] == str(target)
        tmp = Path(seen["TEMP"])
        assert seen["TMP"] == seen["TMPDIR"] == str(tmp)
        assert tmp.parent.parent == tmp_path / ".localm-setup-tmp"
        assert seen["script"].parent.parent == tmp_path / ".localm-setup-tmp"
        assert not tmp.exists() and not seen["script"].exists()
        assert seen["UV_NO_CONFIG"] == "1" and "PYTHONPATH" not in seen

    def test_a_tampered_installer_is_never_run(self, gui, tmp_path, monkeypatch):
        self._pin(gui, tmp_path)
        with pytest.raises(gui.StepFailed, match="checksum"):
            gui.install_store_uv(gui.Plan(), lambda s: None,
                                 download=lambda u, d: d.write_bytes(b"tampered"),
                                 run_installer=lambda *a: pytest.fail("ran it"))

    def test_an_installed_uv_of_another_version_is_a_failure(self, gui, tmp_path,
                                                            monkeypatch):
        installer = self._pin(gui, tmp_path)
        monkeypatch.setattr(gui, "uv_version", lambda e: "0.11.0")

        def run(script, env, emit):
            self._uv(Path(env["UV_INSTALL_DIR"]), gui)
            return 0

        with pytest.raises(gui.StepFailed, match="not the pinned 0.13.6"):
            gui.install_store_uv(gui.Plan(), lambda s: None,
                                 download=lambda u, d: d.write_bytes(installer),
                                 run_installer=run)

    def test_a_shared_store_uv_is_recorded_by_its_full_path(self, gui, tmp_path,
                                                           monkeypatch):
        own = self._uv(gui.store_dirs(False)["uv"], gui)
        self._uv(tmp_path / "sysbin", gui)
        monkeypatch.setenv("PATH", str(tmp_path / "sysbin"))
        seen = []
        monkeypatch.setattr(gui, "_run", lambda cmd, emit, plan, **kw: seen.append(
            [str(c) for c in cmd]) or 0)
        _step(gui, gui.Plan(backend="own", portable_store=False),
              "Writing the install record").run(lambda s: None)
        rec = seen[0]
        assert rec[rec.index("--uv-exe") + 1] == str(own)

    def test_a_contained_uv_is_recorded_as_uv_dir_and_not_as_a_shared_one(
            self, gui, tmp_path, monkeypatch):
        self._uv(tmp_path / ".uv", gui)
        cmds = _commands_for(gui, monkeypatch, gui.Plan(backend="own", portable_store=True))
        rec = [c for c in cmds if "localm.install_manifest" in c][0]
        assert rec[rec.index("--uv-dir") + 1] == str(tmp_path / ".uv")
        assert "--uv-exe" not in rec

    def test_the_launcher_preset_starts_the_window_on_shared(self, gui, monkeypatch):
        assert gui.STORE_ENV == "LOCALM_GUI_STORE"
        monkeypatch.setenv(gui.STORE_ENV, "shared")
        assert os.environ.get(gui.STORE_ENV) == "shared"


class TestLauncherStorePreset:
    """setup-gui.bat / setup-gui.sh pass the Portable or Shared answer to the window."""

    @staticmethod
    def _wizard(gui):
        tk = pytest.importorskip("tkinter")
        from tkinter import filedialog, ttk
        root = build_tk_root(tk.Tk)
        root.withdraw()
        return root, gui.Wizard(root, tk, ttk, filedialog)

    def test_the_window_starts_on_portable_by_default(self, gui, monkeypatch):
        monkeypatch.delenv(gui.STORE_ENV, raising=False)
        root, wizard = self._wizard(gui)
        try:
            assert wizard.store_var.get() is True
            assert wizard.current_plan().portable_store is True
        finally:
            root.destroy()

    def test_the_window_starts_on_shared_when_the_launcher_says_so(self, gui, monkeypatch):
        monkeypatch.setenv(gui.STORE_ENV, "shared")
        root, wizard = self._wizard(gui)
        try:
            assert wizard.store_var.get() is False
            assert wizard.current_plan().portable_store is False
        finally:
            root.destroy()


def _fake_system32(gui, tmp_path, monkeypatch):
    """A Windows system folder holding powershell, attrib, taskkill and cmd."""
    sysdir = tmp_path.parent / f"{tmp_path.name}-System32"
    (sysdir / "WindowsPowerShell" / "v1.0").mkdir(parents=True, exist_ok=True)
    (sysdir / "WindowsPowerShell" / "v1.0" / "powershell.exe").write_bytes(b"")
    for name in ("attrib", "taskkill", "cmd"):
        (sysdir / f"{name}.exe").write_bytes(b"")
    monkeypatch.setattr(gui, "windows_system_dir", lambda: sysdir)
    return sysdir


# --------------------------------------------------------------------------- #
#  Every program by its full path                                              #
# --------------------------------------------------------------------------- #

class TestNoBareProgram:
    """Every child process the install steps, the uv step and the Windows
    helpers start names its program by full path: subprocess is replaced at the
    module boundary and argv[0] checked. In-process code the window imports
    (localm.hwdetect for the backend recommendation, the uninstall) is not
    covered here."""

    @pytest.fixture()
    def spawned(self, gui, tmp_path, monkeypatch):
        calls = []

        class Proc:
            returncode = 0
            stdout: list = []
            stdin = None

            def __init__(self, argv):
                self.argv = argv

            def wait(self):
                return 0

        def record(argv, **kw):
            calls.append(([str(a) for a in argv], kw))
            return Proc(argv)

        monkeypatch.setattr(gui.subprocess, "run", record)
        monkeypatch.setattr(gui.subprocess, "Popen", record)
        return calls

    @staticmethod
    def _uv(directory, gui):
        directory.mkdir(parents=True, exist_ok=True)
        exe = directory / ("uv.exe" if gui.IS_WINDOWS else "uv")
        exe.write_bytes(b"")
        exe.chmod(0o755)
        return exe

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_every_install_step_runs_a_full_path(self, gui, tmp_path, monkeypatch,
                                                 spawned, portable):
        self._uv(gui.store_dirs(portable)["uv"], gui)
        monkeypatch.setattr(gui, "make_shortcut",
                            lambda plan, emit: str(tmp_path / "LocaLM.lnk"))
        plan = gui.Plan(backend="vulkan", app_window=True, add_to_path=True,
                        portable_store=portable, plugins=("coder",))
        for step in gui.build_steps(plan):
            if step.label.startswith("Creating the Python environment"):
                (tmp_path / ".venv").mkdir(exist_ok=True)
            try:
                step.run(lambda s: None)
            except gui.StepFailed:
                pass
        assert spawned, "nothing was started"
        for argv, kw in spawned:
            assert os.path.isabs(argv[0]), f"{argv} searches PATH for {argv[0]}"
            assert kw.get("creationflags") == gui._NO_WINDOW, argv
            env = kw.get("env")
            assert env is not None and env.get("UV_NO_CONFIG") == "1", argv

    def test_the_windows_helpers_run_from_the_system_folder(self, gui, tmp_path,
                                                             monkeypatch, spawned):
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        sysdir = _fake_system32(gui, tmp_path, monkeypatch)
        gui.find_venv_lockers(tmp_path)
        gui.intended_shortcut_path()
        try:
            gui.make_shortcut(gui.Plan(shortcut="launcher"), lambda s: None)
        except Exception:
            pass
        gui.run_system("attrib", "-R")
        gui.run_system("taskkill", "/PID", "1")
        names = [a[0] for a, _ in spawned]
        ps = str(sysdir / "WindowsPowerShell" / "v1.0" / "powershell.exe")
        assert names.count(ps) >= 3, names
        assert str(sysdir / "attrib.exe") in names and str(sysdir / "taskkill.exe") in names
        assert all(os.path.isabs(n) for n in names), names

    def test_a_missing_system_program_is_never_looked_up_on_path(self, gui, tmp_path,
                                                                  monkeypatch, spawned):
        monkeypatch.setattr(gui, "IS_WINDOWS", True)
        monkeypatch.setattr(gui, "windows_system_dir", lambda: tmp_path / "nothing")
        with pytest.raises(FileNotFoundError):
            gui.run_system("attrib", "-R")
        assert gui.find_venv_lockers(tmp_path) == []
        assert spawned == []

    def test_posix_programs_come_from_the_fixed_system_folders(self, gui, tmp_path,
                                                                monkeypatch):
        monkeypatch.setattr(gui, "IS_WINDOWS", False)
        first, second = tmp_path / "usr-bin", tmp_path / "bin"
        for d in (first, second):
            d.mkdir()
        tool = second / "sh"
        tool.write_bytes(b"")
        tool.chmod(0o755)
        monkeypatch.setattr(gui, "POSIX_SYSTEM_DIRS", (str(first), str(second)))
        monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
        if os.name == "nt":
            monkeypatch.setattr(gui.os, "access", lambda p, m: True)
        assert gui.system_program("sh") == str(tool)
        assert gui.system_program("curl") is None

    @pytest.mark.parametrize("portable", [True, False], ids=["portable", "shared"])
    def test_the_uv_version_probe_and_installer_run_full_paths(self, gui, tmp_path,
                                                                monkeypatch, spawned,
                                                                portable):
        exe = self._uv(gui.store_dirs(portable)["uv"], gui)
        gui.uv_version(exe)
        script = tmp_path / "uv-installer.ps1"
        script.write_bytes(b"")
        monkeypatch.setenv("PATH", str(tmp_path / "poisoned"))
        gui._run_uv_installer(script, gui._helper_env(TEMP=str(tmp_path)), lambda s: None)
        assert len(spawned) == 2, spawned
        for argv, kw in spawned:
            assert os.path.isabs(argv[0]), argv
            assert kw.get("creationflags") == gui._NO_WINDOW, argv
        installer_env = spawned[1][1]["env"]
        assert installer_env["PATH"] == gui.system_path()
        assert str(tmp_path / "poisoned") not in installer_env["PATH"]


@pytest.mark.skipif(os.name != "nt", reason="runs Windows PowerShell")
@pytest.mark.parametrize("folder", ["clone", "clone [x]"], ids=["plain", "brackets"])
def test_the_installer_runs_from_a_bracketed_folder(gui, tmp_path, monkeypatch, folder):
    """A stand-in for Astral's installer that makes its temp folder from
    GetTempPath, unpacks with Expand-Archive -Path and moves the binary with
    positional Copy-Item / Remove-Item, as the pinned installer does; it
    installs from a folder whose path holds [ ] with nothing written outside."""
    root = tmp_path / folder
    root.mkdir()
    monkeypatch.setattr(gui, "ROOT", root)
    real_ps = os.path.join(os.environ["SystemRoot"], "System32", "WindowsPowerShell",
                           "v1.0", "powershell.exe")
    monkeypatch.setattr(gui, "system_program",
                        lambda name: real_ps if name == "powershell" else None)
    stub = (
        b"$ErrorActionPreference = 'Stop'\r\n"
        b"$parent = [System.IO.Path]::GetTempPath()\r\n"
        b"$tmp = New-Item -ItemType Directory -Path (Join-Path $parent ([guid]::NewGuid()))\r\n"
        b"Add-Content -LiteralPath $env:STUB_LOG -Value $tmp\r\n"
        b"$dir_path = \"$tmp\\uv.zip\"\r\n"
        b"Add-Type -AssemblyName System.IO.Compression.FileSystem\r\n"
        b"$src = Join-Path $tmp 'src'\r\n"
        b"$null = [IO.Directory]::CreateDirectory($src)\r\n"
        b"[IO.File]::WriteAllText((Join-Path $src 'uv.exe'), 'fake')\r\n"
        b"[IO.Compression.ZipFile]::CreateFromDirectory($src, $dir_path)\r\n"
        b"[IO.Directory]::Delete($src, $true)\r\n"
        b"Expand-Archive -Path $dir_path -DestinationPath \"$tmp\";\r\n"
        b"$dest_dir = New-Item -Force -ItemType Directory -Path $env:UV_INSTALL_DIR\r\n"
        b"Copy-Item \"$tmp\\uv.exe\" -Destination \"$dest_dir\" -ErrorAction Stop\r\n"
        b"Remove-Item \"$tmp\\uv.exe\" -Recurse -Force -ErrorAction Stop\r\n"
    )
    TestStoreUv._pin(gui, root, stub)
    log = tmp_path / "stub.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setattr(gui, "uv_version", lambda e: "0.13.6")
    lines = []
    exe = gui.install_store_uv(gui.Plan(), lines.append,
                               download=lambda u, d: d.write_bytes(stub))
    assert exe == root / ".uv" / "uv.exe" and exe.is_file(), lines
    temps = log.read_text(encoding="utf-8").splitlines()
    assert temps and all(Path(t).is_relative_to(root / ".localm-setup-tmp") for t in temps)
    assert not any(Path(t).exists() for t in temps)


# --------------------------------------------------------------------------- #
#  The setup temp folder                                                       #
# --------------------------------------------------------------------------- #

class TestSetupTemp:
    class _Window:
        def __init__(self, gui):
            import queue
            self.lines = queue.Queue()
            self.logged = []
            self._emit = self.logged.append

    def _run_worker(self, gui, steps):
        window = self._Window(gui)
        gui.Wizard._worker(window, steps)
        return window

    def test_a_run_that_reaches_its_end_removes_it(self, gui, tmp_path):
        def step(emit):
            gui._ensure_setup_temp()
            (gui.setup_temp() / "uv-setuptools-x.lock").write_bytes(b"")
        self._run_worker(gui, [gui.Step("one", step)])
        assert not (tmp_path / ".localm-setup-tmp").exists()

    def test_a_run_stopped_by_a_required_step_keeps_it(self, gui, tmp_path):
        def step(emit):
            gui._ensure_setup_temp()
            raise gui.StepFailed("broken")
        self._run_worker(gui, [gui.Step("one", step)])
        assert (tmp_path / ".localm-setup-tmp").is_dir()

    def test_the_app_started_after_setup_never_gets_the_removed_folder(
            self, gui, tmp_path, monkeypatch):
        inside = str(tmp_path / ".localm-setup-tmp")
        for key in ("TEMP", "TMP", "TMPDIR"):
            monkeypatch.setenv(key, inside)
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "otherhome"))
        monkeypatch.setenv("PYTHONPATH", str(tmp_path / "evil"))
        env = gui.launch_env(gui.Plan(portable_data=True))
        assert env["TEMP"] == env["TMP"] == env["TMPDIR"] == str(tmp_path / "home" / "tmp")
        assert "LOCALM_HOME" not in env and "PYTHONPATH" not in env

    def test_the_apps_own_temp_is_left_alone_otherwise(self, gui, tmp_path, monkeypatch):
        monkeypatch.setenv("TEMP", str(tmp_path / "usertemp"))
        assert gui.launch_env(gui.Plan())["TEMP"] == str(tmp_path / "usertemp")


# --------------------------------------------------------------------------- #
#  Locations the user picks                                                    #
# --------------------------------------------------------------------------- #

def test_a_custom_data_folder_anywhere_is_used_as_picked(gui, tmp_path):
    """A folder the user picks outside the install, even with [ ] and spaces in
    its name, is recorded as picked and the started app's temp goes there."""
    picked = tmp_path.parent / f"{tmp_path.name}-my data [x]" / "localm"
    plan = gui.Plan(portable_data=False, data_path=str(picked), backend="own")
    _step(gui, plan, "Recording where data lives").run(lambda s: None)
    assert Path(_record(tmp_path)["data_dir"]) == picked
    assert picked.is_dir()
    env = gui.launch_env(plan)
    assert "LOCALM_HOME" not in env


def test_the_own_build_folder_is_passed_as_picked(gui, tmp_path, monkeypatch):
    picked = str(tmp_path.parent / f"{tmp_path.name}-builds [x]" / "llama")
    seen = []
    monkeypatch.setattr(gui, "_run", lambda cmd, emit, plan, **kw: seen.append(
        [str(c) for c in cmd]) or 0)
    _step(gui, gui.Plan(backend="own", own_build_dir=picked),
          "Copying in your own llama.cpp build").run(lambda s: None)
    assert seen[0][seen[0].index("--from") + 1] == picked


def test_a_shared_rerun_keeps_the_folders_own_tooling_recorded(gui, tmp_path, monkeypatch):
    """A folder set up Portable and then again Shared: its own .python, .cache
    and .uv stay in the record, so uninstall still removes them, and nothing in
    the Shared store is claimed by this folder."""
    for name in (".python", ".cache", ".uv"):
        (tmp_path / name).mkdir()
    gui.install_manifest().record(tmp_path, **gui._tooling_fields(tmp_path, True))
    for d in gui.store_dirs(False).values():
        d.mkdir(parents=True, exist_ok=True)
    seen = []
    monkeypatch.setattr(gui, "_run", lambda cmd, emit, plan, **kw: seen.append(
        [str(c) for c in cmd]) or 0)
    _step(gui, gui.Plan(backend="own", portable_store=False),
          "Writing the install record").run(lambda s: None)
    rec = seen[0]
    assert "--python-dir" not in rec and "--runtime-contained" not in rec
    record = _record(tmp_path)
    assert Path(record["python_dir"]) == (tmp_path / ".python").resolve()
    assert record["runtime_contained"] is True
