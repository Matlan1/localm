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
