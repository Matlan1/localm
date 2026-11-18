# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first install of every setup variant takes localm's packages from uv.lock,
and a Portable install keeps uv's scratch files inside the folder.

setup.bat, setup.sh and installer/gui.py (which setup-gui.bat and setup-gui.sh
open) used to run ``uv pip install -e`` and so resolved the newest versions
pyproject.toml allows, unchecked against uv.lock, while CI tests the locked set
and ``localm update --repair`` installs it. They now ask
``localm.setup_spec install-args`` for the ``uv sync --frozen`` arguments, the
same ones an update uses.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from localm import pyinstall, setup_spec

ROOT = Path(__file__).resolve().parents[1]
SETUP_BAT = (ROOT / "setup.bat").read_text(encoding="utf-8")
SETUP_SH = (ROOT / "setup.sh").read_text(encoding="utf-8")
GUI_SRC = (ROOT / "installer" / "gui.py").read_text(encoding="utf-8")


# ------------------------------- install-args ------------------------------ #

def test_install_args_are_the_updates_locked_sync_with_the_setup_extras(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    args = setup_spec.install_args(setup_spec.extras(True), tmp_path)
    assert args == pyinstall.sync_command("uv", ["coder", "voice", "monitor", "desktop"])[1:]
    assert args[:4] == ["sync", "--frozen", "--inexact", "--no-config"]
    assert args.count("--extra") == 4
    assert setup_spec.LOCK_NAME == pyinstall.LOCK_NAME
    assert setup_spec.lock_file(tmp_path) == pyinstall.lock_file(tmp_path)


def test_install_args_are_none_without_a_lock(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert setup_spec.install_args("coder", tmp_path) is None


def test_the_cli_prints_the_arguments_and_fails_loudly_without_a_lock(tmp_path):
    with_lock = subprocess.run(
        [sys.executable, "-m", "localm.setup_spec", "install-args", "coder,voice"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert with_lock.returncode == 0, with_lock.stderr
    assert with_lock.stdout.split() == [
        "sync", "--frozen", "--inexact", "--no-config",
        "--extra", "coder", "--extra", "voice"]

    bare = subprocess.run(
        [sys.executable, "-m", "localm.setup_spec", "install-args", "coder"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert bare.returncode == 1
    assert bare.stdout == ""
    assert "uv.lock was not found" in bare.stderr


def test_every_extra_setup_installs_is_in_the_lock():
    """An extra added to setup and not to uv.lock would make the locked install
    fail; this ties the two together."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = next(p for p in lock["package"] if p["name"] == "localm")
    locked = set(project["optional-dependencies"])
    for extra in setup_spec.extras(True).split(","):
        assert extra in locked, f"{extra} is installed by setup but is not in uv.lock"


# ------------------------------ every variant ------------------------------ #

def test_each_setup_variant_installs_through_install_args():
    assert "localm.setup_spec install-args" in SETUP_BAT
    assert "localm.setup_spec install-args" in SETUP_SH
    assert "spec.install_args(" in GUI_SRC


def test_the_unlocked_install_is_only_the_loud_fallback():
    """In setup.bat and setup.sh the unlocked first install sits right after the
    message that says it is not the locked set; in the window it follows the
    emit of the same message."""
    for text, fallback in (
        (SETUP_BAT, 'uv pip install -p .venv -e ".[%EXTRAS%]"'),
        (SETUP_SH, 'uv pip install -p .venv -e ".[${EXTRAS}]"'),
    ):
        first = text.index(fallback)
        head = text[max(0, first - 400):first]
        assert "instead of the locked set" in head, head
    window = GUI_SRC.index('_run(uv_argv("pip", "install", "-p", ".venv", "-e", f".[{plan.extras}]")')
    assert "instead of the locked set" in GUI_SRC[window - 400:window]


# -------------------------- Portable keeps scratch --------------------------- #

@pytest.fixture()
def gui(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "localm_installer_gui_lock", ROOT / "installer" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["localm_installer_gui_lock"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for key in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(key, str(tmp_path / "system-temp"))
    return mod


def test_the_window_points_a_portable_installs_temp_folders_into_the_folder(gui, tmp_path):
    env = gui._env_for(gui.Plan(portable_store=True))
    scratch = tmp_path / ".cache" / "tmp"
    for key in ("TEMP", "TMP", "TMPDIR"):
        assert env[key] == str(scratch)
    assert scratch.is_dir()


def test_the_window_leaves_a_shared_installs_temp_folders_alone(gui, tmp_path):
    env = gui._env_for(gui.Plan(portable_store=False))
    for key in ("TEMP", "TMP", "TMPDIR"):
        assert env[key] == str(tmp_path / "system-temp")
    assert not (tmp_path / ".cache" / "tmp").exists()


def test_the_console_setups_point_a_portable_installs_temp_folders_into_the_folder():
    store_choice = SETUP_BAT.split('if "%STOREPICK%"=="1" (')[1]
    portable_bat = store_choice.split(") else (")[0]
    assert 'set "TEMP=!CD!\\.cache\\tmp"' in portable_bat
    assert 'set "TMP=!CD!\\.cache\\tmp"' in portable_bat
    portable_sh = SETUP_SH.split('if [ "$spick" = 1 ]; then')[1].split("\nelse\n")[0]
    assert 'export TMPDIR="$(pwd)/.cache/tmp"' in portable_sh
    shared_bat = store_choice.split(") else (")[1].split("\n)\n")[0]
    assert "TEMP=" not in shared_bat
