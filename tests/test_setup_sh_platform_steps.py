# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh steps that depend on the platform or on --yes: the "provision later"
hints, the plugin picker under --yes, and the launcher and application menu
entry, which are Linux-only.

Each test extracts one block of setup.sh and runs it in bash with stub commands,
so no install happens."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.test_setup_gpu_detect import SETUP_SH, _bash, _make_uname_stub, _mark_executable

pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")

SRC = SETUP_SH.read_text(encoding="utf-8")


def _block(start: str, end: str) -> str:
    """setup.sh from the line *start* up to and including the first *end* after it."""
    i = SRC.index(start)
    j = SRC.index(end, i + len(start)) + len(end)
    return SRC[i:j]


def _run(script: str, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run([_bash(), "-c", script], capture_output=True, text=True,
                            cwd=cwd, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
    return result


def _recording_localm(root: Path) -> Path:
    """A stub .venv/bin/localm (and python) that appends its arguments to calls.txt."""
    bin_dir = root / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("localm", "python"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{(root / "calls.txt").as_posix()}"\n',
                        encoding="utf-8")
        _mark_executable(stub)
    return root / "calls.txt"


# --------------------------- provision-later hint -------------------------- #

def _platform_flags_block() -> str:
    return _block("IS_APPLE_SILICON=0\n", 'SL_BACKENDS="<vulkan|cuda|hip|sycl|cpu>"\nfi\n')


@pytest.mark.parametrize("os_name, arch, expected", [
    ("Darwin", "arm64", "<metal|cpu>"),
    ("Darwin", "x86_64", "cpu"),
    ("Linux", "x86_64", "<vulkan|cuda|hip|sycl|cpu>"),
    ("Linux", "aarch64", "<vulkan|cuda|hip|sycl|cpu>"),
])
def test_provision_later_hint_names_this_platforms_backends(tmp_path, os_name, arch, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_uname_stub(bin_dir, os_name=os_name, arch=arch)
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    out = _run(_platform_flags_block() + 'printf "%s|%s|%s\\n" "$IS_MACOS" "$IS_APPLE_SILICON" '
               '"$SL_BACKENDS"\n', tmp_path, env).stdout.strip()
    is_macos, is_apple, backends = out.split("|")
    assert backends == expected
    assert is_macos == ("1" if os_name == "Darwin" else "0")
    assert is_apple == ("1" if (os_name, arch) == ("Darwin", "arm64") else "0")


def test_every_provision_later_hint_uses_the_platform_list():
    assert SRC.count("<vulkan|cuda|hip|sycl|cpu>") == 1
    assert SRC.count("setup-llama --backend $SL_BACKENDS") == 3


# ------------------------------ plugin picker ------------------------------ #

def _plugin_block() -> str:
    return _block('if [ "$LOCALM_BIN_OK" = 1 ] && [ "$YES" = 1 ]; then\n', "\nfi\n")


@pytest.mark.parametrize("yes, runs_picker", [(1, False), (0, True)])
def test_yes_skips_the_interactive_plugin_picker(tmp_path, yes, runs_picker):
    calls = _recording_localm(tmp_path)
    out = _run(f'say() {{ printf "%s\\n" "$*"; }}\nYES={yes}\nLOCALM_BIN_OK=1\n'
               + _plugin_block(), tmp_path).stdout
    ran = calls.exists() and "localm plugin setup" in calls.read_text(encoding="utf-8")
    assert ran is runs_picker
    if not runs_picker:
        assert "Skipped (--yes) - choose later with:  .venv/bin/localm plugin setup" in out


# ------------------------ launcher and menu entry -------------------------- #

def _launcher_block() -> str:
    return _block('LAUNCHER_FILE=""\n', "\nfi\n")


@pytest.mark.parametrize("is_macos, builds", [(1, False), (0, True)])
def test_the_launcher_is_built_on_linux_only(tmp_path, is_macos, builds):
    calls = _recording_localm(tmp_path)
    out = _run(f'say() {{ printf "%s\\n" "$*"; }}\nIS_MACOS={is_macos}\nLOCALM_BIN_OK=1\n'
               + _launcher_block(), tmp_path).stdout
    ran = calls.exists() and "make-launcher" in calls.read_text(encoding="utf-8")
    assert ran is builds
    assert ("no app launcher or menu entry is created" in out) is (not builds)


def _menu_entry_block() -> str:
    return _block('SHORTCUT=""\n', '\nesac\n')


@pytest.mark.parametrize("is_macos", [1, 0])
def test_the_menu_entry_is_offered_on_linux_only(tmp_path, is_macos):
    _recording_localm(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { printf "%s" "$1" >&2; echo Y; }\n'
        'jr() { :; }\nstep_begin() { :; }\nstep_done() { :; }\n'
        f'HOME="{home.as_posix()}"\nIS_MACOS={is_macos}\n'
        + _menu_entry_block()
    )
    result = _run(script, tmp_path)
    entry = home / ".local" / "share" / "applications" / "localm.desktop"
    asked = "application menu entry" in result.stderr
    if is_macos:
        assert not asked and not entry.exists()
        assert "No desktop entry created." not in result.stdout
    else:
        assert asked and entry.exists()
