# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh steps that depend on the platform or on --yes: the "provision later"
hints, the plugin step under --yes, and the launcher and application menu
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
    result = subprocess.run([_bash(), "-c", "set -euo pipefail\n" + script],
                            capture_output=True, text=True, cwd=cwd, env=env, timeout=20)
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

def _hint_block() -> str:
    return _block('# The "provision later" hint.\n', "\nesac\n")


@pytest.mark.parametrize("spec, expected", [
    ("metal|cpu", "<metal|cpu>"),
    ("cpu", "cpu"),
    ("vulkan|cuda|hip|sycl|cpu", "<vulkan|cuda|hip|sycl|cpu>"),
    ("backend", "<backend>"),
    ("", "<backend>"),
])
def test_provision_later_hint_names_the_offered_backends(tmp_path, spec, expected):
    out = _run(f'BACKEND_HINT="{spec}"\n' + _hint_block() + 'printf "%s\\n" "$SL_BACKENDS"\n',
               tmp_path).stdout.strip()
    assert out == expected


def test_the_hint_comes_from_the_spec(tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "python"
    stub.write_text('#!/bin/sh\n[ "$3" = backend-hint ] && echo "vulkan|cpu"\nexit 0\n',
                    encoding="utf-8")
    _mark_executable(stub)
    line = _block('BACKEND_HINT="$(.venv/bin/python -m localm.setup_spec backend-hint',
                  "\n")
    out = _run(line + _hint_block() + 'printf "%s\\n" "$SL_BACKENDS"\n', tmp_path).stdout
    assert out.strip() == "<vulkan|cpu>"


@pytest.mark.parametrize("os_name, arch, is_macos, is_apple", [
    ("Darwin", "arm64", "1", "1"),
    ("Darwin", "x86_64", "1", "0"),
    ("Linux", "x86_64", "0", "0"),
    ("Linux", "aarch64", "0", "0"),
])
def test_platform_flags_follow_uname(tmp_path, os_name, arch, is_macos, is_apple):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_uname_stub(bin_dir, os_name=os_name, arch=arch)
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    block = _block("IS_MACOS=0\n", 'if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; '
                   'then IS_MACOS=1; fi\n')
    out = _run(block + 'printf "%s\\n" "$IS_MACOS"\n', tmp_path, env).stdout.strip()
    assert out == is_macos
    assert "IS_APPLE_SILICON" not in SRC


def test_every_provision_later_hint_uses_the_offered_list():
    assert "<vulkan|cuda|hip|sycl|cpu>" not in SRC
    assert SRC.count("setup-llama --backend $SL_BACKENDS") == 3


# ------------------------------ plugin picker ------------------------------ #

def _plugin_block() -> str:
    return _block('if [ "$LOCALM_BIN_OK" = 1 ] && [ "$YES" = 1 ]; then\n', "\nfi\n")


@pytest.mark.parametrize("yes, expected", [
    (1, "localm plugin setup --defaults"),
    (0, "localm plugin setup"),
])
def test_yes_installs_the_default_plugin_set(tmp_path, yes, expected):
    calls = _recording_localm(tmp_path)
    _run(f'say() {{ printf "%s\\n" "$*"; }}\nYES={yes}\nLOCALM_BIN_OK=1\n'
         + _plugin_block(), tmp_path)
    assert calls.read_text(encoding="utf-8").strip() == expected


def test_a_failed_plugin_step_does_not_stop_setup(tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "localm"
    stub.write_text("#!/bin/sh\nexit 4\n", encoding="utf-8")
    _mark_executable(stub)
    out = _run('say() { printf "%s\\n" "$*"; }\nYES=1\nLOCALM_BIN_OK=1\n'
               + _plugin_block(), tmp_path).stdout
    assert "[!] Skipped - choose later with:  .venv/bin/localm plugin setup" in out


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


def _run_menu_entry(tmp_path: Path, *, answer: str, is_macos: int = 0,
                    branded: bool = False, winmode: str = "in your browser"):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _mark_executable(python)
    if branded:
        exe = tmp_path / ".venv" / "bin" / "LocaLM"
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        _mark_executable(exe)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        f'ask() {{ printf "%s" "$1" >&2; a="{answer}"; echo "${{a:-$2}}"; }}\n'
        'jr() { :; }\nstep_begin() { :; }\nstep_done() { :; }\n'
        f'HOME="{home.as_posix()}"\nIS_MACOS={is_macos}\nWINMODE="{winmode}"\n'
        + _menu_entry_block()
        + '\nprintf "SHORTCUT=%s\\n" "$SHORTCUT"\n'
    )
    result = _run(script, tmp_path)
    apps = home / ".local" / "share" / "applications"
    return result, apps


def _here(tmp_path: Path) -> str:
    """$(pwd) for *tmp_path* as bash prints it."""
    return _run('printf "%s" "$(pwd)"\n', tmp_path).stdout


def _exec_line(apps: Path) -> str:
    text = (apps / "localm.desktop").read_text(encoding="utf-8")
    return next(ln for ln in text.splitlines() if ln.startswith("Exec="))[5:]


@pytest.mark.parametrize("is_macos", [1, 0])
def test_the_menu_entry_is_offered_on_linux_only(tmp_path, is_macos):
    result, apps = _run_menu_entry(tmp_path, answer="1", is_macos=is_macos)
    asked = "Pick 1, 2 or 3 [1]: " in result.stderr
    if is_macos:
        assert not asked and not (apps / "localm.desktop").exists()
        assert "Create an application menu entry?" not in result.stdout
    else:
        assert asked and (apps / "localm.desktop").exists()


def test_the_question_offers_the_three_choices_setup_bat_does(tmp_path):
    result, _apps = _run_menu_entry(tmp_path, answer="3", winmode="in its own app window")
    for line in ("  Create an application menu entry?",
                 "    [1] LocaLM launcher - choose GUI / chat / server / coder each time you start",
                 "    [2] Straight to the GUI - skips that menu, opens in its own app window",
                 "    [3] No menu entry"):
        assert line in result.stdout.splitlines()


@pytest.mark.parametrize("answer", ["1", ""])
def test_choice_1_and_the_default_open_the_launcher(tmp_path, answer):
    result, apps = _run_menu_entry(tmp_path, answer=answer)
    here = _here(tmp_path)
    assert _exec_line(apps) == f"{here}/localm-launcher.sh"
    text = (apps / "localm.desktop").read_text(encoding="utf-8")
    assert f"Path={here}" in text.splitlines()
    assert f"SHORTCUT={apps.as_posix()}/localm.desktop" in result.stdout


@pytest.mark.parametrize("branded, target", [
    (True, "/.venv/bin/LocaLM -m localm gui"),
    (False, "/.venv/bin/localm gui"),
])
def test_choice_2_opens_the_gui(tmp_path, branded, target):
    _result, apps = _run_menu_entry(tmp_path, answer="2", branded=branded)
    assert _exec_line(apps) == _here(tmp_path) + target


def test_choice_3_creates_nothing(tmp_path):
    result, apps = _run_menu_entry(tmp_path, answer="3")
    assert "No menu entry created." in result.stdout
    assert not apps.exists() or not any(apps.iterdir())
    assert result.stdout.splitlines()[-1] == "SHORTCUT="


def _case_sensitive(path: Path) -> bool:
    probe = path / "CaseProbe"
    probe.write_text("x", encoding="utf-8")
    try:
        return not (path / "caseprobe").exists()
    finally:
        probe.unlink()


def test_an_entry_the_window_wrote_for_this_folder_is_replaced(tmp_path):
    home = tmp_path / "home"
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    old = apps / "LocaLM.desktop"
    old.write_text(f"[Desktop Entry]\nExec={_here(tmp_path)}/.venv/bin/localm gui\n"
                   f"Path={_here(tmp_path)}\n", encoding="utf-8")
    result, apps = _run_menu_entry(tmp_path, answer="1")
    assert (f"  Removed the older menu entry {apps.as_posix()}/LocaLM.desktop for this folder."
            in result.stdout.splitlines())
    assert sorted(p.name for p in apps.iterdir()) == ["localm.desktop"]
    assert _exec_line(apps).endswith("/localm-launcher.sh")


def test_an_entry_for_another_folder_is_left_alone(tmp_path):
    home = tmp_path / "home"
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    if not _case_sensitive(apps):
        pytest.skip("LocaLM.desktop and localm.desktop are one file on this filesystem")
    other = apps / "LocaLM.desktop"
    other.write_text("[Desktop Entry]\nExec=/somewhere/else/localm-launcher.sh\n",
                     encoding="utf-8")
    result, apps = _run_menu_entry(tmp_path, answer="1")
    assert "Removed the older menu entry" not in result.stdout
    assert other.read_text(encoding="utf-8").endswith("/somewhere/else/localm-launcher.sh\n")
    assert (apps / "localm.desktop").exists()


# ------------------------------ setup-llama call --------------------------- #

def _setup_llama_call_block() -> str:
    return _block('  SL_YES=""\n', 'failed during setup."\n')


@pytest.mark.parametrize("yes, expected", [
    (1, "localm setup-llama --backend vulkan --yes"),
    (0, "localm setup-llama --backend vulkan"),
])
def test_yes_is_passed_on_to_setup_llama(tmp_path, yes, expected):
    calls = _recording_localm(tmp_path)
    _run(f'handle_provision_failure() {{ :; }}\nYES={yes}\nBACKEND=vulkan\nSL_FORCE=""\n'
         + _setup_llama_call_block(), tmp_path)
    assert calls.read_text(encoding="utf-8").strip() == expected


def test_an_entry_for_a_folder_that_only_shares_the_prefix_is_left_alone(tmp_path):
    home = tmp_path / "home"
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    if not _case_sensitive(apps):
        pytest.skip("LocaLM.desktop and localm.desktop are one file on this filesystem")
    other = apps / "LocaLM.desktop"
    other.write_text(f"[Desktop Entry]\nPath={_here(tmp_path)}-dev\n", encoding="utf-8")
    result, apps = _run_menu_entry(tmp_path, answer="1")
    assert "Removed the older menu entry" not in result.stdout
    assert other.exists()


@pytest.mark.parametrize("answer", ["y", "Y", "yes"])
def test_a_habitual_yes_at_the_menu_entry_question_takes_the_default(tmp_path, answer):
    result, apps = _run_menu_entry(tmp_path, answer=answer)
    assert "  (y taken as [1])" in result.stdout.splitlines()
    assert _exec_line(apps).endswith("/localm-launcher.sh")
