# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat run from a folder whose name holds characters cmd.exe treats as syntax
(``&``, ``!``, ``%``, ``^``, parentheses, spaces): the banners name the folder
exactly and nothing in the name is run as a command."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_BAT = ROOT / "setup.bat"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup.bat runs under cmd.exe")

NAMES = [
    "Lo cal&M!x",
    "a&b",
    "one&echo INJECTED&two",
    "bang!x",
    "pct%y",
    "caret^z",
    "(paren)s",
    "all & of ! them % ^ (here)",
    "it's here",
    "a^b!c",
]

AWKWARD = [
    "it's here",
    "a^b!c",
    "a&b'c!d%e^f (g)",
    "café '&!%^",
]


def _run(folder: Path, *args: str, abort_after: str | None = None, answers: str = "\n\n\n\n"):
    env = dict(os.environ)
    env.pop("LOCALM_SETUP_ABORT_AFTER", None)
    if abort_after:
        env["LOCALM_SETUP_ABORT_AFTER"] = abort_after
    captured = folder.parent / (folder.name.encode("ascii", "replace").hex() + ".out.txt")
    with open(captured, "wb") as sink:
        r = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", r".\setup.bat", *args],
                           cwd=str(folder), env=env,
                           input=answers.replace("\n", "\r\n").encode("utf-8"),
                           stdout=sink, stderr=subprocess.STDOUT, timeout=120)
    return r.returncode, captured.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(params=NAMES)
def folder(request, tmp_path):
    d = tmp_path / request.param
    d.mkdir()
    shutil.copy(SETUP_BAT, d / "setup.bat")
    return d


def test_the_install_banner_names_the_folder_and_runs_nothing(folder):
    rc, out = _run(folder, abort_after="portable-choice")
    assert rc == 99, out
    assert f"LocaLM setup - self-contained install in: \"{folder}\"" in out
    assert "is not recognized" not in out
    assert "INJECTED" not in out.replace(str(folder), "")


def test_the_uninstall_banner_names_the_folder_and_runs_nothing(folder):
    _rc, out = _run(folder, "uninstall", "--yes")
    assert f'   "{folder}"' in out
    assert "is not recognized" not in out
    assert "INJECTED" not in out.replace(str(folder), "")


def _oem_encoding() -> str:
    import ctypes
    return f"cp{ctypes.windll.kernel32.GetOEMCP()}"


def _bootstrap(bat: str) -> str:
    start = bat.index('cd /d "%~dp0"')
    end = bat.index("set LOCALM_SETUP=1") + len("set LOCALM_SETUP=1")
    return bat[start:end]


def _awkward_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


def _run_probe(folder: Path, body: str, env_extra: dict[str, str] | None = None):
    text = ("@echo off\n" + body).replace("\r\n", "\n").replace("\n", "\r\n")
    (folder / "probe.bat").write_bytes(text.encode(_oem_encoding(), "replace"))
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", r".\probe.bat"], cwd=str(folder),
                          env=env, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)


def _decode(raw: bytes) -> str:
    return raw.decode(_oem_encoding(), "replace")


def _shortcut_block(bat: str, pick: str) -> str:
    first = bat.index('if "%SCPICK%"=="1" (')
    second = bat.index('if "%SCPICK%"=="2" (')
    third = bat.index('if "%SCPICK%"=="3"')
    block = bat[first:second] if pick == "1" else bat[second:third]
    assert "[Environment]::GetFolderPath('Desktop')" in block
    return block.replace("[Environment]::GetFolderPath('Desktop')", "$env:W26_DESKTOP")


def _subroutine(bat: str, label: str) -> str:
    start = bat.index(f"\n:{label}\n") + 1
    end = bat.index("exit /b 0", start) + len("exit /b 0")
    return bat[start:end]


def _read_shortcut(lnk: Path) -> dict[str, str]:
    out = lnk.parent / "lnk.json"
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:W26_LNK);"
              "$o = @{target=$s.TargetPath; cwd=$s.WorkingDirectory; icon=$s.IconLocation; args=$s.Arguments};"
              "[IO.File]::WriteAllText($env:W26_OUT, ($o | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60,
                   env={**os.environ, "W26_LNK": str(lnk), "W26_OUT": str(out)})
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bat_text():
    return SETUP_BAT.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("name", NAMES + AWKWARD)
def test_the_install_folders_keep_the_exact_folder_name(tmp_path, bat_text, name):
    start = bat_text.index('set "CONTAINED=0"')
    end = bat_text.index("call :step_done portable-choice")
    region = bat_text[start:end].replace("call :flush\n", "").replace(
        'set /p "STOREPICK=  Pick 1 or 2 [1]: "\n', "")
    folder = _awkward_dir(tmp_path, name)
    out_file = tmp_path / "vars.txt"
    probe = (_bootstrap(bat_text) + '\nset "STOREPICK=1"\n' + region +
             "\nsetlocal DisableDelayedExpansion\n"
             "(set PYDIR& set CACHEDIR& set UV_PYTHON_INSTALL_DIR& set UV_CACHE_DIR& "
             f'set LOCALM_SETUP_ROOT) > "{out_file}"\n')
    r = _run_probe(folder, probe)
    got = dict(line.split("=", 1) for line in _decode(out_file.read_bytes()).splitlines() if "=" in line)
    assert got.get("PYDIR") == f"{folder}\\.python", (got, _decode(r.stdout + r.stderr))
    assert got.get("CACHEDIR") == f"{folder}\\.cache", got
    assert got.get("UV_PYTHON_INSTALL_DIR") == f"{folder}\\.python", got
    assert got.get("UV_CACHE_DIR") == f"{folder}\\.cache", got
    assert got.get("LOCALM_SETUP_ROOT") == str(folder), got


@pytest.mark.parametrize("pick", ["1", "2"])
@pytest.mark.parametrize("name", NAMES[:3] + AWKWARD)
def test_the_desktop_shortcut_points_into_the_folder_whatever_its_name(tmp_path, bat_text, pick, name):
    block = _shortcut_block(bat_text, pick)
    folder = _awkward_dir(tmp_path, name)
    desktop = tmp_path / "desk"
    desktop.mkdir()
    probe = _bootstrap(bat_text) + f'\nset "SCPICK={pick}"\nset "SCPATH="\n' + block + "\n"
    r = _run_probe(folder, probe, {"W26_DESKTOP": str(desktop)})
    lnk = desktop / "LocaLM.lnk"
    assert lnk.is_file(), _decode(r.stdout + r.stderr)
    info = _read_shortcut(lnk)
    assert info["cwd"] == str(folder), info
    assert info["icon"].startswith(f"{folder}\\assets\\localm.ico"), info
    if pick == "1":
        assert info["target"] == f"{folder}\\localm-launcher.bat", info
    else:
        assert info["target"] == f"{folder}\\.venv\\Scripts\\localm.exe", info
        assert info["args"] == "gui", info


def test_the_app_shortcut_prefers_the_branded_exe_in_an_awkward_folder(tmp_path, bat_text):
    block = _shortcut_block(bat_text, "2")
    folder = _awkward_dir(tmp_path, "it's here")
    exe = folder / ".venv" / "localm-app" / "LocaLM.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    desktop = tmp_path / "desk"
    desktop.mkdir()
    probe = _bootstrap(bat_text) + '\nset "SCPICK=2"\nset "SCPATH="\n' + block + "\n"
    _run_probe(folder, probe, {"W26_DESKTOP": str(desktop)})
    info = _read_shortcut(desktop / "LocaLM.lnk")
    assert info["target"] == str(exe), info
    assert info["args"] == "-m localm gui", info


@pytest.mark.parametrize("name", ["plain", "it's here", "a^b!c", "a&b'c!d%e^f (g)"])
def test_the_onedrive_probe_recognises_a_folder_under_a_synced_root(tmp_path, bat_text, name):
    sync_root = tmp_path / "sync root"
    sync_root.mkdir()
    folder = sync_root / name
    folder.mkdir()
    out_file = tmp_path / "od.txt"
    probe = (_bootstrap(bat_text) + "\ncall :onedrive_root\n"
             f'setlocal DisableDelayedExpansion\n(set ONEDRIVE_ROOT) > "{out_file}"\nexit /b 0\n' +
             _subroutine(bat_text, "onedrive_root") + "\n")
    _run_probe(folder, probe, {"OneDrive": str(sync_root), "OneDriveConsumer": "", "OneDriveCommercial": ""})
    assert _decode(out_file.read_bytes()).strip() == f"ONEDRIVE_ROOT={sync_root}"


@pytest.mark.parametrize("name", ["plain", "it's here", "a^b!c"])
def test_the_locker_probe_finds_a_process_running_from_the_venv(tmp_path, bat_text, name):
    folder = _awkward_dir(tmp_path, name)
    scripts = folder / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "cmd.exe"
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Copy-Item -LiteralPath (Join-Path $env:SystemRoot 'System32/cmd.exe') -Destination $env:W26_DEST"],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60,
                   env={**os.environ, "W26_DEST": str(runner)})
    stmp = folder / ".localm-setup-tmp"
    stmp.mkdir()
    child = subprocess.Popen([str(runner), "/c", "ping -n 40 127.0.0.1 >nul"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        probe = (_bootstrap(bat_text) + '\nset "STMP=.localm-setup-tmp"\ncall :find_venv_lockers\n'
                 "setlocal DisableDelayedExpansion\n"
                 "if defined LOCKERS (echo LOCKERS=yes) else (echo LOCKERS=no)\nexit /b 0\n" +
                 _subroutine(bat_text, "find_venv_lockers") + "\n")
        r = _run_probe(folder, probe)
        assert "LOCKERS=yes" in _decode(r.stdout), _decode(r.stdout + r.stderr)
        listed = (stmp / "localm_lockers.txt").read_text(encoding="utf-8", errors="replace")
        assert f"{child.pid}|cmd.exe" in listed
    finally:
        child.kill()
        child.wait()


def test_no_path_is_pasted_into_a_single_quoted_powershell_string(bat_text):
    assert not re.search(r"'[^'\r\n]*%CD%", bat_text)
    assert "%CD:!=^!%" not in bat_text
