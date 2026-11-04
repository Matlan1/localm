# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup never edits a shell startup file or the user PATH to install uv.

Astral's installer adds its install folder to the shell startup files (Linux and
macOS) or the user PATH in the registry (Windows) unless UV_NO_MODIFY_PATH is
set. Every setup variant sets it for the installer, uses the uv it just
installed by full path for the rest of the run, and finds that uv again on a
later run although it is on no PATH.

The real scripts run from a throwaway clone with the pinned installer replaced
by a stub that behaves the way Astral's does: it honours UV_NO_MODIFY_PATH and,
without it, edits the startup files (or writes a marker standing in for the
registry write on Windows). No network is touched and no real profile, PATH or
registry value is read or written.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests._exe_stub import write_exe_stub
from tests.test_setup_uv_installer_temp import (
    _BASH, _layout, _native, _needs_bash, _needs_windows, _NO_WINDOW, _pinned_copy,
)

ROOT = Path(__file__).resolve().parents[1]
_STARTUP_FILES = (".bashrc", ".profile", ".zshrc")
_ORIGINAL = "# my own startup file\nexport EDITOR=vi\n"

_SH_INSTALLER = (
    b'#!/bin/sh\n'
    b'printf "NO_MODIFY_PATH=[%s]\\n" "${UV_NO_MODIFY_PATH:-}" >> "$STUB_LOG"\n'
    b'dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"\n'
    b'mkdir -p "$dest"\n'
    b'printf "#!/bin/sh\\n" > "$dest/uv"\n'
    b'chmod +x "$dest/uv"\n'
    b'if [ -z "${UV_NO_MODIFY_PATH:-}" ] && [ -z "${UV_UNMANAGED_INSTALL:-}" ]; then\n'
    b'  for f in .bashrc .profile .zshrc; do\n'
    b'    printf "export PATH=\\"$dest:\\$PATH\\"\\n" >> "$HOME/$f"\n'
    b'  done\n'
    b'fi\n'
)
_PS_INSTALLER = (
    b'Add-Content -LiteralPath $env:STUB_LOG -Value ("NO_MODIFY_PATH=[" + $env:UV_NO_MODIFY_PATH + "]")\r\n'
    b'$dest = $env:UV_INSTALL_DIR\r\n'
    b'if (-not $dest) { $dest = Join-Path $env:USERPROFILE ".local\\bin" }\r\n'
    b'$null = New-Item -ItemType Directory -Force -Path $dest\r\n'
    b'Copy-Item -LiteralPath $env:STUB_EXE -Destination (Join-Path $dest "uv.exe")\r\n'
    b'if (-not $env:UV_NO_MODIFY_PATH -and -not $env:UV_UNMANAGED_INSTALL) {\r\n'
    b'  Add-Content -LiteralPath (Join-Path $env:USERPROFILE "user-path-edited.txt") -Value $dest\r\n'
    b'}\r\n'
)


def _seed_startup_files(home: Path) -> dict[str, bytes]:
    for name in _STARTUP_FILES:
        (home / name).write_bytes(_ORIGINAL.encode())
    return {name: (home / name).read_bytes() for name in _STARTUP_FILES}


def _recorded(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8-sig").split() if log.is_file() else []


def _sh_env(tmp_path: Path, home: Path, log: Path, releases: Path) -> dict:
    stubs = tmp_path / "stubbin"
    stubs.mkdir(exist_ok=True)
    curl = stubs / "curl"
    curl.write_bytes((
        '#!/bin/sh\nout=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift; fi\n'
        '  shift\ndone\n'
        f'cp "{releases.as_posix()}/uv-installer.sh" "$out"\n').encode())
    curl.chmod(0o755)
    path = [str(stubs), str(Path(_BASH).parent)] + (
        [] if os.name == "nt" else ["/usr/bin", "/bin"])
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_NO_MODIFY_PATH", "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR",
                "XDG_BIN_HOME", "XDG_DATA_HOME"):
        env.pop(var, None)
    env.update(HOME=str(home), STUB_LOG=log.as_posix(), PATH=os.pathsep.join(path),
               TMPDIR=str(tmp_path / "systemtemp"))
    (tmp_path / "systemtemp").mkdir(exist_ok=True)
    return env


def _setup_sh(clone: Path, env: dict, answers: str, abort_after: str):
    env = dict(env, LOCALM_SETUP_ABORT_AFTER=abort_after)
    r = subprocess.run([_BASH, "setup.sh"], cwd=str(clone), env=env,
                       input=answers.encode("utf-8"), capture_output=True, timeout=180,
                       creationflags=_NO_WINDOW)
    return r.returncode, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


def _sh_clone(tmp_path: Path, name: str):
    clone, _, home, log = _layout(tmp_path, name)
    releases = tmp_path / "releases"
    _pinned_copy("setup.sh", clone, releases, _SH_INSTALLER, bat=False)
    return clone, home, log, _sh_env(tmp_path, home, log, releases)


@_needs_bash
def test_setup_sh_shared_install_leaves_the_shell_startup_files_alone(tmp_path):
    clone, home, log, env = _sh_clone(tmp_path, "clone")
    before = _seed_startup_files(home)
    rc, out = _setup_sh(clone, env, "2\nY\n", "uv-bootstrap")
    assert rc == 99 and "stopping after 'uv-bootstrap'" in out, out
    assert _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
    assert (home / ".local" / "bin" / "uv").is_file(), out
    assert {n: (home / n).read_bytes() for n in _STARTUP_FILES} == before


@_needs_bash
def test_setup_sh_shared_install_creates_no_startup_file_that_was_missing(tmp_path):
    clone, home, log, env = _sh_clone(tmp_path, "clone")
    rc, out = _setup_sh(clone, env, "2\nY\n", "uv-bootstrap")
    assert rc == 99, out
    assert [n for n in _STARTUP_FILES if (home / n).exists()] == []


@_needs_bash
def test_setup_sh_shared_second_install_reuses_the_uv_that_is_on_no_path(tmp_path):
    clone, home, log, env = _sh_clone(tmp_path, "clone")
    rc, out = _setup_sh(clone, env, "2\nY\n", "uv-bootstrap")
    assert rc == 99 and _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
    uv = home / ".local" / "bin" / "uv"
    uv.write_bytes(b'#!/bin/sh\nprintf "%s\\n" "$0" >> "$UV_CALLS"\n'
                   b'case "$1" in venv) mkdir -p .venv ;; esac\n')
    calls = tmp_path / "uv-calls.txt"
    second = tmp_path / "second"
    second.mkdir()
    _pinned_copy("setup.sh", second, tmp_path / "releases-2", _SH_INSTALLER, bat=False)
    rc, out = _setup_sh(second, dict(env, UV_CALLS=calls.as_posix()), "2\nY\n", "venv")
    assert rc == 99 and "stopping after 'venv'" in out, out
    assert "is not installed" not in out and _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
    used = {_native(ln) for ln in calls.read_text().split()}
    assert used and {Path(p).resolve() for p in used} == {uv.resolve()}, out


@_needs_bash
def test_setup_sh_portable_install_still_stays_in_the_folder(tmp_path):
    clone, home, log, env = _sh_clone(tmp_path, "clone")
    before = _seed_startup_files(home)
    rc, out = _setup_sh(clone, env, "1\nY\n", "uv-portable")
    assert rc == 99, out
    assert (clone / ".uv" / "uv").is_file() and not (home / ".local").exists(), out
    assert {n: (home / n).read_bytes() for n in _STARTUP_FILES} == before


def _win_clone(tmp_path: Path, script: str, installer: bytes):
    clone, scratch, home, log = _layout(tmp_path, "clone")
    _pinned_copy(script, clone, tmp_path / "releases", installer, bat=True)
    return clone, scratch, home, log


def _win_env(scratch: Path, home: Path, log: Path, stub_exe: Path) -> dict:
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "LOCALM_GUI_STORE", "LOCALM_GUI_UV",
                "LOCALM_GUI_UV_INSTALLED", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_NO_MODIFY_PATH", "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "STMP",
                "XDG_BIN_HOME", "XDG_DATA_HOME"):
        env.pop(var, None)
    env.update(PATH=os.pathsep.join([str(system32), str(system32 / "WindowsPowerShell" / "v1.0")]),
               TEMP=str(scratch), TMP=str(scratch), USERPROFILE=str(home),
               HOMEDRIVE=home.drive, HOMEPATH=str(home)[len(home.drive):], STUB_LOG=str(log),
               STUB_EXE=str(stub_exe))
    return env


def _run_cmd(clone: Path, script: str, env: dict, answers: str) -> tuple[int, str]:
    proc = subprocess.Popen([os.environ.get("COMSPEC", "cmd.exe"), "/c", ".\\" + script],
                            cwd=str(clone), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=_NO_WINDOW)
    try:
        out, _ = proc.communicate(answers.replace("\n", "\r\n").encode("utf-8"), timeout=60)
    except subprocess.TimeoutExpired:
        subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"),
                        "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=_NO_WINDOW)
        out, _ = proc.communicate()
        pytest.fail(f"{script} did not finish:\n{out.decode('utf-8', 'replace')}")
    return proc.returncode, out.decode("utf-8", "replace")


@_needs_windows
def test_setup_bat_shared_install_leaves_the_user_path_alone(tmp_path):
    clone, scratch, home, log = _win_clone(tmp_path, "setup.bat", _PS_INSTALLER)
    env = _win_env(scratch, home, log, write_exe_stub(tmp_path / 'program' / 'stub.exe'))
    env["LOCALM_SETUP_ABORT_AFTER"] = "uv-bootstrap"
    rc, out = _run_cmd(clone, "setup.bat", env, "2\nY\n\n\n")
    assert rc == 99 and "stopping after 'uv-bootstrap'" in out, out
    assert _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
    assert (home / ".local" / "bin" / "uv.exe").is_file(), out
    assert not (home / "user-path-edited.txt").exists()


def _probe_default_uv_dir(tmp_path: Path, home: Path, **extra) -> str:
    text = (ROOT / "setup.bat").read_bytes().decode("utf-8")
    start = text.index("\r\n:add_default_uv_dir\r\n") + 2
    sub = text[start:text.index("goto :eof", start) + len("goto :eof")]
    probe = tmp_path / "probe.bat"
    probe.write_bytes(("@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
                       "call :add_default_uv_dir\r\nwhere uv\r\nexit /b 0\r\n" + sub + "\r\n").encode())
    env = _win_env(tmp_path / "scratch", home, tmp_path / "log.txt", tmp_path / "unused.exe")
    env.update(HOMEPATH="\\no-such-home", **extra)
    rc, out = _run_cmd(tmp_path, "probe.bat", env, "")
    return out.strip()


@_needs_windows
def test_setup_bat_puts_a_uv_in_the_install_folder_on_the_path_of_the_run_only(tmp_path):
    home = tmp_path / "home"
    uv = write_exe_stub(home / ".local" / "bin" / "uv.exe")
    assert _probe_default_uv_dir(tmp_path, home) == str(uv)


@_needs_windows
def test_setup_bat_finds_no_uv_when_the_install_folders_hold_none(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert "Could not find files" in _probe_default_uv_dir(tmp_path, home)


@_needs_windows
def test_setup_bat_looks_in_the_install_folders_before_it_installs_uv():
    text = (ROOT / "setup.bat").read_bytes().decode("utf-8")
    shared_check = text.index("call :add_default_uv_dir")
    assert shared_check < text.index(":uv_missing\r\n")


@_needs_windows
def test_setup_gui_bat_shared_install_leaves_the_user_path_alone(tmp_path):
    clone, scratch, home, log = _win_clone(tmp_path, "setup-gui.bat", _PS_INSTALLER)
    script = clone / "setup-gui.bat"
    run = '"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\\gui.py"'
    text = script.read_bytes().decode("utf-8")
    assert text.count(run) == 1, "the window launch line moved"
    script.write_bytes(text.replace(run, "echo WINDOW-UV=%UVEXE%& cmd /c exit 0").encode("utf-8"))
    env = _win_env(scratch, home, log, write_exe_stub(tmp_path / 'program' / 'stub.exe'))
    rc, out = _run_cmd(clone, "setup-gui.bat", env, "2\nY\n\n\n")
    assert _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
    assert f"WINDOW-UV={home / '.local' / 'bin' / 'uv.exe'}" in out, out
    assert not (home / "user-path-edited.txt").exists()
    rc, out = _run_cmd(clone, "setup-gui.bat", env, "2\nY\n\n\n")
    assert "is not installed" not in out, out
    assert f"WINDOW-UV={home / '.local' / 'bin' / 'uv.exe'}" in out, out
    assert _recorded(log) == ["NO_MODIFY_PATH=[1]"], out
