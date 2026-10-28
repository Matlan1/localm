# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing uv leaves nothing in the system temp folder.

Astral's installer downloads its archive into the temp folder of the process
that runs it (GetTempPath on Windows, ``mktemp -d`` on Linux and macOS) and does
not always remove it. Every setup variant points that temp folder into
``.localm-setup-tmp`` inside the install folder for the installer alone and
removes it afterwards.

Each real script runs from a throwaway clone with the system temp folder set to
an empty directory. The pinned installer is served from a local directory
through a file:// URL (Windows) or a stub ``curl`` (bash), and stands in for
Astral's: it writes a partial archive into its temp folder, as the real one
does, and drops a uv where the real one would. No network is used.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from localm import uv_tool

ROOT = Path(__file__).resolve().parents[1]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_needs_windows = pytest.mark.skipif(os.name != "nt", reason="the .bat setups run only on Windows")
_needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_HOST = "https://github.com/astral-sh/uv/releases/download/"
_PS_STUB = (
    b"$tmp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid())\r\n"
    b"$null = New-Item -ItemType Directory -Path $tmp\r\n"
    b"[IO.File]::WriteAllText((Join-Path $tmp 'uv.zip'), 'partial download')\r\n"
    b"Add-Content -LiteralPath $env:STUB_LOG -Value $tmp\r\n"
    b"$null = Get-Item -Path (Join-Path $tmp 'uv.zip') -ErrorAction Stop\r\n"
    b"$dest = $env:UV_INSTALL_DIR\r\n"
    b"if (-not $dest) { $dest = Join-Path $env:USERPROFILE '.local\\bin' }\r\n"
    b"$null = New-Item -ItemType Directory -Force -Path $dest\r\n"
    b"[IO.File]::WriteAllBytes((Join-Path $dest 'uv.exe'), [byte[]]@())\r\n"
)
_SH_STUB = (
    b'#!/bin/sh\n'
    b'tmp="$(mktemp -d)"\n'
    b'printf "partial download" > "$tmp/uv.tar.gz"\n'
    b'printf "%s\\n" "$tmp" >> "$STUB_LOG"\n'
    b'dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"\n'
    b'mkdir -p "$dest"\n'
    b'printf "#!/bin/sh\\n" > "$dest/uv"\n'
    b'chmod +x "$dest/uv"\n'
)


def _pinned_copy(name: str, clone: Path, releases: Path, stub: bytes, *, bat: bool) -> None:
    """Copies *name* into *clone* with the installer pin pointed at *stub*,
    served from *releases*."""
    sha = hashlib.sha256(stub).hexdigest()
    text = (ROOT / name).read_bytes().decode("utf-8")
    version = "0.13.6"
    if bat:
        pin_line = 'set "UV_INSTALLER_SHA256='
        new_pin = f'set "UV_INSTALLER_SHA256={sha}"'
        assert text.count(_HOST) == 1
        base = releases.as_uri() + "/"
        assert "%" not in base and "!" not in base
        text = text.replace(_HOST, base)
        (releases / version).mkdir(parents=True)
        (releases / version / "uv-installer.ps1").write_bytes(stub)
    else:
        pin_line = 'UV_INSTALLER_SHA256="'
        new_pin = f'UV_INSTALLER_SHA256="{sha}"'
        releases.mkdir(parents=True)
        (releases / "uv-installer.sh").write_bytes(stub)
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(pin_line)]
    assert len(hits) == 1, f"{name}: the installer pin moved"
    i = hits[0]
    ending = "\r\n" if lines[i].endswith("\r\n") else "\n"
    lines[i] = lines[i][:len(lines[i]) - len(lines[i].lstrip())] + new_pin + ending
    (clone / name).write_bytes("".join(lines).encode("utf-8"))


def _layout(tmp_path: Path, clone_name: str = "clone") -> tuple[Path, Path, Path, Path]:
    clone, scratch, home = tmp_path / clone_name, tmp_path / "systemtemp", tmp_path / "home"
    for d in (clone, scratch, home):
        d.mkdir()
    return clone, scratch, home, tmp_path / "stub.log"


def _installer_temps(log: Path) -> list[str]:
    assert log.is_file(), "the stub installer never ran"
    return [ln.strip() for ln in log.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]


def _windows_env(scratch: Path, home: Path, log: Path, **extra) -> dict:
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "LOCALM_GUI_STORE", "LOCALM_GUI_UV",
                "LOCALM_GUI_UV_INSTALLED", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "STMP"):
        env.pop(var, None)
    env.update(PATH=os.pathsep.join([str(system32), str(system32 / "WindowsPowerShell" / "v1.0")]),
               TEMP=str(scratch), TMP=str(scratch), USERPROFILE=str(home),
               HOMEDRIVE=home.drive, HOMEPATH=str(home)[len(home.drive):], STUB_LOG=str(log))
    env.update(extra)
    return env


def _cmd(script: Path, clone: Path, env: dict, answers: str) -> subprocess.CompletedProcess:
    return subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", str(script)], cwd=str(clone),
                          env=env, input=answers.replace("\n", "\r\n").encode("utf-8"),
                          capture_output=True, timeout=180, creationflags=_NO_WINDOW)


def _assert_nothing_left(scratch: Path, clone: Path, temps: list[str]) -> None:
    assert temps, "the stub installer recorded no temp folder"
    for t in temps:
        assert Path(t).resolve().is_relative_to((clone / ".localm-setup-tmp").resolve()), temps
        assert not Path(t).exists(), f"the installer's temp folder was left behind: {t}"
    assert list(scratch.iterdir()) == []


@_needs_windows
@pytest.mark.parametrize("pick,step,uv_exe", [
    ("1", "uv-portable", Path(".uv") / "uv.exe"),
    ("2", "uv-bootstrap", None),
])
def test_setup_bat_leaves_nothing_in_the_system_temp_folder(tmp_path, pick, step, uv_exe):
    clone, scratch, home, log = _layout(tmp_path)
    _pinned_copy("setup.bat", clone, tmp_path / "releases", _PS_STUB, bat=True)
    env = _windows_env(scratch, home, log, LOCALM_SETUP_ABORT_AFTER=step)
    r = _cmd(clone / "setup.bat", clone, env, f"{pick}\nY\n")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    assert r.returncode == 99, out
    assert f"stopping after '{step}'" in out, out
    installed = clone / uv_exe if uv_exe else home / ".local" / "bin" / "uv.exe"
    assert installed.is_file(), out
    _assert_nothing_left(scratch, clone, _installer_temps(log))
    leftovers = sorted(p.name for p in (clone / ".localm-setup-tmp").iterdir())
    assert [n for n in leftovers if n.startswith(("uv-tmp-", "uv-installer-"))] == [], leftovers


_GUI_UV_RUN = '"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\\gui.py"'


@_needs_windows
@pytest.mark.parametrize("pick,uv_exe", [("1", Path(".uv") / "uv.exe"), ("2", None)])
def test_setup_gui_bat_leaves_nothing_in_the_system_temp_folder(tmp_path, pick, uv_exe):
    clone, scratch, home, log = _layout(tmp_path)
    _pinned_copy("setup-gui.bat", clone, tmp_path / "releases", _PS_STUB, bat=True)
    script = clone / "setup-gui.bat"
    text = script.read_bytes().decode("utf-8")
    assert text.count(_GUI_UV_RUN) == 1, "the window launch line moved"
    script.write_bytes(text.replace(_GUI_UV_RUN, "echo WINDOW-UV=%UVEXE%& cmd /c exit 0")
                       .encode("utf-8"))
    r = _cmd(script, clone, _windows_env(scratch, home, log), f"{pick}\nY\n")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    installed = clone / uv_exe if uv_exe else home / ".local" / "bin" / "uv.exe"
    assert f"WINDOW-UV={installed}" in out, out
    _assert_nothing_left(scratch, clone, _installer_temps(log))
    assert not (clone / ".localm-setup-tmp").exists()


def _bash_env(tmp_path: Path, scratch: Path, home: Path, log: Path, releases: Path) -> dict:
    stubs = tmp_path / "stubbin"
    stubs.mkdir()
    curl = stubs / "curl"
    curl.write_bytes((
        '#!/bin/sh\nout=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift; fi\n'
        '  shift\ndone\n'
        f'cp "{releases.as_posix()}/uv-installer.sh" "$out"\n').encode())
    curl.chmod(0o755)
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):
        env.pop(var, None)
    env.update(TMPDIR=scratch.as_posix(), TEMP=str(scratch), TMP=str(scratch),
               HOME=str(home), STUB_LOG=log.as_posix(),
               PATH=str(stubs) + os.pathsep + env.get("PATH", ""))
    return env


def _bash_path(p: Path) -> str:
    """*p* as the bash in use names it (Git Bash on Windows takes D:/x as-is)."""
    return p.as_posix()


@_needs_bash
def test_setup_sh_leaves_nothing_in_the_system_temp_folder(tmp_path):
    clone, scratch, home, log = _layout(tmp_path)
    releases = tmp_path / "releases"
    _pinned_copy("setup.sh", clone, releases, _SH_STUB, bat=False)
    env = _bash_env(tmp_path, scratch, home, log, releases)
    env["LOCALM_SETUP_ABORT_AFTER"] = "uv-portable"
    r = subprocess.run([shutil.which("bash"), "setup.sh", "--yes"], cwd=str(clone), env=env,
                       capture_output=True, text=True, timeout=180, creationflags=_NO_WINDOW)
    out = r.stdout + r.stderr
    assert r.returncode == 99, out
    assert "stopping after 'uv-portable'" in out, out
    assert (clone / ".uv" / "uv").is_file(), out
    _assert_nothing_left(scratch, clone, [_native(t) for t in _installer_temps(log)])
    assert not (clone / ".localm-setup-tmp").exists()


@_needs_bash
def test_setup_gui_sh_leaves_nothing_in_the_system_temp_folder(tmp_path):
    clone, scratch, home, log = _layout(tmp_path)
    releases = tmp_path / "releases"
    _pinned_copy("setup-gui.sh", clone, releases, _SH_STUB, bat=False)
    env = _bash_env(tmp_path, scratch, home, log, releases)
    script = (f'set -euo pipefail; cd "{_bash_path(clone)}"; '
              f'eval "$(sed -n \'/^UV_INSTALLER_VERSION=/,/^run_uv_installer()/p\' setup-gui.sh '
              f'| sed \'$d\')"; '
              f'eval "$(sed -n \'/^run_uv_installer()/,/^}}/p\' setup-gui.sh)"; '
              'export UV_INSTALL_DIR="$PWD/.uv" UV_UNMANAGED_INSTALL="$PWD/.uv"; '
              'run_uv_installer; echo INSTALLED')
    r = subprocess.run([shutil.which("bash"), "-c", script], cwd=str(clone), env=env,
                       capture_output=True, text=True, timeout=180, creationflags=_NO_WINDOW)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "INSTALLED" in out, out
    assert (clone / ".uv" / "uv").is_file(), out
    _assert_nothing_left(scratch, clone, [_native(t) for t in _installer_temps(log)])
    assert not (clone / ".localm-setup-tmp").exists()


def _native(path: str) -> str:
    """A path bash printed, in the form this Python opens (Git Bash prints /d/x for D:\\x)."""
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == "/":
        return f"{path[1].upper()}:{path[2:]}"
    return path


def test_bootstrap_uv_leaves_nothing_in_the_system_temp_folder(tmp_path, monkeypatch):
    clone, scratch, _, log = _layout(tmp_path)
    installer = b"the pinned installer"
    sha = hashlib.sha256(installer).hexdigest()
    (clone / "setup.sh").write_text(
        f'UV_INSTALLER_VERSION="9.8.7"\nUV_INSTALLER_SHA256="{sha}"\n', encoding="utf-8")
    (clone / "setup.bat").write_text(
        f'set "UV_INSTALLER_VERSION=9.8.7"\nset "UV_INSTALLER_SHA256={sha}"\n', encoding="utf-8")
    for var in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(var, str(scratch))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    child = (
        "import os, tempfile, uuid\n"
        "d = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)\n"
        "os.mkdir(d)\n"
        "open(os.path.join(d, 'uv.zip'), 'wb').write(b'partial download')\n"
        "open(os.environ['STUB_LOG'], 'a').write(d + '\\n')\n"
        "dest = os.environ['UV_INSTALL_DIR']\n"
        "os.makedirs(dest, exist_ok=True)\n"
        f"open(os.path.join(dest, {uv_tool.UV_NAME!r}), 'wb').close()\n"
        "os.chmod(os.path.join(dest, " + repr(uv_tool.UV_NAME) + "), 0o755)\n")

    def run(script, env):
        return subprocess.run([sys.executable, "-c", child], env=dict(env, STUB_LOG=str(log)),
                              stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW).returncode

    exe = uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(installer),
                               run_installer=run)
    assert exe == clone / ".uv" / uv_tool.UV_NAME
    _assert_nothing_left(scratch, clone, _installer_temps(log))
    assert not (clone / ".localm-setup-tmp").exists()


@_needs_windows
def test_bootstrap_uv_avoids_a_bracketed_temp_folder_on_windows(tmp_path, monkeypatch):
    clone, scratch, _, _ = _layout(tmp_path, "clone[x]")
    installer = b"the pinned installer"
    sha = hashlib.sha256(installer).hexdigest()
    (clone / "setup.bat").write_text(
        f'set "UV_INSTALLER_VERSION=9.8.7"\nset "UV_INSTALLER_SHA256={sha}"\n', encoding="utf-8")
    for var in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(var, str(scratch))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    seen = []

    def run(script, env):
        seen.append(env["TEMP"])
        assert env["TEMP"] == env["TMP"] == env["TMPDIR"]
        assert "[" not in env["TEMP"] and Path(env["TEMP"]).is_dir()
        (Path(env["TEMP"]) / "uv.zip").write_bytes(b"partial download")
        Path(env["UV_INSTALL_DIR"]).mkdir(parents=True, exist_ok=True)
        (Path(env["UV_INSTALL_DIR"]) / uv_tool.UV_NAME).write_bytes(b"")
        return 0

    uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(installer), run_installer=run)
    assert len(seen) == 1 and not Path(seen[0]).exists()
    assert list(scratch.iterdir()) == []
    assert not (clone / ".localm-setup-tmp").exists()
