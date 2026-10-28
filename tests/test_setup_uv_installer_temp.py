# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing uv leaves nothing behind in the system temp folder.

Astral's installer downloads its archive into the temp folder of the process
that runs it (GetTempPath on Windows, ``mktemp -d`` on Linux and macOS) and does
not remove it. Every setup variant gives the installer a temp folder of its own
for that one run, under ``.localm-setup-tmp`` in the install folder, and removes
it when the installer has finished or failed. A run that is killed outright
leaves its partial download inside the install folder, never outside it.

Each real script runs from a throwaway clone with the system temp folder set to
an empty directory. The pinned installer is served from a local directory
through a file:// URL (Windows) or a stub ``curl`` (bash), and stands in for
Astral's: it writes a partial archive into its temp folder, as the real one
does, then either drops a uv where the real one would or fails. Each test
asserts the end state: the installer's temp folder is gone and the system temp
folder is empty. No network is used.
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


def _bash() -> str | None:
    """A bash that shares this process's file system and keeps the PATH it is
    given. On Windows that is Git for Windows' MSYS bash, ``<git>/usr/bin/bash.exe``:
    never the WSL launcher, and never the ``<git>/bin/bash.exe`` shim, which puts
    its own folders ahead of the PATH it is given. Found next to the bash on PATH, or
    next to git when the bash on PATH is another one. None when there is none."""
    found = shutil.which("bash")
    if os.name != "nt":
        return found
    system_roots = tuple(os.path.normcase(os.environ[v]).rstrip("\\") + "\\"
                         for v in ("SystemRoot", "windir") if os.environ.get(v))
    candidates = []
    for exe in (found, shutil.which("git")):
        if exe and not os.path.normcase(exe).startswith(system_roots):
            candidates += [Path(exe).parent.parent / "usr" / "bin" / "bash.exe",
                           Path(exe).with_name("bash.exe")]
    for bash in candidates:
        if bash.is_file() and (bash.parent / "msys-2.0.dll").is_file():
            return str(bash)
    return None


_BASH = _bash()
_needs_bash = pytest.mark.skipif(
    _BASH is None, reason="needs a bash on this file system (on Windows, Git for Windows' bash)")

_HOST = "https://github.com/astral-sh/uv/releases/download/"
_PS_STUB = (
    b"$tmp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid())\r\n"
    b"$null = New-Item -ItemType Directory -Path $tmp\r\n"
    b"[IO.File]::WriteAllText((Join-Path $tmp 'uv.zip'), 'partial download')\r\n"
    b"Add-Content -LiteralPath $env:STUB_LOG -Value $tmp\r\n"
    b"$null = Get-Item -Path (Join-Path $tmp 'uv.zip') -ErrorAction Stop\r\n"
    b"if ($env:STUB_FAIL -eq '1') { exit 3 }\r\n"
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
    b'[ "${STUB_FAIL:-0}" = 1 ] && exit 3\n'
    b'dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"\n'
    b'mkdir -p "$dest"\n'
    b'printf "#!/bin/sh\\n" > "$dest/uv"\n'
    b'chmod +x "$dest/uv"\n'
)
_OUTCOMES = [pytest.param(False, id="installed"), pytest.param(True, id="installer-failed")]


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
    return [_native(ln.strip())
            for ln in log.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]


def _native(path: str) -> str:
    """A path bash printed, in the form this Python opens (Git Bash prints /d/x for D:\\x)."""
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == "/":
        return f"{path[1].upper()}:{path[2:]}"
    return path


def _assert_nothing_left(scratch: Path, log: Path) -> None:
    """The end state: the installer's own temp folder is gone and the system
    temp folder holds nothing."""
    temps = _installer_temps(log)
    assert temps, "the stub installer recorded no temp folder"
    for t in temps:
        assert not Path(t).exists(), f"the installer's temp folder was left behind: {t}"
    assert sorted(p.name for p in scratch.iterdir()) == []


def _windows_env(scratch: Path, home: Path, log: Path, fails: bool, **extra) -> dict:
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "LOCALM_GUI_STORE", "LOCALM_GUI_UV",
                "LOCALM_GUI_UV_INSTALLED", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "STMP"):
        env.pop(var, None)
    env.update(PATH=os.pathsep.join([str(system32), str(system32 / "WindowsPowerShell" / "v1.0")]),
               TEMP=str(scratch), TMP=str(scratch), USERPROFILE=str(home),
               HOMEDRIVE=home.drive, HOMEPATH=str(home)[len(home.drive):], STUB_LOG=str(log),
               STUB_FAIL="1" if fails else "0")
    env.update(extra)
    return env


def _cmd(script: Path, clone: Path, env: dict, answers: str) -> tuple[int, str]:
    r = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", ".\\" + script.name], cwd=str(clone),
                       env=env, input=answers.replace("\n", "\r\n").encode("utf-8"),
                       capture_output=True, timeout=180, creationflags=_NO_WINDOW)
    return r.returncode, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


_AWKWARD_FOLDERS = [
    pytest.param("clone [x]", id="brackets"),
    pytest.param("clone]x", id="closing-bracket"),
    pytest.param("[a] b", id="leading-bracket"),
    pytest.param("it's here", id="apostrophe"),
    pytest.param("a&b", id="ampersand"),
    pytest.param("pct%y", id="percent"),
    pytest.param("bang!x", id="bang"),
    pytest.param("a^b (c)", id="caret-and-parens"),
]


@_needs_windows
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("pick,step,uv_exe", [
    ("1", "uv-portable", Path(".uv") / "uv.exe"),
    ("2", "uv-bootstrap", None),
])
def test_setup_bat_leaves_nothing_in_the_system_temp_folder(tmp_path, pick, step, uv_exe, fails):
    _setup_bat_installs_uv(tmp_path, "clone", pick, step, uv_exe, fails)


@_needs_windows
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("folder", _AWKWARD_FOLDERS)
def test_setup_bat_installs_uv_from_an_awkward_folder(tmp_path, folder, fails):
    _setup_bat_installs_uv(tmp_path, folder, "1", "uv-portable", Path(".uv") / "uv.exe", fails)


def _setup_bat_installs_uv(tmp_path, clone_name, pick, step, uv_exe, fails):
    clone, scratch, home, log = _layout(tmp_path, clone_name)
    _pinned_copy("setup.bat", clone, tmp_path / "releases", _PS_STUB, bat=True)
    env = _windows_env(scratch, home, log, fails, LOCALM_SETUP_ABORT_AFTER=step)
    rc, out = _cmd(clone / "setup.bat", clone, env, f"{pick}\nY\n\n\n")
    installed = clone / uv_exe if uv_exe else home / ".local" / "bin" / "uv.exe"
    if fails:
        assert rc == 1 and "uv still is not callable" in out, out
        assert not installed.exists(), out
    else:
        assert rc == 99 and f"stopping after '{step}'" in out, out
        assert installed.is_file(), out
    _assert_nothing_left(scratch, log)


_GUI_UV_RUN = '"%UVEXE%" run --no-project --isolated --python 3.12 python "installer\\gui.py"'


@_needs_windows
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("pick,uv_exe", [("1", Path(".uv") / "uv.exe"), ("2", None)])
def test_setup_gui_bat_leaves_nothing_in_the_system_temp_folder(tmp_path, pick, uv_exe, fails):
    _setup_gui_bat_installs_uv(tmp_path, "clone", pick, uv_exe, fails)


@_needs_windows
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("folder", _AWKWARD_FOLDERS)
def test_setup_gui_bat_installs_uv_from_an_awkward_folder(tmp_path, folder, fails):
    _setup_gui_bat_installs_uv(tmp_path, folder, "1", Path(".uv") / "uv.exe", fails)


def _setup_gui_bat_installs_uv(tmp_path, clone_name, pick, uv_exe, fails):
    clone, scratch, home, log = _layout(tmp_path, clone_name)
    _pinned_copy("setup-gui.bat", clone, tmp_path / "releases", _PS_STUB, bat=True)
    script = clone / "setup-gui.bat"
    text = script.read_bytes().decode("utf-8")
    assert text.count(_GUI_UV_RUN) == 1, "the window launch line moved"
    script.write_bytes(text.replace(_GUI_UV_RUN, "set UVEXE& cmd /c exit 0")
                       .encode("utf-8"))
    rc, out = _cmd(script, clone, _windows_env(scratch, home, log, fails), f"{pick}\nY\n\n\n")
    installed = clone / uv_exe if uv_exe else home / ".local" / "bin" / "uv.exe"
    if fails:
        assert rc == 1 and "uv still is not callable" in out, out
    else:
        assert f"UVEXE={installed}" in out, out
    _assert_nothing_left(scratch, log)
    assert not (clone / ".localm-setup-tmp").exists()


def _bash_env(tmp_path: Path, scratch: Path, home: Path, log: Path, releases: Path,
              fails: bool) -> dict:
    stubs = tmp_path / "stubbin"
    stubs.mkdir()
    curl = stubs / "curl"
    curl.write_bytes((
        '#!/bin/sh\nout=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then out="$2"; shift; fi\n'
        '  shift\ndone\n'
        'printf "%s\\n" "$out" >> "$CURL_LOG"\n'
        f'cp "{releases.as_posix()}/uv-installer.sh" "$out"\n').encode())
    curl.chmod(0o755)
    env = dict(os.environ)
    for var in ("LOCALM_SETUP_ABORT_AFTER", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):
        env.pop(var, None)
    env.update(TMPDIR=scratch.as_posix(), TEMP=str(scratch), TMP=str(scratch),
               HOME=str(home), STUB_LOG=log.as_posix(), STUB_FAIL="1" if fails else "0",
               CURL_LOG=_curl_log(tmp_path).as_posix(),
               PATH=os.pathsep.join([str(stubs), str(Path(_BASH).parent), env.get("PATH", "")]))
    return env


def _curl_log(tmp_path: Path) -> Path:
    return tmp_path / "curl.log"


def _assert_stub_curl_ran(tmp_path: Path, out: str) -> None:
    assert _curl_log(tmp_path).is_file(), f"the script ran a curl other than the stub\n{out}"


_SH_FOLDERS = [pytest.param("clone", id="plain"), pytest.param("clone [x]", id="brackets"),
               pytest.param("it's a&b", id="apostrophe-ampersand")]


@_needs_bash
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("folder", _SH_FOLDERS)
def test_setup_sh_leaves_nothing_in_the_system_temp_folder(tmp_path, folder, fails):
    clone, scratch, home, log = _layout(tmp_path, folder)
    releases = tmp_path / "releases"
    _pinned_copy("setup.sh", clone, releases, _SH_STUB, bat=False)
    env = _bash_env(tmp_path, scratch, home, log, releases, fails)
    env["LOCALM_SETUP_ABORT_AFTER"] = "uv-portable"
    r = subprocess.run([_BASH, "setup.sh", "--yes"], cwd=str(clone), env=env,
                       capture_output=True, text=True, timeout=180, creationflags=_NO_WINDOW)
    out = r.stdout + r.stderr
    _assert_stub_curl_ran(tmp_path, out)
    if fails:
        assert r.returncode in (1, 99), out
        assert not (clone / ".uv" / "uv").exists(), out
    else:
        assert r.returncode == 99 and "stopping after 'uv-portable'" in out, out
        assert (clone / ".uv" / "uv").is_file(), out
    _assert_nothing_left(scratch, log)
    assert not (clone / ".localm-setup-tmp").exists()


@_needs_bash
@pytest.mark.parametrize("fails", _OUTCOMES)
@pytest.mark.parametrize("folder", _SH_FOLDERS)
def test_setup_gui_sh_leaves_nothing_in_the_system_temp_folder(tmp_path, folder, fails):
    clone, scratch, home, log = _layout(tmp_path, folder)
    releases = tmp_path / "releases"
    _pinned_copy("setup-gui.sh", clone, releases, _SH_STUB, bat=False)
    env = _bash_env(tmp_path, scratch, home, log, releases, fails)
    script = (f'set -euo pipefail; cd "{clone.as_posix()}"; '
              f'eval "$(sed -n \'/^UV_INSTALLER_VERSION=/,/^run_uv_installer()/p\' setup-gui.sh '
              f'| sed \'$d\')"; '
              f'eval "$(sed -n \'/^run_uv_installer()/,/^}}/p\' setup-gui.sh)"; '
              'export UV_INSTALL_DIR="$PWD/.uv" UV_UNMANAGED_INSTALL="$PWD/.uv"; '
              'run_uv_installer; echo INSTALLED')
    r = subprocess.run([_BASH, "-c", script], cwd=str(clone), env=env,
                       capture_output=True, text=True, timeout=180, creationflags=_NO_WINDOW)
    out = r.stdout + r.stderr
    _assert_stub_curl_ran(tmp_path, out)
    if fails:
        assert r.returncode == 1 and "Could not download or run Astral's uv installer" in out, out
    else:
        assert r.returncode == 0 and "INSTALLED" in out, out
        assert (clone / ".uv" / "uv").is_file(), out
    _assert_nothing_left(scratch, log)
    assert not (clone / ".localm-setup-tmp").exists()


def _pinned_clone(clone: Path, installer: bytes) -> None:
    sha = hashlib.sha256(installer).hexdigest()
    (clone / "setup.sh").write_text(
        f'UV_INSTALLER_VERSION="9.8.7"\nUV_INSTALLER_SHA256="{sha}"\n', encoding="utf-8")
    (clone / "setup.bat").write_text(
        f'set "UV_INSTALLER_VERSION=9.8.7"\nset "UV_INSTALLER_SHA256={sha}"\n', encoding="utf-8")


def _system_temp(monkeypatch, scratch: Path) -> None:
    for var in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(var, str(scratch))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)


@pytest.mark.parametrize("fails", _OUTCOMES)
def test_bootstrap_uv_leaves_nothing_in_the_system_temp_folder(tmp_path, monkeypatch, fails):
    clone, scratch, _, log = _layout(tmp_path)
    installer = b"the pinned installer"
    _pinned_clone(clone, installer)
    _system_temp(monkeypatch, scratch)
    child = (
        "import os, sys, tempfile, uuid\n"
        "d = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)\n"
        "os.mkdir(d)\n"
        "open(os.path.join(d, 'uv.zip'), 'wb').write(b'partial download')\n"
        "open(os.environ['STUB_LOG'], 'a').write(d + '\\n')\n"
        "if os.environ['STUB_FAIL'] == '1':\n"
        "    sys.exit(3)\n"
        "dest = os.environ['UV_INSTALL_DIR']\n"
        "os.makedirs(dest, exist_ok=True)\n"
        f"open(os.path.join(dest, {uv_tool.UV_NAME!r}), 'wb').close()\n"
        f"os.chmod(os.path.join(dest, {uv_tool.UV_NAME!r}), 0o755)\n")

    def run(script, env):
        env = dict(env, STUB_LOG=str(log), STUB_FAIL="1" if fails else "0")
        return subprocess.run([sys.executable, "-c", child], env=env,
                              stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW).returncode

    def bootstrap():
        return uv_tool.bootstrap_uv(clone, download=lambda u, d: d.write_bytes(installer),
                                    run_installer=run)

    if fails:
        with pytest.raises(uv_tool.UvBootstrapError, match="exited 3"):
            bootstrap()
    else:
        assert bootstrap() == clone / ".uv" / uv_tool.UV_NAME
    _assert_nothing_left(scratch, log)
    assert not (clone / ".localm-setup-tmp").exists()


@_needs_windows
def test_bootstrap_uv_avoids_a_bracketed_temp_folder_on_windows(tmp_path, monkeypatch):
    clone, scratch, _, _ = _layout(tmp_path, "clone[x]")
    installer = b"the pinned installer"
    _pinned_clone(clone, installer)
    _system_temp(monkeypatch, scratch)
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
