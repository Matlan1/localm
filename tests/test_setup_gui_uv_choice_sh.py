# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.sh: which uv opens the window, with no question in the terminal.

The launcher asks nothing. It uses the folder's own ./.uv/uv, reuses the
per-user uv for an earlier Shared install (an install record and no ./.uv), and
otherwise installs uv into ./.uv. The window's first page asks Portable or
Shared; when Shared is chosen it exits 46 (the per-user uv was there) or 47 (it
installed it), and the launcher removes what this run put into the folder and
opens the window again on the per-user uv.

The real script runs from a throwaway clone with curl and sha256sum replaced by
stubs (the stub installer drops a fake uv into $UV_INSTALL_DIR) and a fake uv
on PATH standing in for a per-user one. Each fake uv stands in for the window:
it reports what it was given, creates ./.python and ./.cache like `uv run` does,
and exits with the next code in WINDOW_CODES_FILE. No network is touched.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_setup_gpu_detect import _bash, _mark_executable

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")

_REPORT = ('#!/bin/sh\necho "WINDOW-UV=$0"\necho "WINDOW-STORE=${LOCALM_GUI_STORE:-}"\n'
           'echo "WINDOW-PYDIR=${UV_PYTHON_INSTALL_DIR:-}"\necho "WINDOW-GUIUV=${LOCALM_GUI_UV:-}"\n'
           'echo "WINDOW-INSTALLED=${LOCALM_GUI_UV_INSTALLED:-}"\n'
           'echo "WINDOW-BOOTSTRAP=${LOCALM_GUI_BOOTSTRAP:-}"\n'
           'echo "WINDOW-REPOINTED=${LOCALM_GUI_REPOINTED:-}"\n'
           'if [ -n "${UV_PYTHON_INSTALL_DIR:-}" ]; then mkdir -p "$UV_PYTHON_INSTALL_DIR"; fi\n'
           'if [ -n "${UV_CACHE_DIR:-}" ]; then mkdir -p "$UV_CACHE_DIR"; fi\n'
           'code=0\n'
           'if [ -n "${WINDOW_CODES_FILE:-}" ] && [ -s "$WINDOW_CODES_FILE" ]; then\n'
           '  code="$(head -n 1 "$WINDOW_CODES_FILE")"\n'
           '  tail -n +2 "$WINDOW_CODES_FILE" > "$WINDOW_CODES_FILE.next"\n'
           '  mv "$WINDOW_CODES_FILE.next" "$WINDOW_CODES_FILE"\n'
           'fi\n'
           'if [ "$code" = 47 ]; then d="$PER_USER_BIN"; mkdir -p "$d"; cp "$0" "$d/uv";'
           ' chmod +x "$d/uv"; fi\n'
           'exit "$code"\n')


def _pinned_sha() -> str:
    text = (ROOT / "setup-gui.sh").read_text(encoding="utf-8")
    return re.search(r'^UV_INSTALLER_SHA256="([0-9a-f]{64})"', text, re.M).group(1)


def _write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    _mark_executable(path)


def _native(p: str) -> str:
    """A path bash printed, in the form Path() gives on this OS."""
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    if os.name == "nt" and m:
        return str(Path(f"{m.group(1).upper()}:/{m.group(2)}"))
    return str(Path(p))


def _posix(p: Path) -> str:
    """*p* as bash spells it (/d/x on Windows), so a PATH built from it splits on ':'."""
    if os.name == "nt" and p.drive:
        return "/" + p.drive[0].lower() + p.as_posix()[len(p.drive):]
    return p.as_posix()


def _tools_path() -> list[str]:
    if os.name == "nt":
        return [str(Path(_bash()).parent)]
    return ["/usr/bin", "/bin"]


def _run(tmp_path: Path, *, system_uv: bool = False, contained_uv: bool = False,
         installer_creates_uv: bool = True, codes: str = "0", record: bool = False,
         existing=(), marker: str = ""):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup-gui.sh", clone / "setup-gui.sh")
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    if contained_uv:
        _write_exec(clone / ".uv" / "uv", _REPORT)
    if record:
        (clone / ".localm-install.json").write_text("{}", encoding="utf-8")
    if marker:
        (clone / ".localm-gui-bootstrap").write_bytes((marker + "\n").encode())
    for name in existing:
        (clone / name).mkdir()
        (clone / name / "keep").write_bytes(b"x")
    stubs = tmp_path / "stubs"
    made = ('dest="${UV_INSTALL_DIR:-$PER_USER_BIN}"\n'
            'mkdir -p "$dest"\n'
            f"cat > \"$dest/uv\" <<'EOT'\n{_REPORT}EOT\n"
            'chmod +x "$dest/uv"\n') if installer_creates_uv else "exit 0\n"
    installer = ('#!/bin/sh\n'
                 'echo "INSTALLER-RAN INSTALL_DIR=[${UV_INSTALL_DIR:-}] '
                 'UNMANAGED=[${UV_UNMANAGED_INSTALL:-}]"\n' + made)
    _write_exec(stubs / "curl", (
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && dest=\"$2\"; shift; done\n"
        f"cat > \"$dest\" <<'STUB_END'\n{installer}STUB_END\n"))
    _write_exec(stubs / "sha256sum", f"#!/bin/sh\necho \"{_pinned_sha()}  $1\"\n")
    path_parts = [str(stubs), *_tools_path()]
    system = tmp_path / "systemuv"
    if system_uv:
        _write_exec(system / "uv", _REPORT)
        path_parts.insert(0, str(system))
    home = tmp_path / "home"
    home.mkdir()
    codes_file = tmp_path / "codes.txt"
    codes_file.write_bytes(("\n".join(codes.split()) + "\n").encode())
    env = dict(os.environ, PATH=os.pathsep.join(path_parts), DISPLAY=":0",
               HOME=_posix(home), WINDOW_CODES_FILE=_posix(codes_file),
               PER_USER_BIN=_posix(home / ".local" / "bin"))
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "LOCALM_GUI_BOOTSTRAP", "LOCALM_GUI_REPOINTED", "XDG_BIN_HOME",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_INSTALL_DIR",
                "UV_UNMANAGED_INSTALL"):
        env.pop(var, None)
    # Bytes, so a Windows text pipe never turns "\n" into "\r\n".
    raw = subprocess.run([_bash(), (clone / "setup-gui.sh").as_posix()], cwd=str(clone),
                         env=env, stdin=subprocess.DEVNULL, capture_output=True,
                         timeout=120)
    out = subprocess.CompletedProcess(raw.args, raw.returncode,
                                      raw.stdout.decode("utf-8", "replace"),
                                      raw.stderr.decode("utf-8", "replace"))
    return clone, system, home, out


def _fields(out, name: str) -> list:
    return [line.split("=", 1)[1].strip() for line in out.stdout.splitlines()
            if line.startswith(name + "=")]


def _field(out, name: str, n: int = 0) -> str:
    found = _fields(out, name)
    assert len(found) > n, f"{name} not reported:\n{out.stdout}\n{out.stderr}"
    return found[n]


def _path_field(out, name: str, n: int = 0) -> str:
    value = _field(out, name, n)
    return _native(value) if value else ""


def _installer_line(out) -> str:
    return next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))


def test_a_fresh_folder_installs_uv_into_the_folder_without_asking(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True)
    dirs = re.findall(r"\[([^\]]*)\]", _installer_line(out))
    assert [_native(d) for d in dirs] == [str(clone / ".uv")] * 2, out.stdout
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _path_field(out, "WINDOW-GUIUV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _path_field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _field(out, "WINDOW-BOOTSTRAP").split() == ["uv", "python", "cache"]
    assert "Pick 1 or 2" not in out.stdout and "[Y/n]" not in out.stdout
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_a_contained_uv_is_used_and_not_named_as_put_there_by_this_run(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                            existing=(".python",))
    assert "INSTALLER-RAN" not in out.stdout
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-BOOTSTRAP").split() == ["cache"]


def test_an_earlier_shared_install_reuses_the_per_user_uv(tmp_path):
    clone, system, _, out = _run(tmp_path, system_uv=True, record=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists()
    assert _path_field(out, "WINDOW-UV") == str(system / "uv")
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == ""


def test_shared_removes_what_this_run_put_in_the_folder_and_reopens_on_the_per_user_uv(tmp_path):
    clone, system, _, out = _run(tmp_path, system_uv=True, codes="46 0")
    assert _path_field(out, "WINDOW-UV", 0) == str(clone / ".uv" / "uv")
    assert _path_field(out, "WINDOW-UV", 1) == str(system / "uv")
    assert _field(out, "WINDOW-STORE", 1) == "shared"
    assert _field(out, "WINDOW-REPOINTED", 1) == "1"
    assert _field(out, "WINDOW-PYDIR", 1) == ""
    assert _field(out, "WINDOW-BOOTSTRAP", 1) == ""
    for name in (".uv", ".python", ".cache"):
        assert not (clone / name).exists(), f"{name} was left behind\n{out.stdout}"
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_shared_keeps_folders_this_run_did_not_create(tmp_path):
    clone, _, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                            existing=(".python", ".cache"), codes="46 0")
    for name in (".uv", ".python", ".cache"):
        assert (clone / name).is_dir(), f"{name} was removed but this run did not create it"
    assert _field(out, "WINDOW-STORE", 1) == "shared"


def test_shared_with_a_uv_the_window_installed_says_so_to_the_window(tmp_path):
    clone, _, home, out = _run(tmp_path, codes="47 0")
    assert _path_field(out, "WINDOW-UV", 1) == str(home / ".local" / "bin" / "uv")
    assert _field(out, "WINDOW-INSTALLED", 1) == "1"
    assert not (clone / ".uv").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_shared_with_no_per_user_uv_after_all_says_uv_is_not_callable(tmp_path):
    clone, _, _, out = _run(tmp_path, codes="46 0")
    assert len(_fields(out, "WINDOW-UV")) == 1
    assert "uv still is not callable" in out.stdout and "per-user uv" in out.stdout
    assert (clone / ".uv" / "uv").is_file(), "the folder's uv was removed with nothing to use"
    assert out.returncode == 1


def test_what_this_run_put_in_the_folder_is_recorded_for_a_later_run(tmp_path):
    clone, _, _, out = _run(tmp_path)
    record = (clone / ".localm-gui-bootstrap").read_text(encoding="ascii").split()
    assert record == ["uv", "python", "cache"], out.stdout


def test_shared_also_removes_what_an_earlier_run_put_in_the_folder(tmp_path):
    clone, system, _, out = _run(tmp_path, system_uv=True, contained_uv=True,
                                 existing=(".python", ".cache"), marker="uv python cache",
                                 codes="46 0")
    assert _field(out, "WINDOW-BOOTSTRAP", 0).split() == ["uv", "python", "cache"]
    for name in (".uv", ".python", ".cache", ".localm-gui-bootstrap"):
        assert not (clone / name).exists(), f"{name} was left behind\n{out.stdout}"
    assert _path_field(out, "WINDOW-UV", 1) == str(system / "uv")
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_the_window_asking_for_shared_twice_is_a_failure_not_a_loop(tmp_path):
    _clone, _, _, out = _run(tmp_path, system_uv=True, codes="46 46 0")
    assert len(_fields(out, "WINDOW-UV")) == 2
    assert "The setup window could not run (exit 46)" in out.stdout
    assert out.returncode == 46


def test_portable_never_falls_back_to_the_system_uv_when_the_install_failed(tmp_path):
    _clone, _, _, out = _run(tmp_path, system_uv=True, installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the system uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_the_launcher_asks_no_question():
    lines = (ROOT / "setup-gui.sh").read_text(encoding="utf-8").splitlines()
    code = [ln for ln in lines if not ln.strip().startswith("#")]
    assert not [ln for ln in code if re.search(r"\bread\b[^;|]*\s-p\b", ln)]
    assert not [ln for ln in lines if "SETUP-Q:" in ln]


def test_no_display_is_refused_before_any_uv_install(tmp_path):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup-gui.sh", clone / "setup-gui.sh")
    stubs = tmp_path / "stubs"
    _write_exec(stubs / "curl", "#!/bin/sh\necho CURL-RAN\nexit 1\n")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PATH=os.pathsep.join([str(stubs), *_tools_path()]),
               HOME=_posix(home))
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL"):
        env.pop(var, None)
    raw = subprocess.run([_bash(), (clone / "setup-gui.sh").as_posix()], cwd=str(clone),
                         env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    out = raw.stdout.decode("utf-8", "replace")
    if "Darwin" in subprocess.run([_bash(), "-c", "uname -s"], capture_output=True,
                                  text=True).stdout:
        pytest.skip("macOS needs no DISPLAY")
    assert raw.returncode == 1, out
    assert "No graphical display was found" in out
    assert "CURL-RAN" not in out
    assert not (clone / ".uv").exists()
