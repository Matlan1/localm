# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.bat: the Portable / Shared question and which uv opens the window.

The launcher asks setup.bat's question once, always, before the window: also
when .\\.uv already exists and when no uv exists anywhere. Portable uses only
.\\.uv\\uv.exe and installs uv there when it is missing; Shared uses the uv on
PATH (never .\\.uv) and installs uv at its per-user location when there is
none. The answer reaches the window as LOCALM_GUI_STORE, the uv as
LOCALM_GUI_UV, and a per-user install as LOCALM_GUI_UV_INSTALLED=1.

The real script runs from a throwaway clone with the uv installer and the
window each replaced by a probe line, a stub installer that drops a uv where
the real one would, and a stub uv on PATH standing in for a system one. No
network is touched. Answers are fed from a file, so each line reaches its own
`set /p`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup-gui.bat runs only on Windows")

_UV_RUN = '"%UVEXE%" run --no-project --python 3.12 python "installer\\gui.py"'
_PS_PREFIX = "powershell -NoProfile -ExecutionPolicy Bypass -Command"
_WINDOW = ("echo WINDOW-UV=%UVEXE%& echo WINDOW-STORE=%LOCALM_GUI_STORE%& "
           "echo WINDOW-PYDIR=%UV_PYTHON_INSTALL_DIR%& echo WINDOW-GUIUV=%LOCALM_GUI_UV%& "
           "echo WINDOW-INSTALLED=%LOCALM_GUI_UV_INSTALLED%& cmd /c exit 0")
_STUB_INSTALLER = (
    "@echo off\r\n"
    'if not "%STUB_CREATES%"=="1" exit /b 0\r\n'
    'set "DEST=%USERPROFILE%\\.local\\bin"\r\n'
    'if defined UV_INSTALL_DIR set "DEST=%UV_INSTALL_DIR%"\r\n'
    'mkdir "%DEST%" 2>nul\r\n'
    'type nul > "%DEST%\\uv.exe"\r\n'
    "exit /b 0\r\n")


def _probe_text() -> str:
    text = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    assert text.count(_UV_RUN) == 1, "the window launch line moved"
    lines = text.splitlines()
    ps = [i for i, line in enumerate(lines) if line.startswith(_PS_PREFIX)]
    assert len(ps) == 1, "the uv installer line moved"
    lines[ps[0]] = ('echo INSTALLER-RAN INSTALL_DIR=[%UV_INSTALL_DIR%] '
                    'UNMANAGED=[%UV_UNMANAGED_INSTALL%]& call "%STUB_INSTALLER%"')
    text = "\n".join(lines).replace(_UV_RUN, _WINDOW)
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _run(tmp_path: Path, *, answers: str, system_uv: bool, contained_uv: bool = False,
         installer_creates_uv: bool = True, contained_on_path: bool = False):
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "probe.bat").write_text(_probe_text(), encoding="utf-8", newline="")
    if contained_uv:
        (clone / ".uv").mkdir()
        (clone / ".uv" / "uv.exe").write_bytes(b"")
    home = tmp_path / "home"
    home.mkdir()
    installer = tmp_path / "stub-installer.bat"
    installer.write_text(_STUB_INSTALLER, encoding="ascii", newline="")
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    path = str(system32)
    stub = tmp_path / "systemuv"
    if system_uv:
        stub.mkdir()
        (stub / "uv.cmd").write_text("@exit /b 1\r\n", encoding="ascii")
        path = str(stub) + os.pathsep + path
    if contained_on_path:
        path = str(clone / ".uv") + os.pathsep + path
    env = dict(os.environ, PATH=path, USERPROFILE=str(home), HOMEDRIVE=home.drive,
               HOMEPATH=str(home)[len(home.drive):], STUB_INSTALLER=str(installer),
               STUB_CREATES="1" if installer_creates_uv else "0")
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_INSTALL_DIR",
                "UV_UNMANAGED_INSTALL"):
        env.pop(var, None)
    answer_file = tmp_path / "answers.txt"
    answer_file.write_text(answers.replace("\n", "\r\n"), encoding="ascii", newline="")
    with answer_file.open("rb") as fh:
        out = subprocess.run(["cmd", "/c", str(clone / "probe.bat")], cwd=str(clone), env=env,
                             stdin=fh, capture_output=True, text=True, timeout=120)
    return clone, stub, home, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def _installer_line(out) -> str:
    return next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))


ASK_STORE = "Pick 1 or 2 [1]:"
ASK_UV = "Install it now with Astral's official installer? [Y/n]:"


def test_portable_with_a_contained_uv_still_asks_the_question_and_uses_it(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\n", system_uv=True, contained_uv=True)
    assert ASK_STORE in out.stdout, out.stdout
    assert ASK_UV not in out.stdout and "INSTALLER-RAN" not in out.stdout
    contained = str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-UV") == contained
    assert _field(out, "WINDOW-GUIUV") == contained
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _field(out, "WINDOW-INSTALLED") == ""
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_portable_ignores_a_system_uv_and_installs_the_contained_one(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\n\n", system_uv=True)
    assert ASK_STORE in out.stdout and ASK_UV in out.stdout, out.stdout
    assert "[!] uv (the Python package manager localm builds on) is not installed." in out.stdout
    dot_uv = str(clone / ".uv")
    assert _installer_line(out) == (
        f"INSTALLER-RAN INSTALL_DIR=[{dot_uv}] UNMANAGED=[{dot_uv}]")
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_the_default_answer_is_portable(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="\n\n", system_uv=True)
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert "INSTALLER-RAN" in out.stdout


def test_no_uv_anywhere_still_asks_the_question_first(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\nY\n", system_uv=False)
    assert out.stdout.index(ASK_STORE) < out.stdout.index(ASK_UV), out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")


def test_shared_uses_the_system_uv_and_says_so_to_the_window(tmp_path):
    clone, stub, _, out = _run(tmp_path, answers="2\n", system_uv=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists(), out.stdout
    assert ASK_UV not in out.stdout
    assert _field(out, "WINDOW-UV") == str(stub / "uv.cmd")
    assert _field(out, "WINDOW-GUIUV") == str(stub / "uv.cmd")
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == "", "Shared must not point uv's Python folder into the clone"
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_shared_ignores_a_contained_uv(tmp_path):
    _clone, stub, _, out = _run(tmp_path, answers="2\n", system_uv=True, contained_uv=True)
    assert _field(out, "WINDOW-UV") == str(stub / "uv.cmd")
    assert _field(out, "WINDOW-STORE") == "shared"


def test_shared_never_takes_the_contained_uv_even_from_path(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="2\nY\n", system_uv=False, contained_uv=True,
                               contained_on_path=True)
    assert ASK_UV in out.stdout and "INSTALLER-RAN" in out.stdout, out.stdout
    assert _field(out, "WINDOW-UV") == str(home / ".local" / "bin" / "uv.exe")
    assert _field(out, "WINDOW-UV") != str(clone / ".uv" / "uv.exe")


def test_shared_with_only_a_contained_uv_installs_the_per_user_one(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="2\nY\n", system_uv=False, contained_uv=True)
    assert ASK_UV in out.stdout, out.stdout
    assert _installer_line(out) == "INSTALLER-RAN INSTALL_DIR=[] UNMANAGED=[]"
    assert _field(out, "WINDOW-UV") == str(home / ".local" / "bin" / "uv.exe")
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert _field(out, "WINDOW-UV") != str(clone / ".uv" / "uv.exe")


def test_no_uv_and_shared_asks_and_installs_per_user_without_an_install_dir(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="2\n\n", system_uv=False)
    assert ASK_STORE in out.stdout and ASK_UV in out.stdout, out.stdout
    assert _installer_line(out) == "INSTALLER-RAN INSTALL_DIR=[] UNMANAGED=[]"
    per_user = str(home / ".local" / "bin" / "uv.exe")
    assert _field(out, "WINDOW-UV") == per_user
    assert _field(out, "WINDOW-GUIUV") == per_user
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == ""
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert not (clone / ".uv").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)


@pytest.mark.parametrize("store", ["1", "2"])
def test_declining_the_uv_install_installs_nothing(tmp_path, store):
    clone, _, _, out = _run(tmp_path, answers=f"{store}\nn\n", system_uv=False)
    assert "INSTALLER-RAN" not in out.stdout and "WINDOW-UV" not in out.stdout
    assert "Nothing was installed." in out.stdout
    assert not (clone / ".uv").exists()
    assert out.returncode == 1


def test_portable_never_falls_back_to_the_system_uv_when_the_install_failed(tmp_path):
    _clone, _, _, out = _run(tmp_path, answers="1\nY\n", system_uv=True,
                             installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the system uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_shared_says_so_when_the_installed_uv_is_not_callable(tmp_path):
    _clone, _, _, out = _run(tmp_path, answers="2\nY\n", system_uv=False,
                             installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_a_folder_with_a_bang_in_its_path_still_asks_and_installs(tmp_path):
    root = tmp_path / "bang!dir"
    root.mkdir()
    clone, _, _, out = _run(root, answers="1\nY\n", system_uv=True)
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")


def test_the_questions_are_worded_like_setup_bat():
    gui = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    console = (ROOT / "setup.bat").read_text(encoding="utf-8")
    for fragment in ("Keep localm's Python tooling", "[1] Portable - everything in this folder",
                     "[2] Shared   - reuse/install uv at its normal per-user location",
                     'set /p "STOREPICK=  Pick 1 or 2 [1]: "',
                     "uv (the Python package manager localm builds on) is not installed.",
                     'set /p "GETUV=  Install it now with Astral\'s official installer? [Y/n]: "'):
        assert fragment in gui and fragment in console, fragment


def test_each_launcher_question_is_tagged_once():
    gui = (ROOT / "setup-gui.bat").read_text(encoding="utf-8").splitlines()
    tags = [ln.strip() for ln in gui if ln.strip().startswith("rem SETUP-Q:")]
    assert tags == ["rem SETUP-Q: tooling-store", "rem SETUP-Q: uv-install"]
    for i, ln in enumerate(gui):
        if ln.strip().lower().startswith("set /p"):
            assert gui[i - 1].strip().startswith("rem SETUP-Q:"), ln
