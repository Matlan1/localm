# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.bat: the Portable / Shared question and which uv opens the window.

The launcher asks setup.bat's question once, always, before the window: also
when .\\.uv already exists and when no uv exists anywhere. Both answers use only
localm's own pinned uv in its store: Portable .\\.uv\\uv.exe, Shared
%LOCALAPPDATA%\\localm\\shared\\uv\\uv.exe. A uv there is used when it reports
the pinned version; otherwise the pinned installer puts it there, without
asking. A uv on PATH is never used. The answer reaches the window as
LOCALM_GUI_STORE, the uv as LOCALM_GUI_UV, and an install into the Shared
folder as LOCALM_GUI_UV_INSTALLED=1.

The real script runs from a throwaway clone with the uv installer and the
window each replaced by a probe line, a stub installer that copies a stand-in
uv (tests/_fake_uv.py) where the real one would, and a decoy uv on PATH. No
network is touched. Answers are fed from a file, so each line reaches its own
`set /p`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests import _fake_uv as fake_uv

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup-gui.bat runs only on Windows")

_UV_RUN = ('"%UVEXE%" run --no-project --isolated --python 3.12 --python-preference only-managed '
           '"installer\\gui.py"')
_PS_PREFIX = '"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -Command'
_WINDOW = ("echo WINDOW-UV=%UVEXE%& echo WINDOW-STORE=%LOCALM_GUI_STORE%& "
           "echo WINDOW-PYDIR=%UV_PYTHON_INSTALL_DIR%& echo WINDOW-CACHE=%UV_CACHE_DIR%& "
           "echo WINDOW-GUIUV=%LOCALM_GUI_UV%& echo WINDOW-INSTALLED=%LOCALM_GUI_UV_INSTALLED%& "
           "echo WINDOW-TEMP=%TEMP%& echo WINDOW-PREF=%UV_PYTHON_PREFERENCE%& cmd /c exit 0")
_STUB_INSTALLER = (
    "@echo off\r\n"
    'if not "%STUB_CREATES%"=="1" exit /b 0\r\n'
    'mkdir "%UV_INSTALL_DIR%" 2>nul\r\n'
    'copy /y "%STUB_UV_EXE%" "%UV_INSTALL_DIR%\\uv.exe" >nul\r\n'
    "exit /b 0\r\n")


@pytest.fixture(scope="module")
def stub_uv(tmp_path_factory):
    return fake_uv.build(tmp_path_factory.mktemp("fakeuv"))


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


def _run(tmp_path: Path, stub_uv: Path, *, answers: str, contained_uv: str | None = None,
         shared_uv: str | None = None, installer_creates_uv: bool = True,
         contained_on_path: bool = False):
    """*contained_uv* / *shared_uv*: put a stand-in uv in that store reporting
    this version ("" = the pinned one)."""
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "probe.bat").write_text(_probe_text(), encoding="utf-8", newline="")
    lad = tmp_path / "localappdata"
    lad.mkdir()
    shared = lad / "localm" / "shared"
    versions = {}
    if contained_uv is not None:
        fake_uv.place(stub_uv, clone / ".uv")
        versions["contained"] = contained_uv
    if shared_uv is not None:
        fake_uv.place(stub_uv, shared / "uv")
        versions["shared"] = shared_uv
    home = tmp_path / "home"
    home.mkdir()
    installer = tmp_path / "stub-installer.bat"
    installer.write_text(_STUB_INSTALLER, encoding="ascii", newline="")
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    decoy = tmp_path / "systemuv"
    decoy.mkdir()
    fake_uv.place(stub_uv, decoy)
    path = str(decoy) + os.pathsep + str(system32)
    if contained_on_path:
        path = str(clone / ".uv") + os.pathsep + path
    env = dict(os.environ, PATH=path, USERPROFILE=str(home), HOMEDRIVE=home.drive,
               HOMEPATH=str(home)[len(home.drive):], LOCALAPPDATA=str(lad),
               STUB_INSTALLER=str(installer), STUB_UV_EXE=str(stub_uv),
               STUB_CREATES="1" if installer_creates_uv else "0",
               UV_PYTHON_INSTALL_DIR=str(tmp_path / "inherited-python"),
               UV_CACHE_DIR=str(tmp_path / "inherited-cache"))
    wrong = {v for v in versions.values() if v}
    assert len(wrong) <= 1, "one reported version per run"
    if wrong:
        env["STUB_UV_VERSION"] = wrong.pop()
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL"):
        env.pop(var, None)
    answer_file = tmp_path / "answers.txt"
    answer_file.write_text(answers.replace("\n", "\r\n"), encoding="ascii", newline="")
    with answer_file.open("rb") as fh:
        out = subprocess.run([str(system32 / "cmd.exe"), "/c", str(clone / "probe.bat")],
                             cwd=str(clone), env=env, stdin=fh, capture_output=True, text=True,
                             timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return clone, shared, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def _installer_line(out) -> str:
    return next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))


ASK_STORE = "Pick 1 or 2 [1]:"


def test_portable_with_a_pinned_contained_uv_still_asks_the_question_and_uses_it(tmp_path, stub_uv):
    clone, _, out = _run(tmp_path, stub_uv, answers="1\n", contained_uv="")
    assert ASK_STORE in out.stdout, out.stdout
    assert "INSTALLER-RAN" not in out.stdout
    contained = str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-UV") == contained
    assert _field(out, "WINDOW-GUIUV") == contained
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _field(out, "WINDOW-CACHE") == str(clone / ".cache")
    assert _field(out, "WINDOW-INSTALLED") == ""
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_portable_ignores_a_path_uv_and_installs_the_contained_one(tmp_path, stub_uv):
    clone, _, out = _run(tmp_path, stub_uv, answers="1\n")
    dot_uv = str(clone / ".uv")
    assert _installer_line(out) == f"INSTALLER-RAN INSTALL_DIR=[{dot_uv}] UNMANAGED=[{dot_uv}]"
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_a_contained_uv_of_another_version_is_replaced(tmp_path, stub_uv):
    clone, _, out = _run(tmp_path, stub_uv, answers="1\n", contained_uv="0.0.1",
                         installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout, out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on a uv of another version"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_the_default_answer_is_portable(tmp_path, stub_uv):
    clone, _, out = _run(tmp_path, stub_uv, answers="\n")
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert "INSTALLER-RAN" in out.stdout


def test_the_tooling_question_is_the_only_question(tmp_path, stub_uv):
    clone, _, out = _run(tmp_path, stub_uv, answers="1\n")
    assert out.stdout.count("[1]:") == 1 and "[Y/n]" not in out.stdout, out.stdout


def test_shared_uses_the_shared_folder_uv_and_says_so_to_the_window(tmp_path, stub_uv):
    clone, shared, out = _run(tmp_path, stub_uv, answers="2\n", shared_uv="")
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists(), out.stdout
    uv = str(shared / "uv" / "uv.exe")
    assert _field(out, "WINDOW-UV") == uv
    assert _field(out, "WINDOW-GUIUV") == uv
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == str(shared / "python")
    assert _field(out, "WINDOW-CACHE") == str(shared / "cache")
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_shared_ignores_a_contained_uv_and_a_path_uv(tmp_path, stub_uv):
    clone, shared, out = _run(tmp_path, stub_uv, answers="2\n", contained_uv="",
                              contained_on_path=True)
    assert _installer_line(out) == (
        f"INSTALLER-RAN INSTALL_DIR=[{shared / 'uv'}] UNMANAGED=[{shared / 'uv'}]")
    assert _field(out, "WINDOW-UV") == str(shared / "uv" / "uv.exe")
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert _field(out, "WINDOW-STORE") == "shared"


def test_no_uv_and_shared_installs_into_the_shared_folder(tmp_path, stub_uv):
    clone, shared, out = _run(tmp_path, stub_uv, answers="2\n")
    assert _installer_line(out) == (
        f"INSTALLER-RAN INSTALL_DIR=[{shared / 'uv'}] UNMANAGED=[{shared / 'uv'}]")
    uv = str(shared / "uv" / "uv.exe")
    assert _field(out, "WINDOW-UV") == uv
    assert _field(out, "WINDOW-GUIUV") == uv
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert not (clone / ".uv").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)


@pytest.mark.parametrize("store", ["1", "2"])
def test_the_window_runs_on_managed_python_with_setup_temp(tmp_path, stub_uv, store):
    clone, _, out = _run(tmp_path, stub_uv, answers=f"{store}\n")
    assert _field(out, "WINDOW-PREF") == "only-managed"
    assert _field(out, "WINDOW-TEMP") == str(clone / ".localm-setup-tmp")
    assert "inherited-" not in out.stdout
    assert not (clone / ".localm-setup-tmp").exists(), "setup temp left after a clean exit"


def test_portable_never_falls_back_to_a_path_uv_when_the_install_failed(tmp_path, stub_uv):
    _clone, _, out = _run(tmp_path, stub_uv, answers="1\n", installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the PATH uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_shared_says_so_when_the_installed_uv_is_not_callable(tmp_path, stub_uv):
    _clone, _, out = _run(tmp_path, stub_uv, answers="2\n", installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_a_folder_with_a_bang_in_its_path_still_asks_and_installs(tmp_path, stub_uv):
    root = tmp_path / "bang!dir"
    root.mkdir()
    clone, _, out = _run(root, stub_uv, answers="1\n")
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")


def test_the_questions_are_worded_like_setup_bat():
    gui = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    console = (ROOT / "setup.bat").read_text(encoding="utf-8")
    for fragment in ("Keep localm's Python tooling", "[1] Portable - everything in this folder",
                     "[2] Shared   - one localm folder in your user profile",
                     'set /p "STOREPICK=  Pick 1 or 2 [1]: "'):
        assert fragment in gui and fragment in console, fragment
    for gone in ("Install it now with Astral's official installer", "winget install"):
        assert gone not in gui and gone not in console, gone


def test_each_launcher_question_is_tagged_once():
    gui = (ROOT / "setup-gui.bat").read_text(encoding="utf-8").splitlines()
    tags = [ln.strip() for ln in gui if ln.strip().startswith("rem SETUP-Q:")]
    assert tags == ["rem SETUP-Q: tooling-store"]
    for i, ln in enumerate(gui):
        if ln.strip().lower().startswith("set /p"):
            assert gui[i - 1].strip().startswith("rem SETUP-Q:"), ln
