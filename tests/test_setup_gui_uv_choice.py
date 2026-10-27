# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.bat: which uv opens the setup window.

A uv found elsewhere on the computer is used only when the user chooses
"Shared" (the question setup.bat asks); otherwise uv is installed into
.\\.uv. The real script runs from a throwaway clone with the uv installer and
the window each replaced by a probe line, and a stub uv on PATH standing in
for the system one. No network is touched.
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


def _probe_text(installer_creates_uv: bool) -> str:
    text = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    assert text.count(_UV_RUN) == 1, "the window launch line moved"
    lines = text.splitlines()
    ps = [i for i, line in enumerate(lines) if line.startswith(_PS_PREFIX)]
    assert len(ps) == 1, "the uv installer line moved"
    lines[ps[0]] = ("echo INSTALLER-RAN& mkdir .uv& type nul > .uv\\uv.exe& cmd /c exit 0"
                    if installer_creates_uv else "echo INSTALLER-RAN& cmd /c exit 0")
    text = "\n".join(lines).replace(
        _UV_RUN, 'echo WINDOW-UV=%UVEXE%& echo WINDOW-STORE=%LOCALM_GUI_STORE%& cmd /c exit 0')
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _run(tmp_path: Path, *, answer: str, system_uv: bool, contained_uv: bool = False,
         installer_creates_uv: bool = True):
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "probe.bat").write_text(_probe_text(installer_creates_uv), encoding="utf-8")
    if contained_uv:
        (clone / ".uv").mkdir()
        (clone / ".uv" / "uv.exe").write_bytes(b"")
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    path = str(system32)
    stub = tmp_path / "systemuv"
    if system_uv:
        stub.mkdir()
        (stub / "uv.cmd").write_text("@exit /b 1\r\n", encoding="ascii")
        path = str(stub) + os.pathsep + path
    env = dict(os.environ, PATH=path)
    env.pop("LOCALM_GUI_STORE", None)
    out = subprocess.run(["cmd", "/c", str(clone / "probe.bat")], cwd=str(clone), env=env,
                         input=answer, capture_output=True, text=True, timeout=120)
    return clone, stub, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def test_a_system_uv_is_not_taken_silently_and_portable_installs_the_contained_one(tmp_path):
    clone, _, out = _run(tmp_path, answer="1\n", system_uv=True)
    assert "Pick 1 or 2" in out.stdout and "INSTALLER-RAN" in out.stdout, out.stdout
    assert "Install it now?" not in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert _field(out, "WINDOW-STORE") == ""
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_the_default_answer_is_portable(tmp_path):
    clone, _, out = _run(tmp_path, answer="\n", system_uv=True)
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")
    assert "INSTALLER-RAN" in out.stdout


def test_choosing_shared_uses_the_system_uv_and_says_so_to_the_window(tmp_path):
    clone, stub, out = _run(tmp_path, answer="2\n", system_uv=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists(), out.stdout
    assert _field(out, "WINDOW-UV") == str(stub / "uv.cmd")
    assert _field(out, "WINDOW-STORE") == "shared"


def test_a_contained_uv_is_used_without_asking(tmp_path):
    clone, _, out = _run(tmp_path, answer="", system_uv=True, contained_uv=True)
    assert "Pick 1 or 2" not in out.stdout and "INSTALLER-RAN" not in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")


def test_no_uv_anywhere_still_offers_the_install(tmp_path):
    clone, _, out = _run(tmp_path, answer="Y\n", system_uv=False)
    assert "Pick 1 or 2" not in out.stdout and "Install it now?" in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")


def test_portable_never_falls_back_to_the_system_uv_when_the_install_failed(tmp_path):
    clone, _, out = _run(tmp_path, answer="1\n", system_uv=True, installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the system uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_a_folder_with_a_bang_in_its_path_still_asks_and_installs(tmp_path):
    root = tmp_path / "bang!dir"
    root.mkdir()
    clone, _, out = _run(root, answer="1\n", system_uv=True)
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv.exe")


def test_the_question_is_worded_like_setup_bat():
    gui = (ROOT / "setup-gui.bat").read_text(encoding="utf-8")
    console = (ROOT / "setup.bat").read_text(encoding="utf-8")
    for fragment in ("[1] Portable - everything in this folder", "Pick 1 or 2 [1]:"):
        assert fragment in gui and fragment in console
