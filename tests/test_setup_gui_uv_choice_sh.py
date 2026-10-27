# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.sh: which uv opens the setup window (Linux/macOS).

A uv found elsewhere on the computer is used only when the user chooses
"Shared" (the question setup.sh asks); otherwise uv is installed into ./.uv.
The real script runs from a throwaway clone with curl and sha256sum replaced
by stubs (the stub installer drops a fake uv into ./.uv) and a fake uv on PATH
standing in for the system one. No network is touched.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                                reason="setup-gui.sh runs under bash on Linux/macOS")

_REPORT = '#!/bin/sh\necho "WINDOW-UV=$0"\necho "WINDOW-STORE=${LOCALM_GUI_STORE:-}"\necho "WINDOW-PYDIR=${UV_PYTHON_INSTALL_DIR:-}"\nexit 0\n'


def _pinned_sha() -> str:
    text = (ROOT / "setup-gui.sh").read_text(encoding="utf-8")
    return re.search(r'^UV_INSTALLER_SHA256="([0-9a-f]{64})"', text, re.M).group(1)


def _write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run(tmp_path: Path, *, answer: str, system_uv: bool, contained_uv: bool = False,
         installer_creates_uv: bool = True):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup-gui.sh", clone / "setup-gui.sh")
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    if contained_uv:
        _write_exec(clone / ".uv" / "uv", _REPORT)
    stubs = tmp_path / "stubs"
    made = ("mkdir -p \"$UV_INSTALL_DIR\"\n"
            f"cat > \"$UV_INSTALL_DIR/uv\" <<'EOT'\n{_REPORT}EOT\n"
            "chmod +x \"$UV_INSTALL_DIR/uv\"\n") if installer_creates_uv else "exit 0\n"
    installer = f"#!/bin/sh\necho INSTALLER-RAN\n{made}"
    _write_exec(stubs / "curl", (
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && dest=\"$2\"; shift; done\n"
        f"cat > \"$dest\" <<'STUB_END'\n{installer}STUB_END\n"))
    _write_exec(stubs / "sha256sum", f"#!/bin/sh\necho \"{_pinned_sha()}  $1\"\n")
    path = str(stubs) + os.pathsep + "/usr/bin" + os.pathsep + "/bin"
    system = tmp_path / "systemuv"
    if system_uv:
        _write_exec(system / "uv", _REPORT)
        path = str(system) + os.pathsep + path
    env = dict(os.environ, PATH=path, DISPLAY=":0", HOME=str(tmp_path / "home"))
    for var in ("LOCALM_GUI_STORE", "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):
        env.pop(var, None)
    (tmp_path / "home").mkdir()
    out = subprocess.run([shutil.which("bash"), str(clone / "setup-gui.sh")], cwd=str(clone),
                         env=env, input=answer, capture_output=True, text=True, timeout=120)
    return clone, system, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def test_a_system_uv_is_not_taken_silently_and_portable_installs_the_contained_one(tmp_path):
    clone, _, out = _run(tmp_path, answer="1\n", system_uv=True)
    assert "[1] Portable - everything in this folder" in out.stdout, (out.stdout, out.stderr)
    assert "INSTALLER-RAN" in out.stdout and "is not installed yet" not in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == ""
    assert _field(out, "WINDOW-PYDIR") == str(clone / ".python")


def test_with_no_answer_possible_the_default_is_portable(tmp_path):
    clone, _, out = _run(tmp_path, answer="", system_uv=True)
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")


def test_choosing_shared_uses_the_system_uv_and_says_so_to_the_window(tmp_path):
    clone, system, out = _run(tmp_path, answer="2\n", system_uv=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists()
    assert _field(out, "WINDOW-UV") == str(system / "uv")
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == ""


def test_a_contained_uv_is_used_without_asking(tmp_path):
    clone, _, out = _run(tmp_path, answer="", system_uv=True, contained_uv=True)
    assert "[1] Portable" not in out.stdout and "INSTALLER-RAN" not in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")


def test_no_uv_anywhere_still_offers_the_install(tmp_path):
    clone, _, out = _run(tmp_path, answer="Y\n", system_uv=False)
    assert "[1] Portable" not in out.stdout and "is not installed yet" in out.stdout
    assert "INSTALLER-RAN" in out.stdout
    assert _field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")


def test_portable_never_falls_back_to_the_system_uv_when_the_install_failed(tmp_path):
    clone, _, out = _run(tmp_path, answer="1\n", system_uv=True, installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on the system uv"
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1
