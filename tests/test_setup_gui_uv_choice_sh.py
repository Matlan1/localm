# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.sh: the Portable / Shared question and which uv opens the window.

The launcher asks setup.sh's question once, always, before the window: also when
./.uv already exists and when no uv exists anywhere. Portable uses only ./.uv/uv
and installs uv there when it is missing; Shared uses the uv on PATH (never
./.uv) and installs uv at its per-user location when there is none. The answer
reaches the window as LOCALM_GUI_STORE, the uv as LOCALM_GUI_UV, and a per-user
install as LOCALM_GUI_UV_INSTALLED=1.

The real script runs from a throwaway clone with curl and sha256sum replaced by
stubs (the stub installer drops a fake uv into $UV_INSTALL_DIR, or into
~/.local/bin when that is unset) and a fake uv on PATH standing in for a system
one. No network is touched.
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
           'echo "WINDOW-INSTALLED=${LOCALM_GUI_UV_INSTALLED:-}"\nexit 0\n')


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


def _run(tmp_path: Path, *, answers: str, system_uv: bool, contained_uv: bool = False,
         installer_creates_uv: bool = True, contained_on_path: bool = False):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "setup-gui.sh", clone / "setup-gui.sh")
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    if contained_uv:
        _write_exec(clone / ".uv" / "uv", _REPORT)
    stubs = tmp_path / "stubs"
    made = ('dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"\n'
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
    if contained_on_path:
        path_parts.insert(0, str(clone / ".uv"))
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PATH=os.pathsep.join(path_parts), DISPLAY=":0",
               HOME=_posix(home))
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_INSTALL_DIR",
                "UV_UNMANAGED_INSTALL"):
        env.pop(var, None)
    # Bytes, so a Windows text pipe never turns "\n" into "\r\n".
    raw = subprocess.run([_bash(), (clone / "setup-gui.sh").as_posix()], cwd=str(clone),
                         env=env, input=answers.encode("utf-8"), capture_output=True,
                         timeout=120)
    out = subprocess.CompletedProcess(raw.args, raw.returncode,
                                      raw.stdout.decode("utf-8", "replace"),
                                      raw.stderr.decode("utf-8", "replace"))
    return clone, system, home, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def _path_field(out, name: str) -> str:
    value = _field(out, name)
    return _native(value) if value else ""


def _installer_line(out) -> str:
    return next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))


QUESTION = "[1] Portable - everything in this folder"
NOT_INSTALLED = "[!] uv (the Python package manager localm builds on) is not installed."


def test_portable_with_a_contained_uv_still_asks_the_question_and_uses_it(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\n", system_uv=True, contained_uv=True)
    assert QUESTION in out.stdout, (out.stdout, out.stderr)
    assert NOT_INSTALLED not in out.stdout and "INSTALLER-RAN" not in out.stdout
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _path_field(out, "WINDOW-GUIUV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _path_field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_portable_ignores_a_system_uv_and_installs_the_contained_one(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\n\n", system_uv=True)
    assert QUESTION in out.stdout and NOT_INSTALLED in out.stdout, (out.stdout, out.stderr)
    line = _installer_line(out)
    dirs = re.findall(r"\[([^\]]*)\]", line)
    assert [_native(d) for d in dirs] == [str(clone / ".uv")] * 2, line
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"


def test_with_no_answer_possible_the_default_is_portable(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="", system_uv=True)
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"


def test_shared_uses_the_system_uv_and_says_so_to_the_window(tmp_path):
    clone, system, _, out = _run(tmp_path, answers="2\n", system_uv=True)
    assert "INSTALLER-RAN" not in out.stdout and not (clone / ".uv").exists()
    assert NOT_INSTALLED not in out.stdout
    assert _path_field(out, "WINDOW-UV") == str(system / "uv")
    assert _path_field(out, "WINDOW-GUIUV") == str(system / "uv")
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == ""
    assert _field(out, "WINDOW-INSTALLED") == ""


def test_shared_ignores_a_contained_uv(tmp_path):
    _clone, system, _, out = _run(tmp_path, answers="2\n", system_uv=True, contained_uv=True)
    assert _path_field(out, "WINDOW-UV") == str(system / "uv")


def test_no_uv_and_shared_asks_and_installs_per_user_without_an_install_dir(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="2\n\n", system_uv=False)
    assert QUESTION in out.stdout and NOT_INSTALLED in out.stdout, (out.stdout, out.stderr)
    assert _installer_line(out) == "INSTALLER-RAN INSTALL_DIR=[] UNMANAGED=[]"
    per_user = str(home / ".local" / "bin" / "uv")
    assert _path_field(out, "WINDOW-UV") == per_user
    assert _path_field(out, "WINDOW-GUIUV") == per_user
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _field(out, "WINDOW-PYDIR") == ""
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert not (clone / ".uv").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)


def test_shared_never_takes_the_contained_uv_even_from_path(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="2\nY\n", system_uv=False, contained_uv=True,
                               contained_on_path=True)
    assert NOT_INSTALLED in out.stdout and "INSTALLER-RAN" in out.stdout, (out.stdout, out.stderr)
    assert _path_field(out, "WINDOW-UV") == str(home / ".local" / "bin" / "uv")
    assert _path_field(out, "WINDOW-UV") != str(clone / ".uv" / "uv")


def test_shared_with_only_a_contained_uv_installs_the_per_user_one(tmp_path):
    _clone, _, home, out = _run(tmp_path, answers="2\nY\n", system_uv=False, contained_uv=True)
    assert _installer_line(out) == "INSTALLER-RAN INSTALL_DIR=[] UNMANAGED=[]"
    assert _path_field(out, "WINDOW-UV") == str(home / ".local" / "bin" / "uv")
    assert _field(out, "WINDOW-INSTALLED") == "1"


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


def test_the_questions_are_worded_like_setup_sh():
    gui = (ROOT / "setup-gui.sh").read_text(encoding="utf-8")
    console = (ROOT / "setup.sh").read_text(encoding="utf-8")
    for fragment in ("Keep localm's Python tooling", "[1] Portable - everything in this folder",
                     "[2] Shared   - reuse/install uv at its normal per-user location",
                     '"  Pick 1 or 2 [1]: "', NOT_INSTALLED,
                     "\"  Install it now with Astral's official installer? [Y/n]: \""):
        assert fragment in gui and fragment in console, fragment


def test_each_launcher_question_is_tagged_once():
    lines = (ROOT / "setup-gui.sh").read_text(encoding="utf-8").splitlines()
    tags = [ln.strip() for ln in lines if ln.strip().startswith("# SETUP-Q:")]
    assert tags == ["# SETUP-Q: uv-install", "# SETUP-Q: tooling-store"]
    for i, ln in enumerate(lines):
        if re.search(r"^\s*read\s+-r\s+-p", ln):
            assert lines[i - 1].strip().startswith("# SETUP-Q:"), ln
