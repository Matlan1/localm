# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-gui.sh: the Portable / Shared question and which uv opens the window.

The launcher asks setup.sh's question once, always, before the window. Both
answers use only localm's own uv, the pinned release: Portable from ./.uv,
Shared from the localm store in the user profile
(``${XDG_DATA_HOME:-$HOME/.local/share}/localm/shared/uv``). A uv there is
reused only when it reports the pinned version; otherwise the pinned installer
puts it there. A uv on PATH is never used. The answer reaches the window as
LOCALM_GUI_STORE and Shared as LOCALM_GUI_UV_INSTALLED=1; LOCALM_GUI_UV is not
set. uv keeps its credentials store in the store's uv folder and writes no
Python shim or registry entry.
The window runs with inherited uv / pip / Python settings cleared, with
only-managed Python, and with every temporary file in ./.localm-setup-tmp.

The real script runs from a throwaway clone whose fixed system folders list a
stub folder first (curl and sha256sum stubs; the stub installer drops a fake uv
into $UV_INSTALL_DIR, or into ~/.local/bin when that is unset), with a fake uv
on PATH standing in for a system one. No network is touched.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._posix_sysdirs import PINNED_UV, bash_path, copy_with_sysdirs, fake_uv_script
from tests.test_setup_gpu_detect import _bash, _mark_executable

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")

_REPORT_BODY = ('echo "WINDOW-UV=$0"\necho "WINDOW-ARGS=$*"\n'
                'echo "WINDOW-STORE=${LOCALM_GUI_STORE:-}"\n'
                'echo "WINDOW-PYDIR=${UV_PYTHON_INSTALL_DIR:-}"\n'
                'echo "WINDOW-CACHE=${UV_CACHE_DIR:-}"\n'
                'echo "WINDOW-GUIUV=${LOCALM_GUI_UV:-}"\n'
                'echo "WINDOW-INSTALLED=${LOCALM_GUI_UV_INSTALLED:-}"\n'
                'echo "WINDOW-TMPDIR=${TMPDIR:-}"\necho "WINDOW-TMP=${TMP:-}"\n'
                'echo "WINDOW-NOCONFIG=${UV_NO_CONFIG:-}"\n'
                'echo "WINDOW-PIPCONFIG=${PIP_CONFIG_FILE:-}"\n'
                'echo "WINDOW-PREF=${UV_PYTHON_PREFERENCE:-}"\n'
                'echo "WINDOW-POISON=${UV_INDEX_URL:-}${PIP_INDEX_URL:-}${PYTHONPATH:-}'
                '${UV_PYTHON_INSTALL_MIRROR:-}${LOCALM_HOME:-}"\n'
                'echo "WINDOW-CRED=${UV_CREDENTIALS_DIR:-}"\n'
                'echo "WINDOW-BIN=${UV_PYTHON_INSTALL_BIN:-}"\n'
                'echo "WINDOW-REG=${UV_PYTHON_INSTALL_REGISTRY:-}"\n'
                'exit 0\n')
_REPORT = fake_uv_script(_REPORT_BODY)


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


def _tools_path() -> list[str]:
    if os.name == "nt":
        return [str(Path(_bash()).parent)]
    return ["/usr/bin", "/bin"]


def _run(tmp_path: Path, *, answers: str, system_uv: bool = True, contained_uv: str | None = None,
         store_uv: str | None = None, installer_creates_uv: bool = True,
         xdg: Path | None = None):
    """Runs setup-gui.sh. *contained_uv* / *store_uv* name the version a fake uv
    already in ./.uv / the Shared store reports (None: no uv there)."""
    clone = tmp_path / "clone"
    clone.mkdir()
    stubs = tmp_path / "stubs"
    copy_with_sysdirs(ROOT / "setup-gui.sh", clone / "setup-gui.sh", stubs)
    shutil.copy2(ROOT / "setup.sh", clone / "setup.sh")
    home = tmp_path / "home"
    home.mkdir()
    store = (xdg if xdg else home / ".local" / "share") / "localm" / "shared"
    if contained_uv:
        _write_exec(clone / ".uv" / "uv", fake_uv_script(_REPORT_BODY, contained_uv))
    if store_uv:
        _write_exec(store / "uv" / "uv", fake_uv_script(_REPORT_BODY, store_uv))
    made = ('dest="${UV_INSTALL_DIR:-$HOME/.local/bin}"\n'
            'mkdir -p "$dest"\n'
            f"cat > \"$dest/uv\" <<'EOT'\n{_REPORT}EOT\n"
            'chmod +x "$dest/uv"\n') if installer_creates_uv else "exit 0\n"
    installer = ('#!/bin/sh\n'
                 'echo "INSTALLER-RAN INSTALL_DIR=[${UV_INSTALL_DIR:-}] '
                 'UNMANAGED=[${UV_UNMANAGED_INSTALL:-}] TMPDIR=[${TMPDIR:-}]"\n' + made)
    _write_exec(stubs / "curl", (
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && dest=\"$2\"; shift; done\n"
        f"cat > \"$dest\" <<'STUB_END'\n{installer}STUB_END\n"))
    _write_exec(stubs / "sha256sum", f"#!/bin/sh\necho \"{_pinned_sha()}  $1\"\n")
    path_parts = [*_tools_path()]
    system = tmp_path / "systemuv"
    if system_uv:
        _write_exec(system / "uv", fake_uv_script('echo "SYSTEM-UV-RAN"\nexit 0\n'))
        path_parts.insert(0, str(system))
    env = dict(os.environ, PATH=os.pathsep.join(path_parts), DISPLAY=":0",
               HOME=bash_path(home), UV_INDEX_URL="http://poison.invalid/uv",
               PIP_INDEX_URL="http://poison.invalid/pip", PYTHONPATH=str(tmp_path),
               UV_PYTHON_INSTALL_MIRROR="http://poison.invalid/mirror",
               LOCALM_HOME=str(tmp_path / "elsewhere"), UV_PYTHON_INSTALL_BIN="1",
               UV_PYTHON_INSTALL_REGISTRY="1")
    for var in ("LOCALM_GUI_STORE", "LOCALM_GUI_UV", "LOCALM_GUI_UV_INSTALLED",
                "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_INSTALL_DIR",
                "UV_UNMANAGED_INSTALL", "XDG_DATA_HOME"):
        env.pop(var, None)
    if xdg:
        env["XDG_DATA_HOME"] = bash_path(xdg)
    # Bytes, so a Windows text pipe never turns "\n" into "\r\n".
    raw = subprocess.run([_bash(), (clone / "setup-gui.sh").as_posix()], cwd=str(clone),
                         env=env, input=answers.encode("utf-8"), capture_output=True,
                         timeout=120)
    out = subprocess.CompletedProcess(raw.args, raw.returncode,
                                      raw.stdout.decode("utf-8", "replace"),
                                      raw.stderr.decode("utf-8", "replace"))
    return clone, store, home, out


def _field(out, name: str) -> str:
    for line in out.stdout.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not reported:\n{out.stdout}\n{out.stderr}")


def _path_field(out, name: str) -> str:
    value = _field(out, name)
    return _native(value) if value else ""


def _installer_dirs(out) -> list[str]:
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("INSTALLER-RAN"))
    return [_native(d) if d else "" for d in re.findall(r"\[([^\]]*)\]", line)]


QUESTION = "[1] Portable - everything in this folder"


def _assert_isolated_window(out, clone: Path) -> None:
    assert "SYSTEM-UV-RAN" not in out.stdout
    assert "--python-preference only-managed" in _field(out, "WINDOW-ARGS")
    assert _field(out, "WINDOW-PREF") == "only-managed"
    assert _field(out, "WINDOW-NOCONFIG") == "1"
    assert _field(out, "WINDOW-PIPCONFIG") == "/dev/null"
    assert _field(out, "WINDOW-POISON") == ""
    assert _field(out, "WINDOW-BIN") == "0" and _field(out, "WINDOW-REG") == "0"
    tmp = str(clone / ".localm-setup-tmp")
    assert _path_field(out, "WINDOW-TMPDIR") == tmp and _path_field(out, "WINDOW-TMP") == tmp
    assert not (clone / ".localm-setup-tmp").exists(), "the setup temp folder was left behind"


def test_portable_with_a_pinned_contained_uv_still_asks_the_question_and_uses_it(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="1\n", contained_uv=PINNED_UV)
    assert QUESTION in out.stdout, (out.stdout, out.stderr)
    assert "INSTALLER-RAN" not in out.stdout
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-GUIUV") == ""
    assert _field(out, "WINDOW-STORE") == "portable"
    assert _path_field(out, "WINDOW-PYDIR") == str(clone / ".python")
    assert _path_field(out, "WINDOW-CACHE") == str(clone / ".cache")
    assert _path_field(out, "WINDOW-CRED") == str(clone / ".uv" / "credentials")
    assert _field(out, "WINDOW-INSTALLED") == ""
    _assert_isolated_window(out, clone)


def test_portable_ignores_a_system_uv_and_installs_the_contained_one(tmp_path):
    clone, _, home, out = _run(tmp_path, answers="1\n")
    assert QUESTION in out.stdout, (out.stdout, out.stderr)
    install_dir, unmanaged, tmpdir = _installer_dirs(out)
    assert install_dir == unmanaged == str(clone / ".uv")
    assert tmpdir.startswith(str(clone / ".localm-setup-tmp"))
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"
    assert not (home / ".local").exists()
    _assert_isolated_window(out, clone)


def test_with_no_answer_possible_the_default_is_portable(tmp_path):
    clone, _, _, out = _run(tmp_path, answers="")
    assert _path_field(out, "WINDOW-UV") == str(clone / ".uv" / "uv")
    assert _field(out, "WINDOW-STORE") == "portable"


@pytest.mark.parametrize("pick", ["1", "2"])
def test_a_uv_of_another_version_in_the_store_is_reinstalled(tmp_path, pick):
    kwargs = {"contained_uv": "0.0.1"} if pick == "1" else {"store_uv": "0.0.1"}
    clone, store, _, out = _run(tmp_path, answers=f"{pick}\n", **kwargs)
    want = clone / ".uv" if pick == "1" else store / "uv"
    assert _installer_dirs(out)[:2] == [str(want)] * 2, out.stdout
    assert _path_field(out, "WINDOW-UV") == str(want / "uv")


def test_shared_never_uses_the_system_uv_and_installs_into_the_localm_store(tmp_path):
    clone, store, home, out = _run(tmp_path, answers="2\n")
    assert QUESTION in out.stdout, (out.stdout, out.stderr)
    assert _installer_dirs(out)[:2] == [str(store / "uv")] * 2
    assert _path_field(out, "WINDOW-UV") == str(store / "uv" / "uv")
    assert _field(out, "WINDOW-GUIUV") == ""
    assert _field(out, "WINDOW-STORE") == "shared"
    assert _path_field(out, "WINDOW-PYDIR") == str(store / "python")
    assert _path_field(out, "WINDOW-CACHE") == str(store / "cache")
    assert _path_field(out, "WINDOW-CRED") == str(store / "uv" / "credentials")
    assert _field(out, "WINDOW-INSTALLED") == "1"
    assert not (clone / ".uv").exists()
    assert not (home / ".local" / "bin").exists()
    assert out.returncode == 0, (out.stdout, out.stderr)
    _assert_isolated_window(out, clone)


def test_shared_reuses_a_pinned_store_uv(tmp_path):
    _clone, store, _, out = _run(tmp_path, answers="2\n", store_uv=PINNED_UV)
    assert "INSTALLER-RAN" not in out.stdout
    assert _path_field(out, "WINDOW-UV") == str(store / "uv" / "uv")


def test_shared_ignores_a_contained_uv(tmp_path):
    clone, store, _, out = _run(tmp_path, answers="2\n", contained_uv=PINNED_UV)
    assert _path_field(out, "WINDOW-UV") == str(store / "uv" / "uv")
    assert _path_field(out, "WINDOW-UV") != str(clone / ".uv" / "uv")


def test_shared_store_follows_an_absolute_xdg_data_home(tmp_path):
    xdg = tmp_path / "xdg"
    _clone, store, _, out = _run(tmp_path, answers="2\n", xdg=xdg)
    assert store == xdg / "localm" / "shared"
    assert _path_field(out, "WINDOW-UV") == str(store / "uv" / "uv")


@pytest.mark.parametrize("pick", ["1", "2"])
def test_a_failed_uv_install_stops_and_never_falls_back_to_the_system_uv(tmp_path, pick):
    _clone, _, _, out = _run(tmp_path, answers=f"{pick}\n", installer_creates_uv=False)
    assert "INSTALLER-RAN" in out.stdout
    assert "WINDOW-UV" not in out.stdout, "the window opened on another uv"
    assert "SYSTEM-UV-RAN" not in out.stdout
    assert "uv still is not callable" in out.stdout
    assert out.returncode == 1


def test_the_questions_are_worded_like_setup_sh():
    gui = (ROOT / "setup-gui.sh").read_text(encoding="utf-8")
    console = (ROOT / "setup.sh").read_text(encoding="utf-8")
    for fragment in ("Keep localm's Python tooling", "[1] Portable - everything in this folder",
                     "[2] Shared   - localm's own uv, Python and downloads in one localm folder",
                     "your user profile, reused by every localm install (faster)",
                     '"  Pick 1 or 2 [1]: "'):
        assert fragment in gui and fragment in console, fragment
    for gone in ("Install it now with Astral's official installer?", "normal per-user location"):
        assert gone not in gui and gone not in console, gone


def test_each_launcher_question_is_tagged_once():
    lines = (ROOT / "setup-gui.sh").read_text(encoding="utf-8").splitlines()
    tags = [ln.strip() for ln in lines if ln.strip().startswith("# SETUP-Q:")]
    assert tags == ["# SETUP-Q: tooling-store"]
    for i, ln in enumerate(lines):
        if re.search(r"^\s*read\s+-r\s+-p", ln):
            assert lines[i - 1].strip().startswith("# SETUP-Q:"), ln
