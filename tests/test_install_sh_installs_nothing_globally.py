# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one-click ``install.sh`` clones the repo and hands over to ``setup.sh``; it
installs nothing itself.

uv is installed by ``setup.sh``, inside the clone for the default Portable choice.
An ``install.sh`` that installed uv first would put it in the user's home folder
and edit their shell startup files before that choice was ever offered.

The real script runs here against stub ``git`` and ``curl`` programs on a minimal
PATH, with HOME pointing into a throwaway folder, so nothing is cloned or
downloaded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="install.sh and setup.sh are the Linux/macOS installer; Windows uses setup.bat",
)

_GIT_STUB = """#!/bin/sh
echo "git $*" >> "$STUB_LOG"
if [ "$1" = clone ]; then
  for last; do :; done
  mkdir -p "$last/.git"
  printf '#!/bin/sh\\necho "setup args: $*" >> "$STUB_LOG"\\n' > "$last/setup.sh"
fi
exit 0
"""

_CURL_STUB = """#!/bin/sh
echo "curl $*" >> "$STUB_LOG"
exit 22
"""


def _bash_path_dirs(bash: str) -> list:
    here = Path(bash).resolve().parent
    dirs = [str(here)]
    usr_bin = here.parent / "usr" / "bin"
    if usr_bin.is_dir():
        dirs.append(str(usr_bin))
    return dirs


def _run_install(tmp_path):
    bash = shutil.which("bash")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("git", _GIT_STUB), ("curl", _CURL_STUB)):
        f = stubs / name
        f.write_bytes(body.replace("\r\n", "\n").encode("utf-8"))
        f.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    log = tmp_path / "stub.log"
    log.write_text("", encoding="utf-8")
    dest = tmp_path / "clone"
    env = {
        "PATH": os.pathsep.join([str(stubs)] + _bash_path_dirs(bash)),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "STUB_LOG": str(log),
        "LOCALM_DIR": str(dest),
        "TMPDIR": str(tmp_path),
    }
    r = subprocess.run([bash, str(INSTALL_SH)], env=env, capture_output=True,
                       text=True, timeout=120)
    return r, log.read_text(encoding="utf-8"), home, dest


def test_installing_without_uv_downloads_nothing_and_hands_over_to_setup(tmp_path):
    r, log, home, dest = _run_install(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "curl" not in log, f"install.sh fetched something itself: {log}"
    assert "git clone" in log
    assert "setup args: --yes" in log, log


def test_nothing_is_written_to_the_home_folder(tmp_path):
    r, log, home, dest = _run_install(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert list(home.rglob("*")) == [], list(home.rglob("*"))


def test_the_script_no_longer_names_the_uv_installer():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "astral.sh" not in text
    assert "LOCALM_UV_BOOTSTRAPPED" not in text
    assert ".local/bin" not in text
