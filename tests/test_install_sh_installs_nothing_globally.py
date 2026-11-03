# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one-click ``install.sh`` downloads localm and hands over to ``setup.sh``;
it installs nothing itself and needs no git.

localm comes as a GitHub source archive fetched with the system curl (else wget)
from the fixed system folders and unpacked with the system tar. uv is installed by
``setup.sh``, inside the install folder for the default Portable choice. A folder
that already holds localm is not downloaded again; its setup runs.

The real script runs with its ``LOCALM_SYSDIRS`` line listing a stub folder first
(a stub curl or wget that copies a prepared archive), a logging git stub first on
PATH, and HOME pointing into a throwaway folder, so nothing is downloaded.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="install.sh and setup.sh are the Linux/macOS installer; Windows uses setup.bat",
)

_LOG_STUB = """#!/bin/sh
echo "{name} $*" >> "$STUB_LOG"
exit 1
"""

_CURL_STUB = """#!/bin/sh
echo "curl $*" >> "$STUB_LOG"
while [ $# -gt 0 ]; do [ "$1" = -o ] && dest="$2"; shift; done
cp "$STUB_ARCHIVE" "$dest"
"""

_WGET_STUB = """#!/bin/sh
echo "wget $*" >> "$STUB_LOG"
while [ $# -gt 0 ]; do [ "$1" = -O ] && dest="$2"; shift; done
cp "$STUB_ARCHIVE" "$dest"
"""

_SETUP_STUB = '#!/bin/sh\necho "setup args: $*" >> "$STUB_LOG"\n'


def _archive(path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = _SETUP_STUB.encode("utf-8")
        info = tarfile.TarInfo("localm-master/setup.sh")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("localm-master/VERSION")
        info.size = 6
        tf.addfile(info, io.BytesIO(b"0.0.0\n"))
    path.write_bytes(buf.getvalue())


def _write_exec(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    path.chmod(0o755)


def _run_install(tmp_path, *, tool="curl", repo=None, existing=False):
    """Runs install.sh with *tool* ("curl" or "wget") as the only download
    program in its system folders."""
    bash = shutil.which("bash")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_exec(stubs / tool, _CURL_STUB if tool == "curl" else _WGET_STUB)
    sysdir = tmp_path / "sys"
    sysdir.mkdir()
    for name in ("tar", "gzip", "mkdir", "rm", "ls", "cp", "sh"):
        found = shutil.which(name)
        assert found, name
        (sysdir / name).symlink_to(found)
    pathstubs = tmp_path / "pathstubs"
    pathstubs.mkdir()
    for name in ("git", "curl", "wget"):
        _write_exec(pathstubs / name, _LOG_STUB.format(name="PATH-" + name))
    script = tmp_path / "install.sh"
    text = INSTALL_SH.read_bytes().decode("utf-8")
    sysline = 'LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"'
    assert text.count(sysline) == 1
    script.write_bytes(text.replace(
        sysline, f'LOCALM_SYSDIRS="{stubs}:{sysdir}"').encode("utf-8"))
    home = tmp_path / "home"
    home.mkdir()
    log = tmp_path / "stub.log"
    log.write_text("", encoding="utf-8")
    archive = tmp_path / "localm.tar.gz"
    _archive(archive)
    dest = tmp_path / "install"
    if existing:
        dest.mkdir()
        _write_exec(dest / "setup.sh", _SETUP_STUB)
    env = {
        "PATH": os.pathsep.join([str(pathstubs), str(Path(bash).parent)]),
        "HOME": str(home),
        "STUB_LOG": str(log),
        "STUB_ARCHIVE": str(archive),
        "LOCALM_DIR": str(dest),
        "TMPDIR": str(tmp_path),
    }
    if repo:
        env["LOCALM_REPO"] = repo
    r = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True,
                       timeout=120)
    return r, log.read_text(encoding="utf-8"), home, dest


@pytest.mark.parametrize("tool", ["curl", "wget"])
def test_downloads_the_archive_without_git_and_hands_over_to_setup(tmp_path, tool):
    r, log, home, dest = _run_install(tmp_path, tool=tool)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PATH-" not in log, log
    assert f"{tool} " in log and "/archive/refs/heads/master.tar.gz" in log, log
    assert "setup args: --yes" in log, log
    assert (dest / "VERSION").read_text() == "0.0.0\n"
    assert not (dest / ".localm-download.tar.gz").exists()


def test_an_existing_install_is_not_downloaded_again(tmp_path):
    r, log, home, dest = _run_install(tmp_path, existing=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "curl" not in log and "PATH-" not in log, log
    assert "setup args: --yes" in log, log


def test_a_repository_outside_github_is_refused(tmp_path):
    r, log, home, dest = _run_install(tmp_path, repo="https://example.invalid/x.git")
    assert r.returncode == 1
    assert "github.com" in r.stdout
    assert log == ""


def test_nothing_is_written_to_the_home_folder(tmp_path):
    r, log, home, dest = _run_install(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert list(home.rglob("*")) == [], list(home.rglob("*"))


def test_the_script_no_longer_names_the_uv_installer_or_git():
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "astral.sh" not in text
    assert "LOCALM_UV_BOOTSTRAPPED" not in text
    assert ".local/bin" not in text
    assert "git clone" not in text and "command -v" not in text
    assert "install it" not in text.lower()
