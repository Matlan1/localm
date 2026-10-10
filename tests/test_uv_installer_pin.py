# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh and setup-gui.sh install uv from a version-pinned installer script that
runs only when its sha256 matches, and never pipe a download into a shell.

The two scripts carry the same pin. The verification helpers run for real here
against a stub ``curl`` that writes known bytes, so no network is used.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("setup.sh", "setup-gui.sh")

_needs_bash = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="setup.sh and setup-gui.sh are the Linux/macOS installers; Windows uses setup.bat",
)


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _pin(name: str, var: str) -> str:
    m = re.search(rf'^{var}="([^"]+)"$', _text(name), re.MULTILINE)
    assert m, f"{name} does not set {var}"
    return m.group(1)


def _helpers(name: str) -> str:
    """The pin constants and the two verification functions of *name*."""
    lines = _text(name).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("UV_INSTALLER_VERSION="))
    closes = [i for i in range(start, len(lines)) if lines[i] == "}"]
    return "\n".join(lines[start:closes[1] + 1]) + "\n"


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_pin_is_a_release_version_and_a_sha256(name):
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pin(name, "UV_INSTALLER_VERSION"))
    assert re.fullmatch(r"[0-9a-f]{64}", _pin(name, "UV_INSTALLER_SHA256"))


def test_both_scripts_carry_the_same_pin():
    assert _pin("setup.sh", "UV_INSTALLER_VERSION") == _pin("setup-gui.sh", "UV_INSTALLER_VERSION")
    assert _pin("setup.sh", "UV_INSTALLER_SHA256") == _pin("setup-gui.sh", "UV_INSTALLER_SHA256")


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_installer_comes_from_the_pinned_release_and_is_never_piped_to_a_shell(name):
    text = _text(name)
    assert ("https://github.com/astral-sh/uv/releases/download/"
            "${UV_INSTALLER_VERSION}/uv-installer.sh") in text
    executable = [ln for ln in text.splitlines()
                  if not ln.lstrip().startswith(("#", "say ", "echo "))]
    piped = [ln for ln in executable if re.search(r"\|\s*(ba|z)?sh\b", ln)]
    assert piped == []


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_installer_runs_only_after_the_verified_fetch(name):
    text = _text(name)
    run = text.index('sh "$UVTMP/uv-installer.sh"')
    fetch = text.rindex('fetch_uv_installer "$UVTMP/uv-installer.sh"', 0, run)
    between = text[fetch:run]
    assert between.count("\n") <= 1 and ("then" in between or "&&" in between)


_CURL_STUB = """#!/bin/sh
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
[ "$STUB_CURL_FAIL" = 1 ] && exit 22
printf 'stub installer bytes' > "$out"
"""

_STUB_BYTES_SHA256 = hashlib.sha256(b"stub installer bytes").hexdigest()


def _run_fetch(tmp_path: Path, name: str, *, expected_sha: str | None = None,
               curl_fails: bool = False, hash_tools: bool = True):
    bash = shutil.which("bash")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    curl = stubs / "curl"
    curl.write_bytes(_CURL_STUB.encode("utf-8"))
    curl.chmod(0o755)
    for tool in ("cut", "sed") + (("sha256sum", "shasum", "openssl") if hash_tools else ()):
        found = shutil.which(tool)
        if found:
            (stubs / tool).symlink_to(found)
    helpers = tmp_path / "helpers.sh"
    helpers.write_text(_helpers(name), encoding="utf-8", newline="\n")
    override = f'UV_INSTALLER_SHA256="{expected_sha}"\n' if expected_sha else ""
    dest = tmp_path / "uv-installer.sh"
    script = (f'say() {{ printf "%s\\n" "$*"; }}\n. "{helpers}"\n{override}'
              f'fetch_uv_installer "{dest}"\n')
    env = {"PATH": str(stubs), "STUB_CURL_FAIL": "1" if curl_fails else "0"}
    r = subprocess.run([bash, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    return r, dest


@_needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_a_matching_checksum_is_accepted(tmp_path, name):
    r, dest = _run_fetch(tmp_path, name, expected_sha=_STUB_BYTES_SHA256)
    assert r.returncode == 0, r.stdout + r.stderr
    assert dest.read_bytes() == b"stub installer bytes"


@_needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_a_checksum_mismatch_is_refused(tmp_path, name):
    r, _ = _run_fetch(tmp_path, name)
    assert r.returncode == 1
    assert "did not match its expected checksum" in r.stdout


@_needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_a_failed_download_is_refused(tmp_path, name):
    r, _ = _run_fetch(tmp_path, name, expected_sha=_STUB_BYTES_SHA256, curl_fails=True)
    assert r.returncode == 1
    assert "Could not download the uv" in r.stdout


@_needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_no_hashing_tool_means_refusal_not_a_skipped_check(tmp_path, name):
    r, _ = _run_fetch(tmp_path, name, expected_sha=_STUB_BYTES_SHA256, hash_tools=False)
    assert r.returncode == 1
    assert "cannot be verified and was not run" in r.stdout
