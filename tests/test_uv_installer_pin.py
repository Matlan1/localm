# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh, setup-gui.sh, setup.bat and setup-gui.bat install uv from a
version-pinned installer script that runs only when its sha256 matches, and never
pipe a download into a shell.

The four scripts carry the same version. The shell verification helpers run for
real here against a stub ``curl`` that writes known bytes, and the Windows
download-verify-run line runs for real against a release directory served
through a file:// URL, so no network is used.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
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


_HASH_PROGRAM = (
    "import hashlib, sys; "
    "print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())")

# What each hashing tool the helper may reach for prints, as macOS and older
# openssl builds print it; each stub hashes the file with Python.
_HASH_STUBS = {
    "shasum": ('[ "$1" = "-a" ] && [ "$2" = "256" ] || exit 2\n'
               'printf "%s  %s\\n" "$("{py}" -c "{prog}" "$3")" "$3"\n'),
    "openssl-new": ('[ "$1" = "dgst" ] && [ "$2" = "-sha256" ] || exit 2\n'
                    'printf "SHA2-256(%s)= %s\\n" "$3" "$("{py}" -c "{prog}" "$3")"\n'),
    "openssl-old": ('[ "$1" = "dgst" ] && [ "$2" = "-sha256" ] || exit 2\n'
                    'printf "SHA256(%s)= %s\\n" "$3" "$("{py}" -c "{prog}" "$3")"\n'),
}


def _run_fetch(tmp_path: Path, name: str, *, expected_sha: str | None = None,
               curl_fails: bool = False, hash_tool: str = "sha256sum"):
    bash = shutil.which("bash")
    stubs = tmp_path / "bin"
    stubs.mkdir(parents=True)
    curl = stubs / "curl"
    curl.write_bytes(_CURL_STUB.encode("utf-8"))
    curl.chmod(0o755)
    real_tools = ("cut", "sed") + (("sha256sum",) if hash_tool == "sha256sum" else ())
    for tool in real_tools:
        found = shutil.which(tool)
        if found:
            (stubs / tool).symlink_to(found)
    if hash_tool in _HASH_STUBS:
        program = _HASH_PROGRAM.replace('"', '\\"')
        body = _HASH_STUBS[hash_tool].format(py=sys.executable, prog=program)
        stub = stubs / hash_tool.split("-")[0]
        stub.write_text("#!/bin/sh\n" + body, encoding="utf-8", newline="\n")
        stub.chmod(0o755)
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
    r, _ = _run_fetch(tmp_path, name, expected_sha=_STUB_BYTES_SHA256, hash_tool="none")
    assert r.returncode == 1
    assert "cannot be verified and was not run" in r.stdout


@_needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("tool", ["shasum", "openssl-new", "openssl-old"])
def test_each_fallback_hashing_tool_verifies_the_installer(tmp_path, name, tool):
    ok, dest = _run_fetch(tmp_path / "ok", name, expected_sha=_STUB_BYTES_SHA256, hash_tool=tool)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert dest.read_bytes() == b"stub installer bytes"
    bad, _ = _run_fetch(tmp_path / "bad", name, hash_tool=tool)
    assert bad.returncode == 1
    assert "did not match its expected checksum" in bad.stdout


BATS = ("setup.bat", "setup-gui.bat")

_needs_windows = pytest.mark.skipif(
    os.name != "nt", reason="setup.bat and setup-gui.bat run only on Windows")


def _bat_pin(name: str, var: str) -> str:
    m = re.search(rf'^\s*set "{var}=([^"]+)"\s*$', _text(name), re.MULTILINE)
    assert m, f"{name} does not set {var}"
    return m.group(1)


@pytest.mark.parametrize("name", BATS)
def test_the_bat_pin_is_a_release_version_and_a_sha256(name):
    assert re.fullmatch(r"\d+\.\d+\.\d+", _bat_pin(name, "UV_INSTALLER_VERSION"))
    assert re.fullmatch(r"[0-9a-f]{64}", _bat_pin(name, "UV_INSTALLER_SHA256"))


def test_the_bat_scripts_carry_the_same_uv_version_as_the_sh_scripts():
    versions = {n: _pin(n, "UV_INSTALLER_VERSION") for n in SCRIPTS}
    versions.update({n: _bat_pin(n, "UV_INSTALLER_VERSION") for n in BATS})
    assert len(set(versions.values())) == 1, versions


def test_both_bat_scripts_carry_the_same_installer_checksum():
    assert (_bat_pin("setup.bat", "UV_INSTALLER_SHA256")
            == _bat_pin("setup-gui.bat", "UV_INSTALLER_SHA256"))


def _bat_install_line(name: str) -> str:
    lines = [ln for ln in _text(name).splitlines()
             if ln.startswith('powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference')]
    assert len(lines) == 1, f"{name} has {len(lines)} uv installer invocations"
    return lines[0]


@pytest.mark.parametrize("name", BATS)
def test_the_bat_installer_comes_from_the_pinned_release_and_is_never_piped_to_iex(name):
    line = _bat_install_line(name)
    assert ("https://github.com/astral-sh/uv/releases/download/'+$env:UV_INSTALLER_VERSION"
            "+'/uv-installer.ps1'") in line
    executable = [ln for ln in _text(name).splitlines()
                  if not ln.lstrip().lower().startswith(("rem ", "echo "))]
    assert [ln for ln in executable if re.search(r"\|\s*iex\b", ln, re.IGNORECASE)] == []


def test_the_bat_install_line_is_identical_in_both_scripts():
    assert _bat_install_line("setup.bat") == _bat_install_line("setup-gui.bat")


@pytest.mark.parametrize("name", BATS)
def test_the_bat_installer_runs_only_after_the_checksum_comparison(name):
    line = _bat_install_line(name)
    compare = line.index("$h -ne $env:UV_INSTALLER_SHA256")
    refuse = line.index("exit 62", compare)
    run = line.index("-File $f")
    assert compare < refuse < run
    assert line.count("-File $f") == 1


def _bat_block(name: str) -> list[str]:
    """The pin, the download-verify-run line and the refusal branches of *name*."""
    lines = _text(name).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith('set "UV_INSTALLER_VERSION='))
    end = next(i for i, ln in enumerate(lines) if ln.startswith('if "%UVRC%"=="62" goto uv_refused'))
    return [ln.strip() for ln in lines[start:end + 1]]


_STUB_INSTALLER = (
    "Get-ExecutionPolicy | Out-Null\n"
    "Set-Content -LiteralPath $env:STUB_MARKER -Value 'ran'\n")


def _pwsh7_module_path() -> str | None:
    exe = shutil.which("pwsh")
    if not exe:
        return None
    r = subprocess.run([exe, "-NoProfile", "-Command", "$env:PSModulePath"],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or None


_PWSH7_PATH = _pwsh7_module_path() if os.name == "nt" else None


def _run_bat_block(tmp_path: Path, name: str, *, release_files: dict[str, bytes],
                   expected_sha: str | None = None, version: str = "0.13.0",
                   ps_module_path: str | None = None):
    """Runs the real block from *name* in a throwaway folder, with the release
    download served from a local directory through a file:// URL."""
    releases = tmp_path / "releases"
    for rel, data in release_files.items():
        target = releases / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    block = _bat_block(name)
    joined = "\r\n".join(block)
    host = "https://github.com/astral-sh/uv/releases/download/"
    assert host in joined
    base = releases.as_uri() + "/"
    assert "%" not in base, "a percent sign in the path would be expanded by cmd"
    joined = joined.replace(host, base)
    joined = re.sub(r'^set "UV_INSTALLER_VERSION=[^"]*"',
                    f'set "UV_INSTALLER_VERSION={version}"', joined, flags=re.MULTILINE)
    if expected_sha:
        joined = re.sub(r'^set "UV_INSTALLER_SHA256=[^"]*"',
                        f'set "UV_INSTALLER_SHA256={expected_sha}"', joined, flags=re.MULTILINE)
    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    work.mkdir()
    scratch.mkdir()
    probe = work / "probe.bat"
    probe.write_bytes(("@echo off\r\nsetlocal EnableExtensions DisableDelayedExpansion\r\n"
                       'cd /d "%~dp0"\r\n' + joined + "\r\n"
                       "echo REACHED rc=%UVRC%\r\nexit /b 0\r\n"
                       ":uv_refused\r\necho REFUSED rc=%UVRC%\r\nexit /b 1\r\n").encode("utf-8"))
    env = dict(os.environ)
    env.update({"TEMP": str(scratch), "TMP": str(scratch),
                "STUB_MARKER": str(tmp_path / "marker.txt")})
    if ps_module_path:
        env["PSModulePath"] = ps_module_path
    r = subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                       env=env, timeout=120, stdin=subprocess.DEVNULL)
    leftovers = list(scratch.glob("uv-installer-*"))
    return r, (tmp_path / "marker.txt").exists(), leftovers


_STUB_SHA = hashlib.sha256(_STUB_INSTALLER.encode("utf-8")).hexdigest()
_STUB_FILES = {"0.13.0/uv-installer.ps1": _STUB_INSTALLER.encode("utf-8")}


@_needs_windows
@pytest.mark.parametrize("name", BATS)
def test_a_matching_installer_is_run_and_its_temp_file_removed(tmp_path, name):
    r, ran, leftovers = _run_bat_block(tmp_path, name, release_files=_STUB_FILES,
                                       expected_sha=_STUB_SHA)
    assert "REACHED rc=0" in r.stdout, r.stdout + r.stderr
    assert ran
    assert leftovers == []


@_needs_windows
@pytest.mark.parametrize("name", BATS)
def test_a_checksum_mismatch_is_refused_and_the_installer_never_runs(tmp_path, name):
    r, ran, leftovers = _run_bat_block(tmp_path, name, release_files=_STUB_FILES)
    assert "REFUSED rc=62" in r.stdout, r.stdout + r.stderr
    assert "did not match its expected checksum" in r.stdout
    assert not ran
    assert leftovers == []


@_needs_windows
@pytest.mark.parametrize("name", BATS)
def test_a_missing_release_is_refused_with_the_download_reason(tmp_path, name):
    r, ran, leftovers = _run_bat_block(tmp_path, name, release_files=_STUB_FILES,
                                       expected_sha=_STUB_SHA, version="9.99.99")
    assert "REFUSED rc=61" in r.stdout, r.stdout + r.stderr
    assert "Could not download the uv 9.99.99 installer" in r.stdout
    assert not ran
    assert leftovers == []


@_needs_windows
@pytest.mark.skipif(_PWSH7_PATH is None, reason="PowerShell 7 is not installed")
@pytest.mark.parametrize("name", BATS)
def test_a_powershell_7_parent_does_not_break_the_installer(tmp_path, name):
    r, ran, _ = _run_bat_block(tmp_path, name, release_files=_STUB_FILES,
                               expected_sha=_STUB_SHA, ps_module_path=_PWSH7_PATH)
    assert "REACHED rc=0" in r.stdout, r.stdout + r.stderr
    assert ran
