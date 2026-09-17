# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat's ``:find_venv_lockers`` / ``:onedrive_root`` subroutines.

``uv venv --clear`` fails with "Access is denied" / "os error 5" (or
"os error 32", a sharing violation) when something still has a file inside
``.venv`` open. Before setup.bat just tells the user to go hunt for the cause
themselves, it looks for it: any process whose executable is running from
INSIDE this folder's ``.venv``, matched by full PATH, never by process name
alone - a same-named process belonging to a different localm clone on the
same machine must never be listed or touched.

Drives the REAL subroutines, sliced out of setup.bat (never hand-retyped),
through a real cmd.exe against real spawned OS processes - the point of the
feature is picking the right process out of the machine's whole process
table, and a mocked process list would prove nothing about that.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "setup.bat"

# A disposable copy of THIS interpreter's own venv shim + a matching
# pyvenv.cfg pointing at the same already-on-disk "home" - not a copy of any
# C:\Windows system binary (tests/conftest.py's syspath guard forbids a test
# touching a real system location; a project-local file is unaffected). The
# shim alone cannot run: it locates its interpreter files via a pyvenv.cfg it
# expects to find one directory up from itself.
_PYVENV_CFG = (
    f"home = {sys.base_prefix}\n"
    "implementation = CPython\n"
    "include-system-site-packages = false\n"
)


@pytest.fixture(scope="module")
def bat():
    return BAT.read_text(encoding="utf-8", errors="replace")


def _subroutine(bat_text: str, label: str) -> str:
    start = bat_text.index(f"\n{label}\n")
    end = bat_text.index("\nexit /b 0", start) + len("\nexit /b 0")
    block = bat_text[start:end]
    assert label in block, "the subroutine moved - update this test"
    return block


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe/WMI only")
class TestFindVenvLockers:
    """The claim under test: a process is picked out by where its executable
    actually lives, not by what it happens to be named."""

    def _spawn_fake_venv_process(self, root: Path, seconds: int = 90) -> subprocess.Popen:
        """A real, long-lived process whose executable path is inside
        <root>/.venv/Scripts/python.exe - this interpreter's own venv shim,
        copied alongside a matching pyvenv.cfg so it can still find its
        interpreter files (see the module-level _PYVENV_CFG comment)."""
        venv_dir = root / ".venv"
        scripts = venv_dir / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        fake_exe = scripts / "python.exe"
        shutil.copy(sys.executable, fake_exe)
        (venv_dir / "pyvenv.cfg").write_text(_PYVENV_CFG, encoding="utf-8")
        proc = subprocess.Popen([str(fake_exe), "-c", f"import time; time.sleep({seconds})"],
                                creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.time() + 10
        while time.time() < deadline and proc.poll() is not None:
            time.sleep(0.05)
        assert proc.poll() is None, "the fake locker process did not start"
        return proc

    def _run_probe(self, tmp_path: Path, block: str) -> subprocess.CompletedProcess:
        probe = tmp_path / "probe.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            f'set "TEMP={tmp_path}"\r\n'
            f'cd /d "{tmp_path}"\r\n'
            "call :find_venv_lockers\r\n"
            "echo LOCKERS=[!LOCKERS!]\r\n"
            "echo ---file---\r\n"
            'type "%TEMP%\\localm_lockers.txt" 2>nul\r\n'
            "echo ---end---\r\n"
            "exit /b 0\r\n"
            + block.replace("\n", "\r\n") + "\r\n",
            encoding="utf-8")
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              cwd=str(tmp_path), timeout=30)

    def test_finds_a_real_process_running_from_this_venv(self, bat, tmp_path):
        block = _subroutine(bat, ":find_venv_lockers")
        proc = self._spawn_fake_venv_process(tmp_path)
        try:
            out = self._run_probe(tmp_path, block)
            assert "LOCKERS=[1]" in out.stdout, (out.stdout, out.stderr)
            assert f"{proc.pid}|python.exe" in out.stdout, (out.stdout, out.stderr)
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_a_same_named_process_in_a_different_folder_is_never_listed(
            self, bat, tmp_path):
        block = _subroutine(bat, ":find_venv_lockers")
        target = tmp_path / "target"
        target.mkdir()
        other = tmp_path / "other_clone"
        other.mkdir()
        # No process under target/.venv at all - only a same-named one
        # elsewhere, which must be invisible to a scan rooted at target.
        proc = self._spawn_fake_venv_process(other)
        try:
            probe_dir = target
            out = self._run_probe(probe_dir, block)
            assert "LOCKERS=[]" in out.stdout, (
                f"a process belonging to a DIFFERENT folder's .venv was "
                f"matched: {out.stdout}")
            assert str(proc.pid) not in out.stdout
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_no_process_at_all_leaves_lockers_empty(self, bat, tmp_path):
        block = _subroutine(bat, ":find_venv_lockers")
        (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
        out = self._run_probe(tmp_path, block)
        assert "LOCKERS=[]" in out.stdout, (out.stdout, out.stderr)


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe/PowerShell only")
class TestOnedriveRoot:
    """The claim under test: a REAL prefix match against the actual
    OneDrive/OneDriveConsumer/OneDriveCommercial env vars, not a guess."""

    def _run_probe(self, tmp_path: Path, block: str, env_lines: list) -> subprocess.CompletedProcess:
        probe = tmp_path / "probe_od.bat"
        probe.write_text(
            "@echo off\r\nsetlocal EnableDelayedExpansion\r\n"
            f'cd /d "{tmp_path}"\r\n'
            + "\r\n".join(env_lines) + "\r\n"
            "call :onedrive_root\r\n"
            "echo ONEDRIVE_ROOT=[!ONEDRIVE_ROOT!]\r\n"
            "exit /b 0\r\n"
            + block.replace("\n", "\r\n") + "\r\n",
            encoding="utf-8")
        return subprocess.run(["cmd", "/c", str(probe)], capture_output=True, text=True,
                              cwd=str(tmp_path), timeout=30)

    def test_cwd_under_the_configured_root_is_detected(self, bat, tmp_path):
        block = _subroutine(bat, ":onedrive_root")
        out = self._run_probe(tmp_path, block, [
            f'set "OneDrive={tmp_path}"',
            'set "OneDriveConsumer="',
            'set "OneDriveCommercial="',
        ])
        assert f"ONEDRIVE_ROOT=[{tmp_path}]" in out.stdout, (out.stdout, out.stderr)

    def test_cwd_not_under_the_configured_root_is_not_flagged(self, bat, tmp_path):
        block = _subroutine(bat, ":onedrive_root")
        out = self._run_probe(tmp_path, block, [
            'set "OneDrive=C:\\some\\unrelated\\place"',
            'set "OneDriveConsumer="',
            'set "OneDriveCommercial="',
        ])
        assert "ONEDRIVE_ROOT=[]" in out.stdout, (out.stdout, out.stderr)

    def test_no_onedrive_vars_set_is_not_flagged(self, bat, tmp_path):
        block = _subroutine(bat, ":onedrive_root")
        out = self._run_probe(tmp_path, block, [
            'set "OneDrive="',
            'set "OneDriveConsumer="',
            'set "OneDriveCommercial="',
        ])
        assert "ONEDRIVE_ROOT=[]" in out.stdout, (out.stdout, out.stderr)


def test_venv_show_failure_still_prints_the_double_caret_marker_exactly_once(bat):
    """test_installer_warning_markers.py's site count depends on this marker
    appearing exactly once in the whole file; this class of change (branching
    the detail text below it) is exactly the shape that could silently
    duplicate it."""
    assert bat.count('echo  [^^!] Could not create the environment.') == 1
