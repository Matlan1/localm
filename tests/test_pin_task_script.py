# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/setup_pin_pipeline_task.ps1: the command it schedules.

The scheduled command is taken from the script's own dry-run output and run for real
against a stand-in Python script, so the log encoding, buffering and exit-code
behaviour that matter to an unattended run are what is tested.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PS1 = _ROOT / "scripts" / "setup_pin_pipeline_task.ps1"
_POWERSHELL = shutil.which("powershell.exe")

pytestmark = pytest.mark.skipif(os.name != "nt" or _POWERSHELL is None,
                                reason="the scheduled task is a Windows Task Scheduler entry")


@pytest.fixture
def scratch_repo(tmp_path):
    """A repo-shaped directory the script accepts: the venv python path exists and so
    does scripts/pin_weekly.py."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("x", encoding="utf-8")
    (tmp_path / "scripts" / "pin_weekly.py").write_text("x", encoding="utf-8")
    (tmp_path / "scripts" / "pin_pipeline.py").write_text("x", encoding="utf-8")
    shutil.copy(_PS1, tmp_path / "scripts" / _PS1.name)
    return tmp_path


def _dry_run(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(repo / "scripts" / _PS1.name), *args],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _scheduled_command(repo: Path, *args: str) -> str:
    out = _dry_run(repo, *args)
    m = re.search(r"^Command:\s+(.*?)\n(?=Log file:)", out, re.S | re.M)
    assert m, out
    return " ".join(m.group(1).split())


def test_the_weekly_command_runs_python_unbuffered_in_utf8_and_keeps_its_exit_code(scratch_repo, tmp_path):
    command = _scheduled_command(scratch_repo)
    assert "-u -X utf8" in command and "Out-File" in command and "-Encoding utf8" in command
    assert "exit $LASTEXITCODE" in command
    assert "pin_weekly.py" in command and "--pin" not in command


def test_the_scheduled_command_logs_utf8_and_returns_the_script_exit_code(scratch_repo, tmp_path):
    command = _scheduled_command(scratch_repo)
    script = tmp_path / "standin.py"
    script.write_text(
        "import sys\nprint('arrow \\u2192 done', flush=True)\nsys.exit(2)\n", encoding="utf-8")
    venv_python = str(scratch_repo / ".venv" / "Scripts" / "python.exe")
    real = command.replace(venv_python, sys.executable).replace(
        str(scratch_repo / "scripts" / "pin_weekly.py"), str(script))
    assert str(script) in real and sys.executable in real
    inner = real.split('-Command "', 1)[1].rsplit('"', 1)[0]
    proc = subprocess.run([_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", inner],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stderr
    log = scratch_repo / "dev-notes" / "pin-pipeline" / "task-log-weekly.txt"
    data = log.read_bytes()
    assert not data.startswith(b"\xff\xfe"), "the log must not be UTF-16"
    assert "arrow → done" in data.decode("utf-8-sig")


def test_a_single_pin_command_still_passes_its_pin_argument(scratch_repo):
    command = _scheduled_command(scratch_repo, "-Pin", "comfyui")
    assert "pin_pipeline.py' --pin comfyui" in command or "pin_pipeline.py --pin comfyui" in command
    assert "-u -X utf8" in command
