# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat: a failed setup-llama no longer ends setup.

It prints what failed and the exact retry command, asks whether to continue
without a runtime (default yes), offers a report either way, and stops with
exit 1 only when the user says no. A run that continued says at the end that
no model can load yet and repeats the retry command, the way setup.sh does.

The real provisioning block and the real :provision_failed subroutine are cut
out of setup.bat and run under cmd.exe, with the setup-llama call replaced by a
command that fails and :offer_report replaced by a line that prints its
arguments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT = (ROOT / "setup.bat").read_text(encoding="utf-8").replace("\r\n", "\n")

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="runs setup.bat blocks under cmd.exe")

BACKEND_CALL = ".venv\\Scripts\\localm setup-llama --backend %BACKEND% %SLFORCE%"
FROM_CALL = '.venv\\Scripts\\localm setup-llama --from "!LLAMABUILD!"'


def _between(start: str, end: str) -> str:
    a = BAT.index(start)
    return BAT[a:BAT.index(end, a)]


def _provision_block(rc: int) -> str:
    block = _between("rem  SLB: the backends setup-llama has a build of",
                     "call :step_done native-runtime")
    assert block.count(BACKEND_CALL) == 1, "the setup-llama --backend call moved"
    assert block.count(FROM_CALL) == 2, "the setup-llama --from call and its retry hint moved"
    fail = f"cmd /c exit /b {rc}"
    return block.replace(BACKEND_CALL, fail).replace(FROM_CALL, fail, 1)


def _subroutine() -> str:
    start = BAT.index("\n:provision_failed\n")
    return BAT[start:BAT.index("\nrem ====", start + 1)]


def _done_lines() -> str:
    return "\n".join(ln for ln in BAT.splitlines() if ln.startswith('if "%RUNTIME_OK%"=="0" '))


def _run(tmp_path: Path, *, backend: str, answers: str, rc: int = 3):
    script = (
        "@echo off\n"
        "setlocal EnableDelayedExpansion\n"
        'set "STMP=stmp"\n'
        f'set "BACKEND={backend}"\n'
        'set "SLFORCE="\n'
        'set "RUNTIME_OK=1"\n'
        + _provision_block(rc)
        + "\necho AFTER RUNTIME_OK=!RUNTIME_OK!\n"
        + _done_lines()
        + "\nexit /b 0\n"
        ":flush\nexit /b 0\n"
        ":offer_report\necho REPORT [%~1] [%~2]\nexit /b 0\n"
        + _subroutine()
        + "\n"
    )
    (tmp_path / "stmp").mkdir(exist_ok=True)
    driver = tmp_path / "driver.bat"
    driver.write_bytes(script.replace("\n", "\r\n").encode("utf-8"))
    answers_file = tmp_path / "answers.txt"
    answers_file.write_bytes(answers.replace("\n", "\r\n").encode("utf-8"))
    with answers_file.open("rb") as stdin:
        return subprocess.run(["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path, stdin=stdin,
                              capture_output=True, timeout=60, check=False)


def _out(done) -> str:
    return done.stdout.decode("utf-8", "replace")


def test_a_failed_backend_provision_continues_by_default(tmp_path):
    done = _run(tmp_path, backend="vulkan", answers="\n")
    out = _out(done)
    assert done.returncode == 0, out
    assert ("[!] Provisioning the native llama.cpp runtime (--backend vulkan) failed during setup."
            in out)
    assert "No model can load until this is fixed. Retry any time with:" in out
    assert ".venv\\Scripts\\localm setup-llama --backend vulkan --force" in out
    assert "Continue the rest of setup without a working runtime? [Y/n]:" in out
    assert ("REPORT [localm setup-llama failed] [Provisioning the native llama.cpp runtime "
            "(--backend vulkan) failed during setup. (continuing setup without a runtime)]") in out
    assert "AFTER RUNTIME_OK=0" in out
    tail = out[out.index("AFTER RUNTIME_OK=0"):]
    assert "[!] No model can load yet - the native llama.cpp runtime did not finish" in tail
    assert "provisioning. Finish it any time with:" in tail
    assert ".venv\\Scripts\\localm setup-llama --backend vulkan --force" in tail


def test_answering_no_offers_a_report_and_stops_setup(tmp_path):
    done = _run(tmp_path, backend="cuda", answers="n\n\n")
    out = _out(done)
    assert done.returncode == 1, out
    assert ("REPORT [localm setup-llama failed] [Provisioning the native llama.cpp runtime "
            "(--backend cuda) failed during setup.]") in out
    assert "continuing setup without a runtime" not in out
    assert "Aborted - re-run setup.bat when ready." in out
    assert "AFTER" not in out


def test_a_failed_own_build_continues_and_names_the_folder_with_a_bang(tmp_path):
    done = _run(tmp_path, backend="own", answers="C:\\some\\bang!path\n\n")
    out = _out(done)
    assert done.returncode == 0, out
    assert "[!] Provisioning the native llama.cpp runtime with --from failed during setup." in out
    retry = '.venv\\Scripts\\localm setup-llama --from "C:\\some\\bang!path"'
    assert out.count(retry) == 2, out
    assert "AFTER RUNTIME_OK=0" in out


def test_a_successful_provision_asks_nothing_and_prints_no_warning(tmp_path):
    done = _run(tmp_path, backend="vulkan", answers="", rc=0)
    out = _out(done)
    assert done.returncode == 0, out
    assert "Continue the rest of setup" not in out and "REPORT" not in out
    assert "AFTER RUNTIME_OK=1" in out
    assert "No model can load yet" not in out


def test_the_runtime_flag_starts_set():
    assert BAT.index('set "RUNTIME_OK=1"') < BAT.index("call :jr begin native-runtime")
