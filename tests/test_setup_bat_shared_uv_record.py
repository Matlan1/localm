# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat's :record_shared_uv records the uv a Shared install used, keeping
a "!" in its path, and records nothing on Portable. The real subroutine runs
under cmd.exe with delayed expansion on (as setup.bat runs), a uv.exe placed in
a folder named with "!", and a stub .venv python that writes its arguments."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="runs a setup.bat block")

ROOT = Path(__file__).resolve().parents[1]


def _subroutine() -> str:
    text = (ROOT / "setup.bat").read_text(encoding="utf-8").replace("\r\n", "\n")
    a = text.index("\n:record_shared_uv\n") + 1
    b = text.index("endlocal\ngoto :eof\n", a) + len("endlocal\ngoto :eof\n")
    return text[a:b]


@pytest.mark.parametrize("contained,recorded", [("0", True), ("1", False)])
def test_the_shared_uv_is_recorded_with_its_bang(tmp_path, contained, recorded):
    uvdir = tmp_path / "tools!x"
    uvdir.mkdir()
    (uvdir / "uv.exe").write_bytes(b"")
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    out = tmp_path / "args.txt"
    (scripts / "python.bat").write_bytes(
        f'@echo off\r\nsetlocal DisableDelayedExpansion\r\necho %*> "{out}"\r\n'.encode())
    driver = tmp_path / "driver.bat"
    driver.write_bytes(("@echo off\nsetlocal EnableDelayedExpansion\n"
                        f'set "CONTAINED={contained}"\ncall :record_shared_uv\nexit /b 0\n'
                        + _subroutine()).replace("\n", "\r\n").encode("utf-8"))
    env = {**os.environ, "PATH": f"{uvdir};{os.environ['SystemRoot']}\\System32"}
    done = subprocess.run(["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    if recorded:
        args = out.read_text(encoding="utf-8")
        assert f'--uv-exe "{uvdir / "uv.exe"}"' in args, args
    else:
        assert not out.exists()


def _shortcut_choice() -> str:
    text = (ROOT / "setup.bat").read_text(encoding="utf-8").replace("\r\n", "\n")
    a = text.index('set "SCPICK="\n')
    b = text.index('if not defined SCPICK set "SCPICK=1"\n', a)
    b = text.index("\n", text.index("y taken as", b)) + 1
    return text[a:b].replace("call :flush\n", "")


@pytest.mark.parametrize("answer,picked", [("y", "1"), ("Yes", "1"), ("", "1"), ("3", "3")])
def test_a_habitual_yes_at_the_shortcut_question_takes_the_default(tmp_path, answer, picked):
    driver = tmp_path / "driver.bat"
    driver.write_bytes(("@echo off\nsetlocal EnableDelayedExpansion\n" + _shortcut_choice()
                        + "echo PICK=[%SCPICK%]\nexit /b 0\n").replace("\n", "\r\n").encode())
    done = subprocess.run(["cmd.exe", "/d", "/c", str(driver)], cwd=tmp_path, input=answer + "\n",
                          capture_output=True, text=True, timeout=60)
    assert f"PICK=[{picked}]" in done.stdout, done.stdout + done.stderr
    assert ("y taken as [1]" in done.stdout) is answer.lower().startswith("y")
