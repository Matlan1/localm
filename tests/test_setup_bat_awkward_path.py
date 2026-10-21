# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.bat run from a folder whose name holds characters cmd.exe treats as syntax
(``&``, ``!``, ``%``, ``^``, parentheses, spaces): the banners name the folder
exactly and nothing in the name is run as a command."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_BAT = ROOT / "setup.bat"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup.bat runs under cmd.exe")

NAMES = [
    "Lo cal&M!x",
    "a&b",
    "one&echo INJECTED&two",
    "bang!x",
    "pct%y",
    "caret^z",
    "(paren)s",
    "all & of ! them % ^ (here)",
]


def _run(folder: Path, *args: str, abort_after: str | None = None, answers: str = "\n\n\n\n"):
    env = dict(os.environ)
    env.pop("LOCALM_SETUP_ABORT_AFTER", None)
    if abort_after:
        env["LOCALM_SETUP_ABORT_AFTER"] = abort_after
    captured = folder.parent / (folder.name.encode("ascii", "replace").hex() + ".out.txt")
    with open(captured, "wb") as sink:
        r = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", r".\setup.bat", *args],
                           cwd=str(folder), env=env,
                           input=answers.replace("\n", "\r\n").encode("utf-8"),
                           stdout=sink, stderr=subprocess.STDOUT, timeout=120)
    return r.returncode, captured.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(params=NAMES)
def folder(request, tmp_path):
    d = tmp_path / request.param
    d.mkdir()
    shutil.copy(SETUP_BAT, d / "setup.bat")
    return d


def test_the_install_banner_names_the_folder_and_runs_nothing(folder):
    rc, out = _run(folder, abort_after="portable-choice")
    assert rc == 99, out
    assert f"LocaLM setup - self-contained install in: \"{folder}\"" in out
    assert "is not recognized" not in out
    assert "INJECTED" not in out.replace(str(folder), "")


def test_the_uninstall_banner_names_the_folder_and_runs_nothing(folder):
    _rc, out = _run(folder, "uninstall", "--yes")
    assert f'   "{folder}"' in out
    assert "is not recognized" not in out
    assert "INJECTED" not in out.replace(str(folder), "")
