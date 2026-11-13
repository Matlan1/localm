# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localcoder`` run from an interpreter without localm's dependencies prints
the venv message, not a ModuleNotFoundError traceback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _bare_python(code: str) -> subprocess.CompletedProcess:
    """Run *code* in the base interpreter with no site-packages (`-S`), the
    checkout first on sys.path."""
    base = getattr(sys, "_base_executable", None) or sys.executable
    prog = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); {code}"
    return subprocess.run([base, "-S", "-c", prog], capture_output=True, text=True,
                          timeout=120, cwd=str(REPO_ROOT), creationflags=_NO_WINDOW)


def test_the_bare_interpreter_really_lacks_click():
    proc = _bare_python("import importlib.util as u; print(bool(u.find_spec('click')))")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_localcoder_without_its_dependencies_shows_the_venv_message():
    proc = _bare_python("from localm.plugins.coder.cli import console_main; console_main()")
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert "localm must be run from its virtual environment" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
