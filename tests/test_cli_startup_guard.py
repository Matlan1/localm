# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CLI's first steps: an interpreter without localm's dependencies gets the
venv message from every entry point, the repository-root stub names the right
launcher, and a help request runs no pending update."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _bare_python(code: str) -> subprocess.CompletedProcess:
    """Run *code* in the base interpreter with no site-packages (`-S`), the
    checkout first on sys.path."""
    base = getattr(sys, "_base_executable", None) or sys.executable
    prog = f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); {code}"
    return subprocess.run([base, "-S", "-c", prog], capture_output=True, text=True,
                          timeout=120, cwd=str(REPO_ROOT), creationflags=_NO_WINDOW)


def _assert_venv_message(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert "localm must be run from its virtual environment" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr


def test_the_bare_interpreter_really_lacks_the_dependencies():
    proc = _bare_python("import importlib.util as u; "
                        "print([m for m in ('click', 'fastapi', 'rich') if u.find_spec(m)])")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_python_m_localm_without_its_dependencies_shows_the_venv_message():
    _assert_venv_message(_bare_python("import runpy; runpy.run_module('localm', run_name='__main__')"))


def test_the_console_script_without_its_dependencies_shows_the_venv_message():
    _assert_venv_message(_bare_python("from localm.cli import console_main; console_main()"))


def test_rich_counts_as_a_dependency_the_guard_checks(monkeypatch):
    import importlib.util

    import localm._venvguard as vg
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "rich" else real(name, *a))
    assert vg._runtime_deps_present() is False


def test_the_repository_stub_names_the_launcher_and_not_the_system_python():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "localm.py")], capture_output=True,
                          text=True, timeout=60, cwd=str(REPO_ROOT), creationflags=_NO_WINDOW)
    assert proc.returncode == 1
    assert "system Python" not in proc.stderr
    launcher = r".venv\Scripts\localm" if sys.platform == "win32" else ".venv/bin/localm"
    assert launcher in proc.stderr


@pytest.fixture
def update_calls(monkeypatch):
    from localm.cli import _core
    calls: list = []
    monkeypatch.setattr(_core, "_update_steps_at_start", lambda command: calls.append(command))
    monkeypatch.setattr("localm.bugreport.install_global_handlers", lambda: None)
    return calls


@pytest.mark.parametrize("argv", [
    ["serve", "-h"], ["gui", "--help"], ["run", "-h"], ["doctor", "-h"], ["image", "-h"],
    ["rag", "add", "-h"],
], ids=lambda a: " ".join(a))
def test_a_help_request_runs_no_pending_update(update_calls, argv):
    from localm.cli import main
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert update_calls == []


@pytest.mark.parametrize("path", [["run"], ["rag", "add"], ["key", "list"]], ids=" ".join)
def test_h_is_help_whether_the_command_was_first_invoked_on_its_own_or_through_localm(path):
    from localm.cli import main
    cmd = main
    for name in path:
        cmd = cmd.commands[name]
    CliRunner().invoke(cmd, ["-h"])
    for argv in (["-h"], ["--help"]):
        alone = CliRunner().invoke(cmd, argv)
        assert alone.exit_code == 0 and "Usage:" in alone.output, alone.output
    through = CliRunner().invoke(main, [*path, "-h"])
    assert through.exit_code == 0, through.output
    assert "No such option" not in through.output


def test_every_localm_command_names_h_as_help():
    from localm.cli import main
    from localm.cli._core import HELP_OPTION_NAMES

    def walk(cmd, path):
        yield path, cmd
        for name, sub in getattr(cmd, "commands", {}).items():
            yield from walk(sub, f"{path} {name}")

    missing = [p for p, c in walk(main, "localm")
               if c.context_settings.get("help_option_names") != HELP_OPTION_NAMES]
    assert missing == []


def test_a_command_still_runs_the_pending_update_first(update_calls):
    from localm.cli import main
    result = CliRunner().invoke(main, ["rag", "list", "--", "-h"])
    assert update_calls == ["rag"], result.output
