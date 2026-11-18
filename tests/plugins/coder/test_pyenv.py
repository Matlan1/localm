# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder runs a project's pytest with the PROJECT's Python."""

import os
import sys

import pytest

from localm.plugins.coder import pyenv, verify
from localm.plugins.coder.tools.shell import (
    _detect_test_runner,
    tool_run_tests,
)


def _make_venv(root, name=".venv", pytest_installed=True):
    """A venv skeleton in this platform's layout (files only, never started)."""
    posix = os.name != "nt"
    venv = root / name
    bindir = venv / ("bin" if posix else "Scripts")
    bindir.mkdir(parents=True)
    exe = bindir / ("python" if posix else "python.exe")
    exe.write_text("")
    (venv / "pyvenv.cfg").write_text("home = x\n")
    sp = (venv / "lib" / "python3.12" / "site-packages") if posix \
        else (venv / "Lib" / "site-packages")
    sp.mkdir(parents=True)
    if pytest_installed:
        (sp / "pytest").mkdir()
        (sp / "pytest" / "__init__.py").write_text("")
    return exe


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(pyenv, "_path_interpreters", lambda cwd: [])
    monkeypatch.setattr(pyenv, "_own_interpreter_has_pytest", lambda: False)
    monkeypatch.setattr(
        "localm.plugins.coder.hook_trust.project_trusted", lambda cwd: True)


@pytest.mark.parametrize("name", [".venv", "venv", "env", ".env"])
def test_project_venv_with_pytest_is_used(tmp_path, name):
    exe = _make_venv(tmp_path, name)
    assert pyenv.find_project_python(tmp_path).python == str(exe)


def test_venv_without_pytest_is_declined_with_an_install_hint(tmp_path):
    _make_venv(tmp_path, pytest_installed=False)
    found = pyenv.find_project_python(tmp_path)
    assert found.python is None
    assert ".venv" in found.reason and "pip install pytest" in found.reason


def test_no_python_at_all_says_what_to_do(tmp_path):
    found = pyenv.find_project_python(tmp_path)
    assert found.python is None
    assert "python -m venv .venv" in found.reason


def test_a_directory_without_pyvenv_cfg_is_not_a_venv(tmp_path):
    exe = _make_venv(tmp_path)
    (tmp_path / ".venv" / "pyvenv.cfg").unlink()
    assert exe.is_file()
    assert pyenv.find_project_python(tmp_path).python is None


def test_project_venv_beats_localm_own_interpreter(tmp_path, monkeypatch):
    exe = _make_venv(tmp_path)
    monkeypatch.setattr(pyenv, "_own_interpreter_has_pytest", lambda: True)
    assert pyenv.find_project_python(tmp_path).python == str(exe)


def test_localm_own_interpreter_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(pyenv, "_own_interpreter_has_pytest", lambda: True)
    assert pyenv.find_project_python(tmp_path).python == sys.executable


def test_path_python_with_pytest_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(pyenv, "_path_interpreters", lambda cwd: ["X:/py"])
    monkeypatch.setattr(pyenv, "_probe_has_pytest", lambda exe: True)
    assert pyenv.find_project_python(tmp_path).python == "X:/py"


def test_path_interpreter_inside_the_project_is_skipped(tmp_path, monkeypatch):
    planted = tmp_path / ("python.exe" if os.name == "nt" else "python3")
    planted.write_text("")
    monkeypatch.undo()
    monkeypatch.setattr(pyenv.shutil, "which", lambda name: str(planted))
    assert pyenv._path_interpreters(tmp_path) == []


def test_probe_runs_a_real_interpreter_in_isolated_mode(tmp_path):
    assert pyenv._probe_has_pytest(sys.executable) is True
    assert pyenv._probe_has_pytest(str(tmp_path / "missing")) is False


def test_detection_never_starts_the_project_interpreter(tmp_path, monkeypatch):
    _make_venv(tmp_path)
    started = []
    monkeypatch.setattr(pyenv.subprocess, "run",
                        lambda *a, **k: started.append(a))
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert verify.detect_verify_command(tmp_path) is not None
    assert started == []


def test_oracle_and_run_tests_share_the_project_python(tmp_path):
    exe = _make_venv(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    assert verify.detect_verify_command(tmp_path)[0] == str(exe)
    assert _detect_test_runner(tmp_path)[0] == str(exe)


def test_run_tests_without_a_project_python_says_so_and_runs_nothing(
        tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("localm.plugins.coder.tools.shell.run_subprocess",
                        lambda *a, **k: called.append(a))
    for runner in ("auto", "pytest"):
        r = tool_run_tests(tmp_path, runner=runner)
        assert not r.ok
        assert "No Python with pytest" in (r.output or r.summary or "")
    assert called == []
