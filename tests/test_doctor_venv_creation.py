# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` must verify that a nested venv can actually be created (the
mechanism the managed-ComfyUI installer depends on), not only that the isolated
worker process can spawn. Without that probe, ComfyUI setup failing with
"[WinError 2]" leaves every doctor check green.
"""

from __future__ import annotations

import importlib
import io

from rich.console import Console

doctor_mod = importlib.import_module("localm.cli.doctor")

_OK = "✓"
_FAIL = "✗"


def _run_check_capturing_output(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(doctor_mod, "console", Console(file=buf, force_terminal=False))
    doctor_mod._check_venv_creation()
    return buf.getvalue()


def test_venv_creation_check_passes_for_a_real_venv(monkeypatch):
    """A REAL `-m venv` invocation on this (unaffected) machine must report OK -
    the exact mechanism managed_comfy_fresh.py uses, exercised for real."""
    out = _run_check_capturing_output(monkeypatch)
    assert "venv creation" in out
    line = next(ln for ln in out.splitlines() if "venv creation" in ln)
    assert _OK in line
    assert _FAIL not in line


def test_venv_creation_check_reports_failure_when_venv_creation_is_broken(monkeypatch):
    """The resolved interpreter cannot actually create a venv; doctor must surface
    that as a FAILED check, never a silent pass."""
    import subprocess

    class _BrokenResult:
        returncode = 1
        stdout = ""
        stderr = "Error: [WinError 2] The system cannot find the file specified"

    # _check_venv_creation() does `import subprocess` locally (same module
    # object/singleton), so patching subprocess.run here is visible to it.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _BrokenResult())

    out = _run_check_capturing_output(monkeypatch)
    assert "venv creation FAILED" in out
    assert "WinError 2" in out


def test_venv_creation_check_reports_failure_when_pip_did_not_land(monkeypatch):
    """`-m venv` can report success (return code 0, the interpreter file present)
    while its own mandatory ensurepip bootstrap failed. The managed-ComfyUI
    installer pip-installs into a venv it just created, so a pip-less venv must
    not read as doctor-green. Runs a REAL `-m venv` (unmocked) and fakes only the
    follow-up `-m pip --version` probe failing, so the check has to read that
    SECOND subprocess call rather than react to any failure."""
    import subprocess

    real_run = subprocess.run

    class _BrokenPip:
        returncode = 1
        stdout = ""
        stderr = "No module named pip"

    def _fake_run(cmd, **kwargs):
        if "pip" in cmd:
            return _BrokenPip()
        return real_run(cmd, **kwargs)

    # _check_venv_creation() does `import subprocess` locally (same module
    # object/singleton), so patching subprocess.run here is visible to it.
    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = _run_check_capturing_output(monkeypatch)
    assert "venv creation FAILED" in out
    assert "no working pip" in out.lower()
    assert "No module named pip" in out
