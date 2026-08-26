# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` must verify the isolated worker spawn
(multiprocessing.get_context("spawn")) that every GGUF model load and the
voice/STT engine depend on, not just that a plain subprocess can run the native
library. The ABI/GPU probes use a plain subprocess.Popen, a different code path,
so they report everything green while every model load fails with "[WinError 2]".
"""

from __future__ import annotations

import localm.cli as cli

_OK = "✓"
_FAIL = "✗"


def test_worker_spawn_check_passes_for_a_real_spawn(cli_runner):
    """A REAL spawn on this (unaffected) machine must report OK - this is the
    exact mechanism ModelRunner/voice._spawn_worker use, exercised for real,
    not mocked."""
    out = cli_runner.invoke(cli.doctor, []).output
    assert "background worker spawn" in out
    line = next(ln for ln in out.splitlines() if "background worker spawn" in ln)
    assert _OK in line
    assert _FAIL not in line


def test_worker_spawn_check_reports_failure_when_spawn_is_broken(monkeypatch):
    """Simulates a child process that can never start and confirms doctor surfaces
    it as a FAILED check with actionable guidance, rather than passing silently
    the way the ABI/GPU probes do."""
    # localm.cli.__init__ re-exports `doctor` as the click Command itself
    # (`doctor = _doctor.doctor`), shadowing the submodule name - go through
    # importlib to get the actual module, not that Command object.
    import importlib
    doctor_mod = importlib.import_module("localm.cli.doctor")
    # The spawn lives in localm.diagnostics (doctor's _check_worker_spawn is a
    # renderer over diagnostics.check_worker_spawn), so that is where `mp` has to
    # be broken. The assertion below reads doctor's RENDERED line.
    diagnostics_mod = importlib.import_module("localm.diagnostics")

    class _BrokenProcess:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise FileNotFoundError(
                "[WinError 2] The system cannot find the file specified")

    class _BrokenContext:
        def Process(self, *a, **k):
            return _BrokenProcess()

    monkeypatch.setattr(diagnostics_mod.mp, "get_context",
                        lambda name: _BrokenContext())

    from rich.console import Console
    import io
    buf = io.StringIO()
    monkeypatch.setattr(doctor_mod, "console", Console(file=buf, force_terminal=False))

    doctor_mod._check_worker_spawn()
    out = buf.getvalue()
    assert "background worker spawn FAILED" in out
