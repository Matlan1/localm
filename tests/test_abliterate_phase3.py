# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 3 regression tests for `localm abliterate` (BUG-4).

BUG-4: the command ignored Heretic's exit status. It only guarded
``FileNotFoundError`` / ``KeyboardInterrupt`` from ``subprocess.run`` and then
unconditionally called ``_register_result`` - so a Heretic run that exited
non-zero (crash, error, or the user aborting without saving) would still try to
register a (missing or half-written) model from the target dir.

These tests drive ``localm abliterate`` through Click's CliRunner with Heretic
fully stubbed: ``runner.find_heretic`` returns a fake "found" location and
``subprocess.run`` returns a fake completed-process whose ``returncode`` we
control. ``_register_result`` is replaced with a spy so we can assert exactly
whether registration was attempted.
"""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import localm.plugins.abliterate.cli as ablcli
from localm.plugins.abliterate.runner import HereticLocation


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess carrying just a returncode."""

    def __init__(self, returncode: int):
        self.returncode = returncode


@pytest.fixture
def abl_env(tmp_path, monkeypatch):
    """Isolate the abliterate command so it reaches the subprocess call.

    cli.py binds HOME_DIR / MODELS_DIR / ensure_dirs at import time, so we patch
    them on the cli module itself. Heretic is reported as found (a repo dir) so
    the command builds a command and runs it without prompting to clone.
    Registration is replaced by a spy.
    """
    home = tmp_path / ".localm"
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ablcli, "HOME_DIR", home)
    monkeypatch.setattr(ablcli, "MODELS_DIR", models)
    monkeypatch.setattr(ablcli, "load_config", lambda: {"heretic_path": str(home)})
    monkeypatch.setattr(ablcli, "ensure_dirs", lambda: None)

    # Heretic is "found" as a checkout dir; build_command will emit `uv run heretic`.
    location = HereticLocation(repo_dir=home, executable=None)
    monkeypatch.setattr(ablcli.runner, "find_heretic", lambda *a, **k: location)

    # Spy on registration so we can assert whether it was invoked.
    calls: list[tuple] = []
    monkeypatch.setattr(
        ablcli, "_register_result", lambda target, reg_name: calls.append((target, reg_name))
    )

    return SimpleNamespace(calls=calls, models=models)


def _run(returncode, monkeypatch, abl_env):
    """Invoke `abliterate` with subprocess.run stubbed to a given returncode."""
    monkeypatch.setattr(
        ablcli.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode)
    )
    return CliRunner().invoke(ablcli.main, ["--model", "acme/widget"])


def test_nonzero_exit_does_not_register(abl_env, monkeypatch):
    """Adversarial case: Heretic exits non-zero -> NO registration, clear failure."""
    result = _run(3, monkeypatch, abl_env)

    assert abl_env.calls == [], "registration must be skipped after a non-zero Heretic exit"
    # The command should surface a failure rather than a success.
    lowered = result.output.lower()
    assert "heretic" in lowered
    assert ("fail" in lowered or "exit" in lowered or "did not" in lowered), (
        f"expected a failure message, got: {result.output!r}"
    )


def test_zero_exit_registers(abl_env, monkeypatch):
    """Happy path: a clean exit (returncode 0) still registers the result."""
    result = _run(0, monkeypatch, abl_env)

    assert len(abl_env.calls) == 1, "a zero exit must still register the saved model"
    target, reg_name = abl_env.calls[0]
    # Registers under the derived name in the (patched) MODELS_DIR.
    assert reg_name == "widget-heretic"
    assert target == (abl_env.models / "widget-heretic").resolve()
    assert result.exit_code == 0


def test_keyboard_interrupt_does_not_register(abl_env, monkeypatch):
    """Aborting Heretic mid-run (KeyboardInterrupt) must not register anything."""

    def _boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ablcli.subprocess, "run", _boom)
    result = CliRunner().invoke(ablcli.main, ["--model", "acme/widget"])

    assert abl_env.calls == [], "registration must be skipped when the run is interrupted"
    assert "interrupt" in result.output.lower()
