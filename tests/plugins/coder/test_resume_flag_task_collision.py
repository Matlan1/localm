# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --resume` (`-r`) takes an optional value: given bare, it resumes
the latest checkpoint; given a value, that value is read as a checkpoint id, not
as the TASK positional. A task typed right after --resume/-r (`-r "fix it"`) is
therefore consumed as the id, and previously crashed once load_checkpoint()
rejected the space-containing string as an invalid checkpoint id."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from localm.plugins.coder.cli._main import main


def _activate_coder_plugin(monkeypatch, home: Path):
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)


def _invoke(tmp_path, monkeypatch, *extra_args):
    _activate_coder_plugin(monkeypatch, tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()
    runner = CliRunner()
    return runner.invoke(
        main,
        ["--no-server", "--url", "http://127.0.0.1:1/v1", "-m", "stub-model",
         "--cwd", str(proj), *extra_args],
        catch_exceptions=True,
        input="\n",   # one blank line so the REPL's first read hits EOF and exits
    )


def test_resume_with_a_task_shaped_value_warns_instead_of_crashing(tmp_path, monkeypatch):
    result = _invoke(tmp_path, monkeypatch, "-r", "continue the refactor")
    assert result.exit_code == 0, result.output
    assert result.exception is None, (
        f"--resume swallowing a task must not raise: {result.exception!r}")
    assert "checkpoint id" in result.output
    assert "--resume ID TASK" in result.output


def test_resume_with_a_task_shaped_value_long_form_also_warns(tmp_path, monkeypatch):
    result = _invoke(tmp_path, monkeypatch, "--resume", "fix the bug in main.py")
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "checkpoint id" in result.output


def test_bare_resume_with_no_saved_session_is_unaffected(tmp_path, monkeypatch):
    """The ordinary "nothing to resume yet" path must still report itself the
    same way, not be swept into the new task-collision warning."""
    result = _invoke(tmp_path, monkeypatch, "--resume")
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "No interrupted session found to resume." in result.output
    assert "checkpoint id" not in result.output


def test_resume_with_an_unknown_but_id_shaped_value_reports_not_found(tmp_path, monkeypatch):
    """A single-word value that LOOKS like a checkpoint id (no spaces) is still
    tried as one and reported as not found, rather than being reinterpreted as
    a task - the collision guard is specifically about task-shaped (multi-word)
    values, not every miss."""
    result = _invoke(tmp_path, monkeypatch, "--resume", "doesnotexist123")
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "No saved session with id 'doesnotexist123'." in result.output
    assert "--resume ID TASK" not in result.output
