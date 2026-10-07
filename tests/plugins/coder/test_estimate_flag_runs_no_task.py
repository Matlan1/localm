# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --estimate TASK` prints one plan and exits: the task itself is
never run, nothing else is written to stdout, and a --resume given alongside is
not acted on."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm.plugins.coder.cli import _main


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """An active coder plugin under a throwaway HOME, and an empty project."""
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    p = tmp_path / "proj"
    p.mkdir()
    return p


@pytest.fixture
def calls(monkeypatch):
    backend = MagicMock()
    backend.model_id = "stub-model"
    agent = MagicMock()
    agent._patch_chunks = []
    estimate = MagicMock(
        side_effect=lambda a, t, fmt: print(json.dumps({"estimate": "plan"}))
        if fmt == "json" else None)
    run_task = MagicMock()
    monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
    monkeypatch.setattr(_main, "build_agent", lambda *a, **k: agent)
    monkeypatch.setattr(_main, "_run_estimate", estimate)
    monkeypatch.setattr(_main, "run_single_task", run_task)
    monkeypatch.setattr(_main, "finish_agent", lambda a: None)
    return agent, estimate, run_task


def _invoke(proj, *args):
    return CliRunner().invoke(
        _main.main, ["-m", "stub-model", "--cwd", str(proj), *args])


def test_estimate_plans_and_does_not_run_the_task(proj, calls):
    agent, estimate, run_task = calls
    result = _invoke(proj, "--estimate", "do x")
    assert result.exit_code == 0, result.output
    estimate.assert_called_once()
    run_task.assert_not_called()


def test_estimate_json_prints_exactly_one_document(proj, calls):
    agent, estimate, run_task = calls
    result = _invoke(proj, "--estimate", "--output-format", "json", "do x")
    assert result.exit_code == 0, result.output
    run_task.assert_not_called()
    docs = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    assert docs == [{"estimate": "plan"}]


def test_estimate_wins_over_resume(proj, calls):
    agent, estimate, run_task = calls
    result = _invoke(proj, "--estimate", "do x", "--resume")
    assert result.exit_code == 0, result.output
    estimate.assert_called_once()
    run_task.assert_not_called()
    agent.load_checkpoint.assert_not_called()


def test_a_one_shot_task_reports_its_progress(proj, calls):
    agent, estimate, run_task = calls
    agent.report_progress = False
    seen = {}

    def _run(a, task):
        seen["report_progress"] = a.report_progress
        return MagicMock(success=True, response="done", denied=())

    run_task.side_effect = _run
    result = _invoke(proj, "do x")
    assert result.exit_code == 0, result.output
    assert seen == {"report_progress": True}
