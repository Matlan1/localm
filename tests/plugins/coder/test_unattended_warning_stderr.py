# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder's "Unattended one-shot" warning goes to stderr, so the stdout of a
one-shot run stays free of it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm.plugins.coder.cli import _main

_WARNING = "Unattended one-shot"


@pytest.fixture
def proj(tmp_path, monkeypatch):
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
    backend = MagicMock()
    backend.model_id = "stub-model"
    agent = MagicMock()
    agent._patch_chunks = []
    monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
    monkeypatch.setattr(_main, "build_agent", lambda *a, **k: agent)
    monkeypatch.setattr(_main, "run_single_task", MagicMock())
    monkeypatch.setattr(_main, "finish_agent", lambda a: None)
    return p


def _invoke(proj, *args):
    return CliRunner().invoke(
        _main.main, ["-m", "stub-model", "--cwd", str(proj), *args])


def test_unattended_warning_is_on_stderr_not_stdout(proj):
    result = _invoke(proj, "do x")
    assert result.exit_code == 0, result.output
    assert _WARNING in result.stderr
    assert _WARNING not in result.stdout


def test_no_warning_when_yes_is_passed(proj):
    result = _invoke(proj, "--yes", "do x")
    assert result.exit_code == 0, result.output
    assert _WARNING not in result.stderr
    assert _WARNING not in result.stdout
