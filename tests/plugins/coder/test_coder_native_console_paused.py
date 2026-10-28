# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder` runs its session with the native console paused, so the
native log lines of the embedding model episodic recall loads go to the debug
log, not into the terminal. The pause ends with the command."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm import debuglog
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


class _Console:
    """Stands in for the stderr duplicate dedup_native_stderr writes to."""

    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def session(monkeypatch):
    """The coder's backend and agent stubbed; the task and the REPL each open a
    real dedup_native_stderr scope and write a native line to fd 2, recording
    whether the native console was paused."""
    console = _Console()
    monkeypatch.setattr(debuglog, "_stable_console_stream", lambda: console)
    backend = MagicMock()
    backend.model_id = "stub-model"
    agent = MagicMock()
    agent._patch_chunks = []
    agent._project_map.file_count.return_value = 0
    seen = []

    def _native_load():
        seen.append(debuglog.native_console_is_paused())
        with debuglog.dedup_native_stderr():
            os.write(2, b"llama_model_loader: loaded meta data with 24 key-value pairs\n")

    def _run_single_task(agent, task):
        _native_load()
        return SimpleNamespace(success=True, response="hi", denied=())

    monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
    monkeypatch.setattr(_main, "build_agent", lambda *a, **k: agent)
    monkeypatch.setattr(_main, "run_single_task", _run_single_task)
    monkeypatch.setattr(_main, "_repl", lambda agent: _native_load())
    monkeypatch.setattr(_main, "finish_agent", lambda a: None)
    return seen, console


def _invoke(proj, *args):
    return CliRunner().invoke(
        _main.main, ["-m", "stub-model", "--cwd", str(proj), "--no-server", *args])


def test_a_one_shot_task_keeps_native_lines_off_the_terminal(proj, session):
    seen, console = session

    result = _invoke(proj, "--output-format", "jsonl", "--yes", "do x")

    assert result.exit_code == 0, result.output
    assert seen == [True]
    assert console.writes == []
    assert "llama_model_loader" not in result.output
    assert not debuglog.native_console_is_paused()


def test_the_interactive_session_keeps_native_lines_off_the_terminal(proj, session):
    seen, console = session

    result = _invoke(proj)

    assert result.exit_code == 0, result.output
    assert seen == [True]
    assert console.writes == []
    assert "llama_model_loader" not in result.output
    assert not debuglog.native_console_is_paused()
