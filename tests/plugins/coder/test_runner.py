# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared one-shot runner (plugins/coder/runner.py).

The CLI and the MCP server both build and run their Agent through it, so the
settings resolution and the Agent construction live in one place. These tests
pin that the runner resolves a project exactly as the CLI's own helper does,
that an unattended task denies the shell tools without ``yes``, that a run
reports its outcome and leaves no checkpoint claimed, and that a timed-out run
is reported as such while the worker is left to finish and release on its own.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from localm.plugins.coder import runner
from localm.plugins.coder.agent.constants import _SHELL_EXEC_TOOLS
from localm.plugins.coder.audit import SessionMode


class _Scripted:
    """One canned reply per chat() call, repeating the last."""

    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage: dict = {"total_tokens": 5}
    last_reasoning = ""

    def __init__(self, responses, gate: threading.Event | None = None):
        self.responses = list(responses)
        self.calls = 0
        self.gate = gate

    def chat(self, messages, **kw):
        if self.gate is not None:
            self.gate.wait()
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def chat_stream(self, messages, on_reasoning=None, **kw):
        yield self.chat(messages, **kw)

    def set_tools(self, defs):
        pass

    def context_capacity(self):
        return None


def _tc(name: str, **args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "args": args}) + "</tool_call>"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    (p / "a.py").write_text("x = 1\n", encoding="utf-8")
    return p


def _agent_patches():
    return (patch("localm.plugins.coder.agent.ProjectMap"),
            patch("localm.plugins.coder.agent.make_audit_log"),
            patch("localm.plugins.coder.agent.load_memory", return_value=""))


def _build(backend, project, task, **kw):
    pm, audit, mem = _agent_patches()
    with pm as MockPM, audit, mem:
        MockPM.build.return_value.file_count.return_value = 0
        return runner.build_agent(
            backend, project, task=task, max_turns=kw.pop("max_turns", 5),
            auto_approve=kw.pop("auto_approve", False),
            always_confirm=kw.pop("always_confirm", ()),
            session_mode=kw.pop("session_mode", SessionMode.LOG), **kw)


# --------------------------------------------------------------------------- #
#  resolve_task_config: the same answer as the CLI helper                     #
# --------------------------------------------------------------------------- #

class TestResolveTaskConfig:
    def test_defaults_without_a_project_config(self, home, project):
        cfg = runner.resolve_task_config(project)
        assert cfg.model is None
        assert cfg.max_turns == runner.DEFAULT_MAX_TURNS
        assert cfg.auto_approve is False
        assert cfg.always_confirm == frozenset()
        assert isinstance(cfg.session_mode, SessionMode)
        assert "max_tokens" in cfg.gen_kw
        assert "seed" not in cfg.gen_kw

    def test_cli_helper_passes_the_runners_values_through_unchanged(self, home, project):
        """The CLI wrapper delegates to resolve_task_config; this pins that it
        hands every field back untouched (the resolution itself is asserted
        directly below, and the CLI's own tests cover its behaviour)."""
        (project / ".localcoder").mkdir()
        (project / ".localcoder" / "config.toml").write_text(
            'model = "cfg-model"\nmax_turns = 7\nmax_tokens = 321\n'
            'temperature = 0.3\nseed = 11\nauto_approve = true\n'
            'always_confirm = ["write_file"]\nmode = "log"\n',
            encoding="utf-8")
        from localm.plugins.coder.cli._main import _resolve_session_config
        cli = _resolve_session_config(project, None, None, None, None, None,
                                      False, False, None, False, None)
        cfg = runner.resolve_task_config(project)
        assert (cfg.model, cfg.max_turns, cfg.auto_approve, set(cfg.always_confirm),
                cfg.session_mode, cfg.gen_kw) == (
            cli[0], cli[1], cli[2], set(cli[3]), cli[4], cli[5])
        assert cfg.model == "cfg-model"
        assert cfg.max_turns == 7
        assert cfg.auto_approve is True
        assert cfg.always_confirm == frozenset({"write_file"})
        assert cfg.session_mode == SessionMode.LOG
        assert cfg.gen_kw == {"temperature": 0.3, "max_tokens": 321, "seed": 11}

    def test_explicit_arguments_win_over_the_project_config(self, home, project):
        (project / ".localcoder").mkdir()
        (project / ".localcoder" / "config.toml").write_text(
            'model = "cfg-model"\nmax_turns = 7\nmode = "log"\n', encoding="utf-8")
        cfg = runner.resolve_task_config(project, model="m", max_turns=2,
                                         mode="full", interactive_confirm=True)
        assert cfg.model == "m"
        assert cfg.max_turns == 2
        assert cfg.session_mode == SessionMode.FULL
        assert set(_SHELL_EXEC_TOOLS) <= set(cfg.always_confirm)

    def test_bad_mode_raises_a_typed_error(self, home, project):
        with pytest.raises(runner.InvalidSessionMode):
            runner.resolve_task_config(project, mode="cloud")

    def test_unreadable_project_config_propagates(self, home, project):
        from localm.plugins.coder.project_config import ProjectConfigUnreadable
        (project / ".localcoder").mkdir()
        (project / ".localcoder" / "config.toml").write_text(
            "this is = not [ toml\n", encoding="utf-8")
        with pytest.raises(ProjectConfigUnreadable):
            runner.resolve_task_config(project)


# --------------------------------------------------------------------------- #
#  build_agent: the unattended shell gate                                     #
# --------------------------------------------------------------------------- #

class TestBuildAgent:
    def test_unattended_task_confirm_gates_the_shell_tools(self, home, project):
        agent = _build(_Scripted(["done"]), project, "do it", auto_approve=False)
        try:
            assert set(_SHELL_EXEC_TOOLS) <= set(agent.always_confirm)
            assert agent.auto_approve is True
        finally:
            agent.close()

    def test_yes_leaves_the_shell_tools_unconfirmed(self, home, project):
        agent = _build(_Scripted(["done"]), project, "do it", auto_approve=True)
        try:
            assert not (set(_SHELL_EXEC_TOOLS) & set(agent.always_confirm))
        finally:
            agent.close()

    def test_interactive_session_is_not_gated(self, home, project):
        agent = _build(_Scripted(["done"]), project, "", auto_approve=False)
        try:
            assert not (set(_SHELL_EXEC_TOOLS) & set(agent.always_confirm))
            assert agent.auto_approve is False
        finally:
            agent.close()

    def test_configured_always_confirm_is_kept(self, home, project):
        agent = _build(_Scripted(["done"]), project, "do it", auto_approve=True,
                       always_confirm=frozenset({"write_file"}))
        try:
            assert "write_file" in agent.always_confirm
        finally:
            agent.close()


# --------------------------------------------------------------------------- #
#  run_single_task / finish_agent                                             #
# --------------------------------------------------------------------------- #

class TestRunSingleTask:
    def test_reports_the_outcome_and_counters(self, home, project):
        backend = _Scripted(["all done"])
        agent = _build(backend, project, "say done", auto_approve=True)
        try:
            result = runner.run_single_task(agent, "say done")
        finally:
            runner.finish_agent(agent)
        assert result.success is True
        assert "all done" in result.response
        assert result.turns >= 1
        assert result.total_tokens >= 5
        assert result.timed_out is False
        assert result.as_dict() == {
            "success": True, "response": result.response,
            "turns": result.turns, "total_tokens": result.total_tokens,
            "denied": []}

    def test_a_run_leaves_no_checkpoint_claimed(self, home, project):
        from localm.plugins.coder.agent.checkpoint import checkpoint_info
        before = checkpoint_info(project)
        agent = _build(_Scripted(["done"]), project, "say done", auto_approve=True)
        try:
            runner.run_single_task(agent, "say done")
        finally:
            runner.finish_agent(agent)
        assert checkpoint_info(project) == before

    def test_shell_tool_is_denied_without_yes_and_runs_with_it(self, home, project):
        marker = project / "marker.txt"
        cmd = f'"{sys.executable}" -c "open(\'marker.txt\', \'w\').close()"'
        script = [_tc("run_shell", command=cmd), "finished"]

        agent = _build(_Scripted(script), project, "make the marker", auto_approve=False)
        try:
            runner.run_single_task(agent, "make the marker")
        finally:
            runner.finish_agent(agent)
        assert not marker.exists(), "run_shell executed without a confirmation channel"

        agent = _build(_Scripted(list(script)), project, "make the marker",
                       auto_approve=True)
        try:
            runner.run_single_task(agent, "make the marker")
        finally:
            runner.finish_agent(agent)
        assert marker.exists(), "run_shell did not execute under yes"


# --------------------------------------------------------------------------- #
#  run_task_with_timeout                                                      #
# --------------------------------------------------------------------------- #

def _wait_until(pred, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class TestRunTaskWithTimeout:
    def test_completes_and_calls_on_finished_once(self, home, project):
        agent = _build(_Scripted(["done"]), project, "t", auto_approve=True)
        finished = []
        result = runner.run_task_with_timeout(
            agent, "t", 30.0, on_finished=lambda: finished.append(1))
        assert result.success is True
        assert result.timed_out is False
        assert finished == [1]

    def test_timeout_reports_and_releases_only_when_the_worker_ends(
            self, home, project, monkeypatch):
        monkeypatch.setattr(runner, "STOP_GRACE_SECONDS", 0.2)
        gate = threading.Event()
        backend = _Scripted(["done"], gate=gate)
        agent = _build(backend, project, "t", auto_approve=True)
        finished = []
        result = runner.run_task_with_timeout(
            agent, "t", 0.3, on_finished=lambda: finished.append(1))
        try:
            assert result.timed_out is True
            assert result.success is False
            assert "timed out" in result.response
            # The worker is still inside its generation: nothing released yet,
            # and the stop was requested for it.
            assert finished == []
            assert agent._stop_requested is True
        finally:
            gate.set()
        assert _wait_until(lambda: finished == [1]), "worker never released"

    def test_worker_exception_propagates_after_release(self, home, project):
        class _Boom(_Scripted):
            def chat(self, messages, **kw):
                raise RuntimeError("model gone")

        agent = _build(_Boom(["x"]), project, "t", auto_approve=True)
        finished = []
        with pytest.raises(RuntimeError, match="model gone"):
            runner.run_task_with_timeout(
                agent, "t", 30.0, on_finished=lambda: finished.append(1))
        assert finished == [1]


# --------------------------------------------------------------------------- #
#  Import graph                                                               #
# --------------------------------------------------------------------------- #

def test_runner_imports_neither_the_cli_module_nor_torch(tmp_path):
    """The MCP server imports the runner in-process: the CLI module reconfigures
    the process's stdout at import and torch must never enter a process that
    hosts the native runtime."""
    code = (
        "import sys\n"
        "import localm.plugins.coder.runner\n"
        "import localm.plugins.coder.backends.shared_engine\n"
        "bad = [m for m in ('torch', 'localm.plugins.coder.cli._main') "
        "if m in sys.modules]\n"
        "print('BAD=' + ','.join(bad))\n"
    )
    env = {"LOCALM_HOME": str(tmp_path)}
    import os
    env.update({k: v for k, v in os.environ.items() if k != "LOCALM_HOME"})
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, env=env,
                          cwd=str(Path(runner.__file__).resolve().parents[3]))
    assert proc.returncode == 0, proc.stderr
    assert "BAD=\n" in proc.stdout or proc.stdout.strip().endswith("BAD="), proc.stdout
