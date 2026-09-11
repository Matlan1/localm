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


# --------------------------------------------------------------------------- #
#  The timeout is enforcing: after it fires, the run can no longer write      #
# --------------------------------------------------------------------------- #

_SLEEP_CMD = f'"{sys.executable}" -c "import time; time.sleep(8)"'


class TestTimeoutIsEnforcing:
    def test_a_tool_call_in_flight_is_killed_and_no_later_write_lands(
            self, home, project, monkeypatch):
        """The model is inside a shell command that outlasts the timeout and
        would write a file in its next turn. The world after the timeout is
        what is asserted: the file is absent, the command was killed, and the
        worker wound down within the grace instead of finishing its plan."""
        monkeypatch.setattr(runner, "STOP_GRACE_SECONDS", 5.0)
        late = project / "late.txt"
        script = [_tc("run_shell", command=_SLEEP_CMD, timeout=60),
                  _tc("write_file", path="late.txt", content="too late"),
                  "finished"]
        backend = _Scripted(script)
        agent = _build(backend, project, "t", auto_approve=True)
        finished = []
        t0 = time.monotonic()
        result = runner.run_task_with_timeout(
            agent, "t", 0.5, on_finished=lambda: finished.append(1))
        assert _wait_until(lambda: finished == [1]), "the worker never closed"
        elapsed = time.monotonic() - t0
        assert not late.exists(), "a file was written after the timeout fired"
        assert elapsed < 6.0, f"the sleeping command was not killed ({elapsed:.1f}s)"
        assert result.timed_out is True
        assert result.success is False
        assert "cancelled" in result.response
        assert agent.cancelled is True
        assert backend.calls == 1, "the model was called again after the cancel"

    def test_a_synchronous_child_is_cancelled_with_its_parent(
            self, home, project, monkeypatch):
        """spawn_agent runs a nested loop on a different Agent object; the
        parent's cancel reaches it through the parent chain, so the child's
        sleeping command is killed and its planned write never lands."""
        monkeypatch.setattr(runner, "STOP_GRACE_SECONDS", 5.0)
        late = project / "late.txt"
        script = ["<tool_call>" + json.dumps({"name": "spawn_agent", "args": {
                      "task": "do the slow thing", "name": "kid"}}) + "</tool_call>",
                  # The child's turns, on the shared backend:
                  _tc("run_shell", command=_SLEEP_CMD, timeout=60),
                  _tc("write_file", path="late.txt", content="too late"),
                  "child finished",
                  "parent finished"]
        backend = _Scripted(script)
        agent = _build(backend, project, "t", auto_approve=True)
        finished = []
        t0 = time.monotonic()
        pm, audit, mem = _agent_patches()
        with pm as MockPM, audit, mem:
            MockPM.build.return_value.file_count.return_value = 0
            result = runner.run_task_with_timeout(
                agent, "t", 0.5, on_finished=lambda: finished.append(1))
            assert _wait_until(lambda: finished == [1]), "the worker never closed"
        elapsed = time.monotonic() - t0
        assert not late.exists(), "the child wrote a file after the parent's timeout"
        assert elapsed < 6.0, f"the child's command was not killed ({elapsed:.1f}s)"
        assert result.timed_out is True
        assert backend.calls == 2, backend.calls

    def test_cancel_reaches_every_child_through_the_parent_chain(self, home, project):
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.tools.agents import inherited_child_kwargs
        parent = _build(_Scripted(["x"]), project, "t", auto_approve=True)
        pm, audit, mem = _agent_patches()
        with pm as MockPM, audit, mem:
            MockPM.build.return_value.file_count.return_value = 0
            child = Agent(**inherited_child_kwargs(
                parent, backend=parent.backend, cwd=project, name="kid",
                max_turns=3, confirm_handler=None))
            grandchild = Agent(**inherited_child_kwargs(
                child, backend=parent.backend, cwd=project, name="grandkid",
                max_turns=3, confirm_handler=None))
        assert not grandchild.cancelled
        child.cancel("child only")
        assert child.cancelled and grandchild.cancelled
        assert not parent.cancelled, "cancelling a child must not cancel its parent"
        assert grandchild.cancel_reason == "child only"
        parent.cancel("timed out")
        assert parent.cancelled and child.cancel_reason == "child only"

    def test_no_tool_runs_after_a_cancel_even_when_the_model_asks(self, home, project):
        from localm.plugins.coder.parser import parse_tool_calls
        marker = project / "marker.txt"
        agent = _build(_Scripted(["x"]), project, "t", auto_approve=True)
        agent.cancel("timed out")
        call, = parse_tool_calls(_tc("write_file", path="marker.txt", content="x"),
                                 tool_names={"write_file"})
        result = agent._execute_tool(call, interactive=False)
        assert result.ok is False
        assert "cancelled" in result.output
        assert not marker.exists()

    def test_a_cancelled_run_does_not_reflect_on_close(self, home, project, monkeypatch):
        """The close-time episode reflection is a model call; a run its host
        cancelled makes none."""
        backend = _Scripted([_tc("write_file", path="new.txt", content="x"), "done"])
        agent = _build(backend, project, "t", auto_approve=True)
        agent._episodic = True
        agent._episode_store = object()
        reflected = []
        monkeypatch.setattr(agent, "_reflect_into_episode",
                            lambda *a, **k: reflected.append(1))
        runner.run_single_task(agent, "t")
        agent.cancel("timed out")
        runner.finish_agent(agent)
        assert reflected == []


class TestReflectionHost:
    def test_the_threaded_runner_reflects_off_the_worker(self, home, project, monkeypatch):
        """In a long-lived host the reflection runs on its own thread; the
        run's caller is not held for a close-time model call."""
        import threading as _threading
        backend = _Scripted([_tc("write_file", path="new.txt", content="x"), "done"])
        agent = _build(backend, project, "t", auto_approve=True)
        agent._episodic = True
        agent._episode_store = object()
        seen = []

        def fake_reflect(*a, **k):
            seen.append(_threading.current_thread().name)

        monkeypatch.setattr(agent, "_reflect_into_episode", fake_reflect)
        runner.run_task_with_timeout(agent, "t", 30.0)
        assert _wait_until(lambda: len(seen) == 1)
        assert seen[0] != "coder-task", "the reflection ran on the worker thread"
        assert agent.reflect_in_background is True

    def test_a_one_shot_process_still_reflects_synchronously(self, home, project, monkeypatch):
        backend = _Scripted([_tc("write_file", path="new.txt", content="x"), "done"])
        agent = _build(backend, project, "t", auto_approve=True)
        agent._episodic = True
        agent._episode_store = object()
        seen = []
        monkeypatch.setattr(agent, "_reflect_into_episode",
                            lambda *a, **k: seen.append(k.get("deadline")))
        runner.run_single_task(agent, "t")
        runner.finish_agent(agent)
        assert seen and seen[0] is not None, "the CLI branch reflects with a deadline"


class TestBackgroundChildrenOfAOneShot:
    def test_a_background_child_still_running_at_the_end_is_cancelled(
            self, home, project, monkeypatch, capsys):
        """A one-shot run that ends with a background sub-agent still going
        reports it, then cancels it: in a long-lived host nothing else would
        ever stop it, and its planned write must not land after the run has
        reported."""
        from localm.plugins.coder import background as bg
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.background import AgentJob, JobRegistry
        from localm.plugins.coder.tools.agents import inherited_child_kwargs
        reg = JobRegistry(kind_caps={"agent": 4})
        monkeypatch.setattr(bg, "_registry", reg)
        late = project / "late.txt"
        parent = _build(_Scripted(["parent done"]), project, "t", auto_approve=True)
        child_backend = _Scripted([_tc("run_shell", command=_SLEEP_CMD, timeout=60),
                                   _tc("write_file", path="late.txt", content="x"),
                                   "child done"])
        pm, audit, mem = _agent_patches()
        with pm as MockPM, audit, mem:
            MockPM.build.return_value.file_count.return_value = 0
            child = Agent(**inherited_child_kwargs(
                parent, backend=child_backend, cwd=project, name="kid",
                max_turns=5, confirm_handler=None))
        t0 = time.monotonic()
        job = reg.submit(lambda: AgentJob(child, "slow", label="kid",
                                          owner=parent.job_owner), kind="agent")
        assert _wait_until(lambda: child_backend.calls >= 1), "the child never started"
        try:
            result = runner.run_single_task(parent, "t")
            assert result.success is True
            assert child.cancelled is True
            assert _wait_until(lambda: job.state != "running"), "the child never stopped"
            elapsed = time.monotonic() - t0
            assert not late.exists(), "the child wrote after the run had reported"
            assert elapsed < 6.0, f"the child's command was not killed ({elapsed:.1f}s)"
            captured = capsys.readouterr()
            text = captured.out + captured.err
        finally:
            reg.shutdown_all()
        assert "STILL RUNNING" in text or "kid" in text


    def test_stopping_a_background_child_leaves_the_shared_backend_usable(
            self, home, project, monkeypatch):
        """The parent and its background child share ONE backend; cancelling
        the child at the end of the run must not refuse the parent's own
        close-time reflection on that backend."""
        from localm.plugins.coder import background as bg
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.background import AgentJob, JobRegistry
        from localm.plugins.coder.backends.shared_engine import SharedEngineBackend
        from localm.plugins.coder.tools.agents import inherited_child_kwargs
        from unittest.mock import MagicMock
        reg = JobRegistry(kind_caps={"agent": 4})
        monkeypatch.setattr(bg, "_registry", reg)
        engine = MagicMock()
        engine.active_requests = 0
        engine.unloading = False
        engine.supports_grammar = False
        engine.count_messages_tokens.return_value = 1
        engine.count_tokens.return_value = 1
        child_replies = iter([_tc("run_shell", command=_SLEEP_CMD, timeout=60),
                              "child done"])

        def chat_stream(messages, **kw):
            if "slow" in str(messages):
                return iter([next(child_replies, "child done")])
            return iter(["parent done"])

        engine.chat_stream.side_effect = chat_stream
        backend = SharedEngineBackend(engine, "m")
        parent = _build(backend, project, "t", auto_approve=True)
        pm, audit, mem = _agent_patches()
        with pm as MockPM, audit, mem:
            MockPM.build.return_value.file_count.return_value = 0
            child = Agent(**inherited_child_kwargs(
                parent, backend=backend, cwd=project, name="kid",
                max_turns=3, confirm_handler=None))
        job = reg.submit(lambda: AgentJob(child, "slow", label="kid",
                                          owner=parent.job_owner), kind="agent")
        try:
            result = runner.run_single_task(parent, "t")
            assert result.success is True
            assert child.cancelled is True and parent.cancelled is False
            assert backend.cancelled is False, "a child cancel aborted the shared backend"
            # The parent can still generate on it (its reflection would).
            assert backend.chat([{"role": "user", "content": "reflect"}]) == "parent done"
            assert _wait_until(lambda: job.state != "running")
        finally:
            reg.shutdown_all()
