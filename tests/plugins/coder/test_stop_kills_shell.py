# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stop (the GUI session's Stop button) kills a running shell command.

The session's stop() asks the agent to stop at its next safe point; a shell
command in flight must be killed then, not waited out to its timeout.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from unittest.mock import patch

import psutil

from localm.plugins.coder import runner
from localm.plugins.coder.audit import SessionMode


class _Scripted:
    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage: dict = {"total_tokens": 5}
    last_reasoning = ""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kw):
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


def _build(backend, project):
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return runner.build_agent(
            backend, project, task="t", max_turns=5, auto_approve=True,
            always_confirm=(), session_mode=SessionMode.LOG)


def _wait_until(pred, timeout=15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def _sleeper_cmd(pid_file) -> str:
    code = (f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "time.sleep(60)")
    return f'"{sys.executable}" -c "{code}"'


def test_stop_kills_the_running_shell_command(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    pid_file = tmp_path / "sleeper.pid"
    late = project / "late.txt"
    backend = _Scripted([_tc("run_shell", command=_sleeper_cmd(pid_file), timeout=60),
                         _tc("write_file", path="late.txt", content="too late"),
                         "finished"])
    agent = _build(backend, project)
    done: list = []
    worker = threading.Thread(
        target=lambda: done.append(agent.run_task("t")), daemon=True)
    worker.start()
    assert _wait_until(lambda: pid_file.exists() and pid_file.read_text().strip()), \
        "the sleeping command never started"
    pid = int(pid_file.read_text().strip())
    assert psutil.pid_exists(pid)

    t0 = time.monotonic()
    agent.request_stop()
    worker.join(15)
    elapsed = time.monotonic() - t0

    assert not worker.is_alive(), "the run was still busy after Stop"
    assert elapsed < 10.0, f"Stop waited for the command ({elapsed:.1f}s)"
    assert _wait_until(lambda: not psutil.pid_exists(pid), timeout=5), \
        "the shell's child process survived Stop"
    assert not late.exists(), "a later tool call ran after Stop"
    assert backend.calls == 1, backend.calls
    assert agent.cancelled is False, "Stop must not cancel the agent for good"


def test_stop_on_a_parent_reaches_a_child_agents_command(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    parent = _build(_Scripted(["x"]), project)
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.tools.agents import inherited_child_kwargs
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        child = Agent(**inherited_child_kwargs(
            parent, backend=parent.backend, cwd=project, name="kid",
            max_turns=3, confirm_handler=None))
    assert child._cancel_check() is False
    parent.request_stop()
    assert child._cancel_check() is True
    assert child.cancelled is False


def test_a_later_run_is_not_killed_by_an_earlier_stop(tmp_path, monkeypatch):
    """run_task clears a stale stop, so the next task's commands run."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    cmd = f'"{sys.executable}" -c "open(\'ran.txt\', \'w\').write(\'y\')"'
    backend = _Scripted([_tc("run_shell", command=cmd, timeout=30), "done"])
    agent = _build(backend, project)
    agent.request_stop()
    agent.run_task("t")
    assert (project / "ran.txt").read_text() == "y"
