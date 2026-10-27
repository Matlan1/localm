# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stop pressed on the root session ends a foreground sub-agent at its next
turn boundary: no further model turn and no further file edit, not only killed
shell commands. A detached (background) sub-agent keeps following only its own
stop."""

from __future__ import annotations

import json
from unittest.mock import patch

from localm.plugins.coder import runner
from localm.plugins.coder.agent import Agent
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.tools.agents import inherited_child_kwargs


class _Scripted:
    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage: dict = {"total_tokens": 5}
    last_reasoning = ""

    def __init__(self, responses, on_call=None):
        self.responses = list(responses)
        self.calls = 0
        self.on_call = on_call

    def chat(self, messages, **kw):
        n = self.calls
        self.calls += 1
        if self.on_call is not None:
            self.on_call(n)
        return self.responses[min(n, len(self.responses) - 1)]

    def chat_stream(self, messages, on_reasoning=None, **kw):
        yield self.chat(messages, **kw)

    def set_tools(self, defs):
        pass

    def context_capacity(self):
        return None


def _tc(tool: str, /, **args) -> str:
    return "<tool_call>" + json.dumps({"name": tool, "args": args}) + "</tool_call>"


def _build(backend, project, max_turns=8):
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return runner.build_agent(
            backend, project, task="t", max_turns=max_turns, auto_approve=True,
            always_confirm=(), session_mode=SessionMode.LOG)


def _child_of(parent, project):
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(**inherited_child_kwargs(
            parent, backend=parent.backend, cwd=project, name="kid",
            max_turns=3, confirm_handler=None))


def test_stop_during_the_childs_generation_runs_none_of_its_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    holder: dict = {}

    def stop_on_child_turn(n):
        if n == 1:
            holder["parent"].request_stop()

    backend = _Scripted([
        _tc("spawn_agent", task="write the file", name="kid", max_turns=5),
        _tc("write_file", path="child.txt", content="x"),
        "child finished",
    ], on_call=stop_on_child_turn)
    parent = _build(backend, project)
    holder["parent"] = parent

    parent.run_task("t")

    assert not (project / "child.txt").exists(), "the child edited a file after Stop"
    assert backend.calls == 2, f"a model turn ran after Stop ({backend.calls})"
    assert parent.cancelled is False
    assert parent._stop_requested is False, "the root did not consume the Stop"


def test_stop_between_the_childs_turns_takes_no_further_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    holder: dict = {}
    backend = _Scripted([
        _tc("spawn_agent", task="two edits", name="kid", max_turns=5),
        _tc("write_file", path="a.txt", content="a"),
        _tc("write_file", path="b.txt", content="b"),
        "child finished",
    ])
    parent = _build(backend, project)
    holder["parent"] = parent
    real = Agent._execute_tool

    def stop_after_first_child_edit(self, call, interactive):
        out = real(self, call, interactive)
        if self.parent is not None and call.name == "write_file":
            holder["parent"].request_stop()
        return out

    with patch.object(Agent, "_execute_tool", stop_after_first_child_edit):
        parent.run_task("t")

    assert (project / "a.txt").exists()
    assert not (project / "b.txt").exists(), "the child kept editing after Stop"
    assert backend.calls == 2, f"the child took another turn after Stop ({backend.calls})"


def test_a_stopped_run_does_not_run_the_rest_of_a_tool_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    agent = _build(_Scripted(["x"]), project)
    agent.request_stop()
    from localm.plugins.coder.parser import ToolCall
    res = agent._execute_tool(
        ToolCall(name="write_file", args={"path": "late.txt", "content": "y"},
                 raw="", start=0, end=0),
        interactive=False)
    assert res.ok is False
    assert not (project / "late.txt").exists()


def test_stop_pending_follows_an_ancestor_only_while_attached(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    parent = _build(_Scripted(["x"]), project)
    child = _child_of(parent, project)
    grandchild = _child_of(child, project)
    assert not child.stop_pending and not grandchild.stop_pending
    parent.request_stop()
    assert child.stop_pending and grandchild.stop_pending
    assert child._cancel_check() and grandchild._cancel_check()
    child.detached = True
    assert not child.stop_pending, "a background child follows only its own stop"
    assert not grandchild.stop_pending
    child.request_stop()
    assert child.stop_pending and grandchild.stop_pending
