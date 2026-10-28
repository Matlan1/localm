# SPDX-License-Identifier: AGPL-3.0-or-later
"""``max_turns`` caps the turns of ONE task, not of the whole session.

A real Agent runs against a scripted backend that answers each LLM call with
canned text: a successful ``read_file`` tool call, or a final answer. Several
tasks on one Agent (the REPL keeps one Agent for the whole session) may use
more than ``max_turns`` turns in total; each new task still gets its own
``max_turns`` turns, while ``agent.turns`` keeps counting the whole session.
"""

from __future__ import annotations

import json

import pytest

from localm.plugins.coder.audit import SessionMode
from tests.conftest import final_answer as _final_answer


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "home" / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "home" / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "home" / "registry.json")
    return tmp_path / "home"


class _Scripted:
    """One canned response per LLM call, in order; the last one repeats."""

    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 5}

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def context_capacity(self):
        return 32768

    def chat(self, messages, **kw):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r

    def chat_stream(self, messages, **kw):
        yield self.chat(messages, **kw)

    def set_tools(self, tool_defs):
        pass


def _read(path: str) -> str:
    return "<tool_call>" + json.dumps(
        {"name": "read_file", "args": {"path": path}}) + "</tool_call>"


def _workspace(tmp_path, count):
    work = tmp_path / "work"
    work.mkdir()
    for i in range(count):
        (work / f"f{i}.txt").write_text(f"line {i}\n", encoding="utf-8")
    return work


def _agent(work, backend, max_turns):
    from localm.plugins.coder.agent import Agent
    return Agent(backend, cwd=work, auto_approve=True, self_verify=False,
                 max_turns=max_turns, mode=SessionMode.LOG)


def test_new_task_runs_after_session_used_more_than_max_turns(home, tmp_path):
    """REPL-style: three chat() tasks of two turns each (six turns in total)
    on an Agent with max_turns=3, then a fourth task. Every task reaches the
    model and finishes with its own answer."""
    work = _workspace(tmp_path, 4)
    script = []
    for i in range(4):
        script += [_read(f"f{i}.txt"), f"Answer number {i}: f{i}.txt holds line {i}."]
    backend = _Scripted(script)
    agent = _agent(work, backend, max_turns=3)

    for i in range(3):
        reply = agent.chat(f"What does f{i}.txt say?")
        assert _final_answer(reply) == f"Answer number {i}: f{i}.txt holds line {i}."
    assert agent.turns == 6
    assert backend.calls == 6

    reply = agent.chat("What does f3.txt say?")

    assert "max_turns" not in reply, (
        "a new task was refused because earlier tasks in the session used up "
        "max_turns between them")
    assert backend.calls == 8
    assert _final_answer(reply) == "Answer number 3: f3.txt holds line 3."
    assert agent.last_run_ok is True
    assert agent.turns == 8


def test_single_task_still_stops_at_max_turns(home, tmp_path):
    """One task that never gives a final answer stops after exactly
    max_turns LLM calls."""
    work = _workspace(tmp_path, 10)
    backend = _Scripted([_read(f"f{i}.txt") for i in range(10)])
    agent = _agent(work, backend, max_turns=3)

    reply = agent.run_task("Read every file.")

    assert reply == "[max_turns=3 reached]"
    assert backend.calls == 3
    assert agent.turns == 3
    assert agent.last_run_ok is False


def test_task_after_earlier_tasks_still_stops_at_its_own_max_turns(home, tmp_path):
    """A task that starts after earlier tasks used turns gets exactly
    max_turns turns of its own, no more and no fewer."""
    work = _workspace(tmp_path, 10)
    script = [_read("f0.txt"), "First answer: line 0."]
    script += [_read(f"f{i}.txt") for i in range(1, 10)]
    backend = _Scripted(script)
    agent = _agent(work, backend, max_turns=3)

    assert _final_answer(agent.chat("What does f0.txt say?")) == "First answer: line 0."
    assert backend.calls == 2

    reply = agent.chat("Read every file.")

    assert reply == "[max_turns=3 reached]"
    assert backend.calls == 5
    assert agent.turns == 5


def test_resumed_session_with_many_turns_still_runs_a_task(home, tmp_path):
    """A checkpoint restored with a session turn count above max_turns (the
    REPL /resume path) still lets the next task reach the model, and the
    restored count keeps growing from there."""
    work = _workspace(tmp_path, 1)
    backend = _Scripted([_read("f0.txt"), "Resumed answer: line 0."])
    agent = _agent(work, backend, max_turns=3)
    agent.resume_checkpoint({
        "messages": [{"role": "user", "content": "earlier task"},
                     {"role": "assistant", "content": "earlier answer"}],
        "turns": 50,
        "total_tokens": 0,
    })
    assert agent.turns == 50

    reply = agent.chat("Continue from where we left off.", record_request=False)

    assert _final_answer(reply) == "Resumed answer: line 0."
    assert backend.calls == 2
    assert agent.turns == 52


def test_reset_starts_the_next_task_from_zero(home, tmp_path):
    """/clear zeroes the session counter and the task start with it, so the
    next task gets max_turns turns and the count restarts at its own turns."""
    work = _workspace(tmp_path, 10)
    script = [_read("f0.txt"), "First answer: line 0."]
    script += [_read(f"f{i}.txt") for i in range(1, 10)]
    backend = _Scripted(script)
    agent = _agent(work, backend, max_turns=3)
    agent.chat("What does f0.txt say?")

    agent.reset()
    reply = agent.run_task("Read every file.")

    assert reply == "[max_turns=3 reached]"
    assert backend.calls == 5
    assert agent.turns == 3
