# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder loop must not end a task as finished, or accept prose as the
answer, when no tool ran or the requested work did not happen; and it must not
nag a person who refused, or a dry-run session whose writes are never made."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.parser import looks_like_tool_attempt


def _make_agent(tmp_path: Path, **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    backend.supports_grammar = True
    kwargs.setdefault("self_verify", False)
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _call(name: str, args: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "args": args}) + "\n</tool_call>"


def _scripted(agent, replies, seen=None):
    it = iter(replies)

    def reply(*_a, **_k):
        if seen is not None:
            seen.append(bool(getattr(agent, "_force_tool_grammar", False)))
        return next(it)
    return patch.object(agent, "_call_llm", side_effect=reply)


def _user_texts(agent) -> list[str]:
    return [str(m.get("content", "")) for m in agent._messages if m["role"] == "user"]


# --- a claim with no tool run (CODC-036-2) --------------------------------

def test_the_escape_does_not_answer_a_request_to_run_the_tests(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8)
    claim = "[no tool needed] All tests have passed successfully."
    with _scripted(agent, [claim] * 6):
        result = agent.run_task("Run the tests with run_tests and tell me what failed.")
    assert agent.last_run_ok is False
    assert "[tool use not achieved" in result
    rung1 = [t for t in _user_texts(agent) if t.startswith("[no tool call]")]
    assert rung1 and "[no tool needed]" not in rung1[0]


def test_the_escape_still_answers_a_plain_question(tmp_path):
    agent = _make_agent(tmp_path, max_turns=6)
    with _scripted(agent, ["Here is the answer in prose, no file involved."]):
        agent.run_task("why is a list not a tuple?")
    assert agent.last_run_ok is True


def test_a_tool_written_as_prose_is_an_attempt_not_an_answer():
    names = {"run_shell", "read_file"}
    assert looks_like_tool_attempt('run_shell({"command": "pytest"})', names)
    assert looks_like_tool_attempt("read_file({...})", names)
    assert not looks_like_tool_attempt("I would call run_shell to do it.", names)


def test_prose_tool_call_ends_the_run_not_finished(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, ['run_shell({"command": "pytest"})'] * 6):
        result = agent.run_task("Run the tests with run_tests and tell me what failed.")
    assert result.startswith("[I tried to call a tool")
    assert agent.last_run_ok is False


# --- plan mode (CODB-058, CODB-058-2) -------------------------------------

def test_plan_mode_prose_plan_is_sent_back_for_exit_plan_mode(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8, interactive_tools=True)
    agent.set_plan_mode(True)
    seen: list[bool] = []
    with _scripted(agent, ["Plan: change add to use plus."] * 3, seen), \
         patch.object(agent, "can_force_tool_calls", return_value=True):
        result = agent.run_task("Change add to use plus.")
    nudges = [t for t in _user_texts(agent) if t.startswith("[plan not submitted]")]
    assert len(nudges) == 2
    assert seen == [False, False, True]
    assert "[not finished: plan mode needs the plan submitted" in result
    assert agent.last_run_ok is False


def test_plan_mode_marker_echoing_the_harness_is_not_accepted(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8, interactive_tools=True)
    agent.set_plan_mode(True)
    echo = "[no tool needed] [no tool call] That request needs you to USE a tool"
    with _scripted(agent, [echo] * 3):
        result = agent.run_task("Change add to use plus.")
    assert "[plan not submitted]" in " ".join(_user_texts(agent))
    assert "[not finished: plan mode" in result


def test_plan_mode_exit_plan_mode_call_settles_it(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8, interactive_tools=True)
    agent.set_plan_mode(True)
    with _scripted(agent, [_call("exit_plan_mode", {"plan": "1. edit add"}),
                           "The plan was not approved."]):
        result = agent.run_task("Change add to use plus.")
    assert not any(t.startswith("[plan not submitted]") for t in _user_texts(agent))
    assert "[not finished: plan mode" not in result


def test_harness_tag_echo_detection():
    from localm.plugins.coder.agent.loop import _echoes_harness_prompt
    assert _echoes_harness_prompt("[no tool call] That request needs")
    assert _echoes_harness_prompt("  [Nothing written] x")
    assert not _echoes_harness_prompt("All done.")


# --- a refusal and a dry run stand (CODB-002, CODB-007) -------------------

def test_a_refused_write_is_not_asked_again(tmp_path):
    asked: list[str] = []

    def handler(call, agent=None):
        asked.append(call.name)
        return False

    agent = _make_agent(tmp_path, max_turns=8, auto_approve=False, confirm_handler=handler)
    with _scripted(agent, [_call("write_file", {"path": "hello2.txt", "content": "no"}),
                           "Understood, I did not write it."]):
        result = agent.run_task("Use the write_file tool to create hello2.txt containing exactly: no")
    assert asked == ["write_file"]
    assert not any(t.startswith("[nothing written]") for t in _user_texts(agent))
    assert "[not finished" not in result
    assert not (tmp_path / "hello2.txt").exists()


def test_a_dry_run_write_is_not_asked_for_again(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8, dry_run=True, auto_approve=True)
    with _scripted(agent, [_call("write_file", {"path": "dry.txt", "content": "x"}),
                           "The write was skipped, as this is a dry run."]):
        result = agent.run_task("Use write_file to create dry.txt containing x.")
    texts = _user_texts(agent)
    assert not any(t.startswith("[nothing written]") for t in texts)
    assert any("Do not repeat it" in t for t in texts)
    assert "[not finished" not in result
    assert not (tmp_path / "dry.txt").exists()


# --- a loose call in an unattended run (CFG-074, CODA-014, CODA-077) ------

def test_a_lenient_denial_binds_the_next_turn_to_the_tool_call_format(tmp_path):
    agent = _make_agent(tmp_path, max_turns=8, auto_approve=True)
    loose = ('Creating it now.\n```json\n{"name": "write_file", "args": '
             '{"path": "hello.txt", "content": "hi"}}\n```')
    seen: list[bool] = []
    with _scripted(agent, [loose, _call("write_file", {"path": "hello.txt", "content": "hi"}),
                           "Created hello.txt."], seen), \
         patch.object(agent, "can_force_tool_calls", return_value=True):
        agent.run_task("create hello.txt containing hi")
    assert seen == [False, True, False]
    note = [t for t in _user_texts(agent) if "[tool-call format] These calls did NOT run" in t]
    assert note and '- write_file {"content": "hi", "path": "hello.txt"}' in note[0]
    assert (tmp_path / "hello.txt").read_text() == "hi"
    assert agent.last_run_ok is True


def test_a_lenient_ask_says_why_on_screen(tmp_path):
    events: list[dict] = []
    agent = _make_agent(tmp_path, max_turns=4, auto_approve=True,
                        confirm_handler=lambda call, agent=None: False)
    agent.on_event = events.append
    loose = ('Creating it now.\n```json\n{"name": "write_file", "args": '
             '{"path": "a.txt", "content": "hi"}}\n```')
    with _scripted(agent, [loose, "Not written."]):
        agent.run_task("create a.txt containing hi")
    infos = [e.get("text", "") for e in events if e["type"] == "info"]
    assert any("not written in the <tool_call> format" in t and "even with auto-approve on" in t
               for t in infos), infos
