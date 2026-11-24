# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model that writes its tool calls as plain JSON still gets its task done
unattended, and nothing it only quoted runs: it is asked once per reply whether
it meant the calls, a result it invents itself ends the reply, and a turn that
must be a call can only name a real tool with that tool's arguments."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.parser import (
    invented_result_at, looks_like_tool_attempt, names_a_tool_as_command,
    unfinished_call)

FENCE = "```"


class _Backend:
    """Answers the canned replies in order, and *check* whenever it is asked
    whether its calls were meant to run."""

    model_id = "stub-model"
    native_tools = False
    supports_grammar = True
    last_usage: dict = {}

    def __init__(self, replies, check="[run]"):
        self.replies = list(replies)
        self.check = check
        self.checks: list = []
        self.calls: list = []

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        if messages[-1]["content"].startswith("[tool-call check]"):
            self.checks.append([dict(m) for m in messages])
            self.check_kw = kw
            return self.check
        self.calls.append(([dict(m) for m in messages], kw))
        return self.replies.pop(0) if self.replies else "Done."

    def chat_stream(self, messages, **kw):
        kw.pop("on_reasoning", None)
        kw.pop("on_status", None)
        for piece in self.chat(messages, **kw):
            yield piece


def _agent(tmp_path: Path, backend, **kwargs):
    from localm.plugins.coder.agent import Agent
    kwargs.setdefault("self_verify", False)
    kwargs.setdefault("auto_approve", True)
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def _loose(name: str, **args) -> str:
    return (FENCE + "json\n" + json.dumps({"name": name, "args": args}) + "\n" + FENCE)


def _xml(tool: str, **args) -> str:
    return "<tool_call>\n" + json.dumps({"name": tool, "args": args}) + "\n</tool_call>"


def _texts(agent, role) -> list[str]:
    return [str(m["content"]) for m in agent._messages if m["role"] == role]


# --- asked once whether the calls were meant ------------------------------

def test_loose_calls_run_unattended_once_the_model_says_they_were_meant(tmp_path):
    first = "Writing both.\n" + _loose("write_file", path="a.txt", content="1") + "\n" \
        + _loose("write_file", path="b.txt", content="2")
    backend = _Backend([first, "Both written."])
    agent = _agent(tmp_path, backend)
    agent.run_task("create a.txt and b.txt")
    assert (tmp_path / "a.txt").read_text() == "1"
    assert (tmp_path / "b.txt").read_text() == "2"
    assert len(backend.checks) == 1
    question = backend.checks[0][-1]["content"]
    assert "- write_file" in question and "b.txt" in question
    assert agent.denied_unconfirmed == []
    assert agent.last_run_ok is True
    assert not any("[tool-call check]" in t for t in _texts(agent, "user"))


def test_a_call_the_model_calls_an_example_never_runs(tmp_path):
    reply = "A call looks like this:\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "unused"], check="[example]")
    agent = _agent(tmp_path, backend)
    result = agent.run_task("explain with an example how a tool call looks")
    assert not (tmp_path / "a.txt").exists()
    assert agent.denied_unconfirmed == []
    assert result.startswith(reply) and agent.last_run_ok is True
    assert len(backend.calls) == 1


def test_an_example_does_not_answer_a_request_for_a_change(tmp_path):
    reply = "Like this:\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, _xml("write_file", path="a.txt", content="1"), "Done."],
                       check="[example]")
    agent = _agent(tmp_path, backend)
    agent.run_task("create a.txt containing 1")
    assert any("you said it was an example" in t for t in _texts(agent, "user"))
    assert (tmp_path / "a.txt").read_text() == "1"


def test_an_example_beside_a_call_that_ran_does_not_end_the_run(tmp_path):
    (tmp_path / "b.txt").write_text("real")
    reply = _loose("read_file", path="b.txt") + "\nand like this:\n" \
        + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "b.txt holds real."], check="[example]")
    agent = _agent(tmp_path, backend)
    assert agent.run_task("what does b.txt say?").startswith("b.txt holds real.")
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize("check", ["", "sure, go ahead", "[run] [example] maybe"])
def test_an_answer_that_is_neither_denies_the_call(tmp_path, check):
    reply = "Writing.\n" + _loose("write_file", path="a.txt", content="1")
    agent = _agent(tmp_path, _Backend([reply, "Done."], check=check))
    agent.run_task("create a.txt")
    assert not (tmp_path / "a.txt").exists()
    assert agent.denied_unconfirmed == [("write_file", "lenient")]


def test_the_question_is_held_to_its_two_answers_by_a_grammar(tmp_path):
    from localm.plugins.coder.agent.context import _CALL_CHECK_GRAMMAR
    reply = "Writing.\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "Done."], check="<think>\nyes\n</think>\n[run]")
    _agent(tmp_path, backend, max_tokens=9, max_tokens_explicit=False).run_task("create a.txt")
    assert backend.check_kw["grammar"] == _CALL_CHECK_GRAMMAR
    assert "max_tokens" not in backend.check_kw
    assert (tmp_path / "a.txt").exists()


@pytest.mark.parametrize("check,ran", [("[run]", True), (" [run]\n", True),
                                       ("I think [run]", False), ("[example]", False),
                                       ("[run] write_file", False)])
def test_without_a_grammar_only_an_exact_answer_counts(tmp_path, check, ran):
    reply = "Writing.\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "Done."], check=check)
    backend.supports_grammar = False
    _agent(tmp_path, backend).run_task("create a.txt")
    assert "grammar" not in backend.check_kw
    assert (tmp_path / "a.txt").exists() is ran


def test_a_person_who_can_be_asked_is_asked_instead_of_the_model(tmp_path):
    reply = "Writing.\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "Not written."])
    asked = []

    def handler(call, agent=None):
        asked.append(call.name)
        return False

    agent = _agent(tmp_path, backend, confirm_handler=handler)
    agent.run_task("create a.txt")
    assert asked == ["write_file"]
    assert backend.checks == []
    assert not (tmp_path / "a.txt").exists()


def test_the_check_never_waives_another_reason_to_confirm(tmp_path):
    reply = "Writing.\n" + _loose("write_file", path="a.txt", content="1")
    backend = _Backend([reply, "Done."])
    agent = _agent(tmp_path, backend, always_confirm={"write_file"})
    agent.run_task("create a.txt")
    assert backend.checks == []
    assert not (tmp_path / "a.txt").exists()
    assert agent.denied_unconfirmed == [("write_file", "unconfirmable")]


def test_the_check_grammar_is_structurally_valid():
    from localm.inference.gbnf import check_grammar_structure
    from localm.plugins.coder.agent.context import _CALL_CHECK_GRAMMAR
    check_grammar_structure(_CALL_CHECK_GRAMMAR)


def test_a_denied_call_that_ran_later_is_not_reported_denied_when_it_failed(tmp_path):
    from localm.plugins.coder.parser import ToolCall
    agent = _agent(tmp_path, _Backend([]))
    args = {"path": "missing.txt", "old": "a", "new": "b"}
    loose = ToolCall(name="edit_file", args=dict(args), raw="", start=0, end=0, lenient=True)
    agent._execute_tool(loose, interactive=False)
    assert agent.denied_unconfirmed == [("edit_file", "lenient")]
    strict = ToolCall(name="edit_file", args=dict(args), raw="", start=0, end=0)
    assert agent._execute_tool(strict, interactive=False).ok is False
    assert agent.denied_unconfirmed == []


# --- a result the model invents ends the reply ----------------------------

INVENTED = ("Looking first.\n" + _loose("read_file", path="a.txt")
            + "\n\n<tool_result>\n" + FENCE + "\nwrong content\n" + FENCE + "\n</tool_result>\n\n"
            + _loose("write_file", path="b.txt", content="from the invented result"))


def test_a_call_after_an_invented_result_does_not_run(tmp_path):
    (tmp_path / "a.txt").write_text("real content")
    agent = _agent(tmp_path, _Backend([INVENTED, "a.txt holds: real content"]))
    agent.run_task("copy a.txt to b.txt")
    assert not (tmp_path / "b.txt").exists()
    fed_back = _texts(agent, "user")[1]
    assert "real content" in fed_back and "[invented result]" in fed_back
    assert "wrong content" not in "\n".join(_texts(agent, "assistant"))


def test_an_invented_result_never_makes_a_loose_call_trusted(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    reply = (_loose("write_file", path="b.txt", content="1")
             + "\n<tool_result>\nwritten\n</tool_result>")
    asked = []

    def handler(call, agent=None):
        asked.append((call.name, call.lenient))
        return False

    agent = _agent(tmp_path, _Backend([reply, "Not written."]), confirm_handler=handler)
    agent.run_task("create b.txt")
    assert asked == [("write_file", True)]


def test_a_result_tag_inside_a_call_is_content_not_an_invented_result(tmp_path):
    body = "line\n<tool_result>\nstill content\n</tool_result>\n"
    reply = _xml("write_file", path="doc.md", content=body)
    assert invented_result_at(reply, {"write_file"}) == -1
    half = '<tool_call>\n{"name": "write_file", "args": {"path": "d", "content": "a\n<tool_result>\nb'
    assert invented_result_at(_xml("read_file", path="x") + "\n" + half,
                              {"write_file", "read_file"}) == -1


@pytest.mark.parametrize("text", [
    "The harness answers in a <tool_result> block.",
    "<tool_result>\nrun_shell(command: \"pytest\")\n</tool_result>",
    _xml("read_file", path="x") + "\nThe answer arrives in a <tool_result> block.",
])
def test_a_result_tag_with_no_call_before_it_or_inside_a_line_is_text(text):
    assert invented_result_at(text, {"read_file", "run_shell"}) == -1


def test_generation_stops_at_the_invented_result(tmp_path):
    (tmp_path / "a.txt").write_text("real content")
    produced = []

    class Streaming(_Backend):
        def chat_stream(self, messages, **kw):
            reply = self.replies.pop(0) if self.replies else "Done."
            for ch in reply:
                produced.append(ch)
                yield ch

    agent = _agent(tmp_path, Streaming([INVENTED, "Done."]), on_event=lambda e: None)
    agent.run_task("copy a.txt to b.txt")
    assert "wrong content" not in "".join(produced)
    assert not (tmp_path / "b.txt").exists()


# --- a call that was cut off ----------------------------------------------

HALF_XML = '<tool_call>\n{"name": "write_file", "args": {"path": "big.py", "content": "x = 1\\ny'
HALF_COMMENTED = (FENCE + 'python\n# write it\n{"name": "write_file", "args": '
                  '{"path": "big.py", "content": "x = 1')
HALF_FENCE = FENCE + 'json\n{"name": "write_file", "args": {"path": "big.py", "content": "x = 1'


@pytest.mark.parametrize("text", [HALF_XML, "Now the file.\n" + HALF_XML, HALF_FENCE,
                                  HALF_COMMENTED,
                                  _xml("read_file", path="a") + "\n" + HALF_XML])
def test_a_call_the_reply_ends_in_is_unfinished(text):
    assert unfinished_call(text, {"write_file", "read_file"}) == "write_file"


@pytest.mark.parametrize("text", [
    _xml("write_file", path="a", content="b"),
    HALF_XML + '"}}\n</tool_call>',
    '<tool_call>\n{"name": "write_file", "args": {"path": "unquoted}}\n</tool_call>',
    FENCE + 'json\n{"name": "somebody", "args": {"x": "1',
    "plain text",
])
def test_a_complete_malformed_or_foreign_call_is_not_unfinished(text):
    assert unfinished_call(text, {"write_file", "read_file"}) == ""


def test_a_flood_of_openers_stays_cheap():
    import time
    text = '<tool_call>{"name": "write_file", "args": {"x": "' * 4000
    start = time.monotonic()
    unfinished_call(text, {"write_file"})
    invented_result_at(text, {"write_file"})
    assert time.monotonic() - start < 5


def test_a_cut_off_call_is_reported_as_cut_off_with_the_limit(tmp_path):
    backend = _Backend([HALF_XML, _xml("write_file", path="big.py", content="x = 1"), "Done."])
    agent = _agent(tmp_path, backend, max_tokens=100, max_tokens_explicit=False)
    agent.run_task("write big.py")
    note = [t for t in _texts(agent, "user") if "ended in the middle of the write_file call" in t]
    assert len(note) == 1
    assert "at most 100 tokens" in note[0] and "200 tokens from here on" in note[0]
    assert "must be valid JSON" not in note[0]
    assert [kw["max_tokens"] for _m, kw in backend.calls] == [100, 200, 200]
    assert (tmp_path / "big.py").exists()


def test_the_limit_is_raised_once_and_only_for_the_run(tmp_path):
    backend = _Backend([HALF_XML, HALF_XML, "I could not write it.", "Hello."])
    agent = _agent(tmp_path, backend, max_tokens=100, max_tokens_explicit=False)
    agent.run_task("write big.py")
    agent.run_task("say hello")
    assert [kw["max_tokens"] for _m, kw in backend.calls][:3] == [100, 200, 200]
    assert backend.calls[-1][1]["max_tokens"] == 100


def test_a_limit_the_user_set_is_not_raised(tmp_path):
    backend = _Backend([HALF_XML, "I could not write it."])
    agent = _agent(tmp_path, backend, max_tokens=100, max_tokens_explicit=True)
    agent.run_task("write big.py")
    assert {kw["max_tokens"] for _m, kw in backend.calls} == {100}


def test_a_cut_off_call_beside_a_complete_one_is_named(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    backend = _Backend([_xml("read_file", path="a.txt") + "\n" + HALF_XML, "Done."])
    agent = _agent(tmp_path, backend)
    agent.run_task("write big.py")
    assert any("ended in the middle of the write_file call" in t for t in _texts(agent, "user"))


def test_a_run_that_only_ever_cuts_off_ends_and_says_so(tmp_path):
    agent = _agent(tmp_path, _Backend([HALF_XML] * 10), max_turns=20)
    result = agent.run_task("write big.py")
    assert "each attempt ended before the call was complete" in result
    assert agent.last_run_ok is False
    assert agent.turns <= 4


# --- calls written as commands or Python ----------------------------------

@pytest.mark.parametrize("text", [
    FENCE + "\nrun_tests\n" + FENCE,
    FENCE + "sh\nwrite_file -args path=\"a\" content=\"b\"\n" + FENCE,
    FENCE + "python\n# change it\nedit_file(path=\"a\", old=\"b\", new=\"c\")\n" + FENCE,
    "Next:\n" + FENCE + "json\n",
])
def test_a_tool_written_as_a_command_is_an_attempt(text):
    assert names_a_tool_as_command(text, {"run_tests", "write_file", "edit_file"})


@pytest.mark.parametrize("text", [
    "I ran run_tests and they pass.",
    FENCE + "python\ndef run_tests_now():\n    pass\n" + FENCE,
    FENCE + "python\nresult = read_file(path)\n" + FENCE,
    FENCE + "json\n{\"a\": 1}\n" + FENCE,
    FENCE + "sh\ngrep -rn foo .\n" + FENCE,
    FENCE + "\ntree\n" + FENCE,
])
def test_other_code_is_not_an_attempt(text):
    names = {"run_tests", "write_file", "read_file", "grep", "tree"}
    assert not names_a_tool_as_command(text, names)
    assert not looks_like_tool_attempt(text, names)


def test_an_answer_that_shows_a_tool_as_a_command_is_not_a_failed_run(tmp_path):
    answer = "Use it like this:\n" + FENCE + "python\nread_file(path=\"a\")\n" + FENCE
    agent = _agent(tmp_path, _Backend([answer] * 8))
    assert "a tool" not in agent.run_task("how do I use read_file?").replace(answer, "")
    assert agent.last_run_ok is True


def test_a_second_broken_attempt_at_an_owed_change_is_forced(tmp_path):
    bad = FENCE + "sh\nwrite_file -args path=a.txt content=1\n" + FENCE
    backend = _Backend([bad, bad, _xml("write_file", path="a.txt", content="1"), "Done."])
    agent = _agent(tmp_path, backend)
    agent.run_task("create a.txt containing 1")
    forced = ["grammar_lazy" not in kw for _m, kw in backend.calls]
    assert forced == [False, False, True, False]
    assert (tmp_path / "a.txt").read_text() == "1"


def test_a_second_broken_attempt_at_a_question_is_not_forced(tmp_path):
    bad = FENCE + "sh\nread_file -args path=a.txt\n" + FENCE
    backend = _Backend([bad, bad, "That is how it looks."])
    agent = _agent(tmp_path, backend)
    agent.run_task("how would a read look?")
    assert all("grammar_lazy" in kw for _m, kw in backend.calls)


# --- a forced turn names a real tool with its arguments -------------------

def _forced_grammar(agent) -> str:
    agent._force_tool_grammar = True
    return agent._llm_kwargs()["grammar"]


def test_a_forced_turn_can_only_name_this_sessions_tools(tmp_path):
    from localm.inference.gbnf import check_grammar_structure
    agent = _agent(tmp_path, _Backend([]), disabled_tools=frozenset({"run_shell"}),
                   always_confirm={"run_tests"})
    grammar = _forced_grammar(agent)
    check_grammar_structure(grammar)
    assert grammar.startswith("root ::= opt-think")
    assert '\\"write_file\\"' in grammar and '\\"content\\"' in grammar
    for name in ("run_shell", "run_tests", "TOOL_NAME"):
        assert f'("\\"{name}\\"")' not in grammar
    assert '-arguments-kv ::= "\\"args\\""' in grammar
    assert '\\"arguments\\"' not in grammar


def test_a_forced_turn_falls_back_to_the_generic_call_grammar(tmp_path):
    from localm.inference.gbnf import TOOL_CALLS_AFTER_THINK
    from localm.inference.tool_calling import ToolsError
    agent = _agent(tmp_path, _Backend([]))
    with patch("localm.inference.tool_calling.tool_grammar", side_effect=ToolsError("no")):
        assert _forced_grammar(agent) == TOOL_CALLS_AFTER_THINK


def test_a_change_shown_twice_instead_of_made_is_forced(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    shown = "Here it is:\n" + FENCE + "python\nx = 2\n" + FENCE
    backend = _Backend([_xml("read_file", path="a.py"), shown, shown + " ",
                        _xml("write_file", path="a.py", content="x = 2\n"), "Done."])
    agent = _agent(tmp_path, backend)
    agent.run_task("change a.py so x is 2")
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assert ["grammar_lazy" not in kw for _m, kw in backend.calls].count(True) == 1


def test_a_code_block_beside_a_call_is_named_as_not_written(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    block = FENCE + "python\nx = 2\ny = 3\nz = 4\nw = 5\n" + FENCE
    mixed = "Changing it.\n" + block + "\nNow reading it.\n" + _xml("read_file", path="a.py")
    backend = _Backend([mixed, _xml("write_file", path="a.py", content="x = 2\n"),
                        "Done. It now is:\n" + block + "\n"])
    agent = _agent(tmp_path, backend)
    agent.run_task("change a.py so x is 2")
    notes = [t for t in _texts(agent, "user") if "The code block in your reply changed no file" in t]
    assert len(notes) == 1 and notes[0] == _texts(agent, "user")[1]


def test_a_code_block_beside_a_call_in_an_answer_is_left_alone(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    mixed = (FENCE + "python\nx = 2\ny = 3\nz = 4\nw = 5\n" + FENCE + "\n"
             + _xml("read_file", path="a.py"))
    agent = _agent(tmp_path, _Backend([mixed, "That is the idea."]))
    agent.run_task("how could a.py look with more variables?")
    assert "changed no file" not in "\n".join(_texts(agent, "user"))


def test_a_prose_refusal_is_never_forced_into_a_call(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    backend = _Backend([_xml("read_file", path="a.py"), "It already holds that.",
                        "Nothing needs to change here."])
    agent = _agent(tmp_path, backend)
    agent.run_task("change a.py so x is 1")
    assert all("grammar_lazy" in kw for _m, kw in backend.calls)


# --- progress while a call is being written -------------------------------

def _clock(monkeypatch, step):
    from localm.plugins.coder.agent import context
    now = [0.0]

    def monotonic():
        now[0] += step
        return now[0]
    monkeypatch.setattr(context.time, "monotonic", monotonic)


BIG = _xml("write_file", path="big.py", content="\n".join(f"x = {i}" for i in range(200)))


def test_a_long_call_reports_progress_as_status_events(tmp_path, monkeypatch):
    _clock(monkeypatch, 0.05)
    events = []
    agent = _agent(tmp_path, _Backend([BIG, "Done."]), on_event=events.append)
    agent.run_task("write big.py")
    said = [e["text"] for e in events if e["type"] == "status" and "write_file" in e["text"]]
    assert said and "big.py" in said[0] and "lines" in said[0]
    assert len(said) < 30


def test_a_long_call_reports_progress_to_the_progress_sink(tmp_path, monkeypatch):
    _clock(monkeypatch, 0.05)
    said = []
    agent = _agent(tmp_path, _Backend([BIG, "Done."]))
    agent.report_progress = True
    agent.progress_sink = said.append
    agent.run_task("write big.py")
    assert any("Preparing write_file" in t for t in said)


def test_a_short_call_reports_no_progress(tmp_path):
    said = []
    agent = _agent(tmp_path, _Backend([_xml("write_file", path="a.txt", content="1"), "Done."]))
    agent.report_progress = True
    agent.progress_sink = said.append
    agent.run_task("write a.txt")
    assert not any("Preparing" in t for t in said)


def test_a_stop_ends_a_call_that_is_still_being_written(tmp_path):
    produced = []

    class Streaming(_Backend):
        def chat_stream(self, messages, **kw):
            for i, ch in enumerate(BIG):
                produced.append(ch)
                if i == 300:
                    agent.request_stop()
                yield ch

    agent = _agent(tmp_path, Streaming([]), on_event=lambda e: None)
    agent.run_task("write big.py")
    assert len(produced) < 400
    assert not (tmp_path / "big.py").exists()


# --- how a run ends -------------------------------------------------------

def test_a_run_out_of_turns_ends_with_the_models_own_account(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    reads = [_xml("read_file", path="a.txt", offset=i) for i in range(1, 4)]
    backend = _Backend(reads + ["I read a.txt; the summary is not written."])
    agent = _agent(tmp_path, backend, max_turns=3)
    result = agent.run_task("summarise a.txt into b.txt")
    assert result.startswith("I read a.txt; the summary is not written.")
    assert "[max_turns=3 reached]" in result
    assert agent.last_run_ok is False
    assert _texts(agent, "assistant")[-1].startswith("I read a.txt;")


def test_a_call_in_the_closing_account_does_not_run(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    backend = _Backend([_xml("read_file", path="a.txt")]
                       + [_xml("write_file", path="late.txt", content="1")])
    agent = _agent(tmp_path, backend, max_turns=1)
    assert agent.run_task("read a.txt") == "[max_turns=1 reached]"
    assert not (tmp_path / "late.txt").exists()


def test_the_model_is_told_how_many_turns_are_left(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i}.txt").write_text(str(i))
    backend = _Backend([_xml("read_file", path=f"f{i}.txt") for i in range(30)] + ["Done."])
    agent = _agent(tmp_path, backend, max_turns=20, turn_budget=10)
    agent.run_task("read every file")
    notes = [t for t in _texts(agent, "user") if t.startswith("[turn budget]")]
    assert "You have used 11 turns" in notes[0]
    assert [t.split()[2] for t in notes[1:]] == ["5", "3"]


def test_a_task_that_ends_early_is_told_nothing_about_turns(tmp_path):
    agent = _agent(tmp_path, _Backend(["Hello."]), max_turns=20)
    agent.run_task("say hello")
    assert not any("[turn budget]" in t for t in _texts(agent, "user"))


def test_failed_verification_stands_in_the_saved_transcript(tmp_path):
    backend = _Backend([_xml("write_file", path="a.txt", content="1")]
                       + ["All done." + " detail" * 400] * 4)
    agent = _agent(tmp_path, backend, verify_cmd="exit 3")
    agent.verify_max_retries = 1
    with patch.object(agent, "_verify_refusal", return_value=None):
        result = agent.run_task("write a.txt")
    assert "[verification FAILED]" in result
    assert _texts(agent, "assistant")[-1].startswith("[verification FAILED]")
    md = agent._write_session_markdown().read_text(encoding="utf-8")
    assert "[verification FAILED]" in md


def test_a_circuit_breaker_stands_at_the_end_of_the_conversation(tmp_path):
    backend = _Backend([_xml("read_file", path=f"missing{i}.txt") for i in range(20)])
    agent = _agent(tmp_path, backend)
    result = agent.run_task("read the files")
    assert "circuit breaker" in result
    assert agent._messages[-1] == {"role": "assistant", "content": result}


def test_a_change_still_owed_after_an_early_ladder_is_asked_for_again(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    backend = _Backend(["I will look at it.", _xml("read_file", path="a.txt"),
                        "The fix is to write y.", _xml("write_file", path="a.txt", content="y"),
                        "Fixed."])
    agent = _agent(tmp_path, backend)
    agent.run_task("fix a.txt so it holds y")
    assert (tmp_path / "a.txt").read_text() == "y"


# --- failing tests are a result, not a broken tool ------------------------

def _run_tests_result(ok: bool):
    from localm.plugins.coder.tools import ToolResult
    if ok:
        return ToolResult.success("passed", summary="tests passed")
    return ToolResult(ok=False, output="1 failed", summary="tests failed (exit 1)")


def test_failing_tests_between_edits_do_not_trip_the_breaker(tmp_path):
    from localm.plugins.coder.agent import TOOL_REGISTRY
    replies = []
    for i in range(5):
        replies += [_xml("write_file", path="a.py", content=f"x = {i}"), _xml("run_tests")]
    agent = _agent(tmp_path, _Backend(replies + ["Still failing."]), max_turns=30)
    with patch.object(TOOL_REGISTRY["run_tests"], "fn",
                      lambda cwd, **kw: _run_tests_result(False)):
        result = agent.run_task("fix the failing test")
    assert "circuit breaker" not in result
    assert (tmp_path / "a.py").read_text() == "x = 4"


def test_failing_tests_with_nothing_changed_still_trip_the_breaker(tmp_path):
    from localm.plugins.coder.agent import TOOL_REGISTRY
    replies = [f"Try {i}: {'different words ' * i}\n" + _xml("run_tests", path=f"t{i}")
               for i in range(8)]
    agent = _agent(tmp_path, _Backend(replies), max_turns=30)
    with patch.object(TOOL_REGISTRY["run_tests"], "fn",
                      lambda cwd, **kw: _run_tests_result(False)):
        result = agent.run_task("fix the failing test")
    assert "circuit breaker: run_tests failed 4 times" in result


def test_a_denied_tool_says_it_stays_denied(tmp_path):
    backend = _Backend([_xml("run_shell", command="echo hi"), "Not possible here."])
    agent = _agent(tmp_path, backend, always_confirm={"run_shell"})
    agent.run_task("run echo hi")
    fed_back = "\n".join(_texts(agent, "user"))
    assert "run_shell is denied every time in this run" in fed_back


# --- the session record ---------------------------------------------------

def test_the_session_record_names_files_a_command_changed(tmp_path):
    agent = _agent(tmp_path, _Backend([]))
    states = [{}, {"shell.txt": "??", "calc/__pycache__/": "??", "kept.py": " M"}]
    with patch.object(agent, "_git_status_map", side_effect=lambda: states.pop(0)):
        agent._shell_baseline_captured = True
        agent._git_baseline = agent._git_status_paths()
        agent._changed_files["kept.py"] = {"original": b"", "writes": 1, "last_tool": "write_file"}
        footer = agent._grounding_footer()
    assert "1 file(s) changed: kept.py" in footer
    assert "1 file(s) changed by commands: shell.txt" in footer
    assert "__pycache__" not in footer


def test_the_session_record_says_when_command_changes_are_unknown(tmp_path):
    from localm.plugins.coder.parser import ToolCall
    agent = _agent(tmp_path, _Backend([]))
    call = ToolCall(name="run_shell", args={"command": "echo hi > shell.txt"},
                    raw="", start=0, end=0)
    with patch.object(agent, "_git_status_map", return_value=None):
        assert agent._execute_tool(call, interactive=False).ok
        footer = agent._grounding_footer()
    assert "1 command(s) ran whose file changes are not recorded" in footer
    assert "no files changed" not in footer


def test_a_read_only_command_leaves_the_record_at_no_files_changed(tmp_path):
    agent = _agent(tmp_path, _Backend([]))
    assert agent._grounding_footer() == "\n\n[session record: no files changed]"


def test_the_session_record_counts_the_files_read(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    backend = _Backend([_xml("read_file", path="a.txt"), _xml("read_file", path="a.txt"),
                        _xml("read_file", path="missing.txt"), "Read 40 files."])
    agent = _agent(tmp_path, backend)
    assert "1 file(s) read]" in agent.run_task("what does a.txt say?")


# --- patch mode, dry run, verbose -----------------------------------------

BAD_HUNK = ("--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n nowhere in the file\n-old\n+new\n")


def test_a_hunk_that_does_not_apply_never_enters_the_patch(tmp_path):
    from localm.plugins.coder.parser import ToolCall
    (tmp_path / "f.py").write_text("x = 1\n")
    agent = _agent(tmp_path, _Backend([]), patch_mode=True)
    bad = ToolCall(name="patch_file", args={"path": "f.py", "diff": BAD_HUNK},
                   raw="", start=0, end=0)
    result = agent._execute_tool(bad, interactive=False)
    assert result.ok is False
    assert "was not captured" in result.output and "could not be applied" in result.output
    good = ToolCall(name="edit_file", args={"path": "f.py", "old": "x = 1", "new": "x = 2"},
                    raw="", start=0, end=0)
    assert agent._execute_tool(good, interactive=False).ok
    patch_text = agent.current_patch()
    assert "nowhere in the file" not in patch_text and "+x = 2" in patch_text


def test_a_refused_edit_batch_names_its_reason_in_patch_mode(tmp_path):
    from localm.plugins.coder.parser import ToolCall
    (tmp_path / "f.py").write_text("x = 1\n")
    agent = _agent(tmp_path, _Backend([]), patch_mode=True)
    call = ToolCall(name="edit_files", args={"edits": [
        {"path": "f.py", "old": "not there", "new": "y"}]}, raw="", start=0, end=0)
    result = agent._execute_tool(call, interactive=False)
    assert result.ok is False and "was not captured:" in result.output
    assert agent.has_patch() is False


@pytest.mark.parametrize("kwargs,call,expected", [
    ({"dry_run": True}, ("write_file", {"path": "a", "content": "b"}), "[dry-run] skipped"),
    ({"patch_mode": True}, ("write_file", {"path": "a", "content": "b"}),
     "[patch-mode] diff captured"),
    ({"patch_mode": True}, ("edit_file", {"path": "a", "old": "x", "new": "y"}),
     "[patch-mode] skipped"),
    ({"patch_mode": True}, ("generate_image", {"prompt": "p"}), "[patch-mode] skipped"),
])
def test_the_mode_tag_is_printed(tmp_path, kwargs, call, expected):
    from rich.console import Console
    import io

    from localm.plugins.coder.agent import execution
    from localm.plugins.coder.parser import ToolCall
    out = io.StringIO()
    agent = _agent(tmp_path, _Backend([]), **kwargs)
    with patch.object(execution, "console", Console(file=out, width=200)):
        agent._execute_tool(ToolCall(name=call[0], args=dict(call[1]), raw="", start=0, end=0),
                            interactive=True)
    assert expected in out.getvalue()


def _one_shot(tmp_path, capsys, *, verbose, sink=None):
    (tmp_path / "a.txt").write_text("\n".join(f"content {i}" for i in range(60)))
    agent = _agent(tmp_path, _Backend([_xml("read_file", path="a.txt"), "Read."]),
                   verbose=verbose)
    agent.report_progress = True
    agent.progress_sink = sink
    capsys.readouterr()
    agent.run_task("what does a.txt say?")
    return capsys.readouterr()


def test_a_verbose_one_shot_prints_tool_output_on_stderr(tmp_path, capsys):
    got = _one_shot(tmp_path, capsys, verbose=True)
    assert "content 3" in got.err and "more lines)" in got.err
    assert "content 3" not in got.out


def test_a_plain_one_shot_prints_no_tool_output(tmp_path, capsys):
    assert "content 3" not in _one_shot(tmp_path, capsys, verbose=False).err


def test_a_progress_sink_gets_no_tool_output(tmp_path, capsys):
    said = []
    got = _one_shot(tmp_path, capsys, verbose=True, sink=said.append)
    assert "content 3" not in got.err and not any("content 3" in t for t in said)


def test_the_stub_backend_is_not_hiding_a_mock(tmp_path):
    assert not isinstance(_agent(tmp_path, _Backend([])).backend, MagicMock)
