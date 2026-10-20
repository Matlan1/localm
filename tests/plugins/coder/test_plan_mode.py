# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan mode (agent/planmode.py) and the interaction tools (tools/interact.py):
the read-only gate in _execute_tool, plan approval, ask_user, the /plan REPL
command, and plan mode surviving a checkpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from localm.plugins.coder import display
from localm.plugins.coder.agent import Agent
from localm.plugins.coder.agent import constants as C
from localm.plugins.coder.agent.planmode import (
    INTERACTION_TOOLS, PLAN_MODE_TOOLS, plan_mode_refusal,
)
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import TOOL_REGISTRY, ToolResult
from localm.plugins.coder.tools.registry import ToolDef


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("auto_approve", True)
        kw.setdefault("self_verify", False)
        return Agent(_Stub(), cwd=cwd, **kw)


def _call(name, **args):
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _answers(monkeypatch, *lines):
    """Script console.input; each prompt pops the next line (an exception
    instance is raised instead)."""
    queue = list(lines)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(str(prompt))
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(display.console, "input", fake_input)
    return prompts


def _tc(name, **args):
    return ("<tool_call>\n" + json.dumps({"name": name, "args": args})
            + "\n</tool_call>")


# ---------------------------------------------------------------------------
#  The gate
# ---------------------------------------------------------------------------

def test_every_destructive_ask_network_or_executing_tool_is_refused_in_plan_mode():
    """Enumerated over the live registry, so a new tool in any of these
    categories is covered without editing this test."""
    executing = C._SHELL_EXEC_TOOLS | {"run_tests"}
    for name, td in TOOL_REGISTRY.items():
        risky = (td.destructive or td.ask_by_default
                 or name in C._NETWORK_TOOLS or name in executing
                 or name in C._BROWSER_TOOLS or name in C._PARENT_AGENT_TOOLS
                 or name == "generate_image")
        if risky:
            assert plan_mode_refusal(name, td) is not None, name


def test_every_plan_mode_tool_is_registered_and_none_is_risky():
    for name in PLAN_MODE_TOOLS:
        td = TOOL_REGISTRY.get(name)
        if name == "use_skill" and td is None:
            continue   # registered only when a skill is installed
        assert td is not None, name
        assert not td.destructive and not td.ask_by_default, name
        assert name not in C._NETWORK_TOOLS | C._SHELL_EXEC_TOOLS, name
        assert plan_mode_refusal(name, td) is None, name


def test_a_listed_tool_flagged_destructive_is_still_refused():
    td = ToolDef(name="read_file", fn=lambda cwd: None, description="",
                 params={}, destructive=True)
    assert plan_mode_refusal("read_file", td) is not None
    td = ToolDef(name="grep", fn=lambda cwd: None, description="",
                 params={}, untrusted_docs=True)
    assert plan_mode_refusal("grep", td) is not None


def test_plan_mode_refuses_a_write_under_auto_approve(tmp_path):
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    result = agent._execute_tool(_call("write_file", path="a.txt", content="x"),
                                 interactive=False)
    assert not (tmp_path / "a.txt").exists()
    assert not result.ok
    assert "plan mode is on" in result.output
    agent.set_plan_mode(False)
    result = agent._execute_tool(_call("write_file", path="a.txt", content="x"),
                                 interactive=False)
    assert result.ok, result.output
    assert (tmp_path / "a.txt").read_text() == "x"


def test_plan_mode_lets_a_read_run(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    result = agent._execute_tool(_call("read_file", path="a.txt"),
                                 interactive=False)
    assert result.ok, result.output
    assert "hello" in result.output


def test_patch_mode_and_dry_run_do_not_bypass_plan_mode(tmp_path):
    agent = _agent(tmp_path, patch_mode=True, dry_run=True)
    agent.set_plan_mode(True)
    result = agent._execute_tool(_call("write_file", path="a.txt", content="x"),
                                 interactive=False)
    assert not result.ok and "plan mode is on" in result.output
    assert agent.current_patch() == ""
    result = agent._execute_tool(_call("run_shell", command="echo hi"),
                                 interactive=False)
    assert not result.ok and "plan mode is on" in result.output


def test_a_dynamic_non_destructive_tool_is_refused_in_plan_mode(tmp_path):
    ran = []
    td = ToolDef(name="mcp_srv_lookup", description="", params={},
                 fn=lambda cwd, **kw: ran.append(1) or ToolResult.success("ok"))
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    with patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY",
                    {"mcp_srv_lookup": td}):
        result = agent._execute_tool(_call("mcp_srv_lookup"), interactive=False)
    assert not result.ok and "plan mode is on" in result.output
    assert ran == []


def test_a_child_of_a_plan_mode_parent_is_gated(tmp_path):
    parent = _agent(tmp_path)
    child = _agent(tmp_path, parent=parent)
    parent.set_plan_mode(True)
    result = child._execute_tool(_call("write_file", path="c.txt", content="x"),
                                 interactive=False)
    assert not result.ok and "plan mode is on" in result.output
    assert not (tmp_path / "c.txt").exists()


def test_an_unknown_tool_in_plan_mode_gets_the_unknown_tool_error(tmp_path):
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    result = agent._execute_tool(_call("no_such_tool"), interactive=False)
    assert "Unknown tool" in result.output


def test_model_supplied_hidden_args_are_overwritten(tmp_path, monkeypatch):
    """A model cannot pass its own _session/_interactive to an interaction
    tool: an unattended run stays unattended."""
    agent = _agent(tmp_path)
    result = agent._execute_tool(
        _call("ask_user", question="q?", _interactive=True, _session=None),
        interactive=False)
    assert not result.ok
    assert "unattended" in result.output


# ---------------------------------------------------------------------------
#  ask_user
# ---------------------------------------------------------------------------

def test_ask_user_maps_an_option_number_to_its_text(tmp_path, monkeypatch):
    prompts = _answers(monkeypatch, "2")
    agent = _agent(tmp_path)
    result = agent._execute_tool(
        _call("ask_user", question="Which cache?", options=["per user", "shared"]),
        interactive=True)
    assert result.ok
    assert result.output == "The user answered: shared"
    assert len(prompts) == 1


def test_ask_user_accepts_free_text_and_an_out_of_range_number(tmp_path, monkeypatch):
    _answers(monkeypatch, "  use redis  ", "7")
    agent = _agent(tmp_path)
    r1 = agent._execute_tool(
        _call("ask_user", question="Which cache?", options=["a", "b"]),
        interactive=True)
    r2 = agent._execute_tool(
        _call("ask_user", question="Which cache?", options=["a", "b"]),
        interactive=True)
    assert r1.output == "The user answered: use redis"
    assert r2.output == "The user answered: 7"


def test_ask_user_empty_answer_and_end_of_input(tmp_path, monkeypatch):
    _answers(monkeypatch, "", EOFError())
    agent = _agent(tmp_path)
    r1 = agent._execute_tool(_call("ask_user", question="q?"), interactive=True)
    r2 = agent._execute_tool(_call("ask_user", question="q?"), interactive=True)
    assert r1.ok and "no answer" in r1.output
    assert not r2.ok and "input ended" in r2.output


def test_ask_user_ctrl_c_interrupts_the_turn(tmp_path, monkeypatch):
    _answers(monkeypatch, KeyboardInterrupt())
    agent = _agent(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        agent._execute_tool(_call("ask_user", question="q?"), interactive=True)


def test_ask_user_caps_a_huge_answer(tmp_path, monkeypatch):
    _answers(monkeypatch, "x" * 10000)
    agent = _agent(tmp_path)
    result = agent._execute_tool(_call("ask_user", question="q?"), interactive=True)
    assert result.output == "The user answered: " + "x" * 4000


@pytest.mark.parametrize("args, fragment", [
    ({}, "non-empty 'question'"),
    ({"question": "   "}, "non-empty 'question'"),
    ({"question": "q?", "options": {"a": 1}}, "list of short answer strings"),
    ({"question": "q?", "options": [str(i) for i in range(10)]}, "at most 9"),
])
def test_ask_user_rejects_bad_arguments(tmp_path, monkeypatch, args, fragment):
    _answers(monkeypatch)
    agent = _agent(tmp_path)
    result = agent._execute_tool(_call("ask_user", **args), interactive=True)
    assert not result.ok
    assert fragment in result.output


def test_ask_user_without_a_terminal(tmp_path):
    gui = _agent(tmp_path, confirm_handler=lambda call: True)
    r = gui._execute_tool(_call("ask_user", question="q?"), interactive=False)
    assert not r.ok and "no way to collect a typed answer" in r.output
    unattended = _agent(tmp_path)
    r = unattended._execute_tool(_call("ask_user", question="q?"), interactive=False)
    assert not r.ok and "Decide yourself" in r.output


def test_ask_user_in_a_sub_agent_is_refused(tmp_path, monkeypatch):
    _answers(monkeypatch)
    child = _agent(tmp_path, parent=_agent(tmp_path))
    r = child._execute_tool(_call("ask_user", question="q?"), interactive=True)
    assert not r.ok and "sub-agent" in r.output


# ---------------------------------------------------------------------------
#  enter_plan_mode / exit_plan_mode
# ---------------------------------------------------------------------------

def test_enter_plan_mode_needs_an_approval_channel(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    r = agent._execute_tool(_call("enter_plan_mode"), interactive=False)
    assert not r.ok and not agent.plan_mode
    r = agent._execute_tool(_call("enter_plan_mode"), interactive=True)
    assert r.ok and agent.plan_mode
    r = agent._execute_tool(_call("enter_plan_mode"), interactive=True)
    assert r.ok and "already" in r.output


def test_exit_plan_mode_approve_leaves_plan_mode(tmp_path, monkeypatch):
    _answers(monkeypatch, "y")
    agent = _agent(tmp_path, auto_approve=False)
    agent.set_plan_mode(True)
    r = agent._execute_tool(_call("exit_plan_mode", plan="1. do x"), interactive=True)
    assert r.ok and "approved" in r.output
    assert not agent.plan_mode
    assert agent.auto_approve is False


def test_exit_plan_mode_revise_keeps_plan_mode_and_returns_feedback(tmp_path, monkeypatch):
    _answers(monkeypatch, "n: also update the docs", "split step 2", "n", "")
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    outs = [agent._execute_tool(_call("exit_plan_mode", plan="1. x"),
                                interactive=True).output for _ in range(4)]
    assert agent.plan_mode
    assert "feedback: also update the docs" in outs[0]
    assert "feedback: split step 2" in outs[1]
    assert "no reason" in outs[2] and "no reason" in outs[3]


def test_exit_plan_mode_autorun_switches_auto_approve_and_verify_for_one_run(
        tmp_path, monkeypatch):
    _answers(monkeypatch, "a")
    agent = _agent(tmp_path, auto_approve=False)
    agent.verify_cmd = None
    agent.set_plan_mode(True)
    with patch("localm.plugins.coder.verify.detect_verify_command",
               return_value="pytest -q"):
        r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"),
                                interactive=True)
    assert r.ok and "`pytest -q` exits 0" in r.output
    assert not agent.plan_mode
    assert agent.auto_approve is True and agent.verify_cmd == "pytest -q"
    agent._end_plan_autorun()
    assert agent.auto_approve is False and agent.verify_cmd is None


def test_autorun_without_a_detectable_check_says_so(tmp_path, monkeypatch):
    _answers(monkeypatch, "auto")
    agent = _agent(tmp_path, auto_approve=False)
    agent.set_plan_mode(True)
    with patch("localm.plugins.coder.verify.detect_verify_command",
               return_value=None):
        r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"),
                                interactive=True)
    assert "No verification command was found" in r.output
    assert agent.auto_approve is True


def test_autorun_ends_when_the_run_ends_even_on_an_error(tmp_path, monkeypatch):
    agent = _agent(tmp_path, auto_approve=False)
    agent._begin_plan_autorun()
    assert agent.auto_approve is True
    with patch.object(agent, "_call_llm", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            agent.chat("go")
    assert agent.auto_approve is False


def test_exit_plan_mode_through_a_gui_handler(tmp_path):
    seen = []
    answers = iter([False, True])

    def handler(call, agent=None):
        seen.append((call.name, call.args))
        return next(answers)

    agent = _agent(tmp_path, confirm_handler=handler)
    agent.set_plan_mode(True)
    r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"), interactive=False)
    assert r.ok and "did not approve" in r.output and agent.plan_mode
    r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"), interactive=False)
    assert r.ok and not agent.plan_mode
    assert seen == [("exit_plan_mode", {"plan": "1. x"})] * 2


def test_exit_plan_mode_unattended_and_misuse(tmp_path, monkeypatch):
    _answers(monkeypatch)
    agent = _agent(tmp_path)
    r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"), interactive=True)
    assert not r.ok and "plan mode, which is off" in r.output
    agent.set_plan_mode(True)
    r = agent._execute_tool(_call("exit_plan_mode", plan="  "), interactive=True)
    assert not r.ok and "non-empty" in r.output
    r = agent._execute_tool(_call("exit_plan_mode", plan="x" * 20001), interactive=True)
    assert not r.ok and "under 20000" in r.output
    r = agent._execute_tool(_call("exit_plan_mode", plan="1. x"), interactive=False)
    assert not r.ok and agent.plan_mode


def test_gui_always_allow_is_never_kept_for_a_plan_approval():
    import threading
    from localm.plugins.coder.sessions import CoderSession

    def fake(tool):
        pending = SimpleNamespace(tool=tool, approved=None,
                                  answered=threading.Event())
        return SimpleNamespace(_lock=threading.Lock(), _pending={"c": pending},
                               allowed_tools=set())

    plan = fake("exit_plan_mode")
    assert CoderSession.answer_confirm(plan, "c", True, always_allow=True)
    assert plan.allowed_tools == set()
    write = fake("write_file")
    assert CoderSession.answer_confirm(write, "c", True, always_allow=True)
    assert write.allowed_tools == {"write_file"}


# ---------------------------------------------------------------------------
#  The loop
# ---------------------------------------------------------------------------

def test_interaction_tools_run_alone_with_the_terminal(tmp_path, monkeypatch):
    """ask_user emitted alongside a read in one response still reaches the
    terminal (the parallel pool runs tools non-interactively)."""
    (tmp_path / "a.txt").write_text("hi\n")
    _answers(monkeypatch, "blue")
    agent = _agent(tmp_path)
    replies = iter([
        _tc("read_file", path="a.txt") + "\n" + _tc("ask_user", question="Colour?"),
        "All set.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(replies)):
        agent.chat("pick a colour")
    tool_msgs = " ".join(str(m.get("content")) for m in agent._messages)
    assert "The user answered: blue" in tool_msgs
    assert "ask_user" in INTERACTION_TOOLS


def test_plan_mode_note_reaches_the_model_but_not_the_review_task(tmp_path):
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    with patch.object(agent, "_call_llm", return_value="Here is what I found."):
        agent.chat("add caching")
    first_user = next(m for m in agent._messages if m["role"] == "user")
    assert "[plan mode]" in str(first_user["content"])
    assert "[plan mode]" not in agent._review_task
    agent.set_plan_mode(False)
    with patch.object(agent, "_call_llm", return_value="ok"):
        agent.chat("next")
    last_user = [m for m in agent._messages if m["role"] == "user"][-1]
    assert "[plan mode]" not in str(last_user["content"])


def test_a_plan_then_approval_then_write_in_one_task(tmp_path, monkeypatch):
    _answers(monkeypatch, "y")
    agent = _agent(tmp_path)
    agent.set_plan_mode(True)
    replies = iter([
        _tc("write_file", path="early.txt", content="x"),
        _tc("exit_plan_mode", plan="1. write late.txt"),
        _tc("write_file", path="late.txt", content="y"),
        "Done.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(replies)):
        agent.chat("do it")
    assert not (tmp_path / "early.txt").exists()
    assert (tmp_path / "late.txt").read_text() == "y"
    assert not agent.plan_mode


# ---------------------------------------------------------------------------
#  REPL and checkpoint
# ---------------------------------------------------------------------------

def test_repl_plan_command_toggles_and_sets(tmp_path):
    from localm.plugins.coder.cli.repl import _handle_command
    agent = _agent(tmp_path)
    _handle_command("/plan", agent)
    assert agent.plan_mode
    _handle_command("/plan", agent)
    assert not agent.plan_mode
    _handle_command("/plan on", agent)
    _handle_command("/plan on", agent)
    assert agent.plan_mode
    _handle_command("/plan sideways", agent)
    assert agent.plan_mode
    _handle_command("/plan off", agent)
    assert not agent.plan_mode


def test_repl_plan_mode_bypasses_goal_mode(tmp_path, monkeypatch):
    from localm.plugins.coder.cli import repl
    agent = _agent(tmp_path)
    agent.goal_cmd = "pytest"
    agent.set_plan_mode(True)
    inputs = iter(["plan the cache"])

    def fake_read():
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    calls = []
    monkeypatch.setattr(repl, "_read_multiline", fake_read)
    monkeypatch.setattr(repl, "_setup_readline", lambda a: None)
    monkeypatch.setattr(repl, "_run_goal_loop", lambda *a, **k: calls.append("goal"))
    monkeypatch.setattr(agent, "chat", lambda text: calls.append("chat"))
    repl._repl(agent)
    assert calls == ["chat"]


def test_plan_mode_survives_a_checkpoint(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"
    proj.mkdir()
    first = _agent(proj, mode=SessionMode.LOG)
    first.set_plan_mode(True)
    first._messages = [{"role": "user", "content": "plan it"}]
    assert first.save_checkpoint()
    second = _agent(proj, mode=SessionMode.LOG)
    data = second.load_checkpoint()
    assert data is not None and data.get("plan_mode") is True
    second.resume_checkpoint(data)
    assert second.plan_mode
    data["plan_mode"] = "yes"
    third = _agent(proj, mode=SessionMode.LOG)
    third.resume_checkpoint(data)
    assert not third.plan_mode
