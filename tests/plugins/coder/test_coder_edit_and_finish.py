# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder model that writes code into its reply, sends edit arguments under
other names, breaks a file's syntax with an edit, or stops right after saying
what it will do next, must still end up editing files through the tools, and
must not be reported as finished when it did not."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.argnorm import bad_args_message, normalize_args
from localm.plugins.coder.agent.loop import _announced_step, _has_file_code_block
from localm.plugins.coder.parser import parse_tool_calls
from localm.plugins.coder.tools import TOOL_REGISTRY
from localm.plugins.coder.tools.files import (
    tool_edit_file, tool_edit_files, tool_patch_file)
from tests.conftest import final_answer as _final_answer


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
    import json
    return "<tool_call>\n" + json.dumps({"name": name, "args": args}) + "\n</tool_call>"


def _scripted(agent, replies):
    it = iter(replies)
    return patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(it))


def _user_texts(agent) -> list[str]:
    return [str(m.get("content", "")) for m in agent._messages if m["role"] == "user"]


_METHOD_FILE = (
    "class Game:\n"
    "    def move(self, move):\n"
    "        return move\n"
)


# --------------------------------------------------------------------------- #
#  Arguments                                                                    #
# --------------------------------------------------------------------------- #

def test_arguments_written_beside_args_are_kept():
    text = ('<tool_call>\n{"name": "edit_file", "args": {"path": "a.py"}, '
            '"old_str": "x = 1", "new": "x = 2"}\n</tool_call>')
    (call,) = parse_tool_calls(text)
    assert call.args == {"path": "a.py"}
    assert call.beside == {"old_str": "x = 1", "new": "x = 2"}
    args, _ = normalize_args(TOOL_REGISTRY["edit_file"], call.args, call.beside)
    assert args == {"path": "a.py", "old": "x = 1", "new": "x = 2"}


def test_a_key_inside_args_wins_over_the_same_key_beside_it():
    text = ('<tool_call>\n{"name": "read_file", "args": {"path": "in.py"}, '
            '"path": "out.py"}\n</tool_call>')
    (call,) = parse_tool_calls(text)
    args, _ = normalize_args(TOOL_REGISTRY["read_file"], call.args, call.beside)
    assert args == {"path": "in.py"}


def test_a_key_beside_args_the_tool_does_not_take_is_dropped():
    text = ('<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}, '
            '"reason": "look at it"}\n</tool_call>')
    (call,) = parse_tool_calls(text)
    args, _ = normalize_args(TOOL_REGISTRY["read_file"], call.args, call.beside)
    assert args == {"path": "a.py"}
    assert bad_args_message("read_file", TOOL_REGISTRY["read_file"], args) is None


def test_the_agent_runs_an_edit_written_beside_args(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=4)
    reply = ('<tool_call>\n{"name": "edit_file", "args": {"path": "a.py"}, '
             '"old": "x = 1", "new": "x = 2"}\n</tool_call>')
    with _scripted(agent, [reply, "Changed x to 2."]):
        agent.run_task("set x to 2 in a.py")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2\n"


def test_other_harness_argument_names_map_onto_edit_file():
    args, renamed = normalize_args(
        TOOL_REGISTRY["edit_file"],
        {"file_path": "a.py", "old_str": "x", "new_str": "y"})
    assert args == {"path": "a.py", "old": "x", "new": "y"}
    assert ("old_str", "old") in renamed


def test_an_alias_is_left_alone_when_the_canonical_name_is_present():
    args, renamed = normalize_args(
        TOOL_REGISTRY["edit_file"], {"path": "a.py", "old": "x", "new": "y", "text": "z"})
    assert args["old"] == "x" and args["text"] == "z"
    assert renamed == []


def test_an_alias_the_tool_declares_itself_is_not_renamed():
    tool = MagicMock()
    tool.params = {"content": {"type": "string"}, "text": {"type": "string"}}
    args, renamed = normalize_args(tool, {"text": "z"})
    assert args == {"text": "z"} and renamed == []


def test_missing_arguments_are_named_with_an_example():
    msg = bad_args_message("edit_file", TOOL_REGISTRY["edit_file"], {"path": "a.py"})
    assert "Missing: old, new." in msg
    assert '"name": "edit_file"' in msg
    assert "positional" not in msg


def test_unknown_arguments_are_named():
    msg = bad_args_message("read_file", TOOL_REGISTRY["read_file"],
                           {"path": "a.py", "colour": "red"})
    assert "Unknown: colour." in msg


def test_arguments_that_bind_get_no_message():
    assert bad_args_message("edit_file", TOOL_REGISTRY["edit_file"],
                            {"path": "a.py", "old": "x", "new": "y"}) is None


def test_the_agent_runs_an_edit_sent_with_other_argument_names(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=4)
    with _scripted(agent, [
        _call("edit_file", {"file_path": "a.py", "old_str": "x = 1", "new_str": "x = 2"}),
        "Changed x to 2.",
    ]):
        agent.run_task("set x to 2 in a.py")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2\n"


def test_the_agent_explains_an_edit_with_missing_arguments(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=4)
    with _scripted(agent, [_call("edit_file", {"path": "a.py"}), "Gave up."]):
        agent.run_task("set x to 2 in a.py")
    results = [t for t in _user_texts(agent) if "<tool_result" in t]
    assert "Missing: old, new." in results[0]
    assert "Bad arguments" not in results[0]


def test_identical_calls_in_one_reply_run_once(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=4)
    read = _call("read_file", {"path": "a.py"})
    ran = []
    real = agent._execute_tool

    def _spy(call, interactive):
        ran.append(call.name)
        return real(call, interactive)

    with _scripted(agent, [read + "\n" + read + "\n" + read, "x is 1."]), \
         patch.object(agent, "_execute_tool", side_effect=_spy):
        agent.run_task("read a.py")
    assert ran == ["read_file"]
    joined = "\n".join(_user_texts(agent))
    assert "[duplicate calls] 2 call(s)" in joined
    assert "could not be parsed" not in joined


# --------------------------------------------------------------------------- #
#  edit_file / edit_files / patch_file                                          #
# --------------------------------------------------------------------------- #

def test_an_edit_whose_new_text_lacks_the_anchor_indentation_is_reindented(tmp_path):
    (tmp_path / "g.py").write_text(_METHOD_FILE, encoding="utf-8")
    result = tool_edit_file(tmp_path, "g.py", "return move",
                            "if move:\n    return move\nreturn None")
    assert result.ok, result.output
    assert "re-indented" in result.output
    text = (tmp_path / "g.py").read_text(encoding="utf-8")
    assert "        if move:\n            return move\n        return None\n" in text
    compile(text, "g.py", "exec")


def test_an_edit_that_breaks_python_syntax_is_refused_and_nothing_written(tmp_path):
    target = tmp_path / "g.py"
    target.write_text(_METHOD_FILE, encoding="utf-8")
    before = target.read_bytes()
    result = tool_edit_file(tmp_path, "g.py", "return move", "return (move")
    assert not result.ok
    assert "Edit NOT applied" in result.output
    assert "syntax error" in result.output
    assert "return (move" in result.output
    assert target.read_bytes() == before


def test_an_edit_to_a_file_that_was_already_broken_still_applies(tmp_path):
    target = tmp_path / "g.py"
    target.write_text("def f(:\n    x = 1\n", encoding="utf-8")
    result = tool_edit_file(tmp_path, "g.py", "x = 1", "x = 2")
    assert result.ok
    assert "x = 2" in target.read_text(encoding="utf-8")
    assert "[syntax check]" in result.output


def test_a_file_type_without_a_syntax_check_is_edited_as_given(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("    a: 1\n", encoding="utf-8")
    result = tool_edit_file(tmp_path, "notes.txt", "a: 1", "a: 2\nb: 3")
    assert result.ok
    assert target.read_text(encoding="utf-8") == "    a: 2\nb: 3\n"


def test_a_json_file_with_comments_is_edited_as_given(tmp_path):
    target = tmp_path / "tsconfig.json"
    target.write_text('{"compilerOptions": {"strict": true}}', encoding="utf-8")
    result = tool_edit_file(tmp_path, "tsconfig.json", '{"compilerOptions"',
                            '{\n  // compiler settings\n  "compilerOptions"')
    assert result.ok, result.output
    assert "// compiler settings" in target.read_text(encoding="utf-8")


def test_edit_files_refuses_a_batch_that_breaks_a_file(tmp_path):
    (tmp_path / "ok.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "g.py").write_text(_METHOD_FILE, encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = tool_edit_files(tmp_path, [
        {"path": "ok.py", "old": "y = 1", "new": "y = 2"},
        {"path": "g.py", "old": "return move", "new": "return (move"},
    ])
    assert not result.ok
    assert "NOT applied" in result.output
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_patch_file_refuses_a_patch_that_breaks_a_file(tmp_path):
    target = tmp_path / "g.py"
    target.write_text(_METHOD_FILE, encoding="utf-8")
    before = target.read_bytes()
    diff = ("--- a/g.py\n+++ b/g.py\n@@ -2,2 +2,2 @@\n"
            "     def move(self, move):\n-        return move\n+        return (move\n")
    result = tool_patch_file(tmp_path, "g.py", diff)
    assert not result.ok
    assert "Patch NOT applied" in result.output
    assert target.read_bytes() == before


def test_a_missed_anchor_says_how_to_add_text(tmp_path):
    (tmp_path / "g.py").write_text(_METHOD_FILE, encoding="utf-8")
    result = tool_edit_file(tmp_path, "g.py", "import random", "import random\nimport os")
    assert not result.ok
    assert "To ADD text" in result.output
    assert "'class Game:'" in result.output


# --------------------------------------------------------------------------- #
#  Code shown instead of written                                               #
# --------------------------------------------------------------------------- #

_CODE_REPLY = (
    "Here is the implementation:\n\n```python\n"
    "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n```\n"
)


def test_code_written_into_the_reply_is_sent_back_to_be_written(tmp_path):
    (tmp_path / "calc.py").write_text("# calc\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [
        _call("read_file", {"path": "calc.py"}),
        _CODE_REPLY,
        _call("write_file", {"path": "calc.py",
                             "content": "def add(a, b):\n    return a + b\n"}),
        "Wrote add to calc.py.",
    ]):
        result = agent.run_task("implement add and sub in calc.py")
    assert any("[code not written]" in t for t in _user_texts(agent))
    assert "def add" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert _final_answer(result) == "Wrote add to calc.py."
    assert agent.last_run_ok is True


def test_code_left_unwritten_after_the_nudge_is_reported_not_finished(tmp_path):
    (tmp_path / "calc.py").write_text("# calc\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [_call("read_file", {"path": "calc.py"}),
                           _CODE_REPLY, _CODE_REPLY + "\nDone."]):
        result = agent.run_task("implement add and sub in calc.py")
    assert "[not finished: nothing was written to any file" in result
    assert agent.last_run_ok is False


def test_the_escape_does_not_answer_a_change_request(tmp_path):
    """The live failure: qwen2.5-coder-7b answered a change request with
    "[no tool needed]" plus code, then claimed the file was already in place."""
    (tmp_path / "calc.py").write_text("# calc\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [
        _call("read_file", {"path": "calc.py"}),
        "[no tool needed] Here it is.\n" + _CODE_REPLY,
        "[no tool needed] The code is already placed in calc.py.",
    ]):
        result = agent.run_task("implement add and sub in calc.py")
    nudges = [t for t in _user_texts(agent) if "[code not written]" in t]
    assert len(nudges) == 1 and "[no tool needed]" not in nudges[0]
    assert "[not finished: nothing was written to any file" in result
    assert agent.last_run_ok is False


def test_the_first_nudge_on_a_change_request_offers_no_escape(tmp_path):
    agent = _make_agent(tmp_path, max_turns=6)
    with _scripted(agent, ["I would write it like this.", "Still prose.", "More prose."]), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        agent.run_task("implement add in calc.py")
    rung1 = [t for t in _user_texts(agent) if t.startswith("[no tool call]")]
    assert rung1 and "[no tool needed]" not in rung1[0]


def test_code_in_an_answer_to_a_question_is_not_sent_back(tmp_path):
    agent = _make_agent(tmp_path, max_turns=4)
    with _scripted(agent, [_CODE_REPLY]):
        agent.run_task("what is the difference between a list and a tuple?")
    assert not any("[code not written]" in t for t in _user_texts(agent))
    assert agent.last_run_ok is True


def test_quoting_a_file_to_answer_a_question_about_it_is_not_sent_back(tmp_path):
    (tmp_path / "calc.py").write_text("# calc\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=6)
    for request in ("How does add() in calc.py work?", "show me calc.py"):
        with _scripted(agent, [_call("read_file", {"path": "calc.py"}), _CODE_REPLY]):
            agent.run_task(request)
        assert not any("[code not written]" in t for t in _user_texts(agent)), request
        assert agent.last_run_ok is True, request


def test_work_written_through_patch_mode_or_a_shell_is_not_sent_back(tmp_path):
    (tmp_path / "calc.py").write_text("# calc\n", encoding="utf-8")
    patched = _make_agent(tmp_path, max_turns=6, patch_mode=True)
    with _scripted(patched, [
        _call("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}),
        _CODE_REPLY,
    ]):
        patched.run_task("implement add and sub in calc.py")
    assert not any("[code not written]" in t for t in _user_texts(patched))
    shell = _make_agent(tmp_path, max_turns=6)
    with _scripted(shell, [_call("run_shell", {"command": "echo hi"}), _CODE_REPLY]):
        shell.run_task("implement add and sub in calc.py")
    assert not any("[code not written]" in t for t in _user_texts(shell))


def test_a_shell_block_is_not_file_code():
    assert not _has_file_code_block("```bash\npip install x\npytest\nls\ncd src\n```")
    assert _has_file_code_block(_CODE_REPLY)
    assert not _has_file_code_block("```python\nx = 1\n```")


# --------------------------------------------------------------------------- #
#  A step announced but never taken                                             #
# --------------------------------------------------------------------------- #

def test_a_reply_that_stops_on_an_announced_step_is_sent_back(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [
        _call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"}),
        "I updated a.py.\n\nLet me run the tests.",
        "The change is complete: x is now 2.",
    ]):
        result = agent.run_task("set x to 2 in a.py")
    nudges = [t for t in _user_texts(agent) if "[unfinished]" in t]
    assert len(nudges) == 1 and "Let me run the tests." in nudges[0]
    assert _final_answer(result) == "The change is complete: x is now 2."
    assert agent.last_run_ok is True


def test_a_run_that_still_stops_on_an_announced_step_is_not_finished(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [
        _call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"}),
        "Let me run the tests.",
        "Now I'll run the tests.",
    ]):
        result = agent.run_task("set x to 2 in a.py")
    assert "[not finished: the run ended on" in result
    assert agent.last_run_ok is False


def test_a_bare_decline_after_the_nudge_still_leaves_the_step_untaken(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, max_turns=8)
    with _scripted(agent, [
        _call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"}),
        "Let me run the tests.",
        "[no tool needed]",
    ]):
        result = agent.run_task("set x to 2 in a.py")
    assert '[not finished: the run ended on "Let me run the tests."' in result
    assert agent.last_run_ok is False


def test_closing_courtesies_are_not_announced_steps():
    for text in ("Let me know if you need anything else.",
                 "I'll be happy to help further!",
                 "Would you like me to run the tests?",
                 "Let me put it simply: it returns a + b.",
                 "Let's see: the function returns the sum.",
                 "I'll note that the tests already pass.",
                 "Done. The tests pass."):
        assert _announced_step(text) == "", text
    for text in ("Let me run the tests.", "Now I'll update main.py:",
                 "**Next, I will add the tests.**", "Okay, let's write the file."):
        assert _announced_step(text), text


def test_an_announcement_inside_a_code_block_does_not_count():
    assert _announced_step("Summary done.\n```python\n# Let me run this\nx = 1\n```") == ""


# --------------------------------------------------------------------------- #
#  "[no tool needed]" with nothing after it                                     #
# --------------------------------------------------------------------------- #

def test_an_empty_decline_keeps_the_earlier_answer(tmp_path):
    agent = _make_agent(tmp_path, max_turns=6)
    with _scripted(agent, ["A plan: use pygame for the board.", "[no tool needed]"]), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        result = agent.run_task("the goal is a tic tac toe game for the project")
    assert _final_answer(result) == "A plan: use pygame for the board."


def test_an_empty_decline_on_an_imperative_request_keeps_climbing(tmp_path):
    agent = _make_agent(tmp_path, max_turns=6)
    forced = []

    def _reply(*a, **k):
        forced.append(bool(getattr(agent, "_force_tool_grammar", False)))
        return "Here is how." if len(forced) == 1 else "[no tool needed]"

    with patch.object(agent, "_call_llm", side_effect=_reply), \
         patch("localm.plugins.coder.agent.parse_tool_calls", return_value=[]):
        agent.run_task("start implementing all the features")
    assert forced[:3] == [False, False, True]
