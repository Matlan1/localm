# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model-owned task list (coder/tools/tasks.py).

test_todos_survive_a_real_checkpoint_resume_cycle drives the REAL dispatch path
(Agent._execute_tool, which injects the session), the REAL checkpoint file under
a temp HOME, and a genuinely FRESH Agent for the same cwd. Both halves of the
persistence wiring are pinned independently: the file must contain the todos,
and the fresh agent must read them back.
"""

import json
from unittest.mock import patch

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools.tasks import (
    DONE, IN_PROGRESS, MAX_ITEMS, MAX_TEXT, PENDING,
    normalize_todos, render_todos, todos_summary,
)
from tests.conftest import final_answer as _final_answer


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(agent, name, **args):
    """Run a tool through the real dispatcher (hidden-arg injection included)."""
    return agent._execute_tool(
        ToolCall(name=name, args=args, raw="", start=0, end=0), interactive=False)


PLAN = ["[x] read the failing test", "[>] fix the parser", "[ ] run the suite"]


# --------------------------------------------------------------------------- #
#  Todos survive a real checkpoint save/resume cycle
# --------------------------------------------------------------------------- #

def test_todos_survive_a_real_checkpoint_resume_cycle(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()

    # 1. A session writes a plan and is interrupted.
    first = _agent(proj)
    assert _call(first, "set_todos", items=PLAN).ok
    first._messages = [{"role": "user", "content": "do the thing"}]
    first.save_checkpoint()

    # 2. The plan is IN the checkpoint file, under HOME and NOT in the project
    #    tree.
    assert (tmp_path / "home") in first._checkpoint_path.parents
    assert not (proj / ".localcoder").exists()
    assert list(proj.iterdir()) == []
    saved = json.loads(first._checkpoint_path.read_text(encoding="utf-8"))
    assert saved["todos"] == [
        {"text": "read the failing test", "status": DONE},
        {"text": "fix the parser",        "status": IN_PROGRESS},
        {"text": "run the suite",         "status": PENDING},
    ]

    # 3. A genuinely fresh Agent for the same cwd resumes and reads it back
    #    (pins the restore half).
    second = _agent(proj)
    assert second.get_todos() == []          # fresh session starts empty
    data = second.load_checkpoint()
    assert data is not None
    second.resume_checkpoint(data)

    result = _call(second, "read_todos")
    assert result.ok
    assert result.output == (
        "[x] read the failing test\n[>] fix the parser\n[ ] run the suite")
    assert "1/3 done" in result.summary and "fix the parser" in result.summary


def test_a_real_agent_turn_writes_and_resumes_the_plan(tmp_path, monkeypatch):
    """The same round trip driven through the REAL loop: a model response is
    parsed into a tool call, dispatched, checkpointed, and resumed - so the
    parser, the loop, the hidden-arg injection, and the checkpoint are all
    exercised together rather than one dispatcher call at a time."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()

    class _Scripted(_Stub):
        """Emits a genuine <tool_call> block, then a plain final answer."""
        def __init__(self):
            self.replies = [
                '<tool_call>\n{"name": "set_todos", "args": {"items": '
                '["[x] read the failing test", "[>] fix the parser", '
                '"[ ] run the suite"]}}\n</tool_call>',
                "Plan written.",
            ]

        def chat(self, messages, **kw):
            return self.replies.pop(0) if self.replies else "Done."

        def chat_stream(self, messages, **kw):
            yield self.chat(messages, **kw)

    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        a = Agent(_Scripted(), cwd=proj, auto_approve=True, self_verify=False,
                  mode=SessionMode.LOG)
        answer = a.run_task("Plan the parser fix.")

    assert _final_answer(answer).strip() == "Plan written."
    assert [t["text"] for t in a.get_todos()] == [
        "read the failing test", "fix the parser", "run the suite"]

    a.save_checkpoint()
    resumed = _agent(proj)
    resumed.resume_checkpoint(resumed.load_checkpoint())
    assert resumed.get_todos() == a.get_todos()


def test_privacy_mode_writes_no_todos_to_disk(tmp_path, monkeypatch):
    """Privacy mode's no-disk promise covers the task list too: the checkpoint
    is a no-op there, so a fresh session finds nothing to resume."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()

    a = _agent(proj, mode=SessionMode.PRIVACY)
    assert _call(a, "set_todos", items=PLAN).ok
    assert a.get_todos()                      # in memory, as usual
    a._messages = [{"role": "user", "content": "hi"}]
    a.save_checkpoint()

    assert not a._checkpoint_path.exists()
    assert not (proj / ".localcoder").exists()
    fresh = _agent(proj, mode=SessionMode.PRIVACY)
    assert fresh.load_checkpoint() is None
    assert fresh.get_todos() == []


def test_resume_of_a_pre_b2_checkpoint_is_not_a_crash(tmp_path, monkeypatch):
    """A checkpoint written before this feature has no todos key at all."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()

    a = _agent(proj)
    a.resume_checkpoint({"version": 1, "turns": 2, "total_tokens": 5,
                         "messages": [{"role": "user", "content": "x"}]})
    assert a.get_todos() == []
    assert a._turns == 2


def test_garbage_todos_in_a_checkpoint_are_dropped_not_trusted(tmp_path, monkeypatch):
    """The checkpoint is plain user-writable JSON: a hand-edited or corrupted
    todos value normalises rather than crashing or landing unvalidated."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()

    a = _agent(proj)
    a.resume_checkpoint({
        "version": 1, "messages": [],
        "todos": [{"text": "kept", "status": "not-a-status"}, None, 42,
                  ["nested"], {"nope": 1}, "[x] also kept"],
    })
    assert a.get_todos() == [
        {"text": "kept", "status": PENDING},    # unknown status -> not done
        {"text": "42", "status": PENDING},      # a number is a task line
        {"text": "also kept", "status": DONE},
    ]

    a.resume_checkpoint({"version": 1, "messages": [], "todos": "not a list"})
    assert a.get_todos() == [{"text": "not a list", "status": PENDING}]

    a.resume_checkpoint({"version": 1, "messages": [], "todos": 17})
    assert a.get_todos() == []


# --------------------------------------------------------------------------- #
#  Compaction: the plan survives it
# --------------------------------------------------------------------------- #

def test_todos_survive_compaction_and_land_in_the_summary(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    assert _call(a, "set_todos", items=PLAN).ok
    a._messages = [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"m{i}"} for i in range(10)]

    assert a._compact_history() is True
    assert a.get_todos()                       # the store itself is untouched
    summary = a._messages[0]["content"]
    assert "Task list (set_todos):" in summary
    assert "[>] fix the parser" in summary


def test_compaction_without_todos_adds_no_task_section(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    a._messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    assert a._compact_history() is True
    assert "Task list" not in a._messages[0]["content"]


# --------------------------------------------------------------------------- #
#  Registration + classification
# --------------------------------------------------------------------------- #

def test_registered_non_destructive_and_free_of_path_args():
    from localm.plugins.coder.tools import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY
    for name in ("set_todos", "read_todos"):
        td = TOOL_REGISTRY[name]
        assert td.destructive is False
        assert td.untrusted_output is False
        # No path-like arg, so the scope allowlist has nothing to confine.
        assert not any(p.endswith(("path", "_file", "_dir")) for p in td.params)
        assert name in SAFE_RESTRICTED_TOOLS


def test_a_restricted_session_keeps_the_task_list(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj, restricted=True)
    assert "set_todos" not in a.disabled_tools
    assert _call(a, "set_todos", items=["[ ] read auth.py"]).ok
    assert a.get_todos() == [{"text": "read auth.py", "status": PENDING}]


def test_native_tool_schema_declares_items_as_a_string_array():
    from localm.plugins.coder.agent.tooldefs import _build_openai_tool_defs
    fn = next(d["function"] for d in _build_openai_tool_defs()
              if d["function"]["name"] == "set_todos")
    items = fn["parameters"]["properties"]["items"]
    assert items["type"] == "array" and items["items"] == {"type": "string"}
    assert fn["parameters"]["required"] == ["items"]


def test_prompt_documents_the_tool_with_a_worked_example():
    from localm.plugins.coder.prompts import _brief_tool_docs, _full_tool_docs
    full = _full_tool_docs()
    assert "## set_todos" in full and "[>] fix the parser" in full
    assert "read_todos()" in _brief_tool_docs()


# --------------------------------------------------------------------------- #
#  Dispatch behaviour
# --------------------------------------------------------------------------- #

def test_dry_run_still_records_todos(tmp_path):
    """dry_run skips DESTRUCTIVE tools; the task list is bookkeeping, so it is
    still recorded."""
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj, dry_run=True)
    assert _call(a, "set_todos", items=PLAN).ok
    assert len(a.get_todos()) == 3


def test_unattended_session_is_not_denied_its_own_task_list(tmp_path):
    """auto_approve=False with no confirm handler fail-closes every tool that
    needs confirmation; the task list does not need one."""
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj, auto_approve=False)
    result = _call(a, "set_todos", items=PLAN)
    assert result.ok, result.output
    assert a.get_todos()


def test_two_set_todos_in_one_parallel_batch_stay_intact(tmp_path):
    """Non-destructive tools run concurrently in one ThreadPoolExecutor batch
    (agent/loop.py). The whole-list swap under the lock means the loser is
    overwritten, never interleaved: the store holds ONE of the two lists whole."""
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    first  = [f"[ ] first-{i}" for i in range(12)]
    second = [f"[ ] second-{i}" for i in range(12)]
    blocks = a._execute_tools(
        [ToolCall(name="set_todos", args={"items": first},  raw="", start=0, end=0),
         ToolCall(name="set_todos", args={"items": second}, raw="", start=0, end=0)],
        interactive=False)
    assert len(blocks) == 2
    texts = [t["text"] for t in a.get_todos()]
    assert texts in ([f"first-{i}" for i in range(12)],
                     [f"second-{i}" for i in range(12)])


def test_the_gui_and_audit_see_every_task_list_write(tmp_path):
    """Surfacing: the tool_call event carries the items (the GUI's tool card
    renders them) and the tool_result carries the rendered list + summary."""
    proj = tmp_path / "proj"; proj.mkdir()
    events = []
    a = _agent(proj, on_event=events.append)
    audit = []
    a._audit.tool_call = lambda name, args: audit.append((name, args))
    assert _call(a, "set_todos", items=PLAN).ok

    call_ev = next(e for e in events if e["type"] == "tool_call")
    assert call_ev["tool"] == "set_todos" and call_ev["args"]["items"] == PLAN
    res_ev = next(e for e in events if e["type"] == "tool_result")
    assert res_ev["ok"] and "[>] fix the parser" in res_ev["output"]
    assert "1/3 done" in res_ev["summary"]
    assert audit == [("set_todos", {"items": PLAN})]


def test_the_injected_session_arg_cannot_be_spoofed_by_the_model(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    assert _call(a, "set_todos", items=["[ ] real"], _session="nonsense").ok
    assert a.get_todos() == [{"text": "real", "status": PENDING}]


def test_a_child_agent_gets_its_own_list(tmp_path):
    """No inheritance: a sub-agent plans its own sub-task."""
    proj = tmp_path / "proj"; proj.mkdir()
    parent = _agent(proj)
    _call(parent, "set_todos", items=PLAN)
    child = _agent(proj, parent=parent)
    assert child.get_todos() == []
    assert len(parent.get_todos()) == 3


# --------------------------------------------------------------------------- #
#  Tool-level behaviour: errors are reported, never swallowed
# --------------------------------------------------------------------------- #

def test_set_todos_without_a_session_reports_the_failure(tmp_path):
    """Rule 5: a write with nowhere to go must NOT report success."""
    from localm.plugins.coder.tools.tasks import tool_read_todos, tool_set_todos
    r = tool_set_todos(tmp_path, items=PLAN)
    assert not r.ok and "nothing was recorded" in r.output
    assert not tool_read_todos(tmp_path).ok


def test_missing_or_wrong_typed_items_error_out(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    _call(a, "set_todos", items=PLAN)

    missing = _call(a, "set_todos")
    assert not missing.ok and "items" in missing.output
    wrong = _call(a, "set_todos", items=42)
    assert not wrong.ok and "must be a list" in wrong.output
    empty_text = _call(a, "set_todos", items=["   ", "[x]  "])
    assert not empty_text.ok and "any task text" in empty_text.output
    # A rejected write leaves the previous plan alone.
    assert len(a.get_todos()) == 3


def test_clearing_the_list_is_allowed(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    _call(a, "set_todos", items=PLAN)
    cleared = _call(a, "set_todos", items=[])
    assert cleared.ok and a.get_todos() == []
    assert _call(a, "read_todos").summary == "todos: none"


def test_partly_unusable_items_say_what_was_dropped(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    r = _call(a, "set_todos", items=["[ ] kept", "", None, "[x] also kept"])
    assert r.ok
    assert "2 of 4 items had no task text" in r.output
    assert len(a.get_todos()) == 2


def test_set_todos_replaces_rather_than_appends(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    _call(a, "set_todos", items=PLAN)
    _call(a, "set_todos", items=["[x] fix the parser", "[>] run the suite"])
    assert a.get_todos() == [
        {"text": "fix the parser", "status": DONE},
        {"text": "run the suite",  "status": IN_PROGRESS},
    ]


def test_the_store_cannot_be_mutated_through_a_returned_list(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    _call(a, "set_todos", items=PLAN)
    snapshot = a.get_todos()
    assert snapshot[0]["status"] == DONE
    snapshot[0]["status"] = PENDING            # would un-finish task 1
    snapshot[1]["text"] = "hijacked"
    snapshot.append({"text": "smuggled", "status": PENDING})

    stored = a.get_todos()
    assert len(stored) == 3
    assert stored[0]["status"] == DONE
    assert stored[1]["text"] == "fix the parser"
    assert all(t["text"] != "smuggled" for t in stored)


# --------------------------------------------------------------------------- #
#  Parsing: the shapes a model actually emits
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("[ ] plain",            {"text": "plain", "status": PENDING}),
    ("plain, no marker",     {"text": "plain, no marker", "status": PENDING}),
    ("[x] done",             {"text": "done", "status": DONE}),
    ("[X] shouty",           {"text": "shouty", "status": DONE}),
    ("[✓] glyph",            {"text": "glyph", "status": DONE}),
    ("[>] working",          {"text": "working", "status": IN_PROGRESS}),
    ("[*] also working",     {"text": "also working", "status": IN_PROGRESS}),
    ("- [x] bulleted",       {"text": "bulleted", "status": DONE}),
    ("* [>] star bullet",    {"text": "star bullet", "status": IN_PROGRESS}),
    ("3. [ ] numbered",      {"text": "numbered", "status": PENDING}),
    ("[done] word marker",   {"text": "word marker", "status": DONE}),
    ("[wip] word marker",    {"text": "word marker", "status": IN_PROGRESS}),
    # An unrecognised bracket is the model's own text, not a status marker, and
    # is kept verbatim.
    ("[?] unknown marker",   {"text": "[?] unknown marker", "status": PENDING}),
    ("[api] fix the handler", {"text": "[api] fix the handler", "status": PENDING}),
    ("[ ]   spaced   out ",  {"text": "spaced out", "status": PENDING}),
    ("[x] multi\nline",      {"text": "multi line", "status": DONE}),
    ({"text": "dict form", "status": "in-progress"},
     {"text": "dict form", "status": IN_PROGRESS}),
    ({"task": "alt key", "status": "COMPLETED"},
     {"text": "alt key", "status": DONE}),
])
def test_item_shapes_normalise(raw, expected):
    assert normalize_todos([raw]) == [expected]


@pytest.mark.parametrize("raw", ["", "   ", "[ ]", "[x]   ", None, [], {}, {"status": "done"}])
def test_items_without_task_text_are_dropped(raw):
    assert normalize_todos([raw]) == []


def test_a_newline_joined_string_is_accepted_as_the_list(tmp_path):
    """A model that ignores the array type and sends one newline-joined blob
    still yields a list, not a single 'task' holding its whole plan."""
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    r = _call(a, "set_todos", items="[x] one\n[>] two\n[ ] three")
    assert r.ok
    assert [t["status"] for t in a.get_todos()] == [DONE, IN_PROGRESS, PENDING]


def test_the_list_is_bounded(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    a = _agent(proj)
    r = _call(a, "set_todos", items=[f"[ ] task {i}" for i in range(MAX_ITEMS + 25)])
    assert r.ok and len(a.get_todos()) == MAX_ITEMS
    assert f"{MAX_ITEMS}-item cap" in r.output

    _call(a, "set_todos", items=["[ ] " + "x" * (MAX_TEXT * 3)])
    assert len(a.get_todos()[0]["text"]) == MAX_TEXT


def test_render_and_summary_are_stable():
    todos = normalize_todos(PLAN)
    assert render_todos(todos) == (
        "[x] read the failing test\n[>] fix the parser\n[ ] run the suite")
    assert todos_summary(todos) == "1/3 done - now: fix the parser"
    assert todos_summary(normalize_todos(["[x] a", "[x] b"])) == "2/2 done"
    assert todos_summary([]) == "no tasks"
    # Round-trips: the rendered form parses back to the same list, including a
    # task whose own text starts with a bracket.
    assert normalize_todos(render_todos(todos).splitlines()) == todos
    tagged = normalize_todos(["[>] [api] fix the handler"])
    assert tagged == [{"text": "[api] fix the handler", "status": IN_PROGRESS}]
    assert normalize_todos(render_todos(tagged).splitlines()) == tagged
