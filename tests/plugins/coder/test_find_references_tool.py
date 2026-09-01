# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the find_references tool (coder/tools/references.py): the
dispatcher wiring that injects the running session's live ProjectMap, and the
tool's own error/formatting behaviour.

test_find_references_survives_a_real_dispatch_and_edit_cycle drives the REAL
dispatch path (Agent._execute_tool, which injects _session) against a REAL
ProjectMap built from real files on disk, matching test_todos.py's pattern of
proving the wiring works end to end rather than by calling the tool function
directly with a hand-supplied session.
"""

from unittest.mock import patch

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.indexer import ProjectMap
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools.references import tool_find_references


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
    """Dispatch a tool the way the product does: through Agent._execute_tool."""
    return agent._execute_tool(
        ToolCall(name=name, args=args, raw="", start=0, end=0), interactive=False)


# --------------------------------------------------------------------------- #
#  Real dispatch + real ProjectMap, end to end                                #
# --------------------------------------------------------------------------- #

def test_find_references_survives_a_real_dispatch_and_edit_cycle(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")
    (proj / "b.py").write_text("def run():\n    return helper(1)\n", encoding="utf-8")
    (proj / "c.py").write_text("result = helper(2)\n", encoding="utf-8")

    agent = _agent(proj)
    # _agent()'s fixture mocks ProjectMap.build entirely (see test_todos.py's
    # identical pattern); swap in the real, real-project index the tool needs.
    agent._project_map = ProjectMap.build(proj)

    result = _call(agent, "find_references", symbol="helper")
    assert result.ok
    assert "b.py:2" in result.output
    assert "c.py:1" in result.output
    assert "2 reference(s)" in result.summary

    # An edit, then the map going dirty (as run_shell/_refresh_map_for_tool
    # would do for real) - find_references must reflect it, not a stale read.
    (proj / "b.py").write_text("def run():\n    return other(1)\n", encoding="utf-8")
    agent._project_map.mark_dirty()

    after = _call(agent, "find_references", symbol="helper")
    assert after.ok
    assert "b.py" not in after.output
    assert "c.py:1" in after.output
    assert "1 reference(s)" in after.summary


def test_find_references_reports_zero_hits_without_erroring(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("x = 1\n", encoding="utf-8")

    agent = _agent(proj)
    agent._project_map = ProjectMap.build(proj)

    result = _call(agent, "find_references", symbol="nonexistent_symbol")
    assert result.ok
    assert "No call sites found" in result.output
    assert "0 reference(s)" in result.summary


def test_the_injected_session_arg_cannot_be_spoofed_by_the_model(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (proj / "b.py").write_text("helper()\n", encoding="utf-8")

    agent = _agent(proj)
    agent._project_map = ProjectMap.build(proj)

    # A model-supplied "_session" must lose to the dispatcher's real injection,
    # exactly like test_todos.py's equivalent for set_todos.
    result = _call(agent, "find_references", symbol="helper", _session="nonsense")
    assert result.ok
    assert "b.py:1" in result.output


# --------------------------------------------------------------------------- #
#  Tool-level behaviour: errors are reported, never swallowed (rule 5)        #
# --------------------------------------------------------------------------- #

def test_find_references_without_a_session_reports_the_failure(tmp_path):
    r = tool_find_references(tmp_path, symbol="helper")
    assert not r.ok
    assert "nothing was searched" in r.output


def test_find_references_rejects_an_empty_symbol(tmp_path):
    class _FakeSession:
        _project_map = ProjectMap.build(tmp_path)

    r = tool_find_references(tmp_path, symbol="   ", _session=_FakeSession())
    assert not r.ok
    assert "non-empty" in r.output


def test_find_references_truncates_a_large_result_set(tmp_path):
    from localm.plugins.coder.tools.references import _MAX_RESULTS

    lines = [f"target({i})" for i in range(_MAX_RESULTS + 20)]
    (tmp_path / "many.py").write_text("\n".join(lines), encoding="utf-8")

    class _FakeSession:
        _project_map = ProjectMap.build(tmp_path)

    r = tool_find_references(tmp_path, symbol="target", _session=_FakeSession())
    assert r.ok
    assert r.truncated is True
    assert "more)" in r.output
