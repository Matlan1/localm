# SPDX-License-Identifier: AGPL-3.0-or-later
"""Each tool_call event and its tool_result carry the same call id, and two calls
of one turn carry different ids, so a consumer can pair them whatever order the
results arrive in."""

from unittest.mock import patch

from localm.audit import SessionMode
from localm.plugins.coder.parser import ToolCall


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, events):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        return Agent(_Stub(), cwd=cwd, self_verify=False, mode=SessionMode.LOG,
                     auto_approve=True, on_event=events.append)


def _read(path):
    return ToolCall(name="read_file", args={"path": str(path)}, raw="", start=0, end=0)


def test_parallel_calls_each_get_their_own_id_on_call_and_result(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    events = []
    agent = _agent(tmp_path, events)
    agent._execute_tools([_read(tmp_path / "a.txt"), _read(tmp_path / "b.txt")],
                         interactive=False)
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 2 and len(results) == 2
    assert all(e.get("call_id") for e in calls + results)
    assert len({e["call_id"] for e in calls}) == 2
    assert {e["call_id"] for e in calls} == {e["call_id"] for e in results}


def test_a_refused_call_still_pairs_its_result(tmp_path):
    events = []
    agent = _agent(tmp_path, events)
    agent.cancel("test")
    agent._execute_tool(_read(tmp_path / "x.txt"), interactive=False)
    [result] = [e for e in events if e["type"] == "tool_result"]
    assert result.get("call_id")
