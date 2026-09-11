# SPDX-License-Identifier: AGPL-3.0-or-later
"""An unattended one-shot run (the in-process MCP coder tool, or `localm coder`
with a task) has nobody to confirm a tool call. A model whose whole response
is an exact ```json-fenced or bare call object must still be able to write
files; a call that is denied for want of a confirmation must be reported on
the run rather than hidden behind success=True; and a denied call the model
re-emits in the <tool_call> format runs and clears the record.

Driven through runner.build_agent + run_single_task on a real Agent with a
scripted backend, the same path run_coder_task and the CLI take.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from localm.plugins.coder import runner as coder_runner
from localm.plugins.coder.runner import describe_denied

FENCE = "```"


def _fenced_call(name: str, **args) -> str:
    return FENCE + "json\n" + json.dumps({"name": name, "args": args}, indent=2) + "\n" + FENCE


def _bare_call(name: str, **args) -> str:
    return json.dumps({"name": name, "args": args})


def _headed_call(name: str, **args) -> str:
    """The lenient shape a model with no tool-call training was seen to emit:
    its own heading, then a bare JSON object."""
    return f"## {name}\n" + _bare_call(name, **args)


def _xml_call(tool: str, **args) -> str:
    return "<tool_call>\n" + json.dumps({"name": tool, "args": args}) + "\n</tool_call>"


class _ScriptedBackend:
    """Answers one canned reply per call, repeating the last, and keeps every
    message list it was handed."""
    model_id = "stub-model"
    native_tools = False
    supports_grammar = False

    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list[dict]] = []

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw) -> str:
        reply = self._replies[min(len(self.seen), len(self._replies) - 1)]
        self.seen.append([dict(m) for m in messages])
        return reply

    def chat_stream(self, messages, **kw):
        return iter([self.chat(messages, **kw)])


def _run(tmp_path: Path, replies, *, yes: bool = False):
    project = tmp_path / "proj"
    project.mkdir()
    cfg = coder_runner.resolve_task_config(project, yes=yes)
    backend = _ScriptedBackend(replies)
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = coder_runner.build_agent(
            backend, project, task="write NOTE.txt", max_turns=cfg.max_turns,
            auto_approve=cfg.auto_approve, always_confirm=cfg.always_confirm,
            session_mode=cfg.session_mode, gen_kw=cfg.gen_kw)
        result = coder_runner.run_single_task(agent, "write NOTE.txt")
    return project, backend, result


def _last_user_text(backend: _ScriptedBackend) -> str:
    """The tool results fed back to the model on its final call."""
    last = backend.seen[-1]
    return "\n".join(m.get("content", "") for m in last if m.get("role") == "user")


def test_an_exact_json_fenced_write_runs_unattended(tmp_path):
    project, _backend, result = _run(
        tmp_path,
        [_fenced_call("write_file", path="NOTE.txt", content="hello from the coder\n"),
         "Done."])
    assert (project / "NOTE.txt").read_text(encoding="utf-8") == "hello from the coder\n"
    assert result.denied == ()
    assert result.success is True


def test_a_bare_exact_call_as_the_whole_response_runs_unattended(tmp_path):
    project, _backend, result = _run(
        tmp_path,
        [_bare_call("write_file", path="NOTE.txt", content="hello from the coder"),
         "Done."])
    assert (project / "NOTE.txt").read_text(encoding="utf-8") == "hello from the coder"
    assert result.denied == ()
    assert result.success is True


def test_a_fenced_write_quoted_inside_prose_is_denied(tmp_path):
    """The same fence with prose around it is a quotation, not an invocation:
    it keeps the confirmation gate and the run says so."""
    project, _backend, result = _run(
        tmp_path,
        ["Here is what I ran earlier:\n"
         + _fenced_call("write_file", path="NOTE.txt", content="hello\n")
         + "\nNothing more to do.",
         "Done."])
    assert not (project / "NOTE.txt").exists(), "a quoted fence wrote a file unconfirmed"
    assert result.denied == (("write_file", "lenient"),)
    assert result.success is False


def test_a_headed_bare_json_write_is_denied_and_the_run_reports_it(tmp_path):
    project, backend, result = _run(
        tmp_path,
        [_headed_call("write_file", path="NOTE.txt", content="hello\n"),
         "I could not write the file."])
    assert not (project / "NOTE.txt").exists(), "a headed bare JSON call wrote a file unconfirmed"
    assert result.denied == (("write_file", "lenient"),)
    assert result.success is False
    assert result.as_dict()["denied"] == [{"tool": "write_file", "reason": "lenient"}]
    # The model was told exactly how to make the call run.
    fed_back = _last_user_text(backend)
    assert "not written in the <tool_call> format" in fed_back
    assert '<tool_call>\n{"name": "write_file", "args": {...}}\n</tool_call>' in fed_back


def test_a_denied_call_re_emitted_as_tool_call_runs_and_clears_the_record(tmp_path):
    project, _backend, result = _run(
        tmp_path,
        [_headed_call("write_file", path="NOTE.txt", content="hello\n"),
         _xml_call("write_file", path="NOTE.txt", content="hello\n"),
         "Done."])
    assert (project / "NOTE.txt").read_text(encoding="utf-8") == "hello\n"
    assert result.denied == ()
    assert result.success is True


def test_a_denied_call_re_emitted_with_other_arguments_stays_recorded(tmp_path):
    project, _backend, result = _run(
        tmp_path,
        [_headed_call("write_file", path="NOTE.txt", content="hello\n"),
         _xml_call("write_file", path="OTHER.txt", content="x\n"),
         "Done."])
    assert not (project / "NOTE.txt").exists()
    assert (project / "OTHER.txt").exists()
    assert result.denied == (("write_file", "lenient"),)
    assert result.success is False


def test_a_shell_call_denied_without_yes_is_reported_as_unconfirmable(tmp_path):
    marker = tmp_path / "proj" / "marker.txt"
    cmd = f'"{sys.executable}" -c "open(\'marker.txt\', \'w\').close()"'
    replies = [_xml_call("run_shell", command=cmd), "Done."]
    project, _backend, result = _run(tmp_path, replies)
    assert not marker.exists(), "run_shell ran without a confirmation channel"
    assert result.denied == (("run_shell", "unconfirmable"),)
    assert result.success is False

    project2 = tmp_path / "proj"
    for p in project2.iterdir():
        if p.is_file():
            p.unlink()
    project2.rmdir()
    _project, _backend, result = _run(tmp_path, replies, yes=True)
    assert marker.exists(), "run_shell did not run under yes"
    assert result.denied == ()
    assert result.success is True


def test_describe_denied_names_every_call_and_the_reason():
    assert describe_denied(()) == ""
    text = describe_denied((("write_file", "lenient"),))
    assert text.startswith("1 tool call was denied and did not run")
    assert "write_file: the call was not written in the <tool_call> format" in text
    text = describe_denied((("write_file", "lenient"), ("write_file", "lenient"),
                            ("run_shell", "unconfirmable")))
    assert text.startswith("3 tool calls were denied and did not run")
    assert "write_file x2:" in text
    assert "run_shell: the tool needs a confirmation an unattended run cannot give" in text


def test_a_sub_agents_denied_write_is_reported_on_the_parent_run(tmp_path):
    """spawn_agent's child inherits the parent's unattended posture, so its
    denied write must reach the parent's outcome: the parent's record names it
    with a sub-agent prefix, the run is not a success, and the tool result the
    parent model reads says so. Driven end to end on one shared scripted
    backend: reply 1 is the parent's spawn, replies 2-3 are the child's, reply
    4 the parent's final answer."""
    project, backend, result = _run(
        tmp_path,
        [_xml_call("spawn_agent", task="write NOTE.txt", name="helper"),
         _headed_call("write_file", path="NOTE.txt", content="hello\n"),
         "child: I could not write the file.",
         "parent: the helper reported a problem."])
    assert not (project / "NOTE.txt").exists(), "the child's lenient write ran unconfirmed"
    assert result.denied == (("sub-agent helper:write_file", "lenient"),)
    assert result.success is False
    fed_back = _last_user_text(backend)
    assert "[sub-agent 'helper': 1 tool call was denied and did not run" in fed_back


def test_a_parents_identical_call_does_not_clear_a_sub_agents_denial(tmp_path):
    project, _backend, result = _run(
        tmp_path,
        [_xml_call("spawn_agent", task="write NOTE.txt", name="helper"),
         _headed_call("write_file", path="NOTE.txt", content="hello\n"),
         "child: could not write.",
         _xml_call("write_file", path="NOTE.txt", content="hello\n"),
         "parent: wrote it myself."])
    assert (project / "NOTE.txt").read_text(encoding="utf-8") == "hello\n"
    assert result.denied == (("sub-agent helper:write_file", "lenient"),)
    assert result.success is False


def test_a_timed_out_run_still_carries_its_denials(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(coder_runner, "STOP_GRACE_SECONDS", 0.2)
    gate = threading.Event()

    class _Gated(_ScriptedBackend):
        def chat(self, messages, **kw):
            if len(self.seen) == 1:
                gate.wait()
            return super().chat(messages, **kw)

    project = tmp_path / "proj"
    project.mkdir()
    cfg = coder_runner.resolve_task_config(project)
    backend = _Gated([_headed_call("write_file", path="NOTE.txt", content="hello\n"), "Done."])
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = coder_runner.build_agent(
            backend, project, task="write NOTE.txt", max_turns=cfg.max_turns,
            auto_approve=cfg.auto_approve, always_confirm=cfg.always_confirm,
            session_mode=cfg.session_mode, gen_kw=cfg.gen_kw)
        try:
            result = coder_runner.run_task_with_timeout(agent, "write NOTE.txt", 0.5)
        finally:
            gate.set()
    assert result.timed_out is True
    assert result.denied == (("write_file", "lenient"),)
    assert result.success is False
