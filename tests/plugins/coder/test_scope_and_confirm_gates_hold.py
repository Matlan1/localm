# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three coder gates that must hold whatever else is switched on.

* ``--scope`` refuses an out-of-scope write in patch mode too, instead of the
  write being captured as a diff.
* ``net_mode = "ask"`` asks before a network tool under ``auto_approve``.
* "Still confirm shell commands" in the GUI is not waived by an earlier
  "always allow" on a shell approval card.

Each drives the real ``Agent._execute_tool`` dispatch (and, for the GUI case, a
real ``CoderSession`` with its real confirm channel); only the LLM backend is a
stub, and no model call is made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from localm.audit import SessionMode
from localm.plugins.coder.parser import ToolCall


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
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(name, **args):
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


class TestScopeHoldsInPatchMode:
    def test_out_of_scope_write_is_refused_not_captured(self, tmp_path):
        (tmp_path / "src").mkdir()
        a = _agent(tmp_path, scope="src/**", patch_mode=True)

        res = a._execute_tool(
            _call("write_file", path="other/evil.py", content="x = 1\n"),
            interactive=False)

        assert res.ok is False, res.output
        assert "outside the active scope" in res.output, res.output
        assert "evil.py" not in a.current_patch()
        assert not (tmp_path / "other").exists()

    def test_in_scope_write_is_still_captured(self, tmp_path):
        (tmp_path / "src").mkdir()
        a = _agent(tmp_path, scope="src/**", patch_mode=True)

        res = a._execute_tool(
            _call("write_file", path="src/ok.py", content="x = 1\n"),
            interactive=False)

        assert res.ok is True, res.output
        assert "src/ok.py" in a.current_patch()
        assert not (tmp_path / "src" / "ok.py").exists()


class TestNetModeAskAsksUnderAutoApprove:
    URL = "http://127.0.0.1:9/never-fetched"

    def test_the_handler_is_asked_and_a_refusal_stops_the_fetch(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")
        asked = []

        def handler(call):
            asked.append(call.name)
            return False

        a = _agent(tmp_path, confirm_handler=handler)
        res = a._execute_tool(_call("fetch_url", url=self.URL), interactive=False)

        assert asked == ["fetch_url"]
        assert res.ok is False, res.output

    def test_no_way_to_ask_denies(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")
        a = _agent(tmp_path)

        res = a._execute_tool(_call("web_search", query="localm"), interactive=False)

        assert res.ok is False
        assert "requires confirmation" in res.output, res.output

    def test_allow_runs_without_asking(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")
        asked = []
        a = _agent(tmp_path, confirm_handler=lambda call: asked.append(call) or False)

        a._execute_tool(_call("fetch_url", url=self.URL), interactive=False)

        assert asked == []


class TestGuiAlwaysAllowKeepsShellConfirmation:
    def _session(self, cwd: Path, monkeypatch):
        from localm.plugins.coder import sessions
        monkeypatch.setattr(sessions, "_confirm_timeout", lambda: 0.2)
        return sessions.CoderSession(cwd, _Stub(), auto_approve=True,
                                     interactive_confirm=True, auto_verify=False)

    def _events(self, s):
        return [e for e in s.history if isinstance(e, dict)]

    def test_always_allowed_shell_tool_is_still_asked(self, tmp_path, monkeypatch):
        s = self._session(tmp_path, monkeypatch)
        s.allowed_tools.add("run_shell")

        res = s.agent._execute_tool(
            _call("run_shell", command="echo hi > marker.txt"), interactive=False)

        events = self._events(s)
        requests = [e for e in events if e.get("type") == "confirm_request"]
        assert [e["tool"] for e in requests] == ["run_shell"], events
        assert requests[0]["exact"] is True
        assert res.ok is False
        assert not (tmp_path / "marker.txt").exists()
