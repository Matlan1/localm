# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI coder sessions: every open stream gets every event, an unanswered
approval is not called a refusal, plan mode works on cards, search_replace
cards show a diff, the closing line counts this task's files, and the routes
say what is true (busy, missing log, owner only, temperature range)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from localm.plugins.coder.agent.execution import _event_output
from localm.plugins.coder.parser import ToolCall
from tests.test_coder_session_controls import _owner, _scoped, _start, _stub


class _Scripted:
    model_id = "stub-model"
    native_tools = False
    supports_native_tools = True
    supports_grammar = False

    def __init__(self, script):
        self.script = list(script)
        self.last_usage = {"prompt_tokens": 1, "total_tokens": 2}

    def set_tools(self, defs):
        pass

    def _next(self):
        return self.script.pop(0) if self.script else "Done."

    def chat(self, messages, **kw):
        return self._next()

    def chat_stream(self, messages, **kw):
        yield self._next()


def _session(app, client, owner, proj, **extra):
    sid = _start(client, owner, proj, **extra)
    return app.state.coder_sessions.get(sid)


def test_every_subscriber_gets_every_event_and_replay_has_no_duplicates(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        s._push({"type": "info", "text": "before"})
        snap, a = s.subscribe(replay=True)
        _, b = s.subscribe()
        s._push({"type": "info", "text": "after"})
        assert [e.get("text") for e in snap].count("before") == 1
        assert "after" not in [e.get("text") for e in snap]
        assert a.get_nowait()["text"] == "after"
        assert b.get_nowait()["text"] == "after"
        s.unsubscribe(b)
        s._push({"type": "info", "text": "later"})
        assert a.get_nowait()["text"] == "later"
        assert b.empty()


def test_events_pushed_with_no_stream_open_reach_the_next_stream(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        _, first = s.subscribe()
        s.unsubscribe(first)
        s._push({"type": "info", "text": "while away"})
        waiting, again = s.subscribe()
        assert [e.get("text") for e in waiting] == ["while away"]
        s._push({"type": "info", "text": "live"})
        assert again.get_nowait()["text"] == "live"


def test_a_stream_opened_after_the_session_closed_ends(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        _, other = s.subscribe()
        s.close()
        _, late = s.subscribe()
        assert late.get_nowait()["type"] == "closed"


def test_an_old_timeout_never_rewords_a_later_refusal():
    from localm.plugins.coder.confirm import (
        consume_unanswered,
        invoke_confirm,
        mark_unanswered,
    )
    mark_unanswered()
    assert invoke_confirm(lambda call: False, object()) is False
    assert consume_unanswered() is False


def test_plan_mode_can_be_turned_off_from_the_settings_route(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"plan_mode": True})
        assert r.status_code == 200 and r.json()["plan_mode"] is True
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"plan_mode": False})
        assert r.status_code == 200 and r.json()["plan_mode"] is False


def test_the_gui_plan_note_does_not_offer_ask_user(tmp_path, monkeypatch):
    from localm.plugins.coder.tools.interact import tool_enter_plan_mode
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        res = tool_enter_plan_mode(proj, _session=s.agent, _interactive=False)
        assert "plan mode" in res.output and "ask_user" not in res.output


def test_an_unanswered_approval_is_not_reported_as_a_refusal(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    monkeypatch.setattr("localm.plugins.coder.sessions._confirm_timeout", lambda: 0.01)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        call = {"name": "write_file", "args": {"path": "a.txt", "content": "x"}}
        s.agent.backend = _Scripted(["<tool_call>" + json.dumps(call) + "</tool_call>"])
        s.agent.run_task("write it")
        texts = [m.get("content", "") for m in s.agent._messages if m["role"] == "user"]
        joined = "\n".join(str(t) for t in texts)
        assert "nobody answered the approval request" in joined
        assert "Rejected by user" not in joined
        infos = [e for e in s.history if e.get("type") == "info"
                 and "timed out" in str(e.get("text", ""))]
        assert infos == []
        resolved = [e for e in s.history if e.get("type") == "confirm_resolved"]
        assert resolved and resolved[-1]["timed_out"] is True
        assert not (proj / "a.txt").exists()


def test_plan_mode_tools_are_on_and_ask_user_is_off(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        assert "exit_plan_mode" not in s.agent.disabled_tools
        assert "enter_plan_mode" not in s.agent.disabled_tools
        assert "ask_user" in s.agent.disabled_tools


def test_a_search_replace_card_shows_the_diff(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    (proj / "a.py").write_text("x = 1\n", encoding="utf-8")
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        diff = s._diff_preview(ToolCall(name="search_replace",
                                        args={"pattern": "x = 1", "replacement": "x = 2",
                                              "glob": "**/*.py"}))
        assert diff and "+x = 2" in diff
        assert (proj / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_the_closing_line_counts_only_this_tasks_files(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj, auto_approve=True)

        def task(path):
            call = {"name": "write_file", "args": {"path": path, "content": "x"}}
            s.agent.backend = _Scripted(["<tool_call>" + json.dumps(call) + "</tool_call>"])
            assert s.send_message(f"write {path}") == "started"
            import time
            deadline = time.time() + 20
            while s.busy and time.time() < deadline:
                time.sleep(0.02)
            finals = [e for e in s.history if e.get("type") == "final"]
            return finals[-1]["changed_files"]

        assert task("a.txt") == ["a.txt"]
        assert task("b.txt") == ["b.txt"]


def test_remember_says_whether_it_added(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        assert s.remember("use tabs")["added"] is True
        assert s.remember("use tabs")["added"] is False


def test_a_disabled_tool_leaves_a_result_event(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        s = _session(app, client, owner, proj)
        s.agent._execute_tool(ToolCall(name="ask_user", args={"question": "?"}),
                              interactive=False)
        results = [e for e in s.history if e.get("type") == "tool_result"
                   and e.get("tool") == "ask_user"]
        assert results and results[-1]["summary"] == "not run: disabled for this session"


def test_a_long_result_event_keeps_its_tail():
    text = "A" * 9000 + "[file truncated: read_output o1]"
    out = _event_output(text)
    assert len(out) < 4100 and out.endswith("[file truncated: read_output o1]")
    assert _event_output("short") == "short"


class TestRoutes:
    def test_temperature_out_of_range_is_refused_with_its_field(self, tmp_path, monkeypatch):
        app, proj, owner = _owner(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r = client.post("/api/coder/sessions", headers=owner,
                            json={"cwd": str(proj), "temperature": 3})
            assert r.status_code == 422
            assert "temperature" in r.text

    def test_undo_while_busy_says_busy_and_idle_says_nothing(self, tmp_path, monkeypatch):
        app, proj, owner = _owner(tmp_path, monkeypatch)
        with TestClient(app) as client:
            sid = _start(client, owner, proj)
            s = _stub(app, sid)
            r = client.post(f"/api/coder/sessions/{sid}/undo", headers=owner)
            assert r.status_code == 409 and r.json()["detail"] == "Nothing to undo"
            s.busy = True
            r = client.post(f"/api/coder/sessions/{sid}/undo", headers=owner)
            assert r.status_code == 409 and "working" in r.json()["detail"]
            s.busy = False

    def test_a_missing_log_file_is_not_blamed_on_privacy_mode(self, tmp_path, monkeypatch):
        app, proj, owner = _owner(tmp_path, monkeypatch)
        with TestClient(app) as client:
            sid = _start(client, owner, proj, mode="log")
            s = app.state.coder_sessions.get(sid)
            monkeypatch.setattr(s, "audit_log_path", lambda: tmp_path / "gone.jsonl")
            r = client.get(f"/api/coder/sessions/{sid}/log", headers=owner)
            assert r.status_code == 404
            assert "privacy" not in r.json()["detail"]
            assert "not on disk" in r.json()["detail"]

    def test_a_non_owner_is_told_it_is_not_authorized(self, tmp_path, monkeypatch):
        app, proj, owner = _owner(tmp_path, monkeypatch)
        with TestClient(app) as client:
            shared = _scoped(app)
            r = client.get("/api/coder/episodes", headers=shared,
                           params={"cwd": str(proj)})
            assert r.status_code == 200 and r.json()["authorized"] is False
            r = client.get("/api/coder/hooks", headers=shared, params={"cwd": str(proj)})
            assert r.status_code == 200 and r.json()["authorized"] is False
