# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder shows what the model is doing before it answers: every backend
relays statuses as ``on_status(text, code)``, the GUI session receives them
as ``status`` events, and the interactive terminal prints each one on its own
line before the assistant label."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import LOADING_MODEL_STATUS
from localm.plugins.coder.backends.http import CoderServerError, HTTPBackend


def _stream_response(chunks):
    lines = [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"data: [DONE]"]
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.__enter__ = lambda self: self
    resp.__exit__ = MagicMock(return_value=False)
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = lines
    return resp


def _delta(**d):
    return {"choices": [{"delta": d, "finish_reason": None}]}


STOP = {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class TestHTTPBackend:
    @patch("requests.post")
    def test_statuses_reach_on_status_and_are_never_yielded(self, mock_post):
        mock_post.return_value = _stream_response([
            _delta(status="Loading model...", status_code="loading_model"),
            _delta(status="Processing prompt...", status_code="processing"),
            _delta(content="hello"), STOP])
        seen = []
        backend = HTTPBackend("http://127.0.0.1:8080/v1", "m")
        pieces = list(backend.chat_stream(
            [{"role": "user", "content": "hi"}],
            on_status=lambda text, code: seen.append((text, code))))
        assert pieces == ["hello"]
        assert seen == [("Loading model...", "loading_model"),
                        ("Processing prompt...", "processing")]

    @patch("requests.post")
    def test_on_status_is_not_sent_to_the_server(self, mock_post):
        mock_post.return_value = _stream_response([_delta(content="x"), STOP])
        backend = HTTPBackend("http://127.0.0.1:8080/v1", "m")
        list(backend.chat_stream([{"role": "user", "content": "hi"}],
                                 on_status=lambda t, c: None))
        body = mock_post.call_args.kwargs["json"]
        assert "on_status" not in body

    @patch("requests.post")
    def test_routing_in_a_headers_chunk_sets_the_routing_note(self, mock_post):
        routing = {"resolved": "a", "requested": "a", "routed": False,
                   "pinned": False, "note": "b was skipped: its last load failed"}
        mock_post.return_value = _stream_response([
            {"choices": [{"delta": {}, "finish_reason": None}],
             "localm_headers": {"X-Localm-Model-Routing": json.dumps(routing)}},
            _delta(content="x"), STOP])
        backend = HTTPBackend("http://127.0.0.1:8080/v1", "m")
        list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        assert backend.routing_note == "b was skipped: its last load failed"

    @patch("requests.post")
    def test_an_in_stream_refusal_raises_like_the_http_error(self, mock_post):
        detail = "Grammar refused: would be ignored"
        mock_post.return_value = _stream_response([
            _delta(content=detail),
            {"choices": [{"delta": {}, "finish_reason": "error"}],
             "localm_error": {"status": 400, "detail": detail}}])
        backend = HTTPBackend("http://127.0.0.1:8080/v1", "m")
        with pytest.raises(CoderServerError) as caught:
            list(backend.chat_stream([{"role": "user", "content": "hi"}]))
        assert str(caught.value).startswith("HTTP 400 error from ")
        assert str(caught.value).endswith(": " + detail)


class TestEngineBackends:
    def test_local_engine_relays_engine_statuses_with_their_codes(self):
        from localm.plugins.coder.backends.local_engine import LocalEngineBackend
        engine = MagicMock()

        def _chat_stream(messages, on_status=None, **kw):
            on_status(LOADING_MODEL_STATUS)
            on_status("Something new...")
            yield "ok"

        engine.chat_stream.side_effect = _chat_stream
        backend = LocalEngineBackend.__new__(LocalEngineBackend)
        backend._engine = engine
        backend._loaded = True
        backend._last_reasoning = ""
        backend._ensure_loaded = lambda: None
        seen = []
        out = list(backend.chat_stream([{"role": "user", "content": "hi"}],
                                       on_status=lambda t, c: seen.append((t, c))))
        assert out == ["ok"]
        assert seen == [(LOADING_MODEL_STATUS, "loading_model"), ("Something new...", None)]


def _make_agent(tmp_path: Path, on_event=None):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    backend.supports_grammar = False
    backend.last_usage = {}
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, on_event=on_event)
    agent._audit = MagicMock()

    def fake_chat_stream(messages, on_reasoning=None, on_status=None, **kw):
        if on_status is not None:
            on_status("Loading model...", "loading_model")
            on_status("Processing prompt...", "processing")
        yield "The "
        if on_status is not None:
            on_status("Generating response...", "generating")
        yield "answer."

    backend.chat_stream.side_effect = fake_chat_stream
    return agent


class TestAgent:
    def test_a_gui_session_receives_status_events(self, tmp_path):
        events = []
        agent = _make_agent(tmp_path, on_event=events.append)
        assert agent._call_llm([{"role": "user", "content": "hi"}],
                               interactive=False) == "The answer."
        statuses = [(e["text"], e["code"]) for e in events if e["type"] == "status"]
        assert statuses[:2] == [("Loading model...", "loading_model"),
                                ("Processing prompt...", "processing")]
        first_status = next(i for i, e in enumerate(events) if e["type"] == "status")
        first_token = next(i for i, e in enumerate(events) if e["type"] == "token")
        assert first_status < first_token

    def test_the_terminal_prints_statuses_then_the_label_once(self, tmp_path):
        from localm.plugins.coder.agent import context
        out = []
        agent = _make_agent(tmp_path)
        with patch.object(context, "print_thinking", lambda *a: out.append("thinking")), \
             patch.object(context, "print_status", lambda t: out.append(f"status:{t}")), \
             patch.object(context, "print_assistant_label", lambda n: out.append("label")), \
             patch.object(context, "print_streaming_token", lambda p: out.append(f"tok:{p}")), \
             patch.object(context, "print_streaming_done", lambda: out.append("done")):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)
        assert out == ["thinking", "status:Loading model...",
                       "status:Processing prompt...", "label",
                       "tok:The ", "tok:answer.", "done"]


class TestOneShotProgress:
    def test_a_run_reporting_progress_prints_each_new_model_status(self, tmp_path):
        from localm.plugins.coder.agent import context
        out = []
        agent = _make_agent(tmp_path)
        agent.report_progress = True
        with patch.object(context, "print_progress", out.append):
            result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
        assert result == "The answer."
        assert out == ["Waiting for test-model...", "Loading model...",
                       "Processing prompt...", "Generating response..."]
        agent.backend.chat.assert_not_called()

    def test_a_silent_run_prints_nothing_and_does_not_stream(self, tmp_path):
        from localm.plugins.coder.agent import context
        out = []
        agent = _make_agent(tmp_path)
        agent.backend.chat.return_value = "quiet"
        with patch.object(context, "print_progress", out.append):
            assert agent._call_llm([{"role": "user", "content": "hi"}],
                                   interactive=False) == "quiet"
        assert out == []
        agent.backend.chat_stream.assert_not_called()

    def test_a_run_reporting_progress_prints_each_tool_call(self, tmp_path):
        from localm.plugins.coder.agent import execution
        from localm.plugins.coder.parser import ToolCall
        from localm.plugins.coder.tools import ToolResult
        calls = []
        agent = _make_agent(tmp_path)
        agent.report_progress = True
        tool_def = MagicMock()
        tool_def.destructive = False
        tool_def.fn = MagicMock(return_value=ToolResult.success("ok"))
        call = ToolCall(name="read_file", args={"path": "a.py"}, raw="", start=0, end=0)
        with patch.dict("localm.plugins.coder.agent.TOOL_REGISTRY", {"read_file": tool_def}), \
             patch.object(execution, "print_progress_tool_call",
                          lambda name, args: calls.append((name, args))):
            agent._execute_tool(call, interactive=False)
        assert calls == [("read_file", {"path": "a.py"})]
