# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder agent's HTTP backend must surface a thinking model's reasoning_content
instead of silently dropping it. context.py has no ThinkSplitter downstream, so
reasoning must flow through a SEPARATE channel, never inline <think> tags, in all
three _call_llm dispatch branches: event-sink, interactive terminal, and silent
(non-interactive/sub-agent)."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_agent(tmp_path: Path, on_event=None) -> object:
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
    return agent


def _reasoning_stream_backend(backend, content_pieces, reasoning_pieces):
    """Configure backend.chat_stream to interleave reasoning (via on_reasoning)
    and content pieces, and last_reasoning to reflect the accumulated total -
    mirroring HTTPBackend's real contract."""
    full_reasoning = "".join(reasoning_pieces)

    def fake_chat_stream(messages, on_reasoning=None, **kw):
        for r in reasoning_pieces:
            if on_reasoning is not None:
                on_reasoning(r)
        yield from content_pieces

    backend.chat_stream.side_effect = fake_chat_stream
    backend.last_reasoning = full_reasoning
    return backend


# ---------------------------------------------------------------------------
#  Event-sink mode (GUI/web session)
# ---------------------------------------------------------------------------

class TestCallLLMEventSink:
    def test_reasoning_emitted_as_separate_event_type(self, tmp_path):
        events = []
        agent = _make_agent(tmp_path, on_event=events.append)
        _reasoning_stream_backend(
            agent.backend, ["The ", "answer."], ["because ", "reasons"])

        result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

        assert result == "The answer."
        reasoning_events = [e for e in events if e["type"] == "reasoning"]
        token_events = [e for e in events if e["type"] == "token"]
        assert [e["text"] for e in reasoning_events] == ["because ", "reasons"]
        assert [e["text"] for e in token_events] == ["The ", "answer."]
        # No token event carries reasoning text, and vice versa.
        assert not any("because" in e["text"] for e in token_events)

    def test_reasoning_recorded_in_audit_separately_from_content(self, tmp_path):
        agent = _make_agent(tmp_path, on_event=lambda e: None)
        _reasoning_stream_backend(agent.backend, ["Hi."], ["scratch"])

        agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

        agent._audit.llm.assert_called_once()
        call = agent._audit.llm.call_args
        assert call.args[0] == "Hi."
        assert call.kwargs["reasoning"] == "scratch"

    def test_no_reasoning_emits_no_reasoning_events(self, tmp_path):
        events = []
        agent = _make_agent(tmp_path, on_event=events.append)
        _reasoning_stream_backend(agent.backend, ["plain answer"], [])

        agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

        assert not [e for e in events if e["type"] == "reasoning"]
        call = agent._audit.llm.call_args
        assert call.kwargs["reasoning"] == ""


# ---------------------------------------------------------------------------
#  Interactive terminal mode
# ---------------------------------------------------------------------------

class TestCallLLMInteractive:
    def test_reasoning_printed_dimmed_separately_from_content(self, tmp_path):
        agent = _make_agent(tmp_path)
        _reasoning_stream_backend(
            agent.backend, ["The ", "answer."], ["because ", "reasons"])

        with patch("localm.plugins.coder.agent.context.print_reasoning_token") as mock_reason, \
             patch("localm.plugins.coder.agent.context.print_streaming_token") as mock_token:
            result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        assert result == "The answer."
        reasoning_calls = [c.args[0] for c in mock_reason.call_args_list]
        token_calls = [c.args[0] for c in mock_token.call_args_list]
        assert reasoning_calls == ["because ", "reasons"]
        assert token_calls == ["The ", "answer."]

    def test_interactive_audit_gets_reasoning_field(self, tmp_path):
        agent = _make_agent(tmp_path)
        _reasoning_stream_backend(agent.backend, ["ok"], ["thinking..."])

        with patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_streaming_token"):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        call = agent._audit.llm.call_args
        assert call.args[0] == "ok"
        assert call.kwargs["reasoning"] == "thinking..."

    def test_interrupted_stream_still_records_partial_text(self, tmp_path):
        """A KeyboardInterrupt mid-stream (Ctrl-C at the terminal) must still
        record and return whatever text streamed before the interrupt, not
        lose it: _stream_and_record wraps the whole consume+record sequence in
        a try/except for exactly that."""
        agent = _make_agent(tmp_path)

        def fake_chat_stream(messages, on_reasoning=None, **kw):
            yield "partial "
            raise KeyboardInterrupt()

        agent.backend.chat_stream.side_effect = fake_chat_stream

        with patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_streaming_token"), \
             patch("localm.plugins.coder.agent.context.print_streaming_done") as mock_done, \
             patch("localm.plugins.coder.agent.context.print_info") as mock_info:
            result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        assert result == "partial "
        mock_done.assert_called_once()
        mock_info.assert_called_once_with("(interrupted)")
        agent._audit.llm.assert_called_once()

    def test_event_sink_and_interactive_share_one_streaming_implementation(self, tmp_path):
        """Both branches must call the SAME _stream_and_record method. When the
        event-sink and interactive branches each carry their own copy of the
        consume-and-record loop, a fix lands in only one of them - which is how
        the interactive branch missed the lazy tool-call grammar."""
        from localm.plugins.coder.agent.context import _ContextMixin

        agent = _make_agent(tmp_path, on_event=lambda e: None)
        calls = []
        original = _ContextMixin._stream_and_record

        def _spy(self, *a, **kw):
            calls.append("event_sink" if self.on_event is not None else "interactive")
            return original(self, *a, **kw)

        _reasoning_stream_backend(agent.backend, ["ok"], [])
        with patch.object(_ContextMixin, "_stream_and_record", _spy):
            agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)
        assert calls == ["event_sink"]

        agent2 = _make_agent(tmp_path)
        _reasoning_stream_backend(agent2.backend, ["ok"], [])
        calls.clear()
        with patch.object(_ContextMixin, "_stream_and_record", _spy), \
             patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_streaming_token"):
            agent2._call_llm([{"role": "user", "content": "hi"}], interactive=True)
        assert calls == ["interactive"]

    def test_interactive_response_and_history_never_contain_raw_think_tags(self, tmp_path):
        """The coder loop has no ThinkSplitter downstream (unlike cli/chat.py),
        so the returned response - which agent/loop.py stores verbatim via
        _add_assistant into conversation history and resends to the model -
        must never contain an inlined <think> tag."""
        agent = _make_agent(tmp_path)
        _reasoning_stream_backend(
            agent.backend, ["Visible answer."], ["secret scratchpad"])

        with patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
             patch("localm.plugins.coder.agent.context.print_streaming_token"):
            result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        assert result == "Visible answer."
        assert "<think>" not in result
        assert "secret scratchpad" not in result


# ---------------------------------------------------------------------------
#  Silent mode (sub-agents / non-interactive run_task)
# ---------------------------------------------------------------------------

class TestCallLLMSilent:
    def test_silent_mode_reads_last_reasoning_into_audit(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.backend.chat.return_value = "Done."
        agent.backend.last_reasoning = "silent scratchpad"

        result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

        assert result == "Done."
        assert "<think>" not in result
        assert "silent scratchpad" not in result
        call = agent._audit.llm.call_args
        assert call.args[0] == "Done."
        assert call.kwargs["reasoning"] == "silent scratchpad"

    def test_silent_mode_backend_without_last_reasoning_does_not_crash(self, tmp_path):
        """A backend with no last_reasoning attribute at all (e.g. a minimal
        test double or a future non-HTTP backend) must not break the call."""
        agent = _make_agent(tmp_path)
        del agent.backend.last_reasoning   # MagicMock: remove the auto-attr
        agent.backend.chat.return_value = "Done."

        result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=False)

        assert result == "Done."
        call = agent._audit.llm.call_args
        assert call.kwargs["reasoning"] == ""
