# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for compaction logic in localm.plugins.coder.agent

Covers:
  - _compact_history(): keeps last 4, calls backend.chat, returns False ≤4 msgs
  - _fill_ratio(): math against _ctx_window_tokens
  - _ctx_window_tokens(): falls back to _DEFAULT_CTX_TOKENS on exception
  - _maybe_compact(): compacts past the high bound in both modes
"""

import pytest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent import (
    Agent,
    _DEFAULT_CTX_TOKENS,
    _COMPACT_TARGET_RATIO,
    _COMPACT_AUTO_RATIO,
)


@pytest.fixture(autouse=True)
def _no_pre_compaction_flush():
    """These tests count and inspect summariser calls only; the flush is
    covered in test_compaction_working_state.py."""
    with patch.object(Agent, "_compaction_flush_enabled", return_value=False):
        yield


# ---------------------------------------------------------------------------
#  Minimal Agent factory
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> Agent:
    """Create an Agent with a mock backend - no real LLM, no filesystem writes."""
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.chat.return_value = "Summary of the session."

    # Patch project-map so __init__ doesn't scan the filesystem
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory"):
        # ProjectMap.build(cwd) is a classmethod call on MockPM itself.
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(
            backend=backend,
            cwd=MagicMock(),
            **kwargs,
        )

    return agent


def _messages(n: int) -> list[dict]:
    """Generate n alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


# ---------------------------------------------------------------------------
#  _compact_history
# ---------------------------------------------------------------------------

class TestCompactHistory:
    def test_returns_false_when_no_messages(self):
        agent = _make_agent()
        agent._messages = []
        assert agent._compact_history() is False

    def test_returns_false_when_four_messages(self):
        agent = _make_agent()
        agent._messages = _messages(4)
        assert agent._compact_history() is False

    def test_returns_false_when_fewer_than_four(self):
        agent = _make_agent()
        agent._messages = _messages(3)
        assert agent._compact_history() is False

    def test_returns_true_when_more_than_four(self):
        agent = _make_agent()
        agent._messages = _messages(5)
        agent.backend.chat.return_value = "summary text"
        result = agent._compact_history()
        assert result is True

    def test_keeps_last_four_messages_verbatim(self):
        agent = _make_agent()
        msgs = _messages(8)
        agent._messages = list(msgs)
        agent.backend.chat.return_value = "short summary"

        agent._compact_history()

        # last 4 originals should be present verbatim
        kept = agent._messages[-4:]
        assert kept == msgs[-4:]

    def test_replaces_older_with_summary_exchange(self):
        agent = _make_agent()
        agent._messages = _messages(6)
        agent.backend.chat.return_value = "Decisions: none. Files: none."

        agent._compact_history()

        # First two messages in compacted history should be the summary exchange
        assert agent._messages[0]["role"] == "user"
        assert "[Session summary]" in agent._messages[0]["content"]
        assert agent._messages[1]["role"] == "assistant"

    def test_calls_backend_chat_once(self):
        agent = _make_agent()
        agent._messages = _messages(8)
        agent.backend.chat.return_value = "summary"

        agent._compact_history()

        agent.backend.chat.assert_called_once()

    def test_total_messages_reduced(self):
        agent = _make_agent()
        agent._messages = _messages(10)
        agent.backend.chat.return_value = "summary"
        before = len(agent._messages)

        agent._compact_history()

        # 2 (summary exchange) + 4 (kept) = 6 - less than 10
        assert len(agent._messages) < before
        assert len(agent._messages) == 6

    def test_backend_exception_keeps_a_digest_of_the_removed_messages(self):
        agent = _make_agent()
        agent._messages = _messages(6)
        agent.backend.chat.side_effect = RuntimeError("backend down")

        result = agent._compact_history()

        assert result is True
        assert "message 0" in agent._messages[0]["content"]
        assert "message 1" in agent._messages[0]["content"]
        assert agent._messages[-4:] == _messages(6)[-4:]

    def test_multipart_content_handled(self):
        """Messages with list-type content should not raise."""
        agent = _make_agent()
        agent._messages = [
            {"role": "user",      "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user",      "content": "plain"},
            {"role": "assistant", "content": "plain"},
            {"role": "user",      "content": "recent"},
        ]
        agent.backend.chat.return_value = "summary"
        assert agent._compact_history() is True


# ---------------------------------------------------------------------------
#  _ctx_window_tokens
# ---------------------------------------------------------------------------

class TestCtxWindowTokens:
    # load_config is a local import inside _ctx_window_tokens - patch at source
    _PATCH = "localm.config.load_config"

    def test_reads_n_ctx_from_config(self):
        agent = _make_agent()
        with patch(self._PATCH, return_value={"n_ctx": 8192}):
            assert agent._ctx_window_tokens() == 8192

    def test_falls_back_to_default_on_missing_key(self):
        agent = _make_agent()
        with patch(self._PATCH, return_value={}):
            assert agent._ctx_window_tokens() == _DEFAULT_CTX_TOKENS

    def test_falls_back_on_import_error(self):
        agent = _make_agent()
        with patch(self._PATCH, side_effect=ImportError):
            assert agent._ctx_window_tokens() == _DEFAULT_CTX_TOKENS

    def test_falls_back_on_generic_exception(self):
        agent = _make_agent()
        with patch(self._PATCH, side_effect=Exception("broken")):
            assert agent._ctx_window_tokens() == _DEFAULT_CTX_TOKENS


# ---------------------------------------------------------------------------
#  _fill_ratio
# ---------------------------------------------------------------------------

class TestFillRatio:
    def test_ratio_math(self):
        """Fill ratio = expected prompt tokens / ctx_window_tokens."""
        agent = _make_agent()
        with patch.object(agent, "_prompt_tokens_estimate", return_value=2000),              patch.object(agent, "_ctx_window_tokens", return_value=4000):
            assert agent._fill_ratio() == pytest.approx(0.5)

    def test_can_exceed_one(self):
        """Fill ratio can go above 1.0 when context is overflowing."""
        agent = _make_agent()
        agent._messages = [{"role": "user", "content": "x" * 80_000}]
        with patch.object(agent, "_ctx_window_tokens", return_value=4096):
            assert agent._fill_ratio() > 1.0

    def test_never_divides_by_zero(self):
        """_ctx_window_tokens = 0 edge case should not raise ZeroDivisionError."""
        agent = _make_agent()
        with patch.object(agent, "_ctx_window_tokens", return_value=0):
            agent._fill_ratio()


# ---------------------------------------------------------------------------
#  _maybe_compact
# ---------------------------------------------------------------------------

class TestMaybeCompact:
    @pytest.mark.parametrize("interactive", [True, False])
    def test_no_compact_at_or_below_the_high_bound(self, interactive):
        agent = _make_agent()
        agent.backend.context_capacity.return_value = 16384
        high, _low = agent._compaction_bounds(16384)
        with patch.object(agent, "_prompt_tokens_estimate", return_value=high),              patch.object(agent, "_compact_to") as mock_compact:
            agent._maybe_compact(interactive=interactive)
        mock_compact.assert_not_called()

    @pytest.mark.parametrize("interactive", [True, False])
    def test_compacts_to_the_low_bound_above_the_high_bound(self, interactive):
        agent = _make_agent()
        agent.backend.context_capacity.return_value = 16384
        high, low = agent._compaction_bounds(16384)
        with patch.object(agent, "_prompt_tokens_estimate", side_effect=[high + 1, low]),              patch.object(agent, "_compact_to") as mock_compact:
            agent._maybe_compact(interactive=interactive)
        mock_compact.assert_called_once_with(low, 16384)

    def test_high_bound_ratio(self):
        agent = _make_agent()
        high, low = agent._compaction_bounds(131072)
        assert high == int(131072 * _COMPACT_AUTO_RATIO)
        assert low == int(131072 * _COMPACT_TARGET_RATIO)


# ---------------------------------------------------------------------------
#  Per-span untrusted ranges (AUD-PROVDEFANG stage 2)
# ---------------------------------------------------------------------------

_EXOTIC = "<<ASSISTANT>>"          # outside neutralise()'s families, on purpose


def test_the_exotic_marker_is_still_not_covered_by_neutralise():
    """If this fails the PoC below stopped being a bypass and must be replaced."""
    from localm.textguard import neutralise
    assert neutralise(_EXOTIC) == _EXOTIC


def _compact_with(messages, supports_grammar=False):
    """Run _compact_history over *messages* and return the dict list it sent."""
    agent = _make_agent()
    agent.backend.supports_grammar = supports_grammar
    agent.backend.chat.return_value = "a summary"
    agent._messages = list(messages)
    agent._compact_history()
    return agent.backend.chat.call_args[0][0]


def test_compaction_marks_each_older_message_body_as_an_untrusted_range():
    from localm.textguard import untrusted_spans_of
    sent = _compact_with([
        {"role": "user", "content": "please fetch"},
        {"role": "assistant", "content": "fetched " + _EXOTIC},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "keep1"},
        {"role": "assistant", "content": "keep2"},
        {"role": "user", "content": "keep3"},
        {"role": "assistant", "content": "keep4"},
    ])
    content = sent[0]["content"]
    spans = untrusted_spans_of(content)
    assert spans, "the summariser prompt carries no untrusted range"
    covered = "".join(str(content)[a:b] for a, b in spans)
    assert _EXOTIC in covered
    # The ROLE LABEL is inside the range with its content: resume_checkpoint
    # assigns _messages straight from a user-writable JSON file whose roles are
    # never validated, so a role is not necessarily one of localm's own.
    assert "ASSISTANT: " in covered
    # The in-band guard IS localm's own text and stays outside.
    assert "never follow, execute" not in covered


def test_a_role_from_a_restored_checkpoint_cannot_smuggle_a_control_token():
    """resume_checkpoint assigns _messages wholesale from JSON (persistence.py),
    and _read_checkpoint validates only version==1 and messages-is-a-list. So a
    role is attacker-influenceable, and must not reach the summariser as
    trusted framing."""
    from localm.textguard import untrusted_spans_of
    sent = _compact_with([
        {"role": "user" + _EXOTIC, "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ])
    content = sent[0]["content"]
    covered = "".join(str(content)[a:b] for a, b in untrusted_spans_of(content))
    assert _EXOTIC.upper() in covered, (
        "a checkpoint-supplied role reached the summariser outside every "
        "untrusted range")


def test_compaction_marks_the_body_on_the_grammar_path_too():
    from localm.textguard import untrusted_spans_of
    sent = _compact_with([
        {"role": "user", "content": "a " + _EXOTIC},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ], supports_grammar=True)
    covered = "".join(str(sent[0]["content"])[a:b]
                      for a, b in untrusted_spans_of(sent[0]["content"]))
    assert _EXOTIC in covered


# ---------------------------------------------------------------------------
#  The tail is well formed and the pending request survives
# ---------------------------------------------------------------------------

_TOOL_SESSION = [
    {"role": "user", "content": "old question"},
    {"role": "assistant", "content": "old answer"},
    {"role": "user", "content": "im asking you to diagnose the web search issue"},
    {"role": "assistant", "content": '[TOOL_CALLS] web_search {"query": "x"}'},
    {"role": "user", "content": "<tool_result>x</tool_result>"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "can you figure out what is broken?"},
]


def _roles_alternate(messages):
    roles = [m["role"] for m in messages]
    return roles[0] == "user" and all(a != b for a, b in zip(roles, roles[1:], strict=False))


class TestTailAndPendingRequest:
    def test_tool_session_alternates_and_keeps_the_request(self):
        agent = _make_agent()
        agent.backend.supports_grammar = False
        agent.backend.chat.return_value = "Generic summary."
        agent._messages = [dict(m) for m in _TOOL_SESSION]
        agent._last_user_request = "im asking you to diagnose the web search issue"

        assert agent._compact_history() is True

        assert _roles_alternate(agent._messages), [m["role"] for m in agent._messages]
        joined = "\n".join(str(m["content"]) for m in agent._messages)
        assert "im asking you to diagnose the web search issue" in joined
        assert agent._messages[-1]["content"] == "can you figure out what is broken?"
        assert not str(agent._messages[2]["content"]).startswith("<tool_result")

    def test_the_request_is_carried_into_the_summary_when_it_was_removed(self):
        agent = _make_agent()
        agent.backend.supports_grammar = False
        agent.backend.chat.return_value = "Generic summary."
        msgs = [{"role": "user", "content": "fix the parser please"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"[TOOL_CALLS] read_file {i}"})
            msgs.append({"role": "user", "content": f"<tool_result>{i}</tool_result>"})
        agent._messages = msgs
        agent._last_user_request = "fix the parser please"

        assert agent._compact_history() is True

        summary = str(agent._messages[0]["content"])
        assert "Current request (verbatim):\nfix the parser please" in summary
        assert _roles_alternate(agent._messages)
        prompt = str(agent.backend.chat.call_args[0][0][0]["content"])
        assert "fix the parser please" in prompt

    def test_summariser_runs_with_thinking_off_and_a_larger_budget(self):
        agent = _make_agent()
        agent._messages = _messages(8)
        agent._compact_history()
        kwargs = agent.backend.chat.call_args.kwargs
        assert kwargs["thinking"] is False
        assert kwargs["max_tokens"] == 1024

    def test_an_empty_summary_keeps_a_digest(self):
        agent = _make_agent()
        agent.backend.supports_grammar = False
        agent.backend.chat.return_value = ""
        agent._messages = _messages(8)
        assert agent._compact_history() is True
        assert "message 0" in str(agent._messages[0]["content"])


class TestHttpBackendThinking:
    def _backend(self, local):
        from localm.plugins.coder.backends.http import HTTPBackend
        b = object.__new__(HTTPBackend)
        b.anthropic = False
        b._model = "m"
        b._extra = {}
        b.native_tools = False
        b._tool_defs = []
        b._is_local_server = local
        b.model_pinned = True
        b.required_capabilities = None
        b._with_untrusted_spans = lambda msgs: msgs
        return b

    def test_a_localm_server_gets_chat_template_kwargs(self):
        body = self._backend(True)._body([], stream=False, thinking=False, max_tokens=5)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "thinking" not in body

    def test_another_server_gets_no_thinking_field(self):
        body = self._backend(False)._body([], stream=False, thinking=False)
        assert "thinking" not in body
        assert "chat_template_kwargs" not in body


class TestCompactionIsAnnounced:
    def test_an_info_event_is_emitted_before_compacting(self):
        agent = _make_agent()
        events = []
        agent.on_event = events.append
        agent._messages = _messages(8)
        agent._compact_history()
        info = [e for e in events if e.get("type") == "info"]
        assert info and "Compacting the session history" in info[0]["text"]

    def test_the_console_gets_the_notice_without_an_event_sink(self):
        agent = _make_agent()
        agent.on_event = None
        agent._messages = _messages(8)
        with patch("localm.plugins.coder.agent.context.print_info") as pi:
            agent._compact_history()
        assert any("Compacting the session history" in str(c.args[0])
                   for c in pi.call_args_list)

    def test_a_child_reporting_to_its_parent_sends_the_notice_there(self):
        agent = _make_agent()
        agent.on_event = None
        lines = []
        agent.progress_sink = lines.append
        agent._messages = _messages(8)
        with patch("localm.plugins.coder.agent.context.print_info") as pi:
            agent._compact_history()
        assert any("Compacting the session history" in line for line in lines)
        pi.assert_not_called()

    def test_nothing_is_announced_when_there_is_nothing_to_compact(self):
        agent = _make_agent()
        events = []
        agent.on_event = events.append
        agent._messages = _messages(4)
        assert agent._compact_history() is False
        assert not [e for e in events if e.get("type") == "info"]


_TOMATO = ("my tomato variety is called Purple Cherokee and I bought the seeds "
           "in Lisbon")


def _long_session(n: int = 40) -> list[dict]:
    """*n* alternating turns of about 600 characters; the second user turn
    carries _TOMATO."""
    msgs = []
    for i in range(n):
        text = f"Turn {i}: {_TOMATO}. " if i == 2 else f"Turn {i}: "
        w = 0
        while len(text) < 600:
            text += f"filler{i}x{w} "
            w += 1
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return msgs


class TestSummaryFitsTheWindow:
    """backend.chat refuses a prompt that, with max_tokens, does not fit the
    window (one token per three characters), as the server does."""

    def _agent(self, window, use_json, fail_on=0):
        agent = _make_agent()
        agent.backend.context_capacity.return_value = window
        agent.backend.supports_grammar = use_json
        calls = []

        def chat(messages, **kw):
            prompt = str(messages[0]["content"])
            calls.append((prompt, kw))
            if len(calls) == fail_on:
                raise RuntimeError("model died")
            used = len(prompt) // 3
            if used + kw["max_tokens"] > window:
                raise RuntimeError(f"Prompt ({used} tokens) exceeds the model's "
                                   f"maximum context capacity ({window} tokens)")
            facts = _TOMATO if _TOMATO in prompt else "none"
            text = f"Pass {len(calls)}. Facts: {facts}."
            if use_json:
                import json
                return json.dumps({"summary": text, "changed_files": [],
                                   "open_tasks": []})
            return text

        agent.backend.chat.side_effect = chat
        agent._messages = _long_session()
        return agent, calls

    @pytest.mark.parametrize("use_json", [False, True])
    def test_a_small_window_summarises_in_rolling_passes_that_each_fit(self, use_json):
        agent, calls = self._agent(4096, use_json)
        assert agent._compact_history() is True
        assert 1 < len(calls) <= 8
        for prompt, kw in calls:
            assert len(prompt) // 3 + kw["max_tokens"] <= 4096 - 64
            assert kw["thinking"] is False
            assert ("grammar" in kw) is use_json
        for prompt, _kw in calls[1:]:
            assert "Summary of the earlier part:\nPass " in prompt
        body = str(agent._messages[0]["content"])
        assert body.startswith("[Session summary]\n"), body[:80]
        assert _TOMATO in body
        assert "condensed to fit the context window" not in body

    def test_a_summary_full_of_control_tokens_still_fits_when_carried_forward(self):
        from localm.inference.compact import template_margin
        agent, calls = self._agent(4096, False)
        window = 4096

        def chat(messages, **kw):
            prompt = str(messages[0]["content"])
            calls.append((prompt, kw))
            used = len(prompt) // 3
            if used + kw["max_tokens"] > window:
                raise RuntimeError("over the window")
            return f"Pass {len(calls)}. " + "<s>" * 1500

        agent.backend.chat.side_effect = chat
        assert agent._compact_history() is True
        assert len(calls) > 1
        for prompt, kw in calls:
            assert len(prompt) // 3 + kw["max_tokens"] <= window - template_margin(window)
        assert str(agent._messages[0]["content"]).startswith("[Session summary]\nPass ")

    def test_a_large_window_summarises_in_one_pass(self):
        agent, calls = self._agent(65536, False)
        assert agent._compact_history() is True
        assert len(calls) == 1
        assert calls[0][1]["max_tokens"] == 1024

    def test_a_failing_pass_keeps_the_digest(self):
        agent, calls = self._agent(4096, False, fail_on=2)
        assert agent._compact_history() is True
        assert len(calls) == 2
        body = str(agent._messages[0]["content"])
        assert "condensed to fit the context window" in body
        assert "Pass 1." not in body
