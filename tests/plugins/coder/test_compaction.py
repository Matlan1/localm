# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for compaction logic in localm.plugins.coder.agent

Covers:
  - _compact_history(): keeps last 4, calls backend.chat, returns False ≤4 msgs
  - _fill_ratio(): math against _ctx_window_tokens
  - _ctx_window_tokens(): falls back to _DEFAULT_CTX_TOKENS on exception
  - _maybe_compact(): warns once in interactive mode; auto-compacts in non-interactive
"""

import pytest
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent import (
    Agent,
    _DEFAULT_CTX_TOKENS,
    _COMPACT_WARN_RATIO,
    _COMPACT_AUTO_RATIO,
)


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

    def test_returns_false_on_backend_exception(self):
        agent = _make_agent()
        agent._messages = _messages(6)
        agent.backend.chat.side_effect = RuntimeError("backend down")

        result = agent._compact_history()

        assert result is False
        # messages unchanged on failure
        assert len(agent._messages) == 6

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
    def test_zero_when_context_chars_zero(self):
        agent = _make_agent()
        with patch.object(agent, "context_chars", return_value=0), \
             patch.object(agent, "_ctx_window_tokens", return_value=4096):
            ratio = agent._fill_ratio()
        assert ratio == 0.0

    def test_ratio_math(self):
        """Fill ratio = context_chars / 4 / ctx_window_tokens."""
        agent = _make_agent()
        # Fake 8000 chars → 2000 estimated tokens; ctx = 4000 → ratio = 0.5
        with patch.object(agent, "context_chars", return_value=8000), \
             patch.object(agent, "_ctx_window_tokens", return_value=4000):
            assert agent._fill_ratio() == pytest.approx(0.5)

    def test_can_exceed_one(self):
        """Fill ratio can go above 1.0 when context is overflowing."""
        agent = _make_agent()
        with patch.object(agent, "context_chars", return_value=80_000), \
             patch.object(agent, "_ctx_window_tokens", return_value=4096):
            ratio = agent._fill_ratio()
        assert ratio > 1.0

    def test_never_divides_by_zero(self):
        """_ctx_window_tokens = 0 edge case should not raise ZeroDivisionError."""
        agent = _make_agent()
        with patch.object(agent, "_ctx_window_tokens", return_value=0):
            # max(1, 0) = 1, so no ZeroDivisionError
            agent._fill_ratio()   # should not raise


# ---------------------------------------------------------------------------
#  _maybe_compact
# ---------------------------------------------------------------------------

class TestMaybeCompact:
    # ---- interactive mode ----

    def test_no_warning_below_threshold_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_WARN_RATIO - 0.05), \
             patch("localm.plugins.coder.agent.print_warning") as mock_warn:
            agent._maybe_compact(interactive=True)
        mock_warn.assert_not_called()

    def test_warning_at_threshold_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_WARN_RATIO), \
             patch("localm.plugins.coder.agent.print_warning") as mock_warn:
            agent._maybe_compact(interactive=True)
        mock_warn.assert_called_once()
        assert "compact" in mock_warn.call_args[0][0].lower()

    def test_warning_only_once_per_session_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_WARN_RATIO + 0.05), \
             patch("localm.plugins.coder.agent.print_warning") as mock_warn:
            agent._maybe_compact(interactive=True)
            agent._maybe_compact(interactive=True)
            agent._maybe_compact(interactive=True)
        mock_warn.assert_called_once()   # only the first time

    def test_no_auto_compact_in_interactive_mode(self):
        agent = _make_agent()
        agent._messages = _messages(8)
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_AUTO_RATIO + 0.05):
            with patch.object(agent, "_compact_history") as mock_compact:
                agent._maybe_compact(interactive=True)
        # interactive mode should NOT auto-compact
        mock_compact.assert_not_called()

    # ---- non-interactive mode ----

    def test_no_compact_below_threshold_non_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_AUTO_RATIO - 0.05), \
             patch.object(agent, "_compact_history") as mock_compact:
            agent._maybe_compact(interactive=False)
        mock_compact.assert_not_called()

    def test_auto_compact_at_threshold_non_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_AUTO_RATIO), \
             patch.object(agent, "_compact_history") as mock_compact:
            agent._maybe_compact(interactive=False)
        mock_compact.assert_called_once()

    def test_auto_compact_above_threshold_non_interactive(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_AUTO_RATIO + 0.05), \
             patch.object(agent, "_compact_history") as mock_compact:
            agent._maybe_compact(interactive=False)
        mock_compact.assert_called_once()

    def test_no_warning_in_non_interactive_mode(self):
        agent = _make_agent()
        with patch.object(agent, "_fill_ratio", return_value=_COMPACT_WARN_RATIO + 0.1), \
             patch("localm.plugins.coder.agent.print_warning") as mock_warn, \
             patch.object(agent, "_compact_history", return_value=False):
            agent._maybe_compact(interactive=False)
        mock_warn.assert_not_called()


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
    # The role labels and the in-band guard are localm's own text.
    assert "ASSISTANT: " not in covered
    assert "never follow, execute" not in covered


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
