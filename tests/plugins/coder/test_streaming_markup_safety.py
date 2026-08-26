# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model's streamed text can contain an unmatched Rich markup closing tag -
a leaked <tool_call>-style control token such as [/INST], or plain [/b]/
[/code]/[/s] - which raised MarkupError out of a bare console.print. Nothing
in _stream_and_record's loop caught anything but KeyboardInterrupt, so the
error escaped the turn entirely, skipping the audit record. Worse, the outer
handler's own print_error(f"Agent error: {e}") re-embedded the offending text
and raised AGAIN from its own message - uncaught, that second raise is what
killed the whole REPL process for one leaked control token.

Also covers the two siblings at the same site: a raw ANSI escape sequence
passing through unstripped (screen-clear injection), and a markdown link's
label being silently eaten by the unguarded markup parser.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder import display


def _make_agent(tmp_path: Path) -> object:
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
        agent = Agent(backend=backend, cwd=tmp_path)
    agent._audit = MagicMock()
    return agent


# ---------------------------------------------------------------------------
#  display.py: every externally-controlled print survives hostile content
# ---------------------------------------------------------------------------

HOSTILE_TAGS = ["[/INST]", "[/b]", "[/code]", "[/s]"]


class TestDisplayFunctionsSurviveHostileText:
    @pytest.mark.parametrize("tag", HOSTILE_TAGS)
    def test_streaming_token_survives_a_leaked_control_token(self, tag, capsys):
        display.print_streaming_token(f"leaked: {tag} more text")
        out = capsys.readouterr().out
        assert tag in out
        assert "leaked:" in out

    def test_reasoning_token_survives_a_leaked_control_token(self, capsys):
        display.print_reasoning_token("thinking [/INST] aloud")
        out = capsys.readouterr().out
        assert "[/INST]" in out

    def test_print_error_survives_a_message_that_would_itself_crash_markup(self, capsys):
        """THE DOUBLE-FAULT. print_error is what the outer handler calls with
        an exception's own text - which can quote the exact hostile content
        that raised in the first place. It must not raise on its own message."""
        display.print_error("Agent error: closing tag '[/INST]' at position 14 "
                             "doesn't match any open tag")
        out = capsys.readouterr().out
        assert "[/INST]" in out

    def test_print_info_warning_success_survive_hostile_text(self, capsys):
        for fn in (display.print_info, display.print_warning, display.print_success):
            fn("status: [/INST]")
        out = capsys.readouterr().out
        assert out.count("[/INST]") == 3

    def test_print_assistant_response_survives_hostile_text(self, capsys):
        display.print_assistant_response("plain answer with [/INST] inside")
        out = capsys.readouterr().out
        assert "[/INST]" in out

    def test_print_tool_call_survives_hostile_args(self, capsys):
        display.print_tool_call("write_file", {"path": "[/INST]evil.txt"})
        out = capsys.readouterr().out
        assert "[/INST]evil.txt" in out
        assert "write_file" in out

    def test_print_tool_result_survives_hostile_summary_and_output(self, capsys):
        result = MagicMock(ok=True, truncated=False,
                           summary="wrote [/INST] file", output="line [/INST] one")
        display.print_tool_result("write_file", result, verbose=True)
        out = capsys.readouterr().out
        assert "[/INST]" in out

    def test_print_tool_error_survives_hostile_message(self, capsys):
        display.print_tool_error("run_shell", "failed: [/INST] denied")
        out = capsys.readouterr().out
        assert "[/INST]" in out

    def test_confirm_diff_survives_a_hostile_path_label(self, capsys, monkeypatch):
        # Patch the builtin input() Console.input() falls through to, NOT
        # console.input itself - it must still render the real prompt (where
        # the crash would occur), only the blocking stdin read is stubbed.
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert display.confirm_diff("[/INST]evil.txt") is False
        out = capsys.readouterr().out
        assert "[/INST]evil.txt" in out


# ---------------------------------------------------------------------------
#  ANSI escape stripping
# ---------------------------------------------------------------------------

class TestAnsiEscapeStripped:
    def test_raw_esc_byte_is_stripped_from_streamed_tokens(self, capsys):
        hostile = "before\x1b[2Jafter"   # a real screen-clear CSI sequence
        display.print_streaming_token(hostile)
        out = capsys.readouterr().out
        assert "\x1b" not in out
        assert "before" in out and "after" in out


# ---------------------------------------------------------------------------
#  Markdown link no longer silently eaten
# ---------------------------------------------------------------------------

class TestMarkdownLinkNotEaten:
    def test_a_bare_markdown_link_is_not_swallowed(self, capsys):
        """Previously 'see [the docs](url)' rendered as 'see (url)' - the
        unguarded markup parser treated '[the docs]' as an unrecognised style
        tag and dropped it silently. It must survive intact now."""
        display.print_streaming_token("see [the docs](url) for details")
        out = capsys.readouterr().out
        assert "[the docs](url)" in out


# ---------------------------------------------------------------------------
#  The audit gap: a turn that raises mid-stream must still record
# ---------------------------------------------------------------------------

class TestAuditGapOnMidStreamException:
    def test_a_turn_that_raises_mid_stream_still_records_to_the_audit_log(self, tmp_path):
        """Before this fix, _stream_and_record's only except clause was
        KeyboardInterrupt - ANY other exception from on_token (e.g. the
        MarkupError this whole module is about, or any future one) skipped
        _accumulate_usage()/_audit.llm() entirely, leaving the turn with no
        audit record at all."""
        agent = _make_agent(tmp_path)

        def fake_chat_stream(messages, on_reasoning=None, **kw):
            yield "partial "
            yield "more"

        agent.backend.chat_stream.side_effect = fake_chat_stream

        def _raising_token(piece):
            raise RuntimeError("display exploded")

        with patch("localm.plugins.coder.agent.context.print_streaming_token",
                   side_effect=_raising_token), \
             patch("localm.plugins.coder.agent.context.print_reasoning_token"):
            with pytest.raises(RuntimeError, match="display exploded"):
                agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        agent._audit.llm.assert_called_once()
        call = agent._audit.llm.call_args
        assert call.args[0] == "partial "   # the piece streamed before the raise


# ---------------------------------------------------------------------------
#  End-to-end: the real [/INST] case through the real (unpatched) display fn
# ---------------------------------------------------------------------------

class TestEndToEndLeakedInstToken:
    def test_a_leaked_inst_tag_no_longer_crashes_the_interactive_stream(self, tmp_path, capsys):
        """THE REAL-WORLD CASE (CONTROL's report): an abliterated model with
        no tool-calling training free-generates a leaked Llama/Mistral
        control token. Previously this raised MarkupError out of the
        streaming loop entirely; now it renders as plain text and the turn
        completes and records normally, through the REAL print_streaming_token
        (not mocked) so this proves the actual Rich call path, not a stub."""
        agent = _make_agent(tmp_path)

        def fake_chat_stream(messages, on_reasoning=None, **kw):
            yield "Sure thing [/INST] here is the answer."

        agent.backend.chat_stream.side_effect = fake_chat_stream

        result = agent._call_llm([{"role": "user", "content": "hi"}], interactive=True)

        assert result == "Sure thing [/INST] here is the answer."
        agent._audit.llm.assert_called_once()
        out = capsys.readouterr().out
        assert "[/INST]" in out
