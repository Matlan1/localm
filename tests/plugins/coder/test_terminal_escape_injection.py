# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-controlled text reaching the terminal must never carry a terminal
control sequence, never be parsed as Rich markup, and never be silently eaten.

display.py sanitizes what it owns (_strip_esc / _sanitized_text). These cover
the sites that BYPASS it: callers that splice externally-controlled values into
a Rich markup f-string, or hand a plain string to a bare console.print, where
Rich parses markup and passes raw ANSI through verbatim.

Three payload classes, because they fail in three different ways:
  RAW_ANSI   a screen clear / cursor reposition reaching the real terminal
  UNMATCHED  an unmatched closing tag, which raises MarkupError from a bare
             console.print and takes the surrounding turn or command with it
  EATEN      a well-formed but unknown tag, which Rich DELETES silently, so
             the user reads a line with a field missing and no error anywhere
"""

from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder import display

ESC = "\x1b"
RAW_ANSI = ESC + "[2J" + ESC + "[H"
UNMATCHED = "[/INST]"
EATEN = "[done, spawn_agent]"
LINK = "[link=http://evil.example/steal]click to continue[/link]"

PAYLOADS = [
    pytest.param(RAW_ANSI, id="raw-ansi"),
    pytest.param(UNMATCHED, id="unmatched-tag"),
    pytest.param(EATEN, id="silently-eaten-tag"),
    pytest.param(LINK, id="link-markup"),
]


def _assert_inert(out: str, payload: str) -> None:
    """The three properties every one of these sites must hold."""
    assert ESC not in out, "a raw ANSI escape reached the terminal"
    # The payload survives as literal text rather than being parsed away. The
    # link case keeps its label; the others keep the whole string.
    needle = "click to continue" if payload == LINK else payload
    assert needle in out, f"content was silently eaten: {payload!r} not in {out!r}"


# ---------------------------------------------------------------------------
#  display.print_assistant_response: the Markdown branch, not just the fallback
# ---------------------------------------------------------------------------

class TestAssistantResponseMarkdownBranch:
    """The fallback branch was already sanitized. The Markdown branch was not,
    and it is the one a real reply takes: any of ``` ** ## '- ' '1. ' routes
    there, which is nearly every coding-model answer."""

    MARKERS = ["- bullet\n", "**bold** ", "## head\n", "1. one\n"]

    @pytest.mark.parametrize("marker", MARKERS)
    def test_a_markdown_shaped_reply_cannot_carry_an_escape(self, marker, capsys):
        display.print_assistant_response(marker + "text" + RAW_ANSI + "INJ")
        out = capsys.readouterr().out
        assert ESC not in out

    def test_the_non_markdown_fallback_is_still_sanitized(self, capsys):
        display.print_assistant_response("plain answer" + RAW_ANSI + "INJ")
        out = capsys.readouterr().out
        assert ESC not in out


# ---------------------------------------------------------------------------
#  agent/loop.py: raw model text on the INTERACTIVE path
# ---------------------------------------------------------------------------

def _make_agent(tmp_path):
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


class TestInteractiveSegmentPrint:
    """When the model narrates AND calls a tool, split_response's text segments
    are printed. Driven end to end through the real Agent, because the existing
    end-to-end case yields a response with NO tool call, so this print never
    runs there."""

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_a_narrating_model_cannot_reach_the_terminal(self, payload, tmp_path, capsys):
        agent = _make_agent(tmp_path)
        narration = "Here is my plan " + payload

        def fake_stream(messages, on_reasoning=None, **kw):
            yield (narration + "\n"
                   '<tool_call>{"name": "read_file", '
                   '"arguments": {"path": "x.py"}}</tool_call>')

        agent.backend.chat_stream.side_effect = fake_stream
        agent.chat("do it")          # must not raise
        _assert_inert(capsys.readouterr().out, payload)


# ---------------------------------------------------------------------------
#  cli/repl.py: the command handlers that splice model data into markup
# ---------------------------------------------------------------------------

class TestReplCommandsSurviveHostileModelData:
    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_memory_renders_a_hostile_memory_file(self, payload, capsys):
        from localm.plugins.coder.cli import repl
        agent = MagicMock()
        agent._memory = "notes " + payload
        repl._handle_command_extended("memory", "", agent)
        _assert_inert(capsys.readouterr().out, payload)

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_changes_renders_a_model_chosen_path(self, payload, capsys):
        from localm.plugins.coder.cli import repl
        agent = MagicMock()
        agent.changed_files.return_value = [{
            "path": "src/" + payload + "x.py", "created": True,
            "exists": True, "writes": 1, "last_tool": "write_file",
        }]
        repl._handle_command_extended("changes", "", agent)
        _assert_inert(capsys.readouterr().out, payload)

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_bg_renders_a_model_supplied_label(self, payload, capsys):
        from localm.plugins.coder.cli import repl
        import localm.plugins.coder.background as bg
        registry = MagicMock()
        registry.list_status.return_value = [{
            "id": "bg1", "kind": "agent", "label": "job " + payload,
            "state": "running", "started_at": 0.0, "result": {},
        }]
        registry.dropped_undrained_by_kind = {}
        with patch.object(bg, "get_registry", return_value=registry):
            repl._handle_command_extended("bg", "", MagicMock())
        _assert_inert(capsys.readouterr().out, payload)

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_review_renders_reviewer_model_output(self, payload, capsys):
        from localm.plugins.coder.cli import repl
        agent = MagicMock()
        agent.session_diff.return_value = "diff --git a/x b/x\n+1\n"
        result = MagicMock()
        result.approved = False
        result.blocking = ["issue " + payload]
        result.notes = "note " + payload
        reviewer = MagicMock()
        reviewer.review.return_value = result
        reviewer.failure_warning.return_value = None
        reviewer.heterogeneous = False
        agent._reviewer = reviewer
        repl._handle_command_extended("review", "", agent)
        _assert_inert(capsys.readouterr().out, payload)


# ---------------------------------------------------------------------------
#  delegated footer: printed as one plain block, so its own brackets matter
# ---------------------------------------------------------------------------

class TestDelegatedFooterIsNotEaten:
    """render_footer emits literal brackets of its own ("[running]", the
    "[truncated - full diff: ...]" notice). Printed as markup those are parsed
    as unknown tags and DELETED, so the status field and the truncation notice
    vanish with no error. The truncation notice disappearing is the one that
    matters: it tells the user the diff they are reading is incomplete."""

    def test_status_and_truncation_notice_survive(self, capsys):
        from localm.plugins.coder.cli import repl
        footer = ("Delegated work (NOT in your working tree):\n"
                  "  mylabel [running, spawn_agent]  3 file(s)\n"
                  "... [truncated - full diff: git -C x diff]")
        agent = MagicMock()
        agent.changed_files.return_value = []
        with patch("localm.plugins.coder.delegated.footer_for",
                   return_value=footer):
            repl._handle_command_extended("changes", "", agent)
        out = capsys.readouterr().out
        assert "[running, spawn_agent]" in out, "the status field was eaten"
        assert "truncated - full diff" in out, "the truncation notice was eaten"


# ---------------------------------------------------------------------------
#  tools/parallel.py: the attribution line before a confirmation prompt
# ---------------------------------------------------------------------------

class TestSubAgentAttributionSurvives:
    """The line naming WHICH child is about to prompt is printed immediately
    before a confirmation prompt, and its print is wrapped in a blanket except.
    A model-chosen label that raised MarkupError therefore deleted the
    attribution and left an unattributed approval prompt."""

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_a_hostile_child_label_cannot_delete_the_attribution(self, payload, capsys):
        from localm.plugins.coder.tools import parallel

        class FakeCall:
            name = "run_shell"

        parallel._announce_asker("child" + payload, FakeCall())
        out = capsys.readouterr().out
        assert "sub-agent" in out, "the attribution line was suppressed"
        assert "run_shell" in out
        assert ESC not in out


# ---------------------------------------------------------------------------
#  The helper itself
# ---------------------------------------------------------------------------

class TestSafeMarkup:
    def test_strips_ansi_and_neutralises_markup_while_keeping_the_text(self):
        assert ESC not in display.safe_markup("a" + RAW_ANSI + "b")
        assert "[/INST]" in display.safe_markup("x [/INST] y")

    def test_accepts_a_non_string(self):
        assert display.safe_markup(7) == "7"
