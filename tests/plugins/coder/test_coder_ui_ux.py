# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the CLI coder UI/UX improvements: scrollback-safe spinner, live
hidden tool-call progress, single-print streaming, compact diff previews with
[f]ull/[a]lways/rejection-reason support, failed-tool error excerpts, todo
checklist rendering, and @file context expansion."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.console import SafeConsole
from localm.plugins.coder import display
from localm.plugins.coder.cli.repl import _expand_at_mentions, _read_multiline
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import ToolResult
from tests.test_cli_stream_scroll import _has_viewport_control


def _make_agent(tmp_path: Path, *, auto_approve: bool = False):
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
        agent = Agent(backend=backend, cwd=tmp_path, auto_approve=auto_approve)
    agent._audit = MagicMock()
    return agent


class TestActivitySpinner:
    def test_spinner_emits_no_viewport_control_sequences(self):
        buf = io.StringIO()
        tty_console = SafeConsole(force_terminal=True, file=buf, highlight=False, width=80)
        spinner = display.ActivitySpinner(
            "Thinking...", console_obj=tty_console, interval=0.02
        )
        spinner.start()
        spinner.update("Preparing write_file", "src/app.py · 42 lines")
        spinner._render_frame_locked(tty_console)
        spinner.stop()

        out = buf.getvalue()
        assert "Thinking..." in out
        assert "Preparing write_file" in out
        assert "src/app.py" in out
        assert "\r" in out
        assert not _has_viewport_control(out), repr(out)

    def test_spinner_is_noop_on_non_tty_console(self):
        buf = io.StringIO()
        non_tty = SafeConsole(force_terminal=False, file=buf, highlight=False)
        with display.ActivitySpinner("Thinking...", console_obj=non_tty):
            pass
        assert buf.getvalue() == ""


class TestHiddenToolCallLiveProgress:
    def test_hidden_write_file_reports_live_progress_without_leaking_tokens(self, tmp_path):
        agent = _make_agent(tmp_path)
        py_lines = "\\n".join(f"def fn_{i}(): return {i}" for i in range(40))
        raw_reply = (
            "I will create the module now.\n"
            '<tool_call>\n{"name": "write_file", "args": '
            f'{{"path": "src/module.py", "content": "{py_lines}"}}}}\n</tool_call>'
        )
        chunks = [raw_reply[i:i + 15] for i in range(0, len(raw_reply), 15)]

        def fake_stream(messages, on_reasoning=None, **kw):
            yield from chunks

        agent.backend.chat_stream.side_effect = fake_stream
        progress_updates = []
        streamed_tokens = []

        orig_init = display._SPINNER_FRAMES  # keep module loaded
        assert orig_init

        from localm.plugins.coder.agent import context
        orig_tp = context._TerminalReply.tool_progress

        def spy_tp(self, label, detail=""):
            progress_updates.append((label, detail))
            return orig_tp(self, label, detail)

        with patch.object(context._TerminalReply, "tool_progress", spy_tp), \
             patch.object(context, "print_streaming_token", side_effect=streamed_tokens.append), \
             patch.object(context, "print_thinking"), \
             patch.object(context, "print_assistant_label"), \
             patch.object(context, "print_streaming_done"):
            agent._call_llm([{"role": "user", "content": "write module"}], interactive=True)

        visible = "".join(streamed_tokens)
        assert "I will create the module now." in visible
        assert "def fn_0" not in visible
        assert any("Preparing write_file" in lbl for lbl, _ in progress_updates)
        assert any("src/module.py" in dtl for _, dtl in progress_updates)


class TestNoDoublePrintInInteractiveLoop:
    def test_streamed_prose_before_tool_call_is_printed_only_once(self, tmp_path, capsys):
        agent = _make_agent(tmp_path, auto_approve=True)
        (tmp_path / "hello.py").write_text("x = 1\n", encoding="utf-8")

        turn_replies = [
            'Checking the file first.\n<tool_call>\n{"name": "read_file", "args": {"path": "hello.py"}}\n</tool_call>',
            "All done.",
        ]

        def fake_stream(messages, on_reasoning=None, **kw):
            reply = turn_replies.pop(0)
            for i in range(0, len(reply), 12):
                yield reply[i:i + 12]

        agent.backend.chat_stream.side_effect = fake_stream
        agent.chat("read hello.py")
        out = capsys.readouterr().out
        assert out.count("Checking the file first.") == 1


class TestDiffPreviewAndConfirmationUX:
    def test_large_new_file_diff_is_capped_at_30_lines_by_default(self):
        big_py = "".join(f"line_{i} = {i}\n" for i in range(120))
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            display.print_diff_preview("", big_py, "big_module.py")
            diff_text = mock_syntax.call_args[0][0]
            assert "more lines truncated - press [f] at prompt for full diff" in diff_text
            assert "line_119" not in diff_text

    def test_confirm_diff_f_expands_full_diff_then_approves(self, monkeypatch):
        big_py = "".join(f"line_{i} = {i}\n" for i in range(80))
        answers = iter(["f", "y"])
        monkeypatch.setattr(display.console, "input", lambda _prompt="": next(answers))

        printed_syntaxes = []
        with patch("localm.plugins.coder.display.Syntax",
                   side_effect=lambda text, *a, **kw: printed_syntaxes.append(text) or text), \
             patch.object(display.console, "print"):
            display.print_diff_preview("", big_py, "app.py")
            approved = display.confirm_diff("app.py")

        assert approved is True
        assert len(printed_syntaxes) == 2
        assert "truncated" in printed_syntaxes[0]
        assert "line_79 = 79" in printed_syntaxes[1]

    def test_session_allow_covers_only_that_tool(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path, auto_approve=False)
        prompts = []
        answers = iter(["a", "n"])

        def fake_input(prompt=""):
            prompts.append(str(prompt))
            return next(answers)

        monkeypatch.setattr(display.console, "input", fake_input)
        first = ToolCall(name="write_file", args={"path": "new.py", "content": "x = 1\n"},
                         raw="", start=0, end=0)
        assert agent._execute_tool(first, interactive=True).ok is True
        assert agent.auto_approve is False

        second = ToolCall(name="write_file", args={"path": "other.py", "content": "y = 2\n"},
                          raw="", start=0, end=0)
        assert agent._execute_tool(second, interactive=True).ok is True
        assert len(prompts) == 1

        shell = ToolCall(name="run_shell", args={"command": "echo hi"},
                         raw="", start=0, end=0)
        res = agent._execute_tool(shell, interactive=True)
        assert len(prompts) == 2
        assert res.ok is False

    def test_session_allow_for_shell_is_exact_command_only(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path, auto_approve=False)
        prompts = []
        answers = iter(["a", "n"])

        def fake_input(prompt=""):
            prompts.append(str(prompt))
            return next(answers)

        monkeypatch.setattr(display.console, "input", fake_input)

        def call(cmd):
            return ToolCall(name="run_shell", args={"command": cmd}, raw="", start=0, end=0)

        agent._execute_tool(call("python --version"), interactive=True)
        agent._execute_tool(call("python --version"), interactive=True)
        assert len(prompts) == 1
        agent._execute_tool(call("python --version && echo x"), interactive=True)
        assert len(prompts) == 2

    def test_session_allow_never_waives_always_confirm_or_lenient(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path, auto_approve=False)
        agent.always_confirm = {"write_file"}
        prompts = []
        monkeypatch.setattr(display.console, "input",
                            lambda prompt="": prompts.append(str(prompt)) or "a")
        for name in ("a.py", "b.py"):
            c = ToolCall(name="write_file", args={"path": name, "content": "z = 0\n"},
                         raw="", start=0, end=0)
            assert agent._execute_tool(c, interactive=True).ok is True
        assert len(prompts) == 2

        agent.always_confirm = set()
        prompts.clear()
        for name in ("c.py", "d.py"):
            c = ToolCall(name="write_file", args={"path": name, "content": "z = 0\n"},
                         raw="", start=0, end=0, lenient=True)
            assert agent._execute_tool(c, interactive=True).ok is True
        assert len(prompts) == 2

    def test_confirm_rejection_reason_is_passed_back_to_agent(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path, auto_approve=False)
        monkeypatch.setattr(
            display.console, "input", lambda _prompt="": "n: keep existing config"
        )

        call = ToolCall(
            name="write_file",
            args={"path": "cfg.py", "content": "x = 2\n"},
            raw="",
            start=0,
            end=0,
        )
        res = agent._execute_tool(call, interactive=True)
        assert res.ok is False
        assert "Rejected by user: keep existing config" in res.output


    def test_free_text_at_prompt_rejects_and_passes_the_text_back(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path, auto_approve=False)
        monkeypatch.setattr(display.console, "input",
                            lambda _prompt="": "put it in src/ instead")
        call = ToolCall(name="write_file", args={"path": "cfg.py", "content": "x = 2\n"},
                        raw="", start=0, end=0)
        res = agent._execute_tool(call, interactive=True)
        assert res.ok is False
        assert "Rejected by user: put it in src/ instead" in res.output
        assert not (tmp_path / "cfg.py").exists()

    def test_bare_enter_and_n_reject_without_a_reason(self):
        assert display._parse_confirm_answer("") == ("no", "")
        assert display._parse_confirm_answer("N") == ("no", "")
        assert display._parse_confirm_answer(" y ") == ("yes", "")


class TestContextWarningText:
    def test_compact_warning_carries_no_literal_markup(self, tmp_path):
        agent = _make_agent(tmp_path)
        warnings = []
        with patch("localm.plugins.coder.agent.print_warning", side_effect=warnings.append), \
             patch.object(type(agent), "_fill_ratio", return_value=0.9):
            agent._maybe_compact(interactive=True)
        assert warnings and "/compact" in warnings[0]
        assert "[bold]" not in warnings[0]


class TestToolResultAndCallFormatting:
    def test_failed_tool_prints_error_excerpt_even_when_not_verbose(self, capsys):
        res = ToolResult(
            ok=False,
            output="<exit_code>1</exit_code>\n<stdout>\nFAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n</stdout>",
            summary="exit 1",
        )
        display.print_tool_result("run_shell", res, verbose=False, elapsed=0.42)
        out = capsys.readouterr().out
        assert "exit 1" in out
        assert "(0.4s)" in out
        assert "AssertionError: 1 != 2" in out
        assert "<exit_code>" not in out

    def test_set_todos_renders_checklist(self, capsys):
        res = ToolResult.success("3 items", summary="3 items (1 done, 1 in_progress, 1 pending)")
        items = [
            {"task": "Inspect display.py", "status": "done"},
            {"task": "Add ActivitySpinner", "status": "in_progress"},
            {"task": "Run pytest", "status": "pending"},
        ]
        display.print_tool_result("set_todos", res, verbose=False, args={"items": items})
        out = capsys.readouterr().out
        assert "[✓] Inspect display.py" in out
        assert "[▸] Add ActivitySpinner" in out
        assert "[ ] Run pytest" in out

    def test_multiline_edit_file_args_summarized_by_line_count(self):
        text = display.tool_call_text(
            "edit_file",
            {"path": "a.py", "old": "line1\nline2\nline3", "new": "line1\nline2"},
        )
        assert "old=<3 lines>" in text
        assert "new=<2 lines>" in text


class TestReplAtMentionsAndPaste:
    def test_expand_at_mentions_attaches_file_and_line_slice(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")

        expanded = _expand_at_mentions(
            "Check @sample.py:2-3 and ignore @property decorator", tmp_path
        )
        assert '<attached_file path="sample.py:2-3">' in expanded
        assert "b = 2\nc = 3" in expanded
        assert "a = 1" not in expanded

    def test_expand_at_mentions_blocks_outside_cwd(self, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("top-secret\n", encoding="utf-8")
        sub = tmp_path / "proj"
        sub.mkdir()

        expanded = _expand_at_mentions("Read @../secret.txt", sub)
        assert "top-secret" not in expanded
        assert "<attached_file" not in expanded

    def test_expand_at_mentions_respects_session_scope(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "ok.py").write_text("inside = 1\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("outside = 1\n", encoding="utf-8")
        expanded = _expand_at_mentions(
            "see @src/ok.py and @other.py", tmp_path,
            scope_allows=lambda p: p.replace("\\", "/").startswith("src/"))
        assert "inside = 1" in expanded
        assert "outside = 1" not in expanded

    def test_expand_at_mentions_leaves_email_and_handles_alone(self, tmp_path):
        text = "mail me@example.com or ping @nobody about it"
        assert _expand_at_mentions(text, tmp_path) == text

    def test_read_multiline_coalesces_pasted_lines(self, monkeypatch):
        monkeypatch.setattr(display.console, "input", lambda _prompt="": "first line")
        buffered = iter(["second line\n", "third line\n"])
        state = {"count": 2}

        def has_data():
            if state["count"] > 0:
                state["count"] -= 1
                return True
            return False

        monkeypatch.setattr("localm.plugins.coder.cli.repl._stdin_has_buffered_data", has_data)
        monkeypatch.setattr("sys.stdin.readline", lambda: next(buffered))

        msg = _read_multiline()
        assert msg == "first line\nsecond line\nthird line"
