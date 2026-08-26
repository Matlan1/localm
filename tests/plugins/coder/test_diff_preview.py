# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for display.print_diff_preview and agent._confirm_tool diff path.
"""

from unittest.mock import MagicMock, patch

from localm.plugins.coder.display import print_diff_preview


# ---------------------------------------------------------------------------
#  print_diff_preview unit tests
# ---------------------------------------------------------------------------

class TestPrintDiffPreview:
    def _capture(self, old, new, label="test.py", max_lines=200):
        """Run print_diff_preview and return the Rich output as a string."""
        # Patch console to write to our buffer (strip markup)
        with patch("localm.plugins.coder.display.console") as mock_console:
            printed = []
            def capture_print(*args, **kwargs):
                printed.append(str(args[0]) if args else "")
            mock_console.print.side_effect = capture_print
            print_diff_preview(old, new, path_label=label, max_lines=max_lines)
        return "\n".join(printed)

    def test_no_diff_message_for_identical_content(self):
        text = "line1\nline2\n"
        output = self._capture(text, text)
        assert "no changes" in output

    def test_diff_shows_for_changed_content(self):
        old = "line1\nline2\n"
        new = "line1\nchanged\n"
        # Should call Syntax - patch it to verify it was called
        with patch("localm.plugins.coder.display.console") as mock_console, \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview(old, new, "test.py")
            # console.print should have been called with a Syntax object
            assert mock_console.print.call_count >= 2  # empty line + Syntax
            mock_syntax.assert_called_once()
            call_args = mock_syntax.call_args[0]
            diff_text = call_args[0]
            assert "-line2" in diff_text
            assert "+changed" in diff_text

    def test_diff_uses_correct_labels(self):
        old = "hello\n"
        new = "world\n"
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview(old, new, "src/foo.py")
            diff_text = mock_syntax.call_args[0][0]
            assert "a/src/foo.py" in diff_text
            assert "b/src/foo.py" in diff_text

    def test_default_labels_when_no_path(self):
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview("old\n", "new\n", "")
            diff_text = mock_syntax.call_args[0][0]
            assert "a/current" in diff_text
            assert "b/proposed" in diff_text

    def test_new_file_shows_all_additions(self):
        new_content = "line1\nline2\nline3\n"
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview("", new_content, "new_file.py")
            diff_text = mock_syntax.call_args[0][0]
            assert "+line1" in diff_text
            assert "+line2" in diff_text
            assert "+line3" in diff_text

    def test_max_lines_truncation(self):
        old = "\n".join(f"line{i}" for i in range(100)) + "\n"
        new = "\n".join(f"X{i}" for i in range(100)) + "\n"
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview(old, new, "big.py", max_lines=10)
            diff_text = mock_syntax.call_args[0][0]
            assert "truncated" in diff_text

    def test_no_truncation_below_limit(self):
        old = "a\n"
        new = "b\n"
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview(old, new, "x.py", max_lines=500)
            diff_text = mock_syntax.call_args[0][0]
            assert "truncated" not in diff_text

    def test_lexer_is_diff(self):
        with patch("localm.plugins.coder.display.console"), \
             patch("localm.plugins.coder.display.Syntax") as mock_syntax:
            print_diff_preview("a\n", "b\n", "x.py")
            _text, lexer = mock_syntax.call_args[0][:2]
            assert lexer == "diff"


# ---------------------------------------------------------------------------
#  Agent._confirm_tool - write_file path
# ---------------------------------------------------------------------------

class TestAgentConfirmTool:
    def _make_agent(self, tmp_path):
        """Build a minimal Agent with a mock backend."""
        from localm.plugins.coder.agent import Agent
        backend = MagicMock()
        backend.model_id = "test-model"
        backend.last_usage = {}
        with patch("localm.plugins.coder.agent.load_memory", return_value=""), \
             patch("localm.plugins.coder.agent.ProjectMap") as mock_pm:
            mock_pm.build.return_value.file_count.return_value = 0
            agent = Agent(backend=backend, cwd=tmp_path, auto_approve=False)
        return agent

    def _call(self, name, **args):
        """Make a minimal mock ToolCall with .name and .args."""
        c = MagicMock()
        c.name = name
        c.args = args
        return c

    def test_write_file_shows_diff_and_approves(self, tmp_path):
        agent = self._make_agent(tmp_path)
        (tmp_path / "foo.py").write_text("old content\n")
        call = self._call("write_file", path="foo.py", content="new content\n")

        with patch("localm.plugins.coder.agent.print_diff_preview") as mock_diff, \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True) as mock_cd:
            result = agent._confirm_tool(call)
        assert result is True
        mock_diff.assert_called_once()
        mock_cd.assert_called_once_with("foo.py")

    def test_write_file_shows_diff_and_rejects(self, tmp_path):
        agent = self._make_agent(tmp_path)
        call = self._call("write_file", path="bar.py", content="x\n")

        with patch("localm.plugins.coder.agent.print_diff_preview"), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=False):
            result = agent._confirm_tool(call)
        assert result is False

    def test_write_file_reads_existing_content(self, tmp_path):
        agent = self._make_agent(tmp_path)
        (tmp_path / "existing.py").write_text("# old\n")
        call = self._call("write_file", path="existing.py", content="# new\n")

        captured_old = {}
        def capture(old, new, path_label=""):
            captured_old["old"] = old
            captured_old["new"] = new

        with patch("localm.plugins.coder.agent.print_diff_preview", side_effect=capture), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        assert captured_old["old"] == "# old\n"
        assert captured_old["new"] == "# new\n"

    def test_write_file_empty_old_for_new_file(self, tmp_path):
        agent = self._make_agent(tmp_path)
        call = self._call("write_file", path="brand_new.py", content="hello\n")

        captured_old = {}
        def capture(old, new, path_label=""):
            captured_old["old"] = old

        with patch("localm.plugins.coder.agent.print_diff_preview", side_effect=capture), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        assert captured_old["old"] == ""

    def test_unreadable_existing_file_warns_at_the_consent_point(
            self, tmp_path, capsys):
        """The file EXISTS and could not be READ, so old_content is "" for a
        reason print_diff_preview cannot express: its contract reads "" as "the
        file doesn't exist yet", and the diff renders the overwrite as a pure
        addition with nothing deleted. A warning is printed at the consent
        point."""
        from pathlib import Path as _P

        agent = self._make_agent(tmp_path)
        (tmp_path / "real.py").write_text("line one\nline two\n", encoding="utf-8")
        call = self._call("write_file", path="real.py", content="brand new\n")

        real_read = _P.read_text

        def flaky(self, *a, **kw):
            if self.name == "real.py":
                raise OSError(13, "Permission denied")
            return real_read(self, *a, **kw)

        with patch.object(_P, "read_text", flaky), \
             patch("localm.plugins.coder.agent.print_diff_preview"), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        out = capsys.readouterr().out.lower()
        assert "could not read the current contents" in out, (
            "the user was asked to approve an overwrite of a file whose "
            "contents we could not read, with no indication of it")

    def test_new_file_does_not_warn(self, tmp_path, capsys):
        """The control: a genuinely new file gets no warning."""
        agent = self._make_agent(tmp_path)
        call = self._call("write_file", path="brand_new.py", content="hello\n")

        with patch("localm.plugins.coder.agent.print_diff_preview"), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        assert "could not read" not in capsys.readouterr().out.lower()

    def test_non_write_file_uses_plain_confirm(self, tmp_path):
        agent = self._make_agent(tmp_path)
        call = self._call("run_shell", command="rm -rf /")

        with patch("localm.plugins.coder.agent.confirm", return_value=True) as mock_confirm, \
             patch("localm.plugins.coder.agent.print_diff_preview") as mock_diff:
            result = agent._confirm_tool(call)
        assert result is True
        mock_confirm.assert_called_once()
        mock_diff.assert_not_called()
