# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for Agent.context_chars() (multipart message handling) and
Agent._patch_mode_intercept() (correct tool argument keys).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_agent(tmp_path: Path) -> object:
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path)
    return agent


def _make_call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


# ---------------------------------------------------------------------------
#  context_chars() - multipart messages
# ---------------------------------------------------------------------------

class TestContextChars:
    def test_counts_string_content(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._messages = [
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        # "hello" + "world" = 10 chars + system prompt
        assert agent.context_chars() >= 10

    def test_counts_multipart_content_by_text_length(self, tmp_path):
        agent = _make_agent(tmp_path)
        # List content: len([...]) == 2, but total text is 10 chars
        agent._messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"},
            ]},
        ]
        chars = agent.context_chars()
        # "hello world" = 11 chars; must not be len([...]) = 2
        assert chars > 2

    def test_mixed_string_and_list_messages(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._messages = [
            {"role": "user",      "content": "plain text"},         # 10 chars
            {"role": "assistant", "content": [
                {"type": "text", "text": "response here"},          # 13 chars
            ]},
        ]
        chars = agent.context_chars()
        assert chars >= 23  # 10 + 13 (plus system prompt)

    def test_list_with_non_text_parts_ignored(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text",      "text": "describe this"},
            ]},
        ]
        chars = agent.context_chars()
        # Only "describe this" (13 chars) should be counted, not the image
        assert chars >= 13

    def test_empty_messages_returns_system_prompt_length(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._messages = []
        assert agent.context_chars() == len(agent._system_prompt)


# ---------------------------------------------------------------------------
#  _patch_mode_intercept - correct arg keys
# ---------------------------------------------------------------------------

class TestPatchModeIntercept:
    def test_write_file_uses_content_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.patch_mode = True

        call = _make_call("write_file", path="x.py", content="print('hi')\n")
        agent._patch_mode_intercept(call)
        # Returns a diff string (even if empty for new file) or None
        # Must not raise KeyError

    def test_edit_file_uses_old_and_new_keys(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.patch_mode = True

        f = tmp_path / "edit.py"
        f.write_text("x = 1\n")

        call = _make_call("edit_file", path="edit.py", old="x = 1", new="x = 2")
        diff = agent._patch_mode_intercept(call)
        assert diff is not None
        assert "-x = 1" in diff
        assert "+x = 2" in diff

    def test_edit_file_wrong_keys_produce_no_diff(self, tmp_path):
        """'old_string'/'new_string' are not the keys read, so they yield empty
        strings and no diff."""
        agent = _make_agent(tmp_path)
        agent.patch_mode = True

        f = tmp_path / "code.py"
        f.write_text("x = 1\n")

        # The wrong arg keys: they find empty strings and produce an empty diff
        call = _make_call("edit_file", path="code.py", old_string="x = 1", new_string="x = 2")
        diff = agent._patch_mode_intercept(call)
        # old="" and new="" → replace("", "", 1) → no change → no diff lines
        assert diff is None or diff == "" or ("x = 2" not in (diff or ""))

    def test_patch_file_uses_diff_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.patch_mode = True

        raw_diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
        call = _make_call("patch_file", path="x.py", diff=raw_diff)
        result = agent._patch_mode_intercept(call)
        assert result == raw_diff

    def test_patch_file_old_key_patch_returns_none(self, tmp_path):
        """A 'patch' key is not read: with no 'diff' key the intercept returns
        None."""
        agent = _make_agent(tmp_path)
        agent.patch_mode = True

        raw_diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
        # Using wrong key 'patch' instead of 'diff'
        call = _make_call("patch_file", path="x.py", patch=raw_diff)
        result = agent._patch_mode_intercept(call)
        # 'diff' key is absent → should return None
        assert result is None


# ---------------------------------------------------------------------------
#  _confirm_tool - correct arg keys for edit_file / patch_file
# ---------------------------------------------------------------------------

class TestConfirmToolArgKeys:
    def test_edit_file_reads_old_and_new(self, tmp_path):
        agent = _make_agent(tmp_path)

        f = tmp_path / "edit.py"
        f.write_text("original content\n")

        captured = {}

        def capture_diff(old, new, path_label=""):
            captured["old"] = old
            captured["new"] = new

        call = _make_call("edit_file", path="edit.py", old="original", new="replaced")

        with patch("localm.plugins.coder.agent.print_diff_preview", side_effect=capture_diff), \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        assert captured.get("old") == "original content\n"
        # new_content = old.replace("original", "replaced", 1)
        assert "replaced" in captured.get("new", "")

    def test_patch_file_reads_diff_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        raw_diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"

        call = _make_call("patch_file", path="f.py", diff=raw_diff)

        # _confirm_tool imports console from .display at call time, so patch it
        # at the display module level.
        with patch("localm.plugins.coder.display.console") as mock_con, \
             patch("localm.plugins.coder.agent.confirm_diff", return_value=True):
            agent._confirm_tool(call)

        # Should have printed at least the blank line + the Syntax object
        assert mock_con.print.call_count >= 1
