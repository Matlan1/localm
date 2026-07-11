# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.plugins.coder.diffutil - the shared diff-computation helper
(CODER-1) that replaced three independent implementations in agent/execution.py
and sessions.py."""

from localm.plugins.coder.diffutil import (
    compute_tool_diff, read_old_content, resolve_new_content,
)


class TestReadOldContent:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_old_content(tmp_path, "nope.py") == ""

    def test_empty_path_returns_empty(self, tmp_path):
        assert read_old_content(tmp_path, "") == ""

    def test_reads_existing_file(self, tmp_path):
        (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
        assert read_old_content(tmp_path, "a.py") == "hello\n"

    def test_unreadable_path_returns_empty_not_raises(self, tmp_path):
        # A directory at that path can't be read as text - must not crash.
        (tmp_path / "adir").mkdir()
        assert read_old_content(tmp_path, "adir") == ""


class TestResolveNewContent:
    def test_write_file_uses_content_key(self):
        assert resolve_new_content(
            "write_file", {"path": "x.py", "content": "new\n"}, "old\n") == "new\n"

    def test_edit_file_replaces_old_with_new(self):
        result = resolve_new_content(
            "edit_file", {"old": "a", "new": "b"}, "a = 1\n")
        assert result == "b = 1\n"

    def test_patch_file_returns_none(self):
        assert resolve_new_content("patch_file", {"diff": "x"}, "old") is None

    def test_unknown_tool_returns_none(self):
        assert resolve_new_content("run_shell", {"command": "ls"}, "old") is None


class TestComputeToolDiff:
    def test_write_file_diff(self):
        diff = compute_tool_diff(
            "write_file", {"path": "a.py", "content": "line2\n"}, "line1\n")
        assert "-line1" in diff
        assert "+line2" in diff
        assert "a/a.py" in diff and "b/a.py" in diff

    def test_edit_file_diff(self):
        diff = compute_tool_diff(
            "edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"}, "x = 1\n")
        assert "-x = 1" in diff
        assert "+x = 2" in diff

    def test_edit_file_no_change_returns_none(self):
        diff = compute_tool_diff(
            "edit_file", {"path": "a.py", "old": "nomatch", "new": "y"}, "x = 1\n")
        assert diff is None

    def test_patch_file_returns_diff_verbatim(self):
        raw = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
        assert compute_tool_diff("patch_file", {"diff": raw}, "") == raw

    def test_patch_file_empty_diff_returns_none(self):
        assert compute_tool_diff("patch_file", {"diff": ""}, "") is None

    def test_unknown_tool_returns_none(self):
        assert compute_tool_diff("run_shell", {"command": "ls"}, "old") is None

    def test_new_file_shows_all_additions(self):
        diff = compute_tool_diff(
            "write_file", {"path": "new.py", "content": "a\nb\n"}, "")
        assert "+a" in diff and "+b" in diff
