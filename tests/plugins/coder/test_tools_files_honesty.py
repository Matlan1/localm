# SPDX-License-Identifier: AGPL-3.0-or-later
"""Honesty and edge-case guards in the coder file tools.

Three properties:
- search_replace: a write failure mid-loop must not leave earlier files modified
  on disk while the error reads as if nothing changed.
- edit_file: an empty `old` must not prepend `new` ('' is "in" every string) and
  report an occurrence count for it.
- read_file: offset/limit on an empty file must not produce a backwards range
  label ("1-0 of 1"); splitlines() disagrees with _line_count about empty."""

from unittest.mock import patch

from localm.plugins.coder.tools import (
    tool_edit_file,
    tool_read_file,
    tool_search_replace,
)


def _norm(raw: bytes) -> str:
    """Universal-newline-normalise raw bytes for a platform-portable
    comparison. The fixtures below write via Path.write_text(), which
    translates \\n -> the platform line separator on write (\\r\\n on
    Windows, unchanged on Linux) - so the RAW bytes ToolResult.changes
    reports (deliberately unnormalised, matching execution.py's own
    snapshot convention) differ by platform even though the file's logical
    content does not. Normalise before comparing so the assertion holds on
    both."""
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


class TestSearchReplacePartialWriteHonesty:
    def _project(self, tmp_path, n=3):
        for i in range(n):
            (tmp_path / f"f{i}.txt").write_text("old value\n", encoding="utf-8")
        return tmp_path

    def test_write_failure_reports_partial_apply(self, tmp_path):
        cwd = self._project(tmp_path)
        real_write = type(cwd).write_text
        calls = {"n": 0}

        def failing_write(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:                       # second file's write fails
                raise OSError("disk full")
            return real_write(self, *args, **kwargs)

        with patch.object(type(cwd), "write_text", failing_write):
            r = tool_search_replace(cwd, pattern="old", replacement="new",
                                    glob="*.txt")
        assert r.ok is False
        assert "PARTIAL APPLY" in r.output
        assert "1 of 3 file(s)" in r.output
        assert "Modified: f0.txt" in r.output
        assert "NOT modified:" in r.output
        assert "f1.txt" in r.output.split("NOT modified:")[1]
        assert "f2.txt" in r.output.split("NOT modified:")[1]
        # The filesystem really is in the reported state.
        assert (cwd / "f0.txt").read_text(encoding="utf-8") == "new value\n"
        assert (cwd / "f1.txt").read_text(encoding="utf-8") == "old value\n"
        # A partial apply still exposes what it actually wrote: the changed-files
        # and undo tracker reads this even on a failed call.
        assert len(r.changes) == 1
        name, old, new = r.changes[0]
        assert name == "f0.txt" and _norm(old) == "old value\n" and new == "new value\n"

    def test_first_write_failure_reports_zero_modified(self, tmp_path):
        cwd = self._project(tmp_path, n=2)

        def always_fail(self, *args, **kwargs):
            raise OSError("read-only filesystem")

        with patch.object(type(cwd), "write_text", always_fail):
            r = tool_search_replace(cwd, pattern="old", replacement="new",
                                    glob="*.txt")
        assert r.ok is False
        assert "0 of 2 file(s)" in r.output
        assert "Modified:" not in r.output          # nothing was written
        assert "NOT modified:" in r.output
        assert r.changes is None                    # nothing to track - nothing changed

    def test_all_writes_succeed_unchanged(self, tmp_path):
        cwd = self._project(tmp_path)
        r = tool_search_replace(cwd, pattern="old", replacement="new",
                                glob="*.txt")
        assert r.ok is True
        assert "Replaced 3 match(es) in 3 file(s)" in r.output
        assert {name for name, _, _ in r.changes} == {"f0.txt", "f1.txt", "f2.txt"}
        assert all(_norm(old) == "old value\n" and new == "new value\n"
                  for _, old, new in r.changes)


class TestEditFileEmptyOld:
    def test_empty_old_rejected(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "a.txt", old="", new="X")
        assert r.ok is False
        assert "empty" in r.output
        # The file is untouched - no silent prepend.
        assert f.read_text(encoding="utf-8") == "hello world\n"

    def test_normal_edit_still_works(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "a.txt", old="world", new="there")
        assert r.ok is True
        assert f.read_text(encoding="utf-8") == "hello there\n"


class TestReadFileEmptyWithRange:
    def test_empty_file_with_offset_limit_label_not_backwards(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        r = tool_read_file(tmp_path, "empty.txt", offset=1, limit=1)
        assert r.ok is True
        assert "1-0" not in r.output                 # the backwards label
        assert "<lines>1-1 of 1</lines>" in r.output

    def test_nonempty_range_label_unchanged(self, tmp_path):
        (tmp_path / "three.txt").write_text("a\nb\nc\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "three.txt", offset=2, limit=2)
        assert r.ok is True
        assert "<lines>2-3 of 4</lines>" in r.output
