# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for file-operation tools in localm.plugins.coder.tools:
  tool_read_file, tool_write_file, tool_edit_file, tool_list_dir, tool_tree
"""

import json

from localm.plugins.coder.tools import (
    tool_read_file,
    tool_write_file,
    tool_edit_file,
    tool_list_dir,
    tool_tree,
)


# ---------------------------------------------------------------------------
#  tool_read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "hello.txt")
        assert r.ok
        assert "hello world" in r.output

    def test_includes_path_and_line_count_in_output(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 2\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "code.py")
        assert r.ok
        # "x = 1\ny = 2\n" → count("\n")+1 = 3 (trailing newline counts)
        assert "<lines>3</lines>" in r.output
        assert "code.py" in r.output

    def test_file_not_found_returns_error(self, tmp_path):
        r = tool_read_file(tmp_path, "nonexistent.py")
        assert not r.ok
        assert "not found" in r.output.lower()

    def test_directory_path_returns_error(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        r = tool_read_file(tmp_path, "subdir")
        assert not r.ok

    def test_path_traversal_rejected(self, tmp_path):
        r = tool_read_file(tmp_path, "../../etc/passwd")
        assert not r.ok
        assert "outside" in r.output.lower() or "permission" in r.output.lower()

    def test_absolute_path_outside_cwd_rejected(self, tmp_path):
        outside = tmp_path.parent / "other.txt"
        outside.write_text("secret", encoding="utf-8")
        r = tool_read_file(tmp_path, str(outside))
        assert not r.ok

    def test_large_file_truncated(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 20_000, encoding="utf-8")
        r = tool_read_file(tmp_path, "big.txt")
        assert r.ok
        assert r.truncated
        assert "truncated" in r.output.lower()

    def test_small_file_not_truncated(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hello\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "small.txt")
        assert r.ok
        assert not r.truncated

    def test_notebook_rendered_as_text(self, tmp_path):
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["print('hi')\n"], "outputs": [], "execution_count": None},
                {"cell_type": "markdown", "source": ["# Title\n"], "outputs": []},
            ]
        }
        f = tmp_path / "test.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        r = tool_read_file(tmp_path, "test.ipynb")
        assert r.ok
        assert "<cells>2</cells>" in r.output
        assert "print" in r.output
        assert "Title" in r.output

    def test_notebook_with_output_shown(self, tmp_path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [{
                "cell_type": "code",
                "source": ["x = 1\n"],
                "outputs": [{"output_type": "stream", "text": ["output text\n"]}],
                "execution_count": 1,
            }]
        }
        f = tmp_path / "out.ipynb"
        f.write_text(json.dumps(nb), encoding="utf-8")
        r = tool_read_file(tmp_path, "out.ipynb")
        assert r.ok
        assert "output text" in r.output

    def test_unicode_content_handled(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("こんにちは\n日本語テスト\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "unicode.txt")
        assert r.ok
        assert "こんにちは" in r.output


# ---------------------------------------------------------------------------
#  tool_write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_creates_new_file(self, tmp_path):
        r = tool_write_file(tmp_path, "new.txt", "hello\n")
        assert r.ok
        assert (tmp_path / "new.txt").read_text() == "hello\n"

    def test_overwrites_existing_file(self, tmp_path):
        f = tmp_path / "exist.txt"
        f.write_text("old\n")
        r = tool_write_file(tmp_path, "exist.txt", "new\n")
        assert r.ok
        assert f.read_text() == "new\n"

    def test_creates_parent_dirs(self, tmp_path):
        r = tool_write_file(tmp_path, "deep/nested/file.txt", "content\n")
        assert r.ok
        assert (tmp_path / "deep" / "nested" / "file.txt").exists()

    def test_path_traversal_rejected(self, tmp_path):
        r = tool_write_file(tmp_path, "../evil.txt", "bad")
        assert not r.ok

    def test_summary_says_created_for_new_file(self, tmp_path):
        r = tool_write_file(tmp_path, "brand_new.py", "pass\n")
        assert r.ok
        assert "created" in r.summary.lower()

    def test_summary_says_updated_for_existing_file(self, tmp_path):
        (tmp_path / "existing.py").write_text("old\n")
        r = tool_write_file(tmp_path, "existing.py", "new\n")
        assert r.ok
        assert "updated" in r.summary.lower()

    def test_syntax_error_in_python_surfaced(self, tmp_path):
        r = tool_write_file(tmp_path, "broken.py", "def foo(\n  pass\n")
        assert r.ok  # write succeeds even with syntax error
        assert "syntax" in r.output.lower() or "syntax" in r.summary.lower()

    def test_valid_python_no_syntax_warning(self, tmp_path):
        r = tool_write_file(tmp_path, "valid.py", "def foo(): pass\n")
        assert r.ok
        assert "syntax error" not in r.output.lower()

    def test_invalid_json_surfaced(self, tmp_path):
        r = tool_write_file(tmp_path, "bad.json", "{key: value}")
        assert r.ok
        assert "syntax" in r.output.lower()

    def test_valid_json_no_warning(self, tmp_path):
        r = tool_write_file(tmp_path, "ok.json", '{"key": "val"}')
        assert r.ok
        assert "syntax error" not in r.output.lower()


# ---------------------------------------------------------------------------
#  tool_edit_file
# ---------------------------------------------------------------------------

class TestEditFile:
    def test_replaces_first_occurrence(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo = 1\nfoo = 2\n")
        r = tool_edit_file(tmp_path, "code.py", old="foo = 1", new="bar = 1")
        assert r.ok
        content = f.read_text()
        assert "bar = 1" in content
        assert "foo = 2" in content  # second occurrence unchanged

    def test_error_when_string_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\n")
        r = tool_edit_file(tmp_path, "code.py", old="not_there", new="replacement")
        assert not r.ok
        assert "not found" in r.output.lower()

    def test_error_when_file_not_found(self, tmp_path):
        r = tool_edit_file(tmp_path, "missing.py", old="x", new="y")
        assert not r.ok

    def test_path_traversal_rejected(self, tmp_path):
        r = tool_edit_file(tmp_path, "../outside.py", old="x", new="y")
        assert not r.ok

    def test_multiple_occurrences_note_in_output(self, tmp_path):
        f = tmp_path / "multi.py"
        f.write_text("x = 1\nx = 1\nx = 1\n")
        r = tool_edit_file(tmp_path, "multi.py", old="x = 1", new="y = 1")
        assert r.ok
        assert "more occurrence" in r.output.lower() or "2 more" in r.output

    def test_syntax_error_surfaced_after_edit(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo(): pass\n")
        r = tool_edit_file(tmp_path, "code.py", old="def foo(): pass", new="def foo(")
        assert r.ok  # write still succeeds
        assert "syntax" in r.output.lower()


class TestEditFileWhitespaceTolerant:
    """A model routinely reconstructs the `old` snippet with slightly different
    whitespace (a wrapped line collapsed to one line, a different indent, a
    trailing space). A byte-exact-only match rejects those legitimate edits and
    the coder is left unable to edit the file at all. edit_file therefore falls
    back to a whitespace-tolerant match, but ONLY when it is unique, so it can
    never silently edit the wrong one of two candidate regions."""

    def test_collapsed_line_wrap_still_matches(self, tmp_path):
        # File has the snippet across two physical lines; the model sends it as
        # one line (the hard wrap collapsed to a space). It must still land.
        f = tmp_path / "code.py"
        f.write_text(
            "a = 1\n"
            "value = compute(first_argument,\n"
            "                second_argument)\n"
            "b = 2\n"
        )
        r = tool_edit_file(
            tmp_path, "code.py",
            old="value = compute(first_argument, second_argument)",
            new="value = compute(only_argument)",
        )
        assert r.ok
        assert "ignoring whitespace" in r.output.lower()
        content = f.read_text()
        assert "value = compute(only_argument)" in content
        # Surrounding lines are untouched.
        assert content.startswith("a = 1\n")
        assert content.endswith("b = 2\n")

    def test_different_indentation_still_matches(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def f():\n        return 1\n")   # 8-space indent on disk
        r = tool_edit_file(
            tmp_path, "code.py",
            old="    return 1",   # model used a 4-space indent
            new="    return 2",
        )
        assert r.ok
        assert "return 2" in f.read_text()

    def test_ambiguous_tolerant_match_refuses_to_guess(self, tmp_path):
        # Two regions match `old` once whitespace is relaxed and neither is an
        # exact match, so the tool refuses rather than picking one.
        original = "a = foo( x )\nb = foo(  x  )\n"   # 1-space and 2-space forms
        f = tmp_path / "code.py"
        f.write_text(original)
        # Precondition: `old` (3 spaces) matches NEITHER region exactly, so the
        # tolerant path runs rather than the exact fast path.
        assert "foo(   x   )" not in original
        r = tool_edit_file(tmp_path, "code.py", old="foo(   x   )", new="foo(y)")
        assert not r.ok
        assert "not found" in r.output.lower()
        # File is untouched by a refused edit.
        assert f.read_text() == original

    def test_exact_match_wins_over_whitespace_variant(self, tmp_path):
        # An exact occurrence and a whitespace-variant occurrence both exist.
        # The exact one is the one edited.
        f = tmp_path / "code.py"
        f.write_text("gap =  1\ngap = 1\n")   # line 1 has two spaces, line 2 one
        r = tool_edit_file(tmp_path, "code.py", old="gap = 1", new="gap = 42")
        assert r.ok
        assert "ignoring whitespace" not in r.output.lower()
        content = f.read_text()
        assert content == "gap =  1\ngap = 42\n"   # only the exact match changed


# ---------------------------------------------------------------------------
#  tool_list_dir
# ---------------------------------------------------------------------------

class TestListDir:
    def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "file.py").write_text("x")
        (tmp_path / "subdir").mkdir()
        r = tool_list_dir(tmp_path, ".")
        assert r.ok
        assert "file.py" in r.output
        assert "subdir/" in r.output

    def test_path_traversal_rejected(self, tmp_path):
        r = tool_list_dir(tmp_path, "../..")
        assert not r.ok

    def test_not_a_directory_returns_error(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("content")
        r = tool_list_dir(tmp_path, "afile.txt")
        assert not r.ok

    def test_nonexistent_path_returns_error(self, tmp_path):
        r = tool_list_dir(tmp_path, "no_such_dir")
        assert not r.ok

    def test_shows_file_sizes(self, tmp_path):
        (tmp_path / "big.bin").write_bytes(b"\x00" * 2048)
        r = tool_list_dir(tmp_path, ".")
        assert r.ok
        assert "big.bin" in r.output
        assert "k" in r.output or "B" in r.output


# ---------------------------------------------------------------------------
#  tool_tree
# ---------------------------------------------------------------------------

class TestTree:
    def test_recursive_output(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass")
        (tmp_path / "README.md").write_text("hi")
        r = tool_tree(tmp_path, ".", max_depth=3)
        assert r.ok
        assert "src/" in r.output
        assert "main.py" in r.output
        assert "README.md" in r.output

    def test_max_depth_respected(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "hidden.py").write_text("x")
        r = tool_tree(tmp_path, ".", max_depth=2)
        assert r.ok
        assert "hidden.py" not in r.output

    def test_ignores_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"")
        r = tool_tree(tmp_path)
        assert r.ok
        assert "__pycache__" not in r.output

    def test_path_traversal_rejected(self, tmp_path):
        r = tool_tree(tmp_path, "../..")
        assert not r.ok
