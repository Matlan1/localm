# SPDX-License-Identifier: AGPL-3.0-or-later
"""
read_file offset/limit slicing, explicit truncation markers (grep, tree,
read_file), edit_file closest-match hints, read_env redaction and
edit_notebook_cell.
"""

import json

import pytest

from localm.plugins.coder.tools import (
    tool_edit_file,
    tool_edit_notebook_cell,
    tool_grep,
    tool_read_env,
    tool_read_file,
    tool_tree,
)


# ---------------------------------------------------------------------------
#  read_file offset / limit
# ---------------------------------------------------------------------------

class TestReadFileSlicing:
    @pytest.fixture
    def big(self, tmp_path):
        lines = "\n".join(f"line {i}" for i in range(1, 101)) + "\n"
        (tmp_path / "big.txt").write_text(lines, encoding="utf-8")
        return tmp_path

    def test_offset_and_limit_slice(self, big):
        r = tool_read_file(big, "big.txt", offset=40, limit=3)
        assert r.ok
        assert "line 40" in r.output and "line 42" in r.output
        assert "line 39" not in r.output and "line 43" not in r.output
        assert "<lines>40-42 of 101</lines>" in r.output

    def test_offset_without_limit_reads_to_end(self, big):
        r = tool_read_file(big, "big.txt", offset=99)
        assert r.ok and "line 99" in r.output and "line 100" in r.output

    def test_limit_without_offset_starts_at_top(self, big):
        r = tool_read_file(big, "big.txt", limit=2)
        assert r.ok and "line 1" in r.output and "line 3" not in r.output

    def test_offset_past_end_is_clear_error(self, big):
        r = tool_read_file(big, "big.txt", offset=9999)
        assert not r.ok and "past the end" in r.output

    def test_truncated_full_read_hints_at_slicing(self, tmp_path):
        (tmp_path / "huge.txt").write_text("x" * 20_000, encoding="utf-8")
        r = tool_read_file(tmp_path, "huge.txt")
        assert r.ok and r.truncated
        assert "re-read specific parts" in r.output

    def test_plain_read_unchanged(self, tmp_path):
        (tmp_path / "s.txt").write_text("hello\n", encoding="utf-8")
        r = tool_read_file(tmp_path, "s.txt")
        assert r.ok and "hello" in r.output and not r.truncated


# ---------------------------------------------------------------------------
#  grep truncation markers
# ---------------------------------------------------------------------------

class TestGrepMarkers:
    def test_per_file_cap_marker(self, tmp_path):
        (tmp_path / "many.txt").write_text(
            "\n".join("needle here" for _ in range(30)), encoding="utf-8")
        r = tool_grep(tmp_path, "needle", glob="many.txt", context=0)
        assert r.ok
        assert "more match(es) in" in r.output

    def test_file_cap_names_unsearched_count(self, tmp_path):
        # Enough matches in early files to blow the 300-line output cap,
        # with files left over afterwards.
        for i in range(8):
            (tmp_path / f"f{i}.txt").write_text(
                "\n".join("needle" for _ in range(25)), encoding="utf-8")
        r = tool_grep(tmp_path, "needle", glob="*.txt", context=2)
        assert r.ok
        assert "were NOT" in r.output and "narrow the search" in r.output


# ---------------------------------------------------------------------------
#  tree limit message
# ---------------------------------------------------------------------------

class TestTreeLimit:
    def test_limit_message_is_actionable(self, tmp_path):
        for i in range(12):
            (tmp_path / f"file{i:02d}.txt").write_text("x", encoding="utf-8")
        r = tool_tree(tmp_path, max_files=5)
        assert r.ok
        assert "file limit 5 reached" in r.output
        assert "raise max_files" in r.output


# ---------------------------------------------------------------------------
#  edit_file closest-match hint
# ---------------------------------------------------------------------------

class TestEditFileClosestMatch:
    def test_near_miss_shows_the_real_text(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
        # Indentation does not match the file: a near miss.
        r = tool_edit_file(tmp_path, "a.py",
                           old="def greet( name ):", new="def hello(name):")
        assert not r.ok
        assert "Closest match" in r.output
        assert "def greet(name):" in r.output     # the actual file text
        assert "whitespace" in r.output

    def test_totally_unrelated_text_has_no_snippet(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        r = tool_edit_file(tmp_path, "a.txt",
                           old="zzzzqqqq12345", new="x")
        assert not r.ok
        assert "Closest match" not in r.output


# ---------------------------------------------------------------------------
#  read_env redaction
# ---------------------------------------------------------------------------

class TestReadEnvRedaction:
    def test_secret_values_never_appear(self, tmp_path):
        (tmp_path / ".env").write_text(
            "API_KEY=sk-supersecret123\n"
            "DB_PASSWORD=hunter2\n"
            "PLAIN_SETTING=visible-value\n",
            encoding="utf-8")
        r = tool_read_env(tmp_path)
        assert r.ok
        assert "sk-supersecret123" not in r.output
        assert "hunter2" not in r.output
        assert "API_KEY" in r.output            # names stay visible
        assert "visible-value" in r.output      # non-secrets pass through
        assert "redacted" in r.summary

    def test_explicit_missing_file_errors(self, tmp_path):
        r = tool_read_env(tmp_path, path="nope.env")
        assert not r.ok and "not found" in r.output


# ---------------------------------------------------------------------------
#  edit_notebook_cell
# ---------------------------------------------------------------------------

def _notebook(cells):
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


class TestEditNotebookCell:
    @pytest.fixture
    def nb_path(self, tmp_path):
        nb = _notebook([
            {"cell_type": "code", "source": ["import os\n"], "outputs": [],
             "metadata": {}, "execution_count": None},
            {"cell_type": "markdown", "source": ["# Title\n"], "metadata": {}},
        ])
        p = tmp_path / "n.ipynb"
        p.write_text(json.dumps(nb), encoding="utf-8")
        return p

    def test_edit_replaces_cell_source(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 0,
                                    "import sys\nprint(sys.path)")
        assert r.ok
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["cells"][0]["source"] == ["import sys\n", "print(sys.path)"]
        # the other cell is untouched
        assert nb["cells"][1]["source"] == ["# Title\n"]

    def test_cell_type_override(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 1, "plain text",
                                    cell_type="raw")
        assert r.ok
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["cells"][1]["cell_type"] == "raw"

    def test_out_of_range_index(self, nb_path, tmp_path):
        r = tool_edit_notebook_cell(tmp_path, "n.ipynb", 9, "x")
        assert not r.ok and "out of range" in r.output

    def test_not_a_notebook(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        r = tool_edit_notebook_cell(tmp_path, "a.py", 0, "x")
        assert not r.ok and "Not a notebook" in r.output
