"""
Tests for search tools in localm.plugins.coder.tools:
  tool_search_files, tool_grep, tool_search_replace
"""

import pytest
from pathlib import Path

from localm.plugins.coder.tools import (
    tool_search_files,
    tool_grep,
    tool_search_replace,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project(tmp_path):
    """A small fake project tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("import main\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n\nDocs here.\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"key": "value"}\n', encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
#  tool_search_files
# ---------------------------------------------------------------------------

class TestSearchFiles:
    def test_finds_python_files(self, project):
        r = tool_search_files(project, "**/*.py")
        assert r.ok
        assert "main.py" in r.output
        assert "utils.py" in r.output

    def test_glob_limited_to_subdir(self, project):
        r = tool_search_files(project, "*.py", path="src")
        assert r.ok
        assert "main.py" in r.output
        assert "README.md" not in r.output

    def test_no_match_returns_success_with_message(self, project):
        r = tool_search_files(project, "**/*.rs")
        assert r.ok
        assert "no files matched" in r.output.lower()

    def test_summary_includes_count(self, project):
        r = tool_search_files(project, "**/*.py")
        assert r.ok
        assert any(c.isdigit() for c in r.summary)

    def test_path_traversal_on_path_arg_rejected(self, project):
        r = tool_search_files(project, "*.py", path="../../..")
        assert not r.ok

    def test_pattern_traversal_filtered(self, project):
        # A pattern that after joining could escape cwd via ../
        # Results outside cwd should be silently excluded, not returned
        r = tool_search_files(project, "../../*.py")
        assert r.ok
        # Either no matches (all filtered) or only inside-cwd matches
        if r.output.strip() and "No files matched" not in r.output:
            for line in r.output.strip().splitlines():
                p = Path(line.strip())
                if p.is_absolute():
                    assert p.is_relative_to(project)


# ---------------------------------------------------------------------------
#  tool_grep
# ---------------------------------------------------------------------------

class TestGrep:
    def test_finds_pattern_in_files(self, project):
        r = tool_grep(project, "def main")
        assert r.ok
        assert "main.py" in r.output
        assert "def main" in r.output

    def test_no_match_returns_ok_with_no_matches(self, project):
        r = tool_grep(project, "zzz_definitely_not_there_zzz")
        assert r.ok
        assert "0 match" in r.summary

    def test_regex_pattern(self, project):
        r = tool_grep(project, r"def \w+")
        assert r.ok
        assert r.output  # should have matches

    def test_invalid_regex_returns_error(self, project):
        r = tool_grep(project, "[unclosed")
        assert not r.ok
        assert "invalid regex" in r.output.lower()

    def test_glob_filter_limits_files(self, project):
        r = tool_grep(project, "import", glob="tests/**/*.py")
        assert r.ok
        assert "test_main.py" in r.output
        # src/main.py should not appear (only tests/ was searched)
        assert "src" + ("/" if "/" in r.output else "\\") + "main.py" not in r.output

    def test_context_lines_included(self, project):
        r = tool_grep(project, "return 42", context=1)
        assert r.ok
        # Should show 1 line of context around the match
        assert "helper" in r.output  # line before "return 42" is "def helper():"

    def test_case_insensitive_matching(self, project):
        r = tool_grep(project, "PROJECT", path="README.md")
        assert r.ok
        assert "Project" in r.output

    def test_path_traversal_rejected(self, project):
        r = tool_grep(project, "x", path="../../..")
        assert not r.ok

    def test_single_file_target(self, project):
        r = tool_grep(project, "key", path="data.json")
        assert r.ok
        assert "key" in r.output


# ---------------------------------------------------------------------------
#  tool_search_replace
# ---------------------------------------------------------------------------

class TestSearchReplace:
    def test_replaces_pattern_in_files(self, project):
        r = tool_search_replace(
            project, pattern=r"def helper\(\)", replacement="def utility()",
            glob="**/*.py"
        )
        assert r.ok
        content = (project / "src" / "utils.py").read_text()
        assert "def utility()" in content

    def test_dry_run_does_not_modify(self, project):
        original = (project / "src" / "main.py").read_text()
        r = tool_search_replace(
            project, pattern="def main", replacement="def entrypoint",
            glob="**/*.py", dry_run=True
        )
        assert r.ok
        assert "dry-run" in r.output.lower()
        # File should be unchanged
        assert (project / "src" / "main.py").read_text() == original

    def test_dry_run_reports_matches(self, project):
        r = tool_search_replace(
            project, pattern=r"def \w+", replacement="def X",
            glob="**/*.py", dry_run=True
        )
        assert r.ok
        assert any(c.isdigit() for c in r.summary)

    def test_no_matches_returns_ok(self, project):
        r = tool_search_replace(
            project, pattern="NOTHING_MATCHES_THIS_XYZ", replacement="ignored"
        )
        assert r.ok
        assert "0 match" in r.output.lower() or "no matches" in r.output.lower()

    def test_invalid_regex_returns_error(self, project):
        r = tool_search_replace(project, pattern="[broken", replacement="x")
        assert not r.ok
        assert "invalid regex" in r.output.lower()

    def test_back_references_in_replacement(self, project):
        (project / "src" / "main.py").write_text("foo = 42\n")
        r = tool_search_replace(
            project, pattern=r"(\w+) = (\d+)", replacement=r"\2 = \1",
            glob="src/main.py"
        )
        assert r.ok
        content = (project / "src" / "main.py").read_text()
        assert "42 = foo" in content

    def test_glob_limits_files_replaced(self, project):
        r = tool_search_replace(
            project, pattern="import main", replacement="import app",
            glob="tests/*.py"
        )
        assert r.ok
        assert "import app" in (project / "tests" / "test_main.py").read_text()
        # src files should be untouched
        assert "import main" not in (project / "src" / "main.py").read_text()

    def test_multiple_files_replaced(self, project):
        r = tool_search_replace(
            project, pattern=r"def \w+\(\):", replacement="def replaced():",
            glob="src/*.py"
        )
        assert r.ok
        # Both src files should be modified
        main_c = (project / "src" / "main.py").read_text()
        utils_c = (project / "src" / "utils.py").read_text()
        assert "def replaced():" in main_c
        assert "def replaced():" in utils_c
