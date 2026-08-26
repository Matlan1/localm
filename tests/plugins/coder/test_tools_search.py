# SPDX-License-Identifier: AGPL-3.0-or-later
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

    def test_dry_run_reports_the_same_changes_shape_a_real_apply_would(self, project):
        """dry_run's preview and a real apply share ONE matching pass (see
        ToolResult.changes), so patch mode can compute an accurate diff via
        dry_run without a second, possibly-diverging implementation. old must
        be the file's REAL current bytes (nothing was written) and new must be
        what a real apply would write.

        Compared against the file's OWN bytes rather than a hardcoded
        literal: Path.write_text() (the `project` fixture) translates \\n to
        the platform line separator on write, so the raw on-disk bytes
        differ between Windows and Linux even though the logical content
        does not - old_bytes must match whichever this platform produced.
        `name` is compared via as_posix() for the same portability reason:
        ToolResult.changes' path uses this tool's existing native-separator
        convention (matching the report text search_replace produces),
        Windows-native here, and the downstream consumer
        (_record_changed_file) re-derives and normalises it to posix
        regardless. This test is about the raw field's CONTENT, not that
        normalisation."""
        r = tool_search_replace(
            project, pattern="def main", replacement="def entrypoint",
            glob="**/*.py", dry_run=True
        )
        assert r.ok
        assert len(r.changes) == 1
        name, old, new = r.changes[0]
        assert Path(name).as_posix() == "src/main.py"
        assert new == "def entrypoint():\n    pass\n"
        # Confirms "nothing was written" too: the old bytes in `changes`
        # still match what is really on disk.
        assert old == (project / "src" / "main.py").read_bytes()

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
        assert r.changes is None

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


# ---------------------------------------------------------------------------
#  glob traversal confinement (security boundary)
#
#  _confine() guards the `path` argument; the `glob` argument fed to
#  Path.glob() is a separate escape vector (Path.glob("../*") traverses
#  upward). grep (read) and search_replace (write) filter their glob results
#  back to cwd, the same way tool_search_files does.
# ---------------------------------------------------------------------------

class TestGlobTraversalConfinement:
    @pytest.fixture()
    def sandbox(self, tmp_path):
        """A project dir with a SECRET file in the parent (outside cwd)."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "inside.py").write_text("inside_marker = 1\n", encoding="utf-8")
        secret = tmp_path / "secret.txt"
        secret.write_text("TOPSECRET_VALUE\n", encoding="utf-8")
        return proj, secret

    def test_grep_cannot_read_outside_cwd_via_parent_glob(self, sandbox):
        proj, _secret = sandbox
        r = tool_grep(proj, "TOPSECRET", glob="../*")
        assert "TOPSECRET_VALUE" not in r.output
        assert "0 match" in r.summary

    def test_grep_cannot_read_outside_cwd_via_recursive_parent_glob(self, sandbox):
        proj, _secret = sandbox
        r = tool_grep(proj, "TOPSECRET", glob="../**/*")
        assert "TOPSECRET_VALUE" not in r.output

    def test_grep_still_searches_inside_cwd(self, sandbox):
        proj, _secret = sandbox
        r = tool_grep(proj, "inside_marker", glob="**/*")
        assert r.ok
        assert "inside.py" in r.output

    def test_search_replace_cannot_write_outside_cwd_via_parent_glob(self, sandbox):
        proj, secret = sandbox
        tool_search_replace(
            proj, pattern="TOPSECRET_VALUE", replacement="PWNED", glob="../*"
        )
        # The sibling secret must be untouched.
        assert secret.read_text(encoding="utf-8") == "TOPSECRET_VALUE\n"

    def test_search_replace_dry_run_excludes_outside_cwd(self, sandbox):
        proj, _secret = sandbox
        r = tool_search_replace(
            proj, pattern="TOPSECRET_VALUE", replacement="PWNED",
            glob="../*", dry_run=True,
        )
        assert "secret.txt" not in r.output

    def test_search_replace_still_replaces_inside_cwd(self, sandbox):
        proj, _secret = sandbox
        r = tool_search_replace(
            proj, pattern="inside_marker", replacement="renamed_marker", glob="**/*"
        )
        assert r.ok
        assert "renamed_marker" in (proj / "inside.py").read_text(encoding="utf-8")
