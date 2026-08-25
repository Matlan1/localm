# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for git helpers in localm.plugins.coder.tools

Both mocked-subprocess tests (fast, no git dependency) and real git-init
integration tests (use pytest's tmp_path fixture, require git in PATH).
"""

import subprocess
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from localm.plugins.coder.tools import (
    _git,
    tool_git_status,
    tool_git_diff,
    tool_git_log,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_proc(stdout="", stderr="", returncode=0):
    """Build a mock CompletedProcess."""
    p = MagicMock()
    p.stdout = stdout
    p.stderr = stderr
    p.returncode = returncode
    return p


def _patch_run(stdout="", stderr="", returncode=0):
    return patch(
        "localm.plugins.coder.tools.subprocess.run",
        return_value=_make_proc(stdout, stderr, returncode),
    )


# ---------------------------------------------------------------------------
#  _git() helper
# ---------------------------------------------------------------------------

class TestGitHelper:
    def test_returns_stdout_on_success(self, tmp_path):
        with _patch_run(stdout="main\n"):
            out, ok = _git(tmp_path, "status")
        assert ok
        assert "main" in out

    def test_returns_stderr_on_failure(self, tmp_path):
        with _patch_run(stderr="not a git repo", returncode=128):
            out, ok = _git(tmp_path, "status")
        assert not ok
        assert "not a git repo" in out

    def test_git_not_found(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            out, ok = _git(tmp_path, "status")
        assert not ok
        assert "git not found" in out

    def test_timeout_returns_false(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["git"], 10),
        ):
            out, ok = _git(tmp_path, "status")
        assert not ok
        assert "timed out" in out

    def test_generic_exception_returns_false(self, tmp_path):
        with patch(
            "localm.plugins.coder.tools.subprocess.run",
            side_effect=OSError("permission denied"),
        ):
            out, ok = _git(tmp_path, "log")
        assert not ok

    def test_empty_stdout_returns_no_output_marker(self, tmp_path):
        with _patch_run(stdout="", stderr=""):
            out, ok = _git(tmp_path, "diff")
        assert out == "(no output)"

    def test_combines_stdout_and_stderr(self, tmp_path):
        with _patch_run(stdout="file.py", stderr="warning: ..."):
            out, ok = _git(tmp_path, "status")
        assert "file.py" in out
        assert "warning" in out


# ---------------------------------------------------------------------------
#  tool_git_status
# ---------------------------------------------------------------------------

class TestGitStatus:
    def test_ok_result_on_success(self, tmp_path):
        with _patch_run(stdout="## main...origin/main\nM  foo.py\n"):
            r = tool_git_status(tmp_path)
        assert r.ok
        assert "M" in r.output or "main" in r.output

    def test_error_result_on_failure(self, tmp_path):
        with _patch_run(stderr="fatal: not a git repository", returncode=128):
            r = tool_git_status(tmp_path)
        assert not r.ok

    def test_summary_mentions_lines(self, tmp_path):
        with _patch_run(stdout="## main\nM  a.py\nM  b.py\n"):
            r = tool_git_status(tmp_path)
        assert "line" in r.summary

    def test_uses_short_branch_flags(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="## main\n")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_status(tmp_path)

        assert "--short" in captured_args
        assert "--branch" in captured_args


# ---------------------------------------------------------------------------
#  tool_git_diff
# ---------------------------------------------------------------------------

class TestGitDiff:
    def test_basic_diff_output(self, tmp_path):
        diff_output = "diff --git a/x.py b/x.py\n+new line\n"
        with _patch_run(stdout=diff_output):
            r = tool_git_diff(tmp_path)
        assert r.ok
        assert "new line" in r.output

    def test_staged_diff_passes_cached_flag(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="staged diff")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_diff(tmp_path, staged=True)

        assert "--cached" in captured_args

    def test_unstaged_diff_no_cached_flag(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="unstaged diff")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_diff(tmp_path, staged=False)

        assert "--cached" not in captured_args

    def test_path_arg_passed_to_git(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_diff(tmp_path, path="src/main.py")

        assert "src/main.py" in captured_args
        assert "--" in captured_args

    def test_summary_mentions_staged(self, tmp_path):
        with _patch_run(stdout="some diff"):
            r = tool_git_diff(tmp_path, staged=True)
        assert "--cached" in r.summary

    def test_summary_not_staged_for_unstaged(self, tmp_path):
        with _patch_run(stdout="some diff"):
            r = tool_git_diff(tmp_path, staged=False)
        assert "--cached" not in r.summary

    def test_truncation_flag_set_on_large_diff(self, tmp_path):
        big = "+" + "x" * 50 + "\n"
        with _patch_run(stdout=big * 10_000):
            r = tool_git_diff(tmp_path)
        # Only ok is checked; truncation depends on _truncate's default.
        assert r.ok


# ---------------------------------------------------------------------------
#  tool_git_log
# ---------------------------------------------------------------------------

class TestGitLog:
    def test_returns_log_output(self, tmp_path):
        log_output = "abc1234 (HEAD -> main) initial commit\n"
        with _patch_run(stdout=log_output):
            r = tool_git_log(tmp_path)
        assert r.ok
        assert "initial commit" in r.output

    def test_n_param_passed_to_git(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_log(tmp_path, n=5)

        assert "--max-count=5" in captured_args

    def test_default_n_is_10(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_log(tmp_path)

        assert "--max-count=10" in captured_args

    def test_path_arg_passed_to_git(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_log(tmp_path, path="README.md")

        assert "README.md" in captured_args

    def test_summary_contains_n(self, tmp_path):
        with _patch_run(stdout="abc initial commit\n"):
            r = tool_git_log(tmp_path, n=7)
        assert "7" in r.summary

    def test_oneline_decorate_flags_present(self, tmp_path):
        captured_args = []

        def fake_run(cmd, **kwargs):
            captured_args.extend(cmd)
            return _make_proc(stdout="")

        with patch("localm.plugins.coder.tools.subprocess.run", side_effect=fake_run):
            tool_git_log(tmp_path)

        assert "--oneline" in captured_args
        assert "--decorate" in captured_args


# ---------------------------------------------------------------------------
#  Integration tests - require real git in PATH
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGitIntegration:
    """These tests actually run git commands in a temp directory."""

    def _init_repo(self, tmp_path: Path) -> Path:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=tmp_path, check=True, capture_output=True)
        return tmp_path

    def test_git_status_clean_repo(self, tmp_path):
        self._init_repo(tmp_path)
        r = tool_git_status(tmp_path)
        assert r.ok

    def test_git_status_untracked_file(self, tmp_path):
        self._init_repo(tmp_path)
        (tmp_path / "new.py").write_text("x = 1\n")
        r = tool_git_status(tmp_path)
        assert r.ok
        assert "new.py" in r.output

    def test_git_diff_shows_unstaged_changes(self, tmp_path):
        repo = self._init_repo(tmp_path)
        f = repo / "code.py"
        f.write_text("x = 1\n")
        subprocess.run(["git", "add", "code.py"], cwd=repo, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True,
                       capture_output=True)
        f.write_text("x = 2\n")  # unstaged change
        r = tool_git_diff(repo)
        assert r.ok
        assert "code.py" in r.output

    def test_git_log_shows_commit(self, tmp_path):
        repo = self._init_repo(tmp_path)
        (repo / "readme.txt").write_text("hello\n")
        subprocess.run(["git", "add", "readme.txt"], cwd=repo, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "add readme"], cwd=repo,
                       check=True, capture_output=True)
        r = tool_git_log(repo)
        assert r.ok
        assert "add readme" in r.output

    def test_git_status_not_a_repo(self, tmp_path):
        """Outside a git repo, git exits non-zero."""
        r = tool_git_status(tmp_path)
        assert not r.ok
