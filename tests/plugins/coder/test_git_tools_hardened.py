# SPDX-License-Identifier: AGPL-3.0-or-later
"""The coder's git tools must not run a program named by a repository's config."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from localm.plugins.coder.tools import git as git_tools

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={**__import__("os").environ, "GIT_AUTHOR_NAME": "t",
                        "GIT_AUTHOR_EMAIL": "t@example.invalid",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})


def _repo(tmp_path: Path, config: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("*.txt diff=evil\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "first")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    with open(repo / ".git" / "config", "a", encoding="utf-8") as fh:
        fh.write(config)
    return repo


_CONFIGS = {
    "fsmonitor": '[core]\n\tfsmonitor = "touch MARK; echo"\n',
    "external": '[diff]\n\texternal = "touch MARK; echo"\n',
    "textconv": '[diff "evil"]\n\ttextconv = "touch MARK; cat"\n',
}


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_read_tools_run_nothing(tmp_path, name):
    repo = _repo(tmp_path, _CONFIGS[name])
    git_tools.tool_git_status(repo)
    git_tools.tool_git_diff(repo)
    git_tools.tool_git_log(repo)
    assert not (repo / "MARK").exists()


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_hardened_runner_alone_runs_nothing(tmp_path, name):
    repo = _repo(tmp_path, _CONFIGS[name])
    git_tools._git(repo, "status", "--short")
    git_tools._git(repo, "diff", "-p")
    git_tools._git(repo, "log", "-p", "--max-count=2")
    assert not (repo / "MARK").exists()


def test_fsmonitor_repo_is_refused_by_the_read_tools(tmp_path):
    repo = _repo(tmp_path, _CONFIGS["fsmonitor"])
    for result in (git_tools.tool_git_status(repo), git_tools.tool_git_diff(repo),
                   git_tools.tool_git_log(repo)):
        assert result.ok is False
        assert "refused" in result.output


def test_clean_repo_still_works(tmp_path):
    repo = _repo(tmp_path, "")
    status = git_tools.tool_git_status(repo)
    assert status.ok and "a.txt" in status.output
    diff = git_tools.tool_git_diff(repo)
    assert diff.ok and "two" in diff.output
    assert git_tools.tool_git_log(repo).ok


def test_unhardened_git_would_have_run_the_driver(tmp_path):
    repo = _repo(tmp_path, _CONFIGS["textconv"])
    subprocess.run(["git", "diff"], cwd=repo, capture_output=True)
    assert (repo / "MARK").exists()


def test_argv_and_env_hardening():
    argv = git_tools.git_argv("diff", "--stat")
    assert argv[:1] == ["git"]
    assert "core.fsmonitor=" in argv and "protocol.ext.allow=never" in argv
    assert "safe.bareRepository=explicit" in argv
    assert argv[argv.index("diff"):] == ["diff", "--no-ext-diff", "--no-textconv", "--stat"]
    assert git_tools.git_argv("status")[-1] == "status"
    assert git_tools.git_env({"PATH": "x", "GIT_EXTERNAL_DIFF": "y"}) == {
        "PATH": "x", "GIT_CONFIG_NOSYSTEM": "1"}


def test_bare_layout_at_project_root_runs_nothing(tmp_path):
    root = tmp_path / "proj"
    for sub in ("objects", "refs"):
        (root / sub).mkdir(parents=True)
    (root / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (root / "config").write_text(
        '[core]\n\tbare = false\n\tworktree = .\n\tfsmonitor = "touch MARK; echo"\n',
        encoding="utf-8")
    git_tools.tool_git_status(root)
    git_tools.tool_git_diff(root)
    git_tools.tool_git_log(root)
    git_tools._git(root, "log", "-p")
    git_tools._git(root, "diff")
    assert not (root / "MARK").exists()
