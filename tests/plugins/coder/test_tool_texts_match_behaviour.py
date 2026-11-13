# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder tool descriptions say what the tools do, and a tool that writes a file
into the project asks first like a file write."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from localm.plugins.coder.agent.constants import (
    _PATCH_MODE_ELIGIBLE_TOOLS,
    _PATCH_MODE_REFUSED_TOOLS,
)
from localm.plugins.coder.tools.registry import TOOL_REGISTRY

_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _repo(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    repo.mkdir()
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "base"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       **flags)
    return repo


class TestAToolThatWritesAFileAsks:
    def test_every_tool_with_an_output_path_is_a_confirmed_write(self):
        writers = [name for name, tool in TOOL_REGISTRY.items()
                   if "output_path" in tool.params]
        assert "browser_screenshot" in writers and "generate_image" in writers
        assert [n for n in writers if not TOOL_REGISTRY[n].destructive] == []
        assert [n for n in writers if n not in _PATCH_MODE_REFUSED_TOOLS
                and n not in _PATCH_MODE_ELIGIBLE_TOOLS] == []


@_git
class TestGitTools:
    def test_create_branch_without_checkout_says_it_was_created(self, tmp_path,
                                                                monkeypatch):
        from localm.plugins.coder.tools.git import tool_git_create_branch
        repo = _repo(tmp_path, monkeypatch)
        res = tool_git_create_branch(repo, "feature-x", checkout=False)
        assert res.ok, res.output
        assert res.output == "Branch 'feature-x' created."

    def test_commit_without_files_includes_new_files_as_described(self, tmp_path,
                                                                   monkeypatch):
        from localm.plugins.coder.tools.git import _git as run_git
        from localm.plugins.coder.tools.git import tool_git_commit
        repo = _repo(tmp_path, monkeypatch)
        (repo / "new.txt").write_text("x\n", encoding="utf-8")
        res = tool_git_commit(repo, "add new")
        assert res.ok, res.output
        out, ok = run_git(repo, "show", "--name-only", "--format=", "HEAD")
        assert ok and "new.txt" in out
        assert "untracked files included" in TOOL_REGISTRY["git_commit"].description


class TestGrepSaysItIgnoresCase:
    def test_description_and_behaviour_agree(self, tmp_path):
        from localm.plugins.coder.tools.files import tool_grep
        (tmp_path / "a.py").write_text("Hello\n", encoding="utf-8")
        res = tool_grep(tmp_path, "hello")
        assert "Hello" in res.output
        assert "ignoring case" in TOOL_REGISTRY["grep"].description


class TestRunTestsNamesItsRunner:
    @pytest.mark.parametrize("cmd, label", [
        ([sys.executable, "-m", "pytest", "--tb=short", "-q"], "pytest"),
        (["C:/tools/node/npm.CMD", "test", "--", "--passWithNoTests"], "npm test"),
        (["/usr/bin/cargo", "test", "--color=never"], "cargo test"),
        (["/usr/bin/make"], "make"),
    ])
    def test_label(self, cmd, label):
        from localm.plugins.coder.tools.shell import _runner_label
        assert _runner_label(cmd) == label
