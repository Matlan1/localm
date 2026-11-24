# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder tools answer truthfully about a path or a remote that is not there.

A search of a folder that does not exist said "0 matches", the tree repeated its
limit line, a call certain to be refused was asked about first, and git_push on a
new branch failed although the tool says it pushes the current branch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from localm.plugins.coder.tools.files import (
    edit_call_refusal, tool_edit_file, tool_edit_files, tool_grep,
    tool_search_files, tool_tree,
)
from localm.plugins.coder.tools import git as git_tools
from localm.plugins.coder.tools.git import (
    tool_git_create_branch, tool_git_diff, tool_git_log, tool_git_push,
)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_IDENT = ("-c", "user.email=t@example.invalid", "-c", "user.name=t")


def _git(*args, cwd: Path) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, creationflags=_NO_WINDOW)
    return done.stdout


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (root / "empty").mkdir()
    for i in range(12):
        (root / "sub" / f"f{i}.txt").write_text("x", encoding="utf-8")
    return root


@pytest.fixture
def repo(tmp_path):
    """A work repository with one commit, its remote 'origin' a local bare repository."""
    bare = tmp_path / "bare.git"
    work = tmp_path / "work"
    _git("init", "-q", "--bare", str(bare), cwd=tmp_path)
    _git("init", "-q", str(work), cwd=tmp_path)
    (work / "a.txt").write_text("one\n", encoding="utf-8")
    _git("add", "a.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "first", cwd=work)
    _git("remote", "add", "origin", str(bare), cwd=work)
    return work, bare


# --------------------------------------------------------------------------
# grep and search_files on a path that is not there
# --------------------------------------------------------------------------

def test_grep_on_a_missing_path_says_the_path_is_not_found(project):
    res = tool_grep(project, "^def ", "P3/")
    assert not res.ok
    assert "Path not found" in res.output
    assert "No matches" not in res.output


def test_grep_on_an_existing_empty_folder_still_reports_no_matches(project):
    res = tool_grep(project, "^def ", "empty")
    assert res.ok
    assert "No matches" in res.output


def test_grep_on_a_file_still_finds_its_match(project):
    res = tool_grep(project, "^def ", "a.py")
    assert res.ok
    assert "def f" in res.output


def test_search_files_on_a_missing_path_says_the_path_is_not_found(project):
    res = tool_search_files(project, "*.py", "P3/")
    assert not res.ok
    assert "Path not found" in res.output


def test_search_files_on_a_file_says_it_is_not_a_folder(project):
    res = tool_search_files(project, "*.py", "a.py")
    assert not res.ok
    assert "Not a directory" in res.output


def test_search_files_in_an_existing_folder_with_no_match_still_says_so(project):
    res = tool_search_files(project, "*.rs", "sub")
    assert res.ok
    assert res.output == "No files matched."


# --------------------------------------------------------------------------
# tree: one limit line
# --------------------------------------------------------------------------

def test_tree_prints_its_limit_line_once_for_a_nested_folder(project):
    res = tool_tree(project, ".", max_files=5)
    assert res.ok
    assert res.output.count("file limit 5 reached") == 1
    assert res.output.rstrip().splitlines()[-1].startswith("    ... (file limit 5")


def test_tree_prints_no_limit_line_when_nothing_was_left_out(project):
    res = tool_tree(project, ".", max_files=13)
    assert "file limit" not in res.output


def test_tree_names_the_limit_when_the_cut_falls_at_the_top_level(project):
    for i in range(4):
        (project / f"top{i}.txt").write_text("x", encoding="utf-8")
    res = tool_tree(project, ".", max_files=14)
    assert res.output.count("file limit 14 reached") == 1


# --------------------------------------------------------------------------
# a call certain to be refused is recognised before anyone is asked
# --------------------------------------------------------------------------

def _doomed_cases(project):
    return [
        ("edit_files", {"edits": [{"path": "a.py", "old": "", "new": "# x\n"}]}),
        ("edit_files", {"edits": [{"path": "missing.py", "old": "x", "new": "y"}]}),
        ("edit_files", {"edits": [{"path": "a.py", "old": "no such text", "new": "y"}]}),
        ("edit_files", {"edits": []}),
        ("edit_file", {"path": "a.py", "old": "", "new": "# x\n"}),
        ("edit_file", {"path": "missing.py", "old": "x", "new": "y"}),
        ("edit_file", {"path": "a.py", "old": "no such text", "new": "y"}),
    ]


def test_edit_call_refusal_gives_the_tools_own_answer(project):
    for name, args in _doomed_cases(project):
        refusal = edit_call_refusal(project, name, args)
        assert refusal, (name, args)
        real = (tool_edit_files(project, args["edits"]) if name == "edit_files"
                else tool_edit_file(project, **args))
        assert not real.ok
        assert real.output == refusal, (name, args)


def test_edit_call_refusal_lets_a_call_that_can_run_through(project):
    ok_files = {"edits": [{"path": "a.py", "old": "return 1", "new": "return 2"}]}
    ok_file = {"path": "a.py", "old": "return 1", "new": "return 2"}
    assert edit_call_refusal(project, "edit_files", ok_files) is None
    assert edit_call_refusal(project, "edit_file", ok_file) is None
    assert edit_call_refusal(project, "write_file", {"path": "x", "content": ""}) is None


def test_edit_call_refusal_writes_nothing(project):
    before = (project / "a.py").read_bytes()
    edit_call_refusal(project, "edit_files",
                      {"edits": [{"path": "a.py", "old": "return 1", "new": "return 2"}]})
    assert (project / "a.py").read_bytes() == before


# --------------------------------------------------------------------------
# git_push
# --------------------------------------------------------------------------

def test_push_of_a_new_branch_with_no_arguments_sets_the_upstream_and_says_so(repo):
    work, bare = repo
    assert tool_git_create_branch(work, "fix-add").ok
    res = tool_git_push(work)
    assert res.ok, res.output
    assert "Upstream set" in res.output
    assert "fix-add" in _git("branch", "--list", "fix-add", cwd=bare)
    upstream = _git("rev-parse", "--abbrev-ref", "fix-add@{upstream}", cwd=work).strip()
    assert upstream == "origin/fix-add"


def test_push_of_a_branch_that_has_an_upstream_changes_no_configuration(repo):
    work, _bare = repo
    tool_git_create_branch(work, "fix-add")
    assert tool_git_push(work).ok
    (work / "a.txt").write_text("two\n", encoding="utf-8")
    _git("add", "a.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "second", cwd=work)
    res = tool_git_push(work)
    assert res.ok, res.output
    assert "Upstream set" not in res.output


def test_push_with_a_named_branch_sets_no_upstream(repo):
    work, bare = repo
    tool_git_create_branch(work, "fix-add")
    res = tool_git_push(work, branch="fix-add")
    assert res.ok, res.output
    assert "Upstream set" not in res.output
    assert "fix-add" in _git("branch", "--list", "fix-add", cwd=bare)


def test_push_in_a_repository_with_no_remote_says_there_is_none(tmp_path):
    work = tmp_path / "solo"
    _git("init", "-q", str(work), cwd=tmp_path)
    (work / "a.txt").write_text("x", encoding="utf-8")
    _git("add", "a.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "first", cwd=work)
    res = tool_git_push(work)
    assert not res.ok
    assert "no remote" in res.output
    assert "git remote add" in res.output


def test_push_to_a_remote_name_the_repository_lacks_lists_the_ones_it_has(repo):
    work, bare = repo
    _git("remote", "add", "backup", str(bare), cwd=work)
    res = tool_git_push(work, remote="upstream")
    assert not res.ok
    assert "no remote named 'upstream'" in res.output
    assert "origin" in res.output and "backup" in res.output


def test_push_with_several_remotes_and_no_origin_asks_for_a_choice(repo):
    work, bare = repo
    _git("remote", "rename", "origin", "first", cwd=work)
    _git("remote", "add", "second", str(bare), cwd=work)
    res = tool_git_push(work)
    assert not res.ok
    assert "no remote named 'origin'" in res.output
    assert "first" in res.output and "second" in res.output


def test_push_from_a_detached_head_without_a_branch_says_so(repo):
    work, bare = repo
    _git("checkout", "-q", "--detach", cwd=work)
    res = tool_git_push(work)
    assert not res.ok
    assert res.output.startswith("git push refused: HEAD is detached")
    assert "branch=<name>" in res.output
    assert _git("branch", "-a", cwd=bare).strip() == ""


def test_push_from_a_detached_head_with_a_named_branch_is_tried(repo):
    work, _bare = repo
    _git("checkout", "-q", "--detach", cwd=work)
    res = tool_git_push(work, branch="HEAD:refs/heads/from-detached")
    assert res.ok, res.output


def test_a_rejected_push_says_why_and_that_the_remote_is_unchanged(repo, tmp_path):
    work, bare = repo
    branch = _git("symbolic-ref", "--short", "HEAD", cwd=work).strip()
    assert tool_git_push(work).ok
    other = tmp_path / "other"
    _git("clone", "-q", str(bare), str(other), cwd=tmp_path)
    (other / "b.txt").write_text("x", encoding="utf-8")
    _git("add", "b.txt", cwd=other)
    _git(*_IDENT, "commit", "-q", "-m", "ahead", cwd=other)
    _git("push", "-q", "origin", branch, cwd=other)
    (work / "c.txt").write_text("y", encoding="utf-8")
    _git("add", "c.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "diverged", cwd=work)
    res = tool_git_push(work)
    assert not res.ok
    assert "git push was rejected: the remote has commits" in res.output
    assert "Nothing was changed on the remote" in res.output


def test_a_push_that_changed_one_branch_does_not_say_nothing_changed(repo, tmp_path):
    work, bare = repo
    branch = _git("symbolic-ref", "--short", "HEAD", cwd=work).strip()
    _git("branch", "side", cwd=work)
    _git("push", "-q", "-u", "origin", branch, "side", cwd=work)
    other = tmp_path / "other"
    _git("clone", "-q", "-b", branch, str(bare), str(other), cwd=tmp_path)
    (other / "b.txt").write_text("x", encoding="utf-8")
    _git("add", "b.txt", cwd=other)
    _git(*_IDENT, "commit", "-q", "-m", "ahead", cwd=other)
    _git("push", "-q", "origin", branch, cwd=other)
    (work / "c.txt").write_text("y", encoding="utf-8")
    _git("add", "c.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "diverged", cwd=work)
    _git("branch", "-f", "side", "HEAD", cwd=work)
    _git("config", "push.default", "matching", cwd=work)
    before = _git("rev-parse", "side", cwd=bare)
    res = tool_git_push(work)
    assert not res.ok
    assert _git("rev-parse", "side", cwd=bare) != before
    assert "git push was rejected: the remote has commits" in res.output
    assert "Nothing was changed" not in res.output


def test_pushing_an_existing_tag_is_not_called_behind(repo):
    work, bare = repo
    _git("tag", "v1", cwd=work)
    assert tool_git_push(work, branch="v1").ok
    (work / "a.txt").write_text("two\n", encoding="utf-8")
    _git("add", "a.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "second", cwd=work)
    _git("tag", "-f", "v1", cwd=work)
    before = _git("rev-parse", "v1", cwd=bare)
    res = tool_git_push(work, branch="v1")
    assert not res.ok
    assert "already exists" in res.output
    assert res.output.startswith("git push failed. Nothing was changed on the remote.")
    assert "the remote has commits" not in res.output
    assert _git("rev-parse", "v1", cwd=bare) == before


def test_a_declined_branch_named_after_a_git_reason_is_not_called_behind(repo):
    work, bare = repo
    hook = bare / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'fetch first' >&2\nexit 1\n", encoding="utf-8",
                    newline="\n")
    hook.chmod(0o755)
    assert tool_git_create_branch(work, "non-fast-forward").ok
    res = tool_git_push(work)
    assert not res.ok
    assert "pre-receive hook declined" in res.output
    assert "the remote has commits" not in res.output


def test_a_push_declined_by_a_remote_hook_is_not_called_behind(repo):
    work, bare = repo
    hook = bare / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'closed for pushes' >&2\nexit 1\n", encoding="utf-8",
                    newline="\n")
    hook.chmod(0o755)
    res = tool_git_push(work)
    assert not res.ok
    assert "pre-receive hook declined" in res.output
    assert "closed for pushes" in res.output
    assert res.output.startswith("git push failed. Nothing was changed on the remote.")
    assert "was rejected" not in res.output
    assert "the remote has commits" not in res.output


def test_a_failing_pre_push_hook_is_reported_as_git_says_it(repo, monkeypatch):
    work, bare = repo
    monkeypatch.setattr(git_tools, "_HARDENING_CONFIG", tuple(
        item for item in git_tools._HARDENING_CONFIG
        if not item.startswith("core.hooksPath=")))
    hook = work / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\necho 'lint says no' >&2\nexit 1\n", encoding="utf-8",
                    newline="\n")
    hook.chmod(0o755)
    res = tool_git_push(work)
    assert not res.ok
    assert "lint says no" in res.output
    assert "failed to push some refs" in res.output
    assert res.output.startswith("git push failed.\n")
    assert "the remote has commits" not in res.output
    assert _git("branch", "-a", cwd=bare).strip() == ""


def test_a_push_never_runs_the_projects_own_pre_push_hook(repo):
    work, bare = repo
    marker = work.parent / "PRE_PUSH_RAN"
    hook = work / ".git" / "hooks" / "pre-push"
    hook.write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\nexit 1\n", encoding="utf-8",
                    newline="\n")
    hook.chmod(0o755)
    res = tool_git_push(work)
    assert res.ok, res.output
    assert not marker.exists()


def test_a_url_remote_is_not_mistaken_for_a_missing_name(repo, tmp_path):
    work, bare = repo
    res = tool_git_push(work, remote=str(bare), branch="HEAD")
    assert res.ok, res.output


# --------------------------------------------------------------------------
# git_diff and git_log on a path that does not exist
# --------------------------------------------------------------------------

def test_git_diff_of_an_unknown_path_says_the_path_is_not_found(repo):
    work, _bare = repo
    res = tool_git_diff(work, path="nope/x.py")
    assert not res.ok
    assert "Path not found" in res.output


def test_git_diff_of_an_unchanged_file_still_says_no_output(repo):
    work, _bare = repo
    res = tool_git_diff(work, path="a.txt")
    assert res.ok
    assert res.output == "(no output)"


def test_git_log_of_an_unknown_path_says_the_path_is_not_found(repo):
    work, _bare = repo
    res = tool_git_log(work, path="nope/x.py")
    assert not res.ok
    assert "Path not found" in res.output


def test_git_log_of_a_deleted_file_still_shows_its_history(repo):
    work, _bare = repo
    _git("rm", "-q", "a.txt", cwd=work)
    _git(*_IDENT, "commit", "-q", "-m", "remove", cwd=work)
    res = tool_git_log(work, path="a.txt")
    assert res.ok
    assert "remove" in res.output
