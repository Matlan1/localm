# SPDX-License-Identifier: AGPL-3.0-or-later
"""git_commit never commits localm's own session files, and a full-mode
transcript is written where plain git ignores it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.tools.git import PRIVATE_PROJECT_PATHS, tool_git_commit
from tests.plugins.coder.test_session_mode import _make_agent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, **_FLAGS).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
        monkeypatch.setenv(k, v)
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "commit", "-q", "--allow-empty", "-m", "base")
    return r


def _write_private(repo):
    for rel in PRIVATE_PROJECT_PATHS:
        p = repo / rel
        if rel.endswith("sessions"):
            p = p / "2026-01-01_000000.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("secret conversation\n", encoding="utf-8")


def _committed(repo):
    return _git(repo, "show", "--name-only", "--format=", "HEAD").split()


def test_commit_without_files_leaves_session_state_out(repo):
    _write_private(repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert tool_git_commit(repo, "fix").ok
    assert _committed(repo) == ["a.py"]


def test_naming_the_session_files_does_not_commit_them(repo):
    _write_private(repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    res = tool_git_commit(repo, "fix", files=["a.py", ".localcoder", "."])
    assert res.ok
    assert _committed(repo) == ["a.py"]


def test_only_session_state_changed_is_nothing_to_commit(repo):
    _write_private(repo)
    res = tool_git_commit(repo, "fix")
    assert res.ok and "Nothing to commit" in res.output


def test_a_transcript_tracked_earlier_is_not_committed_again(repo):
    _write_private(repo)
    _git(repo, "add", "-f", ".localcoder")
    _git(repo, "commit", "-q", "-m", "old")
    (repo / ".localcoder" / "sessions" / "2026-01-01_000000.md").write_text(
        "more secret\n", encoding="utf-8")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert tool_git_commit(repo, "fix").ok
    assert _committed(repo) == ["a.py"]


def test_other_localcoder_files_are_still_committed(repo):
    (repo / ".localcoder").mkdir()
    (repo / ".localcoder" / "config.toml").write_text("mode = 'full'\n", encoding="utf-8")
    assert tool_git_commit(repo, "cfg").ok
    assert _committed(repo) == [".localcoder/config.toml"]


def test_a_failed_unstage_is_an_error_not_a_commit(repo, monkeypatch):
    from localm.plugins.coder.tools import git as g
    real = g._git

    def fake(cwd, *a, **k):
        if a and a[0] == "reset":
            return "boom", False
        return real(cwd, *a, **k)

    monkeypatch.setattr(g, "_git", fake)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    res = tool_git_commit(repo, "fix")
    assert not res.ok
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "1"


def test_first_commit_in_a_new_repo_works_and_leaves_session_state_out(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}.items():
        monkeypatch.setenv(k, v)
    r = tmp_path / "fresh"
    r.mkdir()
    _git(r, "init", "-q")
    _write_private(r)
    (r / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert tool_git_commit(r, "first").ok
    assert _committed(r) == ["a.py"]


def test_child_worktree_commit_leaves_session_state_out(repo):
    from localm.plugins.coder.tools.git import git_commit_all_in
    _write_private(repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert git_commit_all_in(repo, "child")[1]
    assert _committed(repo) == ["a.py"]


def _staged(repo):
    return _git(repo, "diff", "--cached", "--name-only").split()


def test_full_mode_transcript_is_ignored_by_plain_git_add(repo):
    agent = _make_agent(repo, SessionMode.FULL)
    agent._messages = [{"role": "user", "content": "hi"}]
    agent._turns = 1
    path = agent.close()
    assert path.exists()
    _git(repo, "add", "-A")
    assert _staged(repo) == [".localcoder/.gitignore"]


def test_repl_history_is_ignored_by_plain_git_add(repo, monkeypatch):
    import atexit
    import types

    from localm.plugins.coder.cli import repl

    saved = []
    fake = types.SimpleNamespace(
        set_completer=lambda *a: None, set_completer_delims=lambda *a: None,
        parse_and_bind=lambda *a: None, read_history_file=lambda *a: None,
        set_history_length=lambda *a: None, get_line_buffer=lambda: "",
        write_history_file=lambda path: Path(path).write_text("typed secret\n",
                                                              encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "readline", fake)
    monkeypatch.setattr(atexit, "register", saved.append)
    agent = _make_agent(repo, SessionMode.FULL)
    repl._setup_readline(agent)
    assert saved
    saved[0]()
    assert (repo / ".localcoder" / "repl_history").read_text(encoding="utf-8")
    _git(repo, "add", "-A")
    assert _staged(repo) == [".localcoder/.gitignore"]


def test_ignore_file_keeps_user_lines_and_adds_only_missing_ones(repo):
    from localm.plugins.coder.tools.git import ensure_private_state_ignored
    d = repo / ".localcoder"
    d.mkdir()
    (d / ".gitignore").write_text("mine.txt\n/repl_history", encoding="utf-8")
    ensure_private_state_ignored(repo)
    ensure_private_state_ignored(repo)
    lines = (d / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines == ["mine.txt", "/repl_history", "/sessions/", "/checkpoint.json"]
    (d / "config.toml").write_text("mode = 'full'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert "config.toml" in " ".join(_staged(repo))


def test_history_is_not_persisted_when_it_cannot_be_kept_out_of_git(repo, monkeypatch):
    import atexit
    import types

    from localm.plugins.coder.cli import repl

    saved = []
    fake = types.SimpleNamespace(
        set_completer=lambda *a: None, set_completer_delims=lambda *a: None,
        parse_and_bind=lambda *a: None, read_history_file=lambda *a: None,
        set_history_length=lambda *a: None, get_line_buffer=lambda: "",
        write_history_file=lambda path: None)
    monkeypatch.setitem(sys.modules, "readline", fake)
    monkeypatch.setattr(atexit, "register", saved.append)

    def boom(_p):
        raise OSError("read-only")

    monkeypatch.setattr(repl, "ensure_private_state_ignored", boom)
    repl._setup_readline(_make_agent(repo, SessionMode.FULL))
    assert saved == []
