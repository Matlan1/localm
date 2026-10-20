# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only shell commands confined to the project run without a prompt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder import display
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.shell_guard import is_read_only

PROJECT = Path("/work/proj")


@pytest.mark.parametrize("command", [
    "ls", "ls -la src", "pwd", "cat README.md", "head -n 20 src/app.py",
    "tail -50 logs/build.log", "wc -l src/*.py", "grep -rn TODO src",
    "rg parse_tool_calls", "git status", "git status --short", "git diff",
    "git diff HEAD~1 -- src/app.py", "git log --oneline -5", "git show HEAD",
    "git branch", "git branch -a", "git blame src/app.py", "git rev-parse HEAD",
    "git ls-files", "tree src", "find src -name '*.py'", "cd src && ls",
    "ls | wc -l", "git log --oneline | head -3", "cat a.txt > /dev/null",
    "sort data.txt", "diff a.txt b.txt",
])
def test_read_only_commands_inside_the_project(command):
    assert is_read_only(command, PROJECT) is True, command


@pytest.mark.parametrize("command", [
    "rm -rf build", "touch x", "mkdir out", "echo hi > out.txt", "cat a >> b",
    "git commit -m x", "git checkout main", "git branch new-feature",
    "git branch -D old", "git tag v1", "git remote add o url", "git push",
    "git -C /etc status", "git -c core.pager=sh log", "git diff --output=x.patch",
    "git stash", "git reset --hard", "cat /etc/passwd", "cat ~/.ssh/id_rsa",
    "cat ../other/secret.txt", "ls ..", "cd .. && ls", "cd /tmp && ls",
    "head $HOME/.bashrc", "cat `which python`", "ls $(pwd)", "sudo ls",
    "env ls", "find . -delete", "find . -exec rm {} ;", "sort -o out.txt in.txt",
    "uniq in.txt out.txt", "rg --pre cat x", "rg --pre=cat x",
    "sort --compress-program=gzip a.txt", "sort -oout.txt a.txt", "tree -o out.txt",
    "git grep -O foo", "git grep --open-files-in-pager=vim foo", "python script.py", "npm test", "pytest", "curl x",
    "sed -n 1p a.txt", "awk '{print}' a.txt", "xargs cat", "ls; rm x",
    "type%USERPROFILE%\\secrets.txt", "", "   ",
])
def test_anything_unproven_is_not_read_only(command):
    assert is_read_only(command, PROJECT) is False, command


def test_no_project_means_nothing_is_read_only():
    assert is_read_only("ls", None) is False


def _make_agent(tmp_path, **kw):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "m"
    backend.native_tools = False
    backend.supports_grammar = False
    backend.last_usage = {}
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, auto_approve=False, **kw)
    agent._audit = MagicMock()
    return agent


def _shell(cmd, **kw):
    return ToolCall(name="run_shell", args={"command": cmd}, raw="", start=0, end=0, **kw)


def test_read_only_command_runs_without_asking(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "n")
    res = agent._execute_tool(_shell("cat a.txt"), interactive=True)
    assert res.ok and "hello" in res.output
    assert prompts == []


def test_writing_command_still_asks(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "n")
    res = agent._execute_tool(_shell("mkdir out"), interactive=True)
    assert len(prompts) == 1 and not res.ok
    assert not (tmp_path / "out").exists()


def test_always_confirm_and_lenient_calls_still_ask(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path, always_confirm={"run_shell"})
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "n")
    agent._execute_tool(_shell("cat a.txt"), interactive=True)
    agent.always_confirm = set()
    agent._execute_tool(_shell("cat a.txt", lenient=True), interactive=True)
    assert len(prompts) == 2


def test_unattended_run_still_denies_without_confirmation_unless_read_only(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    ok = agent._execute_tool(_shell("cat a.txt"), interactive=False)
    denied = agent._execute_tool(_shell("mkdir out"), interactive=False)
    assert ok.ok
    assert not denied.ok and "requires confirmation" in denied.output


def test_setting_turns_it_off(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".localcoder").mkdir()
    (tmp_path / ".localcoder" / "config.toml").write_text(
        "auto_allow_read_only_shell = false\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "n")
    agent._execute_tool(_shell("cat a.txt"), interactive=True)
    assert len(prompts) == 1
