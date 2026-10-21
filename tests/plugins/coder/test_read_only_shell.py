# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only shell commands confined to the project run without a prompt."""

from __future__ import annotations

import os
import sys
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
    "git grep -O foo", "git grep --open-files-in-pager=vim foo",
    "python script.py", "npm test", "pytest", "curl x",
    "sed -n 1p a.txt", "awk '{print}' a.txt", "xargs cat", "ls; rm x",
    "cat ~root/.bashrc", "type C:foo", "type a.txt:stream", "tree --output x",
    "ls\nrm -rf x", "ls\r\nrm x", "cat //server/share/x",
    "cat {/etc/passwd,x}", "cat .*/x", "cat ./.*", "cat [ab].txt", "ls ..?",
    "type %USERPROFILE%\\secrets.txt", "", "   ",
])
def test_anything_unproven_is_not_read_only(command):
    assert is_read_only(command, PROJECT) is False, command


def test_no_project_means_nothing_is_read_only():
    assert is_read_only("ls", None) is False


_BS = chr(92)


@pytest.mark.parametrize("command", [
    # The program must be the bare allowlisted name.
    "./tools/grep TODO src", "tools/cat a.txt", "tools" + _BS + "cat.bat a.txt",
    "grep.exe x", "/usr/bin/cat a.txt",
    # Options that run a program, write a file, read a named file or follow
    # symlinks, attached, clustered or abbreviated.
    "git grep -O./x.sh foo", "git grep -nO./x.sh foo", "git grep --open=vim foo",
    "git diff --out=x.patch", "git log --show-signature", "git show --textconv HEAD",
    "rg --hostname-bin=./x.sh foo", "rg --pre=./x.sh foo", "rg -z foo", "rg -L foo",
    "rg --follow foo", "rg -fpatterns.txt foo", "sort -ro a.txt a.txt",
    "sort --out=x a.txt", "sort --files0-from=l.txt", "sort -T tmp a.txt",
    "wc --files0-from=l.txt", "du --files0-from=l.txt", "du -L src",
    "grep -fFILE x", "grep -rf FILE x", "grep --exclude-from=../x y", "grep -R x src",
    "tree -R -H x", "tree -l src", "find -L src", "find src -follow",
    "diff -r a b", "diff --from-file=../x a", "file -C -m x", "ls -LR src",
    "findstr /s x a.txt", "dir /s",
    "git blame -Srevs.txt a.txt", "tree --infofile=x src", "file -z a.gz",
    # Operands that only look like options or switches.
    "cat -- -/../../etc/passwd", "cat /ab", "cat a.txt -/../../etc/passwd",
    "cat a.txt -" + _BS + ".." + _BS + "secret",
    # cmd.exe delayed expansion.
    "type !USERPROFILE!" + _BS + ".ssh" + _BS + "id_rsa",
    # An option value naming a path outside the project.
    "grep --exclude-from=/etc/passwd x src",
    # Quoting cmd.exe and sh read differently.
    "type 'x & rd /s /q build & '", 'type "a' + _BS + '" & rd /s /q build"',
    'type ^" & rd /s /q build ^"', "ls " + _BS + '" ; rm x ; echo ' + _BS + '"',
    "echo " + _BS + "' ; rm x ; echo " + _BS + "'",
])
def test_options_programs_and_quoting_that_escape_the_check(command):
    assert is_read_only(command, PROJECT) is False, command


@pytest.mark.parametrize("command", [
    "grep TODO src", 'grep "^def" src', "grep -rn 'TODO' src", "git diff -U3",
    "rg -n parse src", "grep --color=never TODO src", "sort -k2 data.txt",
    "type src" + _BS + "app.py", "dir /b", "dir /b src", "findstr /i x a.txt",
    "findstr /n x a.txt", "where /q grep", "cd /d src",
])
def test_ordinary_read_only_forms_still_skip_the_prompt(command):
    assert is_read_only(command, PROJECT) is True, command


def _symlink(link: Path, target: Path, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        if directory and sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return
        pytest.skip("symlinks are not available to this account")


def test_a_symlink_leaving_the_project_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "a.txt").write_text("a", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    assert is_read_only("cat src/a.txt", proj) is True
    assert is_read_only("cat src/*", proj) is True
    _symlink(proj / "src" / "link.txt", secret)
    assert is_read_only("cat src/link.txt", proj) is False
    assert is_read_only("cat src/*", proj) is False
    assert is_read_only("wc -l src/*.txt", proj) is False


def test_a_directory_symlink_leaving_the_project_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("x", encoding="utf-8")
    _symlink(proj / "out", outside, directory=True)
    assert is_read_only("cat out/x.txt", proj) is False
    assert is_read_only("cd out && cat x.txt", proj) is False
    assert is_read_only("ls out", proj) is False


def test_dotdot_after_a_directory_link_that_climbs_out_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "a.txt").write_text("a", encoding="utf-8")
    sibling = tmp_path / "sibling"
    (sibling / ".git").mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("s", encoding="utf-8")
    assert is_read_only("cat src/../a.txt", proj) is True
    _symlink(proj / "jup", sibling, directory=True)
    assert is_read_only("cat jup/.git/../../secret.txt", proj) is False
    assert is_read_only("grep s jup/.git/../../secret.txt", proj) is False
    assert is_read_only("cd jup/.git/../.. && cat secret.txt", proj) is False
    assert is_read_only("cat src/../a.txt", proj) is True


def test_dotdot_after_a_link_back_into_the_project_that_climbs_out_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    (proj / "x" / "y").mkdir(parents=True)
    (proj / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "z.txt").write_text("outside", encoding="utf-8")
    _symlink(proj / "x" / "y" / "j", proj, directory=True)
    assert is_read_only("cat x/y/j/z.txt", proj) is True
    assert is_read_only("cat x/y/j/../../../z.txt", proj) is False


def test_a_glob_after_a_link_and_dotdot_is_not_read_only_even_with_no_match_inside(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    sibling = tmp_path / "sibling"
    (sibling / ".git").mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("s", encoding="utf-8")
    _symlink(proj / "jup", sibling, directory=True)
    assert is_read_only("cat jup/.git/../../secre*", proj) is False
    assert is_read_only("cat j*/../../secret.txt", proj) is False


def test_a_long_path_is_checked_without_resolving_every_prefix(tmp_path, monkeypatch):
    import localm.plugins.coder.shell_guard as sg
    proj = tmp_path / "proj"
    proj.mkdir()
    calls = []
    real = os.path.realpath

    def counting(path, *a, **k):
        calls.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(sg.os.path, "realpath", counting)
    assert is_read_only("cat " + "x/" * 3000 + "a.txt", proj) is True
    assert len(calls) < 20, len(calls)


def test_git_after_cd_through_a_link_checks_the_repository_git_uses(tmp_path):
    proj = tmp_path / "proj"
    (proj / "vendor" / "repo2" / "sub").mkdir(parents=True)
    (proj / "vendor" / "repo2" / ".git").mkdir()
    (proj / "vendor" / "repo2" / ".git" / "config").write_text(
        "[core]\n\tpager = ./p.sh\n", encoding="utf-8")
    (proj / "x").mkdir()
    assert is_read_only("cd vendor/repo2/sub && git log", proj) is False
    _symlink(proj / "x" / "j", proj / "vendor" / "repo2" / "sub", directory=True)
    assert is_read_only("cd x && ls", proj) is True
    assert is_read_only("cd x/j && git log", proj) is False


def test_a_gitdir_whose_dotdot_climbs_out_through_a_link_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    (proj / "x" / "y").mkdir(parents=True)
    (proj / "store").mkdir()
    (proj / "store" / "config").write_text("[core]\n\trepositoryformatversion = 0\n",
                                           encoding="utf-8")
    _symlink(proj / "x" / "y" / "j", proj, directory=True)
    (proj / ".git").write_text("gitdir: store\n", encoding="utf-8")
    assert is_read_only("git status", proj) is True
    (proj / ".git").write_text("gitdir: x/y/j/../../../store\n", encoding="utf-8")
    assert is_read_only("git status", proj) is False


def test_a_project_file_that_a_command_name_finds_is_not_read_only(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / "bin").mkdir(parents=True)
    monkeypatch.setenv("PATH", "bin" + os.pathsep + str(proj / "abs"))
    assert is_read_only("grep x a.txt", proj) is True
    (proj / "bin" / "grep").mkdir()
    assert is_read_only("grep x a.txt", proj) is True
    (proj / "bin" / "grep").rmdir()
    (proj / "bin" / "grep").write_text("#!/bin/sh\n", encoding="utf-8")
    assert is_read_only("grep x a.txt", proj) is False
    assert is_read_only("cat a.txt", proj) is True
    (proj / "abs").mkdir()
    (proj / "abs" / "cat").write_text("#!/bin/sh\n", encoding="utf-8")
    assert is_read_only("cat a.txt", proj) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows searches the current folder")
def test_a_batch_file_in_the_folder_shadows_the_command_on_windows(tmp_path):
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    assert is_read_only("cd src && type a.txt", proj) is True
    (proj / "src" / "type.bat").write_text("@echo off\n", encoding="utf-8")
    assert is_read_only("cd src && type a.txt", proj) is False
    (proj / "grep.cmd").write_text("@echo off\n", encoding="utf-8")
    assert is_read_only("grep x a.txt", proj) is False


@pytest.mark.parametrize("config, read_only", [
    ("[core]\n\trepositoryformatversion = 0\n\tfsmonitor = true\n", True),
    ("[core]\n\tfsmonitor = ./hook.sh\n", False),
    ("[core]\n\tpager = ./p.sh\n", False),
    ('[filter "lfs"]\n\tclean = git-lfs clean -- %f\n', False),
    ('[diff "x"]\n\ttextconv = ./t.sh\n', False),
    ("[diff]\n\texternal = ./d.sh\n", False),
    ("[include]\n\tpath = other.cfg\n", False),
    ("[log]\n\tshowSignature = false\n", True),
    ("[log]\n\tshowSignature\n", False),
    ("[gpg]\n\tprogram = ./g.sh\n", False),
    ("[core] fsmonitor = ./hook.sh\n", False),
    ("[core]\n\thooksPath = .husky\n", False),
])
def test_git_config_that_runs_programs_is_not_read_only(tmp_path, config, read_only):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    (proj / ".git" / "config").write_text(config, encoding="utf-8")
    assert is_read_only("git status", proj) is read_only
    assert is_read_only("ls", proj) is True


def test_git_config_is_found_through_a_gitdir_file_and_a_nested_repo(tmp_path):
    proj = tmp_path / "proj"
    (proj / "sub").mkdir(parents=True)
    store = proj / "store"
    store.mkdir()
    (store / "config").write_text("[core]\n\tfsmonitor = ./hook.sh\n", encoding="utf-8")
    (proj / "sub" / ".git").write_text("gitdir: ../store\n", encoding="utf-8")
    assert is_read_only("git status", proj) is True
    assert is_read_only("cd sub && git status", proj) is False
    (proj / ".git").write_text("gitdir: store\n", encoding="utf-8")
    assert is_read_only("git status", proj) is False


def test_a_git_hook_script_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    hooks = proj / ".git" / "hooks"
    hooks.mkdir(parents=True)
    (proj / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (hooks / "pre-commit.sample").write_text("#!/bin/sh\n", encoding="utf-8")
    (hooks / "post-index-change.d").mkdir()
    assert is_read_only("git status", proj) is True
    (hooks / "post-index-change").write_text("#!/bin/sh\n", encoding="utf-8")
    assert is_read_only("git status", proj) is False
    (hooks / "post-index-change").unlink()
    sub_hooks = proj / ".git" / "modules" / "lib" / "hooks"
    sub_hooks.mkdir(parents=True)
    assert is_read_only("git status", proj) is True
    (sub_hooks / "post-index-change").write_text("#!/bin/sh\n", encoding="utf-8")
    assert is_read_only("git status", proj) is False


def test_a_gitdir_outside_the_project_is_not_read(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "elsewhere-w20"
    outside.mkdir()
    (outside / "config").write_text("[core]\n\tfsmonitor = ./hook.sh\n", encoding="utf-8")
    (proj / ".git").write_text(f"gitdir: {outside.as_posix()}\n", encoding="utf-8")
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
    assert is_read_only("git status", proj) is True
    assert opened and not any("elsewhere-w20" in f for f in opened)


@pytest.mark.parametrize("target", ["//w20-no-such-host/share/.git",
                                    _BS * 2 + "w20-no-such-host" + _BS + "share",
                                    _BS * 2 + "?" + _BS + "C:" + _BS + "x"])
def test_a_network_gitdir_is_refused_without_being_touched(tmp_path, monkeypatch, target):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".git").write_text("gitdir: " + target + "\n", encoding="utf-8")
    resolved = []
    real_realpath = os.path.realpath
    monkeypatch.setattr(os.path, "realpath",
                        lambda p, *a, **k: resolved.append(str(p)) or real_realpath(p, *a, **k))
    assert is_read_only("git status", proj) is False
    assert not any("w20-no-such-host" in p or "?" in p for p in resolved)


def test_a_submodule_config_that_runs_programs_is_not_read_only(tmp_path):
    proj = tmp_path / "proj"
    sub = proj / ".git" / "modules" / "lib"
    sub.mkdir(parents=True)
    (proj / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (sub / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    assert is_read_only("git status", proj) is True
    (sub / "config").write_text("[core]\n\tfsmonitor = ../../../hook.sh\n", encoding="utf-8")
    assert is_read_only("git status", proj) is False


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


def test_plan_mode_still_refuses_a_read_only_command(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    agent.set_plan_mode(True)
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "y")
    res = agent._execute_tool(_shell("cat a.txt"), interactive=True)
    assert not res.ok and "hello" not in res.output
    assert prompts == []


@pytest.mark.parametrize("command", [
    "sort -ro a.txt a.txt", "./tools/grep TODO .", "git grep -O./x.sh foo",
])
def test_an_executing_or_writing_form_of_an_allowed_command_asks(tmp_path, monkeypatch,
                                                                command):
    (tmp_path / "a.txt").write_text("b\na\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    prompts = []
    monkeypatch.setattr(display.console, "input",
                        lambda prompt="": prompts.append(prompt) or "n")
    res = agent._execute_tool(_shell(command), interactive=True)
    assert len(prompts) == 1 and not res.ok
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "b\na\n"


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


def test_a_run_that_cannot_ask_still_denies_read_only_commands(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    res = agent._execute_tool(_shell("cat a.txt"), interactive=False)
    assert not res.ok and "requires confirmation" in res.output


def test_gui_session_runs_read_only_without_a_card_and_says_so(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    asked, events = [], []
    agent.confirm_handler = lambda call: asked.append(call.name) or False
    agent.on_event = events.append
    ok = agent._execute_tool(_shell("cat a.txt"), interactive=False)
    denied = agent._execute_tool(_shell("mkdir out"), interactive=False)
    assert ok.ok and asked == ["run_shell"] and not denied.ok
    assert any("ran without asking" in str(e.get("text", "")) for e in events)


@pytest.mark.parametrize("value", ["false", "0", 0, "off", False])
def test_setting_off_spellings(tmp_path, monkeypatch, value):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    agent = _make_agent(tmp_path)
    monkeypatch.setattr("localm.plugins.coder.project_config.load_project_config",
                        lambda cwd: {"auto_allow_read_only_shell": value})
    assert agent._read_only_shell(_shell("cat a.txt")) is False


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
