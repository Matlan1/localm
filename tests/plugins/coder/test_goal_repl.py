# SPDX-License-Identifier: AGPL-3.0-or-later
"""The REPL's /goal command and its plain-text routing.

cli/goal.py's _run_goal_loop already iterates a task until a verify command
exits 0 (test_goal_loop.py covers the loop itself; test_verify_oracle.py covers
the in-run exit-code gate --until and /verify both use). Before this file, that
iterating loop was reachable only from the non-interactive `--until` CLI path;
the REPL had no equivalent, so a plain-text message always got exactly one
agent.chat() turn.

This file covers the REPL wiring: /goal (status/off/auto/explicit), mirroring
how /verify is shaped, and the plain-text dispatch in _repl() itself - with
goal_cmd unset a message still gets a single chat() turn (unchanged default
behaviour); with goal_cmd set it is routed through _run_goal_loop instead.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from localm.plugins.coder.audit import SessionMode


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / "home"
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    return h


def _proj(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    return p


# --------------------------------------------------------------------------- #
#  Agent construction: goal_cmd defaults off, mirroring verify_cmd's shape    #
# --------------------------------------------------------------------------- #

def test_goal_cmd_defaults_to_off(home, tmp_path):
    agent = _agent(_proj(tmp_path))
    assert agent.goal_cmd is None
    assert agent.goal_max_iters == 5


# --------------------------------------------------------------------------- #
#  /goal: status / off / auto / explicit                                      #
# --------------------------------------------------------------------------- #

def test_goal_no_arg_reports_off_by_default(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_info",
                        lambda msg: called.setdefault("msg", msg))
    repl_mod._handle_command("/goal", agent)
    assert "(off)" in called["msg"]
    assert agent.goal_cmd is None


def test_goal_no_arg_reports_the_current_command_and_iteration_cap(
        home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    agent.goal_cmd = "pytest -x"
    agent.goal_max_iters = 7
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_info",
                        lambda msg: called.setdefault("msg", msg))
    repl_mod._handle_command("/goal", agent)
    assert "pytest -x" in called["msg"]
    assert "7" in called["msg"]
    # Unlike /verify, a failure re-runs the WHOLE task rather than ending the
    # turn - the status message must say so, not just show the command.
    assert "whole task" in called["msg"].lower()


def test_goal_explicit_command_sets_goal_cmd(home, tmp_path):
    agent = _agent(_proj(tmp_path))
    from localm.plugins.coder.cli import repl as repl_mod
    repl_mod._handle_command("/goal pytest -x", agent)
    assert agent.goal_cmd == "pytest -x"
    assert agent.goal_max_iters == 5   # untouched by /goal itself


def test_goal_off_clears_goal_cmd(home, tmp_path):
    agent = _agent(_proj(tmp_path))
    agent.goal_cmd = "pytest -x"
    from localm.plugins.coder.cli import repl as repl_mod
    repl_mod._handle_command("/goal off", agent)
    assert agent.goal_cmd is None


def test_goal_auto_uses_the_project_configured_check(home, tmp_path):
    proj = _proj(tmp_path)
    (proj / ".localcoder").mkdir()
    (proj / ".localcoder" / "config.toml").write_text('verify = "make check"\n',
                                                       encoding="utf-8")
    agent = _agent(proj)
    from localm.plugins.coder.cli import repl as repl_mod
    repl_mod._handle_command("/goal auto", agent)
    assert agent.goal_cmd == "make check"


def test_goal_auto_reports_none_found_and_leaves_goal_cmd_off(
        home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))   # empty project, no test setup at all
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_warning",
                        lambda msg: called.setdefault("msg", msg))
    repl_mod._handle_command("/goal auto", agent)
    assert agent.goal_cmd is None
    assert "No obvious check" in called["msg"]


def test_goal_command_is_in_slash_command_list_and_help():
    from localm.plugins.coder.cli.repl import _SLASH_COMMANDS
    assert "/goal" in _SLASH_COMMANDS
    from localm.plugins.coder.display import HELP_TEXT
    assert "/goal" in HELP_TEXT


# --------------------------------------------------------------------------- #
#  _repl(): plain-text dispatch                                               #
# --------------------------------------------------------------------------- #

def _drive_repl_with_one_message(repl_mod, agent, monkeypatch, message):
    """Feed _repl() exactly one plain-text message, then end the session as if
    the user hit Ctrl+D - the same EOFError _read_multiline already lets
    propagate to _repl's own except clause."""
    inputs = iter([message, EOFError()])

    def fake_input(prompt=""):
        nxt = next(inputs)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(repl_mod.console, "input", fake_input)
    repl_mod._repl(agent)


def test_plain_text_uses_chat_when_goal_is_off(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    assert agent.goal_cmd is None

    chat_calls = []
    monkeypatch.setattr(agent, "chat",
                        lambda msg: chat_calls.append(msg) or "done")

    from localm.plugins.coder.cli import repl as repl_mod
    goal_calls = []
    monkeypatch.setattr(repl_mod, "_run_goal_loop",
                        lambda *a, **kw: goal_calls.append((a, kw)))

    _drive_repl_with_one_message(repl_mod, agent, monkeypatch, "hello agent")

    assert chat_calls == ["hello agent"]
    assert goal_calls == []


def test_plain_text_uses_goal_loop_when_armed(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    agent.goal_cmd = "pytest -x"
    agent.goal_max_iters = 3

    chat_calls = []
    monkeypatch.setattr(agent, "chat",
                        lambda msg: chat_calls.append(msg) or "done")

    from localm.plugins.coder.cli import repl as repl_mod
    goal_calls = []

    def _fake_goal_loop(a, task, cmd, max_iters, work_dir):
        goal_calls.append((a, task, cmd, max_iters, work_dir))
        return (True, "ok")

    monkeypatch.setattr(repl_mod, "_run_goal_loop", _fake_goal_loop)

    _drive_repl_with_one_message(repl_mod, agent, monkeypatch, "fix the bug")

    assert chat_calls == []
    assert len(goal_calls) == 1
    a, task, cmd, max_iters, work_dir = goal_calls[0]
    assert a is agent
    assert task == "fix the bug"
    assert cmd == "pytest -x"
    assert max_iters == 3
    assert work_dir == agent.cwd


def test_slash_command_is_never_routed_through_goal_loop(home, tmp_path,
                                                          monkeypatch):
    """A /command line must still go through _handle_command, never be treated
    as a task even when goal mode is armed."""
    agent = _agent(_proj(tmp_path))
    agent.goal_cmd = "pytest -x"

    from localm.plugins.coder.cli import repl as repl_mod
    goal_calls = []
    monkeypatch.setattr(repl_mod, "_run_goal_loop",
                        lambda *a, **kw: goal_calls.append((a, kw)))
    called = {}
    monkeypatch.setattr(repl_mod, "print_info",
                        lambda msg: called.setdefault("msg", msg))

    _drive_repl_with_one_message(repl_mod, agent, monkeypatch, "/cwd")

    assert goal_calls == []
    assert str(agent.cwd) in called["msg"]
