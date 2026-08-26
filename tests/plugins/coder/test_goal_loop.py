# SPDX-License-Identifier: AGPL-3.0-or-later
"""Goal mode: the coder iterates on a task until a verification command exits 0,
with the command's exit code (not the model) as the un-gameable judge.
"""

from __future__ import annotations

import pytest

import localm.plugins.coder.cli as cli


# --------------------------------------------------------------------------- #
#  _run_verify: real subprocess, exit code + output                           #
# --------------------------------------------------------------------------- #

def test_run_verify_reports_exit_code(tmp_path):
    assert cli._run_verify("exit 0", tmp_path)[0] == 0
    assert cli._run_verify("exit 7", tmp_path)[0] == 7


def test_run_verify_captures_output(tmp_path):
    code, out = cli._run_verify("echo marker_text", tmp_path)
    assert code == 0
    assert "marker_text" in out


# --------------------------------------------------------------------------- #
#  _run_goal_loop: iterate until the command passes, or stop at the cap       #
# --------------------------------------------------------------------------- #

class _FakeAgent:
    def __init__(self):
        self.run_task_calls: list = []
        self.continue_calls: list = []

    def run_task(self, task):
        self.run_task_calls.append(task)
        return "did the task"

    def continue_task(self, message):
        self.continue_calls.append(message)
        return "applied a fix"


def test_goal_loop_passes_on_first_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_verify", lambda cmd, wd: (0, "ok"))
    agent = _FakeAgent()
    success, response = cli._run_goal_loop(agent, "do it", "pytest -x", 5, tmp_path)
    assert success is True
    assert len(agent.run_task_calls) == 1
    assert agent.continue_calls == []           # passed first try -> no fix turn


def test_goal_loop_fixes_then_passes(tmp_path, monkeypatch):
    seq = iter([(1, "1 failed"), (0, "all passed")])
    monkeypatch.setattr(cli, "_run_verify", lambda cmd, wd: next(seq))
    agent = _FakeAgent()
    success, _ = cli._run_goal_loop(agent, "do it", "pytest -x", 5, tmp_path)
    assert success is True
    assert len(agent.continue_calls) == 1       # one failure fed back, then a pass
    assert "pytest -x" in agent.continue_calls[0]
    assert "1 failed" in agent.continue_calls[0]


def test_goal_loop_gives_up_honestly_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_verify", lambda cmd, wd: (1, "still failing"))
    agent = _FakeAgent()
    success, _ = cli._run_goal_loop(agent, "do it", "pytest -x", 3, tmp_path)
    assert success is False
    # 3 verify attempts; a fix turn after the 1st and 2nd failure, none after the
    # 3rd (the cap) -> 2 fix turns.
    assert len(agent.run_task_calls) == 1
    assert len(agent.continue_calls) == 2


def test_goal_loop_stops_at_once_when_the_check_could_not_run(tmp_path,
                                                              monkeypatch):
    """X4 in the --until path: the command never STARTED, so no fix turn can
    reach it. Iterating burns the whole budget asking the model to fix a
    condition it cannot touch.

    The launch fact rides on the outcome, so this builds a real VerifyOutcome
    rather than a bare tuple; a bare tuple means "it ran", which the companion
    test below covers."""
    from localm.plugins.coder.verify import VerifyOutcome
    calls = []

    def _verify(cmd, wd):
        calls.append(cmd)
        return VerifyOutcome(125, "failed to run: [WinError 2]",
                             launch_failed=True)

    monkeypatch.setattr(cli, "_run_verify", _verify)
    agent = _FakeAgent()
    success, _ = cli._run_goal_loop(agent, "do it", "npm test", 5, tmp_path)
    assert success is False              # nothing was verified
    assert len(calls) == 1               # and never retried
    assert agent.continue_calls == []    # no fix turn


def test_goal_loop_still_retries_a_command_not_found_that_ran(tmp_path,
                                                              monkeypatch):
    """The control for the test above. Exit 127 from a check that DID start (a
    shell whose script is missing, npm whose test binary is missing) is fixable
    by the model - it has a shell and can create the script, chmod +x, or
    install the dependency - so the loop must still retry it."""
    monkeypatch.setattr(cli, "_run_verify",
                        lambda cmd, wd: (127, "sh: ./check.sh: not found"))
    agent = _FakeAgent()
    success, _ = cli._run_goal_loop(agent, "do it", "./check.sh", 3, tmp_path)
    assert success is False
    assert len(agent.continue_calls) == 2       # retried


def test_goal_task_wrap_forbids_editing_the_check():
    wrapped = cli._goal_task_wrap("add a feature", "pytest -x")
    assert "add a feature" in wrapped
    assert "pytest -x" in wrapped
    assert "do not modify" in wrapped.lower()


# --------------------------------------------------------------------------- #
#  CLI wiring                                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def test_cli_until_requires_a_task(home, monkeypatch):
    from click.testing import CliRunner
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(PluginManager, "is_active", lambda self, name: True)
    r = CliRunner().invoke(cli.main, ["--until", "pytest"])
    assert r.exit_code != 0
    assert "requires a TASK" in r.output
