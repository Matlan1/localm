# SPDX-License-Identifier: AGPL-3.0-or-later
"""Goal mode: iterate the task until a verification command exits 0.

Success is judged by the command's exit code (an un-gameable oracle), not the
model, so it cannot declare premature success."""

from __future__ import annotations

from pathlib import Path

import localm.plugins.coder.cli as _cli
from ..agent import Agent
from ..display import print_error, print_info, print_success, print_warning
from ..tools.base import _partial_on_timeout, run_subprocess

def _run_verify(command: str, work_dir: Path, timeout: int = 600) -> "tuple[int, str]":
    """Run the verification *command* in *work_dir*; return (exit_code, output).

    The command is run through the platform shell wrapper (the same list form the
    run_shell tool uses), so a compound command like 'pytest -x && ruff check'
    works. The harness - not the model - runs it, so its exit code is an
    un-gameable judge of success. On a timeout, any output the command produced
    before the kill is preserved (CODER-2) - this used to be silently dropped,
    unlike the run_shell tool's own timeout handling."""
    result = run_subprocess(command, work_dir, timeout=timeout, shell_wrap=True)
    if result.timed_out:
        return 124, (
            "verification command timed out after %ds%s"
            % (timeout, _partial_on_timeout(result)))
    if result.not_found or result.error is not None:
        return 125, "failed to run verification command: %s" % (
            result.error or "command not found")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _goal_task_wrap(task: str, until_cmd: str) -> str:
    return (
        f"{task}\n\n"
        f"This task is verified by running `{until_cmd}`, which must exit 0. "
        "Fix the underlying code until it passes. Do NOT modify the verification "
        "target itself (the tests or the check) to force a pass - that defeats the "
        "purpose. After you finish, the command is run for you and any failure is "
        "reported back for another attempt."
    )


def _goal_feedback(until_cmd: str, code: int, output: str) -> str:
    tail = (output or "").strip()[-4000:] or "(no output)"
    return (
        f"The verification command `{until_cmd}` failed with exit code {code}. "
        f"Output:\n{tail}\n\n"
        "Diagnose the real cause and fix it. Do not modify the check itself."
    )


def _run_goal_loop(agent: Agent, task: str, until_cmd: str, max_iters: int,
                   work_dir: Path) -> "tuple[bool, str]":
    """Iterate: run the task, run the verify command, feed failures back, until it
    exits 0 or the iteration cap is hit. Returns (success, last_response).

    Success is judged solely by the command's exit code, so the model cannot
    declare a premature success; on exhaustion it reports failure honestly rather
    than papering over it."""
    # Live-attribute access so a test patching cli._run_verify is honoured (the
    # name moved into this submodule when cli.py became a package).
    _run_verify = _cli._run_verify
    print_info(
        f"Goal mode: iterating on the task until `{until_cmd}` exits 0 "
        f"(max {max_iters} iteration(s)).")
    response = agent.run_task(_goal_task_wrap(task, until_cmd))
    for attempt in range(1, max_iters + 1):
        code, output = _run_verify(until_cmd, work_dir)
        if code == 0:
            print_success(
                f"Goal met: `{until_cmd}` exited 0 after {attempt} iteration(s).")
            return True, response
        if attempt == max_iters:
            print_error(
                f"Goal NOT met: `{until_cmd}` still failing (exit {code}) after "
                f"{max_iters} iteration(s). Reporting failure rather than a false "
                "success.")
            return False, response
        print_warning(
            f"Verification failed (exit {code}); fixing and retrying "
            f"({attempt}/{max_iters})...")
        response = agent.continue_task(_goal_feedback(until_cmd, code, output))
    return False, response
