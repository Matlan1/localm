# SPDX-License-Identifier: AGPL-3.0-or-later
"""Goal mode: iterate the task until a verification command exits 0.

Success is judged by the command's exit code (an un-gameable oracle), not the
model, so it cannot declare premature success."""

from __future__ import annotations

from pathlib import Path

import localm.plugins.coder.cli as _cli
from ..agent import Agent
from ..display import print_error, print_info, print_success, print_warning
from ..verify import (
    inconclusive_reason as _inconclusive_reason,
    is_inconclusive as _is_inconclusive,
    launch_failed as _launch_failed,
    run_verify as _run_verify,
    verify_feedback as _goal_feedback,
)

# _run_verify and _goal_feedback live in ../verify.py (the agent loop runs the same
# check at its pre-done boundary in REPL/GUI sessions) and are re-bound here under
# their original names: the CLI package re-exports them, and _run_goal_loop below
# reads the name live so a patched `cli._run_verify` takes effect.
__all__ = ["_run_verify", "_goal_feedback", "_goal_task_wrap", "_run_goal_loop"]


def _goal_task_wrap(task: str, until_cmd: str) -> str:
    return (
        f"{task}\n\n"
        f"This task is verified by running `{until_cmd}`, which must exit 0. "
        "Fix the underlying code until it passes. Do NOT modify the verification "
        "target itself (the tests or the check) to force a pass - that defeats the "
        "purpose. After you finish, the command is run for you and any failure is "
        "reported back for another attempt."
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
        outcome = _run_verify(until_cmd, work_dir)
        code, output = outcome
        did_not_start = _launch_failed(outcome)
        if code == 0:
            print_success(
                f"Goal met: `{until_cmd}` exited 0 after {attempt} iteration(s).")
            return True, response
        if _is_inconclusive(until_cmd, code, did_not_start):
            # The check never ran (could not start, or collected nothing). No
            # code change can affect that, so iterating on it would burn the
            # whole budget asking the model to fix a condition it cannot reach.
            # Not success either: nothing was verified, and for a caller reading
            # the exit code that has to stay distinct from a green run.
            #
            # Narrow: a missing script or a missing dependency surfaces as an
            # ordinary non-zero exit, NOT as this branch, and keeps its retries -
            # the model has a shell and can create the script, chmod +x, or
            # install the dep.
            print_error(
                f"Goal NOT verified: `{until_cmd}` "
                f"{_inconclusive_reason(until_cmd, code, did_not_start)} "
                f"(exit {code}) after {attempt} iteration(s). Nothing was "
                "actually checked, so this is reported as unverified rather "
                "than as a pass or as a code failure to fix.")
            return False, response
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
