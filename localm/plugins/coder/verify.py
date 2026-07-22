# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verification oracle: run a check command and read its exit code.

This is the un-gameable core of goal mode. The HARNESS runs the command in a
subprocess and judges success solely by its exit code, so the model cannot
declare a premature success no matter what it claims in prose.

The primitives live here rather than in ``cli/goal.py`` because the oracle is no
longer CLI-only: the agent loop runs the same check at its pre-done boundary in
interactive REPL and GUI sessions (``agent/loop.py``). ``cli/goal.py`` imports
and re-exports these names, so ``cli._run_verify`` / ``cli._goal_feedback`` stay
exactly where the CLI and its tests expect them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .tools.base import _partial_on_timeout, run_subprocess

# A command is either a shell string (user-supplied: pipes, &&, redirects all
# work) or an argv list (auto-detected: no shell quoting to get wrong, which
# matters because the Python runner is sys.executable and that path can contain
# spaces).
VerifyCommand = Union[str, list]

# How long a single verification run may take before it is killed.
VERIFY_TIMEOUT_S = 600


def run_verify(command: VerifyCommand, work_dir: Path,
               timeout: int = VERIFY_TIMEOUT_S) -> "tuple[int, str]":
    """Run the verification *command* in *work_dir*; return (exit_code, output).

    A STRING command is run through the platform shell wrapper (the same list
    form the run_shell tool uses), so a compound command like 'pytest -x && ruff
    check' works. A LIST command is executed directly as argv, which is how
    auto-detected commands arrive (see :func:`detect_verify_command`) - joining
    them into a shell string would break on an interpreter path containing
    spaces. Either way the harness - not the model - runs it, so its exit code is
    an un-gameable judge of success. On a timeout, any output the command
    produced before the kill is preserved (CODER-2) - this used to be silently
    dropped, unlike the run_shell tool's own timeout handling."""
    result = run_subprocess(command, work_dir, timeout=timeout,
                            shell_wrap=isinstance(command, str))
    if result.timed_out:
        return 124, (
            "verification command timed out after %ds%s"
            % (timeout, _partial_on_timeout(result)))
    if result.not_found or result.error is not None:
        return 125, "failed to run verification command: %s" % (
            result.error or "command not found")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def command_text(command: VerifyCommand) -> str:
    """Render *command* for display in a prompt or a console message."""
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def verify_feedback(until_cmd: VerifyCommand, code: int, output: str) -> str:
    """The failure message fed back to the model for another fix attempt.

    Carries the anti-gaming instruction: a weak model's cheapest route to a green
    check is to weaken the check, so every feedback message says not to."""
    tail = (output or "").strip()[-4000:] or "(no output)"
    return (
        f"The verification command `{command_text(until_cmd)}` failed with exit "
        f"code {code}. Output:\n{tail}\n\n"
        "Diagnose the real cause and fix it. Do not modify the check itself."
    )


# Exit codes meaning the command never started: 125 is what run_verify returns
# when the launch itself raised, 126 ("found but not executable") and 127 ("not
# found") are what a POSIX shell returns for a shell-string command.
#
# Honest limit, because no exit code can fix it: Windows `cmd /C <missing>`
# returns 1, which is indistinguishable from a genuine test failure (measured).
# Auto-detected commands are argv lists, so their launch failure surfaces as a
# FileNotFoundError -> 125 and IS distinguishable; a user-supplied shell string
# on Windows is not, and is left to be reported as the failure it looks like.
LAUNCH_FAILURE_CODES = frozenset({125, 126, 127})


def is_launch_failure(code: int) -> bool:
    """True when *code* means the verification command could not be started."""
    return code in LAUNCH_FAILURE_CODES


def is_inconclusive(command: VerifyCommand, code: int) -> bool:
    """True when *code* means "the check did not actually run anything".

    Two cases: the command could not be STARTED at all (see
    :data:`LAUNCH_FAILURE_CODES`), and pytest's exit 5 (no tests collected).
    Neither is a failure the model can fix by editing code, so looping on one
    would burn every retry to no purpose; neither is a pass either, so neither
    may be reported as one. "Not reported as a pass" is concrete: the caller
    settles the gate, says plainly that nothing was verified, and records
    ``last_verify_state == "inconclusive"`` so a programmatic consumer (the GUI's
    final event) can tell this apart from a green run rather than reading an
    unqualified ok. ``tool_run_tests`` makes the same distinction for the same
    reason."""
    return is_launch_failure(code) or (
        code == 5 and "pytest" in command_text(command))


def inconclusive_reason(command: VerifyCommand, code: int) -> str:
    """Why :func:`is_inconclusive` held, phrased for the user and the log."""
    if is_launch_failure(code):
        return "could not run"
    return "collected no tests"


# ------------------------------------------------------------------ #
#  Auto-detection: is there an obvious project check to run?           #
# ------------------------------------------------------------------ #

def detect_verify_command(cwd: Path) -> Optional[VerifyCommand]:
    """The project's obvious check command, or None when there isn't one.

    Resolution order:

    1. An explicit ``verify`` key in ``.localcoder/config.toml`` - the user's own
       choice, used verbatim (as a shell string).
    2. An auto-detected test runner, but ONLY when the project shows positive
       evidence of having a test setup (see :func:`_has_project_check`).

    The command SHAPE is delegated to ``tools/shell.py``'s ``_detect_test_runner``
    so the oracle runs exactly what the ``run_tests`` tool would run - one
    detection, not two that can drift apart. What is added here is the confidence
    gate: ``_detect_test_runner`` always returns something (it falls back to
    pytest), which is right for a tool the model asked to call but wrong for an
    automatic gate. Without the gate, a project with no tests at all would get
    pytest, exit 5 forever, and the oracle would look broken rather than absent.

    The gate asks two things, and BOTH have to hold: does this project have a
    test setup, and can its runner actually be started here. The second is not
    theoretical - it is why this repository's own `npm test` became an oracle
    that could only ever return "failed to run" (see :func:`_has_project_check`).
    """
    from .project_config import load_project_config
    configured = load_project_config(cwd).get("verify")
    if configured:
        return str(configured)
    if not _has_project_check(cwd):
        return None
    from .tools.shell import _detect_test_runner
    return _detect_test_runner(cwd)


def _has_project_check(cwd: Path) -> bool:
    """Positive evidence that *cwd* has a RUNNABLE test setup.

    The branch ORDER mirrors ``_detect_test_runner`` exactly, so whichever branch
    accepts here is the same one that will pick the command. Any change to that
    function's precedence must be mirrored here.

    Runnable, not merely present: a project file proves the project's SHAPE, not
    that its runner is installed. Without the second half, this repo's own
    package.json detected `npm test` on a box where an argv-list npm cannot
    start, the oracle returned 125 on every gated turn, and the model was asked
    to fix a condition no code change can reach. A check that cannot run is not
    a check, so the answer is no oracle, not a doomed one."""
    if (cwd / "Cargo.toml").is_file():
        # `cargo test` also compiles, so it is never vacuous.
        return _runner_available("cargo")
    if (cwd / "go.mod").is_file():
        return _runner_available("go")     # likewise `go test ./...`
    if (cwd / "package.json").is_file():
        lock = "yarn" if (cwd / "yarn.lock").is_file() else "npm"
        return _npm_has_test_script(cwd) and _runner_available(lock)
    return _has_python_tests(cwd) and _pytest_importable()


def _runner_available(name: str) -> bool:
    """True when *name* resolves to something argv-list execution can launch."""
    from .tools.shell import resolve_runner
    if resolve_runner(name) is not None:
        return True
    _log_no_oracle("%s is not on PATH" % name)
    return False


def _pytest_importable() -> bool:
    """True when pytest can be imported by the interpreter the check would use.

    The detected python command runs ``sys.executable -m pytest``, so the
    interpreter is always launchable - but with no pytest importable in it the
    check exits 1 with "No module named pytest" on every run. That is the same
    unfixable-by-the-model condition as a missing binary, only wearing an exit
    code that looks like a real test failure, which makes it worse. It is a live
    case rather than a hypothetical one: localm's own interpreter is what runs
    the coder, and a user's Python project is not obliged to share it."""
    import importlib.util
    try:
        found = importlib.util.find_spec("pytest") is not None
    except Exception:                                          # noqa: BLE001
        # A broken meta-path finder cannot confirm the runner, and an oracle we
        # cannot vouch for is the thing this gate exists to prevent.
        found = False
    if not found:
        _log_no_oracle("pytest is not importable by this interpreter")
    return found


def _log_no_oracle(why: str) -> None:
    """Record why detection declined, so an absent gate stays discoverable.

    Silence here would be the hidden problem: the user sees no verification line
    and cannot tell whether the project has no check or the runner is missing."""
    import logging
    logging.getLogger(__name__).debug(
        "no verification oracle: %s", why)


def _npm_has_test_script(cwd: Path) -> bool:
    """True when package.json actually defines a ``test`` script.

    File existence is not enough: `npm test` on a package.json without one fails
    with "missing script: test" on every run, which the model cannot fix."""
    import json
    try:
        data = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
    except Exception:
        return False         # unreadable/malformed -> no evidence, not a check
    scripts = data.get("scripts")
    return isinstance(scripts, dict) and bool(scripts.get("test"))


def _has_python_tests(cwd: Path) -> bool:
    """True when the project looks like it has a pytest suite: a pytest config
    section, or test files where pytest would look for them."""
    for name, marker in (("pytest.ini", None),
                         ("pyproject.toml", "[tool.pytest"),
                         ("tox.ini", "[pytest]"),
                         ("setup.cfg", "[tool:pytest]")):
        path = cwd / name
        if not path.is_file():
            continue
        if marker is None:
            return True
        try:
            if marker in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    for directory in (cwd, cwd / "tests", cwd / "test"):
        if not directory.is_dir():
            continue
        try:
            for entry in directory.iterdir():
                if not entry.is_file():
                    continue
                if entry.name.startswith("test_") and entry.suffix == ".py":
                    return True
                if entry.name.endswith("_test.py"):
                    return True
        except OSError:
            continue
    return False
