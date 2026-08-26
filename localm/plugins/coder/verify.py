# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verification oracle: run a check command and read its exit code.

This is the un-gameable core of goal mode. The HARNESS runs the command in a
subprocess and judges success solely by its exit code, so the model cannot
declare a premature success no matter what it claims in prose.

The agent loop runs the same check at its pre-done boundary in interactive REPL
and GUI sessions (``agent/loop.py``). ``cli/goal.py`` imports and re-exports
these names, so ``cli._run_verify`` / ``cli._goal_feedback`` stay exactly where
the CLI and its tests expect them.
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


class VerifyOutcome(tuple):
    """``(exit_code, output)``, plus ``launch_failed``.

    Callers unpack it as a 2-tuple. ``launch_failed`` says the command never
    STARTED, which :func:`run_verify` knows directly (it caught the OSError) and
    which no exit code carries: npm returns 127 when a test script's binary is
    missing, a check that very much RAN and FAILED, and `npm test` is exactly
    what auto-detection produces.

    A plain tuple (a test double patching ``cli._run_verify``) has no such
    attribute, and :func:`launch_failed` reads it as False."""

    def __new__(cls, code: int, output: str, launch_failed: bool = False):
        outcome = super().__new__(cls, (code, output))
        outcome.launch_failed = launch_failed
        return outcome


def launch_failed(outcome) -> bool:
    """Did *outcome*'s command fail to start? False for anything not saying so."""
    return bool(getattr(outcome, "launch_failed", False))


def run_verify(command: VerifyCommand, work_dir: Path,
               timeout: int = VERIFY_TIMEOUT_S) -> VerifyOutcome:
    """Run the verification *command* in *work_dir*; return (exit_code, output).

    A STRING command is run through the platform shell wrapper (the same list
    form the run_shell tool uses), so a compound command like 'pytest -x && ruff
    check' works. A LIST command is executed directly as argv, which is how
    auto-detected commands arrive (see :func:`detect_verify_command`) - joining
    them into a shell string would break on an interpreter path containing
    spaces. Either way the harness - not the model - runs it, so its exit code is
    an un-gameable judge of success. On a timeout, any output the command
    produced before the kill is preserved."""
    result = run_subprocess(command, work_dir, timeout=timeout,
                            shell_wrap=isinstance(command, str))
    if result.timed_out:
        return VerifyOutcome(124, (
            "verification command timed out after %ds%s"
            % (timeout, _partial_on_timeout(result))))
    if result.not_found or result.error is not None:
        # The launch itself raised, so this is the ONE place that knows, first
        # hand, that nothing ran. It is recorded on the outcome; 125 remains a
        # display code, never the evidence.
        return VerifyOutcome(125, "failed to run verification command: %s" % (
            result.error or "command not found"), launch_failed=True)
    return VerifyOutcome(result.returncode,
                         (result.stdout or "") + (result.stderr or ""))


def command_text(command: VerifyCommand) -> str:
    """Render *command* for display in a prompt or a console message."""
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def verify_feedback(until_cmd: VerifyCommand, code: int, output: str) -> str:
    """The failure message fed back to the model for another fix attempt.

    Carries the anti-gaming instruction: every feedback message tells the model
    not to weaken the check."""
    tail = (output or "").strip()[-4000:] or "(no output)"
    return (
        f"The verification command `{command_text(until_cmd)}` failed with exit "
        f"code {code}. Output:\n{tail}\n\n"
        "Diagnose the real cause and fix it. Do not modify the check itself."
    )


def is_inconclusive(command: VerifyCommand, code: int,
                    did_not_start: bool = False) -> bool:
    """True when the check did not actually run anything.

    Two cases. The command never STARTED - which the caller must pass in as
    *did_not_start*, from :func:`launch_failed` on the outcome, because it is
    knowledge only :func:`run_verify` has and no exit code reliably carries. And
    pytest's exit 5 (no tests collected).

    Neither is a failure the model can fix by editing code, and neither is a
    pass. The caller settles the gate, says plainly that nothing was verified,
    and records ``last_verify_state == "inconclusive"`` so a programmatic
    consumer (the GUI's final event) can tell this apart from a green run.
    ``tool_run_tests`` makes the same distinction.

    125/126/127 are NOT treated as inconclusive. A command that ran perfectly
    well can return them - npm exits 127 when a test script's binary is missing,
    and `npm test` is exactly what auto-detection produces - so such a check is
    reported as the failure it looks like and the model gets its retries."""
    return did_not_start or (code == 5 and "pytest" in command_text(command))


def inconclusive_reason(command: VerifyCommand, code: int,
                        did_not_start: bool = False) -> str:
    """Why :func:`is_inconclusive` held, phrased for the user and the log."""
    if did_not_start:
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
    so the oracle runs exactly what the ``run_tests`` tool would run. What is
    added here is the confidence gate: ``_detect_test_runner`` always returns
    something (it falls back to pytest), so without the gate a project with no
    tests at all would get pytest and exit 5 forever.

    The gate asks two things, and BOTH have to hold: does this project have a
    test setup, and can its runner actually be started here (see
    :func:`_has_project_check`).
    """
    from .project_config import ProjectConfigUnreadable, load_project_config
    try:
        configured = load_project_config(cwd).get("verify")
    except ProjectConfigUnreadable as e:
        # Say so rather than quietly auto-detecting: the project may configure a
        # DIFFERENT verify command. A note, not a refusal - this runs mid
        # session.
        from localm.debuglog import logger
        logger.warning("verify: ignoring the project config (%s); falling back "
                       "to auto-detection", e)
        configured = None
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
    that its runner is installed. A check that cannot run is not a check, so the
    answer is no oracle rather than one that returns 125 on every gated turn."""
    if (cwd / "Cargo.toml").is_file():
        # `cargo test` also compiles, so it is never vacuous.
        return _runner_available("cargo")
    if (cwd / "go.mod").is_file():
        return _runner_available("go")     # likewise `go test ./...`
    if (cwd / "package.json").is_file():
        # .exists(), not .is_file(), to match _detect_test_runner's own
        # yarn.lock test exactly: the runner checked here has to be the runner
        # that will be picked there.
        lock = "yarn" if (cwd / "yarn.lock").exists() else "npm"
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
    """True when pytest is importable HERE, in the running localm process.

    The detected python command runs ``sys.executable -m pytest``, so the
    interpreter is always launchable - but with no pytest importable in it the
    check exits 1 with "No module named pytest" on every run, which the model
    cannot fix.

    Same interpreter, but NOT quite the same question: ``-m`` prepends the
    project directory to sys.path, while this asks about the localm process's
    own import state, so a project with a vendored pytest can be declined here
    even though the command would have worked."""
    import importlib.util
    try:
        found = importlib.util.find_spec("pytest") is not None
    except Exception:                                          # noqa: BLE001
        # A broken meta-path finder cannot confirm the runner, so the gate
        # declines.
        found = False
    if not found:
        _log_no_oracle("pytest is not importable by this interpreter")
    return found


def _log_no_oracle(why: str) -> None:
    """Record why detection declined, so an absent gate stays discoverable.

    Without it the user sees no verification line and cannot tell whether the
    project has no check or the runner is missing."""
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
