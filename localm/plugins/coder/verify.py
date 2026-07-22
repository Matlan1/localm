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


def is_inconclusive(command: VerifyCommand, code: int) -> bool:
    """True when *code* means "the check did not actually run anything".

    Today that is only pytest's exit 5 (no tests collected). It is NOT a failure
    the model can fix by editing code, so looping on it would burn every retry to
    no purpose; it is NOT a pass either, so it must never be reported as one.
    The caller treats it as inconclusive and says so (see agent/loop.py).
    ``tool_run_tests`` makes the same distinction for the same reason."""
    return code == 5 and "pytest" in command_text(command)


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
    """Positive evidence that *cwd* has a runnable test setup.

    The branch ORDER mirrors ``_detect_test_runner`` exactly, so whichever branch
    accepts here is the same one that will pick the command. Any change to that
    function's precedence must be mirrored here."""
    if (cwd / "Cargo.toml").is_file():
        return True          # `cargo test` also compiles, so it is never vacuous
    if (cwd / "go.mod").is_file():
        return True          # likewise `go test ./...`
    if (cwd / "package.json").is_file():
        return _npm_has_test_script(cwd)
    return _has_python_tests(cwd)


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
