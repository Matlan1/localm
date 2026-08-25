# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verification oracle: run a check command and read its exit code."""

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
    """``(exit_code, output)``, plus the one fact an exit code cannot carry."""

    def __new__(cls, code: int, output: str, launch_failed: bool = False):
        outcome = super().__new__(cls, (code, output))
        outcome.launch_failed = launch_failed
        return outcome


def launch_failed(outcome) -> bool:
    """Did *outcome*'s command fail to start? False for anything not saying so."""
    return bool(getattr(outcome, "launch_failed", False))


def run_verify(command: VerifyCommand, work_dir: Path,
               timeout: int = VERIFY_TIMEOUT_S) -> VerifyOutcome:
    """Run the verification *command* in *work_dir*; return (exit_code, output)."""
    result = run_subprocess(command, work_dir, timeout=timeout,
                            shell_wrap=isinstance(command, str))
    if result.timed_out:
        return VerifyOutcome(124, (
            "verification command timed out after %ds%s"
            % (timeout, _partial_on_timeout(result))))
    if result.not_found or result.error is not None:
        # The launch itself raised, so this is the ONE place that knows, first
        # hand, that nothing ran. Say so on the outcome; 125 remains only a
        # display code, never the evidence (see VerifyOutcome).
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
    """The failure message fed back to the model for another fix attempt."""
    tail = (output or "").strip()[-4000:] or "(no output)"
    return (
        f"The verification command `{command_text(until_cmd)}` failed with exit "
        f"code {code}. Output:\n{tail}\n\n"
        "Diagnose the real cause and fix it. Do not modify the check itself."
    )


def is_inconclusive(command: VerifyCommand, code: int,
                    did_not_start: bool = False) -> bool:
    """True when the check did not actually run anything."""
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
    """The project's obvious check command, or None when there isn't one."""
    from .project_config import ProjectConfigUnreadable, load_project_config
    try:
        configured = load_project_config(cwd).get("verify")
    except ProjectConfigUnreadable as e:
        # Say so rather than quietly auto-detecting: the project may configure a
        # DIFFERENT verify command, so falling through silently would run an
        # oracle the user did not choose and report its result as theirs. A note
        # is the right altitude here (unlike the CLI's refusal): this runs mid
        # session, where aborting would cost more than it protects, and a wrong
        # verify command is a quality gate rather than a safety one.
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
    """Positive evidence that *cwd* has a RUNNABLE test setup."""
    if (cwd / "Cargo.toml").is_file():
        # `cargo test` also compiles, so it is never vacuous.
        return _runner_available("cargo")
    if (cwd / "go.mod").is_file():
        return _runner_available("go")     # likewise `go test ./...`
    if (cwd / "package.json").is_file():
        # .exists(), not .is_file(), to match _detect_test_runner's own yarn.lock
        # test exactly: the runner checked here has to be the runner that will be
        # picked there, or the gate vouches for a command nobody runs.
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
    """True when pytest is importable HERE, in the running localm process."""
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
    """Record why detection declined, so an absent gate stays discoverable."""
    import logging
    logging.getLogger(__name__).debug(
        "no verification oracle: %s", why)


def _npm_has_test_script(cwd: Path) -> bool:
    """True when package.json actually defines a ``test`` script."""
    import json
    try:
        data = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
    except Exception:
        return False         # unreadable/malformed -> no evidence, not a check
    scripts = data.get("scripts")
    return isinstance(scripts, dict) and bool(scripts.get("test"))


def _has_python_tests(cwd: Path) -> bool:
    """True when the project looks like it has a pytest suite: a pytest config section, or test files where pytest would look for them."""
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
