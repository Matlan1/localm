# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-execution tools: ``run_shell`` (with the shell-vs-arglist routing and
privacy-env hook) and ``run_tests`` (test-runner autodetection)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import ToolResult, _truncate

def _needs_shell(command: str) -> bool:
    """Return True when the command uses shell operators that require a real shell."""
    # Characters that only mean something inside a shell
    shell_chars = {"&", "|", ";", "<", ">", "$", "`", "~", "(", ")", "{", "}", "!", "\\", "*", "?"}
    in_single = False
    in_double = False
    for ch in command:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double and ch in shell_chars:
            return True
    return False


def tool_run_shell(
    cwd: Path,
    command: str,
    timeout: int = 30,
    _privacy: bool = False,
) -> ToolResult:
    """
    Execute a shell command.

    When the command contains no shell operators (pipes, redirects, globs,
    variable expansion, etc.) it is parsed with ``shlex.split`` and run as
    a plain argument list - no shell injection possible.  Otherwise it falls
    back to the system shell (cmd /C on Windows, /bin/sh -c elsewhere).

    In privacy mode (``_privacy=True``) the subprocess environment has
    shell-history variables zeroed.
    """
    import shlex

    shell_cmd: list[str]
    if _needs_shell(command):
        # Complex command - must go through a shell
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/C", command]
        else:
            shell_cmd = ["/bin/sh", "-c", command]
    else:
        try:
            shell_cmd = shlex.split(command, posix=(sys.platform != "win32"))
        except ValueError:
            # Malformed quoting - fall back to shell
            if sys.platform == "win32":
                shell_cmd = ["cmd", "/C", command]
            else:
                shell_cmd = ["/bin/sh", "-c", command]
        else:
            # Shell builtins (echo, dir, type, …) have no executable on disk -
            # argument-list mode would fail with "file not found". Detect via
            # PATH lookup and route those through the shell instead.
            import shutil as _shutil
            if not shell_cmd or _shutil.which(shell_cmd[0]) is None:
                if sys.platform == "win32":
                    shell_cmd = ["cmd", "/C", command]
                else:
                    shell_cmd = ["/bin/sh", "-c", command]

    env: dict | None = None
    if _privacy:
        from ..privacy import subprocess_privacy_env
        env = subprocess_privacy_env()

    try:
        proc = subprocess.run(
            shell_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult.error(str(e))

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += ("\n" if combined else "") + "STDERR:\n" + proc.stderr

    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)
    rc = proc.returncode
    status = "ok" if rc == 0 else f"exit {rc}"

    return ToolResult(
        ok=(rc == 0),
        output=f"<exit_code>{rc}</exit_code>\n<output>\n{output}\n</output>",
        summary=f"$ {command[:60]}  [{status}]",
        truncated=trunc,
    )


def _detect_test_runner(cwd: Path) -> list[str]:
    """Return the command list for the most appropriate test runner in *cwd*."""
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "test", "--color=never"]
    if (cwd / "go.mod").exists():
        return ["go", "test", "./..."]
    if (cwd / "package.json").exists():
        lock = "yarn" if (cwd / "yarn.lock").exists() else "npm"
        return [lock, "test", "--passWithNoTests"]
    # Python - prefer pytest; fall back to unittest. Use the SAME interpreter that
    # runs localm (sys.executable), not a bare "python" off PATH - on many machines
    # PATH `python` is a different env (uv/conda/system) without pytest or the
    # project deps, which made run_tests report "No module named pytest" on a suite
    # that passes under the project venv.
    return [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"]


def tool_run_tests(
    cwd: Path,
    runner: str = "auto",
    path: str = ".",
    extra_args: str = "",
) -> ToolResult:
    """
    Run the project's test suite and return the result.

    Parameters
    ----------
    runner:
        ``auto`` (default) detects from project files; or specify
        ``pytest``, ``cargo``, ``go``, ``npm``, ``yarn`` explicitly.
    path:
        Subdirectory or file to limit the test run (default: whole project).
    extra_args:
        Additional arguments appended verbatim to the test command.
    """
    if runner == "auto":
        cmd = _detect_test_runner(cwd)
    elif runner == "pytest":
        cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"]
    elif runner == "cargo":
        cmd = ["cargo", "test", "--color=never"]
    elif runner == "go":
        cmd = ["go", "test", "./..."]
    elif runner in ("npm", "yarn"):
        cmd = [runner, "test", "--passWithNoTests"]
    else:
        return ToolResult.error(
            f"Unknown runner '{runner}'. Use: auto, pytest, cargo, go, npm, yarn"
        )

    if path and path != ".":
        cmd.append(path)
    if extra_args:
        cmd.extend(extra_args.split())

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return ToolResult.error(
            f"Test runner not found: {cmd[0]}. "
            "Make sure it is installed and on PATH."
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error("Test run timed out after 120s")
    except Exception as e:
        return ToolResult.error(str(e))

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += ("\n" if combined else "") + proc.stderr
    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)

    ok = proc.returncode == 0
    if ok:
        status = "passed"
    elif proc.returncode == 5 and "pytest" in " ".join(cmd):
        # pytest exit 5 = no tests collected - not a failure, but worth
        # distinguishing so the agent doesn't "fix" passing code
        ok = True
        status = "no tests found"
    else:
        status = f"failed (exit {proc.returncode})"
    return ToolResult(
        ok=ok,
        output=f"<runner>{' '.join(cmd[:2])}</runner>\n"
               f"<status>{status}</status>\n"
               f"<output>\n{output}\n</output>",
        summary=f"tests {status}",
        truncated=trunc,
    )
