# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-execution tools: ``run_shell`` (with the shell-vs-arglist routing and
privacy-env hook) and ``run_tests`` (test-runner autodetection)."""

from __future__ import annotations

import sys
from pathlib import Path

from .base import (
    ToolResult,
    _partial_on_timeout,
    _truncate,
    platform_shell,
    run_subprocess,
)

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


def _split_command(command: str) -> list[str]:
    """Split *command* into an argument list, with the quote characters REMOVED.

    Removing them is the point. A quoted path is the normal way to pass a path
    containing spaces, and what must reach the process is the path, not the
    quotes around it. This used to split with ``shlex.split(posix=False)`` on
    Windows, which does not do quote removal BY DESIGN, so the token stayed
    ``"a dir with spaces\\f.txt"``, quotes and all, and the process could not
    open it.

    So: posix mode, which does the removal - plus one Windows adjustment.

    Post-stripping the quotes off posix=False's tokens is not the same fix and
    does not work, because posix=False also gets the token BOUNDARIES wrong: it
    honours a quote only where one OPENS a token, so ``--message="a b"`` splits
    into ``['--message="a', 'b"']``. A wrong boundary cannot be repaired after
    the fact. Posix mode groups a mid-token quote the way Windows does.

    ``lex.escape = ""`` is the load-bearing Windows line, and the reason plain
    ``posix=True`` is not the fix either: posix rules read a backslash as an
    escape, which turns ``dir sub\\dir\\f.txt`` into ``dir subdirf.txt`` and
    drops a separator from a UNC ``"\\\\host\\share"``. On Windows a backslash is
    a path separator and nothing else, so the escape character is cleared.

    ``lex.commenters`` is cleared for the same reason :func:`shlex.split` clears
    it: otherwise ``#`` opens a comment and truncates the rest, silently losing
    the message in ``git commit -m "fix #42"``.

    Malformed quoting still raises ``ValueError``, which the caller turns into
    the shell fallback.
    """
    import shlex

    lex = shlex.shlex(command, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    if sys.platform == "win32":
        lex.escape = ""
    return list(lex)


def _shell_argv(command: str) -> "list[str] | str":
    """Route *command* to an argument list, or to the platform shell.

    When the command contains no shell operators (pipes, redirects, globs,
    variable expansion, etc.) it is parsed with :func:`_split_command` and
    returned as a plain argument list - no shell injection possible. Otherwise
    it falls back to the system shell (cmd /C on Windows, /bin/sh -c elsewhere),
    whose launch form is a raw command-line STRING on Windows and a list on
    POSIX (see :func:`base.platform_shell`).

    This is the ONE place that decision is made, so the blocking ``run_shell``
    and the background ``run_shell_background`` cannot drift into different
    security postures.
    """
    if _needs_shell(command):
        # Complex command - must go through a shell
        return platform_shell(command)
    try:
        argv = _split_command(command)
    except ValueError:
        # Malformed quoting - fall back to shell
        return platform_shell(command)
    # Shell builtins (echo, dir, type, …) have no executable on disk -
    # argument-list mode would fail with "file not found". Detect via
    # PATH lookup and route those through the shell instead. This lookup is why
    # _split_command must remove quotes: which() of a still-quoted token is
    # None, so a quoted absolute executable used to fall through to the shell
    # route and be mangled there rather than run directly.
    import shutil as _shutil
    if not argv or _shutil.which(argv[0]) is None:
        return platform_shell(command)
    return argv


def _privacy_env(_privacy: bool) -> dict | None:
    """Subprocess environment with shell-history variables zeroed, or None."""
    if not _privacy:
        return None
    from ..privacy import subprocess_privacy_env
    return subprocess_privacy_env()


def tool_run_shell(
    cwd: Path,
    command: str,
    timeout: int = 30,
    _privacy: bool = False,
) -> ToolResult:
    """
    Execute a shell command and wait for it to finish.

    Argument-list vs shell routing is decided by :func:`_shell_argv`.

    In privacy mode (``_privacy=True``) the subprocess environment has
    shell-history variables zeroed.
    """
    shell_cmd = _shell_argv(command)
    env = _privacy_env(_privacy)

    result = run_subprocess(shell_cmd, cwd, timeout=timeout, env=env)

    if result.timed_out:
        return ToolResult.error(
            f"Command timed out after {timeout}s{_partial_on_timeout(result)}")
    if result.error is not None:
        return ToolResult.error(result.error)

    combined = ""
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + "STDERR:\n" + result.stderr

    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)
    rc = result.returncode
    status = "ok" if rc == 0 else f"exit {rc}"

    return ToolResult(
        ok=(rc == 0),
        output=f"<exit_code>{rc}</exit_code>\n<output>\n{output}\n</output>",
        summary=f"$ {command[:60]}  [{status}]",
        truncated=trunc,
    )


# --------------------------------------------------------------------------- #
#  Background execution                                                        #
# --------------------------------------------------------------------------- #
#  run_shell blocks, so the coder could not start a dev server and then talk to
#  it, or run a long build while doing anything else. These three tools are the
#  async half. They deliberately reuse _shell_argv above, so the background path
#  makes the SAME shell-vs-argv security decision as the blocking one; the job
#  bookkeeping lives in the kind-agnostic registry in ../background.py.

def _job_not_found(registry, job_id: str) -> ToolResult:
    """Error naming the ids that DO exist, so a wrong id is self-correcting."""
    known = registry.ids()
    listing = ", ".join(known) if known else "none"
    return ToolResult.error(
        f"No background job with id '{job_id}'. Known job ids: {listing}.")


def _render_job(job) -> tuple[str, str, bool, dict]:
    """Render a job as ``(output, summary, truncated, status)`` for a ToolResult.

    The status snapshot is RETURNED rather than left for the caller to re-read.
    A second ``job.status()`` would be an independent read of state the watcher
    thread mutates, so a job finishing between the two calls would render a
    "running" body while the caller's ok/exit-code logic saw the finished one.
    One snapshot, one story.
    """
    st = job.status()
    out, err, dropped = job.output()

    body = out.strip()
    if err.strip():
        body += ("\n" if body else "") + "STDERR:\n" + err.strip()
    if not body:
        body = "(no output yet)" if st["state"] == "running" else "(no output)"
    body, trunc = _truncate(body)

    exit_code = (st["result"] or {}).get("exit_code")
    lines = [
        f"<job>{st['id']}</job>",
        f"<state>{st['state']}</state>",
        f"<pid>{st.get('pid')}</pid>",
        f"<elapsed>{st['elapsed']:.1f}s</elapsed>",
    ]
    if st["state"] != "running":
        lines.append(f"<exit_code>{exit_code}</exit_code>")
    if st["error"]:
        lines.append(f"<error>{st['error']}</error>")
    if dropped:
        # Never present a trimmed tail as if it were the whole output.
        lines.append(
            f"<dropped_chars>{dropped}</dropped_chars>  "
            "(oldest output was discarded to bound memory)")
    for warning in st["warnings"]:
        lines.append(f"<warning>{warning}</warning>")
    lines.append(f"<output>\n{body}\n</output>")

    if st["state"] == "running":
        status = f"running {st['elapsed']:.0f}s"
    elif st["state"] == "done":
        status = "ok" if exit_code == 0 else f"exit {exit_code}"
    else:
        status = st["state"]
    summary = f"{st['id']} $ {st['label'][:40]}  [{status}]"
    return "\n".join(lines), summary, trunc, st


def tool_run_shell_background(
    cwd: Path,
    command: str,
    _privacy: bool = False,
) -> ToolResult:
    """
    Start a shell command in the background and return a job id immediately.

    Use for anything long-running you need to keep talking to or working
    alongside: a dev server you then curl, a long build, a watcher. Poll it with
    ``check_shell_job`` and stop it with ``kill_shell_job``. For a command you
    just need the result of, use ``run_shell`` instead.

    Argument-list vs shell routing and privacy-mode env handling are identical
    to :func:`tool_run_shell`; only the waiting differs.
    """
    from ..background import JobCapacityError, ShellJob, get_registry

    argv = _shell_argv(command)
    env = _privacy_env(_privacy)
    registry = get_registry()

    try:
        job = registry.submit(
            lambda: ShellJob(argv, cwd, label=command, env=env), kind="shell")
    except JobCapacityError as e:
        return ToolResult.error(str(e))
    except FileNotFoundError as e:
        return ToolResult.error(
            f"Could not start '{command}': {e}. Check the executable is on PATH.")
    except OSError as e:
        return ToolResult.error(f"Could not start '{command}': {e}")

    return ToolResult(
        ok=True,
        output=(
            f"<job>{job.id}</job>\n"
            f"<state>running</state>\n"
            f"<pid>{job.pid}</pid>\n"
            f"<note>Started in the background. Poll it with "
            f"check_shell_job(job_id=\"{job.id}\") and stop it with "
            f"kill_shell_job(job_id=\"{job.id}\").</note>"
        ),
        summary=f"{job.id} started $ {command[:50]}",
    )


def tool_check_shell_job(cwd: Path, job_id: str) -> ToolResult:
    """
    Check a background job: its state, exit code once finished, and the output
    buffered so far. Safe to call repeatedly; output accumulates until the job
    is pruned.
    """
    from ..background import get_registry

    registry = get_registry()
    job = registry.get(job_id)
    if job is None:
        return _job_not_found(registry, job_id)

    # The SAME snapshot the body was rendered from: re-reading here would let a
    # job that finished in between be described as running while ok said it
    # failed, and that ok=False then feeds the consecutive-failure breaker.
    output, summary, trunc, st = _render_job(job)
    # Mirror run_shell: a finished job that FAILED reports ok=False, so the model
    # sees the failure rather than a green "check succeeded". A job the model
    # killed on purpose is not a failure though - reporting one would feed the
    # consecutive-failure circuit breaker for doing exactly the right thing.
    ok = (st["state"] in ("running", "killed")
          or (st["result"] or {}).get("exit_code") == 0)
    return ToolResult(ok=ok, output=output, summary=summary, truncated=trunc)


def tool_kill_shell_job(cwd: Path, job_id: str) -> ToolResult:
    """
    Stop a background job and its whole process tree, then report the final
    state and buffered output.
    """
    from ..background import get_registry

    registry = get_registry()
    job = registry.get(job_id)
    if job is None:
        return _job_not_found(registry, job_id)

    outcome = job.kill()
    output, summary, trunc, _st = _render_job(job)
    output = f"<kill_result>{outcome}</kill_result>\n" + output
    # A kill that could not stop the process must never report success.
    ok = not outcome.startswith("kill FAILED")
    return ToolResult(ok=ok, output=output,
                      summary=f"{job_id} {outcome}", truncated=trunc)


def resolve_runner(name: str) -> "str | None":
    """The launchable path to *name*, or None when it is not installed.

    Returns the RESOLVED path rather than the bare name because these commands
    are run as an argv list, and argv-list execution on Windows goes through
    CreateProcess, which only launches real executables. npm, yarn and npx ship
    as `.CMD` shims, so `['npm', ...]` raises WinError 2 even with npm installed
    and on PATH, while the full `...\\npm.CMD` path runs fine (both measured).
    That is why a bare `shutil.which` truthiness check is not enough here: which
    finds npm.CMD, and the argv that names it `npm` still cannot start.

    Absolutised, because which() on Windows searches the CURRENT directory first
    and returns what it joined, so the answer can be a relative `.\\npm.CMD`.
    The result is stored on the session and run later with cwd set to the
    PROJECT directory, which is not always the directory which() searched - a
    relative answer would then resolve somewhere else, or against a same-named
    file the project happens to contain."""
    import os
    import shutil as _shutil
    found = _shutil.which(name)
    return os.path.abspath(found) if found else None


def _detect_test_runner(cwd: Path) -> list[str]:
    """Return the command list for the most appropriate test runner in *cwd*.

    Always returns a command (pytest is the fallback), so callers that want
    positive evidence of a runnable check must gate separately - the verify
    oracle does exactly that in ``verify._has_project_check``. When a runner is
    not on PATH the bare name is kept, so the caller's "runner not found"
    message names something the user recognises."""
    if (cwd / "Cargo.toml").exists():
        return [resolve_runner("cargo") or "cargo", "test", "--color=never"]
    if (cwd / "go.mod").exists():
        return [resolve_runner("go") or "go", "test", "./..."]
    if (cwd / "package.json").exists():
        lock = "yarn" if (cwd / "yarn.lock").exists() else "npm"
        return [resolve_runner(lock) or lock, "test", "--passWithNoTests"]
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
        cmd = [resolve_runner("cargo") or "cargo", "test", "--color=never"]
    elif runner == "go":
        cmd = [resolve_runner("go") or "go", "test", "./..."]
    elif runner in ("npm", "yarn"):
        # Resolved for the same reason as the auto branch: an explicitly asked
        # for `npm` is no more launchable as a bare argv[0] on Windows.
        cmd = [resolve_runner(runner) or runner, "test", "--passWithNoTests"]
    else:
        return ToolResult.error(
            f"Unknown runner '{runner}'. Use: auto, pytest, cargo, go, npm, yarn"
        )

    if path and path != ".":
        cmd.append(path)
    if extra_args:
        cmd.extend(extra_args.split())

    result = run_subprocess(cmd, cwd, timeout=120)

    if result.not_found:
        return ToolResult.error(
            f"Test runner not found: {cmd[0]}. "
            "Make sure it is installed and on PATH."
        )
    if result.timed_out:
        return ToolResult.error(
            f"Test run timed out after 120s{_partial_on_timeout(result)}")
    if result.error is not None:
        return ToolResult.error(result.error)

    combined = ""
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)

    ok = result.returncode == 0
    if ok:
        status = "passed"
    elif result.returncode == 5 and "pytest" in " ".join(cmd):
        # pytest exit 5 = no tests collected - not a failure, but worth
        # distinguishing so the agent doesn't "fix" passing code
        ok = True
        status = "no tests found"
    else:
        status = f"failed (exit {result.returncode})"
    return ToolResult(
        ok=ok,
        output=f"<runner>{' '.join(cmd[:2])}</runner>\n"
               f"<status>{status}</status>\n"
               f"<output>\n{output}\n</output>",
        summary=f"tests {status}",
        truncated=trunc,
    )
