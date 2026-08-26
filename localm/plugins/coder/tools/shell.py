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

    A quoted path is the normal way to pass a path containing spaces, and what
    must reach the process is the path, not the quotes around it. Posix mode is
    what does that removal; ``shlex.split(posix=False)`` leaves the quotes on
    the token, and it also gets the token BOUNDARIES wrong, honouring a quote
    only where one OPENS a token, so ``--message="a b"`` splits into
    ``['--message="a', 'b"']``.

    ``lex.escape = ""`` is the load-bearing Windows line: posix rules read a
    backslash as an escape, which turns ``dir sub\\dir\\f.txt`` into
    ``dir subdirf.txt`` and drops a separator from a UNC ``"\\\\host\\share"``.
    On Windows a backslash is a path separator and nothing else, so the escape
    character is cleared.

    ``lex.commenters`` is cleared for the same reason :func:`shlex.split` clears
    it: otherwise ``#`` opens a comment and truncates the rest, losing the
    message in ``git commit -m "fix #42"``.

    Malformed quoting raises ``ValueError``, which the caller turns into the
    shell fallback.
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
    # Shell builtins (echo, dir, type, …) have no executable on disk, so
    # argument-list mode would fail with "file not found". They are resolved via
    # resolve_runner() and unresolvable names are routed through the shell.
    # This lookup is why _split_command must remove quotes: a still-quoted token
    # resolves to None and falls through to the shell route.
    #
    # resolve_runner(), not a bare shutil.which() truthiness check: npm, yarn
    # and npx ship as .CMD shims on Windows, and this argv list is launched via
    # CreateProcess, which cannot start a bare "npm" even though which() finds
    # npm.CMD on PATH. Only the resolved path, with its extension, launches.
    if not argv:
        return platform_shell(command)
    resolved = resolve_runner(argv[0])
    if resolved is None:
        return platform_shell(command)
    return [resolved, *argv[1:]]


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
#  run_shell blocks; these three tools are the async half, for a dev server the
#  coder then talks to or a long build it works alongside. They reuse
#  _shell_argv above, so the background path makes the SAME shell-vs-argv
#  security decision as the blocking one; the job bookkeeping lives in the
#  kind-agnostic registry in ../background.py.

def _job_not_found(registry, job_id: str) -> ToolResult:
    """Error naming the ids that DO exist, so a wrong id is self-correcting."""
    known = registry.ids()
    listing = ", ".join(known) if known else "none"
    return ToolResult.error(
        f"No background job with id '{job_id}'. Known job ids: {listing}.")


def _render_job(job) -> tuple[str, str, bool, dict]:
    """Render a job as ``(output, summary, truncated, status)`` for a ToolResult.

    The status snapshot is RETURNED rather than left for the caller to re-read:
    a second ``job.status()`` is an independent read of state the watcher thread
    mutates, so a job finishing between the two calls would render a "running"
    body while the caller's ok/exit-code logic saw the finished one.
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
    _owner: str | None = None,
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
            lambda: ShellJob(argv, cwd, label=command, env=env, owner=_owner),
            kind="shell")
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
    # failed, and that ok=False feeds the consecutive-failure breaker.
    output, summary, trunc, st = _render_job(job)
    # Mirror run_shell: a finished job that FAILED reports ok=False. A job the
    # model killed itself is not a failure and does not feed the
    # consecutive-failure circuit breaker.
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

    Returns the RESOLVED path rather than the bare name: these commands run as
    an argv list, and argv-list execution on Windows goes through CreateProcess,
    which only launches real executables. npm, yarn and npx ship as `.CMD`
    shims, so `['npm', ...]` raises WinError 2 even with npm installed and on
    PATH, while the full `...\\npm.CMD` path runs. A bare `shutil.which`
    truthiness check is therefore not enough.

    Absolutised, because which() on Windows searches the CURRENT directory first
    and returns what it joined, so the answer can be a relative `.\\npm.CMD`.
    The result is stored on the session and run later with cwd set to the
    PROJECT directory, where a relative answer would resolve somewhere else."""
    import os
    import shutil as _shutil
    found = _shutil.which(name)
    return os.path.abspath(found) if found else None


# Test runners that actually HAVE a --passWithNoTests flag. An allowlist:
# sending the flag to a runner that does not know it is at best inert and at
# worst fatal (`node --test --passWithNoTests` exits 9 with "bad option"), so an
# unlisted runner gets no flag.
_NO_TESTS_OK_RUNNERS = frozenset({"jest", "vitest"})

# How a package script may spell a runner's binary; stripped before matching, so
# `node_modules/.bin/jest.cmd` reads as jest while a script that merely mentions
# one (`node tools/jest-codemod.js`) does not.
_SCRIPT_BIN_SUFFIXES = (".js", ".cjs", ".mjs", ".cmd", ".bat", ".exe")


def _last_command_of(script: str) -> str:
    """The FINAL command in a package script line.

    A package manager appends forwarded arguments to the END of the script, so
    the last command is the only one that can receive them: in the script
    ``node a.js && node b.js`` a forwarded argument reaches ``b.js`` only, and
    in ``jest && eslint .`` an appended ``--passWithNoTests`` would land on
    eslint rather than jest.
    """
    import re
    return re.split(r"&&|\|\||[;|&]", script)[-1]


def _runner_basename(token: str) -> str:
    """*token*'s program name: directory dropped, launcher suffix stripped,
    lowercased.

    So a package script's ``node_modules/.bin/jest.cmd`` reads as ``jest``, and
    the absolute ``...\\nodejs\\npm.CMD`` that :func:`resolve_runner` hands back
    reads as ``npm``.
    """
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in _SCRIPT_BIN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _script_runner_takes_no_tests_flag(cwd: Path) -> bool:
    """True when this package's ``test`` script ends in a runner that has a
    ``--passWithNoTests`` flag (see :data:`_NO_TESTS_OK_RUNNERS`)."""
    import json
    try:
        data = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return False        # unreadable or malformed: no evidence, so no flag
    scripts = data.get("scripts")
    script = scripts.get("test") if isinstance(scripts, dict) else None
    if not isinstance(script, str):
        return False
    return any(_runner_basename(token) in _NO_TESTS_OK_RUNNERS
               for token in _last_command_of(script).split())


def _js_test_command(cwd: Path, package_manager: str) -> list[str]:
    """The ``npm``/``yarn`` test command for *cwd*.

    ONE definition, used by both :func:`_detect_test_runner` and ``run_tests``'s
    explicit npm/yarn branch, so the command the verify oracle runs and the
    command the model's own tool runs cannot drift apart.

    ``--passWithNoTests`` keeps a project whose suite is still empty from being
    billed a failed verification. HOW it has to be passed differs per package
    manager:

    * npm SWALLOWS the bare flag: it reads an unrecognised option as one of its
      own configs, so the script sees only ``npm_config_passwithnotests`` in the
      environment, and npm forwards just nopt's POSITIONAL remainder to the
      script (``npm/lib/npm.js``: ``this.argv = [...parsedArgv.remain]``). It
      must therefore go through npm's ``--`` separator.
    * yarn is the OPPOSITE and must NOT be given the separator. yarn classic
      forwards the bare flag correctly, and warns that a future yarn "will
      forward any explicit -- as-is to the scripts", which would hand the runner
      a literal ``--`` and demote the flag to a positional argument.

    Positional arguments need no separator on either (``npm test somepath``
    reaches the script), so ``run_tests`` can append its ``path`` after this
    command.

    The flag goes only to a runner that HAS it (see
    :data:`_NO_TESTS_OK_RUNNERS`); anything else gets a plain ``<pm> test``.
    """
    cmd = [resolve_runner(package_manager) or package_manager, "test"]
    if _script_runner_takes_no_tests_flag(cwd):
        # npm forwards nothing flag-shaped without the separator; yarn forwards
        # it fine and warns that a future version will pass an explicit `--`
        # straight through to the runner.
        separator = ["--"] if package_manager == "npm" else []
        cmd += separator + ["--passWithNoTests"]
    return cmd


def _append_caller_args(cmd: list[str], args: list[str]) -> list[str]:
    """*cmd* with the caller's *args* appended so the RUNNER receives them.

    npm needs its ``--`` separator or anything flag-shaped never leaves npm: it
    reads an unrecognised option as one of its own configs and forwards only
    nopt's POSITIONAL remainder to the package script (``npm/lib/npm.js``:
    ``this.argv = [...parsedArgv.remain]``). ``npm test --watch`` hands the
    script ``ARGV=[]``, while ``npm test -- --watch`` hands it ``["--watch"]``.

    The separator also fronts the POSITIONAL ``path``, which needs none of its
    own: ``npm test -- somepath`` and ``npm test somepath`` both deliver
    ``["somepath"]``.

    npm only:

    * yarn is the opposite. yarn classic forwards a bare flag correctly and
      warns that a future yarn "will forward any explicit -- as-is to the
      scripts", which would hand the runner a literal ``--`` and demote the flag
      to a positional argument.
    * pytest, cargo and go parse their own argv, so their flags already arrive.

    One separator, never two: ``npm test -- -- --watch`` delivers a literal
    ``["--", "--watch"]`` to the script. A command that already carries one
    keeps it, and a caller who wrote their own leading ``--`` gets theirs
    honoured rather than doubled.
    """
    if not args:
        return cmd
    if _runner_basename(cmd[0]) == "npm" and "--" not in cmd and args[0] != "--":
        return cmd + ["--"] + args
    return cmd + args


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
        return _js_test_command(cwd, lock)
    # Python - prefer pytest; fall back to unittest. Uses the SAME interpreter
    # that runs localm (sys.executable), not a bare "python" off PATH: on many
    # machines PATH `python` is a different env (uv/conda/system) without pytest
    # or the project deps.
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
        Additional arguments appended verbatim to the test command (for npm,
        past its ``--`` separator, or npm would swallow anything flag-shaped).
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
        # Same builder as the auto branch, so an explicitly requested `npm`
        # gets the same resolved path (a bare name is not launchable as argv[0]
        # on Windows) and the same --passWithNoTests handling.
        cmd = _js_test_command(cwd, runner)
    else:
        return ToolResult.error(
            f"Unknown runner '{runner}'. Use: auto, pytest, cargo, go, npm, yarn"
        )

    caller_args: list[str] = []
    if path and path != ".":
        caller_args.append(path)
    if extra_args:
        caller_args += extra_args.split()
    # Appended through one place, because npm drops flag-shaped arguments that
    # do not follow its `--` separator (see _append_caller_args).
    cmd = _append_caller_args(cmd, caller_args)

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
        # pytest exit 5 = no tests collected: reported distinctly, not as a
        # failure
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
