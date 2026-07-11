# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared tool primitives: the ``ToolResult`` value type, the cwd-confinement
and output-truncation helpers every tool builds on, and the canonical
subprocess-execution primitive (``run_subprocess``, CODER-2)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

@dataclass
class ToolResult:
    ok:      bool
    output:  str
    summary: str = ""       # one-line display shown in the console
    truncated: bool = False

    @classmethod
    def success(cls, output: str, summary: str = "") -> "ToolResult":
        return cls(ok=True, output=output, summary=summary)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        # summary is a ONE-LINE console display (see field comment). Keep the
        # full message in output and only a capped first line here, so a long
        # diagnostic (a timeout's partial output, git stderr) cannot flood the
        # interactive console or the audit/event summary field.
        head = message.splitlines()[0] if message else ""
        if len(head) > 200:
            head = head[:200] + "..."
        return cls(ok=False, output=message, summary=f"ERROR: {head}")

    def to_xml(self, tool_name: str) -> str:
        status = "ok" if self.ok else "error"
        trunc  = ' truncated="true"' if self.truncated else ""
        return (
            f'<tool_result name="{tool_name}" status="{status}"{trunc}>\n'
            f"{self.output}\n"
            f"</tool_result>"
        )


_MAX_OUTPUT = 8_000   # chars - truncate large outputs to spare context


def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return (
        text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:],
        True,
    )


def _partial_on_timeout(exc) -> str:
    """Format any output a timed-out subprocess produced before it was killed.

    ``subprocess.TimeoutExpired`` carries the stdout/stderr captured up to the
    kill; dropping it hides exactly the diagnostics the model needs (the test
    progress or last log line before the hang). On POSIX the attributes can be
    bytes even in text mode (a CPython quirk), so decode defensively. Returns
    '' when nothing was captured.
    """
    parts = []
    for label, data in (("", getattr(exc, "stdout", None)),
                        ("STDERR:\n", getattr(exc, "stderr", None))):
        if not data:
            continue
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", errors="replace")
        data = data.strip()
        if data:
            parts.append(label + data)
    if not parts:
        return ""
    text, _ = _truncate("\n".join(parts))
    return "\n[partial output captured before the timeout]\n" + text


def _confine(cwd: Path, path: str) -> Path:
    """
    Resolve *path* against *cwd* and verify it stays inside *cwd*.

    Raises ``PermissionError`` with a clear message if the resolved path
    escapes the working directory (path traversal attempt or accidental
    absolute path outside the project root).
    """
    resolved = (cwd / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    cwd_resolved = cwd.resolve()
    if not resolved.is_relative_to(cwd_resolved):
        raise PermissionError(
            f"'{path}' resolves outside the working directory '{cwd_resolved}'. "
            "All file operations must stay within the project root."
        )
    return resolved


@dataclass
class SubprocessResult:
    """Outcome of one :func:`run_subprocess` call.

    A completed process sets *returncode*/*stdout*/*stderr* (``ok`` is
    ``returncode == 0``). A timeout sets *timed_out* and carries whatever the
    process produced before the kill in *stdout*/*stderr* - possibly bytes even
    in text mode (a CPython ``TimeoutExpired`` quirk) - pass the result straight
    to :func:`_partial_on_timeout` to format it. A launch failure sets
    *not_found* (missing executable) or *error* (any other exception); *ok* is
    always False for either.
    """
    ok: bool
    returncode: Optional[int] = None
    stdout: object = ""
    stderr: object = ""
    timed_out: bool = False
    not_found: bool = False
    error: Optional[str] = None


def run_subprocess(
    argv_or_cmd: Union[list, str],
    cwd: Path,
    *,
    timeout: float,
    shell_wrap: bool = False,
    env: Optional[dict] = None,
) -> SubprocessResult:
    """
    Run a subprocess, capturing stdout+stderr as text with a timeout.

    *argv_or_cmd* is an argument list, run directly, unless *shell_wrap* is
    true - then it must be a command STRING, routed through the platform
    shell (``cmd /C`` on Windows, ``/bin/sh -c`` elsewhere) so shell operators
    (pipes, redirects, ``&&``) work.

    On a timeout, the process's captured stdout/stderr up to the kill is
    preserved on the result, not dropped - format it for display with
    :func:`_partial_on_timeout`. This is the canonical subprocess-execution
    primitive for the coder's tools/shell.py, tools/git.py, and cli/goal.py
    (CODER-2) - previously four independent copies of this run+capture+timeout
    sequence, with only shell.py's own callers getting partial-output-on-timeout
    and git.py/goal.py silently dropping it.
    """
    if shell_wrap:
        if sys.platform == "win32":
            argv = ["cmd", "/C", argv_or_cmd]
        else:
            argv = ["/bin/sh", "-c", argv_or_cmd]
    else:
        argv = argv_or_cmd

    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace", env=env,
        )
    except subprocess.TimeoutExpired as e:
        return SubprocessResult(
            ok=False, timed_out=True, stdout=e.stdout, stderr=e.stderr)
    except FileNotFoundError as e:
        return SubprocessResult(ok=False, not_found=True, error=str(e))
    except Exception as e:
        return SubprocessResult(ok=False, error=str(e))

    return SubprocessResult(
        ok=(proc.returncode == 0), returncode=proc.returncode,
        stdout=proc.stdout or "", stderr=proc.stderr or "",
    )
