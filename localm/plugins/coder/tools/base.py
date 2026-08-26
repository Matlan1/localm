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

from localm import pathsafe

@dataclass
class ToolResult:
    ok:      bool
    output:  str
    summary: str = ""       # one-line display shown in the console
    truncated: bool = False
    # For a tool whose target files are discovered at RUNTIME rather than
    # named in its call args (search_replace's glob+regex sweep has no
    # `path` arg to read ahead of time), the (relative_path, old_bytes,
    # new_text) of every file it touched - or would touch, when called with
    # its own dry_run=True. None for every other tool, and None here too
    # when nothing matched. Both a caller that must NOT touch disk (patch
    # mode, previewing via dry_run) and a caller recording what a REAL write
    # just changed (the changed-files/undo tracker) read this one field, fed
    # by the same matching pass.
    changes: "list[tuple[str, bytes, str]] | None" = None

    @classmethod
    def success(cls, output: str, summary: str = "",
                changes: "list[tuple[str, bytes, str]] | None" = None) -> "ToolResult":
        return cls(ok=True, output=output, summary=summary, changes=changes)

    @classmethod
    def error(cls, message: str,
              changes: "list[tuple[str, bytes, str]] | None" = None) -> "ToolResult":
        # summary is a ONE-LINE console display (see field comment). Keep the
        # full message in output and only a capped first line here, so a long
        # diagnostic (a timeout's partial output, git stderr) cannot flood the
        # interactive console or the audit/event summary field.
        head = message.splitlines()[0] if message else ""
        if len(head) > 200:
            head = head[:200] + "..."
        # changes: a PARTIAL apply (some files written before a mid-batch
        # failure) still tracks those files as changed/undoable - they really
        # did change on disk, and reporting the call as an error must not make
        # that mutation invisible.
        return cls(ok=False, output=message, summary=f"ERROR: {head}", changes=changes)

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
    kill - the test progress or last log line before the hang. On POSIX the
    attributes can be bytes even in text mode (a CPython quirk), so they are
    decoded defensively. Returns '' when nothing was captured.
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

    Delegates to ``pathsafe.confined_absolute_or_under``, the shared
    absolute-or-relative confinement primitive. That refuses a UNC/device
    *path* before it can reach ``Path(path).resolve()`` (the syscall dials SMB
    and can hang for minutes), and rejects an NTFS Alternate Data Stream or
    short-name alias in *path* on a character and identity check.

    ``"."``/``""`` naming *cwd* itself is a supported contract here (a tool
    listing "the project root" is a legitimate request). The shared primitive
    treats a self-referential result as invalid, matching its delete-oriented
    callers, so that one case is handled in this function rather than by
    relaxing the primitive for every caller.
    """
    if path in (".", ""):
        return cwd.resolve()
    try:
        return pathsafe.confined_absolute_or_under(cwd, path)
    except ValueError as e:
        raise PermissionError(
            f"'{path}' resolves outside the working directory '{cwd.resolve()}'. "
            f"All file operations must stay within the project root. ({e})"
        )


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


def platform_shell(command: str) -> Union[list, str]:
    """The launchable form of *command* run through the platform shell.

    Returns a STRING on Windows and an argument LIST on POSIX, because that is
    what each platform needs to receive the command line UNCHANGED - and this is
    the one definition of that, so no caller has to rediscover it.

    On POSIX an argv list is handed to ``execv`` verbatim, so ``/bin/sh -c`` plus
    the command as one element already arrives intact.

    On Windows it must NOT be ``["cmd", "/C", command]``. ``subprocess`` renders
    an argv list with :func:`subprocess.list2cmdline`, which escapes every
    embedded quote MSVCRT-style as ``\\"`` - syntax ``cmd.exe`` does not speak -
    so a quoted path, the normal way to pass a path containing spaces, reaches
    cmd mangled and cannot be opened. Pre-compensating is not possible:
    list2cmdline turns EVERY quote into ``\\"``, so no list element can put a
    bare quote on the command line. A command STRING is passed to
    ``CreateProcess`` verbatim instead, so cmd applies its own quoting rules to
    exactly what was written.
    """
    if sys.platform == "win32":
        return "cmd /C " + command
    return ["/bin/sh", "-c", command]


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
    shell by :func:`platform_shell` so shell operators (pipes, redirects,
    ``&&``) work. An already-routed caller (``tools/shell.py:_shell_argv``) may
    also pass the string that function returns for the Windows shell route; a
    bare string is a raw Windows command line, never a POSIX one.

    On a timeout, the process's captured stdout/stderr up to the kill is
    preserved on the result, not dropped - format it for display with
    :func:`_partial_on_timeout`. This is the canonical subprocess-execution
    primitive for the coder's tools/shell.py, tools/git.py, and cli/goal.py.
    """
    argv = platform_shell(argv_or_cmd) if shell_wrap else argv_or_cmd

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
