# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared tool primitives: the ``ToolResult`` value type and the cwd-confinement
and output-truncation helpers every tool builds on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
