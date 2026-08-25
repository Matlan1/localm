# SPDX-License-Identifier: AGPL-3.0-or-later
"""Environment-reading tool: reads .env + the process environment with
secret-looking values redacted."""

from __future__ import annotations

import os
from pathlib import Path

from .base import ToolResult, _confine

# Env var names (substring match, case-insensitive) whose values are secrets
_SECRET_MARKERS = (
    "key", "token", "secret", "password", "passwd", "credential",
    "auth", "cookie", "private", "cert", "signature", "dsn",
)


def _redact_env_value(name: str, value: str) -> str:
    """Replace secret-looking values with a redaction marker."""
    lowered = name.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return f"<redacted, {len(value)} chars>"
    return value


def tool_read_env(cwd: Path, path: str = "") -> ToolResult:
    """
    Read environment configuration with secrets stripped.

    Without arguments, reads the project's ``.env`` file (if present) and the
    active process environment. With ``path``, reads that env-format file
    instead. Values of variables whose names look secret (KEY, TOKEN, SECRET,
    PASSWORD, ...) are redacted; only their length is shown.
    """
    lines: list[str] = []

    try:
        env_file = _confine(cwd, path) if path else (cwd / ".env")
    except PermissionError as e:
        return ToolResult.error(str(e))
    if env_file.is_file():
        lines.append(f"# {env_file.name}")
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            name = name.strip().removeprefix("export ").strip()
            lines.append(f"{name}={_redact_env_value(name, value.strip())}")
    elif path:
        return ToolResult.error(f"Env file not found: {path}")

    if not path:
        lines.append("")
        lines.append("# process environment")
        for name in sorted(os.environ):
            lines.append(f"{name}={_redact_env_value(name, os.environ[name])}")

    output = "\n".join(lines).strip() or "(no environment variables found)"
    n_vars = sum(1 for l in lines if "=" in l)
    return ToolResult.success(output, summary=f"read_env ({n_vars} vars, secrets redacted)")
