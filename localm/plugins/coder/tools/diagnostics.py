# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-edit diagnostics: a ruff pass over the Python files a file tool just wrote.

Runs ``ruff check --select F,E9`` (pyflakes and syntax-level errors) on the written
files and returns a short block for the tool result. Nothing is installed: when ruff
does not resolve, the result is None and the tool result is unchanged.

Process-spawning, so a restricted session never reaches it (the file tools take a
``_restricted`` flag, default-deny). Project opt-out: ``diagnostics = false`` in
``.localcoder/config.toml``."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger(__name__)

MAX_DIAGNOSTICS = 10
TIMEOUT_SECONDS = 5.0
MAX_FILE_BYTES = 1024 * 1024
_RULES = "F,E9"


@functools.lru_cache(maxsize=8)
def _resolve_ruff(path_value: str, pathext: str) -> Optional[str]:
    """The absolute path of the first ``ruff`` executable in the absolute entries of
    *path_value*, or None.

    Empty and relative entries are skipped, and the current directory is never
    searched, so an executable inside the project tree runs only when the user put
    its directory on PATH (an activated virtualenv does)."""
    suffixes = [""]
    if sys.platform == "win32":
        suffixes = [e for e in pathext.lower().split(os.pathsep) if e in (".exe", ".com")]
    for entry in path_value.split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        for suffix in suffixes:
            candidate = os.path.join(entry, "ruff" + suffix)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def find_ruff() -> Optional[str]:
    """The ruff executable on the current PATH, or None."""
    return _resolve_ruff(os.environ.get("PATH", ""), os.environ.get("PATHEXT", ""))


def diagnostics_enabled(cwd: Path) -> bool:
    """False when the project's ``.localcoder/config.toml`` sets ``diagnostics = false``.

    An unreadable config disables diagnostics: the opt-out cannot be read, so it
    cannot be honoured."""
    from ..project_config import ProjectConfigUnreadable, load_project_config
    try:
        cfg = load_project_config(cwd)
    except ProjectConfigUnreadable as e:
        log.debug("post-edit diagnostics skipped, project config unreadable: %s", e)
        return False
    return cfg.get("diagnostics", True) is not False


def _concise_lines(stdout: str) -> list[str]:
    """The finding lines of ruff's concise output: every non-blank line that is not
    ruff's trailing summary (``Found N errors.`` / ``[*] N fixable ...``)."""
    lines = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Found ", "No fixes", "[*]", "All checks")):
            continue
        lines.append(stripped)
    return lines


def run_diagnostics(cwd: Path, paths: Sequence[Path]) -> Optional[str]:
    """The ``[ruff]`` block for the Python files in *paths*, or None when there is
    nothing to say (ruff absent, opted out, no Python file, or no findings).

    Never raises. A timeout or a ruff failure returns a one-line note instead of
    None, because silence would read as a clean result."""
    py = [p for p in paths if p.suffix.lower() == ".py"]
    if not py:
        return None
    exe = find_ruff()
    if exe is None or not diagnostics_enabled(cwd):
        return None
    targets = []
    for p in py:
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        targets.append(os.path.relpath(p, cwd))
    if not targets:
        return None
    argv = [exe, "check", "--select", _RULES, "--output-format", "concise",
            "--no-fix", "--no-cache", "--force-exclude", "--", *targets]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
            errors="replace", stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS,
            **kwargs)
    except subprocess.TimeoutExpired:
        return f"[ruff] diagnostics skipped: timed out after {TIMEOUT_SECONDS:g}s"
    except OSError as e:
        log.debug("post-edit diagnostics: could not start ruff: %s", e)
        return f"[ruff] diagnostics skipped: could not start ruff ({e.__class__.__name__})"
    if proc.returncode == 0:
        return None
    if proc.returncode != 1:
        detail = [ln.strip() for ln in proc.stderr.splitlines()
                  if ln.strip() and ln.strip() != "ruff failed"]
        reason = (detail or [f"exit {proc.returncode}"])[0][:160]
        return f"[ruff] diagnostics skipped: ruff failed ({reason})"
    findings = _concise_lines(proc.stdout)
    if not findings:
        return None
    shown = findings[:MAX_DIAGNOSTICS]
    more = len(findings) - len(shown)
    head = f"[ruff {_RULES}] {len(findings)} issue(s):"
    body = "\n".join(f"  {line}" for line in shown)
    tail = f"\n  ... and {more} more" if more else ""
    return f"{head}\n{body}{tail}"
