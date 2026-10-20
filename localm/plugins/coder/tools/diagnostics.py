# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-edit diagnostics: a ruff pass over the Python files a file tool just wrote.

Runs ``ruff check --select F,E9`` (pyflakes and syntax-level errors) on the written
files and returns a short block for the tool result. Nothing is installed: when ruff
does not resolve, the result is None and the tool result is unchanged.

Process-spawning, so a restricted session never reaches it (the file tools take a
``_restricted`` flag, default-deny). Project opt-out: ``diagnostics = false`` in
``.localcoder/config.toml``. After repeated timeouts or failures in one project the
pass stops for that project until the process restarts."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger(__name__)

# The file tools that run the pass; each takes the session's ``_restricted`` flag.
DIAGNOSED_TOOLS = frozenset({"write_file", "edit_file", "edit_files", "patch_file"})

MAX_DIAGNOSTICS = 10
TIMEOUT_SECONDS = 5.0
MAX_FILE_BYTES = 1024 * 1024
MAX_FAILURES = 2
_RULES = "F,E9"
_DEFAULT_PATHEXT = ".EXE;.COM"
_OFF_WORDS = frozenset({"false", "off", "no", "0"})
_FINDING = re.compile(r"^.+?:\d+:\d+: \S")

# Consecutive timeouts/failures per project directory.
_failures: dict[str, int] = {}


def _resolve_ruff(path_value: str, pathext: str) -> Optional[str]:
    """The absolute path of the first ``ruff`` executable in the absolute entries of
    *path_value*, or None.

    Empty and relative entries are skipped, and the current directory is never
    searched, so an executable inside the project tree runs only when the user put
    its directory on PATH (an activated virtualenv does). On Windows only ``.exe``
    and ``.com`` count."""
    suffixes = [""]
    if sys.platform == "win32":
        listed = (pathext or _DEFAULT_PATHEXT).lower().split(os.pathsep)
        suffixes = [e for e in listed if e in (".exe", ".com")] or [".exe", ".com"]
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
    """False when the project's ``.localcoder/config.toml`` turns ``diagnostics`` off
    (``false``, ``0`` or one of ``"false"``, ``"off"``, ``"no"``, ``"0"``).

    An unreadable config disables diagnostics: the opt-out cannot be read, so it
    cannot be honoured."""
    from ..project_config import ProjectConfigUnreadable, load_project_config
    try:
        cfg = load_project_config(cwd)
    except ProjectConfigUnreadable as e:
        log.debug("post-edit diagnostics skipped, project config unreadable: %s", e)
        return False
    value = cfg.get("diagnostics", True)
    if isinstance(value, str):
        return value.strip().lower() not in _OFF_WORDS
    return bool(value)


def _target_arg(path: Path, cwd: Path) -> str:
    """*path* as a ruff argument: relative to *cwd*, or absolute when it cannot be
    made relative (another drive) or its relative form would start with ``@`` (ruff
    reads such an argument as a file of arguments)."""
    try:
        arg = os.path.relpath(path, cwd)
    except ValueError:
        return os.path.abspath(path)
    return os.path.abspath(path) if arg.startswith("@") else arg


def _finding_lines(stdout: str) -> list[str]:
    """The ``path:line:col: CODE message`` lines of ruff's concise output."""
    return [ln.strip() for ln in stdout.splitlines() if _FINDING.match(ln.strip())]


def _note_failure(cwd: Path, note: str) -> str:
    """Count a timeout or failure for *cwd* and return *note*, with a closing remark
    once the project has reached MAX_FAILURES."""
    key = str(cwd)
    _failures[key] = _failures.get(key, 0) + 1
    if _failures[key] >= MAX_FAILURES:
        return f"{note}; post-edit diagnostics are off for this project until restart"
    return note


def run_diagnostics(cwd: Path, paths: Sequence[Path]) -> Optional[str]:
    """The ``[ruff]`` block for the Python files in *paths*, or None when there is
    nothing to say (ruff absent, opted out, stopped after repeated failures, no
    Python file, or no findings).

    A timeout or a ruff failure returns a one-line note instead of None, because
    silence would read as a clean result."""
    py = [p for p in paths if p.suffix.lower() == ".py"]
    if not py or _failures.get(str(cwd), 0) >= MAX_FAILURES:
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
        targets.append(_target_arg(p, cwd))
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
        return _note_failure(
            cwd, f"[ruff] diagnostics skipped: timed out after {TIMEOUT_SECONDS:g}s")
    except OSError as e:
        log.debug("post-edit diagnostics: could not start ruff: %s", e)
        return _note_failure(
            cwd, f"[ruff] diagnostics skipped: could not start ruff ({e.__class__.__name__})")
    if proc.returncode not in (0, 1):
        detail = [ln.strip() for ln in proc.stderr.splitlines()
                  if ln.strip() and ln.strip() != "ruff failed"]
        reason = (detail or [f"exit {proc.returncode}"])[0].replace(str(cwd), ".")[:160]
        return _note_failure(cwd, f"[ruff] diagnostics skipped: ruff failed ({reason})")
    _failures.pop(str(cwd), None)
    if proc.returncode == 0:
        return None
    findings = _finding_lines(proc.stdout)
    if not findings:
        return "[ruff] reported issues but its output could not be read"
    shown = findings[:MAX_DIAGNOSTICS]
    more = len(findings) - len(shown)
    head = f"[ruff {_RULES}] {len(findings)} issue(s):"
    body = "\n".join(f"  {line}" for line in shown)
    tail = f"\n  ... and {more} more" if more else ""
    return f"{head}\n{body}{tail}"
