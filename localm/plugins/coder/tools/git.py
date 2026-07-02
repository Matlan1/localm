# SPDX-License-Identifier: AGPL-3.0-or-later
"""Git tools: the shared ``_git`` runner plus the status/diff/log read commands
and the commit/push/create-branch write commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .base import ToolResult, _partial_on_timeout, _truncate

def _git(cwd: Path, *args: str, timeout: int = 10) -> tuple[str, bool]:
    """Run a git command and return (output, ok)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        out = (proc.stdout + proc.stderr).strip() or "(no output)"
        return out, proc.returncode == 0
    except FileNotFoundError:
        return "git not found in PATH", False
    except subprocess.TimeoutExpired as e:
        return f"git {args[0]} timed out{_partial_on_timeout(e)}", False
    except Exception as e:
        return str(e), False


def tool_git_status(cwd: Path) -> ToolResult:
    """Return the output of `git status --short` in the working directory."""
    out, ok = _git(cwd, "status", "--short", "--branch")
    return ToolResult(ok=ok, output=out, summary=f"git status ({len(out.splitlines())} lines)")


def tool_git_diff(cwd: Path, path: str = "", staged: bool = False) -> ToolResult:
    """
    Return `git diff` output.

    Parameters
    ----------
    path   : limit diff to this file or directory (optional)
    staged : if True, show staged changes (`git diff --cached`)
    """
    args = ["diff", "--stat", "-p"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args, timeout=15)
    out, trunc = _truncate(out)
    return ToolResult(ok=ok, output=out, summary="git diff" + (" --cached" if staged else ""), truncated=trunc)


def tool_git_log(cwd: Path, n: int = 10, path: str = "") -> ToolResult:
    """Return the last n commits as a compact log."""
    args = ["log", f"--max-count={n}", "--oneline", "--decorate"]
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args)
    return ToolResult(ok=ok, output=out, summary=f"git log -{n}")


def tool_git_commit(
    cwd: Path,
    message: str,
    files: Optional[list] = None,
) -> ToolResult:
    """
    Stage files and create a git commit.

    Parameters
    ----------
    message:
        Commit message.
    files:
        List of file paths to stage.  When omitted, stages all tracked
        modifications (``git add -A``).
    """
    # Stage
    if files:
        for f in files:
            out, ok = _git(cwd, "add", "--", f)
            if not ok:
                return ToolResult.error(f"Failed to stage '{f}': {out}")
    else:
        out, ok = _git(cwd, "add", "-A")
        if not ok:
            return ToolResult.error(f"Failed to stage changes: {out}")

    # Commit
    out, ok = _git(cwd, "commit", "-m", message, timeout=30)
    if not ok:
        if "nothing to commit" in out.lower():
            return ToolResult.success(
                "Nothing to commit, working tree clean.",
                summary="git commit - nothing to commit",
            )
        return ToolResult.error(f"git commit failed: {out}")

    return ToolResult.success(out, summary=f"git commit: {message[:60]}")


def tool_git_push(
    cwd: Path,
    remote: str = "origin",
    branch: str = "",
) -> ToolResult:
    """Push the current branch to a remote."""
    args = ["push", remote]
    if branch:
        args.append(branch)
    out, ok = _git(cwd, *args, timeout=60)
    if not ok:
        return ToolResult.error(f"git push failed: {out}")
    return ToolResult.success(out, summary=f"git push {remote}")


def tool_git_create_branch(
    cwd: Path,
    name: str,
    checkout: bool = True,
) -> ToolResult:
    """
    Create a new git branch.

    Parameters
    ----------
    name:
        Branch name, e.g. ``feat/my-feature``.
    checkout:
        Switch to the new branch after creating it (default: True).
    """
    if checkout:
        out, ok = _git(cwd, "checkout", "-b", name)
    else:
        out, ok = _git(cwd, "branch", name)
    if not ok:
        return ToolResult.error(f"git branch failed: {out}")
    action = "created and checked out" if checkout else "created"
    return ToolResult.success(out or f"Branch '{name}' {action}.", summary=f"git branch {name}")
