# SPDX-License-Identifier: AGPL-3.0-or-later
"""Git tools: the shared ``_git`` runner plus the status/diff/log read commands and the commit/push/create-branch write commands."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .base import ToolResult, _partial_on_timeout, _truncate, run_subprocess

def _git(cwd: Path, *args: str, timeout: int = 10,
         env: Optional[dict] = None) -> tuple[str, bool]:
    """Run a git command and return (output, ok)."""
    result = run_subprocess(["git", *args], cwd, timeout=timeout, env=env)
    if result.not_found:
        return "git not found in PATH", False
    if result.timed_out:
        return f"git {args[0]} timed out{_partial_on_timeout(result)}", False
    if result.error is not None:
        return result.error, False
    out = (result.stdout + result.stderr).strip() or "(no output)"
    return out, result.ok


def tool_git_status(cwd: Path) -> ToolResult:
    """Return the output of `git status --short` in the working directory."""
    out, ok = _git(cwd, "status", "--short", "--branch")
    return ToolResult(ok=ok, output=out, summary=f"git status ({len(out.splitlines())} lines)")


def tool_git_diff(cwd: Path, path: str = "", staged: bool = False) -> ToolResult:
    """Return `git diff` output."""
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
    """Stage files and create a git commit."""
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


# --------------------------------------------------------------------------
# Worktree helpers (used by tools/parallel.py; deliberately not registry tools)
# --------------------------------------------------------------------------

# Every worktree this feature creates is named with this prefix, so an orphan left
# behind by a hard kill is identifiable as OURS and can be reaped without guessing.
# A crash between "add" and "remove" cannot be prevented, only made discoverable.
WORKTREE_PREFIX = "coder-child-"

# Where child worktrees live, relative to the repo root. Matches the convention this
# repo already uses for agent worktrees.
WORKTREE_SUBDIR = Path(".claude") / "worktrees"


def git_repo_root(cwd: Path) -> Optional[Path]:
    """Absolute path of the repo containing *cwd*, or None if it is not a repo."""
    out, ok = _git(cwd, "rev-parse", "--show-toplevel")
    if not ok:
        return None
    first = out.splitlines()[0].strip() if out else ""
    return Path(first) if first else None


def git_current_branch(cwd: Path) -> str:
    """Current branch name, or '' when detached or unavailable."""
    out, ok = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    if not ok:
        return ""
    name = out.splitlines()[0].strip() if out else ""
    return "" if name == "HEAD" else name


def git_is_dirty(cwd: Path) -> bool:
    """True when the working tree has staged, unstaged, or untracked changes."""
    out, ok = _git(cwd, "status", "--porcelain")
    if not ok:
        return False
    return bool(out.strip()) and out.strip() != "(no output)"


def _ensure_locally_ignored(repo: Path, rel: str) -> None:
    """Make *rel* ignored via .git/info/exclude (local, untracked)."""
    _, already = _git(repo, "check-ignore", "-q", rel)
    if already:
        return
    exclude = repo / ".git" / "info" / "exclude"
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        if rel in existing.splitlines():
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        sep = "" if (not existing or existing.endswith("\n")) else "\n"
        exclude.write_text(
            f"{existing}{sep}# added by localm coder parallel dispatch\n{rel}\n",
            encoding="utf-8",
        )
    except OSError:
        # Cosmetic only: without this the child worktree merely shows as untracked
        # in the user's git status. Never worth failing a dispatch over.
        pass


def git_worktree_add(repo: Path, path: Path, branch: str,
                     base: str = "HEAD") -> tuple[str, bool]:
    """Create a worktree at *path* on a NEW branch *branch* based at *base*."""
    if not branch or branch in {"master", "main", "HEAD"}:
        return (f"refusing to create a child worktree on '{branch}': child work "
                "must go on its own fresh branch, never a shared one"), False
    # An existing branch would mean reusing state from a previous run; the caller
    # generates unique names, so a collision here is a bug worth reporting loudly
    # rather than silently resuming into someone else's branch.
    _, exists = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if exists:
        return f"branch '{branch}' already exists; refusing to reuse it", False

    path.parent.mkdir(parents=True, exist_ok=True)
    # Ignore the worktrees DIRECTORY once, not each child path: a per-child entry
    # would grow .git/info/exclude by a line on every dispatch, forever.
    try:
        path.relative_to(repo)
    except ValueError:
        pass
    else:
        _ensure_locally_ignored(repo, WORKTREE_SUBDIR.as_posix() + "/")

    out, ok = _git(repo, "worktree", "add", "-b", branch, str(path), base, timeout=60)
    return out, ok


def git_worktree_remove(repo: Path, path: Path) -> tuple[str, bool]:
    """Remove the worktree at *path*."""
    out, ok = _git(repo, "worktree", "remove", str(path), timeout=60)
    if ok:
        return out, True

    low = out.lower()
    if "locked" in low:
        reason = "the worktree is locked"
    elif "contains modified or untracked files" in low or "dirty" in low:
        reason = ("it still has uncommitted changes (the child's work was not "
                  "committed - not removing, so nothing is lost)")
    elif ("permission" in low or "being used" in low or "busy" in low
          or "not empty" in low):
        reason = ("a live process still holds it (this is real, not spurious - "
                  "something is still running in that directory)")
    else:
        reason = "git declined"
    return f"could not remove worktree {path}: {reason}: {out}", False


# `git worktree prune -n -v` reports each record it WOULD drop as
# "Removing worktrees/<name>: <reason>", where <name> is the administrative
# record under .git/worktrees/ and derives from the worktree directory's own
# basename. Verified against git 2.54.0.
_PRUNE_RECORD_RE = re.compile(r"^Removing\s+worktrees/(?P<name>.+?):")


def git_prune_child_worktrees(repo: Path) -> tuple[str, bool]:
    """Prune stale worktree records, but ONLY when every one of them is ours."""
    # Force git's own messages to English FOR THIS PROBE. The line we parse is
    # gettext-translated (git's source emits `Removing %s/%s: %s` through _()),
    # so on a build that ships message catalogs a localized line would fail the
    # regex below - and the fail-closed branch would then report OUR OWN records
    # as foreign, print a message that is simply untrue, and disable the cleanup
    # permanently on that machine. Merged onto os.environ, never passed bare:
    # env REPLACES the environment, and a git without PATH does not run at all.
    probe_env = {**os.environ, "LC_ALL": "C", "LANGUAGE": ""}
    out, ok = _git(repo, "worktree", "prune", "-n", "-v", timeout=30,
                   env=probe_env)
    if not ok:
        return f"could not check what `git worktree prune` would remove: {out}", False

    ours: list[str] = []
    foreign: list[str] = []
    for line in out.splitlines():
        text = line.strip()
        if not text or text == "(no output)":
            continue
        match = _PRUNE_RECORD_RE.match(text)
        if match is None:
            # An unrecognised line (a git output-format change, or a translated
            # build) means we cannot tell whose record it is. Fail closed.
            foreign.append(text)
        elif match.group("name").startswith(WORKTREE_PREFIX):
            ours.append(match.group("name"))
        else:
            foreign.append(match.group("name"))

    if foreign:
        return (
            "skipped `git worktree prune`: it would also drop worktree records "
            "this plugin does not own (" + "; ".join(foreign) + "). A worktree on "
            "an unmounted drive or network share looks missing to git, so pruning "
            "would discard it; restore one with `git worktree repair`."
        ), False
    if not ours:
        return "(nothing to prune)", True
    return _git(repo, "worktree", "prune", timeout=30)


def git_list_child_worktrees(repo: Path) -> list[Path]:
    """Every registered worktree whose directory name marks it as one of ours."""
    out, ok = _git(repo, "worktree", "list", "--porcelain", timeout=30)
    if not ok:
        return []
    found: list[Path] = []
    for line in out.splitlines():
        m = re.match(r"^worktree\s+(.*)$", line.strip())
        if not m:
            continue
        p = Path(m.group(1).strip())
        if p.name.startswith(WORKTREE_PREFIX):
            found.append(p)
    return found


def git_commit_all_in(cwd: Path, message: str) -> tuple[str, bool]:
    """Stage and commit everything in *cwd*, including untracked files."""
    out, ok = _git(cwd, "add", "-A", timeout=30)
    if not ok:
        return f"failed to stage child work: {out}", False
    out, ok = _git(cwd, "commit", "-m", message, timeout=60)
    if not ok and "nothing to commit" in out.lower():
        return "nothing to commit", True
    return out, ok


def tool_git_create_branch(
    cwd: Path,
    name: str,
    checkout: bool = True,
) -> ToolResult:
    """Create a new git branch."""
    if checkout:
        out, ok = _git(cwd, "checkout", "-b", name)
    else:
        out, ok = _git(cwd, "branch", name)
    if not ok:
        return ToolResult.error(f"git branch failed: {out}")
    action = "created and checked out" if checkout else "created"
    return ToolResult.success(out or f"Branch '{name}' {action}.", summary=f"git branch {name}")
