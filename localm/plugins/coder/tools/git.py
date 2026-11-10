# SPDX-License-Identifier: AGPL-3.0-or-later
"""Git tools: the shared ``_git`` runner plus the status/diff/log read commands
and the commit/push/create-branch write commands.

Also holds the git-worktree helpers used by ``tools/parallel.py`` to give each
concurrently-dispatched child agent its own isolated checkout. Those are plain
helpers, NOT entries in TOOL_REGISTRY: handing the model a raw ``git worktree add``
would let it create checkouts at arbitrary paths for no benefit the parallel
dispatch does not already provide. Keeping them un-exposed keeps the blast radius
to the one caller that needs them."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .base import ToolResult, _partial_on_timeout, _truncate, run_subprocess

_NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"

_HARDENING_CONFIG = (
    "core.fsmonitor=",
    f"core.hooksPath={_NULL_DEVICE}",
    "protocol.ext.allow=never",
    "safe.bareRepository=explicit",
    "diff.external=",
)

_NO_DRIVER_COMMANDS = frozenset({"diff", "log", "show"})

_DROPPED_ENV = ("GIT_EXTERNAL_DIFF", "GIT_PAGER", "PAGER")


def git_argv(*args: str) -> list[str]:
    """Full argv for a git command that cannot run a program named by the
    repository: fsmonitor, hooks, ext:: transports and a discovered bare
    repository are switched off, and diff, log and show skip external diff
    and textconv drivers."""
    argv = ["git"]
    for item in _HARDENING_CONFIG:
        argv += ["-c", item]
    if args and args[0] in _NO_DRIVER_COMMANDS:
        return [*argv, args[0], "--no-ext-diff", "--no-textconv", *args[1:]]
    return [*argv, *args]


def git_env(env: Optional[dict] = None) -> dict:
    """Environment for a hardened git: *env* (default ``os.environ``) copied,
    with program-naming variables removed. The system and global config stay in
    effect (line endings, credential helper, symlinks); the ``-c`` settings in
    :func:`git_argv` override them for every hardened key. See
    test_system_line_ending_config_still_applies."""
    out = dict(os.environ if env is None else env)
    for name in _DROPPED_ENV:
        out.pop(name, None)
    return out


def repo_runs_programs(cwd: Path) -> bool:
    """True when the repository at *cwd* has local configuration or hooks that
    can make a read command run a program, or that cannot be inspected."""
    from .. import shell_guard
    try:
        return shell_guard._git_runs_programs(str(cwd), str(cwd))
    except Exception:
        return True


def run_git(argv_args: tuple, cwd: Path, *, timeout: float,
            env: Optional[dict] = None):
    """Run a hardened git and return the raw SubprocessResult."""
    return run_subprocess(git_argv(*argv_args), cwd, timeout=timeout, env=git_env(env))


def _git(cwd: Path, *args: str, timeout: int = 10,
         env: Optional[dict] = None) -> tuple[str, bool]:
    """Run a hardened git command and return (output, ok).

    *env* is the base environment (default ``os.environ``); the hardening
    variables are applied on top of it.
    """
    result = run_git(args, cwd, timeout=timeout, env=env)
    if result.not_found:
        return "git not found in PATH", False
    if result.timed_out:
        return f"git {args[0]} timed out{_partial_on_timeout(result)}", False
    if result.error is not None:
        return result.error, False
    out = (result.stdout + result.stderr).strip() or "(no output)"
    return out, result.ok


_RISKY_REFUSAL = ("refused: this repository's configuration or hooks can run "
                  "programs, so git is not run on it")


def _refuse_risky(cwd: Path, name: str) -> Optional[ToolResult]:
    if repo_runs_programs(cwd):
        return ToolResult.error(f"{name} {_RISKY_REFUSAL}")
    return None


def tool_git_status(cwd: Path) -> ToolResult:
    """Return the output of `git status --short` in the working directory."""
    refused = _refuse_risky(cwd, "git status")
    if refused is not None:
        return refused
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
    refused = _refuse_risky(cwd, "git diff")
    if refused is not None:
        return refused
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
    refused = _refuse_risky(cwd, "git log")
    if refused is not None:
        return refused
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
    """Push the current branch to a remote.

    A *remote* that is empty after stripping, or a *remote* or *branch* that
    begins with ``-``, is refused; the two are passed after ``--`` so git never
    reads either as an option."""
    remote = (remote or "origin").strip()
    branch = (branch or "").strip()
    for label, value in (("remote", remote), ("branch", branch)):
        if value.startswith("-"):
            return ToolResult.error(
                f"git push refused: the {label} {value!r} begins with '-', which "
                "git would read as an option.")
    if not remote:
        return ToolResult.error("git push refused: the remote is empty.")
    args = ["push", "--", remote]
    if branch:
        args.append(branch)
    out, ok = _git(cwd, *args, timeout=60)
    if not ok:
        return ToolResult.error(f"git push failed: {out}")
    return ToolResult.success(out, summary=f"git push {remote}")


# --------------------------------------------------------------------------
# Worktree helpers (used by tools/parallel.py; not registry tools)
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
    """Current branch name, or "" when detached or unavailable."""
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
    """Make *rel* ignored via .git/info/exclude (local, untracked).

    A worktree created INSIDE the repo shows up as untracked in the user's
    ``git status`` unless the path is ignored, which is noise we would be
    inflicting on their working tree. ``.git/info/exclude`` is the right lever:
    it is local-only and never committed, so we are not editing the user's
    tracked ``.gitignore`` behind their back. Best-effort by design - failing to
    tidy git status must never fail a dispatch - but we only skip silently when
    the path is ALREADY ignored, which is the benign case.
    """
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
    """Create a worktree at *path* on a NEW branch *branch* based at *base*.

    Always creates a fresh branch (``worktree add -b``). That is what keeps this
    safe on two counts the caller depends on:
    - the new worktree is never checked out on master (or any existing branch), so
      it cannot collide with a branch already checked out elsewhere; and
    - the SHARED main checkout is never switched, branched, or reset - ``worktree
      add`` only ever touches the new directory.
    """
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
    """Remove the worktree at *path*.

    Deliberately NEVER passes ``--force``. A removal that fails because the tree is
    dirty or because a live process still holds it is a REAL condition, not a
    spurious error to bulldoze: ``--force`` would delete a child's uncommitted work,
    which is precisely the outcome this feature exists to prevent. The caller
    commits the child's work to its branch BEFORE removing, so a dirty tree here
    means something went wrong and the operator needs to know.
    """
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
# basename.
_PRUNE_RECORD_RE = re.compile(r"^Removing\s+worktrees/(?P<name>.+?):")


def git_prune_child_worktrees(repo: Path) -> tuple[str, bool]:
    """Prune stale worktree records, but ONLY when every one of them is ours.

    `git worktree prune` takes no pathspec (`usage: git worktree prune [-n] [-v]
    [--expire <expire>]`), so a bare call drops the administrative record of
    EVERY registered worktree whose directory is currently missing - in the
    USER's repo, not just localm's. Git's own documentation names the victim: a
    worktree "stored on a portable device or network share which is not always
    mounted" is indistinguishable from an abandoned one, and recovering it needs
    `git worktree repair`.

    So: ask git what it WOULD prune, and prune only when every record listed is
    one of ours (the ``coder-child-`` prefix). Anything else, INCLUDING a line
    that cannot be parsed, fails CLOSED - the prune is skipped and the reason is
    reported. A leftover record of our own is harmless and already surfaced in
    the dispatch report; a dropped foreign record cannot be recovered from here.

    Returns (message, ok). ok=False means nothing was pruned and the message says
    what stopped it, for the caller to surface rather than swallow.
    """
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
    """Every registered worktree whose directory name marks it as one of ours.

    Used to surface orphans left by a hard kill, so they can be reaped instead of
    silently accumulating.
    """
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
    """Stage and commit everything in *cwd*, including untracked files.

    Committing a child's work is what makes the worktree disposable: the BRANCH is
    the durable artifact the human reviews and merges, the worktree is transient.
    Committing is NOT merging - the parent's tree is untouched either way.
    """
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
