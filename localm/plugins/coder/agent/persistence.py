# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durable + in-session state: project-map build, the resume checkpoint
(save/load/clear/resume + path resolution), the changed-files tracker, the
cumulative session diff, and undo. Mixed into Agent."""

from __future__ import annotations

import datetime
import difflib
import json
from pathlib import Path
from typing import Optional

import localm.plugins.coder.agent as _agent
from ..indexer import ProjectMap
from ..audit import SessionMode
from .checkpoint import (
    _checkpoint_path_for, _index_deadline, _legacy_checkpoint_path_for,
    _read_checkpoint,
)


class _PersistenceMixin:
    def changed_files(self) -> list[dict]:
        """
        Files this session has written, with change counts.

        Each entry: ``{path, writes, created, exists, last_tool}`` where
        *created* means the file did not exist before this session touched it
        and *exists* is its current on-disk state (False = since deleted).
        """
        # Snapshot first - the GUI reads this from another thread while the
        # agent loop may be inserting entries.
        snapshot = dict(self._changed_files)
        out = []
        for key in sorted(snapshot):
            e = snapshot[key]
            abs_path = (self.cwd / key)
            out.append({
                "path": key,
                "writes": e["writes"],
                "created": e["original"] is None,
                "exists": abs_path.is_file(),
                "last_tool": e["last_tool"],
            })
        return out

    def session_diff(self, path: Optional[str] = None) -> str:
        """
        Cumulative unified diff of everything this session changed.

        Compares each tracked file's first-seen original content against its
        current on-disk state - so three successive edits to one file show as
        one combined diff. Pass *path* for a single file, None for all.
        Returns "" when nothing was changed (or the path is untracked).
        """
        snapshot = dict(self._changed_files)   # cross-thread read safety
        keys = [path] if path else sorted(snapshot)
        parts: list[str] = []
        for key in keys:
            entry = snapshot.get(key)
            if entry is None:
                continue
            original = entry["original"]
            old_text = (original.decode("utf-8", errors="replace")
                        if original is not None else "")
            abs_path = (self.cwd / key)
            try:
                new_text = (abs_path.read_text(encoding="utf-8", errors="replace")
                            if abs_path.is_file() else "")
            except Exception:
                new_text = ""
            diff = "".join(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{key}" if original is not None else "/dev/null",
                tofile=f"b/{key}" if new_text else "/dev/null",
            ))
            if diff:
                parts.append(diff)
        return "\n".join(parts)

    def _record_changed_file(self, path_arg: str, old_content: bytes | None,
                             tool: str) -> None:
        """Track a successful file write in the changed-files map."""
        abs_path = (self.cwd / path_arg).resolve()
        try:
            key = abs_path.relative_to(self.cwd.resolve()).as_posix()
        except ValueError:
            key = str(abs_path)
        entry = self._changed_files.get(key)
        if entry is None:
            self._changed_files[key] = {
                "original": old_content, "writes": 1, "last_tool": tool,
            }
        else:
            entry["writes"] += 1
            entry["last_tool"] = tool

    def _absorb_child_state(self, child) -> None:
        """Fold a spawned child agent's changed-files and error trace into this
        parent (audit cluster 11).

        A child from ``spawn_agent`` shares this cwd but is never ``close()``d, so
        without this its delegated file changes and failures would never reach an
        episode. Merging them here lets the parent's single close-time episode
        cover the delegated work too. Best-effort: called guarded so bookkeeping
        never breaks the tool."""
        from .constants import _MAX_ERROR_TRACE
        for key, centry in getattr(child, "_changed_files", {}).items():
            pentry = self._changed_files.get(key)
            if pentry is None:
                # Keep the child's first-seen original so session_diff() shows the
                # full change; copy so later parent edits do not mutate the child.
                self._changed_files[key] = dict(centry)
            else:
                pentry["writes"] = pentry.get("writes", 0) + centry.get("writes", 0)
                if centry.get("last_tool"):
                    pentry["last_tool"] = centry["last_tool"]
        child_errors = getattr(child, "_error_trace", None)
        if child_errors:
            self._error_trace.extend(child_errors)
            if len(self._error_trace) > _MAX_ERROR_TRACE:
                self._error_trace = self._error_trace[-_MAX_ERROR_TRACE:]

    def _git_status_map(self) -> "dict[str, str] | None":
        """Map of dirty path -> 2-char ``git status --porcelain`` code in cwd, or
        None when cwd is not a git work tree or git is unavailable. Best-effort
        helper for episodic change detection - never raises. The code lets the
        caller tell an untracked new file (``??``) from a tracked edit so the diff
        can be built correctly."""
        from ..tools.base import run_subprocess
        try:
            r = run_subprocess(["git", "status", "--porcelain"], self.cwd, timeout=10)
        except Exception:
            return None
        if r.not_found or r.error is not None or r.returncode != 0:
            return None      # not a git work tree, or git unavailable
        out: dict[str, str] = {}
        for line in str(r.stdout or "").splitlines():
            if len(line) < 4:
                continue
            code = line[:2]
            p = line[3:].strip()
            if " -> " in p:                     # rename: "R  old -> new"
                p = p.split(" -> ", 1)[1].strip()
            p = p.strip().strip('"')
            if p:
                out[p] = code
        return out

    def _git_status_paths(self) -> "frozenset[str] | None":
        """The set of dirty (changed/untracked) paths at cwd, or None when cwd is
        not a git work tree. The pre-shell baseline for episodic change detection."""
        m = self._git_status_map()
        return None if m is None else frozenset(m)

    def _git_delta_diff(self, paths: list, status: dict) -> str:
        """A unified diff SCOPED to *paths* (this session's detected delta), so a
        pre-existing dirty file OUTSIDE the delta never leaks into the work log.

        Tracked edits among the delta come from ``git diff HEAD -- <paths>`` (a
        pathspec, not the whole tree); untracked new files (``??``, which
        ``git diff`` omits entirely) get a capped content snapshot appended so the
        session's actual output still appears. Best-effort - '' on any failure."""
        from ..tools.base import run_subprocess
        parts: list[str] = []
        tracked = [p for p in paths if status.get(p, "") != "??"]
        if tracked:
            for argv in (["git", "diff", "HEAD", "--", *tracked],
                         ["git", "diff", "--", *tracked]):
                try:
                    r = run_subprocess(argv, self.cwd, timeout=15)
                except Exception:
                    r = None
                if r is not None and r.returncode == 0 and r.error is None:
                    if r.stdout:
                        parts.append(str(r.stdout))
                    break
        for p in paths:
            if status.get(p, "") != "??":
                continue
            fp = self.cwd / p
            try:
                if fp.is_file():
                    text = fp.read_text(encoding="utf-8", errors="replace")[:4000]
                    body = "".join("+" + ln + "\n" for ln in text.splitlines())
                    parts.append(f"--- /dev/null\n+++ b/{p}\n{body}")
            except Exception:
                pass          # unreadable/binary new file: skip its snapshot
        return "\n".join(p for p in parts if p)

    def _detect_shell_changes(self) -> "tuple[list[dict], str]":
        """Best-effort detection of files changed via run_shell (git apply, a
        formatter, codegen) that the write-tool tracker never recorded.

        Returns ``(changed_files_list, unified_diff)``, or ``([], "")`` when no
        run_shell ran, cwd is not a git work tree, or nothing new changed since the
        pre-shell baseline. BOTH the file list and the diff are scoped to THIS
        session by subtracting the baseline captured before the first run_shell, so
        a pre-existing dirty tree is not misattributed (audit cluster 11)."""
        if not self._shell_baseline_captured or self._git_baseline is None:
            return [], ""
        current = self._git_status_map()
        if current is None:
            return [], ""
        delta = sorted(set(current) - self._git_baseline)
        if not delta:
            return [], ""
        # Only `path` is consumed by the reflection; keep the other fields to match
        # the changed_files() dict shape for any future reader (created is True for
        # an untracked new file, "??").
        changed = [{
            "path": p,
            "writes": 1,
            "created": current.get(p, "") == "??",
            "exists": (self.cwd / p).is_file(),
            "last_tool": "run_shell",
        } for p in delta]
        from localm.debuglog import logger
        logger.debug("episodic memory: %d file(s) detected via git after a "
                     "shell-driven session", len(delta))
        return changed, self._git_delta_diff(delta, current)

    def undo_list(self) -> list[dict]:
        """The undo stack, most recent first: ``[{tool, path}, ...]``."""
        return [{"tool": e["tool"], "path": str(e["path"])}
                for e in reversed(self._undo_stack)]

    def _build_project_map(self, cwd: Path) -> ProjectMap:
        """Index the project with a config-driven deadline, and surface a one-line
        note when a large tree is slow or truncated so a session started on a huge
        root (e.g. C:\\) shows progress instead of appearing to hang (CODER-1)."""
        # Live-attribute access so a test patching agent.ProjectMap is honoured
        # (the name moved into this submodule when agent.py became a package).
        ProjectMap = _agent.ProjectMap
        import time
        t0 = time.monotonic()
        pm = ProjectMap.build(cwd, deadline_s=_index_deadline())
        took = time.monotonic() - t0
        if pm.truncated or took > 2.0:
            suffix = (" (large project - index truncated; the agent can still "
                      "list_dir / search_files)" if pm.truncated else "")
            self._emit("info",
                       text=f"Indexed {pm.file_count()} files in {took:.1f}s{suffix}")
        return pm

    @property
    def _checkpoint_path(self) -> Path:
        # Session data lives under HOME, not in the project tree (CODER-4).
        return _checkpoint_path_for(self.cwd)

    @property
    def _legacy_checkpoint_path(self) -> Path:
        return _legacy_checkpoint_path_for(self.cwd)

    def save_checkpoint(self) -> None:
        """Persist current conversation state so it can be resumed later.

        No-op in privacy mode - the checkpoint contains the full
        conversation, which privacy mode promises never to write to disk."""
        if self.mode == SessionMode.PRIVACY:
            return
        data = {
            "version": 1,
            "interrupted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "turns": self._turns,
            "total_tokens": self._total_tokens,
            "messages": self._messages,
        }
        p = self._checkpoint_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # never let checkpoint failure crash the session

    def clear_checkpoint(self) -> None:
        """Remove any saved checkpoint for this working directory (new HOME
        location and the legacy in-project one)."""
        for p in (self._checkpoint_path, self._legacy_checkpoint_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def load_checkpoint(self) -> dict | None:
        """
        Read the checkpoint file if it exists and is valid.

        Checks the new HOME location first, then the legacy in-project path so a
        checkpoint saved by an older build can still be resumed (CODER-4).
        Returns the parsed dict, or None if no checkpoint is found.
        """
        for p in (self._checkpoint_path, self._legacy_checkpoint_path):
            data = _read_checkpoint(p)
            if data is not None:
                return data
        return None

    def resume_checkpoint(self, data: dict) -> None:
        """Restore agent state from a checkpoint dict."""
        self._messages     = data["messages"]
        self._turns        = data.get("turns", len(self._messages))
        self._total_tokens = data.get("total_tokens", 0)

    def _undo_one(self, entry: dict) -> tuple[str, bool]:
        """Revert a single undo-stack entry. Returns (description, ok)."""
        path: Path = entry["path"]
        old: bytes | None = entry["old_content"]
        try:
            if old is None:
                # File didn't exist before - delete it
                if path.exists():
                    path.unlink()
                return f"deleted {path} (file was new)", True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(old)
            lines = old.count(b"\n") + 1
            return f"restored {path} ({lines} lines)", True
        except Exception as e:
            return f"FAILED to restore {path}: {e}", False

    def undo(self) -> str | None:
        """
        Revert the last undoable file operation (write_file, edit_file,
        edit_files, patch_file, edit_notebook_cell).

        A multi-file call (edit_files) pushes one stack entry PER FILE but is a
        single operation, so all entries sharing its call id are reverted
        together. Undoing only the last of them would leave the other files
        edited while reporting the operation undone - a half-undone state the
        caller was told does not exist.

        Returns a human-readable summary of what was restored, or None if the
        undo stack is empty.
        """
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        tool: str = entry["tool"]
        call_id = entry.get("call_id")
        group = [entry]
        # An entry with no call id is never grouped (single-file tools, and any
        # entry predating call ids), so their behaviour is unchanged.
        if call_id is not None:
            while self._undo_stack and self._undo_stack[-1].get("call_id") == call_id:
                group.append(self._undo_stack.pop())

        parts, failures = [], []
        for item in group:
            desc, ok = self._undo_one(item)
            (parts if ok else failures).append(desc)

        if len(group) == 1 and not failures:
            return f"Undid {tool}: {parts[0]}"
        summary = f"Undid {tool}: {len(parts)} of {len(group)} file(s) - " + "; ".join(parts)
        if failures:
            # RULE 5: a partial undo must never read as a complete one.
            summary += ("\nWARNING: the undo did NOT fully succeed: "
                        + "; ".join(failures)
                        + " - these files still hold the change.")
        return summary
