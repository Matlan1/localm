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

    def undo(self) -> str | None:
        """
        Revert the last undoable file operation (write_file, edit_file, patch_file).

        Returns a human-readable summary of what was restored, or None if the
        undo stack is empty.
        """
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        path: Path    = entry["path"]
        old: bytes | None = entry["old_content"]
        tool: str     = entry["tool"]
        try:
            if old is None:
                # File didn't exist before - delete it
                if path.exists():
                    path.unlink()
                return f"Undid {tool}: deleted {path} (file was new)"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(old)
                lines = old.count(b"\n") + 1
                return f"Undid {tool}: restored {path} ({lines} lines)"
        except Exception as e:
            return f"Undo failed: {e}"
