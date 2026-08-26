# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durable + in-session state: project-map build, the resume checkpoint
(save/load/clear/resume + path resolution), the changed-files tracker, the
cumulative session diff, and undo. Mixed into Agent."""

from __future__ import annotations

import base64
import dataclasses
import datetime
import difflib
import json
from pathlib import Path
from typing import Optional

import localm.plugins.coder.agent as _agent
from localm.debuglog import logger
from localm.storekit import atomic_write
from ..indexer import ProjectMap
from ..audit import SessionMode
from .checkpoint import (
    _checkpoint_path_for, _derive_title, _index_deadline,
    _legacy_checkpoint_path_for, _legacy_home_checkpoint_path_for,
    _project_map_path_for, _read_checkpoint, list_checkpoints,
    migrate_legacy_checkpoint,
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
        parent.

        A child from ``spawn_agent`` shares this cwd but is never ``close()``d, so
        without this its delegated file changes and failures would never reach an
        episode. Merging them here lets the parent's single close-time episode
        cover the delegated work too. Best-effort: called guarded, so bookkeeping
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

    def _drain_background_agents(self) -> list:
        """Fold finished background sub-agents into this parent and describe them.

        Called at the TOP of the parent's turn, on the PARENT's own thread: the
        worker thread never touches parent state, so ``_changed_files`` /
        ``_error_trace`` keep the single-threaded invariant every other writer
        relies on, and absorption happens at a deterministic point rather than
        whenever a thread happened to finish. A lock would make the mutation
        atomic but NOT ordered - it could still interleave with the parent's own
        ``_track_write`` mid-turn, so ``session_diff()`` and the close-time
        episode could observe a half-absorbed view.

        Returns one note per finished job for the caller to put in front of the
        model. Best-effort: bookkeeping must never break the turn.
        """
        from .constants import _MAX_ERROR_TRACE
        try:
            from ..background import get_registry
            registry = get_registry()
            finished = registry.drain_finished(kind="agent")
        except Exception:
            return []

        # Completions the registry had to evict ahead of this drain.
        # Reported, because absorption is drain-only: an evicted child's summary,
        # branch and diff are unrecoverable.
        #
        # Its OWN try, AFTER the drain: drain_finished CONSUMES (it marks the batch
        # drained), so a failure inside the same try would discard completions that
        # were already handed over.
        try:
            lost = registry.take_dropped_undrained("agent")
        except Exception:
            lost = 0

        notes: list[str] = []
        if lost:
            notes.append(
                f"WARNING: {lost} background sub-agent completion(s) were "
                "discarded before this turn could collect them, so their "
                "results are lost. Re-run that work if you still need it.")
        for st in finished:
            try:
                job_id = st.get("id", "?")
                label = st.get("label", "sub-agent")
                result = st.get("result") or {}
                state = st.get("state", "done")
                job = registry.get(job_id)

                # ERROR TRACE ONLY. The child ran in its OWN worktree, so its
                # _changed_files keys are relative to a tree this parent does not
                # have; merging them would make session_diff() resolve them against
                # the PARENT's cwd. Errors carry no path, so that half stays valid.
                child = getattr(job, "child", None) if job is not None else None
                child_errors = getattr(child, "_error_trace", None)
                if child_errors:
                    self._error_trace.extend(child_errors)
                    if len(self._error_trace) > _MAX_ERROR_TRACE:
                        self._error_trace = self._error_trace[-_MAX_ERROR_TRACE:]

                # Did the child actually succeed? A job reaches state done
                # whenever its thread returned, and run_task RETURNS a failure
                # message rather than raising, so done alone covers a child that
                # hit max_turns or tripped its circuit breaker. The child records
                # its own verdict in the payload (background.py); the live child
                # object is the fallback for a job that predates it.
                child_ok = result.get("ok")
                if child_ok is None:
                    child_ok = getattr(child, "last_run_ok", True) if child else True
                if state != "done":
                    status = state
                else:
                    status = "ok" if child_ok else "error"

                # A POINTER to the child's work, never a merge into the parent's
                # own change map, for the same reason.
                branch = result.get("branch")
                if branch:
                    from .. import delegated as _delegated
                    _delegated.record(self, _delegated.DelegatedChangeSet(
                        label=label,
                        branch=branch,
                        file_count=result.get("file_count", 0),
                        source="background",
                        status=status,
                        base=result.get("base", ""),
                        diff=result.get("diff", ""),
                    ))

                if st.get("error"):
                    notes.append(
                        f"Background sub-agent '{label}' ({job_id}) FAILED: "
                        f"{st['error']}")
                else:
                    body = result.get("summary") or "(no output)"
                    where = (f" Its changes are committed on branch '{branch}' "
                             f"({result.get('file_count', 0)} file(s)) and are NOT "
                             "in your working tree.") if branch else ""
                    # Say plainly that it did not finish its task. The summary text
                    # states it.
                    verdict = "finished" if child_ok else "DID NOT COMPLETE its task"
                    notes.append(
                        f"Background sub-agent '{label}' ({job_id}) {verdict} after "
                        f"{result.get('turns', 0)} turn(s).{where}\n\n{body}")

                for w in (st.get("warnings") or []):
                    notes.append(f"  (warning from {job_id}: {w})")
                if result.get("cleanup_warning"):
                    notes.append(
                        f"  (cleanup warning from {job_id}: "
                        f"{result['cleanup_warning']})")
            except Exception:
                # One malformed job must not cost the others their absorption.
                continue
        return notes

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
        a pre-existing dirty tree is not misattributed."""
        if not self._shell_baseline_captured or self._git_baseline is None:
            return [], ""
        current = self._git_status_map()
        if current is None:
            return [], ""
        delta = sorted(set(current) - self._git_baseline)
        if not delta:
            return [], ""
        # Only `path` is consumed by the reflection; the other fields match the
        # changed_files() dict shape (created is True for an untracked new file,
        # "??").
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
        root shows progress instead of appearing to hang.

        Tries the cross-session cache first, falling back to a full walk when
        there is nothing usable to reconcile. Either way, the RESULT is persisted
        for the next session: a fresh build() leaves a cache the next session can
        reconcile from, and a reconciled-from-cache map is re-saved too, so this
        session's own edits (captured in-memory via refresh_file()/mark_dirty())
        are what the NEXT session reconciles against, not a stale earlier
        snapshot.

        Makes exactly ONE call into ProjectMap: the cache lookup is folded inside
        build() rather than being a second classmethod call here.

        No-op cache in privacy mode - the cache records this project's file paths
        and extracted symbol names, the same promise save_checkpoint() below makes
        about the conversation. cache_path=None makes build() do a fresh in-memory
        walk with no cache read or write."""
        # Live attribute lookup, so a patched agent.ProjectMap is honoured.
        ProjectMap = _agent.ProjectMap
        import time
        cache_path = None if self.mode == SessionMode.PRIVACY else _project_map_path_for(cwd)
        t0 = time.monotonic()
        pm = ProjectMap.build(cwd, deadline_s=_index_deadline(), cache_path=cache_path)
        took = time.monotonic() - t0
        if pm.truncated or took > 2.0:
            suffix = (" (large project - index truncated; the agent can still "
                      "list_dir / search_files)" if pm.truncated else "")
            source = "from cache" if getattr(pm, "from_cache", False) else "scanned"
            self._emit("info",
                       text=f"Indexed {pm.file_count()} files ({source}) "
                            f"in {took:.1f}s{suffix}")
        return pm

    @property
    def _checkpoint_path(self) -> Path:
        # Session data lives under HOME, not in the project tree. Keyed on THIS
        # agent's own stable checkpoint id, not the project alone - see
        # _checkpoint_path_for.
        return _checkpoint_path_for(self.cwd, self._checkpoint_id)

    @property
    def _legacy_checkpoint_path(self) -> Path:
        return _legacy_checkpoint_path_for(self.cwd)

    @property
    def _legacy_home_checkpoint_path(self) -> Path:
        return _legacy_home_checkpoint_path_for(self.cwd)

    def save_checkpoint(self) -> None:
        """Persist current conversation state so it can be resumed later.

        No-op in privacy mode - the checkpoint contains the full
        conversation, which privacy mode promises never to write to disk.
        The task list rides along in the same file and is therefore covered by
        the same promise: in privacy mode it stays in memory only."""
        if self.mode == SessionMode.PRIVACY:
            return
        data = {
            "version": 1,
            "interrupted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "turns": self._turns,
            "total_tokens": self._total_tokens,
            # A short display title for a resume listing - captured once, from the
            # RAW first task/message text before any episodic-memory preamble is
            # prepended (see _session_title's capture sites in loop.py). Re-derived
            # from the same stored source on every save rather than only at
            # first-save, so an older checkpoint missing this field still gets a
            # real title the next time it is saved.
            "title": _derive_title(self._session_title),
            "messages": self._messages,
            # The model's own task list (tools/tasks.py). An older build ignores
            # the extra key, and an older checkpoint restores as no todos.
            "todos": self.get_todos(),
            # Pointers to work delegated children COMMITTED on their own branches.
            # The branches are durable; this keeps the pointer to them durable too,
            # so a resumed session still shows the footer. The change-sets are
            # small, frozen and JSON-shaped so they can ride along here.
            "delegated": [
                dataclasses.asdict(cs)
                for cs in (getattr(self, "_delegated", None) or [])
            ],
            # The changed-files tracker (GUI "files changed" / session_diff), so a
            # server restart or GUI reconnect mid-session keeps every prior write's
            # record. Each entry's "original" is the pre-change snapshot (bytes, or
            # None for a newly-created file), base64-encoded because JSON has no
            # bytes type.
            "changed_files": {
                key: {
                    "original": (base64.b64encode(entry["original"]).decode("ascii")
                                 if entry.get("original") is not None else None),
                    "writes": entry.get("writes", 1),
                    "last_tool": entry.get("last_tool", "unknown"),
                }
                for key, entry in self._changed_files.items()
            },
        }
        p = self._checkpoint_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(p, json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            # Never let checkpoint failure crash the session (AGENTS.md rule 5:
            # surfaced, not silenced - a resume checkpoint failing to save is
            # exactly the case a user needs to know about).
            logger.warning("save_checkpoint: failed to persist %s", p, exc_info=True)

    def clear_checkpoint(self) -> None:
        """Remove THIS agent's own saved checkpoint - its own per-session id under
        HOME, plus the two legacy shapes (a project can only ever have ONE
        checkpoint under either legacy layout). It does NOT reach any OTHER
        session's per-id checkpoint - see _checkpoint_path_for.

        Best-effort per path: a delete that fails is LOGGED at warning and does
        not raise, so this returning is not by itself proof the checkpoint is
        gone."""
        for p in (self._checkpoint_path, self._legacy_checkpoint_path,
                  self._legacy_home_checkpoint_path):
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                # Non-fatal: this is cleanup riding on OTHER work, so raising would
                # misreport that work.
                #
                # Not silent either: a checkpoint that outlives a clear can be
                # adopted by a later session the user believes is gone. WARNING,
                # not debug, so it reaches the always-on recent-activity ring a bug
                # report dumps (that ring is INFO+).
                from localm.debuglog import logger
                logger.warning(
                    "coder: could not clear checkpoint %s: %s; it remains on "
                    "disk and a later session may resume from it", p, e)

    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> dict | None:
        """
        Read a saved checkpoint for this cwd, or None if none is found.

        *checkpoint_id* is None (the default) to load the MOST RECENT checkpoint,
        the zero-argument continue-where-I-left-off behaviour, unaffected by
        several sessions being able to coexist in the same project. Pass a
        specific id (from list_checkpoints()) to resume a particular one instead
        of the newest.

        On success this sets self._checkpoint_id, so a LATER save_checkpoint()
        writes back to the SAME file the resumed conversation came from, rather
        than minting a fresh one on every save - including when the checkpoint
        came from a legacy location, which is migrated into the per-session layout
        as part of this same load (see migrate_legacy_checkpoint). Explicit-id
        lookups never fall back to a legacy path.
        """
        if checkpoint_id is not None:
            data = _read_checkpoint(_checkpoint_path_for(self.cwd, checkpoint_id))
            if data is not None:
                self._checkpoint_id = checkpoint_id
            return data
        entries = list_checkpoints(self.cwd)
        if entries:
            newest_id = entries[0]["id"]
            data = _read_checkpoint(_checkpoint_path_for(self.cwd, newest_id))
            if data is not None:
                self._checkpoint_id = newest_id
                return data
        for legacy_path in (self._legacy_home_checkpoint_path,
                            self._legacy_checkpoint_path):
            data = _read_checkpoint(legacy_path)
            if data is not None:
                self._checkpoint_id = migrate_legacy_checkpoint(
                    self.cwd, legacy_path, data)
                return data
        return None

    def resume_checkpoint(self, data: dict) -> None:
        """Restore agent state from a checkpoint dict.

        This is the single restore path for every caller (the REPL /resume, the
        CLI resume flag, and the GUI session), so restoring the task list here
        is what makes it survive a pause/resume everywhere. The file is plain
        user-writable JSON, so the todos go back through normalize_todos rather
        than being trusted as-is."""
        from ..tools.tasks import normalize_todos
        from .checkpoint import _first_user_text
        self._messages     = data["messages"]
        self._turns        = data.get("turns", len(self._messages))
        self._total_tokens = data.get("total_tokens", 0)
        # Keep the restored session's own title stable across resume plus a later
        # save: the guard in loop.py's run_task/chat only sets this once per Agent,
        # so a fresh Agent being resumed into needs it seeded here or its NEXT
        # instruction would retitle the whole session. Falls back to re-deriving
        # from the messages for a checkpoint saved before title existed;
        # load_checkpoint()'s migration path already backfills it.
        self._session_title = data.get("title") or _first_user_text(self._messages)
        self.set_todos(normalize_todos(data.get("todos")))
        self._restore_delegated(data.get("delegated"))
        self._restore_changed_files(data.get("changed_files"))

    def _restore_changed_files(self, raw) -> None:
        """Rebuild the changed-files tracker from a checkpoint.

        The file is plain user-writable JSON (same posture as _restore_delegated):
        each entry is filtered to the expected shape and skipped if it does not
        fit, rather than trusted as-is. A checkpoint saved before this field
        existed has no "changed_files" key at all - *raw* is then None and this
        leaves the tracker empty.
        """
        restored: dict[str, dict] = {}
        if isinstance(raw, dict):
            for key, entry in raw.items():
                if not isinstance(key, str) or not isinstance(entry, dict):
                    continue
                original_b64 = entry.get("original")
                try:
                    original = (base64.b64decode(original_b64)
                                if original_b64 is not None else None)
                except Exception:
                    continue   # corrupt encoding: drop this entry, not the resume
                writes = entry.get("writes", 1)
                if not isinstance(writes, int):
                    writes = 1
                last_tool = entry.get("last_tool", "unknown")
                if not isinstance(last_tool, str):
                    last_tool = "unknown"
                restored[key] = {"original": original, "writes": writes,
                                 "last_tool": last_tool}
        self._changed_files = restored

    def _restore_delegated(self, raw) -> None:
        """Rebuild the delegated-work pointers from a checkpoint.

        The file is plain user-writable JSON, so each entry is filtered to the
        dataclass's own fields and skipped if it does not fit, rather than
        splatted in and trusted - the same posture normalize_todos takes.

        NOTE ON IN-FLIGHT JOBS: a background sub-agent itself does NOT survive
        here, and nothing pretends it does. Its thread dies with the
        process, so a resumed session records no job id for it - there is no
        orphaned reference that silently resolves to nothing. What survives is
        what is genuinely durable: the branch its committed work sits on.
        """
        from ..delegated import DelegatedChangeSet
        if not isinstance(raw, list):
            return
        fields = {f.name for f in dataclasses.fields(DelegatedChangeSet)}
        restored = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                restored.append(DelegatedChangeSet(
                    **{k: v for k, v in item.items() if k in fields}))
            except Exception:
                continue          # a malformed entry must not break the resume
        if restored:
            self._delegated = restored

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
        together.

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
        # entry predating call ids).
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
            # A partial undo must never read as a complete one.
            summary += ("\nWARNING: the undo did NOT fully succeed: "
                        + "; ".join(failures)
                        + " - these files still hold the change.")
        return summary
