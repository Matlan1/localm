# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Background jobs for the GUI: model pulls and image generation.

A job wraps a ``localm`` CLI subprocess. Its stdout/stderr lines are pushed
onto a queue that the web layer streams to the browser as SSE. Subprocesses
run without a TTY, so interactive prompts fall back to their non-interactive
defaults.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from localm.model_manager import PROGRESS_SENTINEL

# Bound on the replay backlog and on each subscriber's queue.
_HISTORY_MAX = 10_000

# Every status a job can hold. "interrupted" means the server stopped while the
# operation was in flight, so its outcome is unknown; "failed" means the work
# itself failed; "cancelled" means someone stopped it.
_JOB_STATUSES = ("running", "done", "failed", "cancelled", "interrupted")


def _safe_put(q: asyncio.Queue, event: dict) -> None:
    """Push onto a subscriber's queue, evicting the oldest entry on overflow
    instead of blocking or raising. Runs on the event loop via
    call_soon_threadsafe, so it must not block."""
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


@dataclass
class Job:
    id: str
    kind: str                      # "pull" | "imagine" | ...
    argv: list
    status: str = "running"        # one of _JOB_STATUSES - see that tuple
    returncode: Optional[int] = None
    result: Optional[str] = None   # kind-specific payload (e.g. output image path)
    created_at: float = field(default_factory=time.time)
    # When the worker thread left. None while the job is still running. The TTL
    # sweep keys on this field.
    finished_at: Optional[float] = None
    # Human-readable name for this operation (start_cli's host_label).
    label: Optional[str] = None
    # Stable id (keystore hash) of the key that created this job, or None when no
    # key is configured at all or no token was presented. The events/cancel routes
    # accept the creator or an admin/owner only. In the default configuration
    # principal_id() is None, so jobs are unowned and job_owner_ok() admits any
    # authenticated caller; ownership discriminates in keyed mode only.
    owner: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None
    # Set by cancel(); in-thread jobs (start_fn) poll this to stop cooperatively,
    # having no subprocess to terminate.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Every event ever pushed (bounded, oldest evicted first). Each SSE connection
    # gets its OWN asyncio.Queue in _subscribers, fed live by push().
    _history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_HISTORY_MAX))
    _subscribers: list = field(default_factory=list)
    _sub_lock: threading.Lock = field(default_factory=threading.Lock)
    # The most recent {"type": "progress", ...} event. Written under _sub_lock in
    # push(), alongside the history append it is derived from.
    _last_progress: Optional[dict] = None
    # Set by mark_outcome(). None (never marked) and "failed" both fall through to
    # start_fn's except-branch behavior; only "done" overrides it. Read and written
    # from the single worker thread running this job's callback, so it needs no
    # lock, unlike _history and _subscribers.
    _outcome: Optional[str] = None
    # Called (best-effort, never allowed to raise into the job) whenever this
    # job's DURABLE state changes: a cancel, and the one-time finished stamp. Set
    # by JobManager when it registers the job. Progress does NOT notify.
    _notify: Optional[Callable[[], None]] = None
    # For a job REBUILT from a persisted row (see from_record): the pid its child
    # process had in the previous run, or None. Only ever reported, never
    # signalled. Absent on a live job, which has _proc.
    restored_child_pid: Optional[int] = None

    # ---- durable record -------------------------------------------------- #
    # A row carries only what a listing and an ownership check need: no argv and
    # no progress, so a recovered row reports an unknown percentage.

    def to_record(self) -> dict:
        """This job as a persistable row. Reads plain fields only, never the
        _sub_lock-guarded history/subscribers, so it is safe to call under the
        manager's own lock."""
        pid = self.restored_child_pid
        proc = self._proc
        if proc is not None:
            try:
                pid = int(proc.pid)
            except Exception:
                pass
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "result": self.result,
            "owner": self.owner,
            "child_pid": pid,
        }

    @classmethod
    def from_record(cls, data: dict) -> "Job":
        """Rebuild a job from a persisted row, or raise ValueError on a row this
        build cannot make sense of.

        The result is a RECORD, not a live job: argv is empty and _proc is None.
        created_at is required and must be a real number."""
        jid = data.get("id")
        if not isinstance(jid, str) or not jid.strip():
            raise ValueError("row has no id")
        status = data.get("status")
        if status not in _JOB_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        created = data.get("created_at")
        if not isinstance(created, (int, float)):
            raise ValueError(f"created_at must be a number, got {created!r}")
        finished = data.get("finished_at")
        if finished is not None and not isinstance(finished, (int, float)):
            raise ValueError(f"finished_at must be a number or null, got {finished!r}")
        rc = data.get("returncode")
        if rc is not None and not isinstance(rc, int):
            rc = None
        pid = data.get("child_pid")
        if not isinstance(pid, int) or pid <= 0:
            pid = None
        label = data.get("label")
        result = data.get("result")
        owner = data.get("owner")
        job = cls(
            id=jid,
            kind=str(data.get("kind") or "unknown"),
            argv=[],
            status=status,
            returncode=rc,
            result=result if isinstance(result, str) else None,
            created_at=float(created),
            finished_at=(float(finished) if finished is not None else None),
            label=label if isinstance(label, str) else None,
            owner=owner if isinstance(owner, str) else None,
        )
        job.restored_child_pid = pid
        return job

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def subscriber_count(self) -> int:
        with self._sub_lock:
            return len(self._subscribers)

    def push(self, event: dict) -> None:
        with self._sub_lock:
            self._history.append(event)
            if event.get("type") == "progress":
                self._last_progress = event
            subs = list(self._subscribers)
        for q, loop in subs:
            loop.call_soon_threadsafe(_safe_put, q, event)

    def progress(self, *, phase: Optional[str] = None,
                 done: Optional[int] = None, total: Optional[int] = None,
                 unit: Optional[str] = None, **extra) -> None:
        """Report structured progress from an IN-PROCESS (``start_fn``) job.

        ``pct`` is derived here, computed exactly as
        ``model_manager.pull._emit_progress`` does, and is **null whenever there
        is no total**. Pass ``done`` without ``total`` for an indeterminate
        count.

        ``unit`` names what ``done``/``total`` are counted in ("bytes",
        "files", "chunks"). ``done``, ``total`` and ``unit`` are OMITTED when
        unknown, never sent as zero.
        """
        pct = None
        if total and done is not None:
            pct = round(done * 100 / total, 1)
            if done > total:
                # Not clamped: an over-100% value is logged and left visible
                # upstream.
                try:
                    from localm.debuglog import logger
                    logger.debug("job %s reported %s of %s %s (over 100%%)",
                                 self.id, done, total, unit or "units")
                except Exception:
                    pass
        event: dict = {"type": "progress", "pct": pct}
        if phase:
            event["phase"] = phase
        if done is not None:
            event["done"] = done
        if total:
            event["total"] = total
        if unit:
            event["unit"] = unit
        event.update(extra)
        self.push(event)

    def subscribe(self) -> asyncio.Queue:
        """Register an independent event stream for one SSE connection: an
        asyncio.Queue pre-loaded with every event pushed so far, plus every
        event push() fans out from here on. Call unsubscribe() when the
        connection closes, or the subscriber leaks for the job's lifetime."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=_HISTORY_MAX)
        with self._sub_lock:
            backlog = list(self._history)
            self._subscribers.append((q, loop))
        for event in backlog:
            q.put_nowait(event)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers[:] = [(sq, sl) for sq, sl in self._subscribers if sq is not q]

    def cancel(self) -> None:
        """Request cancellation. Sets the cooperative flag (polled by in-thread
        jobs like media generation) and terminates the subprocess when there is
        one (CLI jobs like model pulls)."""
        if self.status == "running":
            self.status = "cancelled"
        self.cancel_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        self._fire_notify()

    def mark_finished(self) -> None:
        """Stamp finished_at once, when the worker thread leaves. Idempotent: a
        second call cannot move the timestamp forward."""
        if self.finished_at is None:
            self.finished_at = time.time()
            # Inside the idempotence guard: the notify persists the terminal state
            # once.
            self._fire_notify()

    def _fire_notify(self) -> None:
        """Tell the owning manager this job's durable state moved. Best-effort: a
        store problem never propagates into the job's own control flow, and the
        failure is logged."""
        cb = self._notify
        if cb is None:
            return
        try:
            cb()
        except Exception:
            try:
                from localm.debuglog import logger
                logger.debug("job %s state notify failed", self.id, exc_info=True)
            except Exception:
                pass

    def mark_outcome(self, status: str) -> None:
        """Record that THIS job's own real work has verifiably finished with
        *status*, for a ``start_fn`` (in-process) callback to call once its
        actual deliverable is complete and BEFORE any risky tail cleanup (a
        VRAM handover, a bookkeeping call) that could still raise.

        start_fn's worker trusts this over inferring the outcome from an
        exception, but ONLY for an exception raised AFTER this was called. A
        callback that never calls it keeps the default rule: an exception
        anywhere in ``fn`` means failed.

        Raises ValueError unless *status* is "done" or "failed"."""
        if status not in ("done", "failed"):
            raise ValueError(
                f"mark_outcome status must be 'done' or 'failed', got {status!r}")
        self._outcome = status

    def summary(self) -> dict:
        """This job as a listing row: enough for a client to render and then
        attach to it. Does not carry argv or the owner id.

        pct/phase come from the last progress event. Both are absent, not zero,
        when the job has not reported progress."""
        with self._sub_lock:
            progress = self._last_progress
        out = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cancellable": self.status == "running",
        }
        if progress:
            pct = progress.get("pct")
            if isinstance(pct, (int, float)):
                out["pct"] = pct
            phase = progress.get("phase")
            if phase:
                out["phase"] = phase
        return out


class _HostAnnouncer:
    """Mirror a background job's lifecycle to the HOST - the person running
    ``localm gui`` - not only to the requesting client.

    The host sees a start line, throttled progress (10% steps) and the end
    status. Output is ephemeral: host stdout plus the debug log, never a
    privacy-mode disk trace. ``line()`` is pure."""

    # How many raw output lines to keep, so a failure can log its tail.
    _TAIL_LINES = 20

    def __init__(self, label: str) -> None:
        self.label = label
        self._last_bucket = -1
        self._recent_lines: collections.deque = collections.deque(maxlen=self._TAIL_LINES)

    def line(self, event: dict) -> Optional[str]:
        """The host message for a job event, or None to stay quiet (non-progress
        lines, or a progress tick still inside the last 10% bucket)."""
        et = event.get("type")
        if et == "progress":
            pct = event.get("pct")
            if not isinstance(pct, (int, float)):
                return None
            bucket = int(pct // 10) * 10
            if bucket <= self._last_bucket:
                return None
            self._last_bucket = bucket
            return f"{self.label}: {bucket}%"
        if et == "end":
            return f"{self.label} {event.get('status', 'finished')}"
        return None

    def _say(self, msg: str) -> None:
        try:
            from localm.debuglog import logger
            logger.info(msg)
        except Exception:
            pass
        try:
            # The GUI server's stdout is the host terminal.
            print(msg, flush=True)
        except Exception:
            pass

    def announce_start(self) -> None:
        self._say(f"{self.label} started (requested from a connected client)")

    def record_line(self, text: str) -> None:
        """Buffer a raw output line (not a progress/end event) for a later
        failure to log."""
        self._recent_lines.append(text)

    def announce_failure_detail(self) -> None:
        """Log the job's last few buffered output lines at ERROR level. No-op if
        nothing was buffered."""
        if not self._recent_lines:
            return
        try:
            from localm.debuglog import logger
            logger.error("%s failed - last output:\n%s", self.label,
                        "\n".join(self._recent_lines))
        except Exception:
            pass

    def emit(self, event: dict) -> None:
        msg = self.line(event)
        if msg is not None:
            self._say(msg)


# --------------------------------------------------------------------------- #
#  Durable record of in-flight operations                                     #
# --------------------------------------------------------------------------- #
# ONE FILE PER WRITER, named <pid>-<run>.json under <data dir>/activity/. The pid
# half is what a later run reads to decide whether the writer is GONE; the run
# half is a nonce minted per store instance, so two managers alive in one process
# never write the same path. Liveness is checked against the PID, not the
# instance_id; pid reuse is handled by the run nonce plus the in-process
# registry, see _ActivityStore.reap.
#
# Posture: atomic temp+replace, owner-restricted, corrupt-file quarantine, and a
# per-row skip on a bad entry. This process only ever WRITES its own file, so
# several localm servers can share one data dir.
#
# The row holds no session content, no prompts and no argv, and the TTL sweep
# drops it an hour after the operation finishes. It is written in every session
# mode, privacy included.
_ACTIVITY_VERSION = 1

# Serialises every activity-store operation IN THIS PROCESS. Lock order in this
# module is always JobManager._lock OUTER, _STORE_LOCK INNER; nothing here ever
# reaches back for the manager's lock.
_STORE_LOCK = threading.RLock()

# How many corrupt-copies to keep across the whole activity DIRECTORY, not per
# file name.
_QUARANTINE_KEEP = 3

# Matches the `owner` field's sha256 digest SHAPE (64 hex) in raw text, so a
# mangled value is left alone.
_OWNER_DIGEST_RE = re.compile(r'("owner"\s*:\s*)"[0-9a-fA-F]{64}"')

# What the digest is replaced WITH: a non-null, non-hex sentinel. The field is
# neither dropped nor nulled, because owner=None means unowned and therefore
# unrestricted. It can never equal a principal_id(), so a recovered row resolves
# to NOBODY and fails CLOSED: unreachable to a scoped key, still reachable to an
# admin/owner key.
_REDACTED_OWNER = "redacted-on-quarantine"


def activity_dir() -> Path:
    """The in-flight operation record dir (``<data dir>/activity``), resolved at
    CALL time so a repointed LOCALM_HOME is honoured. Not ``<data dir>/run``,
    where the instance reaper globs ``*.json`` and would try to parse these as
    instance entries."""
    from localm.config import home_dir
    return (home_dir() / "activity").resolve()


def _redact_owner_digests(raw: str) -> str:
    """Strip owner key digests out of a corrupt record file before it is copied
    aside, keeping everything else.

    Warns when a digest-shaped value survives the substitution."""
    if not raw:
        return raw
    out, n = _OWNER_DIGEST_RE.subn(rf'\1"{_REDACTED_OWNER}"', raw)
    if re.search(r"\b[0-9a-fA-F]{64}\b", out):
        try:
            from localm.debuglog import logger
            logger.warning(
                "activity store: quarantine copy still holds a digest-shaped "
                "value after redacting %d owner field(s); the file is corrupt, "
                "so it may carry one in a form this cannot match", n)
        except Exception:
            pass
    return out


class _ActivityStore:
    """Reads and writes this process's in-flight operation record.

    Every method is BEST-EFFORT: ``write`` returns False instead of raising, and
    ``reap`` returns whatever it could read. Each failure logs."""

    def __init__(self, root: Optional[Path] = None, *,
                 pid: Optional[int] = None, run: Optional[str] = None) -> None:
        self._root_override = Path(root).resolve() if root is not None else None
        self._pid = int(pid) if pid is not None else os.getpid()
        self._run = str(run) if run else uuid.uuid4().hex[:12]

    @property
    def root(self) -> Path:
        return self._root_override if self._root_override is not None else activity_dir()

    @property
    def path(self) -> Path:
        return self.root / f"{self._pid}-{self._run}.json"

    @property
    def pid(self) -> int:
        return self._pid

    # ---- write ----------------------------------------------------------- #
    def write(self, rows: list) -> bool:
        """Persist *rows* atomically, owner-restricted. Returns success."""
        with _STORE_LOCK:
            return self._write(rows)

    def _write(self, rows: list) -> bool:
        try:
            d = self.root
            d.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(d, 0o700)
            except OSError:
                pass          # POSIX only; the file ACL is set below
            payload = {"version": _ACTIVITY_VERSION, "pid": self._pid,
                       "operations": rows}
            # Writes owner-only from the moment the bytes first exist, on Windows
            # as well as POSIX. Its FIXED "<name>.tmp" temp name requires the
            # destination path to stay unique per (process, store), with
            # _STORE_LOCK serialising every writer in this process.
            from localm.config import atomic_write_private
            atomic_write_private(
                self.path, json.dumps(payload, indent=2, ensure_ascii=False))
            return True
        except Exception as e:
            try:
                from localm.debuglog import logger
                logger.warning(
                    "activity store: could not persist the in-flight operation "
                    "record to %s (%s); a restart will not be able to report "
                    "what was running", self.path, e)
            except Exception:
                pass
            return False

    def clear(self) -> None:
        """Remove this process's record file, for when there is nothing left to
        record."""
        with _STORE_LOCK:
            self._clear()

    def _clear(self) -> None:
        # Broad, like write(): no caller's control flow changes because of a store
        # problem, and even computing the path can fail (home_dir() is lazy).
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            try:
                from localm.debuglog import logger
                logger.debug("activity store: could not remove the record file (%s)", e)
            except Exception:
                pass

    # ---- read + reconcile ------------------------------------------------- #
    def reap(self, *, skip=None) -> list:
        """Adopt every record file left by a writer that is GONE, and return their
        rows. An adopted file is deleted, so its rows end up in exactly one live
        process's record.

        *skip* is the set of paths owned by managers alive IN THIS PROCESS
        (including the caller's own). Pid liveness cannot answer the same-process
        question, so only the in-process registry tells our own live record apart
        from a leftover of a process that held this pid before us.

        A live foreign writer's file is left completely alone: not read, not
        adopted, not deleted."""
        with _STORE_LOCK:
            return self._reap(skip=skip)

    def _reap(self, *, skip=None) -> list:
        skipped = {Path(s) for s in (skip or ())}
        try:
            files = sorted(self.root.glob("*.json"))
        except OSError as e:
            try:
                from localm.debuglog import logger
                logger.debug("activity store: could not list %s (%s)",
                             self.root, e)
            except Exception:
                pass
            return []
        out: list = []
        for f in files:
            if f in skipped:
                continue
            pid = self._pid_of(f)
            if pid is not None and pid != self._pid:
                from localm.instances import pid_alive
                if pid_alive(pid):
                    continue      # another live server owns it
            rows = self._read(f)
            if rows is None:
                continue          # unreadable/corrupt: already reported by _read
            out.extend(rows)
            try:
                f.unlink()
            except OSError as e:
                # A file left behind can be adopted again by a third process, so
                # its rows may be reported twice.
                try:
                    from localm.debuglog import logger
                    logger.warning(
                        "activity store: adopted the operation records in %s but "
                        "could not remove it (%s); they may be reported twice",
                        f.name, e)
                except Exception:
                    pass
        return out

    @staticmethod
    def _pid_of(path: Path) -> Optional[int]:
        """The writer pid a record filename encodes, or None when the name does
        not follow the scheme.

        Only the pid half is parsed. The run nonce after it is never compared to
        anything; _live_store_paths identifies our own live files."""
        try:
            return int(path.stem.split("-", 1)[0])
        except (TypeError, ValueError, IndexError):
            return None

    def _read(self, path: Path) -> Optional[list]:
        """The rows in *path*, or None when it could not be read at all."""
        raw = None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except OSError as e:
            # Reported, then left in place for a later run.
            try:
                from localm.debuglog import logger
                logger.warning("activity store: %s is unreadable (%s); the "
                               "operations it records cannot be recovered",
                               path.name, e)
            except Exception:
                pass
            return None
        except json.JSONDecodeError as e:
            self._quarantine(path, raw, e)
            return None
        ops = data.get("operations") if isinstance(data, dict) else None
        if not isinstance(ops, list):
            self._quarantine(path, raw, "no operations list")
            return None
        return [r for r in ops if isinstance(r, dict)]

    def _quarantine(self, path: Path, raw, err) -> None:
        """Copy a corrupt record file aside before anything removes it, and warn.
        The copy is redacted (_redact_owner_digests) and the set of copies is
        pruned (_prune_quarantine)."""
        try:
            from localm.debuglog import logger
        except Exception:
            logger = None
        try:
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            if raw is not None:
                backup.write_text(_redact_owner_digests(raw), encoding="utf-8")
                # Restricted the way the live file is, check-and-retry like
                # atomic_write_private.
                from localm.config import restrict_file_perms
                if not restrict_file_perms(backup):
                    restrict_file_perms(backup)
            if logger is not None:
                logger.warning(
                    "activity store: %s is corrupt (%s); backed up to %s. The "
                    "operations it recorded cannot be reported, but the file is "
                    "preserved rather than lost", path.name, err, backup.name)
            self._prune_quarantine()
        except OSError as e:
            if logger is not None:
                logger.warning("activity store: corrupt record file %s could not "
                               "be backed up (%s); its content may be lost",
                               path.name, e)

    def _prune_quarantine(self) -> None:
        """Keep only the newest _QUARANTINE_KEEP corrupt-copies in the dir.

        Bounded across the whole DIRECTORY, not per file name. Sorted by the
        timestamp SUFFIX, not lexically.

        Best-effort: failing to delete an old backup never breaks the recovery
        path that just wrote a new one."""
        try:
            from localm.debuglog import logger
        except Exception:
            logger = None
        try:
            backups = list(self.root.glob("*.json.corrupt-*"))
        except OSError as e:
            if logger is not None:
                logger.debug(
                    "activity store: could not list quarantine copies (%s)", e)
            return

        def _stamp(pth: Path) -> int:
            try:
                return int(pth.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                return 0

        backups.sort(key=lambda b: (_stamp(b), b.name))
        for old in (backups[:-_QUARANTINE_KEEP] if _QUARANTINE_KEEP else backups):
            try:
                old.unlink()
            except OSError as e:
                if logger is not None:
                    logger.debug("activity store: could not prune %s (%s)",
                                 old.name, e)


# --------------------------------------------------------------------------- #
#  Child-process termination on process exit                                  #
# --------------------------------------------------------------------------- #
# Every live JobManager in this process, reachable at module scope. A WeakSet, so
# a manager built by a test (or by an app that is torn down) is not kept alive
# here.
_MANAGERS: "weakref.WeakSet" = weakref.WeakSet()

def _live_store_paths() -> set:
    """The record-file paths owned by managers that are ALIVE in this process,
    which is what reconciliation needs (see _ActivityStore.reap). Reads the
    private _store attribute of sibling managers."""
    out = set()
    for manager in list(_MANAGERS):
        try:
            out.add(manager._store.path)
        except Exception:
            continue
    return out


# How long a child gets to exit after a graceful signal before it is killed.
_CHILD_GRACE_S = 3.0


def _terminate_process_tree(proc, *, grace: float = _CHILD_GRACE_S) -> None:
    """Terminate *proc* and, where the platform allows it, its descendants.

    A start_cli child is ``python -m localm <cmd>``, and several of those commands
    spawn their OWN children (comfy setup runs git and pip).

    The tree half is BEST-EFFORT:

    * Windows uses ``taskkill /F /T``, which walks the child tree.
    * POSIX walks the tree with psutil, which is an OPTIONAL dependency. Without
      it only the direct child is reached, and that limit is LOGGED.

    Sets no start_new_session/creationflags on the Popen itself, so signal
    delivery for existing jobs is unchanged."""
    try:
        from localm.debuglog import logger
    except Exception:
        logger = None
    try:
        pid = int(proc.pid)
    except Exception:
        return
    if sys.platform == "win32":
        try:
            done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                  capture_output=True, timeout=10)
            if done.returncode == 0:
                return
            if logger is not None:
                # taskkill reports its ordinary failures by exit code, not by
                # raising. Falls through to the handle fallback below.
                logger.debug("taskkill exited %s for job child %s; falling back "
                             "to the process handle", done.returncode, pid)
        except (OSError, subprocess.SubprocessError) as e:
            if logger is not None:
                logger.debug("taskkill unavailable for job child %s (%s); "
                             "falling back to the process handle", pid, e)
        _kill_handle(proc)
        return

    kids = []
    try:
        import psutil
        kids = psutil.Process(pid).children(recursive=True)
    except ImportError:
        if logger is not None:
            logger.debug("psutil is not installed, so only the direct child of "
                         "job pid %s is signalled; any grandchildren it spawned "
                         "are left running", pid)
    except Exception as e:
        if logger is not None:
            logger.debug("could not walk the child tree of job pid %s (%s); "
                         "signalling the direct child only", pid, e)
    for kid in kids:
        try:
            kid.terminate()
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=grace)
    except Exception:
        _kill_handle(proc)
    for kid in kids:
        try:
            if kid.is_running():
                kid.kill()
        except Exception:
            pass


def _kill_handle(proc) -> None:
    """Force-kill via the Popen handle, which is bound to the launched object and
    can never hit a recycled pid."""
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def terminate_children_for_exit(*, grace: float = _CHILD_GRACE_S) -> int:
    """Terminate every in-flight job child process in this process, and return how
    many were signalled.

    Called from http_server._do_shutdown and _do_restart. Both end in a call that
    bypasses atexit (os._exit / os.execv), the job worker threads are daemons so
    their ``finally`` may never run, and the Popen carries no creationflags.

    Termination is unconditional, so every child ends in one known state, which
    the next start reports as "interrupted"."""
    total = 0
    for manager in list(_MANAGERS):
        try:
            total += manager.terminate_children_for_exit(grace=grace)
        except Exception:
            try:
                from localm.debuglog import logger
                logger.debug("terminating job children failed for one manager",
                             exc_info=True)
            except Exception:
                pass
    return total


class JobManager:
    """Registry of background jobs. Finished jobs stay queryable for an hour.

    The registry is the in-memory dict below - what SSE subscribers, progress and
    cancellation all attach to - with a small record of each operation's LIFECYCLE
    mirrored to ``<data dir>/activity/<pid>-<run>.json``. Progress is not
    mirrored.

    On construction it RECONCILES: records left by a writer process that is gone
    are adopted, and any of them still marked ``running`` become ``interrupted``
    (see _JOB_STATUSES)."""

    _TTL_S = 3600

    def __init__(self, *, store: Optional["_ActivityStore"] = None,
                 reconcile: bool = True) -> None:
        self._jobs: dict[str, Job] = {}
        # An RLock: persisting happens while holding it, and the persist path is
        # reached both directly and through Job._notify. Lock order in this module
        # is _lock OUTER, Job._sub_lock INNER.
        self._lock = threading.RLock()
        self._store = store if store is not None else _ActivityStore()
        _MANAGERS.add(self)
        if reconcile:
            self._reconcile_from_disk()

    # ---- durability ------------------------------------------------------- #
    def _register(self, job: Job) -> None:
        """Track *job*, sweep expired records, and persist - the shared tail of
        start_cli and start_fn."""
        job._notify = self._persist
        with self._lock:
            # _gc() runs here and nowhere else, so this one write also carries any
            # eviction it just made.
            self._gc()
            self._jobs[job.id] = job
            self._persist_locked()

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        """Mirror the registry to disk. Caller holds _lock.

        Best-effort: _ActivityStore.write reports its own failures and returns
        False instead of raising. An EMPTY registry removes the file instead of
        writing an empty one."""
        rows = [j.to_record() for j in self._jobs.values()]
        if rows:
            self._store.write(rows)
        else:
            self._store.clear()

    def _reconcile_from_disk(self) -> None:
        """Adopt the operation records of writer processes that are gone.

        A row still marked ``running`` is reported as ``interrupted``: this server
        stopped while it was in flight, so its outcome is unknown. It is never
        reported as ``failed``.

        ``finished_at`` is stamped at DETECTION, never derived from the file's
        mtime: _gc() sweeps on finished_at, and a pre-crash timestamp would put
        the row past the TTL cutoff the instant it was recovered."""
        try:
            rows = self._store.reap(skip=_live_store_paths())
        except Exception:
            try:
                from localm.debuglog import logger
                logger.warning("activity store: reconciliation failed; this "
                               "server cannot report what a previous run was "
                               "doing", exc_info=True)
            except Exception:
                pass
            return
        if not rows:
            return
        now = time.time()
        recovered = []
        interrupted = 0
        skipped = 0
        orphan_pids = []
        for row in rows:
            try:
                job = Job.from_record(row)
            except (ValueError, TypeError):
                skipped += 1      # skip a corrupt row rather than fail the load
                continue
            if job.status == "running":
                job.status = "interrupted"
                job.finished_at = now
                interrupted += 1
                if job.restored_child_pid is not None:
                    try:
                        from localm.instances import pid_alive
                        if pid_alive(job.restored_child_pid):
                            orphan_pids.append(job.restored_child_pid)
                    except Exception:
                        pass
            elif job.finished_at is None:
                # A terminal row with no stamp cannot be swept: give it the same
                # detection stamp.
                job.finished_at = now
            self._seed_recovered_history(job)
            recovered.append(job)
        with self._lock:
            for job in recovered:
                job._notify = self._persist
                self._jobs[job.id] = job
            self._gc()
            self._persist_locked()
        try:
            from localm.debuglog import logger
            logger.info(
                "recovered %d operation record(s) from a previous run "
                "(%d interrupted)", len(recovered), interrupted)
            if skipped:
                logger.warning("activity store: skipped %d unreadable operation "
                               "record(s)", skipped)
            if orphan_pids:
                # An orphan is neither adopted nor killed, only reported.
                logger.warning(
                    "a previous run left child process(es) %s still running; they "
                    "are no longer tracked and their output goes nowhere",
                    ", ".join(str(p) for p in orphan_pids))
        except Exception:
            pass

    @staticmethod
    def _seed_recovered_history(job: Job) -> None:
        """Give a recovered job a terminal event stream, so a client that
        reattaches over ``GET /api/jobs/{id}/events`` gets a history and an
        ``end`` frame instead of keepalives forever."""
        if job.status == "interrupted":
            job.push({"type": "line",
                      "text": "The server stopped while this operation was in "
                              "flight, so its outcome is unknown. Its output was "
                              "not kept."})
        else:
            job.push({"type": "line",
                      "text": "Recovered after a server restart. The live output "
                              "of this operation was not kept."})
        job.push({"type": "end", "status": job.status,
                  "returncode": job.returncode, "result": job.result})

    # ---- exit ------------------------------------------------------------- #
    def terminate_children_for_exit(self, *, grace: float = _CHILD_GRACE_S) -> int:
        """Terminate the child process of every still-running start_cli job, and
        return how many were signalled.

        Enumerates under the lock and kills OUTSIDE it.

        Does NOT go through Job.cancel(). The registry is left saying ``running``,
        which is what makes the next start reconcile these rows to
        ``interrupted``."""
        with self._lock:
            live = [(j.id, j._proc) for j in self._jobs.values()
                    if j.status == "running" and j._proc is not None]
        count = 0
        for job_id, proc in live:
            try:
                if proc.poll() is not None:
                    continue          # already exited on its own
            except Exception:
                continue
            try:
                from localm.debuglog import logger
                logger.debug("terminating job %s child pid %s on exit",
                             job_id, getattr(proc, "pid", "?"))
            except Exception:
                pass
            _terminate_process_tree(proc, grace=grace)
            count += 1
        return count

    def start_cli(self, kind: str, cli_args: list, *,
                  result_path: str | None = None,
                  extra_env: dict | None = None,
                  host_label: str | None = None,
                  owner: str | None = None) -> Job:
        """
        Run ``python -m localm <cli_args>`` as a job.

        result_path, when given, is stored on the job as the expected output
        artifact (e.g. the image file an imagine job writes). extra_env adds
        environment variables for the subprocess (e.g. progress reporting).
        host_label, when given, mirrors the job's start/progress/end to the host
        console + debug log. owner, when given, is the creating key's principal
        id - only that key (or an admin/owner) may later stream or cancel the job.
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            argv=[sys.executable, "-X", "utf8", "-m", "localm", *cli_args],
            result=result_path,
            owner=owner,
            label=host_label,
        )
        self._register(job)

        def _run():
            announcer = _HostAnnouncer(host_label) if host_label else None
            if announcer:
                announcer.announce_start()
            # An explicit {"type": "outcome"} sentinel frame overrides the
            # exit-code guess below. None means no such frame arrived, and the
            # exit-code rule then applies.
            reported_outcome = None
            try:
                env = None
                if extra_env:
                    env = os.environ.copy()
                    env.update(extra_env)
                job._proc = subprocess.Popen(
                    job.argv,
                    stdin=subprocess.DEVNULL,   # no inherited TTY, so interactive
                    # dedup/overwrite prompts take their non-interactive default.
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
                # Re-persist now that the child EXISTS, so the record carries its
                # pid; the create-time write ran ahead of the Popen.
                self._persist()
                for line in job._proc.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    # Structured download-progress lines → progress events;
                    # everything else streams verbatim as a log line.
                    if PROGRESS_SENTINEL in line:
                        _, _, payload = line.partition(PROGRESS_SENTINEL)
                        try:
                            data = json.loads(payload)
                        except ValueError:
                            continue
                        # Popped, not merely read: **data below must never carry
                        # its own "type" key.
                        etype = data.pop("type", "progress")
                        if etype == "outcome":
                            # An internal producer -> job-runner signal, never
                            # forwarded to subscribers; it only corrects the
                            # status decision below.
                            status = data.get("status")
                            if status in ("done", "failed"):
                                reported_outcome = status
                            continue
                        job.push({"type": "progress", **data})
                        if announcer:
                            announcer.emit({"type": "progress", **data})
                        continue
                    job.push({"type": "line", "text": line})
                    if announcer:
                        announcer.record_line(line)
                job._proc.wait()
                job.returncode = job._proc.returncode
                if job.status != "cancelled":
                    if reported_outcome is not None:
                        job.status = reported_outcome
                    else:
                        job.status = "done" if job.returncode == 0 else "failed"
            except Exception as e:
                job.status = "failed"
                job.push({"type": "line", "text": f"job error: {e}"})
                if announcer:
                    announcer.record_line(f"job error: {e}")
            finally:
                # Stamp BEFORE the end event goes out, so a subscriber that lists
                # jobs on "end" never sees this one still in flight.
                job.mark_finished()
                if announcer:
                    if job.status == "failed":
                        announcer.announce_failure_detail()
                    announcer.emit({"type": "end", "status": job.status})
                job.push({
                    "type": "end",
                    "status": job.status,
                    "returncode": job.returncode,
                    "result": job.result,
                })

        threading.Thread(target=_run, daemon=True).start()
        return job

    def start_fn(self, kind: str, fn, *, result_path: str | None = None,
                 owner: str | None = None, label: str | None = None) -> Job:
        """
        Run a Python callable as a job in a worker thread.

        ``fn`` receives the job and should return True on success. It may call
        ``job.push({"type": "line", ...})`` for log output, ``job.progress(...)``
        for a structured percentage or count that a listing can render without
        replaying the stream, and may update ``job.result``. A fraction
        formatted into a line is invisible to ``/api/activity``, to the CLI and
        to MCP. If ``fn``'s own real work is done but it still runs risky tail
        cleanup afterward (a VRAM handover, a bookkeeping call), call
        ``job.mark_outcome("done")`` first, so a later exception in the tail
        cannot misreport a completed operation as failed. owner, when given,
        binds the job to the creating key's principal id so only that key (or an
        admin/owner) may stream/cancel it. label, when given, is the
        human-readable operation name a listing shows; there is no console
        mirroring for in-thread jobs.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=[], result=result_path,
                  owner=owner, label=label)
        self._register(job)

        def _run():
            try:
                ok = fn(job)
                if job.status != "cancelled":
                    job.status = "done" if ok else "failed"
            except Exception as e:
                if job._outcome == "done":
                    # mark_outcome("done") was called ahead of the exception, so
                    # the mark wins. None and "failed" both fall through to the
                    # else branch below.
                    job.status = "done"
                    job.push({"type": "line",
                              "text": f"(cleanup after success failed: {e})"})
                else:
                    job.status = "failed"
                    job.push({"type": "line", "text": f"job error: {e}"})
            finally:
                # Stamp before "end" is emitted, as in start_cli.
                job.mark_finished()
                job.push({
                    "type": "end",
                    "status": job.status,
                    "returncode": job.returncode,
                    "result": job.result,
                })

        threading.Thread(target=_run, daemon=True).start()
        return job

    def snapshot(self, visible=None) -> list:
        """Every tracked job as a listing row, newest first.

        *visible*, when given, is called with each job's ``owner`` and decides
        whether that job appears. It takes a PREDICATE, not a principal id, so
        the owner never leaves this class and ``summary()`` never carries it.

        This is the ONLY way to learn a job exists without already holding its
        id.

        Returns summaries, never Job objects: callers must not reach into the
        live object's history/subscribers outside the lock. Ownership is NOT
        filtered here - that is the caller's job, because it needs the request
        to answer it (see job_owner_ok), and in open mode there are no owners to
        filter on at all.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        if visible is not None:
            jobs = [j for j in jobs if visible(j.owner)]
        return sorted((j.summary() for j in jobs),
                      key=lambda s: s["created_at"], reverse=True)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def has_running(self, kind: str) -> bool:
        """Whether a job of *kind* is currently running, so a caller can tell
        "still actively installing" apart from "abandoned mid-install" without
        its own job-tracking state."""
        with self._lock:
            return any(j.kind == kind and j.status == "running"
                      for j in self._jobs.values())

    def _gc(self) -> None:
        """Drop jobs that finished more than _TTL_S ago.

        Keyed on finished_at, NOT created_at: a job that RAN for longer than the
        TTL would otherwise be past the cutoff the moment it finished.

        A still-running job is never swept at any age. created_at travels in the
        summary, so a client can tell a six-second job from a six-hour one.
        """
        cutoff = time.time() - self._TTL_S
        stale = [
            jid for jid, j in self._jobs.items()
            if j.status != "running"
            and (j.finished_at if j.finished_at is not None else j.created_at) < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]
