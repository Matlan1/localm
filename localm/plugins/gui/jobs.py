# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Background jobs for the GUI: model pulls and image generation.

A job wraps a ``localm`` CLI subprocess. Its stdout/stderr lines are pushed
onto a queue that the web layer streams to the browser as SSE, so the GUI
reuses the exact CLI logic (progress bars, dedup checks, split GGUF handling)
without duplicating any of it. Subprocesses run without a TTY, which makes
all interactive prompts fall back to their safe non-interactive defaults.
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

# Bound on both the replay backlog and each subscriber's queue, so a very
# long-lived job cannot grow unbounded.
_HISTORY_MAX = 10_000

# Every status a job can hold. "interrupted" means THIS SERVER STOPPED while the
# operation was in flight, so its outcome is unknown - distinct from "failed"
# (the work itself failed) and "cancelled" (someone stopped it).
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
    # When the worker thread left, i.e. when this job stopped being in flight.
    # None while it is still running. Separate from created_at: the TTL sweep
    # keys on this field, so a long job that finished a second ago survives.
    finished_at: Optional[float] = None
    # Human-readable name for this operation, in the same words the host console
    # uses (start_cli's host_label).
    label: Optional[str] = None
    # Stable id (keystore hash) of the key that created this job, or None when no
    # key is configured at all or no token was presented. The events/cancel routes
    # accept the creator or an admin/owner only.
    #
    # principal_id() returns None whenever no owner key and no keystore are
    # configured, and the loopback GUI's shell token is neither of those, so in
    # the DEFAULT configuration jobs are UNOWNED and job_owner_ok() then admits
    # any authenticated caller. Ownership discriminates in KEYED mode only, so
    # anything built on this field must be correct in both modes.
    owner: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None
    # Set by cancel(); in-thread jobs (start_fn, e.g. media gen) poll this to stop
    # cooperatively, having no subprocess to terminate.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Every event ever pushed (bounded, oldest evicted first), so a viewer that
    # subscribes mid-job replays the full stream from the start. Each SSE
    # connection gets its OWN asyncio.Queue in _subscribers, fed live by push().
    _history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_HISTORY_MAX))
    _subscribers: list = field(default_factory=list)
    _sub_lock: threading.Lock = field(default_factory=threading.Lock)
    # The most recent {"type": "progress", ...} event, so a listing can report
    # pct/phase without replaying the whole history. Written under _sub_lock in
    # push(), alongside the history append it is derived from.
    _last_progress: Optional[dict] = None
    # Set by mark_outcome(). None (never marked) and "failed" both fall through to
    # start_fn's except-branch behavior; only "done" overrides it. Read and written
    # from the single worker thread running this job's callback, same as
    # status/result/returncode above - no lock needed, unlike _history and
    # _subscribers, which SSE subscribers also read cross-thread.
    _outcome: Optional[str] = None
    # Called (best-effort, never allowed to raise into the job) whenever this
    # job's DURABLE state changes: a cancel, and the one-time finished stamp. Set
    # by JobManager when it registers the job, so Job needs no knowledge of the
    # store. Progress does NOT notify, so a per-second tick never rewrites the
    # whole store file.
    _notify: Optional[Callable[[], None]] = None
    # For a job REBUILT from a persisted row (see from_record): the pid its child
    # process had in the previous run, or None. Used only to report that an orphan
    # may still be running, never to signal it - that pid may have been recycled.
    # Absent on a live job, which has _proc.
    restored_child_pid: Optional[int] = None

    # ---- durable record -------------------------------------------------- #
    # A row carries only what a LISTING and an ownership check need. Not argv,
    # which summary() also withholds from clients. Progress is not persisted
    # either, so a recovered row reports an UNKNOWN percentage, not a stale one.

    def to_record(self) -> dict:
        """This job as a persistable row. Reads plain fields only, never the
        _sub_lock-guarded history/subscribers, so it is safe to call under the
        manager's own lock without adding a lock edge."""
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
        build cannot make sense of (the caller skips it and counts it, matching
        the scheduled-jobs store's per-entry posture).

        The result is a RECORD, not a live job: argv is empty and _proc is None,
        the same shape a start_fn job already has. created_at is required and
        must be a real number, because snapshot() sorts on it - a None there
        would raise inside the listing route rather than at load time, which is
        the wrong place to find out."""
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

        Only ``start_cli``'s stdout reader ever PRODUCED a
        ``{"type": "progress"}`` event, keyed on PROGRESS_SENTINEL, so no
        in-thread job ever reported a percentage - not the three RAG kinds, the
        three media kinds, embed-setup or embedding-warmup. ``rag-reembed``
        computes a true ``n/total`` and could only ever print it as prose.

        ``push`` itself was never the obstacle: it is public and latches any
        progress-typed dict, so a hand-rolled one always reached ``summary()``.
        What was missing is this affordance, and that matters because
        ``done * 100 / total`` with no total either raises or - after the guard
        most people reach for - reports a fabricated 0%. Deriving it in ONE
        place is the point; four call sites deriving it independently is four
        chances to reintroduce exactly the defect ADR-0008 removed.

        ``pct`` is derived here rather than accepted from the caller so the two
        producers cannot drift: it is computed exactly as
        ``model_manager.pull._emit_progress`` does, and is **null whenever there
        is no total**. An operation that has not established a denominator is at
        an UNKNOWN percentage, never at 0% (ADR-0008 R1). Pass ``done`` without
        ``total`` for an honest indeterminate count.

        ``unit`` names what ``done``/``total`` are counted in ("bytes",
        "files", "chunks") so a surface can render "412 MB of 1.9 GB" or
        "37 of 128 files" rather than a bare percentage. ``done``, ``total`` and
        ``unit`` are OMITTED when unknown rather than sent as zero.
        """
        pct = None
        if total and done is not None:
            pct = round(done * 100 / total, 1)
            if done > total:
                # Not clamped: an over-100% value is logged here and left visible
                # upstream. Local import matching _HostAnnouncer._say below:
                # debuglog is imported inside the call site throughout this module.
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
        asyncio.Queue pre-loaded with every event pushed so far (so a viewer
        that connects mid-job, or reconnects, still gets the full history) plus
        every event push() fans out from here on. Call unsubscribe() when the
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
        """Stamp finished_at once, when the worker thread leaves. Idempotent so
        a second call (or a cancel that raced the thread's own exit) cannot move
        the timestamp forward and give the job a second lease on the TTL."""
        if self.finished_at is None:
            self.finished_at = time.time()
            # Inside the idempotence guard: the notify persists the terminal state
            # once.
            self._fire_notify()

    def _fire_notify(self) -> None:
        """Tell the owning manager this job's durable state moved. Best-effort by
        contract: persistence is a convenience layered on top of a job, and a
        store problem must never propagate into the job's own control flow
        (rule 5 keeps it visible - the manager's own handler logs it)."""
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
        exception, but ONLY for an exception that happens AFTER this was
        called - a callback that never calls it keeps today's rule unchanged
        (an exception anywhere in ``fn`` means failed). Mirrors start_cli's
        own ``{"type": "outcome"}`` sentinel-frame contract (#1126) for the
        in-process job path, which has no subprocess/stdout boundary to carry
        a sentinel frame across, so the signal has to be an ordinary method
        call on the same in-memory object instead."""
        if status not in ("done", "failed"):
            raise ValueError(
                f"mark_outcome status must be 'done' or 'failed', got {status!r}")
        self._outcome = status

    def summary(self) -> dict:
        """This job as a listing row: enough for a client to render and then
        attach to it, without exposing argv (which carries the resolved model
        spec and any host path the caller passed) or the owner id.

        pct/phase come from the last progress event rather than from history, so
        a caller never has to replay 10,000 events to learn how far along a
        download is. Both are absent, not zero, when the job has not reported
        progress - a pull that has not yet read a byte-count is at an UNKNOWN
        percentage, not at 0%."""
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
    ``localm gui`` - not only to the requesting client (G2).

    A model pull started from a phone/PWA ran silently as far as the host was
    concerned: its output went only to the per-job event queue the browser reads.
    Now the host also sees a start line, throttled progress (10% steps so it does
    not spam), and the end status. Output is ephemeral (host stdout + the debug
    log), never a privacy-mode disk trace - the model spec is operational, not
    session content. ``line()`` is pure so the throttling is unit-testable."""

    # How many raw output lines to keep, so a failure can log its actual tail (a
    # git/pip/native error) rather than only the bare summary.
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
        """Buffer a raw output line (not a progress/end event) so a later
        failure can log its actual tail - see class docstring."""
        self._recent_lines.append(text)

    def announce_failure_detail(self) -> None:
        """Log the job's last few output lines at ERROR level so a debug-log
        digest or a bug report carries the real reason it failed, not just
        the bare "<label> failed" summary. No-op if nothing was buffered."""
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
# (which gui/web.py's fallback can create) never write the same path. Liveness is
# checked against the PID, not the instance_id, because a restart re-execs and
# re-advertises with a fresh instance_id (http_server._do_restart); pid REUSE is
# handled by the run nonce plus the in-process registry, see _ActivityStore.reap.
#
# Posture: atomic temp+replace, owner-restricted, corrupt-file quarantine, and a
# per-row skip on a bad entry. The quarantine helpers below are a local
# implementation: kernel code must not depend on an optional plugin. This process
# only ever WRITES its own file, so a file it could not READ is never a file it is
# about to overwrite, and several localm servers can share one data dir without
# any of them clobbering another's record.
#
# The row holds no session content, no prompts and no argv, and the TTL sweep
# drops it an hour after the operation finishes. It is written in every session
# mode, privacy included.
_ACTIVITY_VERSION = 1

# Serialises every activity-store operation IN THIS PROCESS: two JobManagers can
# coexist in one process, each with its own lock, and reconciliation additionally
# reads and DELETES files a sibling manager may be reaping at the same moment.
# Lock order in this module is always JobManager._lock OUTER, _STORE_LOCK INNER
# (persisting happens under the manager's lock); nothing here ever reaches back
# for the manager's lock.
_STORE_LOCK = threading.RLock()

# How many corrupt-copies to keep across the whole activity dir. Per-pid file
# names mean this prunes across the DIRECTORY, not per file name.
_QUARANTINE_KEEP = 3

# `owner` holds the sha256 of the creating key, the digest the keystore stores
# (http_server.principal_id returns key_hash). A quarantine copy is made from a
# file that FAILED to parse, so the redaction works on raw text. Matches the
# digest SHAPE (64 hex), leaving a mangled value alone rather than replacing it
# blind.
_OWNER_DIGEST_RE = re.compile(r'("owner"\s*:\s*)"[0-9a-fA-F]{64}"')

# What the digest is replaced WITH. job_owner_ok (inference/http_server.py) treats
# owner=None as unowned and therefore unrestricted, so the field is neither
# dropped nor nulled. A non-null, non-hex sentinel can never equal a
# principal_id() (always 64 hex), so a recovered row resolves to NOBODY and fails
# CLOSED: unreachable to a scoped key, still reachable to an admin/owner key.
_REDACTED_OWNER = "redacted-on-quarantine"


def activity_dir() -> Path:
    """The in-flight operation record dir (``<data dir>/activity``), resolved at
    CALL time so a test that repoints LOCALM_HOME is honoured (home_dir() is
    itself lazy). Deliberately NOT ``<data dir>/run``: the instance reaper globs
    ``*.json`` there and would try to parse these as instance entries."""
    from localm.config import home_dir
    return (home_dir() / "activity").resolve()


def _redact_owner_digests(raw: str) -> str:
    """Strip owner key digests out of a corrupt record file before it is copied
    aside, keeping everything else the copy exists to preserve.

    Reports at warning when a digest-shaped value survives: a partially corrupt
    file can hold one in a position this cannot match, and a redaction that
    silently half-happened must not look like one that fully did (rule 5)."""
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

    Every method is BEST-EFFORT for the caller's purposes: ``write`` returns
    False rather than raising, and ``reap`` returns whatever it could read. A
    persistence problem must never break the operation being recorded, and must
    never be silent either - each failure logs (rule 5)."""

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
            # The shared temp+restrict+replace primitive: rows carry the owner key
            # digest, so the file must be owner-only from the moment the bytes
            # first exist, on Windows as well as POSIX. Its FIXED "<name>.tmp" temp
            # name requires that the destination path stays unique per (process,
            # store) and that _STORE_LOCK serialises every writer in this process.
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
        record - so a data dir does not accumulate empty files."""
        with _STORE_LOCK:
            self._clear()

    def _clear(self) -> None:
        # Broad, like write(): no caller's control flow ever changes because of a
        # store problem, and the path itself is resolved lazily (home_dir()), so
        # even computing it can fail.
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
        (including the caller's own). It exists because pid liveness cannot answer
        the same-process question: this process is obviously alive, so a file
        bearing our pid is either our own live record or a leftover from a
        DIFFERENT process that held this pid before us, and only the in-process
        registry can tell those apart.

        A live foreign writer's file is left completely alone - not read, not
        adopted, not deleted - so two servers sharing a data dir never take each
        other's operations."""
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
                # The rows are now held by this process, so a file left behind can
                # be adopted again by a third process and reported twice. Report it
                # rather than claiming a clean handover.
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
        not follow the scheme (a hand-dropped file: read it, do not guess).

        Only the pid half is parsed. The run nonce after it is never compared to
        anything: a file bearing OUR pid but a different nonce is a leftover from
        a process that held this pid before us, and _live_store_paths - not the
        nonce - is what identifies our own live files."""
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
            # Neither collapsed to an empty list nor raised: this process never
            # writes another writer's file, so there is nothing here to erase.
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
        """Copy a corrupt record file aside before anything removes it, and warn
        (rule 5: a data-loss risk must be visible, never silent). Redacted and
        pruned - see _redact_owner_digests and _prune_quarantine for why each is
        needed and why neither replaces the other."""
        try:
            from localm.debuglog import logger
        except Exception:
            logger = None
        try:
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            if raw is not None:
                backup.write_text(_redact_owner_digests(raw), encoding="utf-8")
                # The copy still describes the user's operations and may carry a
                # residual digest, so it is restricted the way the live file is.
                # Check-and-retry like atomic_write_private.
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

        Bounded across the whole DIRECTORY rather than per file name, because
        per-pid names would otherwise give every pid its own unbounded series.
        Sorted by the timestamp SUFFIX, not lexically: the pid prefix sorts first,
        so a lexical order would prune by pid instead of by age.

        Best-effort by design: failing to delete an old backup must never break
        the recovery path that just successfully wrote a new one."""
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
# Every live JobManager in this process, reachable at module scope. os._exit /
# os.execv (http_server's _do_shutdown / _do_restart) bypass atexit and never
# touch a start_cli CHILD, and they take no app and hold no manager.
#
# A WeakSet, so a manager built by a test (or by an app that is torn down) does
# not keep itself alive here.
_MANAGERS: "weakref.WeakSet" = weakref.WeakSet()

def _live_store_paths() -> set:
    """The record-file paths owned by managers that are ALIVE in this process.

    Reconciliation needs this because pid liveness cannot answer the same-process
    question - see _ActivityStore.reap. Reading the private _store attribute of a
    sibling manager is deliberate: the set is meaningless to anyone outside this
    module, so exposing it as API would invite a caller to act on it."""
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
    spawn their OWN children (comfy setup runs git and pip - see
    media/managed_comfy_provision.py, which notes it sits one process-hop inside
    this one). Killing only the direct child strands those.

    The tree half is BEST-EFFORT and says so rather than overclaiming:

    * Windows uses ``taskkill /F /T``, which walks the child tree. There is no
      graceful tree-wide signal on Windows, and Popen.terminate() is itself
      TerminateProcess, so nothing is lost by going straight to it here.
    * POSIX walks the tree with psutil, which is an OPTIONAL dependency (pyproject
      declares it under gpu/monitor/dev, not core). Without it only the direct
      child is reached, and that limit is LOGGED rather than glossed.

    Deliberately no start_new_session/creationflags change to the Popen itself:
    that would alter signal delivery for every existing job (a host Ctrl+C would
    stop reaching a pull), which is a separate decision from this one."""
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
                # raising: "not found" when the child exited between our poll and
                # this call, or access-denied. Fall through to the handle fallback
                # below instead of returning.
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
    """Force-kill via the Popen handle - bound to the object we launched, so it can
    never hit a recycled pid."""
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
    their ``finally`` may never run, and the Popen carries no creationflags - so
    without this a stop or restart simply ABANDONS the child.

    ADR-0008 inferred that from the code shape and recorded that it was NOT
    measured. It is measured now, and the answer has two halves rather than the
    one the inference expected. MEASURED 2026-08-19 on Windows, both children
    spawned exactly as start_cli does and then abandoned by an os._exit(0):

        writes NOTHING to stdout    SURVIVED, and kept working (its heartbeat
                                    advanced after the parent was gone)
        writes and FLUSHES          DIED at its next write, on the broken pipe

    Neither outcome is acceptable, which is why this is unconditional. A quiet
    child (a git clone, a pip install, a long file write between progress
    emissions) keeps running untracked, holding VRAM in the media case and still
    writing into the shared data dir. A chatty child - a pull flushes a progress
    line constantly - is instead torn down at an arbitrary instant with no cleanup
    and no record, which is not a graceful stop but a crash that happens to look
    like nothing occurred. Terminating deliberately replaces both with one known
    state, which the next start then reports as "interrupted"."""
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

    DURABLE since 2026-08-19 (ADR-0008's option E). The registry is still the
    in-memory dict below - that is what SSE subscribers, progress and cancellation
    all attach to - with a small record of each operation's LIFECYCLE mirrored to
    ``<data dir>/activity/<pid>.json`` so a restart can still say what was running.
    Progress is not mirrored, deliberately: see _ActivityStore's module comment.

    On construction it RECONCILES: records left by a writer process that is gone
    are adopted, and any of them still marked ``running`` become ``interrupted``,
    because nobody measured whether that work succeeded (see _JOB_STATUSES)."""

    _TTL_S = 3600

    def __init__(self, *, store: Optional["_ActivityStore"] = None,
                 reconcile: bool = True) -> None:
        self._jobs: dict[str, Job] = {}
        # An RLock, not a Lock: persisting happens while holding it, so the file
        # can never be written out of order relative to the dict it mirrors, and
        # the persist path is reached both directly and through Job._notify. Lock
        # order in this module is _lock OUTER, Job._sub_lock INNER, the order
        # snapshot() -> summary() already establishes. to_record reads plain fields
        # only, so persisting adds no new lock edge.
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
            # eviction it just made; there is no separate persist for the sweep.
            self._gc()
            self._jobs[job.id] = job
            self._persist_locked()

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        """Mirror the registry to disk. Caller holds _lock.

        Best-effort by contract: _ActivityStore.write reports its own failures and
        returns False rather than raising, so a full disk cannot fail a model pull.
        An EMPTY registry removes the file instead of writing an empty one, so a
        data dir does not accumulate one file per process that ever started."""
        rows = [j.to_record() for j in self._jobs.values()]
        if rows:
            self._store.write(rows)
        else:
            self._store.clear()

    def _reconcile_from_disk(self) -> None:
        """Adopt the operation records of writer processes that are gone.

        A row still marked ``running`` is reported as ``interrupted``: this server
        stopped while it was in flight, so its outcome is genuinely unknown. It is
        NOT reported as ``failed`` - ADR-0008 R3 - because a pull that was 99% done
        may well have finished, and claiming otherwise would be a fabrication in
        the one direction a user cannot check.

        ``finished_at`` is stamped at DETECTION rather than derived from the file's
        mtime, and that is load-bearing rather than lazy: _gc() sweeps on
        finished_at, so a timestamp from before the crash would put the row past
        the TTL cutoff the instant it was recovered - evicting it before the user
        who just restarted the server could ever see it. That is the exact defect
        _gc()'s own docstring records for created_at."""
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
        """Give a recovered job a terminal event stream.

        Without this, a client that reattaches to a recovered id over
        ``GET /api/jobs/{id}/events`` gets an empty history and then keepalives
        forever, because the worker thread that would have pushed the ``end``
        frame died with the previous process. The stream has to be able to say
        "this is over" for the same reason the status has to."""
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
        return how many were signalled. See the module-level function of the same
        name for why this exists at all.

        Enumerates under the lock and kills OUTSIDE it: a tree kill runs a
        subprocess with its own timeout, and holding a lock other threads need
        across that would stall them for the duration.

        Deliberately does NOT go through Job.cancel(): cancel means "the user asked
        to stop this", and a shutdown is not that. The registry is left saying
        ``running``, which is what makes the next start reconcile these rows to
        ``interrupted`` - the honest word for what actually happened to them."""
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
        console + debug log (G2) so a client-initiated pull is visible there.
        owner, when given, is the creating key's principal id - only that key (or
        an admin/owner) may later stream or cancel the job (KEY-SCOPE-2).
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
            # An explicit {"type": "outcome"} sentinel frame (_shared._emit_outcome)
            # overrides the exit-code guess below. None means no such frame arrived
            # (an older CLI build, a job kind that never emits one, or a crash
            # before it could be sent), and the exit-code rule then applies
            # unchanged. It can only correct a misleading exit code, never turn
            # silence into "done".
            reported_outcome = None
            try:
                env = None
                if extra_env:
                    env = os.environ.copy()
                    env.update(extra_env)
                job._proc = subprocess.Popen(
                    job.argv,
                    stdin=subprocess.DEVNULL,   # no inherited TTY, so interactive
                    # dedup/overwrite prompts (pull, imagine) take their safe
                    # non-interactive default (skip) instead of aborting the job on
                    # an unfed terminal stdin.
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
                # Re-persist now that the child EXISTS: the create-time write
                # happened before this Popen, so it recorded no pid. That pid is
                # what lets a later run report an orphaned child.
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
                        # popped, not merely read: **data below must never carry
                        # its own "type" key, or a dict literal's later-key-wins
                        # rule would let a payload override the "progress" label
                        # assigned here.
                        etype = data.pop("type", "progress")
                        if etype == "outcome":
                            # An internal producer -> job-runner signal
                            # (_shared._emit_outcome), never forwarded to
                            # subscribers; it only corrects the status decision
                            # below.
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
                # jobs on "end" never sees this one still claiming to be in flight.
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
        replaying the stream, and may update ``job.result``. Prefer
        ``job.progress`` over formatting numbers into a line: a fraction that
        only exists inside prose is invisible to ``/api/activity``, to the CLI
        and to MCP. If ``fn``'s own real work is done but it still runs risky
        tail cleanup afterward (a VRAM handover, a bookkeeping call), call
        ``job.mark_outcome("done")`` first - see that method's docstring - so a
        later exception in the tail cannot misreport a completed operation as
        failed. owner, when given, binds the job to the creating key's
        principal id so only that key (or an admin/owner) may stream/cancel it.
        label, when given, is the human-readable operation name a listing shows
        (the start_cli equivalent is host_label, which doubles as the host
        console prefix; there is no console mirroring for in-thread jobs).
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
                    # The callback called mark_outcome("done") before this raised,
                    # so the mark wins over the exception. None (never marked) and
                    # "failed" both fall through to the else branch below.
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
        whether that job appears. It takes a PREDICATE rather than a principal
        id so the owner never has to leave this class: the caller supplies the
        policy (``job_owner_ok`` needs the request to evaluate it), the manager
        keeps the identity, and ``summary()`` still never carries it.

        This is the ONLY way to learn a job exists without already holding its
        id. Until it existed, a job id was handed out exactly once - in the body
        of the POST that started the job - so a second client, or the same tab
        after a reload, had no way to ask what was running even though the
        server knew (ADR-0008).

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
        """Whether a job of *kind* is currently running - so a caller can tell
        "still actively installing" apart from "abandoned mid-install" without
        its own job-tracking state (e.g. the managed-ComfyUI status check,
        which must not call a stalled/incomplete install "corrupt" while its
        own setup job is genuinely still in flight)."""
        with self._lock:
            return any(j.kind == kind and j.status == "running"
                      for j in self._jobs.values())

    def _gc(self) -> None:
        """Drop jobs that finished more than _TTL_S ago.

        Keyed on finished_at, NOT created_at. The class docstring has always
        promised "finished jobs stay queryable for an hour"; keying the sweep on
        created_at did not deliver that, because a job that RAN for longer than
        the TTL was already past the cutoff the moment it finished, so a
        two-hour pull became unqueryable the instant it succeeded. finished_at
        makes the code match the promise.

        A still-running job is never swept at any age, and that is deliberate
        rather than an oversight: evicting a live job would strand its SSE
        subscribers and lose the record while the work carries on. An operation
        that has legitimately been running for hours is reported as running,
        which is true; created_at travels in the summary so a client can tell a
        six-second job from a six-hour one instead of being told only "running".
        """
        cutoff = time.time() - self._TTL_S
        stale = [
            jid for jid, j in self._jobs.items()
            if j.status != "running"
            and (j.finished_at if j.finished_at is not None else j.created_at) < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]
