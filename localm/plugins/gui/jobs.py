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
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from localm.model_manager import PROGRESS_SENTINEL

# Bound on both the replay backlog and each subscriber's queue, mirroring the
# old single-queue's maxsize so a very long-lived job cannot grow unbounded.
_HISTORY_MAX = 10_000


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
    status: str = "running"        # running | done | failed | cancelled
    returncode: Optional[int] = None
    result: Optional[str] = None   # kind-specific payload (e.g. output image path)
    created_at: float = field(default_factory=time.time)
    # Stable id (keystore hash) of the key that created this job, or None only when
    # NO token was presented at all (a fully anonymous request). The events/cancel
    # routes accept the creator or an admin/owner only (KEY-SCOPE-2), so a leaked
    # job id is not enough to touch another key's job. (Note: the open-mode loopback
    # GUI still presents its shell token, so its jobs are owned by that token's id,
    # not None - tokenless callers simply cannot reach them.)
    owner: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None
    # Set by cancel(); in-thread jobs (start_fn, e.g. media gen) poll this to
    # stop cooperatively since there is no subprocess to terminate.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Every event ever pushed (bounded, oldest evicted first) so a viewer that
    # subscribes mid-job still sees the full stream from the start. Each SSE
    # connection then gets its OWN asyncio.Queue in _subscribers, fed live by
    # push() - a plain queue.Queue is single-consumer, so two concurrent viewers
    # used to split a job's events between them instead of each seeing all of them.
    _history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_HISTORY_MAX))
    _subscribers: list = field(default_factory=list)
    _sub_lock: threading.Lock = field(default_factory=threading.Lock)

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
            subs = list(self._subscribers)
        for q, loop in subs:
            loop.call_soon_threadsafe(_safe_put, q, event)

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


class _HostAnnouncer:
    """Mirror a background job's lifecycle to the HOST - the person running
    ``localm gui`` - not only to the requesting client (G2).

    A model pull started from a phone/PWA ran silently as far as the host was
    concerned: its output went only to the per-job event queue the browser reads.
    Now the host also sees a start line, throttled progress (10% steps so it does
    not spam), and the end status. Output is ephemeral (host stdout + the debug
    log), never a privacy-mode disk trace - the model spec is operational, not
    session content. ``line()`` is pure so the throttling is unit-testable."""

    # A job's per-line CLI output (record_line) used to reach ONLY the
    # ephemeral per-job SSE history a browser tab happened to have open - never
    # the debug log or a bug report, even though it carries the actual reason
    # a job failed (a git/pip/native error). #621: a ComfyUI setup failure
    # left nothing more informative than "ComfyUI setup failed" anywhere a bug
    # report could read from. Keep a bounded tail so a failure can log it.
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
            # The GUI server's stdout is the host terminal, so a plain print is the
            # one channel guaranteed to surface this to the host.
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


class JobManager:
    """Registry of background jobs. Finished jobs stay queryable for an hour."""

    _TTL_S = 3600

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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
        )
        with self._lock:
            self._gc()
            self._jobs[job.id] = job

        def _run():
            announcer = _HostAnnouncer(host_label) if host_label else None
            if announcer:
                announcer.announce_start()
            try:
                env = None
                if extra_env:
                    env = os.environ.copy()
                    env.update(extra_env)
                job._proc = subprocess.Popen(
                    job.argv,
                    stdin=subprocess.DEVNULL,   # no inherited TTY: interactive
                    # dedup/overwrite prompts (pull, imagine) must take their
                    # safe non-interactive default (skip), never click.Abort on
                    # an unfed terminal stdin and fail the job with "Aborted!".
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
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
                    job.status = "done" if job.returncode == 0 else "failed"
            except Exception as e:
                job.status = "failed"
                job.push({"type": "line", "text": f"job error: {e}"})
                if announcer:
                    announcer.record_line(f"job error: {e}")
            finally:
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
                 owner: str | None = None) -> Job:
        """
        Run a Python callable as a job in a worker thread.

        ``fn`` receives the job and should return True on success. It may call
        ``job.push({"type": "line", ...})`` to report progress and may update
        ``job.result``. owner, when given, binds the job to the creating key's
        principal id so only that key (or an admin/owner) may stream/cancel it.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=[], result=result_path,
                  owner=owner)
        with self._lock:
            self._gc()
            self._jobs[job.id] = job

        def _run():
            try:
                ok = fn(job)
                if job.status != "cancelled":
                    job.status = "done" if ok else "failed"
            except Exception as e:
                job.status = "failed"
                job.push({"type": "line", "text": f"job error: {e}"})
            finally:
                job.push({
                    "type": "end",
                    "status": job.status,
                    "returncode": job.returncode,
                    "result": job.result,
                })

        threading.Thread(target=_run, daemon=True).start()
        return job

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
        cutoff = time.time() - self._TTL_S
        stale = [
            jid for jid, j in self._jobs.items()
            if j.status != "running" and j.created_at < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]
