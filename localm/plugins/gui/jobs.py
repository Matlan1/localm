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
    # When the worker thread left, i.e. when this job stopped being in flight.
    # None while it is still running. Deliberately SEPARATE from created_at,
    # because the TTL sweep keys on this one: a two-hour job that finished a
    # second ago must survive, and keying the sweep on created_at evicted
    # exactly that job (it was already past the cutoff the moment it finished).
    finished_at: Optional[float] = None
    # Human-readable name for this operation, in the same words the host console
    # already uses (start_cli's host_label), so a listing can describe a job
    # without re-deriving a label from argv.
    label: Optional[str] = None
    # Stable id (keystore hash) of the key that created this job, or None when NO
    # key is configured at all or no token was presented. The events/cancel routes
    # accept the creator or an admin/owner only (KEY-SCOPE-2), so a leaked job id
    # is not enough to touch another key's job.
    #
    # READ THE OPEN-MODE CASE CAREFULLY, because it is the DEFAULT and an earlier
    # version of this comment had it backwards: principal_id() returns None
    # whenever no owner key and no keystore are configured, and the loopback GUI's
    # shell token is NEITHER of those - so in the default configuration jobs are
    # UNOWNED, and job_owner_ok() then admits any authenticated caller. Ownership
    # discriminates in KEYED mode only. Anything built on this field must be
    # correct in both modes and must not be described as if ownership were always
    # enforced.
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
    # The most recent {"type": "progress", ...} event, so a listing can report
    # pct/phase without replaying the whole history. Written under _sub_lock in
    # push(), alongside the history append it is derived from.
    _last_progress: Optional[dict] = None

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
                # Not clamped: a numerator past its denominator is a bug in the
                # CALLER, and silently pinning it to 100% would hide the bug
                # while also making a false completion claim (AGENTS.md rule 5).
                # Surfaced here so it is attributable, and left visible upstream.
                # Local import matching _HostAnnouncer._say below: debuglog is
                # imported inside the call site throughout this module.
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

    def mark_finished(self) -> None:
        """Stamp finished_at once, when the worker thread leaves. Idempotent so
        a second call (or a cancel that raced the thread's own exit) cannot move
        the timestamp forward and give the job a second lease on the TTL."""
        if self.finished_at is None:
            self.finished_at = time.time()

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
            label=host_label,
        )
        with self._lock:
            self._gc()
            self._jobs[job.id] = job

        def _run():
            announcer = _HostAnnouncer(host_label) if host_label else None
            if announcer:
                announcer.announce_start()
            # An explicit {"type": "outcome"} sentinel frame (_shared._emit_outcome)
            # overrides the exit-code guess below. None means no such frame arrived
            # - an older CLI build, a job kind that never emits one, or a crash
            # before it could be sent - and the exit-code rule is then EXACTLY
            # today's behavior: this can only ever correct a misleading exit code,
            # never invent a "done" out of silence.
            reported_outcome = None
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
                        # popped, not merely read: **data below must never carry
                        # its own "type" key, or a dict literal's later-key-wins
                        # rule would let a payload override the "progress" label
                        # this reader assigns everything else on this channel.
                        etype = data.pop("type", "progress")
                        if etype == "outcome":
                            # An internal producer -> job-runner signal (see
                            # _shared._emit_outcome), never forwarded to
                            # subscribers - it exists solely to correct the
                            # status decision below, not to be rendered.
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
                # Stamp BEFORE the end event goes out: a subscriber that reacts
                # to "end" by listing jobs must not see this one still claiming
                # to be in flight.
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
        and to MCP. owner, when given, binds the job to the creating key's
        principal id so only that key (or an admin/owner) may stream/cancel it.
        label, when given, is the human-readable operation name a listing shows
        (the start_cli equivalent is host_label, which doubles as the host
        console prefix; there is no console mirroring for in-thread jobs).
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=[], result=result_path,
                  owner=owner, label=label)
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
                # See the start_cli equivalent: stamp before "end" is emitted.
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
