# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background job registry for the coder plugin.

The coder's process tools were blocking-only, so it could not start a dev server
and then talk to it, or run a long build while doing anything else. This module
is the async half: a GENERIC job registry (``JobRegistry``) plus the concrete
:class:`ShellJob`.

Generic on purpose, because it is shared infrastructure. ``BackgroundJob`` knows
only about a lifecycle (running -> done/killed/failed), an OPAQUE per-kind result
payload, a kill hook, and a bounded output buffer; everything process-specific
lives in ``ShellJob``. A background sub-agent is a second subclass with
``kind = "agent"`` and its own payload, not a second registry: two parallel job
systems in one plugin would be a design smell.

The registry therefore offers three things a non-shell caller needs:

* **Per-kind caps.** ``kind_caps`` holds a separate ceiling per job kind (shell
  jobs and agent jobs exhaust different resources, so one shared number would be
  wrong for both).
* **Atomic check-and-insert.** The cap check and the spawn happen under one lock
  acquisition, so two near-simultaneous submits can never both observe a free
  slot and both be admitted.
* **Drain, not just poll.** ``drain_finished()`` returns everything that finished
  since the last drain, for a caller that absorbs completions at a turn boundary
  rather than polling a known id.

Concurrency style follows the GUI coder sessions (``sessions.py``): daemon
threads plus a bounded buffer that drops the oldest entry when full. That
module's ``queue.Queue`` is per-session and not reusable as a general table, so
the pattern is reproduced here rather than lifted.

Three invariants are load-bearing:

1. **No pid-reuse window.** ``Popen.poll()`` is called ONLY under the job lock,
   and this module is the single reaping call site for its children. So a
   ``poll() is None`` observed under that lock proves the child is still
   unreaped: on POSIX it is a live process or a zombie (either way the pid is
   not reusable), and on Windows we still hold the process handle (which
   reserves the pid). Killing by pid is therefore safe at that instant. psutil's
   ``create_time`` is pinned alongside the pid as a second, independent check,
   but correctness does not depend on it: psutil is an OPTIONAL dependency here,
   not a core one.
2. **Kill reaps the TREE.** A build or dev server spawns children; killing only
   the direct child strands them. POSIX gets its own session/process group
   (``start_new_session``) and is killed with ``killpg``; Windows uses
   ``taskkill /F /T``, which walks the child tree. We never kill by port or by
   image name - only by a pid we have just proven is still ours.
3. **Bounded memory.** A chatty process cannot grow the buffer without limit,
   and anything dropped is COUNTED and reported (never silently discarded).
"""

from __future__ import annotations

import atexit
import codecs
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Optional

# Per-kind ceilings on jobs running at once. Separate numbers because the kinds
# exhaust different resources: shell jobs are OS processes the model started and
# may forget about, agent jobs each hold a model context. A kind with no entry
# falls back to _DEFAULT_CAP. Rejecting past the cap is a CLEAR error the model
# can act on, never a silent queue that looks like it started.
_KIND_CAPS: dict[str, int] = {
    "shell": 4,
}
_DEFAULT_CAP = 4

# Per-stream output cap. Chars, not lines: a progress bar that only emits '\r'
# never produces a line, so a line-based cap would let one "line" grow forever.
_RING_MAX_CHARS = 256_000

# Finished jobs stay queryable (that is the whole point of check_shell_job after
# completion) but are pruned oldest-first so a long session cannot accumulate
# them without limit. Finished jobs never count toward a cap.
_KEEP_FINISHED = 16

_POLL_INTERVAL = 0.05     # seconds between liveness polls
_READ_CHUNK = 65_536      # bytes per raw pipe read
_KILL_GRACE = 3.0         # seconds to wait after a graceful terminate
_DRAIN_GRACE = 2.0        # seconds to wait for reader threads at finish


class JobError(RuntimeError):
    """Base class for job-registry errors."""


class JobCapacityError(JobError):
    """Raised when a per-kind concurrent-job cap would be exceeded."""


# --------------------------------------------------------------------------- #
#  Bounded output buffer                                                       #
# --------------------------------------------------------------------------- #

class RingBuffer:
    """A capped FIFO of text chunks, dropping the OLDEST when full.

    Stores raw chunks rather than lines so that output with no newlines (a
    ``\\r`` progress bar, a binary-ish blob) is capped just like any other.
    ``dropped`` counts the characters evicted, so a caller can say how much was
    lost instead of presenting a truncated tail as if it were everything.
    """

    def __init__(self, max_chars: int = _RING_MAX_CHARS) -> None:
        self._chunks: deque[str] = deque()
        self._chars = 0
        self._dropped = 0
        self._max = max(1, int(max_chars))
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._chunks.append(text)
            self._chars += len(text)
            while self._chars > self._max and len(self._chunks) > 1:
                old = self._chunks.popleft()
                self._chars -= len(old)
                self._dropped += len(old)
            # A single chunk bigger than the whole cap cannot be dropped without
            # losing everything, so trim it from the FRONT and keep the tail
            # (the newest output is the part the model needs).
            if self._chars > self._max and self._chunks:
                only = self._chunks.pop()
                keep = only[-self._max:]
                self._dropped += len(only) - len(keep)
                self._chunks.append(keep)
                self._chars = len(keep)

    def read(self) -> tuple[str, int]:
        """Return ``(text, dropped_chars)``."""
        with self._lock:
            return "".join(self._chunks), self._dropped


# --------------------------------------------------------------------------- #
#  pid identity helpers (the belt-and-braces half of invariant 1)              #
# --------------------------------------------------------------------------- #

def _process_create_time(pid: int) -> Optional[float]:
    """The process start timestamp, or None when psutil is unavailable.

    psutil is an optional dependency (pyproject declares it only in extras), so
    None is the normal case on a core install, not an error.
    """
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _still_the_same_process(pid: int, create_time: Optional[float]) -> bool:
    """Is *pid* still the process we started?

    Returns True when we cannot tell (no psutil, or an unexpected probe error) -
    the caller's lock-plus-unreaped-child argument is the primary guarantee, and
    this check only ever adds a veto. Returns False only on positive evidence of
    a mismatch (the pid is gone, or its start time no longer matches).
    """
    if create_time is None:
        return True
    try:
        import psutil
    except Exception:
        return True
    try:
        # Tolerance covers platform clock granularity on create_time; it is far
        # tighter than any realistic pid-recycle interval.
        return abs(psutil.Process(pid).create_time() - create_time) < 0.05
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return True


# --------------------------------------------------------------------------- #
#  Job base class                                                              #
# --------------------------------------------------------------------------- #

class BackgroundJob:
    """One unit of asynchronous work tracked by :class:`JobRegistry`.

    The record is deliberately kind-agnostic::

        {id, kind, label, state, started_at, finished_at, result, error, warnings}

    ``result`` is an OPAQUE per-kind payload (a shell job puts ``exit_code``
    there; an agent job would put its own summary), so adding a kind never means
    reshaping the table. Subclasses implement :meth:`_poll`, :meth:`_terminate`
    and optionally :meth:`_result_for` / :meth:`_drain`.
    """

    kind = "job"

    def __init__(self, label: str) -> None:
        self.id = "job_" + uuid.uuid4().hex[:8]
        self.label = label
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.state = "running"          # running | done | killed | failed
        # Opaque, per-kind terminal payload. None while running.
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        # Non-fatal problems worth surfacing rather than swallowing (a reader
        # thread that died, a kill that had to fall back). Reported by status().
        self.warnings: list[str] = []
        # Set by JobRegistry.drain_finished once this job's completion has been
        # handed to its owner, so a drain-based consumer never sees it twice and
        # pruning can prefer jobs somebody has already collected.
        self.drained = False
        self._lock = threading.RLock()
        self._watcher: Optional[threading.Thread] = None

    # -- subclass hooks ---------------------------------------------------- #

    def _poll(self):
        """Terminal value if finished, else None. Called ONLY under the lock."""
        raise NotImplementedError

    def _terminate(self, *, force: bool) -> None:
        """Signal the job (and its whole tree) to stop. Under ``self._lock``."""
        raise NotImplementedError

    def _result_for(self, poll_value) -> Optional[dict]:
        """Convert the terminal poll value into the opaque per-kind payload."""
        return None

    def _drain(self, timeout: float) -> bool:
        """Flush any in-flight output. True when fully drained."""
        return True

    # -- lifecycle --------------------------------------------------------- #

    def start_watcher(self) -> None:
        self._watcher = threading.Thread(
            target=self._watch, name=f"bgjob-{self.id}", daemon=True)
        self._watcher.start()

    def _watch(self) -> None:
        while True:
            with self._lock:
                if self.state != "running":
                    return
                value = self._poll()
                if value is not None:
                    self._finish("done", self._result_for(value))
                    return
            time.sleep(_POLL_INTERVAL)

    def _finish(self, state: str, result: Optional[dict],
                error: Optional[str] = None) -> None:
        """Record the terminal state. Caller must hold ``self._lock``."""
        if not self._drain(_DRAIN_GRACE):
            # Not fatal: a detached grandchild can hold the pipe open after the
            # job itself exited. Say so rather than presenting a possibly
            # incomplete tail as the full output.
            self.warnings.append(
                "output readers did not reach EOF within "
                f"{_DRAIN_GRACE:g}s - a detached child may still hold the pipe; "
                "the captured output may be incomplete")
        self.result = result
        self.error = error
        self.finished_at = time.time()
        self.state = state

    def kill(self) -> str:
        """Stop the job and its process tree. Returns a human-readable outcome."""
        with self._lock:
            if self.state != "running":
                return f"already {self.state}"
            # Reaping happens here, under the lock, so the pid cannot have been
            # recycled between this check and the kill below (invariant 1).
            value = self._poll()
            if value is not None:
                self._finish("done", self._result_for(value))
                return "already finished"

            self._terminate(force=False)
            value = self._wait_for_exit(_KILL_GRACE)
            if value is None:
                self._terminate(force=True)
                value = self._wait_for_exit(_KILL_GRACE)
            if value is None:
                # Do NOT report success for a kill that did not happen.
                msg = "the process did not exit after a forced kill"
                self.warnings.append(msg)
                self._finish("failed", None, error=msg)
                return "kill FAILED - the process is still running"
            self._finish("killed", self._result_for(value))
            return "killed"

    def _wait_for_exit(self, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            value = self._poll()
            if value is not None:
                return value
        return None

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def status(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "state": self.state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed": self.elapsed(),
                "result": dict(self.result) if self.result is not None else None,
                "error": self.error,
                "warnings": list(self.warnings),
            }


# --------------------------------------------------------------------------- #
#  Shell job                                                                   #
# --------------------------------------------------------------------------- #

class ShellJob(BackgroundJob):
    """A background OS process with its stdout/stderr drained into ring buffers.

    *argv* is an already-routed argument list - the caller decides shell vs
    argument-list mode (``tools/shell.py:_shell_argv``), so the background path
    and the blocking ``run_shell`` path make that security decision in exactly
    one place.

    Its opaque result payload is ``{"exit_code": int}``.
    """

    kind = "shell"

    def __init__(self, argv: list, cwd: Path, *, label: str,
                 env: Optional[dict] = None,
                 max_chars: int = _RING_MAX_CHARS) -> None:
        super().__init__(label)
        self.stdout = RingBuffer(max_chars)
        self.stderr = RingBuffer(max_chars)

        kwargs: dict = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            # A background job must never block waiting on input it cannot get,
            # and must never steal the interactive coder's stdin.
            "stdin": subprocess.DEVNULL,
            # bufsize=0 gives raw FileIO pipes, so a read() returns whatever is
            # available instead of blocking for a full buffer or a newline.
            "bufsize": 0,
            "env": env,
        }
        if sys.platform == "win32":
            # Its own group, so a Ctrl+C in the coder console does not tear down
            # background jobs the model is still using.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Its own session/process group, so killpg reaps the whole tree.
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(argv, **kwargs)
        self.pid = self._proc.pid
        self._create_time = _process_create_time(self.pid)
        self._readers = [
            self._start_reader(self._proc.stdout, self.stdout, "stdout"),
            self._start_reader(self._proc.stderr, self.stderr, "stderr"),
        ]
        self.start_watcher()

    @property
    def exit_code(self) -> Optional[int]:
        return (self.result or {}).get("exit_code")

    def _result_for(self, poll_value) -> Optional[dict]:
        return {"exit_code": poll_value}

    # -- output draining ---------------------------------------------------- #

    def _start_reader(self, stream, ring: RingBuffer, name: str) -> threading.Thread:
        t = threading.Thread(target=self._read_stream, args=(stream, ring, name),
                             name=f"bgjob-{self.id}-{name}", daemon=True)
        t.start()
        return t

    def _read_stream(self, stream, ring: RingBuffer, name: str) -> None:
        """Drain one pipe into *ring* until EOF.

        Deliberately touches no job state except ``warnings``: it must never
        take ``self._lock``, or a bounded join under that lock could deadlock.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = stream.read(_READ_CHUNK)
                if not chunk:
                    break
                ring.append(decoder.decode(chunk))
        except (OSError, ValueError) as e:
            # Reaching here means the pipe broke in a way a normal EOF does not
            # cover. Record it: output may be missing, and silently returning
            # would present a partial capture as complete.
            self.warnings.append(f"{name} reader stopped early: {type(e).__name__}: {e}")
        finally:
            ring.append(decoder.decode(b"", final=True))
            try:
                stream.close()
            except Exception:
                pass

    def _drain(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        for t in self._readers:
            remaining = deadline - time.time()
            if remaining > 0:
                t.join(timeout=remaining)
        return not any(t.is_alive() for t in self._readers)

    # -- process control ---------------------------------------------------- #

    def _poll(self):
        return self._proc.poll()

    def _terminate(self, *, force: bool) -> None:
        if not _still_the_same_process(self.pid, self._create_time):
            # Positive evidence the pid is no longer ours. Never signal it by
            # number; fall back to the Popen handle, which is bound to the
            # object we launched and cannot hit a recycled pid.
            self.warnings.append(
                f"pid {self.pid} no longer matches the process we started - "
                "killing via the process handle only, not by pid")
            self._kill_via_handle(force=force)
            return
        if sys.platform == "win32":
            self._terminate_tree_windows()
        else:
            self._terminate_tree_posix(force=force)

    def _terminate_tree_windows(self) -> None:
        # taskkill /T walks the child tree; /F is the only reliable mode for it
        # (there is no graceful tree-wide signal on Windows).
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.pid)],
                capture_output=True, timeout=10,
            )
            return
        except FileNotFoundError:
            self.warnings.append("taskkill not found - falling back to psutil/handle kill")
        except Exception as e:
            self.warnings.append(f"taskkill failed ({type(e).__name__}: {e}) - falling back")
        self._kill_children_via_psutil()
        self._kill_via_handle(force=True)

    def _terminate_tree_posix(self, *, force: bool) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            pgid = os.getpgid(self.pid)
        except OSError:
            pgid = None
        # If start_new_session somehow did not take, the child shares OUR group
        # and killpg would take down localm itself. Never signal our own group.
        if pgid is not None and pgid != os.getpgrp():
            try:
                os.killpg(pgid, sig)
                return
            except OSError as e:
                self.warnings.append(f"killpg({pgid}) failed: {e} - falling back")
        else:
            self.warnings.append(
                "job is in this process's own group - killing the child only, "
                "its descendants may survive")
        self._kill_children_via_psutil()
        self._kill_via_handle(force=force)

    def _kill_children_via_psutil(self) -> None:
        """Best-effort descendant sweep for the fallback paths.

        Only reachable when the OS-level tree kill above was unavailable, and
        only ever touches processes psutil reports as descendants of OUR pid -
        never a port or an image name.
        """
        try:
            import psutil
        except Exception:
            return
        try:
            for child in psutil.Process(self.pid).children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
        except Exception:
            pass

    def _kill_via_handle(self, *, force: bool) -> None:
        try:
            if force:
                self._proc.kill()
            else:
                self._proc.terminate()
        except Exception as e:
            self.warnings.append(f"handle kill failed: {type(e).__name__}: {e}")

    # -- reporting ---------------------------------------------------------- #

    def output(self) -> tuple[str, str, int]:
        """``(stdout_text, stderr_text, dropped_chars)`` captured so far."""
        out, out_dropped = self.stdout.read()
        err, err_dropped = self.stderr.read()
        return out, err, out_dropped + err_dropped

    def status(self) -> dict:
        st = super().status()
        st["pid"] = self.pid
        return st


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

class JobRegistry:
    """Tracks background jobs, caps how many run at once per kind, reaps at exit.

    Kind-agnostic by design: it holds :class:`BackgroundJob` instances, so a
    background sub-agent registers here too and inherits the caps, the lookup,
    the drain, and the shutdown behaviour.
    """

    def __init__(self, kind_caps: Optional[dict] = None,
                 default_cap: int = _DEFAULT_CAP,
                 keep_finished: int = _KEEP_FINISHED) -> None:
        self.kind_caps = dict(_KIND_CAPS if kind_caps is None else kind_caps)
        self.default_cap = default_cap
        self.keep_finished = keep_finished
        # Completions evicted before any drain collected them. Should stay 0;
        # a non-zero value means a drain-based consumer silently missed results.
        self.dropped_undrained = 0
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        # A job the model started must not outlive the localm process: it was
        # detached into its own group/session precisely so it survives signals,
        # which is exactly what makes it an orphan if we just exit.
        atexit.register(self.shutdown_all)

    def cap_for(self, kind: str) -> int:
        return self.kind_caps.get(kind, self.default_cap)

    def running(self, kind: Optional[str] = None) -> list:
        with self._lock:
            return [j for j in self._jobs.values()
                    if j.state == "running" and (kind is None or j.kind == kind)]

    def submit(self, factory: Callable[[], BackgroundJob], kind: str) -> BackgroundJob:
        """Create and register a job of *kind*, enforcing that kind's cap.

        The cap check and the spawn happen under ONE lock acquisition, so there
        is no TOCTOU window: two near-simultaneous submits cannot both observe a
        free slot and both be admitted. *factory* is only called once a slot is
        secured, so a rejected submit never spawns anything. Raises
        :class:`JobCapacityError` when the cap is reached; anything *factory*
        raises (a missing executable, say) propagates and nothing is registered.
        """
        with self._lock:
            cap = self.cap_for(kind)
            live = [j for j in self._jobs.values()
                    if j.state == "running" and j.kind == kind]
            if len(live) >= cap:
                detail = ", ".join(f"{j.id} ({j.label[:40]})" for j in live)
                raise JobCapacityError(
                    f"Background {kind} job limit reached ({len(live)}/{cap}). "
                    f"Running: {detail}. Kill one with kill_shell_job before "
                    "starting another.")
            job = factory()
            if job.kind != kind:
                raise JobError(
                    f"factory produced a '{job.kind}' job but the slot was "
                    f"reserved for '{kind}'")
            self._jobs[job.id] = job
            self._prune_locked()
            return job

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def ids(self, kind: Optional[str] = None) -> list:
        with self._lock:
            return [j.id for j in self._jobs.values()
                    if kind is None or j.kind == kind]

    def list_status(self, kind: Optional[str] = None) -> list:
        with self._lock:
            jobs = [j for j in self._jobs.values()
                    if kind is None or j.kind == kind]
        return [j.status() for j in jobs]

    def drain_finished(self, kind: Optional[str] = None) -> list:
        """Status of every job that finished since the last drain, then mark them.

        For a caller that absorbs completions at a turn boundary instead of
        polling a known id. Each finished job is returned by exactly one drain.
        Draining does NOT remove the job, so a later poll-by-id still works.
        """
        with self._lock:
            jobs = [j for j in self._jobs.values()
                    if j.state != "running" and not j.drained
                    and (kind is None or j.kind == kind)]
            for job in jobs:
                job.drained = True
        # status() takes each job's own lock - do it outside the registry lock.
        return [j.status() for j in jobs]

    def _prune_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.state != "running"]
        excess = len(finished) - self.keep_finished
        if excess <= 0:
            return
        # Drop jobs somebody has already collected FIRST, so a completion that
        # no drain has seen yet survives as long as possible. (Stable sort keeps
        # oldest-first within each group.)
        ordered = sorted(finished, key=lambda j: not j.drained)
        for job in ordered[:excess]:
            self._jobs.pop(job.id, None)
            if not job.drained:
                # The table must stay bounded, so once EVERY retained completion
                # is undrained something has to go. That means a drain-based
                # consumer will never see this one: count it rather than let it
                # vanish silently (a lost completion looks identical to "nothing
                # finished", which is exactly the failure we must not hide).
                self.dropped_undrained += 1

    def shutdown_all(self) -> int:
        """Kill every running job. Returns how many were killed. Never raises."""
        killed = 0
        for job in self.running():
            try:
                job.kill()
                killed += 1
            except Exception:
                pass
        return killed


_registry: Optional[JobRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    """The process-wide job registry, created on first use.

    Process-wide rather than per-session: the tools are dispatched as plain
    functions that receive only *cwd*, with no session handle to hang a registry
    off. That is not a trust boundary - only a full-capability (non-restricted)
    session can reach these tools at all, and such a session already has
    ``run_shell``, i.e. arbitrary code execution.
    """
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry


def reset_registry() -> None:
    """Kill everything and drop the singleton (used by tests)."""
    global _registry
    with _registry_lock:
        reg, _registry = _registry, None
    if reg is not None:
        reg.shutdown_all()
        try:
            atexit.unregister(reg.shutdown_all)
        except Exception:
            pass
