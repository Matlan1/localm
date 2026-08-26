# SPDX-License-Identifier: AGPL-3.0-or-later
"""Background job registry for the coder plugin.

A GENERIC job registry (``JobRegistry``) plus the concrete :class:`ShellJob` and
:class:`AgentJob`. ``BackgroundJob`` knows only about a lifecycle (running ->
done/killed/failed), an OPAQUE per-kind result payload, a kill hook, and a
bounded output buffer; everything process-specific lives in ``ShellJob``.

The registry offers three things a non-shell caller needs:

* **Per-kind caps.** ``kind_caps`` holds a separate ceiling per job kind, since
  shell jobs and agent jobs exhaust different resources.
* **Atomic check-and-insert.** The cap check and the spawn happen under one lock
  acquisition, so two near-simultaneous submits can never both observe a free
  slot and both be admitted.
* **Drain, not just poll.** ``drain_finished()`` returns everything that finished
  since the last drain, for a caller that absorbs completions at a turn boundary
  rather than polling a known id.

Concurrency style, shared with the GUI coder sessions (``sessions.py``): daemon
threads plus a bounded buffer that drops the oldest entry when full.

This table is NOT ``localm/plugins/builtin/jobs/``, which owns the word "job" in
the user-facing product (a GUI Jobs tab, ``localm job ...``, ``/api/jobs``). That
plugin is schedule-centric (every ``Job`` carries a ``schedule_kind`` validated
against ``SCHEDULE_KINDS``), stores durable user data under ``<data dir>/jobs/``,
is an optional install, and its scheduler starts only when the plugin is loaded
under a running event loop - so a plain ``localcoder`` REPL, which has no event
loop, would silently never run work routed through it. Jobs here are in-process
threads scoped to one session and killed at exit.

Naming: nothing here is a bare "job" - ``check_shell_job`` / ``kill_shell_job``
read as shell job control (jobs/bg/fg/kill %1), not as a schedule entry. A
user-facing listing command is ``/bg``, never ``/jobs``.

Three invariants are load-bearing:

1. **No pid-reuse window.** ``Popen.poll()`` is called ONLY under the job lock,
   and this module is the single reaping call site for its children. So a
   ``poll() is None`` observed under that lock proves the child is still
   unreaped: on POSIX it is a live process or a zombie (either way the pid is
   not reusable), and on Windows the process handle is still held (which
   reserves the pid). Killing by pid is therefore safe at that instant. psutil's
   ``create_time`` is pinned alongside the pid as a second, independent check,
   but correctness does not depend on it: psutil is an OPTIONAL dependency here,
   not a core one.
2. **Kill reaps the TREE, and says so only once it has CHECKED.** A build or dev
   server spawns children; killing only the direct child strands them. POSIX
   gets its own session/process group (``start_new_session``) and is killed with
   ``killpg``; Windows uses ``taskkill /F /T``, which walks the child tree.
   Nothing is ever killed by port or by image name - only by a pid just proven
   to be still ours. Tree-wide delivery is not the same as the tree being DEAD:
   the direct child exiting is all ``Popen.poll()`` can ever show, and a
   descendant that handles the signal and then hangs would satisfy it while
   still holding its port. So the tree is pinned before the kill
   (``_snapshot_tree``) and re-checked after (``_verify_tree_gone``), survivors
   are killed by their pinned identity, and anything left - or an install where
   psutil cannot tell - is reported as a warning rather than folded into a flat
   "killed".
3. **Bounded memory.** A chatty process cannot grow the buffer without limit,
   and anything dropped is COUNTED and reported, never silently discarded.
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
from typing import Any, Callable, Optional

# Per-kind ceilings on jobs running at once. A kind with no entry falls back to
# _DEFAULT_CAP. Exceeding a cap raises rather than queueing.
_KIND_CAPS: dict[str, int] = {
    "shell": 4,
    # Each background sub-agent holds a model context. child_limit is the
    # authoritative gate; this entry caps the half the registry can see.
    "agent": 2,
}
_DEFAULT_CAP = 4

# Per-stream output cap, in characters.
_RING_MAX_CHARS = 256_000

# Retained finished jobs, PER KIND, pruned oldest-first. Finished jobs never
# count toward a cap and stay queryable until pruned.
_KEEP_FINISHED = 16

_POLL_INTERVAL = 0.05     # seconds between liveness polls
_READ_CHUNK = 65_536      # bytes per raw pipe read
_KILL_GRACE = 3.0         # seconds to wait after a graceful terminate
_DRAIN_GRACE = 2.0        # seconds to wait for reader threads at finish


# Sentinel for "do not filter by owner" in list_status / dropped_for. Distinct
# from None, which is a real owner value: a job started outside any agent.
_ANY_OWNER = object()


class JobError(RuntimeError):
    """Base class for job-registry errors."""


class JobCapacityError(JobError):
    """Raised when a per-kind concurrent-job cap would be exceeded."""


# --------------------------------------------------------------------------- #
#  Bounded output buffer                                                       #
# --------------------------------------------------------------------------- #

class RingBuffer:
    """A capped FIFO of text chunks, dropping the OLDEST when full.

    Stores raw chunks rather than lines, so output with no newlines (a ``\\r``
    progress bar, a binary-ish blob) is capped like any other. ``dropped`` counts
    the characters evicted, so a caller can say how much was lost instead of
    presenting a truncated tail as if it were everything.
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
            # A chunk bigger than the whole cap is trimmed from the FRONT,
            # keeping the tail.
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
#  pid identity helpers                                                       #
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


def _describe(job) -> str:
    """Name a job for an error message, without trusting it to be well-formed.

    Never raises. Reading ``.id`` / ``.label`` is a call into someone else's
    object, and this runs on the atexit path where a reported failure must not
    become a traceback.
    """
    try:
        return f"{job.id} ({str(job.label)[:60]})"
    except Exception:
        return "<unidentifiable job>"


def _decode(raw) -> str:
    """Captured subprocess bytes as a one-line string, for a warning message."""
    if not raw:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return " ".join(raw.split())


def _still_the_same_process(pid: int, create_time: Optional[float]) -> bool:
    """Is *pid* still the process we started?

    Returns True when it cannot tell (no psutil, or an unexpected probe error):
    the caller's lock-plus-unreaped-child argument is the primary guarantee and
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
        # Tolerance for platform clock granularity on create_time.
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

    The record is kind-agnostic::

        {id, kind, label, state, started_at, finished_at, result, error, warnings}

    ``result`` is an OPAQUE per-kind payload (a shell job puts ``exit_code``
    there; an agent job puts its own summary), so adding a kind never means
    reshaping the table. Subclasses implement :meth:`_poll`, :meth:`_terminate`
    and optionally :meth:`_result_for` / :meth:`_drain`.
    """

    kind = "job"

    def __init__(self, label: str, owner: Optional[str] = None) -> None:
        self.id = "job_" + uuid.uuid4().hex[:8]
        self.label = label
        # Which agent session started this, or None. Never returned by status();
        # it lets a caller narrow a listing to its own jobs.
        self.owner = owner
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.state = "running"          # running | done | killed | failed
        # Opaque, per-kind terminal payload. None while running.
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        # Non-fatal problems, reported by status().
        self.warnings: list[str] = []
        # Set by JobRegistry.drain_finished once this completion has been handed
        # to its owner. Pruning prefers jobs already collected.
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

    def _snapshot_tree(self) -> None:
        """Record the descendants alive BEFORE the kill. Under ``self._lock``.

        Must run while the job is still alive: once the direct child exits and is
        reaped, its children are re-parented and can no longer be found from its
        pid, so a post-hoc lookup would report a clean tree it never saw.
        Default: a kind with no process tree has nothing to record.
        """

    def _verify_tree_gone(self) -> None:
        """Confirm the snapshot's descendants really died. Under ``self._lock``.

        Called once the direct child has exited. Appends a warning rather than
        raising: the direct child IS dead by then, so the honest report is
        "killed, but N descendants survived", not a blanket kill failure.
        Default: nothing to verify.
        """

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
            self.warnings.append(
                "output readers did not reach EOF within "
                f"{_DRAIN_GRACE:g}s - a detached child may still hold the pipe; "
                "the captured output may be incomplete")
        # Publication order matters: state is assigned LAST, so any thread that
        # sees a non-running state also sees a fully populated record.
        self.result = result
        self.error = error
        self.finished_at = time.time()
        self.state = state

    def kill(self) -> str:
        """Stop the job and its process tree. Returns a human-readable outcome."""
        with self._lock:
            if self.state != "running":
                return f"already {self.state}"
            # Reaped here, under the lock, so the pid cannot be recycled between
            # this check and the kill below.
            value = self._poll()
            if value is not None:
                self._finish("done", self._result_for(value))
                return "already finished"

            # Pin the tree before reaping: descendants are unreachable from the
            # child's pid once it is reaped.
            self._snapshot_tree()
            self._terminate(force=False)
            value = self._wait_for_exit(_KILL_GRACE)
            if value is None:
                self._terminate(force=True)
                value = self._wait_for_exit(_KILL_GRACE)
            if value is None:
                msg = "the process did not exit after a forced kill"
                self.warnings.append(msg)
                self._finish("failed", None, error=msg)
                return "kill FAILED - the process is still running"
            # _wait_for_exit observes only the DIRECT child, so confirm the tree
            # died before reporting a kill.
            self._verify_tree_gone()
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

    *argv* is an already-routed launch form - the caller decides shell vs
    argument-list mode (``tools/shell.py:_shell_argv``), so the background path
    and the blocking ``run_shell`` path make that security decision in exactly
    one place. Usually an argument list; a STRING is the Windows shell route's
    raw command line, which ``CreateProcess`` receives verbatim (an argv list
    would be re-quoted by ``list2cmdline`` in syntax cmd.exe misreads - see
    ``tools/base.py:platform_shell``).

    Its opaque result payload is ``{"exit_code": int}``.
    """

    kind = "shell"

    def __init__(self, argv: list | str, cwd: Path, *, label: str,
                 env: Optional[dict] = None,
                 max_chars: int = _RING_MAX_CHARS,
                 owner: Optional[str] = None) -> None:
        super().__init__(label, owner=owner)
        self.stdout = RingBuffer(max_chars)
        self.stderr = RingBuffer(max_chars)

        kwargs: dict = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            # Never blocks on input, and never takes the coder's stdin.
            "stdin": subprocess.DEVNULL,
            # Raw FileIO pipes: a read() returns whatever is available.
            "bufsize": 0,
            "env": env,
        }
        if sys.platform == "win32":
            # Own group, so a Ctrl+C in the coder console does not reach it.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Own session/process group, so killpg reaps the whole tree.
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(argv, **kwargs)
        self.pid = self._proc.pid
        self._create_time = _process_create_time(self.pid)
        # (pid, create_time) of every descendant seen just before a kill. None
        # means "not looked at / could not look", which is not the same as an
        # empty list. The reason is kept alongside.
        self._tree_snapshot: Optional[list] = None
        self._tree_unverified_reason: Optional[str] = None
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

        Touches no job state except ``warnings``, and must never take
        ``self._lock``: a bounded join under that lock could deadlock.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = stream.read(_READ_CHUNK)
                if not chunk:
                    break
                ring.append(decoder.decode(chunk))
        except (OSError, ValueError) as e:
            # The pipe broke in a way a normal EOF does not cover; output may be
            # missing.
            self.warnings.append(f"{name} reader stopped early: {type(e).__name__}: {e}")
        finally:
            ring.append(decoder.decode(b"", final=True))
            try:
                stream.close()
            except Exception:
                # Every byte is already in the ring, and Popen closes its own
                # pipes at collection.
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
            # The pid is no longer ours. Signal via the Popen handle, which is
            # bound to the object we launched.
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
        # taskkill /T walks the child tree; /F is its only reliable mode.
        try:
            done = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.pid)],
                capture_output=True, timeout=10,
            )
            if done.returncode == 0:
                return
            # taskkill reports its ordinary failures by exit code plus a stderr
            # message rather than raising: exit 128 when the direct child is
            # already gone, exit 255 when a descendant exits during a /T sweep.
            # The fallback sweep below and _verify_tree_gone() decide what
            # actually survived.
            detail = (_decode(done.stderr) or _decode(done.stdout)
                      or "no output")
            self.warnings.append(
                f"taskkill exited {done.returncode} ({detail}) - the process "
                "tree may not be fully dead; falling back to psutil/handle kill")
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
        # Never signal our own group: if start_new_session did not take, the
        # child shares it and killpg would take down localm itself.
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

    # -- tree verification ---------------------------------------------------- #

    def _snapshot_tree(self) -> None:
        """Pin every live descendant as ``(pid, create_time)`` before signalling.

        Must be taken while the child is still alive: once it exits and is reaped
        its children are re-parented (to init on POSIX, to nothing followable on
        Windows), so they are unreachable from our pid afterwards. The
        create_time pin is what makes killing a survivor safe later - the same
        identity check the direct child uses, so a recycled pid can never be
        signalled.
        """
        self._tree_snapshot = None
        self._tree_unverified_reason = None
        try:
            import psutil
        except Exception:
            # psutil is optional. Left as None, not [], so the verifier reports
            # "could not check" rather than an empty tree.
            self._tree_unverified_reason = "psutil is not installed"
            return
        try:
            self._tree_snapshot = [
                (child.pid, child.create_time())
                for child in psutil.Process(self.pid).children(recursive=True)
            ]
        except Exception as e:
            # Nothing to pin. The reason is kept so the warning names the real
            # one.
            self._tree_snapshot = None
            self._tree_unverified_reason = (
                f"the process tree could not be read: {type(e).__name__}")

    def _surviving_descendants(self) -> Optional[list]:
        """Snapshot entries still alive under their ORIGINAL identity.

        ``None`` means it could not look; ``[]`` means it looked and the tree is
        clean. The two are never collapsed, which would turn "unverified" into
        "verified good".
        """
        if self._tree_snapshot is None:
            return None
        try:
            import psutil
        except Exception:
            return None
        alive = []
        for pid, created in self._tree_snapshot:
            try:
                proc = psutil.Process(pid)
                if abs(proc.create_time() - created) < 0.05:
                    alive.append(proc)
            except Exception:
                # Gone, or unreadable. Neither counts as a survivor.
                continue
        return alive

    def _verify_tree_gone(self) -> None:
        survivors = self._surviving_descendants()
        if survivors is None:
            reason = self._tree_unverified_reason or "the tree could not be read"
            self.warnings.append(
                f"could not verify the process tree died ({reason}): the kill "
                "was delivered tree-wide, but a descendant that ignored it "
                "would go unnoticed here")
            return
        if not survivors:
            return
        # Every entry is pinned by (pid, create_time) to a process seen as our
        # own descendant, so this cannot signal a recycled pid.
        for proc in survivors:
            try:
                proc.kill()
            except Exception:
                # Already gone, or not ours to signal; the re-check below is the
                # authority on what survived.
                pass
        deadline = time.time() + _KILL_GRACE
        remaining = survivors
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            remaining = self._surviving_descendants()
            if not remaining:
                return
        if remaining:
            pids = ", ".join(str(p.pid) for p in remaining[:10])
            self.warnings.append(
                f"{len(remaining)} descendant process(es) survived the kill "
                f"(pid(s) {pids}) - the direct child is gone but its tree is "
                "not")

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
#  Agent job                                                                   #
# --------------------------------------------------------------------------- #

class AgentJob(BackgroundJob):
    """A background sub-agent.

    The child Agent is built by the CALLER on the parent's thread and handed in
    already constructed, so a construction error (bad role, unreadable preload)
    surfaces synchronously in the spawn tool's result instead of turning into a
    job that immediately fails. This job only owns RUNNING it.

    ``result`` payload: ``{"summary": str, "turns": int}``.

    NOT PREEMPTIBLE. ``Agent.run_task`` is a blocking call with no cooperative
    cancellation anywhere in the agent package (the only interruption path is a
    KeyboardInterrupt in the INTERACTIVE loop, which a worker thread cannot
    raise). ``_terminate`` therefore cannot stop a turn already in flight: it
    records a warning and marks intent, and the daemon thread dies with the
    process. There is no ``kill_agent_job`` tool.
    """

    kind = "agent"

    def __init__(self, child: Any, task: str, *, label: str,
                 token: Any = None, finalize: Any = None,
                 owner: Optional[str] = None) -> None:
        super().__init__(label, owner=owner)
        self._child = child
        self._task = task
        # Optional teardown run on THIS job's worker thread once the child stops:
        # ``finalize(child) -> dict`` merged into the terminal payload. It must
        # never touch parent state; the parent folds the payload in at its own
        # turn boundary.
        self._finalize = finalize
        # The child_limit slot this job holds. Released on FINISH, not on submit.
        self._token = token
        self._outcome: Optional[dict] = None
        self._runner = threading.Thread(
            target=self._run, name=f"bgagent-{self.id}", daemon=True)
        self._runner.start()
        # Must come last; without it the job never leaves the running state.
        self.start_watcher()

    @property
    def child(self) -> Any:
        """The child Agent, for the parent's turn-boundary absorption."""
        return self._child

    def _run(self) -> None:
        """Body of the worker thread. Never touches parent state - the parent
        absorbs at ITS turn boundary, so there is no cross-thread mutation of
        the parent's _changed_files / _error_trace from here."""
        try:
            text = self._child.run_task(self._task)
            # run_task RETURNS its failure message rather than raising, so record
            # the child's own verdict.
            outcome = {"summary": text, "turns": getattr(self._child, "turns", 0),
                       "ok": bool(getattr(self._child, "last_run_ok", True))}
        except Exception as exc:                      # noqa: BLE001 - recorded
            # Becomes the job's terminal error.
            outcome = {"error": f"{type(exc).__name__}: {exc}"}
        # Teardown runs even when the child failed, so its worktree is removed.
        if self._finalize is not None:
            try:
                extra = self._finalize(self._child)
                if extra:
                    outcome.update(extra)
            except Exception as exc:                  # noqa: BLE001 - surfaced
                # Recorded as a visible warning.
                self.warnings.append(f"teardown failed: {type(exc).__name__}: {exc}")
        with self._lock:
            self._outcome = outcome

    def _poll(self):
        return self._outcome

    def _result_for(self, poll_value) -> Optional[dict]:
        # Neutralise the child's text at the single point where it becomes
        # readable by the parent loop.
        from .provenance import neutralise
        payload = {
            "summary": neutralise(poll_value.get("summary") or ""),
            "turns": poll_value.get("turns", 0),
        }
        if poll_value.get("error"):
            payload["error"] = poll_value["error"]
        # Isolation facts produced by finalize(), carried through verbatim. The
        # diff is NOT neutralised; the parent renders it in a labelled block.
        for key in ("branch", "base", "file_count", "diff", "worktree",
                    "cleanup_warning"):
            if key in poll_value:
                payload[key] = poll_value[key]
        return payload

    def _terminate(self, *, force: bool) -> None:
        # A blocking run_task cannot be preempted; record that instead.
        self.warnings.append(
            "a background sub-agent cannot be stopped mid-turn; it will finish "
            "or die with the process")

    def _finish(self, state: str, result: Optional[dict],
                error: Optional[str] = None) -> None:
        # Promote a child-raised exception to the job's own error field.
        if result and result.get("error") and not error:
            error, state = result["error"], "failed"
        try:
            super()._finish(state, result, error)
        finally:
            # Release the shared child budget; release() is idempotent and
            # tolerates None.
            from .child_limit import release
            release(self._token)
            self._token = None


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

class JobRegistry:
    """Tracks background jobs, caps how many run at once per kind, reaps at exit.

    Kind-agnostic: it holds :class:`BackgroundJob` instances, so a background
    sub-agent registers here too and inherits the caps, the lookup, the drain,
    and the shutdown behaviour.
    """

    def __init__(self, kind_caps: Optional[dict] = None,
                 default_cap: int = _DEFAULT_CAP,
                 keep_finished: int = _KEEP_FINISHED) -> None:
        self.kind_caps = dict(_KIND_CAPS if kind_caps is None else kind_caps)
        self.default_cap = default_cap
        # Per kind: the table's total bound is this times the number of kinds
        # that have finished work.
        self.keep_finished = keep_finished
        # Completions evicted before any drain collected them, per kind.
        # Cumulative and never reset.
        self.dropped_undrained_by_kind: dict[str, int] = {}
        # The same losses awaiting a report, consumed by take_dropped_undrained
        # so each loss is reported once.
        self._unreported_drops: dict[str, int] = {}
        # The same cumulative losses, split by owner as well as kind.
        self._dropped_by_owner: dict[tuple, int] = {}
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        # Background jobs must not outlive the localm process.
        atexit.register(self.shutdown_all)

    @property
    def dropped_undrained(self) -> int:
        """Total completions evicted before any drain collected them.

        The sum across kinds. Prefer ``dropped_undrained_by_kind`` when the
        number is going in front of a person: the kinds do not mean the same
        thing (see that attribute).
        """
        return sum(self.dropped_undrained_by_kind.values())

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
        A factory that returns the WRONG kind has already produced a live job, so
        that job is STOPPED before the :class:`JobError` is raised - otherwise it
        would leak somewhere nothing can reach it.
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
            if job.kind == kind:
                self._jobs[job.id] = job
                self._prune_locked()
                return job

        # Kind mismatch. The factory has already produced a LIVE job, and an
        # unregistered job is invisible to kill_shell_job and shutdown_all, so
        # stop it here. Stopped outside the registry lock, because a kill can
        # take a grace period.
        detail = ""
        try:
            outcome = job.kill()
            if outcome.startswith("kill FAILED"):
                detail = f" The stray job could not be stopped: {outcome}."
        except Exception as e:            # noqa: BLE001 - folded into the error
            # Folded into the error raised below.
            detail = (f" The stray job could not be stopped: "
                      f"{type(e).__name__}: {e}.")
        raise JobError(
            f"factory produced a '{job.kind}' job but the slot was "
            f"reserved for '{kind}'.{detail}")

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def ids(self, kind: Optional[str] = None) -> list:
        with self._lock:
            return [j.id for j in self._jobs.values()
                    if kind is None or j.kind == kind]

    def list_status(self, kind: Optional[str] = None, owner=_ANY_OWNER) -> list:
        """Every tracked job, optionally narrowed to one *kind* and/or *owner*.

        *owner* defaults to the "do not filter" sentinel. Pass a real owner id
        (or None) to get exactly that owner's jobs - see BackgroundJob.owner.
        """
        with self._lock:
            jobs = [j for j in self._jobs.values()
                    if (kind is None or j.kind == kind)
                    and (owner is _ANY_OWNER or j.owner == owner)]
        return [j.status() for j in jobs]

    def dropped_for(self, owner=_ANY_OWNER) -> dict:
        """Uncollected completions evicted from the table, per kind.

        The per-owner view of ``dropped_undrained_by_kind``. Cumulative and
        never reset, like that total: it is the standing session diagnostic, not
        the report-once channel (that is take_dropped_undrained). Kinds with no
        losses are omitted, so an empty dict means nothing was discarded rather
        than nothing was looked at.
        """
        with self._lock:
            if owner is _ANY_OWNER:
                return dict(self.dropped_undrained_by_kind)
            out: dict = {}
            for (job_owner, kind), n in self._dropped_by_owner.items():
                if job_owner == owner:
                    out[kind] = out.get(kind, 0) + n
            return out

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
        """Trim retained completions. Called from submit, which is the only path
        that GROWS the table - so the table stays bounded even though a job
        finishing does not itself prune (between submits the count can sit at
        keep_finished plus whatever finished since, itself bounded by the caps).

        Each KIND is budgeted separately, so a kind nobody drains cannot evict
        another kind's uncollected completion (see _KEEP_FINISHED).
        """
        by_kind: dict[str, list] = {}
        for job in self._jobs.values():
            if job.state != "running":
                by_kind.setdefault(job.kind, []).append(job)

        for kind, finished in by_kind.items():
            excess = len(finished) - self.keep_finished
            if excess <= 0:
                continue
            # Drop already-collected jobs first; the stable sort keeps
            # oldest-first within each group.
            ordered = sorted(finished, key=lambda j: not j.drained)
            for job in ordered[:excess]:
                self._jobs.pop(job.id, None)
                if not job.drained:
                    # This completion will never reach a drain-based consumer:
                    # count it and queue it for report.
                    self.dropped_undrained_by_kind[kind] = (
                        self.dropped_undrained_by_kind.get(kind, 0) + 1)
                    self._unreported_drops[kind] = (
                        self._unreported_drops.get(kind, 0) + 1)
                    key = (job.owner, kind)
                    self._dropped_by_owner[key] = (
                        self._dropped_by_owner.get(key, 0) + 1)

    def take_dropped_undrained(self, kind: Optional[str] = None) -> int:
        """Uncollected completions lost since the last call, and RESET the count.

        The reporting half of ``dropped_undrained``. A drain-based consumer calls
        this next to its ``drain_finished`` and tells the user what it lost,
        because from the consumer's side a discarded completion is
        indistinguishable from "nothing finished". Consumed exactly once, like
        the drain itself, so a turn-boundary caller warns per loss instead of
        every turn forever. The cumulative ``dropped_undrained`` total is NOT
        reset here and stays readable all session (``/bg`` shows it).
        """
        with self._lock:
            if kind is None:
                total = sum(self._unreported_drops.values())
                self._unreported_drops.clear()
                return total
            return self._unreported_drops.pop(kind, 0)

    def shutdown_all(self) -> int:
        """Kill every running job. Returns how many were killed. Never raises.

        This is the atexit hook, so it is the LAST chance to say anything: a job
        that could not be killed is a live process the coder started outliving
        localm. ``kill()`` reports that failure by RETURN VALUE ("kill FAILED
        - ..."), not by raising, so counting every non-raising call as a success
        would report the orphan case as a clean shutdown. Failures are counted
        out and printed instead.
        """
        running = self.running()
        if running:
            # Each kill can take a grace period, so announce the pause.
            self._report_at_exit(
                f"localm: stopping {len(running)} background job(s) started by "
                "the coder...")

        killed = 0
        failures: list[str] = []
        for job in running:
            # The whole body is inside the try: this runs from an atexit hook
            # and must never raise.
            try:
                outcome = job.kill()
                if outcome.startswith("kill FAILED"):
                    failures.append(f"{job.id} ({job.label[:60]}): {outcome}")
                else:
                    killed += 1
            except Exception as e:        # noqa: BLE001 - reported, not swallowed
                # Recorded rather than raised out of the atexit hook.
                failures.append(f"{_describe(job)}: {type(e).__name__}: {e}")

        for failure in failures:
            self._report_at_exit(
                f"localm: WARNING - a background job may still be running after "
                f"exit: {failure}")
        return killed

    @staticmethod
    def _report_at_exit(message: str) -> None:
        """Print a teardown message, tolerating interpreter shutdown.

        Runs from atexit, where ``sys.stderr`` can already be closed or replaced
        by None. Failing to PRINT must not become an exception escaping the hook,
        so it tries and gives up only when the stream itself is gone.
        """
        try:
            stream = sys.stderr
            if stream is None:
                return
            print(message, file=stream, flush=True)
        except Exception:
            # The stream is gone during interpreter teardown.
            pass


_registry: Optional[JobRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    """The process-wide job registry, created on first use.

    Process-wide rather than per-session: the tools are dispatched as plain
    functions that receive only *cwd*, with no session handle to hang a registry
    off. This is not a trust boundary - only a full-capability (non-restricted)
    session can reach these tools at all, and such a session already has
    ``run_shell``.
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
