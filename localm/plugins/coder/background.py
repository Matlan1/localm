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

Why this is NOT the builtin jobs plugin
---------------------------------------
``localm/plugins/builtin/jobs/`` already exists and owns the word "job" in the
user-facing product: a GUI Jobs tab, ``localm job ...``, ``/api/jobs``. This
table is a DIFFERENT thing that happens to share a noun, and the five reasons
are recorded here so nobody later "consolidates" the two (which would break the
CLI case outright) or flags this module as duplicating shipped infrastructure.
Each was checked against the source, not assumed:

1. That plugin is SCHEDULE-CENTRIC. Every ``Job`` carries ``schedule_kind``
   ("interval" | "cron") and ``store.py`` validates it against ``SCHEDULE_KINDS``.
   A one-shot background command has no schedule; we would have to invent one.
2. It is DURABLE USER DATA, stored under ``<data dir>/jobs/`` (``store.jobs_dir``)
   and, per ``docs/jobs.md``, its "recorded results are saved in every session
   mode, including privacy". An ephemeral in-session thread is not user data and
   must not land in the user's jobs store or the Jobs tab.
3. It is an OPTIONAL plugin (``localm plugin install jobs``; docs/jobs.md line 5).
   A core coder capability cannot require an optional plugin to be present.
4. Decisively: its scheduler "starts only when the plugin is loaded under a
   running event loop" (``jobs/plug.py``), i.e. inside ``localm gui`` / ``serve``.
   A plain ``localcoder`` REPL has no event loop, so background work routed
   through it would SILENTLY NEVER RUN - exactly the case this module exists for.
5. Lifetimes are opposite: theirs survive restarts, ours are in-process threads
   scoped to one session and killed at exit.

The existing integration also runs the other way (a jobs task_kind of "coder"
runs a coder agent). This is coder -> background work, the reverse direction.

Naming: nothing here is a bare "job" - ``check_shell_job`` / ``kill_shell_job``
read as shell job control (jobs/bg/fg/kill %1), not as a schedule entry. A
user-facing listing command should be ``/bg``, never ``/jobs``; "tasks" was
considered and rejected because ``task`` is already the coder's own core noun
(``localcoder [TASK]``, ``run_task``), which would be a worse ambiguity inside
one component.

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
2. **Kill reaps the TREE, and says so only once it has CHECKED.** A build or dev
   server spawns children; killing only the direct child strands them. POSIX
   gets its own session/process group (``start_new_session``) and is killed with
   ``killpg``; Windows uses ``taskkill /F /T``, which walks the child tree. We
   never kill by port or by image name - only by a pid we have just proven is
   still ours. Delivery being tree-wide is not the same as the tree being DEAD,
   though: the direct child exiting is all ``Popen.poll()`` can ever show, and a
   descendant that handles the signal and then hangs would satisfy it while
   still holding its port. So the tree is pinned before the kill
   (``_snapshot_tree``) and re-checked after (``_verify_tree_gone``), survivors
   are killed by their pinned identity, and anything left - or an install where
   psutil cannot tell us - is reported as a warning rather than folded into a
   flat "killed".
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
from typing import Any, Callable, Optional

# Per-kind ceilings on jobs running at once. Separate numbers because the kinds
# exhaust different resources: shell jobs are OS processes the model started and
# may forget about, agent jobs each hold a model context. A kind with no entry
# falls back to _DEFAULT_CAP. Rejecting past the cap is a CLEAR error the model
# can act on, never a silent queue that looks like it started.
_KIND_CAPS: dict[str, int] = {
    "shell": 4,
    # 2, not the default 4: each background sub-agent holds a model context, and
    # the box's practical ceiling is 2 resident models. This MUST be declared -
    # an unlisted kind silently falls back to _DEFAULT_CAP=4, which would be
    # double the intended ceiling with no error at all. The authoritative gate is
    # child_limit (it also counts C2's synchronous parallel children, which never
    # reach this registry); this entry is the defensive backstop for the half the
    # registry can see.
    "agent": 2,
}
_DEFAULT_CAP = 4

# Per-stream output cap. Chars, not lines: a progress bar that only emits '\r'
# never produces a line, so a line-based cap would let one "line" grow forever.
_RING_MAX_CHARS = 256_000

# Finished jobs stay queryable (that is the whole point of check_shell_job after
# completion) but are pruned oldest-first so a long session cannot accumulate
# them without limit. Finished jobs never count toward a cap.
#
# PER KIND, not one global budget. That distinction was invisible while shell was
# the only kind, but the kinds have opposite consumers: the agent kind is DRAINED
# at a turn boundary, while nothing drains the shell kind at all (both drain call
# sites filter kind="agent"), so shell completions stay undrained for the life of
# the process. Sharing one budget, they would eventually evict an undrained
# SUB-AGENT completion - and since absorption is drain-only, that child's summary,
# branch and diff would be unrecoverable - purely because unrelated shell commands
# finished. Budgeting per kind means one kind can never crowd out another.
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


def _describe(job) -> str:
    """Name a job for an error message, without trusting it to be well-formed.

    Used on the atexit path, where the job we are describing is the one that
    just misbehaved: reading ``.id`` / ``.label`` is a call into someone else's
    object and must not be what turns a reported failure into a traceback.
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
            # Not fatal: a detached grandchild can hold the pipe open after the
            # job itself exited. Say so rather than presenting a possibly
            # incomplete tail as the full output.
            self.warnings.append(
                "output readers did not reach EOF within "
                f"{_DRAIN_GRACE:g}s - a detached child may still hold the pipe; "
                "the captured output may be incomplete")
        # PUBLICATION ORDER IS LOAD-BEARING: state goes LAST. drain_finished()
        # selects on state without taking this job's lock, so any thread that
        # sees state != "running" must already be able to see a fully populated
        # record. Assigning state first would let a drain collect a completion
        # with result still None. Do not reorder these four lines.
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

            # Pin the tree while it still exists - after the child is reaped its
            # descendants are unreachable from its pid (see _snapshot_tree).
            self._snapshot_tree()
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
            # _wait_for_exit only ever observes the DIRECT child, so reaching
            # here proves delivery, not tree-wide termination: a descendant that
            # handles SIGTERM and then hangs satisfies the loop above while still
            # holding its port. Invariant 2 claims the TREE dies, so check it
            # before reporting "killed" instead of assuming it.
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
        # (pid, create_time) of every descendant seen just before a kill. None
        # means "not looked at / could not look", which is NOT the same as the
        # empty list "looked, found none" - see _verify_tree_gone. The reason is
        # kept alongside so the warning states the real one.
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
                # Safe to ignore, and this is why: the loop above only exits at
                # EOF or on a pipe error it has already recorded as a warning, so
                # every byte this stream will ever produce is in the ring by now.
                # Closing is pure resource cleanup, and Popen closes its own
                # pipes at collection regardless, so a failure here loses nothing
                # and hides nothing.
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
            done = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.pid)],
                capture_output=True, timeout=10,
            )
            if done.returncode == 0:
                return
            # taskkill reports its ORDINARY failures by exit code plus a stderr
            # message, never by raising here: "process not found" when the direct
            # child exited between kill()'s poll and this call (its detached
            # grandchildren then survive), or access-denied against a
            # higher-integrity process. Returning unconditionally would report a
            # tree kill that never ran, and would skip the fallback sweep below.
            # The POSIX path already warns when killpg fails; match it.
            #
            # The same "process not found" race can also land on a NON-root
            # descendant: /T snapshots the tree once and then terminates each
            # pid it found in turn, so under heavy scheduler contention one
            # descendant can legitimately exit in the gap between that snapshot
            # and taskkill reaching its specific pid. Windows then reports THAT
            # pid as "There is no running instance of the task" (exit 255 for a
            # multi-pid /T call, versus 128 for the single direct-child case
            # above) even though the rest of the tree, including that pid, ends
            # up fully dead - the fallback sweep below and _verify_tree_gone()'s
            # independent (pid, create_time) check are what actually confirm
            # that, not this exit code. Observed for real under `pytest -n
            # auto`: exit 255 naming a grandchild pid, tree fully dead after.
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

    # -- tree verification (invariant 2 is about the TREE, not the handle) ---- #

    def _snapshot_tree(self) -> None:
        """Pin every live descendant as ``(pid, create_time)`` before signalling.

        Taken while the child is still alive on purpose: once it exits and is
        reaped its children are re-parented (to init on POSIX, to nothing we can
        follow on Windows), so they are unreachable from our pid afterwards. The
        create_time pin is what makes killing a survivor safe later - it is the
        same identity check the direct child uses, so a recycled pid can never be
        signalled.
        """
        self._tree_snapshot = None
        self._tree_unverified_reason = None
        try:
            import psutil
        except Exception:
            # psutil is an OPTIONAL dependency, so this is the ordinary case on a
            # core install, not an error. Left as None (not []) so the verifier
            # says "could not check" instead of claiming a tree it never looked at.
            self._tree_unverified_reason = "psutil is not installed"
            return
        try:
            self._tree_snapshot = [
                (child.pid, child.create_time())
                for child in psutil.Process(self.pid).children(recursive=True)
            ]
        except Exception as e:
            # The child may have exited already; nothing to pin, and an empty
            # answer here would be a claim we cannot support. The REASON is kept
            # because "psutil is missing" and "the lookup failed" are different
            # facts, and reporting the wrong one is its own small dishonesty.
            self._tree_snapshot = None
            self._tree_unverified_reason = (
                f"the process tree could not be read: {type(e).__name__}")

    def _surviving_descendants(self) -> Optional[list]:
        """Snapshot entries still alive under their ORIGINAL identity.

        ``None`` means we could not look; ``[]`` means we looked and the tree is
        clean. Collapsing those two would turn "unverified" into "verified good",
        which is the exact failure this check exists to prevent.
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
                # Gone, or unreadable. Neither is positive evidence of a
                # survivor, and we only ever act on positive evidence.
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
        # Positive evidence that delivery is not termination. Every entry is
        # pinned by (pid, create_time) to a process we saw as our own descendant,
        # so this can never signal a recycled pid.
        for proc in survivors:
            try:
                proc.kill()
            except Exception:
                # Already gone, or not ours to signal. The re-check below is the
                # authority on what actually survived, so a failure here is not
                # worth a warning of its own.
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
    """A background sub-agent: the second kind, exactly as this module intended.

    The child Agent is built by the CALLER on the parent's thread and handed in
    already constructed, so a construction error (bad role, unreadable preload)
    surfaces synchronously in the spawn tool's result instead of turning into a
    job that immediately fails. This job only owns RUNNING it.

    ``result`` payload: ``{"summary": str, "turns": int}``.

    NOT PREEMPTIBLE, and we say so rather than pretending. ``Agent.run_task`` is
    a blocking call with no cooperative cancellation anywhere in the agent
    package (the only interruption path is a KeyboardInterrupt in the INTERACTIVE
    loop, which a worker thread cannot raise). So ``_terminate`` cannot stop a
    turn already in flight: it records a warning and marks intent, and the daemon
    thread dies with the process. That is why this PR ships no ``kill_agent_job``
    tool - offering one that cannot actually stop the work would be a facade.
    """

    kind = "agent"

    def __init__(self, child: Any, task: str, *, label: str,
                 token: Any = None, finalize: Any = None) -> None:
        super().__init__(label)
        self._child = child
        self._task = task
        # Optional teardown run on THIS job's worker thread once the child stops:
        # ``finalize(child) -> dict`` merged into the terminal payload. The
        # isolation teardown (commit the child's branch, capture its diff, remove
        # its worktree) lives there rather than here, so this module keeps knowing
        # nothing about git. It runs on the worker because only the worker knows
        # when the child finished - but it must NEVER touch parent state; the
        # parent folds the payload in at its own turn boundary.
        self._finalize = finalize
        # The child_limit slot this job holds. Released on FINISH, not on submit:
        # the budget is about children that are RUNNING, and a job that has been
        # submitted but not yet finished is still occupying the box.
        self._token = token
        self._outcome: Optional[dict] = None
        self._runner = threading.Thread(
            target=self._run, name=f"bgagent-{self.id}", daemon=True)
        self._runner.start()
        # LAST, per the base class contract: without this the job never leaves
        # "running".
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
            # run_task RETURNS its failure message rather than raising (max_turns
            # reached, circuit breaker tripped), so reaching this line is not the
            # same as succeeding. Record the child's OWN verdict, or the parent
            # cannot tell a failed sub-agent from a finished one and reports it ok.
            outcome = {"summary": text, "turns": getattr(self._child, "turns", 0),
                       "ok": bool(getattr(self._child, "last_run_ok", True))}
        except Exception as exc:                      # noqa: BLE001 - recorded
            # Surfaced as the job's terminal error, never swallowed.
            outcome = {"error": f"{type(exc).__name__}: {exc}"}
        # Teardown runs even when the child failed: its worktree still exists and
        # would leak otherwise, and a failed child may still have committed work
        # worth pointing at.
        if self._finalize is not None:
            try:
                extra = self._finalize(self._child)
                if extra:
                    outcome.update(extra)
            except Exception as exc:                  # noqa: BLE001 - surfaced
                # A teardown failure is REAL (a worktree may be left behind), so
                # it becomes a visible warning rather than a silent pass.
                self.warnings.append(f"teardown failed: {type(exc).__name__}: {exc}")
        with self._lock:
            self._outcome = outcome

    def _poll(self):
        return self._outcome

    def _result_for(self, poll_value) -> Optional[dict]:
        # Defang here, at the single choke point where a child's text becomes
        # readable by the parent: a sub-agent may have quoted untrusted web/MCP
        # content verbatim, and this string re-enters the PARENT loop as a
        # trusted tool result. Same reasoning as the synchronous path.
        from .provenance import neutralise
        payload = {
            "summary": neutralise(poll_value.get("summary") or ""),
            "turns": poll_value.get("turns", 0),
        }
        if poll_value.get("error"):
            payload["error"] = poll_value["error"]
        # Isolation facts produced by finalize(), carried through verbatim so the
        # parent can record a DelegatedChangeSet at ITS turn boundary. The diff is
        # NOT neutralised: it is machine-read git output the parent renders in a
        # clearly-labelled block, never merged into session_diff().
        for key in ("branch", "base", "file_count", "diff", "worktree",
                    "cleanup_warning"):
            if key in poll_value:
                payload[key] = poll_value[key]
        return payload

    def _terminate(self, *, force: bool) -> None:
        # Cannot preempt a blocking run_task (see the class docstring). Record it
        # instead of reporting a stop that did not happen.
        self.warnings.append(
            "a background sub-agent cannot be stopped mid-turn; it will finish "
            "or die with the process")

    def _finish(self, state: str, result: Optional[dict],
                error: Optional[str] = None) -> None:
        # Promote a child-raised exception to the job's own error field so a
        # failed delegation is visible in check_agent_job, not buried in result.
        if result and result.get("error") and not error:
            error, state = result["error"], "failed"
        try:
            super()._finish(state, result, error)
        finally:
            # Release the shared child budget exactly once; release() is
            # idempotent and tolerates None, so a double _finish is harmless.
            from .child_limit import release
            release(self._token)
            self._token = None


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
        # PER KIND (see _KEEP_FINISHED): the table's total bound is this times
        # the number of kinds that have finished work, not this on its own.
        self.keep_finished = keep_finished
        # Completions evicted before any drain collected them, PER KIND.
        # Cumulative and never reset: the session-long diagnostic.
        #
        # What a non-zero count MEANS depends on the kind, which is why it is
        # kept per kind rather than as one number. For a kind with a drain-based
        # consumer (agent) it is a real, silent loss: absorption is drain-only,
        # so the payload is unrecoverable. For a kind polled by id (shell) it is
        # ordinary bounded-table housekeeping and is NOT silent - check_shell_job
        # answers "No background job with id X. Known job ids: ..." So a caller
        # must not render both with the same alarm, or the warning cries wolf on
        # every long session and stops being read.
        self.dropped_undrained_by_kind: dict[str, int] = {}
        # The same losses awaiting a REPORT to that kind's consumer. Separate
        # from the cumulative total because the two have different jobs: the
        # total is a standing diagnostic, this one is consumed by
        # take_dropped_undrained so a turn-boundary consumer warns about each
        # loss exactly once instead of repeating it every turn forever.
        self._unreported_drops: dict[str, int] = {}
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        # A job the model started must not outlive the localm process: it was
        # detached into its own group/session precisely so it survives signals,
        # which is exactly what makes it an orphan if we just exit.
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

        # Kind mismatch. The factory has ALREADY produced a LIVE job (a ShellJob
        # has spawned its OS process; an AgentJob its worker thread and its
        # child_limit token), so rejecting it without stopping it would leak
        # exactly what the cap check above exists to prevent - and the leak would
        # be unreachable, because kill_shell_job and shutdown_all only ever see
        # REGISTERED jobs and this one never gets registered. Unreachable in tree
        # today (both call sites pass a matching kind); it is a trap for whoever
        # adds the third kind, which is the whole point of a generic registry.
        #
        # Stopped OUTSIDE the registry lock on purpose: a kill can take a grace
        # period, and holding the lock through it would stall every other
        # registry call. Nothing can race us for a job that was never registered.
        detail = ""
        try:
            outcome = job.kill()
            if outcome.startswith("kill FAILED"):
                detail = f" The stray job could not be stopped: {outcome}."
        except Exception as e:            # noqa: BLE001 - folded into the error
            # Never silently dropped: if we could not stop it, the caller is the
            # only one who can still learn a process was left behind.
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
            # Drop jobs somebody has already collected FIRST, so a completion
            # that no drain has seen yet survives as long as possible. (Stable
            # sort keeps oldest-first within each group.)
            ordered = sorted(finished, key=lambda j: not j.drained)
            for job in ordered[:excess]:
                self._jobs.pop(job.id, None)
                if not job.drained:
                    # The table must stay bounded, so once EVERY retained
                    # completion of a kind is undrained something has to go. That
                    # means a drain-based consumer will never see this one: count
                    # it AND queue it for report rather than let it vanish (a lost
                    # completion looks identical to "nothing finished", which is
                    # exactly the failure we must not hide).
                    self.dropped_undrained_by_kind[kind] = (
                        self.dropped_undrained_by_kind.get(kind, 0) + 1)
                    self._unreported_drops[kind] = (
                        self._unreported_drops.get(kind, 0) + 1)

    def take_dropped_undrained(self, kind: Optional[str] = None) -> int:
        """Uncollected completions lost since the last call, and RESET the count.

        The reporting half of ``dropped_undrained``: a counter nobody reads is
        bookkeeping, not honesty. A drain-based consumer calls this next to its
        ``drain_finished`` and tells the user what it lost, because from the
        consumer's side a discarded completion is indistinguishable from "nothing
        finished". Consumed exactly once, like the drain itself, so a turn-
        boundary caller warns per loss instead of every turn forever. The
        cumulative ``dropped_undrained`` total is deliberately NOT reset here -
        that one stays readable all session (``/bg`` shows it).
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
        we could not kill is a live process the coder started outliving localm -
        precisely the orphan the hook exists to prevent, and precisely what the
        shipped "stopped at exit rather than orphaned" claim promises does not
        happen. ``kill()`` reports that failure by RETURN VALUE ("kill FAILED
        - ..."), not by raising, so counting every non-raising call as a success
        would report the orphan case as a clean shutdown. Failures are counted
        out and printed instead (AGENTS.md rule 5: a safety step that failed must
        never report success).
        """
        running = self.running()
        if running:
            # Each kill can take a grace period (seconds on POSIX, longer on
            # Windows with two taskkill timeouts), and several stubborn jobs make
            # that add up. Bounded, so it is not a hang - but a silent multi-
            # second pause at exit reads exactly like one.
            self._report_at_exit(
                f"localm: stopping {len(running)} background job(s) started by "
                "the coder...")

        killed = 0
        failures: list[str] = []
        for job in running:
            # The whole body is inside the try, INCLUDING building the failure
            # string: "Never raises" has to hold for an atexit hook, and reading
            # job.id / job.label is itself a call into someone else's object.
            try:
                outcome = job.kill()
                if outcome.startswith("kill FAILED"):
                    failures.append(f"{job.id} ({job.label[:60]}): {outcome}")
                else:
                    killed += 1
            except Exception as e:        # noqa: BLE001 - reported, not swallowed
                # Must not raise out of an atexit hook, but the reason must not
                # be lost either: an exception here means we do not even know
                # whether the process died, which is worse than a known failure.
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
        but staying silent about an orphan is not acceptable either - so we try,
        and give up only when the stream itself is gone.
        """
        try:
            stream = sys.stderr
            if stream is None:
                return
            print(message, file=stream, flush=True)
        except Exception:
            # The stream is gone (closed during interpreter teardown). There is
            # nowhere left to report to; raising here would only replace a
            # reportable orphan with an atexit traceback.
            pass


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
