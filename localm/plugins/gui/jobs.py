"""
Background jobs for the GUI: model pulls and image generation.

A job wraps a ``localm`` CLI subprocess. Its stdout/stderr lines are pushed
onto a queue that the web layer streams to the browser as SSE, so the GUI
reuses the exact CLI logic (progress bars, dedup checks, split GGUF handling)
without duplicating any of it. Subprocesses run without a TTY, which makes
all interactive prompts fall back to their safe non-interactive defaults.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    id: str
    kind: str                      # "pull" | "imagine" | ...
    argv: list
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10_000))
    status: str = "running"        # running | done | failed | cancelled
    returncode: Optional[int] = None
    result: Optional[str] = None   # kind-specific payload (e.g. output image path)
    created_at: float = field(default_factory=time.time)
    _proc: Optional[subprocess.Popen] = None

    def push(self, event: dict) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def cancel(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self.status = "cancelled"
            proc.terminate()


class JobManager:
    """Registry of background jobs. Finished jobs stay queryable for an hour."""

    _TTL_S = 3600

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start_cli(self, kind: str, cli_args: list, *, result_path: str | None = None) -> Job:
        """
        Run ``python -m localm <cli_args>`` as a job.

        result_path, when given, is stored on the job as the expected output
        artifact (e.g. the image file an imagine job writes).
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            argv=[sys.executable, "-X", "utf8", "-m", "localm", *cli_args],
            result=result_path,
        )
        with self._lock:
            self._gc()
            self._jobs[job.id] = job

        def _run():
            try:
                job._proc = subprocess.Popen(
                    job.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in job._proc.stdout:
                    line = line.rstrip()
                    if line:
                        job.push({"type": "line", "text": line})
                job._proc.wait()
                job.returncode = job._proc.returncode
                if job.status != "cancelled":
                    job.status = "done" if job.returncode == 0 else "failed"
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

    def start_fn(self, kind: str, fn, *, result_path: str | None = None) -> Job:
        """
        Run a Python callable as a job in a worker thread.

        ``fn`` receives the job and should return True on success. It may call
        ``job.push({"type": "line", ...})`` to report progress and may update
        ``job.result``.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, argv=[], result=result_path)
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

    def _gc(self) -> None:
        cutoff = time.time() - self._TTL_S
        stale = [
            jid for jid, j in self._jobs.items()
            if j.status != "running" and j.created_at < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]
