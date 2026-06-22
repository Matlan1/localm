# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job persistence: the Job dataclass and the JobStore.

Job definitions live in ``<data dir>/jobs/jobs.json`` (a single JSON file,
written atomically via a temp file + os.replace). Each job's run results live in
``<data dir>/jobs/results/<job_id>/<iso-ts>.json`` (one file per run, holding the
prompt, output, status, and timing).

Results are EXPLICIT user data, like generated images: they are saved in every
privacy mode (the user asked for this job to run and keep its output). What a
job RUN writes as a session trace (audit JSONL, transcripts) still honours
``effective_mode`` - that is the runner's concern, not the store's.

Every path the store touches is resolved and confined under the jobs dir, so a
crafted job id (``../../etc``) can never escape it.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Process-wide lock serialising the read-modify-write of jobs.json. The scheduler
# records results from a worker thread (run_in_executor) while route handlers
# create/edit jobs on the event-loop thread, and several JobStore instances may
# point at the same file - so an instance lock is not enough. Home-scale data,
# so one coarse lock is fine.
_STORE_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
#  Job definition                                                             #
# --------------------------------------------------------------------------- #

# Task kinds and schedule kinds the store accepts. Kept here (not in the
# scheduler) so the store can validate a job def before it is ever persisted.
TASK_KINDS = ("chat", "coder", "memory")
SCHEDULE_KINDS = ("interval", "cron")

# A job id is a short opaque token; we also accept any string but confine it to
# a single path segment when it is used to build a results directory.
_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass
class Job:
    """A scheduled recurring task definition.

    schedule_kind == "interval": ``schedule`` is the period in SECONDS (int).
    schedule_kind == "cron":     ``schedule`` is a 5-field cron string
                                 ("minute hour dom month dow").
    task_kind == "chat":  run ``prompt`` against the inference engine.
    task_kind == "coder": run a coder agent for ``prompt`` in ``cwd``.
    """

    name: str
    schedule_kind: str = "interval"
    schedule: "int | str" = 3600
    task_kind: str = "chat"
    prompt: str = ""
    model: Optional[str] = None
    cwd: Optional[str] = None
    scope: Optional[str] = None        # coder file-access glob, optional
    enabled: bool = True
    id: str = ""
    created: Optional[float] = None
    last_run: Optional[float] = None
    last_status: Optional[str] = None      # "ok" | "error" | None (never run)
    last_result_id: Optional[str] = None   # iso-ts stem of the last result file

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if self.created is None:
            self.created = time.time()
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on a malformed job def (called at construction and
        on update so a bad def never reaches disk or the scheduler)."""
        if not str(self.name).strip():
            raise ValueError("job name is required")
        if self.task_kind not in TASK_KINDS:
            raise ValueError(
                f"task_kind must be one of {TASK_KINDS}, got {self.task_kind!r}")
        if self.schedule_kind not in SCHEDULE_KINDS:
            raise ValueError(
                f"schedule_kind must be one of {SCHEDULE_KINDS}, "
                f"got {self.schedule_kind!r}")
        if self.schedule_kind == "interval":
            try:
                secs = int(self.schedule)
            except (TypeError, ValueError):
                raise ValueError("interval schedule must be an integer of seconds")
            if secs < 1:
                raise ValueError("interval schedule must be >= 1 second")
            self.schedule = secs
        else:  # cron
            from localm.plugins.builtin.jobs.scheduler import validate_cron
            if not isinstance(self.schedule, str) or not self.schedule.strip():
                raise ValueError("cron schedule must be a 5-field cron string")
            validate_cron(self.schedule)        # raises ValueError on a bad field
        # memory jobs synthesise from session logs and need no user prompt.
        if self.task_kind != "memory" and not str(self.prompt).strip():
            raise ValueError("prompt is required")
        if self.task_kind == "coder" and not (self.cwd and str(self.cwd).strip()):
            raise ValueError("coder jobs require a cwd")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        # Only keep known fields so a forward-compat file with extra keys loads.
        known = {f for f in cls.__dataclass_fields__}      # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
#  Store                                                                       #
# --------------------------------------------------------------------------- #

def jobs_dir() -> Path:
    """The jobs data dir (``<data dir>/jobs``), resolved at call time so a test
    that monkeypatches the home dir is honoured."""
    from localm.config import home_dir
    return (home_dir() / "jobs").resolve()


class JobStore:
    """Persist job definitions and run results under the jobs data dir.

    All filesystem access is confined under :func:`jobs_dir` (resolve +
    is_relative_to), so neither a job id nor a result timestamp can be used to
    write or read outside it. The definitions file is written atomically (temp +
    os.replace) so a crash mid-write cannot corrupt jobs.json.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = (Path(root).resolve() if root is not None else jobs_dir())
        self._defs_file = self._root / "jobs.json"
        self._results_root = self._root / "results"

    # ---- path confinement --------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    def _confine(self, p: Path) -> Path:
        """Resolve *p* and guarantee it stays under the jobs root."""
        rp = p.resolve()
        if rp != self._root and not rp.is_relative_to(self._root):
            raise ValueError(f"path escapes the jobs dir: {p}")
        return rp

    def _result_dir(self, job_id: str) -> Path:
        """Results dir for *job_id*, confined to one path segment under
        results/. A crafted id (``..`` / separators) is sanitised first, then
        the resolved path is re-checked against the root."""
        safe = _ID_RE.sub("_", str(job_id)).strip("._") or "job"
        d = self._confine(self._results_root / safe)
        if d.parent != self._results_root.resolve():
            raise ValueError(f"invalid job id for a results dir: {job_id!r}")
        return d

    # ---- internal IO -------------------------------------------------------
    def _ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> dict:
        if not self._defs_file.is_file():
            return {}
        try:
            data = json.loads(self._defs_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        out: dict = {}
        for entry in jobs:
            if not isinstance(entry, dict):
                continue
            try:
                job = Job.from_dict(entry)
            except (ValueError, TypeError):
                continue        # skip a corrupt entry rather than fail the load
            out[job.id] = job
        return out

    def _write_all(self, jobs: dict) -> None:
        """Atomically write the whole defs file (temp + os.replace)."""
        self._ensure_root()
        payload = {"version": 1,
                   "jobs": [j.to_dict() for j in jobs.values()]}
        # Unique temp name (pid + thread + uuid) so concurrent writers never
        # share a temp path; os.replace is atomic, so the last writer wins
        # cleanly rather than corrupting or clobbering the file.
        tmp = self._defs_file.with_name(
            f"{self._defs_file.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self._defs_file)

    # ---- CRUD --------------------------------------------------------------
    def list(self) -> list:
        """All jobs, sorted by creation time (oldest first)."""
        jobs = self._read_all()
        return sorted(jobs.values(), key=lambda j: (j.created or 0, j.id))

    def get(self, job_id: str) -> Optional[Job]:
        return self._read_all().get(job_id)

    def add(self, job: Job) -> Job:
        job.validate()
        with _STORE_LOCK:
            jobs = self._read_all()
            jobs[job.id] = job
            self._write_all(jobs)
        return job

    def update(self, job_id: str, **changes) -> Job:
        """Apply *changes* to an existing job and persist. Re-validates the
        result. Raises KeyError if the job does not exist."""
        with _STORE_LOCK:
            jobs = self._read_all()
            job = jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            for key, value in changes.items():
                if key in Job.__dataclass_fields__ and key != "id":   # never remap id
                    setattr(job, key, value)
            job.validate()
            jobs[job.id] = job
            self._write_all(jobs)
        return job

    def remove(self, job_id: str) -> bool:
        """Delete a job def and its results dir. Returns True if it existed."""
        with _STORE_LOCK:
            jobs = self._read_all()
            if job_id not in jobs:
                return False
            del jobs[job_id]
            self._write_all(jobs)
        # Best-effort: drop the job's results too (confined to the jobs dir).
        try:
            import shutil
            d = self._result_dir(job_id)
            if d.is_dir():
                shutil.rmtree(d)
        except (OSError, ValueError):
            pass
        return True

    # ---- results -----------------------------------------------------------
    def record_result(self, job_id: str, result: dict) -> str:
        """Persist one run *result* for *job_id* and return its result id (the
        iso-ts file stem). Also stamps the job's last_run / last_status /
        last_result_id. Results are saved in every privacy mode (explicit user
        data). A timestamp collision is disambiguated with a counter suffix."""
        d = self._result_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        stamp = _iso_stamp(result.get("finished") or result.get("started")
                           or time.time())
        result_id = stamp
        path = self._confine(d / f"{result_id}.json")
        n = 1
        while path.exists():
            result_id = f"{stamp}_{n}"
            path = self._confine(d / f"{result_id}.json")
            n += 1
        record = {"job_id": job_id, "result_id": result_id, **result}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        # Stamp the job def (best-effort: a run for a now-deleted job is fine).
        with _STORE_LOCK:
            jobs = self._read_all()
            job = jobs.get(job_id)
            if job is not None:
                job.last_run = result.get("finished") or time.time()
                job.last_status = result.get("status")
                job.last_result_id = result_id
                jobs[job.id] = job
                self._write_all(jobs)
        return result_id

    def list_results(self, job_id: str) -> list:
        """Run results for *job_id*, newest first. Each entry is the stored
        record dict (prompt + output + status + timing)."""
        try:
            d = self._result_dir(job_id)
        except ValueError:
            return []
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("*.json"),
                        key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return out


def _iso_stamp(ts: float) -> str:
    """Filesystem-safe ISO-ish timestamp (no colons) for a result filename."""
    lt = time.localtime(ts)
    return time.strftime("%Y-%m-%dT%H-%M-%S", lt)
