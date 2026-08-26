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

from localm.pathsafe import is_unc_or_device_path

# Process-wide lock serialising the read-modify-write of jobs.json. The scheduler
# records results from a worker thread (run_in_executor) while route handlers
# create/edit jobs on the event-loop thread, and several JobStore instances may point
# at the same file, so an instance lock is not enough.
_STORE_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
#  Job definition                                                             #
# --------------------------------------------------------------------------- #

# Task kinds and schedule kinds the store accepts. Held here rather than in the
# scheduler, so the store can validate a job def before it is persisted.
TASK_KINDS = ("chat", "coder", "memory", "rag")
SCHEDULE_KINDS = ("interval", "cron")

# A job id is a short opaque token. Any string is accepted, but confined to a single
# path segment when building a results directory.
_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")

# --------------------------------------------------------------------------- #
#  Quarantine copies of a corrupt jobs.json                                    #
# --------------------------------------------------------------------------- #
# How many corrupt-copies to keep.
_QUARANTINE_KEEP = 3

# `owner` holds the sha256 of the creating key, the same digest the keystore
# stores. A quarantine copy is made from a file that FAILED to parse, so the
# redaction works on raw text. It matches the digest SHAPE (64 hex), so a mangled
# or truncated value is left alone.
_OWNER_DIGEST_RE = re.compile(r'("owner"\s*:\s*)"[0-9a-fA-F]{64}"')

# What the digest is replaced with. It must be non-null and non-hex:
# job_owner_ok (inference/http_server.py) treats owner=None as unrestricted, and
# a principal_id() is always 64 hex, so this sentinel resolves to nobody and
# fails CLOSED - unreachable to a scoped key, still reachable to an admin/owner
# key.
_REDACTED_OWNER = "redacted-on-quarantine"


def _redact_owner_digests(raw: str) -> str:
    """Strip owner key digests out of a corrupt jobs.json before it is copied
    aside, keeping everything the copy exists to preserve.

    Only the ACCESS-CONTROL field goes: schedule, prompt, model, cwd and name are
    left exactly as they were, so the copy still holds the user's scheduled jobs.

    Logs a WARNING when a digest-shaped value survives: a partially corrupt file
    can hold one in a position this does not match, so a half-completed redaction
    is reported rather than looking like a complete one."""
    if not raw:
        return raw
    out, n = _OWNER_DIGEST_RE.subn(rf'\1"{_REDACTED_OWNER}"', raw)
    leftover = re.search(r"\b[0-9a-fA-F]{64}\b", out)
    if leftover:
        from localm.debuglog import logger
        logger.warning(
            "jobs store: quarantine copy still holds a digest-shaped value "
            "after redacting %d owner field(s); the file is corrupt, so it may "
            "carry one in a form this cannot match", n)
    return out


@dataclass
class Job:
    """A scheduled recurring task definition.

    schedule_kind == "interval": ``schedule`` is the period in SECONDS (int).
    schedule_kind == "cron":     ``schedule`` is a 5-field cron string
                                 ("minute hour dom month dow").
    task_kind == "chat":  run ``prompt`` against the inference engine.
    task_kind == "coder": run a coder agent for ``prompt`` in ``cwd``.
    task_kind == "rag":   re-sync the RAG ``collection`` against the folders it
                          was indexed from (no prompt, no chat model).
    """

    name: str
    schedule_kind: str = "interval"
    schedule: "int | str" = 3600
    task_kind: str = "chat"
    prompt: str = ""
    model: Optional[str] = None
    cwd: Optional[str] = None
    scope: Optional[str] = None        # coder file-access glob, optional
    # rag jobs: the knowledge collection to re-sync.
    collection: Optional[str] = None
    # Coder jobs run RESTRICTED by default: read plus confined edit, no shell, no
    # network, no sub-agents. The owner opts a specific job into the full,
    # shell-capable coder by setting allow_shell; the API route gates that opt-in
    # to owner / coder:full callers.
    allow_shell: bool = False
    enabled: bool = True
    id: str = ""
    created: Optional[float] = None
    last_run: Optional[float] = None
    last_status: Optional[str] = None      # "ok" | "error" | None (never run)
    last_result_id: Optional[str] = None   # iso-ts stem of the last result file
    # Principal id (keystore hash of the creating key, or None for a tokenless /
    # open-mode creation) that OWNS this job. Per-job routes accept only the owner or
    # an admin/owner key, so one jobs-scoped key cannot read, edit, delete or run
    # another principal's scheduled jobs. Persisted so it survives a restart;
    # stripped from the API response.
    owner: Optional[str] = None
    # Whether the credential that created this job was the OWNER KEY itself (or
    # an owner session) rather than a minted keystore key. Stamped at creation
    # because it is not recoverable later: at run time a revoked scoped key and a
    # rotated-away owner key both hash to no keystore entry, and the runner needs
    # the difference to re-validate shell. Legacy jobs default to False and fall
    # back to the runner's key-value comparison. Internal, like `owner`: never
    # sent to a client.
    owner_is_owner_key: bool = False

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
        # memory jobs synthesise from session logs and rag jobs re-walk indexed
        # folders: both are fully specified without a user prompt.
        if self.task_kind not in ("memory", "rag") and not str(self.prompt).strip():
            raise ValueError("prompt is required")
        if self.task_kind == "coder" and not (self.cwd and str(self.cwd).strip()):
            raise ValueError("coder jobs require a cwd")
        if self.task_kind == "rag":
            if not (self.collection and str(self.collection).strip()):
                raise ValueError("rag jobs require a collection")
            # The NAME is validated here as well as at run time. Existence is
            # NOT checked: a collection may be created after the schedule, and
            # the runner reports a missing one.
            from localm.rag.store import check_collection_name
            self.collection = check_collection_name(str(self.collection).strip())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        # Only keep known fields so a forward-compat file with extra keys loads.
        known = {f for f in cls.__dataclass_fields__}      # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def cwd_unc_error(cwd) -> Optional[str]:
    """None when *cwd* is safe to use as a coder job's working directory, else
    a ready-to-show refusal message.

    ONE function owning the wording, called by every writer (plug.py's POST/PUT,
    cli.py's job_add) and by the runner's authoritative run-time re-check
    (runner.py's _run_coder)."""
    if cwd and is_unc_or_device_path(str(cwd)):
        return "cwd must be a local directory path, not a UNC or device path."
    return None


# --------------------------------------------------------------------------- #
#  Store                                                                       #
# --------------------------------------------------------------------------- #

def jobs_dir() -> Path:
    """The jobs data dir (``<data dir>/jobs``), resolved at call time so a
    relocated home dir is honoured."""
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
            return {}            # genuinely absent: an empty store is correct
        raw = None
        try:
            raw = self._defs_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except OSError as e:
            # Unreadable (locked / IO error): do NOT collapse to empty, or the next
            # write would overwrite a file that may be intact. Raise so the caller
            # cannot silently wipe it.
            raise RuntimeError(
                f"jobs store unreadable ({self._defs_file}): {e}") from e
        except json.JSONDecodeError as e:
            # Corrupt JSON: back it up so a subsequent write cannot destroy it,
            # warn, and start empty (distinct from the absent case).
            self._quarantine_corrupt(raw, e)
            return {}
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        out: dict = {}
        skipped = 0
        for entry in jobs:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            try:
                job = Job.from_dict(entry)
            except (ValueError, TypeError):
                skipped += 1    # skip a corrupt entry rather than fail the load
                continue
            out[job.id] = job
        if skipped:
            from localm.debuglog import logger
            logger.warning("jobs store: skipped %d unreadable job entr%s",
                           skipped, "y" if skipped == 1 else "ies")
        return out

    def _quarantine_corrupt(self, raw, err) -> None:
        """Copy a corrupt defs file aside before anything overwrites it, and warn.

        The copy is redacted (``_redact_owner_digests``) and the older copies
        pruned (``_prune_quarantine``)."""
        from localm.debuglog import logger
        try:
            stamp = int(time.time())
            backup = self._defs_file.with_name(
                f"{self._defs_file.name}.corrupt-{stamp}")
            if raw is not None:
                backup.write_text(_redact_owner_digests(raw), encoding="utf-8")
                # The backup still describes the user's jobs, so it is restricted
                # the way the live file is. Check-and-retry, like _write_all.
                from localm.config import restrict_file_perms
                if not restrict_file_perms(backup):
                    restrict_file_perms(backup)
            logger.warning(
                "jobs store %s is corrupt (%s); backed up to %s and starting "
                "with an empty job list - your scheduled jobs are preserved in "
                "the backup, not lost", self._defs_file, err, backup.name)
            self._prune_quarantine()
        except OSError as e:
            # The caller still proceeds with an empty job list, exactly as on
            # success.
            logger.warning("jobs store: corrupt defs file could not be backed "
                           "up (%s); the corrupt content may be lost on the "
                           "next write", e)

    def _prune_quarantine(self) -> None:
        """Keep only the newest ``_QUARANTINE_KEEP`` corrupt-copies, bounding
        older ones that still carry unredacted digests.

        Best-effort: a failure to delete an old backup does not break the
        recovery path that just wrote a new one, and is logged at debug."""
        from localm.debuglog import logger
        try:
            backups = sorted(
                self._defs_file.parent.glob(f"{self._defs_file.name}.corrupt-*"))
        except OSError as e:
            logger.debug("jobs store: could not list quarantine copies (%s)", e)
            return
        # Names are <file>.corrupt-<unix-ts>, so lexical order is chronological for
        # any timestamp of the same width, up to the year 2286.
        for old in backups[:-_QUARANTINE_KEEP] if _QUARANTINE_KEEP else backups:
            try:
                old.unlink()
            except OSError as e:
                logger.debug("jobs store: could not prune %s (%s)", old.name, e)

    def _write_all(self, jobs: dict) -> None:
        """Atomically write the whole defs file (temp + os.replace)."""
        self._ensure_root()
        payload = {"version": 1,
                   "jobs": [j.to_dict() for j in jobs.values()]}
        # Unique temp name (pid + thread + uuid) so concurrent writers never
        # share a temp path; os.replace is atomic, so the last writer wins.
        tmp = self._defs_file.with_name(
            f"{self._defs_file.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        # Each job def records `owner`, the sha256 of the creating key, so the
        # file is restricted to this account the way auth.key is. The TEMP file
        # is what gets restricted: it already holds the whole payload, and
        # os.replace carries the source's ACL onto the destination. The
        # destination is retried only when the first attempt did not happen.
        from localm.config import restrict_file_perms
        ok = restrict_file_perms(tmp)
        os.replace(tmp, self._defs_file)
        if not ok:
            restrict_file_perms(self._defs_file)

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
                try:
                    self._write_all(jobs)
                except Exception as e:
                    # The result file IS persisted (result_id above); only the
                    # job's last_run/last_status stamp failed, so the API will
                    # under-report this run until the next successful write.
                    from localm.debuglog import logger
                    logger.warning(
                        "jobs: result %s for job %s is persisted but its metadata "
                        "stamp failed (%s); last_run will under-report this run",
                        result_id, job_id, e)
        return result_id

    def list_results(self, job_id: str, *, limit: Optional[int] = None,
                     offset: int = 0) -> list:
        """Run results for *job_id*, newest first. Each entry is the stored
        record dict (prompt + output + status + timing).

        With *limit* / *offset*, only that page of files is READ into memory: the
        page is selected before reading, not sliced after loading everything.
        limit=None reads the whole history."""
        try:
            d = self._result_dir(job_id)
        except ValueError:
            return []
        if not d.is_dir():
            return []
        files = sorted(d.glob("*.json"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
        if offset:
            files = files[max(0, offset):]
        if limit is not None:
            files = files[:max(0, limit)]
        out = []
        for p in files:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return out


def _iso_stamp(ts: float) -> str:
    """Filesystem-safe ISO-ish timestamp (no colons) for a result filename."""
    lt = time.localtime(ts)
    return time.strftime("%Y-%m-%dT%H-%M-%S", lt)
