# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-process write lock for a single RAG collection."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from localm.debuglog import logger as _log

# How often the holder refreshes its heartbeat. Everything else is a multiple of
# this, so tuning one value keeps the ratios sane.
HEARTBEAT_INTERVAL = 5.0
# A holder that has not refreshed its heartbeat for this long is presumed
# crashed and its lock is reclaimed. NOT a limit on how long a lock may be held:
# a live holder beats every HEARTBEAT_INTERVAL, so a run of any length stays
# fresh. 12 missed beats, so a badly stalled but living process (an antivirus
# scan, a paused debugger) is not mistaken for a dead one.
STALE_AFTER = 60.0
# When the pin PROVES the holder is gone (pid absent, or recycled into another
# process), waiting out the full STALE_AFTER only delays recovery from a crash.
# Deliberately still LONGER than the heartbeat failures a live holder tolerates
# (_Heartbeat keeps going through a transient utime failure): if this dropped to
# a beat or two, any WRONG "dead" verdict would instantly outrace a living
# holder's own margin, which is how a safety accelerator turns into a lock thief.
DEAD_HOLDER_GRACE = 4 * HEARTBEAT_INTERVAL
# How long a would-be writer waits for the lock before refusing. Bounded on
# purpose: an unbounded wait turns a stuck peer into a hung CLI or a hung job.
WAIT_TIMEOUT = 30.0
# Only mention waiting once it has actually lasted; the uncontended case (the
# overwhelming majority) stays silent.
WAIT_NOTICE_AFTER = 1.0
_POLL = 0.05
_POLL_CAP = 0.5

# Operator overrides, read at call time (so a test or a script can set them
# without a restart). Documented in docs/rag.md.
ENV_WAIT = "LOCALM_RAG_LOCK_WAIT"
ENV_STALE = "LOCALM_RAG_LOCK_STALE"


class CollectionLockedError(RuntimeError):
    """Another writer holds this collection's write lock and did not release it within the wait budget."""

    def __init__(self, name: str, holder: Optional[dict], waited: float,
                 last_alive: Optional[float] = None,
                 lockpath: Optional[Path] = None, same_process: bool = False,
                 kind: str = "Collection"):
        # *kind* names WHAT is locked, for the message only. It defaults to
        # "Collection" so every existing RAG raise site reads exactly as before;
        # agent memory passes "Memory namespace" (see memory/store.py), because
        # the same machinery now serialises both and "Collection 'a1b2..'" would
        # be a lie in a memory refusal.
        self.kind = kind
        self.collection = name
        self.holder = holder
        self.waited = waited
        self.lockpath = lockpath
        who = ("another thread of this same localm process" if same_process
               else describe_holder(holder, last_alive))
        tail = ""
        if lockpath is not None and holder is None and not same_process:
            # Nothing could be read about the holder, so give the user the one
            # concrete thing they can act on rather than an unexplained refusal.
            tail = (f" Its lock file is {lockpath}; if you are certain no localm "
                    f"process is using it, deleting that file releases it.")
        super().__init__(
            f"{kind} '{name}' is being written by {who}. "
            f"Waited {_duration(waited)} and gave up; nothing was changed. Let "
            f"that run finish and try again.{tail}")


def wait_budget() -> float:
    """The current wait-before-refusing budget, honouring the env override."""
    return _env_float(ENV_WAIT, WAIT_TIMEOUT)


def lock_path_for(collection_dir: Path) -> Path:
    """The lock file for the collection stored at *collection_dir* (a sibling ``<name>.lock``, see the module docstring)."""
    return collection_dir.with_name(collection_dir.name + ".lock")


def describe_holder(rec: Optional[dict], last_alive: Optional[float] = None) -> str:
    """A human sentence naming who holds a lock, from its record."""
    if not isinstance(rec, dict):
        return ("another localm process (its lock record is unreadable, so it "
                "cannot say which)")
    pid = rec.get("pid")
    op = rec.get("op") or "a write"
    who = f"another localm process (pid {pid})" if pid else "another localm process"
    bits = [f"{who} running {op}"]
    started = rec.get("started")
    if isinstance(started, (int, float)):
        bits.append(f"held for {_duration(time.time() - started)}")
    if isinstance(last_alive, (int, float)):
        bits.append(f"last heartbeat {_duration(time.time() - last_alive)} ago")
    return ", ".join(bits)


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _env_float(name: str, default: float) -> float:
    """An operator override, or *default* if it is unset or not a usable number."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if value <= 0:
        _log.warning("%s=%r is not a positive number of seconds; using the "
                     "default of %.0fs", name, raw, default)
        return default
    return value


_machine_id_cache: Optional[str] = None


def _machine_id() -> str:
    """An opaque, stable id for THIS PID SPACE."""
    global _machine_id_cache
    if _machine_id_cache is None:
        parts = [sys.platform]
        try:
            parts.append(platform.node() or "")
        except Exception:
            parts.append("")
        try:
            # Linux/containers: distinguishes two pid namespaces on one host.
            parts.append(str(os.stat("/proc/self/ns/pid").st_ino))
        except OSError:
            pass                  # not Linux, or not exposed: the rest still holds
        if not any(p for p in parts[1:]):
            # We learned nothing machine-specific. A SHARED constant here would
            # make two unrelated boxes compare equal and start trusting each
            # other's pids, so fail toward "no two processes match": a value
            # unique to this process makes every foreign record read as
            # another pid space, which only ever costs a slower crash recovery.
            parts.append(uuid.uuid4().hex)
        _machine_id_cache = hashlib.sha256(
            "\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:16]
    return _machine_id_cache


def _create_time(pid: int) -> Optional[float]:
    """Process start time for *pid*, or None when it cannot be determined (psutil absent - it is an extra, not a core dependency - or the process is already gone)."""
    try:
        import psutil
    except Exception:
        return None
    try:
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _holder_liveness(rec: dict) -> str:
    """``'dead'``, ``'alive'`` or ``'unknown'`` for the process in *rec*."""
    if rec.get("machine") != _machine_id():
        return "unknown"          # another pid space: its pids say nothing here
    pid = rec.get("pid")
    pinned = rec.get("pid_create_time")
    if not isinstance(pid, int) or pid <= 0:
        return "unknown"
    if not isinstance(pinned, (int, float)):
        # The holder could not pin its own start time, so the bare number is not
        # an identity: some unrelated process may now own it. Not evidence.
        return "unknown"
    try:
        import psutil
    except Exception:
        return "unknown"
    try:
        current = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return "dead"
    except Exception:
        return "unknown"          # access denied, a broken psutil: not evidence
    # A pid that now belongs to a process which started AFTER the holder pinned
    # it was recycled: the holder itself is gone. Tolerance covers the different
    # rounding psutil applies per platform, not a real difference in start time.
    return "alive" if abs(current - pinned) <= 1.0 else "dead"


def _read_record(lockpath: Path):
    """``(record_or_None, mtime_or_None)`` for the lock file."""
    try:
        mtime = lockpath.stat().st_mtime
    except OSError:
        return None, None
    try:
        raw = lockpath.read_bytes()
    except OSError:
        return None, mtime
    try:
        rec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, mtime
    return (rec if isinstance(rec, dict) else None), mtime


def _is_stale(rec: Optional[dict], mtime: Optional[float], stale_after: float) -> bool:
    """Whether the holder stopped proving it was alive."""
    if mtime is None:
        return False              # the file vanished; the caller re-tries the create
    # A holder whose clock runs ahead of ours yields a negative age. Clamp to 0
    # (treat as fresh) rather than letting arithmetic decide to steal a lock.
    age = max(0.0, time.time() - mtime)
    if age > stale_after:
        return True
    if isinstance(rec, dict) and _holder_liveness(rec) == "dead":
        return age > DEAD_HOLDER_GRACE
    return False


class _Heartbeat(threading.Thread):
    """Keeps the holder's lock file looking alive until the lock is released."""

    def __init__(self, lockpath: Path, record: dict, interval: float):
        super().__init__(name=f"rag-lock-{record.get('op', 'write')}", daemon=True)
        self._lockpath = lockpath
        self._record = record
        self._interval = interval
        # NOT `_stop`: threading.Thread already uses that name internally, and
        # shadowing it breaks join() at interpreter level.
        self._stopping = threading.Event()
        self._failures = 0

    def run(self) -> None:
        while not self._stopping.wait(self._interval):
            if not self._beat():
                return

    def _beat(self) -> bool:
        """One refresh."""
        if self._stopping.is_set():
            # Released while we were sleeping. Touching the file now could
            # refresh a lock the releasing thread is about to remove.
            return False
        rec, _ = _read_record(self._lockpath)
        if rec is not None and rec.get("token") != self._record["token"]:
            _note(f"another localm process took over the write lock on "
                  f"'{self._record.get('collection')}' while this process was "
                  f"still writing to it. Both may now be writing; check the "
                  f"collection with 'localm rag list' when both runs finish.")
            return False
        if self._stopping.is_set():
            return False
        try:
            os.utime(self._lockpath, None)
        except FileNotFoundError:
            # Our lock file is gone while we are demonstrably alive: somebody
            # judged us stale and removed it. Deliberately NOT re-created - a
            # fresh file here would collide with whoever is taking over, and
            # re-creating a lock during release is how a phantom lock outlives
            # the run that owned it. Report and stand down instead.
            _note(f"the write lock file for '{self._record.get('collection')}' "
                  f"was removed while this process still held it. Another "
                  f"localm process may now be writing to the same collection.")
            return False
        except OSError as e:
            # Transient (an antivirus scanner holding the file, a full disk).
            # Missing ONE beat is harmless - STALE_AFTER is twelve of them - so
            # keep going rather than abandoning a lock we still hold. A run of
            # them is NOT harmless: it ends with somebody reclaiming this lock
            # while we are still writing, so it escalates instead of staying a
            # debug line nobody reads (AGENTS.md rule 5).
            self._failures += 1
            if self._failures == 3:
                _note(f"cannot refresh the write lock on "
                      f"'{self._record.get('collection')}' ({e}). If this keeps "
                      f"failing another localm process will treat this run as "
                      f"crashed and start writing to the same collection.")
            else:
                _log.debug("rag lock heartbeat failed for %s (%s)",
                           self._lockpath.name, e)
            return True
        self._failures = 0
        return True

    def stop(self) -> None:
        self._stopping.set()
        self.join(timeout=self._interval + 1.0)


def _note(message: str) -> None:
    """Surface an unusual lock event through BOTH channels, always."""
    print(f"[localm] note: {message}", file=sys.stderr)
    _log.warning("rag lock: %s", message)


@contextlib.contextmanager
def collection_write_lock(lockpath: Path, *, collection: str, op: str,
                          timeout: Optional[float] = None,
                          stale_after: Optional[float] = None,
                          on_wait: Optional[Callable[[str], None]] = None,
                          kind: str = "Collection"):
    """Hold the cross-process write lock for a collection, or refuse."""
    timeout = _env_float(ENV_WAIT, WAIT_TIMEOUT) if timeout is None else timeout
    stale_after = (_env_float(ENV_STALE, STALE_AFTER)
                   if stale_after is None else stale_after)
    pid = os.getpid()
    record = {
        "token": uuid.uuid4().hex,
        "pid": pid,
        "pid_create_time": _create_time(pid),
        "machine": _machine_id(),
        "collection": collection,
        "op": op,
        "started": time.time(),
    }
    token = record["token"]
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    started_waiting = time.time()
    announced = False
    attempt = 0

    while True:
        try:
            fd = os.open(str(lockpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:
            # WINDOWS ONLY, and it is a WAIT, not a failure. On Windows a lock
            # file that exists but is momentarily inaccessible - the holder's
            # unlink in flight, a scanner or backup holding a handle - reports
            # ERROR_ACCESS_DENIED here rather than the ERROR_FILE_EXISTS that
            # becomes FileExistsError. Letting it propagate aborts the user's
            # command with "localm hit an unexpected error" over a condition
            # that clears on its own in milliseconds, which is what reddened
            # the release gate for 0.1.5rc3.
            #
            # Treated as "a holder may be there and I could not read it": it
            # waits on the SAME deadline as every other contended path, so it
            # can never spin longer than the caller's timeout, and it reports a
            # normal lock timeout if it never clears.
            #
            # NOT applied on POSIX: there this genuinely means the directory is
            # not writable, which no amount of waiting fixes, and where raising
            # at once is the honest answer.
            if os.name != "nt":
                raise
            rec, mtime = _read_record(lockpath)
            waited = time.time() - started_waiting
            if time.time() >= deadline:
                raise CollectionLockedError(collection, rec, waited, mtime,
                                            lockpath, kind=kind)
            if on_wait and not announced and waited >= WAIT_NOTICE_AFTER:
                announced = True
                on_wait(f"waiting for the write lock on '{collection}': "
                        f"{describe_holder(rec, mtime)}")
            time.sleep(min(_POLL * (attempt + 1), _POLL_CAP))
            attempt += 1
            continue
        except FileExistsError:
            rec, mtime = _read_record(lockpath)
            if (mtime is not None and _is_stale(rec, mtime, stale_after)
                    and _reclaim(lockpath, rec, mtime, stale_after)):
                continue          # removed: retry the create straight away
            # Anything else - a live holder, or a stale lock we could NOT remove
            # (a permissions fault, a handle another process still has open, a
            # fresher lock that appeared under us) - waits on the ONE budget
            # below. Retrying a reclaim from here instead would spin without ever
            # consulting the deadline: a hang, dressed as a busy loop.
            waited = time.time() - started_waiting
            if time.time() >= deadline:
                raise CollectionLockedError(collection, rec, waited, mtime,
                                            lockpath, kind=kind)
            if on_wait and not announced and waited >= WAIT_NOTICE_AFTER:
                announced = True
                on_wait(f"waiting for the write lock on '{collection}': "
                        f"{describe_holder(rec, mtime)}")
            time.sleep(min(_POLL * (attempt + 1), _POLL_CAP))
            attempt += 1
            continue
        # We created the file, so from here every failure must remove OUR file,
        # or a transient error leaks a lock nobody owns that blocks every writer
        # of this collection until it goes stale.
        try:
            try:
                os.write(fd, json.dumps(record).encode("utf-8"))
            finally:
                os.close(fd)
            beat = _Heartbeat(lockpath, record, HEARTBEAT_INTERVAL)
            # Inside the cleanup block on purpose: a thread that cannot start
            # (thread exhaustion) would otherwise leave a lock file behind with
            # nothing refreshing it, blocking this collection until it went stale.
            beat.start()
        except BaseException:
            try:
                lockpath.unlink()
            except OSError:
                pass
            raise
        break

    try:
        yield
    finally:
        beat.stop()
        if beat.is_alive():
            # It should have exited the moment the stop event was set. Still
            # running means it is stuck in a filesystem call - say so instead of
            # leaving an unexplained refreshed timestamp behind.
            _note(f"the heartbeat thread for '{collection}' did not stop; it is "
                  f"stuck in a filesystem call and may keep this collection's "
                  f"lock looking alive for a moment after this run ended.")
        # Fencing release: remove the file ONLY on a POSITIVE match of the token
        # this call wrote. Two cases must NOT delete it. Another process
        # reclaimed it as stale while we were legitimately still inside the
        # critical section - deleting THEIR live lock would let a third writer in
        # and rebuild this very race through its own recovery path (config.py
        # learned that one the hard way). And we could not read the record at
        # all, which includes the moment a successor has created its file but not
        # yet written into it: "I cannot read it" must never be taken as "it must
        # be mine". A lock we wrongly leave behind costs one staleness window; a
        # live lock we wrongly delete costs a lost update.
        rec, _ = _read_record(lockpath)
        if isinstance(rec, dict) and rec.get("token") == token:
            try:
                lockpath.unlink()
            except OSError as e:
                _note(f"could not remove the write lock file for '{collection}' "
                      f"({e}); it will be reclaimed as stale in "
                      f"{stale_after:.0f}s.")
        elif isinstance(rec, dict):
            _note(f"the write lock on '{collection}' was taken over by another "
                  f"localm process before this write finished; leaving their "
                  f"lock in place.")
        elif lockpath.exists():
            _note(f"the write lock file for '{collection}' is no longer readable, "
                  f"so this run cannot prove the lock is still its own; leaving "
                  f"it rather than risk deleting another writer's. It is "
                  f"reclaimed as stale in {stale_after:.0f}s.")


def _reclaim(lockpath: Path, rec: Optional[dict], mtime: Optional[float],
             stale_after: float) -> bool:
    """Remove a lock whose holder stopped proving it was alive."""
    current, current_mtime = _read_record(lockpath)
    if current_mtime is None:
        return True               # already gone: the acquire loop can proceed
    if not _is_stale(current, current_mtime, stale_after):
        return False              # somebody is alive on it now: not ours to remove
    if isinstance(current, dict) != isinstance(rec, dict) or (
            isinstance(current, dict) and isinstance(rec, dict)
            and current.get("token") != rec.get("token")):
        return False              # a different lock than the one we judged
    try:
        lockpath.unlink()
    except FileNotFoundError:
        pass                      # already gone; the acquire loop re-tries anyway
    except OSError as e:
        # We judged it abandoned but cannot remove it (no permission, or another
        # process holds a handle to it). Say so and let the caller wait out its
        # budget and refuse: silently looping on an unremovable file would hang.
        _note(f"the write lock on '{(rec or {}).get('collection', lockpath.stem)}' "
              f"looks abandoned but could not be removed ({e}); waiting instead.")
        return False
    age = time.time() - current_mtime
    _note(f"reclaimed the write lock on "
          f"'{(rec or {}).get('collection', lockpath.stem)}': its holder "
          f"({describe_holder(rec, current_mtime)}) had not reported for "
          f"{_duration(age)}, so it appears to have crashed without releasing it.")
    return True
