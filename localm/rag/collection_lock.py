# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-process write lock for a single RAG collection.

``store._collection_lock`` serialises writers inside ONE process. It cannot
serialise `localm rag add|resync|repair|rm` (its own OS process, its own lock
registry) against a running server's scheduled re-sync of the same collection:
both ``_load()`` the same state, mutate their copy and ``_save()``, so one
update is silently lost and interleaved meta/chunks/vectors can surface later
as a degraded index. This module closes that, per collection, across processes.

A hold here has NO wall-clock limit. Indexing a folder legitimately runs for
minutes or hours, so a ``config._cross_process_lock``-style rule that reclaims
ANY holder older than 30 s would reap a LIVE holder and let both write. The
holder instead proves it is alive with a HEARTBEAT, and staleness is keyed on
the age of that heartbeat (``STALE_AFTER``), not on how long the lock has been
held.

THE HEARTBEAT IS THE LOCK FILE'S MTIME, refreshed with ``os.utime``, not a
timestamp field rewritten inside the record. Rewriting the record means
replacing the file, and a replace cannot be made conditional: a holder whose
write stalls past the staleness window (an
unresponsive network share, a long antivirus hold) would land its now-stale
record ON TOP of the record of whichever process legitimately reclaimed the
lock meanwhile - destroying the successor's identity and letting a third writer
in behind it. ``os.utime`` only ever moves a timestamp, so the worst a stalled
holder can do is make a live successor's lock look a few seconds fresher than
it is, which is harmless because that successor IS alive. The record itself is
written exactly once, at acquisition, and never rewritten.

The lock file is ``<data dir>/rag/<name>.lock``, a SIBLING of the collection
directory rather than a file inside it: ``delete_collection``'s rmtree would
destroy an inside lock while it was held, and a stray file in the collection
directory reads as collection data. Collection names are ``[A-Za-z0-9_-]{1,64}``
(``check_collection_name``), so ``<name>.lock`` can never collide with a
collection directory, and ``collection_names()`` only lists directories that
hold a meta.json, so the lock file is never mistaken for a collection.

Identity is the per-acquisition ``token`` (uuid4), never the pid: pids are
reused across process lifetimes, so a leaked lock file can carry the very pid
the OS later hands to a new localm process. The record ALSO pins ``(pid, pid_create_time)``, which is what makes a pid
usable as evidence at all: same pid + different create time means the number
was recycled and the real holder is gone. That pin is only consulted when the
record was written by a process that shares this one's pid space
(``machine``), because a LOCALM_HOME on a network share - or on the Windows
side of a WSL mount - can hold a pid from an entirely different pid table,
where a local lookup would be worse than useless. It needs psutil, which is an
EXTRA here, not a core dependency; without it the pin is inert and staleness
falls back to heartbeat age alone, which is the primary rule anyway. Nothing
depends on the accelerator being available.

Failure is never silent and never optimistic:

  * A lock that cannot be acquired raises ``CollectionLockedError``. There is no
    path that proceeds to write without holding it.
  * A lock file whose record is corrupt or unreadable is treated as HELD (until
    its mtime goes stale), never as free.
  * Release removes the file only on a POSITIVE token match. "I cannot read it"
    is never taken as "it must be mine".
  * A reclaim is printed, and so is the case where this process's own hold was
    reclaimed while it was still running.
"""

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
# this.
HEARTBEAT_INTERVAL = 5.0
# A holder that has not refreshed its heartbeat for this long is presumed
# crashed and its lock is reclaimed. NOT a limit on how long a lock may be held:
# a live holder beats every HEARTBEAT_INTERVAL, so a run of any length stays
# fresh. 12 missed beats, so a badly stalled but living process (an antivirus
# scan, a paused debugger) is not mistaken for a dead one.
STALE_AFTER = 60.0
# When the pin PROVES the holder is gone (pid absent, or recycled into another
# process), waiting out the full STALE_AFTER only delays recovery from a crash.
# Still LONGER than the heartbeat failures a live holder tolerates (_Heartbeat
# keeps going through a transient utime failure), so a WRONG "dead" verdict
# cannot outrace a living holder's own margin.
DEAD_HOLDER_GRACE = 4 * HEARTBEAT_INTERVAL
# How long a would-be writer waits for the lock before refusing. Bounded: an
# unbounded wait turns a stuck peer into a hung CLI or a hung job.
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
    """Another writer holds this collection's write lock and did not release it
    within the wait budget. Nothing was written."""

    def __init__(self, name: str, holder: Optional[dict], waited: float,
                 last_alive: Optional[float] = None,
                 lockpath: Optional[Path] = None, same_process: bool = False,
                 kind: str = "Collection"):
        # *kind* names WHAT is locked, for the message only. It defaults to
        # "Collection" for the RAG raise sites; agent memory passes "Memory
        # namespace" (see memory/store.py), since the same machinery serialises
        # both.
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
    """The current wait-before-refusing budget, honouring the env override.

    Public so a caller that has to bound its OWN waiting (delete_collection
    bounds the in-process half too) uses the same number the file lock does,
    rather than inventing a second one that could drift from the docs."""
    return _env_float(ENV_WAIT, WAIT_TIMEOUT)


def lock_path_for(collection_dir: Path) -> Path:
    """The lock file for the collection stored at *collection_dir* (a sibling
    ``<name>.lock``, see the module docstring)."""
    return collection_dir.with_name(collection_dir.name + ".lock")


def describe_holder(rec: Optional[dict], last_alive: Optional[float] = None) -> str:
    """A human sentence naming who holds a lock, from its record.

    Says only what the record knows: the pid, what it is doing, how long it has
    held the lock and when it last proved it was alive. No hostname and no
    command line."""
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
    """An operator override, or *default* if it is unset or not a usable number.

    A malformed value is reported rather than silently ignored."""
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
    """An opaque, stable id for THIS PID SPACE.

    Not just the host: a pid only means something within one pid table, and a
    hostname does not identify one. WSL2 defaults its hostname to the Windows
    machine name, and a LOCALM_HOME shared across that boundary (a /mnt/c path)
    would otherwise let each side look the other's pids up in its own process
    table, find nothing, and declare a perfectly live holder dead. So the
    platform and, where the kernel exposes it, the pid namespace go into the id
    as well.

    Hashed rather than stored plainly: the node name is a personal identifier
    and this record is written into the user's data directory. Only ever
    compared for equality, so the hash is as good as the name."""
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
    """Process start time for *pid*, or None when it cannot be determined
    (psutil absent - it is an extra, not a core dependency - or the process is
    already gone)."""
    try:
        import psutil
    except Exception:
        return None
    try:
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _holder_liveness(rec: dict) -> str:
    """``"dead"``, ``"alive"`` or ``"unknown"`` for the process in *rec*.

    Only ever used to reclaim a crashed holder EARLIER than the heartbeat rule
    would. It never keeps a stale lock alive, so "unknown" (no psutil, another
    pid space, nothing pinned) costs nothing but a slower recovery."""
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
    """``(record_or_None, mtime_or_None)`` for the lock file.

    ``(None, mtime)`` means the file EXISTS but its record could not be read: a
    hand-edit, a truncated file, an ACL that permits stat but not read, or the
    brief moment between another process creating the file and writing its
    record into it. That is treated as held until the file itself goes stale -
    never as free, which would let a second writer in exactly when the on-disk
    state is already suspect. The stat is taken SEPARATELY, and first, so an
    unreadable file still gets a staleness clock instead of being unjudgeable
    and therefore held for ever."""
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
    """Whether the holder stopped proving it was alive.

    The clock is the lock file's mtime, which the holder refreshes (see the
    module docstring), so a corrupt record is judged by exactly the same rule as
    a readable one."""
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
    """Keeps the holder's lock file looking alive until the lock is released.

    A daemon thread, so it can never keep a process alive; and it only ever
    touches the lock file's timestamp, so it cannot interfere with the indexing
    run it is vouching for, nor overwrite anybody's record."""

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
        """One refresh. False when we no longer hold the lock and must stop."""
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
            # judged us stale and removed it. NOT re-created - a fresh file here
            # would collide with whoever is taking over, and a lock re-created
            # during release outlives the run that owned it. Report and stand
            # down instead.
            _note(f"the write lock file for '{self._record.get('collection')}' "
                  f"was removed while this process still held it. Another "
                  f"localm process may now be writing to the same collection.")
            return False
        except OSError as e:
            # Transient (an antivirus scanner holding the file, a full disk).
            # Missing ONE beat is harmless - STALE_AFTER is twelve of them - so
            # keep going rather than abandoning a lock we still hold. A RUN of
            # them ends with somebody reclaiming this lock while we are still
            # writing, so it escalates rather than staying a debug line.
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
    """Surface an unusual lock event through BOTH channels, always.

    stderr is for whoever is watching a terminal; the log is the durable record.
    The log is never conditional: every localm entry point installs the
    always-on ring buffer (debuglog.install_ring_buffer, called from
    cli/_core.py), which is what a bug report dumps, and a run launched without
    a console has no usable stderr at all."""
    print(f"[localm] note: {message}", file=sys.stderr)
    _log.warning("rag lock: %s", message)


@contextlib.contextmanager
def collection_write_lock(lockpath: Path, *, collection: str, op: str,
                          timeout: Optional[float] = None,
                          stale_after: Optional[float] = None,
                          on_wait: Optional[Callable[[str], None]] = None,
                          kind: str = "Collection"):
    """Hold the cross-process write lock for a collection, or refuse.

    Raises ``CollectionLockedError`` if another process still holds it after
    *timeout* seconds. It never returns without the lock: there is no
    "carry on unprotected" path.

    *on_wait* is called with a progress line if the wait actually lasts (see
    WAIT_NOTICE_AFTER), so a CLI can say why it is sitting there instead of
    looking hung. Callers pass their existing progress channel.
    """
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
            # that clears on its own in milliseconds.
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
            # below. Retrying a reclaim from here instead would spin without
            # ever consulting the deadline.
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
            # Inside the cleanup block: a thread that cannot start (thread
            # exhaustion) would otherwise leave a lock file behind with nothing
            # refreshing it, blocking this collection until it went stale.
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
    """Remove a lock whose holder stopped proving it was alive. True when the
    file is gone afterwards and the caller may retry its create.

    Re-reads the file and re-applies the SAME staleness test before unlinking,
    so a lock that was released and freshly re-taken between our judgement and
    this call is left alone - including the case where neither record could be
    read, where a token comparison would be meaningless but the refreshed mtime
    still says the new holder is alive. The residual window (a lock created
    between this re-check and the unlink below) cannot be closed with plain
    files; the fencing token stops it from cascading, since the wrongly-removed
    holder's own release will not then delete a third party's lock."""
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
