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
directory reads as collection data. ``check_collection_name`` forbids ``.``
in a collection name, so ``<name>.lock`` can never collide with a
collection directory, and ``collection_names()`` only lists directories that
hold a meta.json, so the lock file is never mistaken for a collection.

Identity is the per-acquisition ``token`` (uuid4), never the pid. The record
also names the holder's ``pid``, the pid space that pid belongs to
(``machine``) and the holder's ``start`` identity
(``instances.process_start_identity``), which no change of the system clock
alters. A waiter decides whether the holder still holds the lock
(``_is_stale``) by what it can establish about that process
(``_holder_liveness``):

  * a holder proven running keeps the lock however old its heartbeat is;
  * a holder proven gone (its pid has exited, or now names a process with
    another start identity) is taken over once its heartbeat is
    ``DEAD_HOLDER_GRACE`` old;
  * any other holder (a record from another pid space, one without a start
    identity, one written by an earlier localm, a probe that cannot answer) is
    taken over once its heartbeat is ``STALE_AFTER`` old AND the waiter has
    itself watched ``CONFIRM_BEATS`` heartbeats go missing, timed on its own
    monotonic clock.

So a heartbeat age read from the wall clock never, on its own, takes the lock
from a holder that is still running or still beating.

Failure is never silent and never optimistic:

  * A lock that cannot be acquired raises ``CollectionLockedError``. There is no
    path that proceeds to write without holding it.
  * A lock file whose record is corrupt or unreadable is treated as HELD by a
    holder whose liveness is unknown, never as free.
  * Release removes the file only on a POSITIVE token match. "I cannot read it"
    is never taken as "it must be mine". A file it cannot remove it empties
    through a handle whose record carries its token, so no record is left
    naming a holder that has released.
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
# A holder whose liveness cannot be established (see _holder_liveness) is
# presumed crashed once its heartbeat is this old and a waiter has seen
# CONFIRM_BEATS heartbeats go missing. NOT a limit on how long a lock may be
# held: a live holder beats every HEARTBEAT_INTERVAL.
STALE_AFTER = 60.0
# Heartbeats a waiter must itself see go missing, timed on its own monotonic
# clock, before STALE_AFTER counts against such a holder. See
# test_the_confirm_window_fits_inside_the_wait_budget.
CONFIRM_BEATS = 3
# A holder proven gone (see _holder_liveness) is taken over once its heartbeat
# is this old.
DEAD_HOLDER_GRACE = 4 * HEARTBEAT_INTERVAL
# Attempts a holder makes at removing its own lock file before emptying it.
_UNLINK_TRIES = 5
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


# Tokens of the acquisitions in this process that currently hold, or are
# acquiring, a lock.
_held_tokens: set = set()
_held_tokens_lock = threading.Lock()


def _holder_liveness(rec: dict) -> str:
    """``"alive"``, ``"dead"`` or ``"unknown"`` for the holder *rec* names.

    ``"alive"``: the record names this pid space and a start identity, and the
    process now running under its pid has that start identity; a record naming
    this process counts only while the acquisition that wrote it still holds
    the lock. ``"dead"``: the pid has exited, now names a process with another
    start identity, or is this process's under a released acquisition.
    ``"unknown"``: anything else, including a record from another pid space,
    one without a start identity (every record written by an earlier localm)
    and a probe that cannot answer.
    """
    from localm import instances
    if rec.get("machine") != _machine_id():
        return "unknown"
    pid = rec.get("pid")
    recorded = rec.get("start")
    if type(pid) is not int or pid <= 0 or not isinstance(recorded, dict):
        return "unknown"
    try:
        if not instances.pid_alive(pid):
            return "dead"
        current = instances.process_start_identity(pid)
    except Exception as e:
        _log.debug("rag lock: liveness probe for pid %s failed (%s)", pid, e)
        return "unknown"
    if instances.start_identity_differs(recorded, current):
        return "dead"
    if not instances.start_identity_matches(recorded, current):
        return "unknown"
    if pid == os.getpid():
        token = rec.get("token")
        with _held_tokens_lock:
            if not (isinstance(token, str) and token in _held_tokens):
                return "dead"
    return "alive"


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


def _is_stale(rec: Optional[dict], mtime: Optional[float], stale_after: float,
              quiet: float) -> bool:
    """Whether the holder of a lock file no longer holds it, by the rules in
    the module docstring.

    The heartbeat's age is the lock file's mtime against this process's wall
    clock. *quiet* is how long, on the caller's monotonic clock, the caller has
    seen this record and mtime stay unchanged. A corrupt or unreadable record
    is judged as a holder whose liveness is unknown."""
    if mtime is None:
        return False              # the file vanished; the caller re-tries the create
    # A holder whose clock runs ahead of ours yields a negative age. Clamp to 0
    # (treat as fresh) rather than letting arithmetic decide to steal a lock.
    age = max(0.0, time.time() - mtime)
    if age <= min(DEAD_HOLDER_GRACE, stale_after):
        return False
    liveness = _holder_liveness(rec) if isinstance(rec, dict) else "unknown"
    if liveness == "alive":
        return False
    if liveness == "dead":
        return age > DEAD_HOLDER_GRACE
    return age > stale_after and quiet >= CONFIRM_BEATS * HEARTBEAT_INTERVAL


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


def _empty_own_lock(lockpath: Path, token: str) -> Optional[bool]:
    """Truncate *lockpath* to nothing through one handle, only when the record
    read through that same handle carries *token*.

    True when it was emptied, False when its record does not carry *token* (or
    cannot be parsed), None when the file could not be opened, read or
    truncated."""
    try:
        with open(lockpath, "r+b") as f:
            try:
                rec = json.loads(f.read().decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return False
            if not isinstance(rec, dict) or rec.get("token") != token:
                return False
            f.seek(0)
            f.truncate()
    except OSError:
        return None
    return True


def _remove_own_lock(lockpath: Path, token: str, collection: str,
                     stale_after: float) -> None:
    """Remove the lock file this acquisition created.

    A refused unlink is retried up to ``_UNLINK_TRIES`` times in all, each
    retry only while the file's record is this acquisition's or unreadable.
    If it is still there, the file is emptied (see _empty_own_lock). Every
    outcome other than a removal is reported."""
    err: Optional[OSError] = None
    for attempt in range(_UNLINK_TRIES):
        if attempt:
            time.sleep(min(_POLL * 2 ** attempt, _POLL_CAP))
            rec, _ = _read_record(lockpath)
            if isinstance(rec, dict) and rec.get("token") != token:
                _note(f"the write lock on '{collection}' was taken over by "
                      f"another localm process before this write finished; "
                      f"leaving their lock in place.")
                return
        try:
            lockpath.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as e:
            err = e
    if not lockpath.exists():
        return
    emptied = _empty_own_lock(lockpath, token)
    if emptied:
        _note(f"could not remove the write lock file for '{collection}' "
              f"({err}); emptied it instead, so it is reclaimed as stale "
              f"after {stale_after:.0f}s.")
    elif emptied is False:
        _note(f"could not remove the write lock file for '{collection}' "
              f"({err}); it is reclaimed as stale after {stale_after:.0f}s.")
    else:
        _note(f"could not remove or empty the write lock file {lockpath} "
              f"({err}). Until this process exits, other localm processes on "
              f"this machine wait for it; deleting that file releases "
              f"'{collection}'.")


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
    token = uuid.uuid4().hex
    with _held_tokens_lock:
        _held_tokens.add(token)
    try:
        with _hold_lock_file(lockpath, token, collection=collection, op=op,
                             timeout=timeout, stale_after=stale_after,
                             on_wait=on_wait, kind=kind):
            yield
    finally:
        with _held_tokens_lock:
            _held_tokens.discard(token)


@contextlib.contextmanager
def _hold_lock_file(lockpath: Path, token: str, *, collection: str, op: str,
                    timeout: Optional[float], stale_after: Optional[float],
                    on_wait: Optional[Callable[[str], None]], kind: str):
    """Acquire, hold and release *lockpath* under *token* for
    :func:`collection_write_lock`, which takes the same arguments."""
    from localm import instances
    timeout = _env_float(ENV_WAIT, WAIT_TIMEOUT) if timeout is None else timeout
    stale_after = (_env_float(ENV_STALE, STALE_AFTER)
                   if stale_after is None else stale_after)
    pid = os.getpid()
    record = {
        "token": token,
        "pid": pid,
        "start": instances.process_start_identity(pid),
        "machine": _machine_id(),
        "collection": collection,
        "op": op,
        "started": time.time(),
    }
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    started_waiting = time.time()
    announced = False
    attempt = 0
    # The (token, mtime) last read from an existing lock file, and when this
    # waiter first read it, on the monotonic clock.
    watched = None
    watched_since = 0.0

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
            now = time.monotonic()
            seen = (rec.get("token") if isinstance(rec, dict) else None, mtime)
            if seen != watched:
                watched, watched_since = seen, now
            quiet = now - watched_since
            if (mtime is not None and _is_stale(rec, mtime, stale_after, quiet)
                    and _reclaim(lockpath, rec, mtime, stale_after, quiet)):
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
            _remove_own_lock(lockpath, token, collection, stale_after)
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
            _remove_own_lock(lockpath, token, collection, stale_after)
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
             stale_after: float, quiet: float) -> bool:
    """Remove a lock whose holder stopped proving it was alive. True when the
    file is gone afterwards and the caller may retry its create.

    Re-reads the file first and leaves it alone when its mtime is no longer the
    *mtime* it was judged by, when its record is not the one judged, or when
    the SAME staleness test no longer holds. The residual window (a lock
    created between this re-check and the unlink below) cannot be closed with
    plain files; the fencing token stops it from cascading, since the
    wrongly-removed holder's own release will not then delete a third party's
    lock."""
    current, current_mtime = _read_record(lockpath)
    if current_mtime is None:
        return True               # already gone: the acquire loop can proceed
    if current_mtime != mtime:
        return False              # its heartbeat moved since it was judged
    if not _is_stale(current, current_mtime, stale_after, quiet):
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
