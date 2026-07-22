# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-process write lock for a single RAG collection.

``store._collection_lock`` serialises writers inside ONE process. It cannot
serialise `localm rag add|resync|repair` (its own OS process, its own lock
registry) against a running server's scheduled re-sync of the same collection:
both ``_load()`` the same state, mutate their copy and ``_save()``, so one
update is silently lost and interleaved meta/chunks/vectors can surface later
as a degraded index. This module closes that, per collection, across processes.

Why not ``config._cross_process_lock``: it reclaims ANY holder older than 30 s
as abandoned. That is right for a config read-modify-write (milliseconds) and
fatal here, where indexing a folder legitimately runs for minutes or hours - a
waiter would reap a LIVE holder and both would write. So a hold here has NO
wall-clock limit at all. The holder instead proves it is alive with a
HEARTBEAT it refreshes every ``HEARTBEAT_INTERVAL``, and staleness is keyed on
the age of that heartbeat (``STALE_AFTER``), not on how long the lock has been
held.

The lock file is ``<data dir>/rag/<name>.lock``, a SIBLING of the collection
directory rather than a file inside it: ``delete_collection``'s rmtree would
destroy an inside lock while it was held, and a stray file in the collection
directory reads as collection data. Collection names are ``[A-Za-z0-9_-]{1,64}``
(``check_collection_name``), so ``<name>.lock`` can never collide with a
collection directory, and ``collection_names()`` only lists directories that
hold a meta.json, so the lock file is never mistaken for a collection.

Identity is the per-acquisition ``token`` (uuid4), never the pid: pids are
reused across process lifetimes, so a leaked lock file can carry the very pid
the OS later hands to a new localm process (REG-586, learned in config.py).
The record ALSO pins ``(pid, pid_create_time)``, which is what makes a pid
usable as evidence at all: same pid + different create time means the number
was recycled and the real holder is gone. That pin is only consulted when the
record was written on this machine (``machine``), because a LOCALM_HOME on a
network share can hold another box's pid, where a local pid lookup would be
meaningless. It needs psutil, which is an EXTRA here, not a core dependency;
without it the pin is inert and staleness falls back to heartbeat age alone,
which is the primary rule anyway. Nothing depends on the accelerator being
available.

Failure is never silent and never optimistic:

  * A lock that cannot be acquired raises ``CollectionLockedError``. There is no
    path that proceeds to write without holding it.
  * A lock file whose record is corrupt or unreadable is treated as HELD (until
    its file mtime goes stale), never as free.
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

from localm.debuglog import debug_enabled
from localm.debuglog import logger as _log
from localm.storekit import atomic_write

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
# Still not zero: a holder that died a moment ago may have a sibling about to
# take over, and a crash is exactly when a little conservatism is cheap.
DEAD_HOLDER_GRACE = 2 * HEARTBEAT_INTERVAL
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
    """Another process holds this collection's write lock and did not release it
    within the wait budget. Nothing was written."""

    def __init__(self, name: str, holder: Optional[dict], waited: float):
        self.collection = name
        self.holder = holder
        self.waited = waited
        super().__init__(
            f"Collection '{name}' is being written by {describe_holder(holder)}. "
            f"Waited {_duration(waited)} and gave up; nothing was changed. Let "
            f"that run finish and try again.")


def lock_path_for(collection_dir: Path) -> Path:
    """The lock file for the collection stored at *collection_dir* (a sibling
    ``<name>.lock``, see the module docstring)."""
    return collection_dir.with_name(collection_dir.name + ".lock")


def describe_holder(rec: Optional[dict]) -> str:
    """A human sentence naming who holds a lock, from its record.

    Deliberately says only what the record honestly knows: the pid, what it is
    doing, how long it has held the lock and when it last proved it was alive.
    No hostname or command line (this file lives in the user's data directory
    and can end up quoted in a bug report)."""
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
    hb = rec.get("heartbeat")
    if isinstance(hb, (int, float)):
        bits.append(f"last heartbeat {_duration(time.time() - hb)} ago")
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

    A malformed value is reported rather than silently ignored: someone who set
    it meant something by it, and a typo that quietly reverts to the default is
    exactly the kind of "it looked like it worked" this codebase does not ship."""
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
    """An opaque, stable id for this machine.

    Hashed rather than stored plainly: the node name is a personal identifier
    and this record is written into the user's data directory (AGENTS.md rule
    2). Only ever compared for equality, so the hash is as good as the name."""
    global _machine_id_cache
    if _machine_id_cache is None:
        try:
            node = platform.node() or ""
        except Exception:            # platform.node() is not documented to raise
            node = ""                # but an unknown host must not break locking
        _machine_id_cache = hashlib.sha256(
            node.encode("utf-8", "replace")).hexdigest()[:16]
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
    machine's record, nothing pinned) costs nothing but a slower recovery."""
    if rec.get("machine") != _machine_id():
        return "unknown"          # another box's pid says nothing about ours
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

    ``(None, mtime)`` means the file EXISTS but its record could not be read
    (a partial write, a hand-edit, a truncated file). That is treated as held
    until the file itself goes stale - never as free, which would let a second
    writer in exactly when the on-disk state is already suspect."""
    try:
        raw = lockpath.read_bytes()
        mtime = lockpath.stat().st_mtime
    except OSError:
        return None, None
    try:
        rec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, mtime
    return (rec if isinstance(rec, dict) else None), mtime


def _last_alive(rec: Optional[dict], mtime: Optional[float]) -> Optional[float]:
    """When the holder last proved it was alive.

    The heartbeat field when it is readable; otherwise the file's own mtime,
    which tracks it anyway because every heartbeat rewrites the file. That
    fallback is what gives a CORRUPT record a staleness clock of its own."""
    if isinstance(rec, dict):
        hb = rec.get("heartbeat")
        if isinstance(hb, (int, float)):
            return float(hb)
    return mtime


def _is_stale(rec: Optional[dict], mtime: Optional[float], stale_after: float) -> bool:
    last = _last_alive(rec, mtime)
    if last is None:
        return False              # the file vanished; the caller re-tries the create
    # A holder whose clock runs ahead of ours yields a negative age. Clamp to 0
    # (treat as fresh) rather than letting arithmetic decide to steal a lock.
    age = max(0.0, time.time() - last)
    if age > stale_after:
        return True
    if isinstance(rec, dict) and _holder_liveness(rec) == "dead":
        return age > DEAD_HOLDER_GRACE
    return False


class _Heartbeat(threading.Thread):
    """Refreshes the holder's heartbeat until the lock is released.

    A daemon thread, so it can never keep a process alive; and it only ever
    touches the lock file, so it cannot interfere with the indexing run it is
    vouching for."""

    def __init__(self, lockpath: Path, record: dict, interval: float):
        super().__init__(name=f"rag-lock-{record.get('op', 'write')}", daemon=True)
        self._lockpath = lockpath
        self._record = record
        self._interval = interval
        # NOT `_stop`: threading.Thread already uses that name internally, and
        # shadowing it breaks join() at interpreter level.
        self._stopping = threading.Event()
        self.lost = False          # our lock was taken over while we held it

    def run(self) -> None:
        while not self._stopping.wait(self._interval):
            if not self._beat():
                return

    def _beat(self) -> bool:
        """One refresh. False when we no longer hold the lock and must stop."""
        rec, _ = _read_record(self._lockpath)
        token = self._record["token"]
        if rec is None and not self._lockpath.exists():
            # The file is gone while we are demonstrably alive: a waiter judged
            # us stale and unlinked us, and has not created its own record yet.
            # Re-establish OURS with an exclusive create, which cannot clobber a
            # record that waiter may have just written - if it got there first we
            # lose the create and find out below that the lock is now theirs.
            if self._create_exclusive():
                _log.warning("rag lock %s was removed while this process still "
                             "held it; re-established it", self._lockpath.name)
                return True
            rec, _ = _read_record(self._lockpath)
        if isinstance(rec, dict) and rec.get("token") != token:
            self.lost = True
            _note(f"another localm process took over the write lock on "
                  f"'{self._record.get('collection')}' while this process was "
                  f"still writing to it. Both may now be writing; check the "
                  f"collection with 'localm rag list' when both runs finish.")
            return False
        if self._stopping.is_set():
            # Released while we were reading. Writing now would RE-CREATE the
            # lock file the releasing thread is about to remove (or has just
            # removed), wedging the collection until that phantom went stale.
            return False
        self._record["heartbeat"] = time.time()
        try:
            atomic_write(self._lockpath, json.dumps(self._record))
        except OSError as e:
            # Transient (an antivirus scanner holding the file, a full disk).
            # Missing ONE beat is harmless - STALE_AFTER is twelve of them - so
            # keep going rather than abandoning a lock we still hold, but say so
            # at debug level so a persistent failure is discoverable.
            _log.debug("rag lock heartbeat write failed for %s (%s)",
                       self._lockpath.name, e)
        return True

    def _create_exclusive(self) -> bool:
        self._record["heartbeat"] = time.time()
        try:
            fd = os.open(str(self._lockpath),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return False
        try:
            os.write(fd, json.dumps(self._record).encode("utf-8"))
        finally:
            os.close(fd)
        return True

    def stop(self) -> None:
        self._stopping.set()
        self.join(timeout=self._interval + 1.0)


def _note(message: str) -> None:
    """Surface an unusual lock event: on stderr for whoever is watching, and in
    the debug log so an unattended run leaves a durable record (AGENTS.md rule
    5 - a problem nobody can see has not been reported).

    The log call is gated on debug mode only to avoid printing the SAME line
    twice: with no handler configured, logging's last-resort handler writes
    WARNING records straight to stderr, right next to the print above."""
    print(f"[localm] note: {message}", file=sys.stderr)
    if debug_enabled():
        _log.warning("rag lock: %s", message)


@contextlib.contextmanager
def collection_write_lock(lockpath: Path, *, collection: str, op: str,
                          timeout: Optional[float] = None,
                          stale_after: Optional[float] = None,
                          on_wait: Optional[Callable[[str], None]] = None):
    """Hold the cross-process write lock for a collection, or refuse.

    Raises ``CollectionLockedError`` if another process still holds it after
    *timeout* seconds. It never returns without the lock: there is no
    "carry on unprotected" path, because an unserialised write is precisely the
    lost update this exists to prevent.

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
        "heartbeat": time.time(),
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
        except FileExistsError:
            rec, mtime = _read_record(lockpath)
            unreadable = rec is None and mtime is None
            if not unreadable:
                if _is_stale(rec, mtime, stale_after):
                    _reclaim(lockpath, rec, mtime)
                    continue
            # `unreadable` means the file existed for the create and was gone (or
            # unopenable) a syscall later. Usually the holder releasing, so retry -
            # but through the SAME deadline, never in a tight spin: a lock file we
            # can never read (a permissions fault) must end in a refusal, not a
            # busy loop that looks like a hang.
            waited = time.time() - started_waiting
            if time.time() >= deadline:
                raise CollectionLockedError(collection, rec, waited)
            if on_wait and not announced and waited >= WAIT_NOTICE_AFTER:
                announced = True
                on_wait(f"waiting for the write lock on '{collection}': "
                        f"{describe_holder(rec)}")
            time.sleep(min(_POLL * (attempt + 1), _POLL_CAP))
            attempt += 1
            continue
        # We created the file, so from here every failure must remove OUR file,
        # or a transient write error leaks a lock nobody owns that blocks every
        # writer of this collection until it goes stale.
        try:
            try:
                os.write(fd, json.dumps(record).encode("utf-8"))
            finally:
                os.close(fd)
        except BaseException:
            try:
                lockpath.unlink()
            except OSError:
                pass
            raise
        break

    beat = _Heartbeat(lockpath, record, HEARTBEAT_INTERVAL)
    beat.start()
    try:
        yield
    finally:
        beat.stop()
        if beat.is_alive():
            # It should have exited the moment the stop event was set. Still
            # running means it is stuck in a filesystem call and could re-write
            # the lock file after the release below - say so instead of leaving a
            # phantom lock to be explained later.
            _note(f"the heartbeat for '{collection}' did not stop; if a lock file "
                  f"reappears for this collection it is that thread, and it goes "
                  f"stale in {stale_after:.0f}s.")
        # Fencing release: remove the file ONLY while it still carries the token
        # this call wrote. If another process reclaimed it as stale while we were
        # legitimately still inside the critical section, deleting it here would
        # delete THEIR live lock and let a third writer in - the very race this
        # mechanism exists to close, rebuilt through its own recovery path
        # (config.py learned this one the hard way).
        rec, _ = _read_record(lockpath)
        if rec is not None and rec.get("token") != token:
            _note(f"the write lock on '{collection}' was taken over by another "
                  f"localm process before this {op} finished; leaving their lock "
                  f"in place.")
        elif rec is not None or lockpath.exists():
            try:
                lockpath.unlink()
            except OSError as e:
                # Leaving it behind is not silent damage: it goes stale and is
                # reclaimed. Say so rather than pretending the release worked.
                _note(f"could not remove the write lock file for '{collection}' "
                      f"({e}); it will be reclaimed as stale in "
                      f"{stale_after:.0f}s.")


def _reclaim(lockpath: Path, rec: Optional[dict], mtime: Optional[float]) -> None:
    """Remove a lock whose holder stopped proving it was alive.

    Re-reads and compares the token first, so a lock that was released and
    freshly re-taken between our staleness judgement and this call is left
    alone. That check narrows, but does not fully close, the window in which an
    unrelated process could create a NEW lock between the compare and the
    unlink; closing it completely needs OS-level file locking, which does not
    work reliably on the network shares a LOCALM_HOME can live on. The residual
    requires a crashed holder AND microsecond-scale interleaving, and the
    fencing token still stops it from cascading: the wrongly-removed holder's
    own release cannot then delete a third party's lock."""
    age = time.time() - (_last_alive(rec, mtime) or time.time())
    current, _ = _read_record(lockpath)
    same = (current is None and rec is None) or (
        isinstance(current, dict) and isinstance(rec, dict)
        and current.get("token") == rec.get("token"))
    if not same:
        return                    # somebody else's fresh lock now: not ours to remove
    _note(f"reclaiming the write lock on "
          f"'{(rec or {}).get('collection', lockpath.stem)}': its holder "
          f"({describe_holder(rec)}) has not reported for {_duration(age)}, so it "
          f"appears to have crashed without releasing it.")
    try:
        lockpath.unlink()
    except OSError:
        pass                      # already gone: the acquire loop re-tries anyway
