# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared store mechanics.

JSONL stores: atomic tmp+replace writes and a lazy per-namespace lock registry,
used by ``localm/rag/store.py`` and ``localm/memory/store.py``.

JSON stores (config, registry, the key store, credentials, sessions and the
other owner-private files): the checked read (``read_json_state``), the
owner-restricted write (``atomic_write_private``) and the cross-process store
lock (``store_lock``), whose staleness follows whether the holder is still
running (see ``localm.procid``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

from localm import procid, sysprog

logger = logging.getLogger("localm")


def atomic_write(path: Path, data: str | Iterable[str]) -> None:
    """Write *data* to *path* atomically: write a unique per-writer temp
    file, then replace *path* with it. *data* is a string, or an iterable of
    strings written one after another.

    The temp name is unique per (pid, thread), so two independent writers
    never collide on the same temp file even outside a lock (e.g. a sidecar
    file write) - on top of, not instead of, each store's own per-namespace
    lock serializing writers to a given target. Retries the replace on
    ``PermissionError``: on Windows, an AV real-time scanner or the Search
    Indexer can transiently hold a handle to *path*.

    The temp file is always cleaned up, so a permanently-failing replace leaves
    no orphan behind.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        if isinstance(data, str):
            tmp.write_text(data, encoding="utf-8")
        else:
            with open(tmp, "w", encoding="utf-8") as fh:
                for piece in data:
                    fh.write(piece)
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        # Removes the temp file when the write or the replace failed; a no-op
        # on success.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as e:
            # Best-effort: the write's outcome is already decided.
            logger.debug("storekit: could not remove temp file %s (%s)", tmp, e)


class NamespaceLockRegistry:
    """Lazily-created ``threading.RLock`` per key, behind one shared creation
    guard.

    Serializes concurrent writers to the SAME on-disk store: two request
    handlers each construct their own ``Collection``/``MemoryStore`` instance,
    ``_load()`` the same state, mutate their own copy and ``_save()``. The lock
    is keyed by name/namespace, not per instance, so it serializes
    process-wide. RLock, so a locked method may call another locked method on
    the same key without deadlocking.
    """

    def __init__(self) -> None:
        self._locks: dict = {}
        self._guard = threading.Lock()

    def get(self, key: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock


def _perm_warn(path: Path, why: str) -> None:
    """Record a failed permission tightening at debug level.

    Debug level rather than a warning: the data dir is already user-scoped, so a
    failure here degrades defence in depth without exposing a plaintext
    secret."""
    logger.debug("could not restrict permissions on %s (%s); the data "
                 "directory's own scoping still applies", path.name, why)


def restrict_file_perms(path: Path, *, mode: int = 0o600) -> bool:
    """Best-effort: restrict *path* to the current user (POSIX chmod *mode*, or
    Windows icacls - which grants the user full control regardless of *mode*,
    since ACLs do not encode POSIX bits; inherited entries are dropped, and a
    file created without inherited entries keeps its default SYSTEM and
    Administrators ACEs). Returns True when the tightening is believed
    to have happened, False when it did not. No-op on failure - the data dir is
    already user-scoped, so this per-file tightening is defence in depth.

    *mode* defaults to 0o600 for a FILE. A caller locking down a DIRECTORY must
    pass 0o700 (or another mode with the execute/search bit set) - 0o600 on a
    directory removes POSIX traversal (x), so every file inside becomes
    unreachable by path even though it still lists in a directory scan.

    On Windows ``os.replace`` carries the source's ACL onto the destination,
    overwriting the destination's inherited one, and POSIX rename likewise
    carries the source's mode. So a caller doing the atomic temp+replace dance
    can restrict the TEMP file (which already holds the whole payload) and skip
    a second call on the destination in the happy path; when this returns False,
    retry on the destination.

    Shared by the project's credential-bearing files and directories -
    ``auth.key``, ``sessions.json``, ``jobs.json``, the local CA material via
    ``tls`` (which passes ``mode=0o700``), and everything written through
    ``atomic_write_private`` - so each gets the same treatment on both platforms.
    """
    try:
        if os.name == "posix":
            os.chmod(path, mode)
        else:
            import subprocess
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if not user:
                _perm_warn(path, "no USERNAME/USER in the environment")
                return False
            # /inheritance:r drops the inherited ACEs (BUILTIN\Users et al) and
            # /grant:r replaces any existing grant for this user, so the user
            # holds an explicit full-control ACE (SYSTEM and Administrators may remain).
            icacls = sysprog.find("icacls")
            if not icacls:
                _perm_warn(path, "icacls was not found in the Windows system folder")
                return False
            r = subprocess.run(
                [icacls, str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                capture_output=True, check=False)
            if r.returncode != 0:
                # icacls FAILS without raising (access denied, an unresolvable
                # principal, a non-NTFS volume), so a non-zero exit is reported
                # and returned as False rather than read as a lock-down that
                # happened.
                _perm_warn(path, (r.stderr or r.stdout or b"").decode(
                    "utf-8", "replace").strip() or f"icacls exit {r.returncode}")
                return False
        return True
    except Exception as e:
        # Best-effort (see docstring): a failure (icacls missing, a filesystem
        # without per-file perms) leaves the home-dir scoping in effect and never
        # raises. It is reported at debug level and returned as False rather than
        # swallowed, so a caller can retry.
        _perm_warn(path, repr(e))
        return False


def atomic_write_private(path: Path, text: str, *, retrying: bool = False) -> bool:
    """Write *text* to *path* atomically, owner-restricted from the moment the
    bytes first exist on disk. Returns whether the PRE-RENAME restriction
    succeeded (see below).

    *retrying* selects the final replace: False (default) is a bare
    ``os.replace``, which never sleeps; True uses ``replace_atomic``'s bounded
    backoff (up to ~1 s) over a transient Windows sharing violation, such as a
    concurrent reader's open handle. Pass True only when the caller serializes
    every writer of *path* across processes (a cross-process lock held for the
    whole write or read-modify-write). The callers that pass True are
    ``auth.set_api_key`` (auth.key), ``auth._save_keystore`` (auth.json) and
    ``model_source_credentials._write_all`` (model_source_credentials.json);
    every other caller uses the bare replace.

    TWO steps are needed and NEITHER covers both platforms alone:

    * the temp file is created via ``os.open`` with an explicit 0600, so on
      POSIX it is never briefly at the umask default. On Windows the mode
      argument writes no ACL at all.
    * :func:`restrict_file_perms` on the TEMP file before the rename covers
      Windows - ``os.replace`` carries the source's ACL (and POSIX mode) onto
      the destination, so the single call covers both names. The destination is
      retried only when the first attempt failed.

    The pre-rename restrict also covers the one case ``os.open`` cannot: a stale
    ``.tmp`` left by an earlier crash is opened, not created, so its existing
    mode survives O_CREAT.

    Call sites: ``auth.key``, ``auth.json`` and the owner-KDF file in auth.py;
    ``model_source_credentials.json``; ``sessions.json``; the instance
    registry entry from ``register_instance``, ``restore_own`` and
    ``set_mode``, and the hold marker from ``register_hold_marker``; the issue
    tracker's record; the job manager's activity store.

    The RETURN VALUE is the pre-rename ``ok``, so a caller that logs its own
    subsystem-named warning on failure keeps that signal.

    When this process holds the cross-process lock on *path* (see
    ``store_lock``), the lock file is re-checked immediately before the
    replace and ``ConfigLockLost`` is raised, with nothing written, if it no
    longer holds that acquisition's token.

    Best-effort by contract: a tightening that fails is reported by
    ``restrict_file_perms`` (which warns) and retried, never raised.

    With *retrying* False the replace is a bare ``os.replace``; some of those
    callers (``sessions._save``, reached from session_login, create_key_ep and
    _gui_index) run on the asyncio event loop.

    ``os.open`` is called WITHOUT ``os.O_BINARY``: on Windows it then opens in
    TEXT mode, so ``os.write`` expands ``\\n`` to ``\\r\\n`` and the resulting
    file matches what ``Path.write_text`` produces on each platform. Adding
    O_BINARY would switch every Windows install's credential and registry files
    to LF on the next write.

    The write LOOPS because ``os.write`` is a single syscall that may consume
    less than the whole buffer (ENOSPC after a partial write returns the short
    count rather than raising); without the loop a silently truncated keystore
    or session store could replace a good one. Raising leaves the temp file
    behind and the destination untouched."""
    assert_store_lock_held(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = text.encode("utf-8")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if not n:
                raise OSError(
                    f"short write persisting {path.name}: {written} of "
                    f"{len(data)} bytes accepted; the previous file is intact")
            written += n
    finally:
        os.close(fd)
    ok = restrict_file_perms(tmp)
    if retrying:
        replace_atomic(tmp, path)
    else:
        os.replace(tmp, path)      # atomic on Windows + POSIX (same dir)
    if not ok:
        restrict_file_perms(path)
    return ok

# A concurrent open handle makes a Windows os.replace / open raise a TRANSIENT
# PermissionError (WinError 5); a bounded retry rides it out. The lock is usually
# microseconds, but a loaded box (antivirus scanning the file, an indexer, a slow
# second process) can hold it tens of ms, so the backoff escalates and the total
# budget is ~1 s before a PERSISTENT failure is re-raised / falls back. See
# replace_atomic / read_json.
_REPLACE_RETRIES = 16
_REPLACE_BACKOFF = 0.01      # seconds; escalates up to the cap
_REPLACE_BACKOFF_CAP = 0.1   # seconds


def _transient_backoff(attempt: int) -> None:
    """Sleep before the next retry, escalating linearly to a cap so a lock that
    lingers on a busy machine is ridden out without a hot spin."""
    time.sleep(min(_REPLACE_BACKOFF * (attempt + 1), _REPLACE_BACKOFF_CAP))


def _is_transient_permission_error(e: OSError) -> bool:
    """True when a PermissionError plausibly reflects a TRANSIENT file lock worth
    riding out, rather than a stable denial no retry can change.

    Windows is the only platform with the race this retry exists for: another
    handle (a concurrent atomic replace, antivirus, the indexer, Windows Search)
    holds the file open for microseconds and open()/os.replace raises
    PermissionError - ERROR_SHARING_VIOLATION (32), or ERROR_ACCESS_DENIED (5)
    when a replace collides with an AV lock. A genuine Windows ACL denial also
    surfaces as winerror 5 and is indistinguishable from the AV case at this
    layer, so Windows keeps retrying it: ~1s to be sure is the documented,
    accepted trade there.

    POSIX has no such race. open() raises EACCES/EPERM only when the mode or ACL
    genuinely denies us (a mode-000 or root-owned file left by an earlier sudo
    run), which is a stable state - the retry can never succeed and just burns
    ~1s of time.sleep while holding _io_lock, which serializes config access
    process-wide, including the per-request auth path (auth.require_auth ->
    load_config). So on POSIX we fall back at once (the failure is still
    surfaced by the caller, never hidden - it just is not retried)."""
    if os.name != "nt":
        return False
    # winerror is absent only if this was not raised by the Windows layer at all
    # (e.g. a test double); treat that as transient to preserve the established
    # Windows behaviour rather than silently narrowing it.
    return getattr(e, "winerror", None) in (5, 32, None)


def replace_atomic(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)`` with a bounded retry on a transient Windows
    sharing violation.

    os.replace IS atomic, but on Windows it raises PermissionError (WinError 5)
    when another handle has *dst* open at that instant: a second localm process
    reading the file, an antivirus / indexer / backup scanner, Windows Search.
    That window is short, so retrying briefly rides it out instead of crashing
    the save. A genuine, persistent permission problem is re-raised after the
    last attempt. On POSIX os.replace does not hit this, so the loop succeeds
    first try; a raise there is a STABLE denial (see
    _is_transient_permission_error) and is re-raised at once rather than
    stalling under _io_lock AND the cross-process lock for a retry that cannot
    succeed."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                raise
            _transient_backoff(attempt)


class ConfigUnreadable(RuntimeError):
    """A JSON store file EXISTS but could not be read, so a read-modify-write
    must not proceed (it would persist defaults over the caller's real data).
    Raised by update_config / update_registry, and by auth.py's keystore
    (create_key/revoke_key) and model_source_credentials.set_credentials,
    which share this same refuse-on-unreadable contract."""


class ConfigLockTimeout(TimeoutError):
    """Another localm process held a store file's cross-process lock for the
    whole wait budget, so the write was not made. Nothing was written; trying
    again later is safe. Raised by ``store_lock``."""


class ConfigLockLost(ConfigLockTimeout):
    """This process's cross-process lock on a store file was taken over (or its
    lock file removed) before the write. Nothing was written; trying again
    later is safe. Raised by ``_HeldLock.assert_held``."""


def read_json(path: Path, default):
    """Read JSON from *path*, falling back to its .bak then *default* on any
    corruption - a damaged file must never take the whole app down."""
    return read_json_checked(path, default)[0]


def read_json_checked(path: Path, default):
    """``(value, read_ok)``. ``read_ok`` is True only when *path* itself parsed,
    or when no file is present at all (a genuinely absent file). It is False
    when a file was PRESENT and *path* could not be read, whether the ``.bak``
    then parsed (the value is the previous generation) or not (the value is
    *default*, indistinguishable from an absent file). Use
    :func:`read_json_state` to tell those two apart.

    Read-only consumers whose default is the safe answer take the fallback
    and keep calling read_json (the key verify path, netname and updater). A
    consumer whose default is looser than a setting the user may have stored
    (netpolicy, require_auth_enabled, audit.effective_mode) calls
    load_config_checked and fails safe on ``read_ok`` False. A
    read-modify-write must not take the fallback: writing it back replaces
    the file with an older generation or with defaults, so update_config /
    update_registry refuse on ``read_ok`` False.

    Note the asymmetry: a file that PARSES but holds the wrong shape (a JSON
    string, a list) is read_ok True and keeps its documented fall-back-to-
    defaults behaviour. Only an unreadable file can be hiding settings that
    still exist."""
    value, state = read_json_state(path, default)
    return value, state in (READ_CURRENT, READ_ABSENT)


READ_CURRENT = "current"
READ_ABSENT = "absent"
READ_FROM_BACKUP = "backup"
READ_UNREADABLE = "unreadable"

_warned_unreadable: set = set()


def _warn_unreadable_once(candidate: Path, error) -> None:
    """Report an unreadable store file once per path and error text, through the
    logger."""
    key = (str(candidate), str(error))
    if key in _warned_unreadable:
        return
    _warned_unreadable.add(key)
    logger.warning("%s is unreadable (%s)", candidate.name, error)


def read_json_state(path: Path, default):
    """``(value, state)`` with *state* one of ``READ_CURRENT`` (*path* parsed),
    ``READ_ABSENT`` (neither file exists; *value* is *default*),
    ``READ_FROM_BACKUP`` (*path* is present but unreadable and its ``.bak``
    parsed; *value* is the previous generation, not the current one) or
    ``READ_UNREADABLE`` (a file is present and nothing parsed; *value* is
    *default*).

    A file that parses but holds the wrong shape is ``READ_CURRENT``. A
    candidate that cannot be read is reported once through the logger."""
    saw_file = False
    served_backup = False
    for candidate in (path, path.with_name(path.name + ".bak")):
        if not candidate.is_file():
            continue
        saw_file = True
        for attempt in range(_REPLACE_RETRIES):
            try:
                with open(candidate, encoding="utf-8") as f:
                    value = json.load(f)
                return value, (READ_FROM_BACKUP if served_backup else READ_CURRENT)
            except PermissionError as e:
                # A transient Windows sharing violation is retried on the same
                # file; a stable denial (any POSIX EACCES) is not, because this
                # loop runs under _io_lock.
                if attempt < _REPLACE_RETRIES - 1 and _is_transient_permission_error(e):
                    _transient_backoff(attempt)
                    continue
                _warn_unreadable_once(candidate, e)
            except (ValueError, OSError, RecursionError) as e:
                # Corrupt or non-UTF-8 JSON, a huge integer, deep nesting or a
                # hard OS error: fall back at once.
                _warn_unreadable_once(candidate, e)
            break
        served_backup = True
    fallback = default() if callable(default) else default
    return fallback, (READ_UNREADABLE if saw_file else READ_ABSENT)


_CROSS_LOCK_TIMEOUT = 10.0      # seconds to wait for a lock held by another process
_CROSS_LOCK_STALE_AGE = 30.0    # a lock file whose holder cannot be identified (no start
                                 # identity, another pid space, unreadable) and that is
                                 # older than this is presumed abandoned and is reclaimed
_CROSS_LOCK_POLL = 0.02         # seconds between acquire attempts; escalates up to
_CROSS_LOCK_POLL_CAP = 0.25     # this cap under sustained contention


def _cross_lock_backoff(attempt: int) -> None:
    time.sleep(min(_CROSS_LOCK_POLL * (attempt + 1), _CROSS_LOCK_POLL_CAP))


def _lock_owner_pid(raw: bytes):
    """Extract the PID from a lock file's ``<pid>:<nonce>`` token, or from a
    bare ``<pid>``. None if unparseable - including an orphaned/empty file left
    by a write that failed partway through (see store_lock's
    acquire-failure cleanup)."""
    try:
        return int(raw.split(b":", 1)[0])
    except (ValueError, IndexError):
        return None


# The fencing tokens of locks THIS process currently holds, keyed by lock path.
# This - not the pid recorded in the file - is what identifies a lock as ours.
# A pid NUMBER is not an identity: the OS reuses pids freely across process
# lifetimes, so a lock LEAKED by a crashed holder can carry the very pid the OS
# later hands to a new localm process. The uuid4 nonce in each token makes the
# match exact - a leaked file can never equal a token we are holding right now,
# whatever pid it records.
_held_lock_tokens: dict = {}
_held_lock_tokens_guard = threading.Lock()


def _lock_is_held_by_us(lockpath: Path, held: bytes) -> bool:
    """True only when *held* is a fencing token THIS process wrote and still
    holds - i.e. a genuine nested acquisition, not a pid collision."""
    if not held:
        return False
    with _held_lock_tokens_guard:
        return _held_lock_tokens.get(str(lockpath)) == held


def _unlink_lock_file(lockpath: Path) -> None:
    """Remove our own released lock file, retrying a transient sharing violation
    (another process reading the file) with replace_atomic's bounded backoff.
    A file that still cannot be removed is reported at WARNING, naming the file.
    See test_release_retries_unlink_while_a_reader_pins_the_lock_file."""
    last: Optional[OSError] = None
    for attempt in range(_REPLACE_RETRIES):
        try:
            lockpath.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last = e
            if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                break
            _transient_backoff(attempt)
    logger.warning("could not remove released lock file %s (%s); config/registry "
                   "writes in every localm process fail on it until it is %.0fs "
                   "old and reclaimed", lockpath.name, last, _CROSS_LOCK_STALE_AGE)


def _read_lock_for_release(lockpath: Path) -> Optional[bytes]:
    """The lock file's bytes, or None when the file is gone or could not be read.
    A transient sharing violation is retried with replace_atomic's bounded
    backoff; a file that still cannot be read is reported at WARNING, naming the
    file, because the release then leaves it in place for stale-age reclaim."""
    last: Optional[OSError] = None
    for attempt in range(_REPLACE_RETRIES):
        try:
            return lockpath.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            last = e
            if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                break
            _transient_backoff(attempt)
    logger.warning("could not read lock file %s on release (%s); leaving it in "
                   "place, so config/registry writes in every localm process "
                   "fail on it until it is %.0fs old and reclaimed",
                   lockpath.name, last, _CROSS_LOCK_STALE_AGE)
    return None







def _lock_token(pid: int, nonce: str) -> bytes:
    """The bytes a holder writes into its lock file: ``<pid>:<nonce>:<pid
    space id>:<start identity as JSON>``. The last two fields are empty when the
    process start identity cannot be read."""
    try:
        start = procid.process_start_identity(pid)
        host = procid.pid_space_id() if start else ""
    except Exception as e:
        logger.debug("lock token: no start identity for pid %s (%s)", pid, e)
        start = host = None
    tail = json.dumps(start, separators=(",", ":"), sort_keys=True) if start else ""
    return f"{pid}:{nonce}:{host or ''}:{tail}".encode("ascii")


def _lock_holder_state(raw: bytes) -> str:
    """``"alive"``, ``"dead"`` or ``"unknown"`` for the holder a lock file's
    bytes *raw* name.

    ``"alive"``: the token names this pid space and a start identity, and the
    process now running under that pid has that start identity; a token naming
    this process counts only while one of its acquisitions holds that exact
    token. ``"dead"``: the pid has exited, now names a process with another
    start identity, or is this process's under a token no acquisition holds.
    ``"unknown"``: anything else (a token without a start identity, another
    pid space, unparseable bytes, a probe that cannot answer)."""
    parts = raw.split(b":", 3)
    if len(parts) != 4 or not parts[2] or not parts[3]:
        return "unknown"
    try:
        pid = int(parts[0])
        host = parts[2].decode("ascii")
        recorded = json.loads(parts[3].decode("ascii"))
    except (ValueError, RecursionError):
        return "unknown"
    if pid <= 0 or not isinstance(recorded, dict):
        return "unknown"
    try:
        if host != procid.pid_space_id():
            return "unknown"
        alive = procid.pid_liveness(pid)
        if alive is None:
            return "unknown"
        if not alive:
            return "dead"
        current = procid.process_start_identity(pid)
    except Exception as e:
        logger.debug("lock liveness probe for pid %s failed (%s)", pid, e)
        return "unknown"
    if procid.start_identity_differs(recorded, current):
        return "dead"
    if not procid.start_identity_matches(recorded, current):
        return "unknown"
    if pid == os.getpid():
        with _held_lock_tokens_guard:
            held = raw in _held_lock_tokens.values()
        return "alive" if held else "dead"
    return "alive"


def _lock_is_reclaimable(raw: bytes, mtime: float) -> bool:
    """Whether a lock file with bytes *raw* and modification time *mtime* may be
    removed: its holder is gone, or it cannot be identified and the file is
    older than _CROSS_LOCK_STALE_AGE. A holder proven running keeps the lock
    however old the file is."""
    state = _lock_holder_state(raw)
    if state == "alive":
        return False
    if state == "dead":
        return True
    return time.time() - mtime > _CROSS_LOCK_STALE_AGE


class _HeldLock:
    """The handle ``store_lock`` yields."""

    def __init__(self, lockpath: Path, token: bytes):
        self._lockpath = lockpath
        self._token = token

    def assert_held(self) -> None:
        """Raise ConfigLockLost unless the lock file still holds this
        acquisition's token. Call immediately before the write the lock
        protects."""
        current = None
        for attempt in range(_REPLACE_RETRIES):
            try:
                current = self._lockpath.read_bytes()
                break
            except FileNotFoundError:
                break
            except OSError as e:
                if attempt == _REPLACE_RETRIES - 1 or not _is_transient_permission_error(e):
                    break
                _transient_backoff(attempt)
        if current != self._token:
            raise ConfigLockLost(
                f"{self._lockpath.name} was taken over by another localm process "
                "before this write; nothing was written")


def assert_store_lock_held(target: Path) -> None:
    """Raise ConfigLockLost when this process holds the cross-process lock on
    *target* and the lock file no longer carries that acquisition's token.
    Does nothing when this process holds no lock on *target*."""
    lockpath = target.with_name(target.name + ".lock")
    with _held_lock_tokens_guard:
        token = _held_lock_tokens.get(str(lockpath))
    if token is not None:
        _HeldLock(lockpath, token).assert_held()


def _reclaim_stale_lock(lockpath: Path, judged: bytes, judged_mtime: float) -> bool:
    """Remove a lock file judged stale, and report whether it was removed.

    Re-stats and re-reads the file immediately before the unlink and leaves it
    alone when its mtime or its bytes are no longer the ones it was judged by, or
    when the staleness test no longer holds: between the judgement and here, the
    holder may have released it and a new holder may have created its own. Two
    waiters that both reach the stale branch therefore cannot both unlink and both
    create. True when the file is already gone, so the caller retries its create.
    A file that cannot be read is judged by the bytes it was first read as, so an
    unreadable stale lock is still reclaimed rather than wedging every writer.
    The residual window (a lock created between this re-read and the unlink)
    cannot be closed with plain files; the fencing token stops it from cascading,
    since the wrongly-removed holder's release will not then delete a third
    party's lock. Mirrors ``rag.collection_lock._reclaim``.
    See test_a_stale_lock_is_not_reclaimed_after_a_new_holder_took_it."""
    try:
        current_mtime = lockpath.stat().st_mtime
    except OSError:
        return True
    if current_mtime != judged_mtime:
        return False
    try:
        current = lockpath.read_bytes()
    except OSError:
        current = judged
    if current != judged:
        return False
    if not _lock_is_reclaimable(current, current_mtime):
        return False
    try:
        lockpath.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


@contextlib.contextmanager
def store_lock(target: Path, *, timeout: Optional[float] = None):
    """Hold an exclusive, cross-process lock on *target* (a sibling ``<name>.lock``
    marker file) for the duration of the ``with`` block.

    *timeout* is the wait budget in seconds; None means _CROSS_LOCK_TIMEOUT.

    Bounded retry with an escalating backoff, matching replace_atomic's
    established pattern, but on a wall-clock budget (_CROSS_LOCK_TIMEOUT) rather
    than a fixed attempt count, since a full read-modify-write held by another
    process can legitimately take longer than a single os.replace. Timing out
    raises TimeoutError rather than proceeding unprotected.

    FENCING TOKEN: each acquisition writes a unique ``<pid>:<nonce>`` token into
    the lock file, not just a bare marker. A holder whose critical section
    legitimately outlasts _CROSS_LOCK_STALE_AGE (not crashed, just slow - heavy
    antivirus scanning, a paused debugger) has its lock reclaimed by a waiter
    that reads staleness as "crashed". With the token, release only removes the
    file if it still holds the token THIS call wrote; if it doesn't (this call's
    lock was reclaimed as stale while still legitimately held), the file is left
    alone - whoever's token is on disk owns cleanup, so a stale-reclaim can never
    cascade into deleting a live holder's lock.

    The same token also turns a same-thread NESTED call (a mutator passed to
    update_config()/update_registry() that calls back into either for the same
    file - not supported, this lock is not reentrant like _io_lock) from a
    confusing _CROSS_LOCK_TIMEOUT-long stall into an immediate, clear error: the
    token on disk matches one in _held_lock_tokens, which is only possible via
    the calling process's own nested acquisition (a sibling thread in this
    process would already be blocked on the outer _rmw_lock before ever reaching
    here). Ownership is decided by that exact token, NOT by the pid recorded in
    the file: pids are reused across process lifetimes, so a leaked lock carrying
    our own pid must still be treated as foreign, and stays eligible for the
    staleness reclaim below.

    STALENESS is decided by whether the holder is still running, not by the
    age of the file: the token names the holder's pid, pid space and process
    start identity (the rule ``rag.collection_lock`` uses). A holder proven
    running keeps the lock however long it holds it (a waiter times out with
    ConfigLockTimeout instead); a holder proven gone (pid exited, or reused by
    another process) is reclaimed at once; a holder that cannot be identified (a
    token without a start identity, another pid space, unreadable bytes) is
    reclaimed once the file is older than _CROSS_LOCK_STALE_AGE. Every reclaim is
    logged.

    Yields a handle whose ``assert_held()`` re-reads the lock file and raises
    ConfigLockLost unless it still holds this acquisition's token; call it
    immediately before the write the lock protects."""
    lockpath = target.with_name(target.name + ".lock")
    token = _lock_token(os.getpid(), uuid.uuid4().hex)
    budget = _CROSS_LOCK_TIMEOUT if timeout is None else timeout
    deadline = time.time() + budget
    attempt = 0
    while True:
        try:
            fd = os.open(str(lockpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = lockpath.read_bytes()
            except OSError:
                held = b""
            if _lock_is_held_by_us(lockpath, held):
                raise RuntimeError(
                    f"{lockpath.name} is already held by this same process "
                    f"(pid {os.getpid()}) - a mutator passed to update_config()/"
                    "update_registry() called back into update_config()/"
                    "update_registry() on the same file. That is not supported "
                    "(the cross-process lock is not reentrant); make both changes "
                    "in one mutator instead of nesting the calls.") from None
            try:
                mtime = lockpath.stat().st_mtime
            except OSError:
                # The lock vanished between our failed create and this stat (the
                # holder just released it) - retry the create immediately rather
                # than waiting out a backoff for a lock that is already gone.
                continue
            age = time.time() - mtime
            if _lock_is_reclaimable(held, mtime) and _reclaim_stale_lock(lockpath, held, mtime):
                print(f"[localm] note: reclaiming stale lock {lockpath.name} "
                      f"({age:.0f}s old) - the process that held it appears to "
                      "have crashed without releasing it.", file=sys.stderr)
                continue
            if time.time() >= deadline:
                raise ConfigLockTimeout(
                    f"timed out after {budget:g}s waiting for "
                    f"{lockpath.name}, held by another localm process") from None
            _cross_lock_backoff(attempt)
            attempt += 1
            continue
        # We created the lock file (os.open succeeded). Everything from here
        # must clean up OUR OWN just-created file on failure, or a transient
        # write error (ENOSPC, a momentary AV lock on the file we just made)
        # would leak an orphaned, unowned lock file that blocks every other
        # config/registry writer install-wide until the next staleness reclaim.
        try:
            try:
                os.write(fd, token)
            finally:
                os.close(fd)
        except BaseException:
            try:
                lockpath.unlink()
            except OSError:
                pass
            raise
        # Record the token only once it is actually ON DISK: a failed write above
        # unlinks the file, so registering earlier would leave us believing we
        # hold a lock that does not exist.
        with _held_lock_tokens_guard:
            _held_lock_tokens[str(lockpath)] = token
        break
    try:
        yield _HeldLock(lockpath, token)
    finally:
        # Forget our claim FIRST, and unconditionally: once we leave this block we
        # no longer hold the lock, whatever happens to the file below. A stale
        # entry here would make a LATER acquisition of the same path read a
        # foreign lock as our own nested call.
        with _held_lock_tokens_guard:
            if _held_lock_tokens.get(str(lockpath)) == token:
                del _held_lock_tokens[str(lockpath)]
        # Fencing-token release (see docstring): only remove the lock file if it
        # still holds the token WE wrote. If it doesn't, another process reclaimed
        # it as stale while we were still legitimately inside our critical section
        # (a write that outlasted _CROSS_LOCK_STALE_AGE) - deleting THEIR live
        # lock here would let a third writer in while they still believe they
        # hold exclusive access, so we leave it alone instead.
        current = _read_lock_for_release(lockpath)
        if current == token:
            _unlink_lock_file(lockpath)
        elif current is not None:
            print(f"[localm] note: {lockpath.name} was reclaimed by another "
                  "localm process while this process still held it (this "
                  f"write took longer than _CROSS_LOCK_STALE_AGE={_CROSS_LOCK_STALE_AGE:.0f}s) "
                  "- not deleting the new holder's lock.", file=sys.stderr)
