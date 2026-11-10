# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-GPU load lease shared by every localm process that uses one data dir.

A process that is about to decide whether a model fits on a GPU, and then load
it, holds the lease of every device the load can touch from that decision until
the load has allocated its memory (or failed). A second process that wants one of
those devices waits, so it reads free VRAM only after the first load's memory is
on the card. The lease covers every kind of localm process (the HTTP server,
``localm run``, ``localm job run``, the MCP server), listening or not.

Keys: a load confined by ``gpu_split_indices`` takes ``gpu<index>`` for each
index; any other GPU load takes :data:`ALL_DEVICES`, which conflicts with every
key. A CPU-only load takes none (:func:`lease_keys`).

On disk (``<data dir>/run/gpu-lease/``): one directory ``<key>.lease`` per held
key, holding ``owner.json`` (pid, pid space, process start identity, a token
unique to the holding process, the key, the purpose and the time). A lease
directory is created whole by renaming a finished temporary directory, so no
reader ever sees a lease without its owner record. Every look at, creation of
and removal of a lease happens under ``gate``, an OS advisory lock on one file
held for a few file operations, which the operating system drops when its
holder exits.

A lease whose holder is gone is reclaimed: the recorded pid is not alive, or a
live pid has a different start identity (the pid was reused). Anything that
cannot be told (another pid space, an unreadable record, a failed probe)
counts as alive. Elapsed time never makes a lease stale.

Within one process the lease is shared by every thread: a thread asking for a
key this process already holds gets it at once (a count), and the directory is
removed when the count of the last holder drops to zero. Loads within one
process are serialised by ``engine._LOAD_LOCK``.

LOCK ORDER: the lease is OUTER to ``engine._LOAD_LOCK`` (and so to every lock
inner to it). A thread holding ``_LOAD_LOCK`` may still ask for a key: the
request waits only on other processes, and no other process ever waits on this
process's ``_LOAD_LOCK``. The module's own locks (``_ACQ_LOCK`` outer,
``_STATE`` inner, then the gate file lock) are held for file operations only and
never while waiting for a lease. See test_lock_order_lease_outer_load_lock_inner.

Every wait is bounded: a caller that cannot get the lease in *timeout* seconds
gets :class:`LoadBusyError`.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

from localm.debuglog import logger

ALL_DEVICES = "all"
LEASE_SUFFIX = ".lease"
OWNER_FILE = "owner.json"
GATE_FILE = "gate"
#: Longest a caller waits for a device lease, or for ``engine._LOAD_LOCK``,
#: before it is answered "busy". Stays below the native load ceiling (900 s),
#: so a caller behind a load that hangs is answered before that load gives up.
#: See test_wait_bound_sits_below_the_native_load_ceiling.
LOAD_WAIT_TIMEOUT_S = 300.0
#: The retry hint a "busy" answer carries, in seconds.
BUSY_RETRY_AFTER_S = 10
_POLL_S = 0.1
_GATE_WAIT_S = 2.0
_TMP_PREFIX = ".tmp-"

#: Unique to this process; marks the leases this process holds.
_TOKEN = secrets.token_hex(8)

_ACQ_LOCK = threading.Lock()
_STATE = threading.Lock()
_counts: dict[str, int] = {}


class LoadBusyError(RuntimeError):
    """A model load could not start within the wait bound because another load
    holds the device (another process's lease, or this process's load lock).
    ``retry_after_s`` is the suggested wait before trying again, ``holder`` a
    short description of who holds it (may be empty)."""

    def __init__(self, message: str, *, retry_after_s: int = BUSY_RETRY_AFTER_S,
                 holder: str = "") -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.holder = holder


def busy_message(waited_s: float, holder: str = "") -> str:
    """The text of a "busy" answer after waiting *waited_s* seconds."""
    who = f" ({holder})" if holder else ""
    return (f"Busy: a model is loading{who}. Waited {waited_s:.0f}s; "
            f"retry in {BUSY_RETRY_AFTER_S}s.")


def lease_root(home: Optional[Path] = None) -> Path:
    """The lease directory of data dir *home* (default: the current data dir)."""
    if home is None:
        from localm import config
        home = config.HOME_DIR
    return Path(home) / "run" / "gpu-lease"


def lease_keys(cfg: Optional[dict] = None, *, device: Optional[str] = None,
               n_gpu_layers: Optional[int] = None) -> tuple[str, ...]:
    """The lease keys of a load under config *cfg* (default: the current
    config). Empty for a CPU-only load (*device* ``"cpu"`` or *n_gpu_layers*
    0); ``gpu<i>`` for each non-negative integer in ``gpu_split_indices``,
    sorted; otherwise ``(ALL_DEVICES,)``."""
    if isinstance(device, str) and device.strip().lower() == "cpu":
        return ()
    if n_gpu_layers == 0:
        return ()
    if cfg is None:
        from localm.config import load_config
        cfg = load_config()
    raw = cfg.get("gpu_split_indices") if isinstance(cfg, dict) else None
    if isinstance(raw, (list, tuple)):
        idx = sorted({i for i in raw if type(i) is int and i >= 0})
        if idx:
            return tuple(f"gpu{i}" for i in idx)
    return (ALL_DEVICES,)


def _conflicts(a: str, b: str) -> bool:
    return a in (b, ALL_DEVICES) or b == ALL_DEVICES


def _win_process(pid: int) -> tuple[Optional[bool], Optional[int]]:
    """``(alive, creation FILETIME)`` of Windows process *pid* from kernel32:
    alive None when it cannot be told; the time None when it cannot be read.
    ``(None, None)`` on any other platform."""
    if sys.platform != "win32":
        return None, None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.OpenProcess(0x1000, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        return (False if err == 87 else None), None
    try:
        code = wintypes.DWORD()
        alive: Optional[bool] = None
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            alive = code.value == 259
        times = [wintypes.FILETIME() for _ in range(4)]
        created = None
        if k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        return alive, created
    finally:
        k32.CloseHandle(handle)


def _start_identity(pid: int):
    """A start identity of *pid* for an owner record: ``{"filetime": int}`` on
    Windows (kernel32), else ``instances.process_start_identity``; None when
    it cannot be read."""
    if sys.platform == "win32":
        try:
            created = _win_process(pid)[1]
        except (OSError, AttributeError, ValueError):
            return None
        return {"filetime": created} if created is not None else None
    from localm import instances
    return instances.process_start_identity(pid)


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is alive; True when it cannot be told."""
    if sys.platform == "win32":
        try:
            alive = _win_process(pid)[0]
        except (OSError, AttributeError, ValueError):
            alive = None
        return True if alive is None else alive
    from localm import instances
    return instances.pid_alive(pid)


def _identity_differs(recorded, current) -> bool:
    """True only when two start identities of the same shape name two
    different processes."""
    if isinstance(recorded, dict) and isinstance(current, dict) \
            and "filetime" in recorded and "filetime" in current:
        return (type(recorded["filetime"]) is int and type(current["filetime"]) is int
                and recorded["filetime"] != current["filetime"])
    from localm import instances
    return instances.start_identity_differs(recorded, current)


def _owner_record(key: str, purpose: str) -> dict:
    from localm import instances
    return {"pid": os.getpid(), "space": instances.pid_space_id(),
            "start": _start_identity(os.getpid()),
            "token": _TOKEN, "key": key, "purpose": purpose,
            "at": time.time()}


def _read_owner(lease_dir: Path) -> Optional[dict]:
    try:
        rec = json.loads((lease_dir / OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return rec if isinstance(rec, dict) else None


def holder_is_gone(rec: Optional[dict]) -> bool:
    """True only when owner record *rec* names a process PROVEN gone: a pid in
    this pid space that is not alive, or alive with a different start
    identity. A record of this process whose token is not this process's is
    gone too (a previous process with the same pid). False for no record, an
    unparsable pid, another pid space and any probe failure."""
    if not isinstance(rec, dict):
        return False
    from localm import instances
    try:
        pid = int(rec.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0 or rec.get("space") != instances.pid_space_id():
        return False
    if pid == os.getpid():
        return rec.get("token") != _TOKEN
    try:
        if not _pid_alive(pid):
            return True
        return _identity_differs(rec.get("start"), _start_identity(pid))
    except Exception as e:
        logger.debug("load lease: cannot tell whether pid %s is alive (%s); "
                     "keeping its lease", pid, e)
        return False


def _describe(rec: Optional[dict]) -> str:
    if not isinstance(rec, dict):
        return "an unreadable lease"
    return f"process {rec.get('pid')}: {rec.get('purpose') or 'a model load'}"


@contextlib.contextmanager
def _gate(root: Path) -> Iterator[bool]:
    """Hold the gate file lock of *root* for the block; yields False when it
    could not be taken within ``_GATE_WAIT_S``."""
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / GATE_FILE, "a+b")
    got = False
    try:
        deadline = time.monotonic() + _GATE_WAIT_S
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK,
                                   errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
        yield got
    finally:
        if got:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                logger.debug("load lease: unlocking the gate failed: %s", e)
        f.close()


def _remove_dir(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)
    if d.exists():
        logger.warning("load lease: could not remove %s", d)


def _sweep_temps(root: Path) -> None:
    """Remove temporary lease directories whose creating process is gone.
    Caller holds the gate."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for d in entries:
        if d.name.startswith(_TMP_PREFIX) and d.is_dir() and holder_is_gone(_read_owner(d)):
            _remove_dir(d)


def _try_take(root: Path, keys: list[str], purpose: str) -> tuple[bool, str]:
    """Create the lease directories of *keys* when no other live process holds
    a conflicting key. Reclaims the leases of gone holders. Caller holds
    ``_ACQ_LOCK``. Returns ``(taken, holder)``; *holder* describes the first
    conflicting live lease, or the gate, when not taken."""
    with _gate(root) as got:
        if not got:
            return False, "another process is checking the GPU lease"
        _sweep_temps(root)
        try:
            held = [d for d in root.iterdir()
                    if d.name.endswith(LEASE_SUFFIX) and d.is_dir()]
        except OSError as e:
            return False, f"cannot read {root}: {e}"
        for d in held:
            other = d.name[:-len(LEASE_SUFFIX)]
            if not any(_conflicts(other, k) for k in keys):
                continue
            rec = _read_owner(d)
            if isinstance(rec, dict) and rec.get("token") == _TOKEN:
                continue
            if holder_is_gone(rec):
                logger.info("load lease: reclaiming the %s lease of %s, which is gone",
                            other, _describe(rec))
                _remove_dir(d)
                continue
            return False, _describe(rec)
        made: list[Path] = []
        try:
            for k in keys:
                final = root / (k + LEASE_SUFFIX)
                if final.is_dir():
                    rec = _read_owner(final)
                    if isinstance(rec, dict) and rec.get("token") == _TOKEN:
                        continue
                    return False, f"the {k} lease of {_describe(rec)} could not be removed"
                tmp = root / f"{_TMP_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
                tmp.mkdir()
                try:
                    (tmp / OWNER_FILE).write_text(
                        json.dumps(_owner_record(k, purpose)), encoding="utf-8")
                    os.rename(tmp, final)
                except OSError:
                    _remove_dir(tmp)
                    raise
                made.append(final)
        except OSError:
            for d in made:
                _remove_dir(d)
            raise
        return True, ""


def _drop(root: Path, keys: Iterable[str]) -> None:
    """Remove this process's lease directories of *keys*. Caller holds
    ``_ACQ_LOCK``."""
    with _gate(root) as got:
        if not got:
            logger.debug("load lease: removing %s without the gate", list(keys))
        for k in keys:
            d = root / (k + LEASE_SUFFIX)
            rec = _read_owner(d)
            if isinstance(rec, dict) and rec.get("token") == _TOKEN:
                _remove_dir(d)


def acquire(keys: Iterable[str], *, timeout: Optional[float] = None,
            purpose: str = "model load", root: Optional[Path] = None) -> tuple[str, ...]:
    """Take the lease of every key in *keys* for this process, waiting up to
    *timeout* seconds (default ``LOAD_WAIT_TIMEOUT_S``). Returns the keys
    taken (pass them to :func:`release`).
    Raises :class:`LoadBusyError` when another process still holds a
    conflicting key after *timeout*. Never raises for a filesystem failure:
    a lease that cannot be written is logged at WARNING and the load goes on
    without it (the keys returned are then empty)."""
    wanted = tuple(sorted(set(keys)))
    if not wanted:
        return ()
    if timeout is None:
        timeout = LOAD_WAIT_TIMEOUT_S
    base = lease_root() if root is None else root
    started = time.monotonic()
    deadline = started + max(0.0, timeout)
    holder = ""
    while True:
        with _ACQ_LOCK:
            with _STATE:
                missing = [k for k in wanted if _counts.get(k, 0) == 0]
            if missing:
                try:
                    taken, holder = _try_take(base, missing, purpose)
                except OSError as e:
                    logger.warning("load lease: cannot write the GPU lease under %s "
                                   "(%s); loading without it", base, e)
                    return ()
            else:
                taken = True
            if taken:
                with _STATE:
                    for k in wanted:
                        _counts[k] = _counts.get(k, 0) + 1
                return wanted
        now = time.monotonic()
        if now >= deadline:
            raise LoadBusyError(busy_message(now - started, holder), holder=holder)
        time.sleep(min(_POLL_S, max(0.0, deadline - now)))


def release(keys: Iterable[str], *, root: Optional[Path] = None) -> None:
    """Give back one hold of each key :func:`acquire` returned. The lease
    directory goes when this process's last hold of the key is released."""
    keys = tuple(keys)
    if not keys:
        return
    base = lease_root() if root is None else root
    with _ACQ_LOCK:
        gone = []
        with _STATE:
            for k in keys:
                n = _counts.get(k, 0)
                if n <= 1:
                    _counts.pop(k, None)
                    if n == 1:
                        gone.append(k)
                else:
                    _counts[k] = n - 1
        if gone:
            try:
                _drop(base, gone)
            except OSError as e:
                logger.warning("load lease: cannot remove the GPU lease %s under "
                               "%s: %s", gone, base, e)


def held_keys() -> dict[str, int]:
    """This process's holds per key (a copy)."""
    with _STATE:
        return dict(_counts)


@contextlib.contextmanager
def held(keys: Optional[Iterable[str]] = None, *, timeout: Optional[float] = None,
         purpose: str = "model load", root: Optional[Path] = None) -> Iterator[tuple]:
    """Hold the lease of *keys* (default: :func:`lease_keys` of the current
    config) for the block. Raises :class:`LoadBusyError` as :func:`acquire`."""
    taken = acquire(lease_keys() if keys is None else keys, timeout=timeout,
                    purpose=purpose, root=root)
    try:
        yield taken
    finally:
        release(taken, root=root)


def acquire_load_lock(lock, *, timeout: Optional[float] = None, what: str = "") -> None:
    """Acquire *lock* (``engine._LOAD_LOCK``) waiting at most *timeout*
    seconds. Raises :class:`LoadBusyError` when it is still held then."""
    started = time.monotonic()
    if timeout is None:
        timeout = LOAD_WAIT_TIMEOUT_S
    if not lock.acquire(True, max(0.0, timeout)):
        raise LoadBusyError(busy_message(time.monotonic() - started,
                                         what or "this process is loading a model"),
                            holder=what)


@contextlib.contextmanager
def bounded(lock, *, timeout: Optional[float] = None, what: str = "") -> Iterator[None]:
    """Hold *lock* (``engine._LOAD_LOCK``) for the block, waiting for it at
    most *timeout* seconds; raises :class:`LoadBusyError` past that."""
    acquire_load_lock(lock, timeout=timeout, what=what)
    try:
        yield
    finally:
        lock.release()


@contextlib.contextmanager
def load_guard(lock, keys: Optional[Iterable[str]] = None, *,
               timeout: Optional[float] = None,
               purpose: str = "model load") -> Iterator[None]:
    """The lease of *keys* (default: the current config's), then *lock*
    (``engine._LOAD_LOCK``), each waited for at most *timeout* seconds in all,
    held for the block and released in reverse order. Raises
    :class:`LoadBusyError` when either cannot be had in time."""
    if timeout is None:
        timeout = LOAD_WAIT_TIMEOUT_S
    started = time.monotonic()
    taken = acquire(lease_keys() if keys is None else keys, timeout=timeout,
                    purpose=purpose)
    try:
        acquire_load_lock(lock, timeout=max(0.0, timeout - (time.monotonic() - started)))
        try:
            yield
        finally:
            lock.release()
    finally:
        release(taken)
