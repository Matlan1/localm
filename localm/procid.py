# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process and host identity: whether a pid is running, when a process or
thread started, and which pid table a pid belongs to.

Stdlib only at module level (``psutil`` is imported inside the functions that
use it). Store locks, instance discovery and every other liveness-based lock
read these answers from here.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("localm")

_LINUX_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def _is_zombie(pid: int) -> bool:
    """True if *pid* has exited but has not been reaped by its parent.

    ``os.kill(pid, 0)`` SUCCEEDS for such a process - a zombie keeps its
    process-table entry until its parent waits on it - so a signal-0 probe
    alone calls a dead process alive. That is not the conservative
    "cannot tell" case pid_alive documents below: the OS can tell, and the
    answer is dead. Every staleness rule in this project is keyed on
    liveness rather than elapsed time (the pull .part lock, the updater
    lock, setup_llama's provision lock, the managed-comfy update lock), so
    a zombie holder read as alive wedges that lock for good. Linux answers
    this in /proc/<pid>/stat; elsewhere (macOS, BSD) psutil's process status
    answers it. Where neither can be read this returns False and the
    signal-0 answer stands unchanged.

    The state is the first field after ``comm``, which is parenthesised and
    may itself contain ``)`` - so split on the LAST ``)``, never the first.
    See test_pid_alive_is_false_for_an_unreaped_child.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read().decode("utf-8", "replace").rpartition(")")[2].split()[0] == "Z"
    except FileNotFoundError:
        if sys.platform.startswith("linux"):
            return False
    except (OSError, IndexError):
        return False
    try:
        import psutil
    except ImportError:
        return False
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


def pid_liveness(pid: int) -> Optional[bool]:
    """Whether *pid* is a live process: True, False, or None when this
    platform cannot tell (Windows without a working psutil, an unexpected
    OSError from the signal-0 probe). A zombie is reported dead, see
    _is_zombie. A pid that is not a positive int is False."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception as e:
            logger.debug("psutil check failed (%s); liveness of pid %s unknown", e, pid)
            return None
    try:
        os.kill(pid, 0)
        return not _is_zombie(pid)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True       # exists, owned by someone else
    except OSError:
        return None


def pid_alive(pid: int) -> bool:
    """Best-effort: is *pid* a live process? Conservative - when we genuinely
    cannot tell (pid_liveness returns None, e.g. Windows without psutil),
    return True so reaping never deletes a live instance's entry. A zombie is
    not that case and is reported dead - see _is_zombie. PID reuse is handled
    by the /whoami handshake, not here."""
    alive = pid_liveness(pid)
    return True if alive is None else alive


def process_start_identity(pid: int) -> dict | None:
    """When process *pid* started, in a form no later change of the system
    clock alters, or None when it cannot be read.

    Linux: ``{"boot": <boot id or None>, "ticks": <start time in clock ticks
    since boot>}``, read from /proc. Windows and macOS: ``{"created": <psutil
    create_time>}``, the creation timestamp the kernel stores when the process
    starts; None there when psutil is not installed. Every other platform:
    None.
    """
    if sys.platform.startswith("linux"):
        # Start ticks come from /proc, not psutil's create_time. See
        # test_a_live_holder_keeps_its_lock_across_a_clock_step.
        try:
            stat = Path(f"/proc/{pid}/stat").read_bytes()
            ticks = int(stat[stat.rindex(b")") + 2:].split()[19])
        except (OSError, ValueError, IndexError):
            return None
        try:
            boot = _LINUX_BOOT_ID.read_text(encoding="ascii").strip() or None
        except (OSError, ValueError):
            boot = None
        return {"boot": boot, "ticks": ticks}
    if sys.platform in ("win32", "darwin"):
        try:
            import psutil
            return {"created": float(psutil.Process(pid).create_time())}
        except Exception:
            return None
    return None


def start_identity_differs(recorded, current) -> bool:
    """True only when *recorded* and *current* are start identities of the
    same shape that name two different processes: ticks under two different
    boot ids, a different start tick under the same boot id, or creation
    times more than a second apart.

    Two boot ids name two boots of one machine only when the caller has
    established that both identities were read on one machine (one pid
    space, see :func:`pid_space_id`); this function does not check that.

    Anything missing, malformed, of different shapes, or ticks under an
    unknown boot id returns False.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    rt, ct = recorded.get("ticks"), current.get("ticks")
    if type(rt) is int and type(ct) is int:
        rb, cb = recorded.get("boot"), current.get("boot")
        if not (isinstance(rb, str) and isinstance(cb, str)):
            return False
        return rb != cb or rt != ct
    rc, cc = recorded.get("created"), current.get("created")
    if type(rc) in (int, float) and type(cc) in (int, float):
        if math.isfinite(rc) and math.isfinite(cc):
            return abs(rc - cc) > 1.0
    return False


_WIN_THREAD_API = None


def _win_thread_api():
    """ctypes bindings for the kernel32 thread calls thread_start_identity
    makes on Windows, created once."""
    global _WIN_THREAD_API
    if _WIN_THREAD_API is None:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenThread.restype = wintypes.HANDLE
        k.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
        k.GetProcessIdOfThread.restype = wintypes.DWORD
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.GetThreadTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME)] * 4
        k.GetThreadTimes.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        _WIN_THREAD_API = (ctypes, wintypes, k)
    return _WIN_THREAD_API


def _win_thread_created(pid: int, tid: int) -> int | None:
    """The creation time of running thread *tid* of process *pid*, in 100 ns
    units since 1601, or None."""
    ctypes, wintypes, k = _win_thread_api()
    h = k.OpenThread(0x0800 | 0x00100000, False, tid)
    if not h:
        return None
    try:
        if k.GetProcessIdOfThread(h) != pid:
            return None
        if k.WaitForSingleObject(h, 0) != 0x00000102:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetThreadTimes(h, *[ctypes.byref(t) for t in times]):
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        k.CloseHandle(h)


def thread_start_identity(pid: int, tid: int) -> dict | None:
    """When thread *tid* of process *pid* started, or None when that is not a
    running thread of that process or cannot be read.

    Linux: ``{"ticks": <start time in clock ticks since boot>}``, read from
    /proc/<pid>/task/<tid>/stat. Windows: ``{"created": <creation time in 100 ns
    units since 1601>}``, for a thread that has not exited. Every other
    platform: None.
    """
    if type(pid) is not int or type(tid) is not int or pid <= 0 or tid <= 0:
        return None
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/task/{tid}/stat").read_bytes()
            return {"ticks": int(stat[stat.rindex(b")") + 2:].split()[19])}
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "win32":
        if tid > 0xFFFFFFFF:
            return None
        try:
            created = _win_thread_created(pid, tid)
        except (OSError, AttributeError) as e:
            logger.debug("thread start identity of %s/%s unreadable (%s)", pid, tid, e)
            return None
        return None if created is None else {"created": created}
    return None


def start_identity_matches(recorded, current) -> bool:
    """True only when *recorded* and *current* are start identities of the
    same shape that name one process: the same start tick under the same boot
    id, or creation times at most a second apart.

    Anything missing, malformed, of different shapes, or ticks under different
    or unknown boot ids returns False. Never True together with
    :func:`start_identity_differs`.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    rt, ct = recorded.get("ticks"), current.get("ticks")
    if type(rt) is int and type(ct) is int:
        rb, cb = recorded.get("boot"), current.get("boot")
        return isinstance(rb, str) and isinstance(cb, str) and rb == cb and rt == ct
    rc, cc = recorded.get("created"), current.get("created")
    if type(rc) in (int, float) and type(cc) in (int, float):
        if math.isfinite(rc) and math.isfinite(cc):
            return abs(rc - cc) <= 1.0
    return False


_PID_SPACE: str | None = None
_LINUX_MACHINE_ID_FILES = (Path("/etc/machine-id"),
                           Path("/var/lib/dbus/machine-id"))


def machine_guid() -> str:
    """This Windows installation's MachineGuid, or "" on other platforms and
    when it cannot be read."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
            value, _ = winreg.QueryValueEx(k, "MachineGuid")
    except (OSError, ImportError):
        return ""
    return str(value).strip()


def linux_machine_id() -> str:
    """This Linux installation's machine-id (/etc/machine-id, else
    /var/lib/dbus/machine-id), or "" on other platforms and when neither can
    be read. Cloned images can share one, so it narrows the host a pid table
    belongs to without identifying it."""
    if not sys.platform.startswith("linux"):
        return ""
    for path in _LINUX_MACHINE_ID_FILES:
        try:
            value = path.read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            continue
        if value:
            return value
    return ""


def pid_space_id() -> str:
    """An opaque id for the pid table this process's pids belong to: the
    platform, the Windows MachineGuid (see :func:`machine_guid`), the Linux
    machine-id (see :func:`linux_machine_id`), the host name and, on Linux, the
    pid namespace, hashed.

    A process that can read none of the machine identifiers, the host name and
    the pid namespace gets an id unique to itself, so no other process's record
    matches it.
    """
    global _PID_SPACE
    if _PID_SPACE is None:
        import hashlib
        import platform
        import uuid
        parts = [sys.platform, machine_guid(), linux_machine_id()]
        try:
            parts.append(platform.node() or "")
        except Exception:
            parts.append("")
        try:
            parts.append(str(os.stat("/proc/self/ns/pid").st_ino))
        except OSError:
            pass
        if not any(parts[1:]):
            parts.append(uuid.uuid4().hex)
        _PID_SPACE = hashlib.sha256(
            "".join(parts).encode("utf-8", "replace")).hexdigest()[:16]
    return _PID_SPACE


def legacy_pid_space_id() -> str | None:
    """The pid-space id that owner records written before the Linux machine-id
    was part of :func:`pid_space_id` carry on this host: the platform, the
    Windows MachineGuid, the host name and, on Linux, the pid namespace,
    hashed. None when none of those but the platform can be read (such a
    record then carries an id unique to its process, which nothing matches)."""
    import hashlib
    import platform
    parts = [sys.platform, machine_guid()]
    try:
        parts.append(platform.node() or "")
    except Exception:
        parts.append("")
    try:
        parts.append(str(os.stat("/proc/self/ns/pid").st_ino))
    except OSError:
        pass
    if not any(parts[1:]):
        return None
    return hashlib.sha256(
        "".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def legacy_record_is_this_host(rec: dict) -> bool:
    """True when owner record *rec*, carrying :func:`legacy_pid_space_id`, was
    written on this host: on Windows when this install has a MachineGuid (it
    is part of that id); on Linux when the record's start identity names this
    boot's boot id. False otherwise, including on every other platform."""
    if not isinstance(rec, dict):
        return False
    if sys.platform == "win32":
        return bool(machine_guid())
    if sys.platform.startswith("linux"):
        start = rec.get("start")
        recorded = start.get("boot") if isinstance(start, dict) else None
        try:
            boot = _LINUX_BOOT_ID.read_text(encoding="ascii").strip() or None
        except (OSError, ValueError):
            return False
        return isinstance(recorded, str) and recorded == boot
    return False


def darwin_host_uuid() -> str:
    """This Mac's hardware UUID from ``gethostuuid``, or "" on other platforms
    and when it cannot be read."""
    if sys.platform != "darwin":
        return ""
    try:
        import ctypes
        import uuid

        class _Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        libc = ctypes.CDLL(None, use_errno=True)
        buf = (ctypes.c_ubyte * 16)()
        wait = _Timespec(1, 0)
        if libc.gethostuuid(buf, ctypes.byref(wait)) != 0:
            return ""
        raw = bytes(buf)
    except (OSError, AttributeError, ValueError):
        return ""
    if not any(raw):
        return ""
    return str(uuid.UUID(bytes=raw))


_HOST_ID: str | None = None
_HOST_ID_READ = False
_LINUX_PID_NS = "/proc/self/ns/pid"


def host_id() -> str | None:
    """An opaque id for this machine's pid table that does not depend on the
    host name: the platform, the Windows MachineGuid, the Linux machine-id,
    the Mac hardware UUID and, on Linux, the pid namespace, hashed. None when
    none of the three machine identifiers can be read, and on Linux when the
    pid namespace cannot be read."""
    global _HOST_ID, _HOST_ID_READ
    if not _HOST_ID_READ:
        import hashlib
        ids = [machine_guid(), linux_machine_id(), darwin_host_uuid()]
        parts = [sys.platform, *ids] if any(ids) else None
        if parts is not None and sys.platform.startswith("linux"):
            try:
                parts.append(str(os.stat(_LINUX_PID_NS).st_ino))
            except OSError:
                parts = None
        if parts is not None:
            _HOST_ID = hashlib.sha256(
                "\0".join(parts).encode("utf-8", "replace")).hexdigest()[:16]
        _HOST_ID_READ = True
    return _HOST_ID
