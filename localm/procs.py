# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which processes of this install are running.

Every process that imports localm writes one small file into ``<root>/.localm-run``
as its first act (:func:`enter`, called by ``localm/__init__.py``), and a long-lived or
detached program that does not import localm is started through :func:`start`, which
writes one for it. A file names the process id, when that process started, what it is,
the process that started it and, when it opened programs for the user, those; no
command line, no environment, no content.

A process never removes its own file, so it is listed until it has really ended. The
file of one that has ended is deleted by the next localm process that starts and by the
next reader. It is the entry of a running process only while

- that process id runs and started at that time,
- the file belongs to this very folder: its name carries the identity of the install
  folder (volume, file id and a number made with the record), so in a copied install,
  or one moved to another volume, the old files are stale; and
- it was written by the account the process runs as, the owner of the folder, or root
  or an administrator.

A short tool has no file: it is a running child of a process that has one, and
:func:`list` finds it there. A program localm opens for the user (a web browser, a
viewer) is opened through :func:`open_for_user` and is never listed and never ended.
Neither is a program below a localm process of which it cannot be told that it is a
tool: see :func:`_users_own`.

Everything that imports localm from this folder is a process of this install, also a
test run, a notebook or a script of the user, and :func:`end_others` ends it.

There is no record for an install that cannot keep one: a pip install, the container
image, a folder this process may not write to, or a system whose process start times
cannot be read. :func:`current` is then None and :func:`problem` says why.

Standard library only, and no import of localm: ``localm/__init__.py`` imports this
module before anything else of localm, and it can be loaded by file path.
"""

import os
import stat
import sys

FOLDER = ".localm-run"

#: The kinds of process :func:`enter` names itself. Anything else is "command"
#: (a localm command), "script" or the dotted name of a localm module.
COMMANDS = frozenset({
    "add", "bench-mtp", "bench-spec", "benchmark", "coder", "comfy", "doctor", "gui",
    "image", "install-packages", "job", "mcp", "memory", "models", "music", "plugin",
    "pull", "rag", "relocate", "rerank", "run", "search", "serve", "setup-browser",
    "setup-embeddings", "setup-llama", "setup-music", "setup-sdcpp", "speak", "status",
    "stop", "update", "video",
})
SCRIPTS = {"localm": "", "localm.exe": "", "localm.py": "", "localcoder": "coder",
           "localcoder.exe": "coder", "launcher.pyw": "launcher", "gui.py": "installer"}

#: How many files one process start looks at to delete those of ended processes.
_SWEEP = 8
#: After this long a file of another machine or container is deleted.
_FOREIGN_S = 7 * 24 * 3600
_MAX_BYTES = 64 * 1024
#: The host of every console window; it ends with the programs it shows.
_CONSOLE_HOSTS = frozenset({"conhost.exe", "openconsole.exe"})
#: The browser process of localm's own app window.
_APP_WINDOW = "msedgewebview2.exe"
_POSIX_GRACE_S = 5.0


class Entry:
    """One running process. *start* is when it started, in a form that only this
    process has on this system; *parent* is ``(pid, start)`` or None; *recorded* is
    whether it has a file of its own; *users* are the ``(pid, start)`` of the programs
    it opened for the user."""

    __slots__ = ("pid", "start", "purpose", "parent", "recorded", "users")

    def __init__(self, pid, start, purpose, parent=None, recorded=True, users=()) -> None:
        self.pid = pid
        self.start = start
        self.purpose = purpose
        self.parent = parent
        self.recorded = recorded
        self.users = frozenset(users)

    @property
    def key(self) -> tuple:
        return (self.pid, self.start)

    def __eq__(self, other) -> bool:
        return isinstance(other, Entry) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        if not self.pid:
            return f"<{self.purpose}>"
        started_by = f" started by {self.parent[0]}" if self.parent else ""
        return f"<{self.purpose} pid {self.pid}{started_by}>"


#: What :func:`end_others` returns when processes may run that it cannot see.
UNSEEN = Entry(0, "", "others may run unseen", recorded=False)


# --------------------------------------------------------------------------- #
#  What the system says about a process                                        #
# --------------------------------------------------------------------------- #
# A row is (parent pid, start, image name or ""). Two starts are compared for equality
# only: 100 ns units since 1601 (Windows), clock ticks since boot (Linux), microseconds
# since 1970 (macOS). Which of two processes started first is asked of _order.
_UNKNOWN = (0, -1, "")
_DLLS: dict = {}


def _win_dll(name: str):
    import ctypes
    if sys.platform != "win32":
        raise OSError("not Windows")
    if name not in _DLLS:
        _DLLS[name] = ctypes.WinDLL(name, use_last_error=True)
    return _DLLS[name]


def _win_row(handle):
    """The row of the process *handle* is open on, or :data:`_UNKNOWN` when the system
    does not say."""
    import ctypes
    k32, nt = _win_dll("kernel32"), _win_dll("ntdll")
    times = (ctypes.c_uint64 * 4)()
    base = ctypes.addressof(times)
    k32.GetProcessTimes.argtypes = [ctypes.c_void_p] * 5
    if not k32.GetProcessTimes(handle, base, base + 8, base + 16, base + 24):
        return _UNKNOWN
    basic = (ctypes.c_void_p * 6)()
    nt.NtQueryInformationProcess.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    if nt.NtQueryInformationProcess(handle, 0, ctypes.addressof(basic),
                                    ctypes.sizeof(basic), None) != 0:
        return _UNKNOWN
    return (int(basic[5] or 0), int(times[0]), "")


def _win_open(pid: int, access: int):
    import ctypes
    if sys.platform != "win32":
        return None
    k32 = _win_dll("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    ctypes.set_last_error(0)
    return k32.OpenProcess(access, 0, pid)


def _win_close(handle) -> None:
    import ctypes
    k32 = _win_dll("kernel32")
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.CloseHandle(handle)


def _win_exited(handle) -> bool:
    """Whether the process has ended; the handle has the right SYNCHRONIZE."""
    import ctypes
    k32 = _win_dll("kernel32")
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    return k32.WaitForSingleObject(handle, 0) == 0


def _win_process(pid: int, table=None):
    import ctypes
    if sys.platform != "win32":
        return None
    handle = _win_open(pid, 0x101000)       # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if handle:
        try:
            return None if _win_exited(handle) else _win_row(handle)
        finally:
            _win_close(handle)
    if ctypes.get_last_error() == 87:       # ERROR_INVALID_PARAMETER: no such process
        return None
    return (_win_table() if table is None else table).get(pid)


def _win_table() -> dict:
    """Every process, from one ``NtQuerySystemInformation(SystemProcessInformation)``."""
    import ctypes
    import struct
    nt = _win_dll("ntdll")
    nt.NtQuerySystemInformation.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    nt.NtQuerySystemInformation.restype = ctypes.c_uint32
    size, need = 1 << 20, ctypes.c_uint32(0)
    for _ in range(8):
        buf = ctypes.create_string_buffer(size)
        status = nt.NtQuerySystemInformation(5, buf, size, ctypes.addressof(need))
        if status == 0:
            break
        if status != 0xC0000004:            # STATUS_INFO_LENGTH_MISMATCH
            raise OSError(f"the running processes cannot be listed (0x{status:08X})")
        size = max(size, need.value) + (1 << 18)
    else:
        raise OSError("the running processes cannot be listed")
    base, raw, rows, at = ctypes.addressof(buf), buf.raw, {}, 0
    while True:
        step, threads = struct.unpack_from("<II", raw, at)
        created, = struct.unpack_from("<q", raw, at + 32)
        length, name_at = struct.unpack_from("<H6xQ", raw, at + 56)
        pid, parent = struct.unpack_from("<QQ", raw, at + 80)
        if threads:
            name = ""
            if length and base <= name_at < base + size:
                off = name_at - base
                name = raw[off:off + length].decode("utf-16-le", "replace").lower()
            rows[pid] = (parent, created, name)
        if not step:
            return rows
        at += step


def _win_order(pid: int, start: int):
    """The number of the process in the order of creation since this system was
    installed (``ProcessTelemetryIdInformation``), which no clock is part of."""
    import ctypes
    import struct
    handle = _win_open(pid, 0x1000)
    if not handle:
        return None
    try:
        row = _win_row(handle)
        if row == _UNKNOWN or row[1] != start:
            return None
        nt = _win_dll("ntdll")
        buf = ctypes.create_string_buffer(8192)
        if nt.NtQueryInformationProcess(handle, 64, ctypes.addressof(buf), 8192, None) != 0:
            return None
        return struct.unpack_from("<Q", buf.raw, 8)[0]
    finally:
        _win_close(handle)


def _linux_row(raw: bytes):
    fields = raw[raw.rindex(b")") + 2:].split()
    if fields[0] in (b"Z", b"X"):
        return None
    return (int(fields[1]), int(fields[19]), "")


def _linux_process(pid: int):
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return _linux_row(fh.read())
    except (OSError, ValueError, IndexError):
        return None if _posix_gone(pid) else _UNKNOWN


def _linux_table() -> dict:
    rows = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat", "rb") as fh:
                    row = _linux_row(fh.read())
            except (OSError, ValueError, IndexError):
                continue
            if row:
                rows[int(name)] = row
    if os.getpid() not in rows:
        raise OSError("the running processes cannot be listed")
    return rows


def _posix_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return True
    except OSError:
        pass
    return False


def _mac_lib():
    import ctypes
    if "proc" not in _DLLS:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _DLLS["proc"] = lib
    return _DLLS["proc"]


def _mac_info(pid: int) -> bytes:
    """``struct proc_bsdinfo`` of *pid*, also of one that has ended and was not waited
    for (without the 1 the system says nothing about it), or b""."""
    import ctypes
    buf = ctypes.create_string_buffer(136)
    ctypes.set_errno(0)
    done = _mac_lib().proc_pidinfo(pid, 3, 1, buf, 136)     # PROC_PIDTBSDINFO
    return buf.raw if done == 136 else b""


def _mac_process(pid: int):
    import ctypes
    import errno
    import struct
    raw = _mac_info(pid)
    if not raw:
        gone = ctypes.get_errno() == errno.ESRCH or _posix_gone(pid)
        return None if gone else _UNKNOWN
    if struct.unpack_from("=I", raw, 4)[0] == 5:            # SZOMB
        return None
    sec, usec = struct.unpack_from("=QQ", raw, 120)
    return (struct.unpack_from("=I", raw, 16)[0], sec * 1000000 + usec, "")


def _mac_table() -> dict:
    import ctypes
    lib = _mac_lib()
    count = lib.proc_listallpids(None, 0)
    pids = (ctypes.c_int * (max(count, 0) + 256))()
    count = lib.proc_listallpids(pids, ctypes.sizeof(pids))
    if count <= 0:
        raise OSError("the running processes cannot be listed")
    rows = {}
    for pid in pids[:count]:
        row = _mac_process(pid) if pid > 0 else None
        if row and row != _UNKNOWN:
            rows[pid] = row
    return rows


def _mac_order(pid: int, start: int):
    """When the process started on the clock that counts from boot
    (``proc_pid_rusage``, ``ri_proc_start_abstime``)."""
    import ctypes
    import struct
    row = _mac_process(pid)
    buf = ctypes.create_string_buffer(96)
    if not row or row[1] != start or _mac_lib().proc_pid_rusage(pid, 0, buf) != 0:
        return None
    return struct.unpack_from("=Q", buf.raw, 80)[0]


def _table() -> dict:
    """{pid: row} of every running process. Raises OSError when this system does not
    say."""
    if sys.platform == "win32":
        return _win_table()
    if sys.platform.startswith("linux"):
        return _linux_table()
    if sys.platform == "darwin":
        return _mac_table()
    raise OSError("the running processes cannot be listed on this system")


def _process(pid: int, table=None):
    """The row of *pid*, None when no such process runs, or :data:`_UNKNOWN` when one
    runs that this process may not look at."""
    if sys.platform == "win32":
        return _win_process(pid, table)
    if sys.platform.startswith("linux"):
        return _linux_process(pid)
    if sys.platform == "darwin":
        return _mac_process(pid)
    raise OSError("process start times cannot be read on this system")


def _own_row():
    if sys.platform == "win32":
        import ctypes
        return _win_row(ctypes.c_void_p(-1))
    return _process(os.getpid())


def _order(pid: int, start: int):
    """A number that is smaller for a process that started earlier on this boot,
    whatever was done to the clock, or None when the system does not say."""
    if sys.platform == "win32":
        return _win_order(pid, start)
    if sys.platform == "darwin":
        return _mac_order(pid, start)
    return start


def _later(one, other):
    """Whether process *one* started after *other*, each ``(pid, start number)``; by the
    clock when the system gives no order."""
    first, second = _order(*one), _order(*other)
    if first is None or second is None:
        first, second = one[1], other[1]
    return first > second


def _account(pid: int):
    """The account *pid* runs as, or None when that cannot be read."""
    if sys.platform == "win32":
        return _win_account(pid)
    if sys.platform == "darwin":
        import struct
        raw = _mac_info(pid)
        return struct.unpack_from("=I", raw, 20)[0] if raw else None
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


# --------------------------------------------------------------------------- #
#  Accounts and rights on Windows                                              #
# --------------------------------------------------------------------------- #

#: The accounts that may change the install anyway: SYSTEM and the administrators.
_WIN_ADMINS = ("S-1-5-18", "S-1-5-32-544")
#: Every account that signs in carries these.
_WIN_EVERYONE = ("S-1-1-0", "S-1-5-11", "S-1-5-32-545")
_WIN_RUN = 0x21                             # read and traverse
_WIN_WRITE = 0x2                            # add a file


def _win_sid_text(sid) -> str:
    import ctypes
    adv, k32 = _win_dll("advapi32"), _win_dll("kernel32")
    text = ctypes.c_void_p()
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    if not sid or not adv.ConvertSidToStringSidW(sid, ctypes.addressof(text)):
        return ""
    try:
        return ctypes.wstring_at(text.value or 0)
    finally:
        k32.LocalFree.argtypes = [ctypes.c_void_p]
        k32.LocalFree(text)


def _win_account(pid: int):
    import ctypes
    adv = _win_dll("advapi32")
    handle = _win_open(pid, 0x1000)
    if not handle:
        return None
    token = ctypes.c_void_p()
    try:
        adv.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        if not adv.OpenProcessToken(handle, 0x8, ctypes.addressof(token)):     # TOKEN_QUERY
            return None
        buf = ctypes.create_string_buffer(512)
        got = ctypes.c_uint32(0)
        adv.GetTokenInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        if not adv.GetTokenInformation(token, 1, ctypes.addressof(buf), 512,   # TokenUser
                                       ctypes.addressof(got)):
            return None
        return _win_sid_text(ctypes.c_void_p.from_buffer(buf).value) or None
    finally:
        if token:
            _win_close(token)
        _win_close(handle)


class _WinSecurity:
    """Owner and rights of one file or folder."""

    def __init__(self, path: str) -> None:
        import ctypes
        adv = _win_dll("advapi32")
        self._free = _win_dll("kernel32").LocalFree
        self._free.argtypes = [ctypes.c_void_p]
        owner, self.dacl, self._sd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        adv.GetNamedSecurityInfoW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_int, ctypes.c_uint32] + [ctypes.c_void_p] * 5
        adv.GetNamedSecurityInfoW.restype = ctypes.c_uint32
        error = adv.GetNamedSecurityInfoW(
            path, 1, 0x5, ctypes.addressof(owner), None, ctypes.addressof(self.dacl), None,
            ctypes.addressof(self._sd))
        if error:
            raise OSError(error, "the rights cannot be read")
        self.owner = _win_sid_text(owner.value)

    def accounts(self) -> set:
        """Each account the rights name, as text."""
        import ctypes
        adv, found, i = _win_dll("advapi32"), set(), 0
        adv.GetAce.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        ace = ctypes.c_void_p()
        while self.dacl and adv.GetAce(self.dacl, i, ctypes.addressof(ace)):
            kind, flags = ctypes.string_at(ace.value or 0, 2)
            if kind == 0 and not flags & 0x8:       # allowed, and not only inherited
                found.add(_win_sid_text((ace.value or 0) + 8))
            i += 1
        return found - {""}

    def mask(self, account: str) -> int:
        """What the rights let *account* do."""
        if not self.dacl:
            return 0x1F01FF
        import ctypes
        adv = _win_dll("advapi32")
        sid = _win_sid(account)
        trustee = _win_trustee(sid)
        got = ctypes.c_uint32(0)
        adv.GetEffectiveRightsFromAclW.argtypes = [ctypes.c_void_p] * 3
        adv.GetEffectiveRightsFromAclW.restype = ctypes.c_uint32
        error = adv.GetEffectiveRightsFromAclW(
            self.dacl, ctypes.addressof(trustee), ctypes.addressof(got))
        self._free(sid)
        if error:
            raise OSError(error, "the rights cannot be read")
        return got.value

    def close(self) -> None:
        self._free(self._sd)


def _win_sid(text: str):
    import ctypes
    adv = _win_dll("advapi32")
    sid = ctypes.c_void_p()
    adv.ConvertStringSidToSidW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
    if not adv.ConvertStringSidToSidW(text, ctypes.addressof(sid)):
        raise OSError(f"no such account: {text}")
    return sid


def _win_trustee(sid):
    import ctypes

    class Trustee(ctypes.Structure):
        _fields_ = [("multiple", ctypes.c_void_p), ("operation", ctypes.c_int),
                    ("form", ctypes.c_int), ("kind", ctypes.c_int), ("name", ctypes.c_void_p)]
    return Trustee(None, 0, 0, 0, sid.value)


def _win_kept_out(install: str, folder: str) -> list:
    """The accounts that may run the install at *install* but may not add a file to
    *folder*."""
    outer, inner = _WinSecurity(install), _WinSecurity(folder)
    try:
        every_write = 0
        for account in _WIN_EVERYONE:
            every_write |= inner.mask(account)
        out = []
        for account in sorted(outer.accounts() | set(_WIN_EVERYONE)):
            try:
                run, write = outer.mask(account), inner.mask(account) | every_write
            except OSError:                     # no account has this id any more
                continue
            if account not in _WIN_ADMINS and run & _WIN_RUN == _WIN_RUN and not write & _WIN_WRITE:
                out.append(account)
        return out
    finally:
        outer.close()
        inner.close()


def _win_let_in(folder: str, accounts) -> None:
    """Let *accounts* add files to *folder* and list it, and each of them do anything to
    the files it adds."""
    import ctypes
    adv = _win_dll("advapi32")
    security = _WinSecurity(folder)
    sids = [_win_sid(a) for a in [*accounts, "S-1-3-0"]]        # CREATOR OWNER
    trustee = type(_win_trustee(sids[0]))

    class Access(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_uint32), ("mode", ctypes.c_int),
                    ("inherit", ctypes.c_uint32), ("trustee", trustee)]
    entries = (Access * len(sids))()
    for i, sid in enumerate(sids):
        last = i == len(sids) - 1
        entries[i] = Access(0x1F01FF if last else 0x1200AB, 1, 0xB if last else 0,
                            trustee(None, 0, 0, 0, sid.value))
    made = ctypes.c_void_p()
    adv.SetEntriesInAclW.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    adv.SetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_int, ctypes.c_uint32] + [ctypes.c_void_p] * 4
    try:
        if adv.SetEntriesInAclW(len(sids), ctypes.addressof(entries), security.dacl,
                                ctypes.addressof(made)) == 0:
            adv.SetNamedSecurityInfoW(folder, 1, 0x4, None, None, made, None)
    finally:
        for sid in sids:
            security._free(sid)
        if made:
            security._free(made)
        security.close()


def _trusted_writer(path: str, folder: str, pid: int) -> bool:
    """Whether the file at *path* was written by an account that may say what *pid* is:
    the one *pid* runs as, the owner of the folder, or root or an administrator. True
    where the system does not say who wrote it."""
    account = _account(pid)
    if account is None:
        return True
    if sys.platform != "win32":
        writer = os.lstat(path).st_uid
        return writer in (account, 0, os.stat(folder).st_uid)
    try:
        mine, above = _WinSecurity(path), _WinSecurity(folder)
    except OSError:
        return True
    try:
        return not mine.owner or mine.owner in (account, above.owner, *_WIN_ADMINS)
    finally:
        mine.close()
        above.close()


def _open_up(install: str, folder: str) -> None:
    """Let everyone who may run the install at *install* add a file to its new
    *folder*, and nobody else; a file belongs to the account that added it."""
    if sys.platform == "win32":
        kept_out = _win_kept_out(install, folder)
        if kept_out:
            _win_let_in(folder, kept_out)
        return
    of = os.stat(install)
    mode = 0o700
    for run, all_of in ((0o050, 0o070), (0o005, 0o007)):
        if of.st_mode & run == run:
            mode |= all_of
    if os.geteuid() == 0:
        os.chown(folder, of.st_uid, of.st_gid)
    elif mode & 0o070 and os.stat(folder).st_gid != of.st_gid:
        try:
            os.chown(folder, -1, of.st_gid)
        except OSError:
            mode &= ~0o070
    os.chmod(folder, mode | (0o1000 if mode & 0o077 else 0))


def _others_kept_out(install: str) -> bool:
    """Whether someone may run the install at *install* without being able to add a
    file to its record."""
    folder = os.path.join(install, FOLDER)
    if sys.platform == "win32":
        return bool(_win_kept_out(install, folder))
    outer, inner = os.stat(install), os.stat(folder)
    for run, write in ((0o050, 0o030), (0o005, 0o003)):
        if outer.st_mode & run == run and inner.st_mode & write != write:
            return True
    return bool(outer.st_mode & 0o050 == 0o050 and inner.st_gid != outer.st_gid
                and inner.st_mode & 0o003 != 0o003)


# --------------------------------------------------------------------------- #
#  Start times, hosts and folders as text                                      #
# --------------------------------------------------------------------------- #

_BOOT = None
_HOSTS: dict = {}


def _boot() -> str:
    """Linux: the boot that start ticks count from. Empty elsewhere."""
    global _BOOT
    if _BOOT is None:
        _BOOT = ""
        if sys.platform.startswith("linux"):
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as fh:
                number = int(fh.read().strip().replace("-", ""), 16)
            _BOOT = f"{number:032x}"
    return _BOOT


def _text(number: int) -> str:
    return f"{_boot()}.{number}" if _boot() else str(number)


def _digits(text: str) -> bool:
    return text.isascii() and text.isdigit() and len(text) < 24


def _number(started: str):
    """The number in *started*, or None when it counts from another boot."""
    boot, _, number = started.rpartition(".")
    return int(number) if boot == _boot() and _digits(number) else None


def _mac_uuid() -> str:
    """This Mac's hardware UUID."""
    import ctypes
    raw, wait = (ctypes.c_ubyte * 16)(), (ctypes.c_long * 2)(1, 0)
    if ctypes.CDLL(None).gethostuuid(raw, wait) != 0:
        raise OSError("this machine has no id")
    return bytes(raw).hex()


def _hash(*parts) -> str:
    value = 0xCBF29CE484222325
    for byte in "\0".join(str(p) for p in parts).encode("utf-8", "replace"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _machine() -> str:
    """The process table this process is in, which the machine's name is not part of:
    one machine, and on Linux one pid namespace."""
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            return str(winreg.QueryValueEx(key, "MachineGuid")[0])
    if sys.platform.startswith("linux"):
        text = str(os.stat("/proc/self/ns/pid").st_ino)
        try:
            with open("/etc/machine-id", encoding="ascii") as fh:
                return text + fh.read().strip()
        except (OSError, ValueError):
            return text + _boot()
    return _mac_uuid()


def _host(mark: str) -> str:
    """This machine's id within the record that has *mark*: without the mark, which no
    other folder has, it says nothing about the machine."""
    if mark not in _HOSTS:
        _HOSTS[mark] = _hash(mark, _machine())
    return _HOSTS[mark]


def _marks(names) -> list:
    """The numbers made with the record, from the names of its files."""
    return sorted(n[:-5] for n in names
                  if len(n) == 21 and n.endswith(".mark") and _hex(n[:-5]))


def _hex(text: str, length: int = 16) -> bool:
    return len(text) == length and not set(text) - set("0123456789abcdef")


def _folder_id(install: str, mark: str) -> str:
    at = os.stat(install)
    return _hash(mark, at.st_dev, at.st_ino)


# --------------------------------------------------------------------------- #
#  The files                                                                   #
# --------------------------------------------------------------------------- #

_CURRENT = None
_PROBLEM = "localm has not been imported"
_PLACE = ("", "", "")       # folder, folder id and host id of this process's record


def root() -> str:
    """The folder of the install this code is part of."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def current():
    """The :class:`Entry` of this process, or None when it has no file; :func:`problem`
    says why."""
    return _CURRENT if _CURRENT is not None and _CURRENT.pid == os.getpid() else None


def problem() -> str:
    """Why this process has no entry. Empty when it has one."""
    if current() is not None:
        return ""
    return _PROBLEM or "a forked child has no entry of its own"


def _parse(name: str):
    """``(folder id, host id, pid, start)`` of the name of an entry, or None."""
    parts = name[:-5].split("-") if name.endswith(".json") else []
    if len(parts) != 4 or not _hex(parts[0]) or not _hex(parts[1]) or not _digits(parts[2]):
        return None
    if not 0 < int(parts[2]) < 1 << 32:
        return None
    boot, _, number = parts[3].rpartition(".")
    if not _digits(number) or not (boot == "" or _hex(boot, 32)):
        return None
    return parts[0], parts[1], int(parts[2]), parts[3]


def _name(at: str, host: str, pid: int, started: str) -> str:
    """The name of the file of an entry: hex, hex, digits, hex and digits, whatever is
    given. Raises ValueError for anything else than the parts of a name."""
    boot, _, number = started.rpartition(".")
    when = f"{int(number)}"
    if boot:
        when = f"{int(boot, 16):032x}.{when}"
    name = f"{int(at, 16):016x}-{int(host, 16):016x}-{int(pid)}-{when}.json"
    if type(pid) is not int or _parse(name) != (at, host, pid, started):
        raise ValueError("not the parts of a name")
    return name


def _content(entry: Entry) -> bytes:
    def one(key) -> str:
        return '{"pid": %d, "start": "%s"}' % key
    text = '{"pid": %d, "start": "%s", "purpose": "%s", "parent": %s' % (
        entry.pid, entry.start, entry.purpose, one(entry.parent) if entry.parent else "null")
    if entry.users:
        text += ', "users": [%s]' % ", ".join(one(k) for k in sorted(entry.users))
    return (text + "}\n").encode("ascii")


def _write(place, entry: Entry) -> None:
    """Write the file of *entry* whole, or not at all."""
    folder, at, host = place
    path = os.path.join(folder, _name(at, host, entry.pid, entry.start))
    new = path[:-5] + ".new"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(new, flags, 0o644)
    except FileExistsError:
        os.unlink(new)
        fd = os.open(new, flags, 0o644)
    try:
        os.write(fd, _content(entry))
    finally:
        os.close(fd)
    for again in range(20):
        try:
            os.replace(new, path)
            return
        except PermissionError:
            if again == 19:
                _unlink(new)
                raise
            import time
            time.sleep(0.01)


def _make_folder(install: str) -> None:
    if (os.path.basename(install).lower() in ("site-packages", "dist-packages")
            or not os.path.isfile(os.path.join(install, "pyproject.toml"))):
        raise OSError("this is not an install that updates itself")
    folder = os.path.join(install, FOLDER)
    try:
        os.mkdir(folder, 0o700)
    except FileExistsError:
        return
    try:
        _open_up(install, folder)
    except OSError:
        pass


def _make_mark(folder: str) -> str:
    mark = os.urandom(8).hex()
    os.close(os.open(os.path.join(folder, mark + ".mark"),
                     os.O_WRONLY | os.O_CREAT, 0o644))
    return mark


def _parent_of(pid: int, row):
    """``(pid, start)`` of the parent that *row*, the row of *pid*, names, or None when
    it has gone or started after its child."""
    other = _process(row[0]) if row[0] else None
    if not other or other == _UNKNOWN or _later((row[0], other[1]), (pid, row[1])):
        return None
    return (row[0], _text(other[1]))


def _purpose(argv) -> str:
    """What the interpreter started as *argv* is."""
    if "--multiprocessing-fork" in argv:
        return "worker"
    args = iter(argv[1:])
    for arg in args:
        if arg == "-m":
            module = next(args, "")
            if module != "localm":
                known = module.startswith("localm.") and module.replace(".", "").replace(
                    "_", "").isalnum() and module.isascii()
                return module if known else "script"
            break
        if arg in ("-X", "-W"):
            next(args, None)
        elif not arg.startswith("-"):
            base = os.path.basename(arg).lower()
            if base not in SCRIPTS:
                return "script"
            if SCRIPTS[base]:
                return SCRIPTS[base]
            break
        elif arg.startswith("-c"):
            return "script"
    return next((a for a in args if a in COMMANDS), "command")


def enter() -> None:
    """Record this process. Called once, by ``localm/__init__.py``, before anything
    else of localm is imported. Never raises: a process that cannot be recorded has no
    entry (:func:`current`)."""
    global _CURRENT, _PLACE, _PROBLEM
    if os.environ.get("LOCALM_CONTAINER_IMAGE"):
        _PROBLEM = "the container image keeps no record"
        return
    try:
        row = _own_row()
        if not row or row == _UNKNOWN:
            raise OSError("the start time of this process cannot be read")
        entry = Entry(os.getpid(), _text(row[1]), _purpose(sys.orig_argv),
                      _parent_of(os.getpid(), row))
        install = root()
        folder = os.path.join(install, FOLDER)
        try:
            names = os.listdir(folder)
        except FileNotFoundError:
            _make_folder(install)
            names = os.listdir(folder)
        marks = _marks(names) or [_make_mark(folder)]
        place = (folder, _folder_id(install, marks[0]), _host(marks[0]))
        _write(place, entry)
        _CURRENT, _PLACE, _PROBLEM = entry, place, ""
        _sweep(install, names, marks)
    except Exception as e:
        _PROBLEM = str(e) or type(e).__name__


def start(args, *, purpose: str, **kwargs):
    """``subprocess.Popen(args, **kwargs)`` for a long-lived or detached program of
    localm that does not import localm, recorded as *purpose* (lower-case words)."""
    import subprocess
    if not (purpose.replace(" ", "").isalnum() and purpose.isascii()):
        raise ValueError(purpose)
    proc = subprocess.Popen(args, **kwargs)
    me = current()
    if me is not None:
        try:
            row = _process(proc.pid)
            if row and row != _UNKNOWN and row[0] == me.pid:
                _write(_PLACE, Entry(proc.pid, _text(row[1]), purpose, me.key))
        except Exception:
            return proc
    return proc


def _own_children() -> set:
    table = _table()
    return {(pid, _text(row[1])) for pid, row in table.items() if row[0] == os.getpid()}


def open_for_user(call, *args, **kwargs):
    """``call(*args, **kwargs)``, which opens something in a program of the user (a web
    browser, a viewer, an editor). The programs this process starts meanwhile are noted
    as the user's: never listed, never ended."""
    global _CURRENT
    me, before = current(), set()
    try:
        before = _own_children() if me is not None else set()
    except Exception:
        me = None
        _note("a program opened for the user cannot be noted")
    try:
        return call(*args, **kwargs)
    finally:
        if me is not None:
            try:
                opened = _own_children() - before
                me = current()
                if opened and me is not None:
                    entry = Entry(me.pid, me.start, me.purpose, me.parent,
                                  users=me.users | opened)
                    _write(_PLACE, entry)
                    _CURRENT = entry
            except Exception:
                _note("a program opened for the user could not be noted")


def _note(text: str) -> None:
    import logging
    logging.getLogger("localm").debug("procs: %s", text, exc_info=True)


# --------------------------------------------------------------------------- #
#  Reading                                                                     #
# --------------------------------------------------------------------------- #

def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _state(pid: int, start: str, table=None):
    """True when the process *pid* that started at *start* runs, False when it does
    not, None when one with that pid runs whose start cannot be read."""
    number = _number(start)
    if number is None:
        return False
    row = table.get(pid) if table else None
    if not row or row[1] != number:
        row = _process(pid, table)
    if row == _UNKNOWN:
        return None
    return bool(row) and row[1] == number


def alive(entry: Entry) -> bool:
    """Whether the process of *entry* still runs. One that cannot be looked at (it
    belongs to another account) counts as running, and so does :data:`UNSEEN`."""
    return not entry.pid or _state(entry.pid, entry.start) is not False


class _Record:
    """The files of one install, by what they are."""

    def __init__(self, install: str, marks) -> None:
        self.install = install
        self.folder = os.path.join(install, FOLDER)
        self.hosts = {_folder_id(install, m): _host(m) for m in marks}
        self._again = True

    def _knows(self, folder_id: str) -> bool:
        """Whether *folder_id* is this folder's. A number that was made after the files
        were listed is looked for once more: it is made before the first file with it."""
        if folder_id not in self.hosts and self._again:
            self._again = False
            for m in _marks(os.listdir(self.folder)):
                self.hosts.setdefault(_folder_id(self.install, m), _host(m))
        return folder_id in self.hosts

    def kind(self, name: str) -> tuple:
        """("mark" | "entry" | "new" | "foreign" | "stale", what the name says)."""
        if _marks([name]):
            return "mark", None
        new = name.endswith(".new")
        said = _parse(name[:-4] + ".json" if new else name)
        if said is None or not self._knows(said[0]):
            return "stale", None
        if said[1] != self.hosts[said[0]]:
            return "foreign", said
        return ("new" if new else "entry"), said

    def aged(self, name: str) -> bool:
        import time
        try:
            return abs(time.time() - os.lstat(os.path.join(self.folder, name)).st_mtime) > _FOREIGN_S
        except OSError:
            return False

    def drop(self, name: str) -> None:
        _unlink(os.path.join(self.folder, name))


def _sweep(install: str, names, marks) -> None:
    """Delete the files of ended processes among :data:`_SWEEP` of *names*. The system
    is asked for its table of processes at most once."""
    record = _Record(install, marks)
    table = None
    first = os.getpid() // 4 % len(names) if names else 0
    for i in range(min(_SWEEP, len(names))):
        name = names[(first + i) % len(names)]
        kind, said = record.kind(name)
        if kind == "stale" or (kind == "foreign" and record.aged(name)):
            record.drop(name)
        elif said is not None and kind != "foreign":
            number = _number(said[3])
            row = None if number is None else _process(said[2], table if table else {})
            if row is None and number is not None and not _gone(said[2]):
                table = table if table is not None else _table()
                row = table.get(said[2])
            if number is None or row is None or (row != _UNKNOWN and row[1] != number):
                record.drop(name)


def _gone(pid: int) -> bool:
    """Whether no process has *pid*, as far as this process may look."""
    if sys.platform != "win32":
        return _posix_gone(pid)
    import ctypes
    handle = _win_open(pid, 0x101000)
    if handle:
        try:
            return _win_exited(handle)
        finally:
            _win_close(handle)
    return ctypes.get_last_error() == 87


def _read(path: str, said) -> "Entry | None":
    """The entry in the file at *path*, None when it is not one. Raises OSError when it
    cannot be read now."""
    import json
    at = os.lstat(path)
    if not stat.S_ISREG(at.st_mode) or at.st_size > _MAX_BYTES:
        return None
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        raw = os.read(fd, _MAX_BYTES)
    finally:
        os.close(fd)
    try:
        data = json.loads(raw.decode("ascii"))
        keys = [(k["pid"], k["start"]) for k in
                [data, *([data["parent"]] if data["parent"] else []), *data.get("users", [])]]
        purpose = data["purpose"]
        right = keys[0] == (said[2], said[3]) and isinstance(purpose, str) and purpose.isascii() \
            and purpose.isprintable() and all(
                type(p) is int and p > 0 and isinstance(s, str) for p, s in keys)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
        return None
    if not right:
        return None
    rest = keys[1:]
    parent = rest.pop(0) if data["parent"] else None
    return Entry(said[2], said[3], purpose, parent, users=rest)


def _recorded(install: str, table=None) -> dict:
    """{pid: Entry} of the running processes that have a file; deletes the files that
    are not theirs."""
    folder = os.path.join(install, FOLDER)
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return {}
    if table is None:
        table = _table()
    record, found = _Record(install, _marks(names)), {}
    for name in names:
        kind, said = record.kind(name)
        if kind == "stale" or (kind == "foreign" and record.aged(name)):
            record.drop(name)
        if said is None or kind == "foreign":
            continue
        pid, at = said[2], said[3]
        if _state(pid, at, table) is False:
            record.drop(name)
            continue
        if kind == "new":
            continue
        path = os.path.join(folder, name)
        try:
            entry = _read(path, said)
            if entry is None:
                record.drop(name)
                continue
            if not _trusted_writer(path, folder, pid):
                continue
        except FileNotFoundError:
            continue
        except OSError:
            entry = Entry(pid, at, "localm")
        found[pid] = entry
    return found


def elsewhere(install=None) -> int:
    """How many files belong to another machine, or to another container, that uses
    the same folder. Whether those processes run cannot be told from here; such a file
    is deleted a week after it was written."""
    install = os.fspath(install or root())
    try:
        names = os.listdir(os.path.join(install, FOLDER))
    except FileNotFoundError:
        return 0
    record = _Record(install, _marks(names))
    return sum(1 for n in names if record.kind(n)[0] == "foreign" and not record.aged(n))


def _children(table: dict) -> dict:
    """{pid: the pids that name it as their parent}. Whether it is, _proven says."""
    out: dict = {}
    for pid, row in table.items():
        if row[0] != pid and row[0] in table:
            out.setdefault(row[0], []).append(pid)
    return out


def _proven(above: int, pid: int, table: dict) -> bool:
    """Whether the process *above* is the parent of *pid*, which names that number: it
    did not start after its child."""
    return not _later((above, table[above][1]), (pid, table[pid][1]))


# --------------------------------------------------------------------------- #
#  Programs of the user                                                        #
# --------------------------------------------------------------------------- #

def _win_image(pid: int) -> str:
    import ctypes
    k32 = _win_dll("kernel32")
    handle = _win_open(pid, 0x1000)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_uint32(32768)
        k32.QueryFullProcessImageNameW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        done = k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.addressof(size))
        return buf.value if done else ""
    finally:
        _win_close(handle)


def _win_console_program(image: str) -> bool:
    """Whether the program file *image* is one for the console, as every tool is and no
    web browser, viewer or editor with a window of its own."""
    import struct
    try:
        with open(image, "rb") as fh:
            raw = fh.read(4096)
        return struct.unpack_from("<H", raw, struct.unpack_from("<I", raw, 0x3C)[0] + 92)[0] == 3
    except (OSError, struct.error):
        return False


def _win_session(pid: int) -> int:
    import ctypes
    k32 = _win_dll("kernel32")
    got = ctypes.c_uint32(0)
    k32.ProcessIdToSessionId.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    return got.value if k32.ProcessIdToSessionId(pid, ctypes.addressof(got)) else -1


def _win_windows():
    """The pids that show a window of their own, console windows left out, or None when
    this process cannot see the windows of the user: it is a service, came over the
    network, or runs on another desktop."""
    import ctypes
    if sys.platform != "win32":
        return None
    user, k32 = _win_dll("user32"), _win_dll("kernel32")
    for fn in (user.GetProcessWindowStation, user.GetThreadDesktop):
        fn.restype = ctypes.c_void_p
    user.GetProcessWindowStation.argtypes = []
    user.GetThreadDesktop.argtypes = [ctypes.c_uint32]
    user.GetUserObjectInformationW.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    text = ctypes.create_unicode_buffer(256)
    for handle, usual in ((user.GetProcessWindowStation(), "WinSta0"),
                          (user.GetThreadDesktop(k32.GetCurrentThreadId()), "Default")):
        if not handle or not user.GetUserObjectInformationW(
                handle, 2, ctypes.addressof(text), 512, None) or text.value != usual:
            return None
    pids: set = set()
    owner = ctypes.c_uint32(0)
    for fn in (user.IsWindowVisible, user.GetWindow, user.GetClassNameW,
               user.GetWindowThreadProcessId):
        fn.restype = ctypes.c_void_p
    user.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    user.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    user.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def see(hwnd, _):
        if user.IsWindowVisible(hwnd) and not user.GetWindow(hwnd, 4):     # GW_OWNER
            user.GetClassNameW(hwnd, text, 64)
            if text.value not in ("ConsoleWindowClass", "PseudoConsoleWindow"):
                user.GetWindowThreadProcessId(hwnd, ctypes.addressof(owner))
                pids.add(owner.value)
        return 1

    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    user.EnumWindows.argtypes = [proto, ctypes.c_void_p]
    return pids if user.EnumWindows(proto(see), None) else None


def _within(path: str, folder: str) -> bool:
    path, folder = (os.path.normcase(os.path.realpath(p)) for p in (path, folder))
    return path.startswith(folder.rstrip("\\/") + os.sep)


def _users_own(install: str, table: dict):
    """A test ``(pid, above, direct)`` for an unrecorded program below a recorded
    process: True when it is, or may be, a program of the user. *above* started it and
    *direct* says that *above* is recorded.

    Windows: a program that is not one for the console, or shows a window. When the
    windows cannot be seen, a console program that localm started itself is a tool (the
    places that open something for the user go through open_for_user), and one further
    down may be the user's. Elsewhere: a program in a session of its own."""
    if sys.platform == "win32":
        windows = _win_windows()
        mine = _win_session(os.getpid())

        def may_be(pid: int, above: int, direct: bool) -> bool:
            if table[pid][2] in _CONSOLE_HOSTS:
                return True
            if table[pid][2] == _APP_WINDOW and (direct or table[above][2] == _APP_WINDOW):
                return False
            image = _win_image(pid)
            if image and _within(image, install):
                return False
            if not image or not _win_console_program(image):
                return True
            if windows is None or _win_session(pid) != mine:
                return not (direct or _is_starter(above, install, table))
            return pid in windows
        return may_be

    def own_session(pid: int, above: int, direct: bool) -> bool:
        try:
            return os.getsid(pid) != os.getsid(above)
        except OSError:
            return True
    return own_session


def _below(top: int, install: str, recorded: dict, table: dict, children: dict, skip) -> dict:
    """{pid: Entry} of the running processes of localm below *top*, without it."""
    found: dict = {}
    users = {k for e in recorded.values() for k in e.users}
    todo = [top]
    while todo:
        above = todo.pop()
        for pid in children.get(above, ()):
            if pid in found or pid == top or not _proven(above, pid, table):
                continue
            entry = recorded.get(pid)
            if entry is None:
                key = (pid, _text(table[pid][1]))
                if key in users or skip(pid, above, above in recorded):
                    continue
                entry = Entry(*key, "tool", (above, _text(table[above][1])), recorded=False)
            found[pid] = entry
            todo.append(pid)
    return found


def _is_starter(pid: int, install: str, table: dict) -> bool:
    """Windows: whether *pid* is a program in the venv of *install* that only starts the
    real interpreter and ends with it (``python.exe`` and ``localm.exe`` of a venv)."""
    if sys.platform != "win32":
        return False
    image = _win_image(pid)
    scripts = os.path.join(os.path.realpath(install), ".venv", "Scripts")
    return bool(image) and os.path.normcase(
        os.path.dirname(os.path.realpath(image))) == os.path.normcase(scripts)


def _starters(pid: int, install: str, table: dict) -> dict:
    """The starters above *pid* (:func:`_is_starter`)."""
    found: dict = {}
    while sys.platform == "win32":
        above = table[pid][0] if pid in table else 0
        if (above not in table or above in found or not _is_starter(above, install, table)
                or not _proven(above, pid, table)):
            break
        pid = above
        found[pid] = Entry(pid, _text(table[pid][1]), "starter",
                           _parent_of(pid, table[pid]), recorded=False)
    return found


def _sorted(entries: dict):
    return sorted(entries.values(), key=lambda e: (_number(e.start) or 0, e.pid))


def _all(install: str, table: dict, recorded: dict) -> dict:
    found = dict(recorded)
    children, skip = _children(table), _users_own(install, table)
    for pid in recorded:
        if pid in table:
            found.update(_below(pid, install, recorded, table, children, skip))
            for above, entry in _starters(pid, install, table).items():
                found.setdefault(above, entry)
    return found


def list(install=None):
    """Every running process of the install at *install* (this one by default), oldest
    first: those that have a file, the short tools below them, and on Windows the venv
    programs that started them. Raises OSError when the processes of this system cannot
    be listed."""
    install = os.fspath(install or root())
    if not os.path.isdir(os.path.join(install, FOLDER)):
        return []
    table = _table()
    return _sorted(_all(install, table, _recorded(install, table)))


def descendants(entry: Entry, install=None):
    """The running processes of localm that *entry* started and those they started,
    without *entry*."""
    install = os.fspath(install or root())
    table = _table()
    if _state(entry.pid, entry.start, table) is not True:
        return []
    recorded = _recorded(install, table)
    return _sorted(_below(entry.pid, install, recorded, table, _children(table),
                          _users_own(install, table)))


# --------------------------------------------------------------------------- #
#  Ending                                                                      #
# --------------------------------------------------------------------------- #

def _ancestors(pid: int, table: dict) -> set:
    """*pid* and every process above it, also one whose place there is not proven."""
    found = {pid}
    while pid in table and table[pid][0] in table and table[pid][0] not in found:
        above = table[pid][0]
        if _later((above, table[above][1]), (pid, table[pid][1])):
            break
        found.add(above)
        pid = above
    return found


def _end(entry: Entry, force: bool) -> None:
    """Ask the process of *entry* to end, or end it. Nothing else is ever reached: the
    start time is read through the same handle that ends it."""
    if sys.platform == "win32":
        import ctypes
        handle = _win_open(entry.pid, 0x1001)           # QUERY_LIMITED | TERMINATE
        if not handle:
            return
        try:
            row = _win_row(handle)
            if row and row != _UNKNOWN and _text(row[1]) == entry.start:
                k32 = _win_dll("kernel32")
                k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                k32.TerminateProcess(handle, 1)
        finally:
            _win_close(handle)
        return
    import signal
    sig = signal.SIGKILL if force else signal.SIGTERM
    fd = None
    try:
        if hasattr(os, "pidfd_open"):
            fd = os.pidfd_open(entry.pid)
        if _state(entry.pid, entry.start) is not True:
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(entry.pid, sig)
    except OSError:
        return
    finally:
        if fd is not None:
            os.close(fd)


def end_others(install=None, *, timeout: float = 20.0, spare=()):
    """End every process of the install except this one, every process above it (a
    localm command that waits for this one among them), what this one started, and
    *spare* (entries or pids). Watchdogs end first. A process that starts meanwhile is
    ended as well, unless its starter was ended in the same moment and it has no file:
    then nothing names it.

    Returns the entries that still run after *timeout* seconds: empty when every process
    of the install but the spared ones has ended, with :data:`UNSEEN` among them when
    processes may run that have no entry (an account that may run the install but not
    write to its record, another machine on the same folder, no record at all). Raises
    OSError when the processes of this system cannot be listed."""
    import time
    install = os.fspath(install or root())
    if not os.path.isdir(os.path.join(install, FOLDER)):
        return [UNSEEN]
    named = {s.pid if isinstance(s, Entry) else s for s in spare}
    for pid in tuple(named):
        named.update(_starters(pid, install, _table()))
    deadline = time.monotonic() + timeout
    asked: dict = {}
    me = os.getpid()
    while True:
        table = _table()
        recorded = _recorded(install, table)
        spared = named | _ancestors(me, table) | set(_below(
            me, install, recorded, table, _children(table), lambda *_: False))
        rest = [e for e in _sorted(_all(install, table, recorded)) if e.pid not in spared]
        now = time.monotonic()
        if not rest or now > deadline:
            try:
                unseen = _others_kept_out(install) or bool(elsewhere(install))
            except OSError:
                unseen = True
            return rest + ([UNSEEN] if unseen else [])
        for entry in sorted(rest, key=lambda e: "watchdog" not in e.purpose):
            first = asked.setdefault(entry.key, now)
            _end(entry, now - first >= _POSIX_GRACE_S)
        time.sleep(0.1)
