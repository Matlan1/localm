# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which processes of this install are running.

Every process that imports localm writes one small file into ``<root>/.localm-run``
as its first act (:func:`enter`, called by ``localm/__init__.py``), and a long-lived or
detached program that does not import localm is started through :func:`start`, which
writes one for it. A file names the process id, when that process started, what it is,
the process that started it and, when it opened programs for the user, those; no
command line, no environment, no content.

A file is believed only when

- its name carries the id of this machine and this folder: in a copied install, or on
  another machine or in another container that uses the folder, it is ignored;
- it is a regular file and, except on Windows, belongs to the user of this process
  (for root: to root or to the owner of the install); and
- what it says is well formed and is what its name says.

Every other file is left alone: not listed, not deleted. A process never removes its
own file, so it is listed until it has really ended. The believed file of a process
that has ended is deleted by the next localm process that starts and by the next
reader.

A short tool has no file: it is a running child of a process that has one, and
:func:`list` finds it there. A program localm opens for the user (a web browser, a
viewer) is opened through :func:`open_for_user` and is never listed and never ended.
Neither is a program below a localm process of which it cannot be told that it is a
tool: see :func:`_users_own`.

Everything that imports localm from this folder is a process of this install, also a
test run, a notebook or a script of the user, and :func:`end_others` ends it.

There is no record for an install that cannot keep one: a pip install, the container
image, a folder this process may not write to, a ``.localm-run`` that is a link or
another user's, or a system whose process start times cannot be read. :func:`current`
is then None and :func:`problem` says why.

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
        started_by = f" started by {self.parent[0]}" if self.parent else ""
        return f"<{self.purpose} pid {self.pid}{started_by}>"


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
    return _win_table().get(pid) if table is None else table.get(pid, _UNKNOWN)


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


# --------------------------------------------------------------------------- #
#  Start times, machine and folder as text                                     #
# --------------------------------------------------------------------------- #

_BOOT = None


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
    return text.isascii() and text.isdigit() and len(text) < 24 and str(int(text)) == text


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


def _hex(text: str, length: int = 16) -> bool:
    return len(text) == length and not set(text) - set("0123456789abcdef")


def _here(install: str) -> str:
    """The id of the folder *install* on this machine. A renamed folder keeps it, a
    copy has another one."""
    at, value = os.stat(install), 0xCBF29CE484222325
    for byte in f"{_machine()} {at.st_dev} {at.st_ino}".encode("utf-8", "replace"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


# --------------------------------------------------------------------------- #
#  The files                                                                   #
# --------------------------------------------------------------------------- #
# Text lines: the name of the file without ".proc", the purpose, "pid-start" of the
# parent or nothing, then "pid-start" of each program opened for the user.

_CURRENT = None
_PROBLEM = "localm has not been imported"
_PLACE = ("", "")           # install and id of the record this process writes to


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


def _is_link(at) -> bool:
    """Whether the lstat result *at* is of a symbolic link or a junction."""
    return stat.S_ISLNK(at.st_mode) or bool(getattr(at, "st_reparse_tag", 0) & 0x20000000)


def _folder(install: str) -> tuple:
    """The folder of the record and the users whose files in it are believed (None: not
    looked at), or ``("", None)`` when there is no record. Raises OSError when what is
    there is not used."""
    folder = os.path.join(install, FOLDER)
    try:
        at = os.lstat(folder)
    except FileNotFoundError:
        return "", None
    if _is_link(at) or not stat.S_ISDIR(at.st_mode):
        raise OSError(f"{FOLDER} is a link or not a folder")
    if sys.platform == "win32":
        return folder, None
    users = {os.geteuid()} if os.geteuid() else {0, os.stat(install).st_uid}
    if at.st_uid not in users:
        raise OSError(f"{FOLDER} belongs to another user")
    return folder, users


def _make_folder(install: str) -> None:
    if (os.path.basename(install).lower() in ("site-packages", "dist-packages")
            or not os.path.isfile(os.path.join(install, "pyproject.toml"))):
        raise OSError("this is not an install that updates itself")
    folder = os.path.join(install, FOLDER)
    try:
        os.mkdir(folder, 0o777 if sys.platform == "win32" else 0o700)
    except FileExistsError:
        return
    if sys.platform != "win32" and os.geteuid() == 0:
        os.chown(folder, os.stat(install).st_uid, -1, follow_symlinks=False)


def _key(text: str):
    """``(pid, start)`` of "pid-start", or None."""
    pid, _, started = text.partition("-")
    boot, dot, number = started.rpartition(".")
    if not _digits(pid) or not 0 < int(pid) < 1 << 32 or not _digits(number):
        return None
    return (int(pid), started) if not dot or _hex(boot, 32) else None


def _said(here: str, name: str):
    """``(pid, start)`` of the file name of an entry of this machine and folder, or None."""
    if not name.startswith(here + "-") or not name.endswith(".proc"):
        return None
    return _key(name[len(here) + 1:-5])


def _name(here: str, pid: int, started: str) -> str:
    """The name of the file of an entry: hex, digits, hex and digits, whatever is given.
    Raises ValueError for anything else than the parts of a name."""
    boot, _, number = started.rpartition(".")
    when = f"{int(number)}"
    if boot:
        when = f"{int(boot, 16):032x}.{when}"
    name = f"{int(here, 16):016x}-{int(pid)}-{when}.proc"
    if type(pid) is not int or not _hex(here) or _said(here, name) != (pid, started):
        raise ValueError("not the parts of a name")
    return name


def _content(name: str, entry: Entry) -> bytes:
    keys = [entry.parent, *sorted(entry.users)]
    lines = [name[:-5], entry.purpose, *("%d-%s" % k if k else "" for k in keys)]
    return ("\n".join(lines) + "\n").encode("ascii")


def _patiently(call, *args):
    """``call(*args)``, tried again for a moment while Windows says that another
    program has the file."""
    for _ in range(20 if sys.platform == "win32" else 0):
        try:
            return call(*args)
        except PermissionError:
            import time
            time.sleep(0.01)
    return call(*args)


def _write(place, entry: Entry, *, replace: bool = True) -> None:
    """Write the file of *entry* whole, or not at all. Without *replace* a file that is
    there already stays."""
    install, here = place
    folder = _folder(install)[0]
    if not folder:
        raise FileNotFoundError(FOLDER)
    name = _name(here, entry.pid, entry.start)
    path = os.path.join(folder, name)
    if not replace and os.path.lexists(path):
        return
    new = f"{path[:-5]}.{os.urandom(4).hex()}.new"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(new, flags, 0o600)
    try:
        try:
            os.write(fd, _content(name, entry))
        finally:
            os.close(fd)
        _patiently(os.replace, new, path)
    except BaseException:
        _unlink(new)
        raise


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
        if not _folder(install)[0]:
            _make_folder(install)
        place = (install, _here(install))
        _write(place, entry)
        _CURRENT, _PLACE, _PROBLEM = entry, place, ""
        _sweep(*place)
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
                _write(_PLACE, Entry(proc.pid, _text(row[1]), purpose, me.key), replace=False)
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
    belongs to another account) counts as running."""
    return _state(entry.pid, entry.start) is not False


def _believed(folder: str, users, name: str, said) -> "Entry | None":
    """The entry in the file *name*, which names the process *said*, or None when it is
    not believed."""
    path = os.path.join(folder, name)
    flags = os.O_RDONLY | sum(getattr(os, f, 0) for f in (
        "O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"))
    try:
        at = os.lstat(path)
        if _is_link(at) or not stat.S_ISREG(at.st_mode):
            return None
        fd = _patiently(os.open, path, flags)
        try:
            now = os.fstat(fd)
            if (not stat.S_ISREG(now.st_mode) or now.st_ino != at.st_ino
                    or now.st_size > _MAX_BYTES or (users and now.st_uid not in users)):
                return None
            lines = os.read(fd, _MAX_BYTES).decode("ascii").split("\n")
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None
    if len(lines) < 4 or lines.pop() or lines[0] != name[:-5] or not lines[1].isprintable():
        return None
    parent, opened = _key(lines[2]), [_key(k) for k in lines[3:]]
    if None in opened or (lines[2] and not parent):
        return None
    return Entry(*said, lines[1], parent, users=opened)


def _sweep(install: str, here: str) -> None:
    """Delete the files of ended processes among :data:`_SWEEP` of them."""
    folder, users = _folder(install)
    names = [n for n in os.listdir(folder) if n.startswith(here)]
    for i in range(min(_SWEEP, len(names))):
        name = names[(os.getpid() // 4 + i) % len(names)]
        said = _said(here, name)
        if said and _state(*said, {}) is False and _believed(folder, users, name, said):
            _unlink(os.path.join(folder, name))


def _recorded(install: str, table: dict) -> dict:
    """{pid: Entry} of the running processes that have a file; deletes the files of
    those that have ended. Raises OSError when the record cannot be read."""
    folder, users = _folder(install)
    if not folder:
        return {}
    here, found = _here(install), {}
    for name in os.listdir(folder):
        said = _said(here, name)
        if said is None:
            continue
        running = _state(*said, table)
        entry = _believed(folder, users, name, said)
        if entry is None:
            continue
        if running is False:
            _unlink(os.path.join(folder, name))
        else:
            found[entry.pid] = entry
    return found


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
    return os.path.dirname(folder) != folder and path.startswith(folder + os.sep)


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
    programs that started them. The caller and what :func:`end_others` spares are among
    them: one that needs the files of the install free looks here for what still runs.
    Empty for an install without a record (:func:`current` is None there). Raises
    OSError when the processes of this system or the record cannot be read."""
    install = os.fspath(install or root())
    if not _folder(install)[0]:
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

    Returns the entries that still run after *timeout* seconds of those it tried to end.
    Empty does not say that the files of the install are free: a spared process may have
    some open, and :func:`list` names it. Raises OSError when the processes of this
    system or the record cannot be read."""
    import time
    install = os.fspath(install or root())
    if not _folder(install)[0]:
        return []
    named = {s.pid if isinstance(s, Entry) else s for s in spare}
    deadline = time.monotonic() + timeout
    asked: dict = {}
    me = os.getpid()
    while True:
        table = _table()
        recorded = _recorded(install, table)
        spared = named | _ancestors(me, table) | set(_below(
            me, install, recorded, table, _children(table), lambda *_: False))
        for pid in named:
            spared.update(_starters(pid, install, table))
        rest = [e for e in _sorted(_all(install, table, recorded)) if e.pid not in spared]
        now = time.monotonic()
        for entry in sorted(rest, key=lambda e: "watchdog" not in e.purpose):
            first = asked.setdefault(entry.key, now)
            _end(entry, now - first >= _POSIX_GRACE_S)
        if not rest:
            return []
        if now > deadline:
            return [e for e in rest if alive(e)]
        time.sleep(0.1)
