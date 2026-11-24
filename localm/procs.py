# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which processes of this install are running.

Every process that imports localm writes one small file into ``<root>/.localm-run``
as its first act (:func:`enter`, called by ``localm/__init__.py``), and a long-lived or
detached program that does not import localm is started through :func:`start`, which
writes one for it. A file names the process id, when that process started, what it is
and the process that started it; no command line, no environment, no content. The
process removes its own file when it ends. A file whose process has gone, or whose
process id now belongs to a process that started at another time, is deleted by the
next localm process that starts or ends and by the next reader.

A short tool has no file: it is a running child of a process that has one, and
:func:`list` finds it there. The parent of a process is only believed when it started
no later than its child.

There is no record for an install that cannot keep one: a pip install, the container
image, a folder this process may not write to, or a system whose process start times
cannot be read. :func:`current` is then None.

Standard library only, and no import of localm: ``localm/__init__.py`` imports this
module before anything else of localm, and it can be loaded by file path.
"""

import os
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

#: The host of every console window; it ends with the programs it shows.
_NOT_OURS = frozenset({"conhost.exe", "openconsole.exe"})
_POSIX_GRACE_S = 5.0


class Entry:
    """One running process. *start* is when it started, in a form that only this
    process has on this system; *parent* is ``(pid, start)`` or None; *recorded* is
    whether it has a file of its own."""

    __slots__ = ("pid", "start", "purpose", "parent", "recorded")

    def __init__(self, pid, start, purpose, parent=None, recorded=True) -> None:
        self.pid = pid
        self.start = start
        self.purpose = purpose
        self.parent = parent
        self.recorded = recorded

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
# A row is (parent pid, start, image name or ""); start is a number that grows with
# time: 100 ns units since 1601 (Windows), clock ticks since boot (Linux),
# microseconds since 1970 (macOS).
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
    k32 = _win_dll("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
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


def _win_process(pid: int):
    handle = _win_open(pid, 0x101000)       # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if handle:
        try:
            row = None if _win_exited(handle) else _win_row(handle)
        finally:
            _win_close(handle)
        if row != _UNKNOWN:
            return row
    return _win_table().get(pid)


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
    except ProcessLookupError:
        return True
    except OSError:
        pass
    return False


def _mac_lib():
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


def _mac_row(lib, pid: int):
    """``struct proc_bsdinfo`` of *pid*: None when it has exited or is a zombie."""
    import ctypes
    import struct
    buf = ctypes.create_string_buffer(136)
    if lib.proc_pidinfo(pid, 3, 0, buf, 136) != 136:     # PROC_PIDTBSDINFO
        return None if _posix_gone(pid) else _UNKNOWN
    if struct.unpack_from("=I", buf.raw, 4)[0] == 5:     # SZOMB
        return None
    sec, usec = struct.unpack_from("=QQ", buf.raw, 120)
    return (struct.unpack_from("=I", buf.raw, 16)[0], sec * 1000000 + usec, "")


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
        row = _mac_row(lib, pid) if pid > 0 else None
        if row and row != _UNKNOWN:
            rows[pid] = row
    return rows


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


def _process(pid: int):
    """The row of *pid*, None when no such process runs, or :data:`_UNKNOWN` when one
    runs that this process may not look at."""
    if sys.platform == "win32":
        return _win_process(pid)
    if sys.platform.startswith("linux"):
        return _linux_process(pid)
    if sys.platform == "darwin":
        return _mac_row(_mac_lib(), pid)
    raise OSError("process start times cannot be read on this system")


def _own_row():
    if sys.platform == "win32":
        import ctypes
        return _win_row(ctypes.c_void_p(-1))
    return _process(os.getpid())


# --------------------------------------------------------------------------- #
#  Start times and hosts as text                                               #
# --------------------------------------------------------------------------- #

_BOOT = None


def _boot() -> str:
    """Linux: the boot that start ticks count from. Empty elsewhere."""
    global _BOOT
    if _BOOT is None:
        _BOOT = ""
        if sys.platform.startswith("linux"):
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as fh:
                _BOOT = fh.read().strip().replace("-", "")
    return _BOOT


def _text(number: int) -> str:
    return f"{_boot()}.{number}" if _boot() else str(number)


def _number(start: str):
    """The number in *start*, or None when it counts from another boot."""
    boot, _, number = start.rpartition(".")
    return int(number) if boot == _boot() and number.isdigit() else None


def _mac_uuid() -> str:
    """This Mac's hardware UUID."""
    import ctypes
    raw, wait = (ctypes.c_ubyte * 16)(), (ctypes.c_long * 2)(1, 0)
    if ctypes.CDLL(None).gethostuuid(raw, wait) != 0:
        raise OSError("this machine has no id")
    return bytes(raw).hex()


def _host() -> str:
    """A short id of the process table this process is in, which the machine's name is
    not part of: one machine, and on Linux one pid namespace."""
    global _HOST
    if _HOST:
        return _HOST
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            text = str(winreg.QueryValueEx(key, "MachineGuid")[0])
    elif sys.platform.startswith("linux"):
        text = str(os.stat("/proc/self/ns/pid").st_ino)
        try:
            with open("/etc/machine-id", encoding="ascii") as fh:
                text += fh.read().strip()
        except (OSError, ValueError):
            text += _boot()
    else:
        text = _mac_uuid()
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8", "replace"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    _HOST = f"{value:016x}"
    return _HOST


# --------------------------------------------------------------------------- #
#  The files                                                                   #
# --------------------------------------------------------------------------- #

_CURRENT = None
_CURRENT_PATH = ""
_HOST = ""


def root() -> str:
    """The folder of the install this code is part of."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def current():
    """The :class:`Entry` of this process, or None when it has no file."""
    return _CURRENT if _CURRENT is not None and _CURRENT.pid == os.getpid() else None


def _name(pid: int, start: str) -> str:
    return f"{_host()}-{pid}-{start}.json"


def _parse(name: str):
    """``(host, pid, start)`` of a file name, or None."""
    parts = name[:-5].split("-") if name.endswith(".json") else []
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), parts[2]


def _write(folder: str, entry: Entry) -> str:
    path = os.path.join(folder, _name(*entry.key))
    parent = "null"
    if entry.parent:
        parent = '{"pid": %d, "start": "%s"}' % entry.parent
    text = '{"pid": %d, "start": "%s", "purpose": "%s", "parent": %s}\n' % (
        entry.pid, entry.start, entry.purpose, parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, text.encode("ascii"))
    finally:
        os.close(fd)
    return path


def _make_folder(install: str) -> bool:
    """Make the folder of the install at *install*, when it is one that keeps a
    record. A folder made by root belongs to the owner of the install."""
    if (os.path.basename(install).lower() in ("site-packages", "dist-packages")
            or not os.path.isfile(os.path.join(install, "pyproject.toml"))):
        return False
    folder = os.path.join(install, FOLDER)
    os.mkdir(folder, 0o755)
    if sys.platform != "win32" and os.geteuid() == 0:
        owner = os.stat(install)
        os.chown(folder, owner.st_uid, owner.st_gid)
    return True


def _parent_of(row):
    """``(pid, start)`` of the parent that *row* names, or None when it has gone or
    started after its child."""
    other = _process(row[0]) if row[0] else None
    if not other or other == _UNKNOWN or other[1] > row[1]:
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
    global _CURRENT, _CURRENT_PATH
    if os.environ.get("LOCALM_CONTAINER_IMAGE"):
        return
    try:
        row = _own_row()
        if not row or row == _UNKNOWN:
            return
        entry = Entry(os.getpid(), _text(row[1]), _purpose(sys.orig_argv), _parent_of(row))
        install = root()
        folder = os.path.join(install, FOLDER)
        try:
            path = _write(folder, entry)
        except FileNotFoundError:
            try:
                if not _make_folder(install):
                    return
            except FileExistsError:
                pass
            path = _write(folder, entry)
        _CURRENT, _CURRENT_PATH = entry, path
        import atexit
        atexit.unregister(leave)
        atexit.register(leave)
        _sweep(folder)
    except Exception:
        return


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _sweep(folder: str) -> None:
    """Delete the files of processes of this machine that have gone."""
    host = _host()
    for name in os.listdir(folder):
        parsed = _parse(name)
        if parsed and parsed[0] == host and _state(parsed[1], parsed[2]) is False:
            _unlink(os.path.join(folder, name))


def leave() -> None:
    """Remove the file of this process, which is about to end, and of those that have
    ended."""
    global _CURRENT
    if current() is None:
        return
    _CURRENT = None
    _unlink(_CURRENT_PATH)
    try:
        _sweep(os.path.dirname(_CURRENT_PATH))
    except Exception:
        return


def start(args, *, purpose: str, **kwargs):
    """``subprocess.Popen(args, **kwargs)`` for a long-lived or detached program that
    does not import localm, recorded as *purpose* (lower-case words)."""
    import subprocess
    if not (purpose.replace(" ", "").isalnum() and purpose.isascii()):
        raise ValueError(purpose)
    proc = subprocess.Popen(args, **kwargs)
    me = current()
    if me is not None:
        try:
            row = _process(proc.pid)
            if row and row != _UNKNOWN and row[0] == me.pid:
                folder = os.path.dirname(_CURRENT_PATH)
                _sweep(folder)
                _write(folder, Entry(proc.pid, _text(row[1]), purpose, me.key))
        except Exception:
            return proc
    return proc


# --------------------------------------------------------------------------- #
#  Reading                                                                     #
# --------------------------------------------------------------------------- #

def _state(pid: int, start: str, table=None):
    """True when the process *pid* that started at *start* runs, False when it does
    not, None when one with that pid runs whose start cannot be read."""
    number = _number(start)
    if number is None:
        return False
    row = table.get(pid) if table else None
    if not row or row[1] != number:
        row = _process(pid)
    if row == _UNKNOWN:
        return None
    return bool(row) and row[1] == number


def alive(entry: Entry) -> bool:
    """Whether the process of *entry* still runs. One that cannot be looked at (it
    belongs to another user) counts as running."""
    return _state(entry.pid, entry.start) is not False


def _recorded(install: str, table=None) -> dict:
    """{pid: Entry} of the running processes that have a file; deletes the files of
    processes of this machine that have gone."""
    folder = os.path.join(install, FOLDER)
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return {}
    if table is None:
        table = _table()
    import json
    host, found = _host(), {}
    for name in names:
        parsed = _parse(name)
        if parsed is None or parsed[0] != host:
            continue
        _, pid, start = parsed
        path = os.path.join(folder, name)
        if _state(pid, start, table) is False:
            _unlink(path)
            continue
        purpose, parent = "localm", None
        try:
            with open(path, encoding="ascii") as fh:
                data = json.load(fh)
            purpose = str(data["purpose"])
            if data["parent"]:
                parent = (int(data["parent"]["pid"]), str(data["parent"]["start"]))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError, KeyError):
            pass
        found[pid] = Entry(pid, start, purpose, parent)
    return found


def elsewhere(install=None) -> int:
    """How many files belong to another machine, or to another container, that uses
    the same folder. Whether those processes run cannot be told from here."""
    try:
        names = os.listdir(os.path.join(install or root(), FOLDER))
    except FileNotFoundError:
        return 0
    host = _host()
    return sum(1 for n in names if (_parse(n) or (host,))[0] != host)


def _children(table: dict) -> dict:
    out: dict = {}
    for pid, row in table.items():
        above = table.get(row[0])
        if above is not None and row[0] != pid and above[1] <= row[1]:
            out.setdefault(row[0], []).append(pid)
    return out


def _win_windows() -> set:
    """The pids that show a window of their own, console windows left out."""
    import ctypes
    if sys.platform != "win32":
        return set()
    user = _win_dll("user32")
    pids: set = set()
    name = ctypes.create_unicode_buffer(64)
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
            user.GetClassNameW(hwnd, name, 64)
            if name.value not in ("ConsoleWindowClass", "PseudoConsoleWindow"):
                user.GetWindowThreadProcessId(hwnd, ctypes.addressof(owner))
                pids.add(owner.value)
        return 1

    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    user.EnumWindows.argtypes = [proto, ctypes.c_void_p]
    user.EnumWindows(proto(see), None)
    return pids


def _users_own(top: int, table: dict):
    """A test for a program below the recorded process *top* that localm opened for the
    user (a web browser, an editor) and that is not a process of localm: it shows a
    window of its own (Windows), or *top* started it in a session of its own without
    recording it (elsewhere)."""
    if sys.platform == "win32":
        windows = _win_windows()
        return lambda pid: pid in windows or table[pid][2] in _NOT_OURS

    def own_session(pid: int) -> bool:
        try:
            return table[pid][0] == top and os.getsid(pid) != os.getsid(top)
        except OSError:
            return False
    return own_session


def _below(top: int, recorded: dict, table: dict, children: dict) -> dict:
    """{pid: Entry} of the running processes below *top*, without it."""
    found: dict = {}
    skip, todo = _users_own(top, table), [top]
    while todo:
        above = todo.pop()
        for pid in children.get(above, ()):
            if pid in found or pid == top:
                continue
            entry = recorded.get(pid)
            if entry is None:
                if skip(pid):
                    continue
                entry = Entry(pid, _text(table[pid][1]), "tool",
                              (above, _text(table[above][1])), recorded=False)
                found[pid] = entry
                todo.append(pid)
            else:
                found[pid] = entry
                found.update(_below(pid, recorded, table, children))
    return found


def descendants(entry: Entry, install=None):
    """The running processes that *entry* started and those they started, without
    *entry*."""
    table = _table()
    if _state(entry.pid, entry.start, table) is not True:
        return []
    recorded = _recorded(install or root(), table)
    return _sorted(_below(entry.pid, recorded, table, _children(table)))


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


def _starters(pid: int, install: str, table: dict) -> dict:
    """Windows: the programs in the venv of *install* that only start *pid* and end
    with it (``python.exe`` and ``localm.exe`` of a venv start the real interpreter)."""
    found: dict = {}
    if sys.platform != "win32":
        return found
    scripts = os.path.normcase(os.path.join(os.path.realpath(install), ".venv", "Scripts"))
    while True:
        row = table.get(pid)
        above = table.get(row[0]) if row else None
        if row is None or above is None or above[1] > row[1] or row[0] in found:
            return found
        image = os.path.normcase(os.path.realpath(_win_image(row[0]) or "."))
        if os.path.dirname(image) != scripts:
            return found
        pid = row[0]
        parent = _parent_of(above)
        found[pid] = Entry(pid, _text(above[1]), "starter", parent, recorded=False)


def _sorted(entries: dict):
    return sorted(entries.values(), key=lambda e: (_number(e.start) or 0, e.pid))


def list(install=None):
    """Every running process of the install at *install* (this one by default), oldest
    first: those that have a file, the short tools below them, and on Windows the venv
    programs that started them. Raises OSError when the processes of this system cannot
    be listed."""
    install = os.fspath(install or root())
    if not os.path.isdir(os.path.join(install, FOLDER)):
        return []
    table = _table()
    recorded = _recorded(install, table)
    found = dict(recorded)
    children = _children(table)
    for pid in recorded:
        if pid in table:
            found.update(_below(pid, recorded, table, children))
            for above, entry in _starters(pid, install, table).items():
                found.setdefault(above, entry)
    return _sorted(found)


# --------------------------------------------------------------------------- #
#  Ending                                                                      #
# --------------------------------------------------------------------------- #

def _spared(install: str, table: dict, recorded: dict) -> set:
    """The pids that are this process, what it started, and the programs waiting for
    it that are not localm."""
    me = os.getpid()
    spared = {me} | set(_below(me, recorded, table, _children(table)))
    pid = me
    while True:
        row = table.get(pid)
        above = table.get(row[0]) if row else None
        if row is None or above is None or above[1] > row[1] or row[0] in spared:
            return spared
        pid = row[0]
        if pid in recorded:
            return spared
        spared.add(pid)


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


def end_others(install=None, *, timeout: float = 20.0):
    """End every process of the install except this one, what it started, and the
    programs that only wait for it. Watchdogs end first, then starters of others before
    what they started; what starts meanwhile is ended as well. Returns the entries that
    still run after *timeout* seconds, empty when none does. Raises OSError when the
    processes of this system cannot be listed."""
    import time
    install = os.fspath(install or root())
    deadline = time.monotonic() + timeout
    asked: dict = {}
    while True:
        table = _table()
        spared = _spared(install, table, _recorded(install, table))
        rest = [e for e in list(install) if e.pid not in spared]
        now = time.monotonic()
        if not rest or now > deadline:
            return rest
        for entry in sorted(rest, key=lambda e: "watchdog" not in e.purpose):
            first = asked.setdefault(entry.key, now)
            _end(entry, now - first >= _POSIX_GRACE_S)
        time.sleep(0.1)
