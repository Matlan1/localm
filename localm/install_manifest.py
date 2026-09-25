# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install provenance ledger - what setup created and where - and the uninstaller
that removes exactly that.

``record()`` merges what a setup step created into ``.localm-install.json`` at
the root of the clone. Setup calls it as it goes (after the environment is
created, when the data folder is chosen, and at the end), so an install that
stops half way still has a record of what it made. Paths are stored absolute.

``prepare_data()`` creates the data folder and records who owns what in it: a
folder setup created (or the clone's own ``home``) belongs to localm entirely;
in a folder that already existed, only localm's own top-level entries
(``DATA_ENTRIES``) that were not already there count as localm's.

``uninstall()`` plans and executes the removal. It removes only recorded items
and localm's own fixed in-clone locations (``.venv`` with its marker,
``.python``, ``.cache``, ``.uv``, ``home``, the runtime lib folder), classifies
anything it is not sure of as a warning (removed only with ``force``), refuses
dangerous targets outright (root, a drive root, $HOME, the clone, an ancestor of
it, a symlink), and never touches a path it has no record or fixed rule for.
The user's saved data (the data folder) is kept unless ``purge_data``.

Removal of the Python runtime the uninstaller itself runs on is DEFERRED: with
``defer_runtime`` (setup scripts and the graphical installer) the in-clone
runtime folders are never deleted in-process; their names are written to
``.localm-uninstall-pending`` and the calling shell deletes them after this
process exits, then deletes the pending file and the manifest.

The manifest is deleted only when everything recorded is gone; after a failure
it stays, so running uninstall again retries.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MANIFEST_NAME = ".localm-install.json"
PENDING_NAME = ".localm-uninstall-pending"
DATA_MARKER = ".localm-data"
HOME_CFG_NAME = "localm-home.cfg"
PORTABLE_HOME = "home"
SCHEMA_VERSION = 3

# Exit codes of `python -m localm.install_manifest uninstall`.
EXIT_OK = 0
EXIT_FAILED = 1          # something could not be removed; nothing deferred
EXIT_PARTIAL = 2         # finished, but an item you asked to remove was refused
EXIT_RUNNING = 3         # LocaLM is running from this folder; nothing touched
EXIT_ABORTED = 4         # the record is newer than this uninstaller

# In-clone runtime folders, by manifest key. Only these names are ever written
# to the pending file, and the shells accept only these names from it.
RUNTIME_DIRS: Dict[str, str] = {"python_dir": ".python", "cache_dir": ".cache",
                                "uv_dir": ".uv"}
DEFERRABLE = (".venv",) + tuple(RUNTIME_DIRS.values())

_LIB_SENTINELS = (".gitignore", ".gitkeep")
_DEFAULT_LIB = ("runtime", "localm_llama_runtime", "lib")
_SHORTCUT_NAMES = ("localm.lnk", "localm.desktop")
_SHIM_NAMES = ("localm", "localm.cmd")
_LAUNCHER_DESKTOP = "LocaLM.desktop"

# The top-level names localm itself creates inside its data folder. In a data
# folder that already existed before setup, only these (and the marker) are
# removed by "delete saved data". Kept in sync with the package by
# test_data_entries_cover_every_data_dir_child.
DATA_FILES = frozenset({
    "config.json", "registry.json", "instance_id.txt", "auth.key", "auth.json",
    "auth.kdf.json", "sessions.json", "model_source_credentials.json",
    "model_meta.json", "coder-projects.json", "prompts.json", "launcher.json",
    "chat-memory.md",
})
DATA_DIRS = frozenset({
    "models", "cache", "logs", "run", "sessions", "bug-reports", "tls", "updates",
    "plugins", "rag", "memory", "chats", "coder", "checkpoints", "jobs",
    "activity", "uploads", "share_inbox", "gui_images", "gui_video", "gui_music",
    "gallery_index", "workflows", "mcp-images", "comfyui", "comfyui-models",
    "skills",
})
DATA_ENTRIES = frozenset({DATA_MARKER}) | DATA_FILES | DATA_DIRS
_DATA_PATTERNS = ("comfy-launch-*.log",)
# LocaLM entries whose names other programs use too. In a data folder with no
# record of what was in it before LocaLM ("legacy"), these are kept.
_LEGACY_KEPT = frozenset({
    "models", "cache", "logs", "plugins", "workflows", "skills", "memory",
    "comfyui", "run", "jobs", "checkpoints", "uploads", "updates", "tls",
    "config.json", "registry.json",
})

_ALL_KEYS = ("venv", "lib_dir", "lib_entries", "home_cfg", "data_dir",
             "data_created", "data_preexisting", "data_parents_created",
             "previous_data_dirs",
             "shortcut", "files", "runtime_contained", "python_dir", "cache_dir",
             "uv_dir", "uv_shared_installed", "path_dir", "command_shim",
             "path_modified")


def is_data_entry(name: str) -> bool:
    """Whether *name* is a top-level entry localm creates in its data folder:
    one of DATA_ENTRIES, a ``comfy-launch-*.log``, or the ``.lock`` / ``.bak`` /
    ``.<suffix>.tmp`` companion of one of localm's own files."""
    import fnmatch
    if name in DATA_ENTRIES:
        return True
    if any(fnmatch.fnmatchcase(name, p) for p in _DATA_PATTERNS):
        return True
    for f in DATA_FILES:
        if name in (f + ".lock", f + ".bak") or (
                name.startswith(f + ".") and name.endswith(".tmp")):
            return True
    return False


def manifest_path(root) -> Path:
    return Path(root) / MANIFEST_NAME


def pending_path(root) -> Path:
    return Path(root) / PENDING_NAME


def _abs(p) -> str:
    return str(Path(p).expanduser().resolve()) if p else ""


def _plain_abs(p) -> str:
    """Absolute without resolving links, so a symlinked folder stays visible as
    one to the guards."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(p)))) if p else ""


def _same(a, b) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(a))) == \
            os.path.normcase(os.path.abspath(str(b)))
    except (TypeError, ValueError):
        return False


def _lib_entries(lib_dir: Optional[Path]) -> list:
    """Snapshot of the top-level names in *lib_dir* (files and folders) other
    than the tracked git sentinels: everything setup-llama provisioned there."""
    out: list = []
    if not lib_dir:
        return out
    try:
        for f in sorted(Path(lib_dir).iterdir()):
            if f.name not in _LIB_SENTINELS:
                out.append(f.name)
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- #
#  Reading and writing the record                                              #
# --------------------------------------------------------------------------- #

def load(root) -> Optional[dict]:
    try:
        data = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _upgrade(m: dict, root: Path) -> dict:
    """A manifest older than schema 3, in schema-3 terms.

    Schema 1 and 2 listed only binary suffixes (``binaries``), and the console
    setups recorded a custom data folder as created even when it already
    existed, so that claim is dropped: such a folder is treated as one that
    already existed, with its earlier contents unknown."""
    out = dict(m)
    if (out.get("schema") or 0) >= 3:
        return out
    if "lib_entries" not in out:
        out["lib_entries"] = list(out.get("binaries") or [])
    data_dir = out.get("data_dir") or ""
    if data_dir and not _same(data_dir, root / PORTABLE_HOME):
        out["data_created"] = False
        out["data_preexisting"] = None
    return out


def _write(root: Path, data: dict) -> Path:
    p = manifest_path(root)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def record(root, *, venv="", lib_dir="", home_cfg="", data_dir="",
           data_created=False, shortcut="", stamp="",
           runtime_contained=False, python_dir="", cache_dir="", uv_dir="",
           path_dir="", command_shim="", path_modified=False,
           files: Iterable[str] = (), uv_shared_installed=False,
           data_preexisting: Optional[List[str]] = None,
           data_parents_created: Iterable[str] = ()) -> Path:
    """Merge what a setup step created into the manifest under *root*.

    Empty values leave the recorded ones alone, flags only ever turn on, and
    ``files`` accumulate, so a later step (or a later setup run) never erases
    what an earlier one created. Passing *lib_dir* re-snapshots its contents.
    Passing *data_dir* sets the data-folder fields; for the folder already
    recorded, the parent folders setup created are kept, and a different
    folder moves the recorded one to ``previous_data_dirs``. *command_shim* is
    stored as given, not resolved, so a symlink stays the symlink.

    Raises ValueError when the existing manifest is newer than this code."""
    root = Path(root).resolve()
    old = load(root) or {}
    if (old.get("schema") or 0) > SCHEMA_VERSION:
        raise ValueError(f"install record schema {old.get('schema')!r} is newer "
                         "than this setup understands")
    data = _upgrade(old, root) if old else {}
    data.pop("binaries", None)
    data["schema"] = SCHEMA_VERSION
    if stamp:
        data["stamp"] = stamp

    for key, value in (("venv", venv), ("home_cfg", home_cfg),
                       ("shortcut", shortcut), ("python_dir", python_dir),
                       ("cache_dir", cache_dir), ("uv_dir", uv_dir),
                       ("path_dir", path_dir)):
        if value:
            data[key] = _abs(value)
    if command_shim:
        data["command_shim"] = _plain_abs(command_shim)
    for key, value in (("runtime_contained", runtime_contained),
                       ("path_modified", path_modified),
                       ("uv_shared_installed", uv_shared_installed)):
        data[key] = bool(data.get(key)) or bool(value)
    if lib_dir:
        data["lib_dir"] = _abs(lib_dir)
        data["lib_entries"] = _lib_entries(Path(lib_dir))
    if data_dir:
        new_dir = _plain_abs(data_dir)
        old_dir = data.get("data_dir") or ""
        parents = [_plain_abs(p) for p in data_parents_created]
        previous = [p for p in data.get("previous_data_dirs") or [] if p]
        if old_dir and _same(old_dir, new_dir):
            parents = list(dict.fromkeys((data.get("data_parents_created") or []) + parents))
        elif old_dir:
            if not any(_same(old_dir, p) for p in previous):
                previous.append(old_dir)
        data["previous_data_dirs"] = [p for p in previous if not _same(p, new_dir)]
        data["data_dir"] = new_dir
        data["data_created"] = bool(data_created)
        data["data_preexisting"] = (None if data_preexisting is None
                                    else sorted(set(data_preexisting)))
        data["data_parents_created"] = parents
    have = list(data.get("files") or [])
    for f in files:
        if f and _abs(f) not in have:
            have.append(_abs(f))
    data["files"] = have
    for key in _ALL_KEYS:
        data.setdefault(key, [] if key in ("lib_entries", "files", "data_parents_created",
                                           "previous_data_dirs") else
                        (False if key in ("data_created", "runtime_contained",
                                          "uv_shared_installed", "path_modified")
                         else (None if key == "data_preexisting" else "")))
    return _write(root, data)


def refresh_lib(lib_dir) -> Optional[Path]:
    """Re-snapshot *lib_dir* (``<clone>/runtime/localm_llama_runtime/lib``) into
    the install record of the clone that holds it. Returns the manifest path,
    or None when that clone has no record, the record is newer than this code,
    or it names a different runtime folder."""
    lib = Path(lib_dir).resolve()
    if len(lib.parents) < 3:
        return None
    root = lib.parents[2]
    m = load(root)
    if m is None or (m.get("schema") or 0) > SCHEMA_VERSION:
        return None
    if m.get("lib_dir") and not _same(m["lib_dir"], lib):
        return None
    return record(root, lib_dir=str(lib))


def read_marker(folder) -> Optional[dict]:
    """The ``.localm-data`` marker of *folder*: ``created`` (setup created the
    folder), ``preexisting`` (what was in it before LocaLM first used it) and
    ``installs`` (the install folders that use it). None when there is no
    marker, it cannot be read, or it carries no ``preexisting`` list."""
    try:
        data = json.loads((Path(folder) / DATA_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("preexisting"), list):
        return None
    return {"created": bool(data.get("created")),
            "preexisting": sorted({str(n) for n in data["preexisting"]}),
            "installs": [str(p) for p in data.get("installs") or [] if isinstance(p, str)]}


def _write_marker(folder: Path, marker: dict) -> None:
    body = {"created_by": "localm setup", "created": bool(marker["created"]),
            "preexisting": sorted(set(marker["preexisting"])),
            "installs": list(marker["installs"])}
    tmp = folder / (DATA_MARKER + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, folder / DATA_MARKER)


def _live_install(path: str) -> bool:
    p = Path(path)
    return (p / MANIFEST_NAME).is_file() or (p / ".venv").is_dir()


def current_data_dir(root) -> str:
    """The data folder the clone at *root* is set up to use: the folder named
    in ``localm-home.cfg``, else ``<root>/home``. "" when that folder does not
    exist or the setting cannot be read."""
    root = Path(root).resolve()
    cfg = root / HOME_CFG_NAME
    if cfg.is_file():
        try:
            line = _read_cfg_line(cfg)
        except (OSError, ValueError, LookupError):
            return ""
        path = os.path.expanduser(line) if line else ""
        return _plain_abs(path) if path and os.path.isabs(path) and os.path.isdir(path) else ""
    home = root / PORTABLE_HOME
    return str(home) if home.is_dir() else ""


def prepare_data(root, *, data_dir="", portable=False) -> Path:
    """Create the data folder, point the clone at it, and record ownership.

    *portable* uses ``<root>/home`` and removes ``localm-home.cfg``; otherwise
    *data_dir* must be a full path, and ``localm-home.cfg`` is written with it
    (UTF-8). The folder's ``.localm-data`` marker records whether setup created
    it, what was in it before LocaLM first used it, and which install folders
    use it; a marker that already carries that record is kept as it is, so
    preparing the same folder again (a repair, another install) never counts
    anything that was there before as LocaLM's. A folder that exists without
    such a record counts everything in it as pre-existing.

    Returns the data folder. Raises ValueError for a relative path, a path that
    is a file, or a folder uninstall could never safely delete (a drive root,
    the home folder, this install folder or one that contains it), OSError when
    the folder cannot be created or the marker written."""
    root = Path(root).resolve()
    cfg = root / HOME_CFG_NAME
    if portable:
        target = root / PORTABLE_HOME
    else:
        text = str(data_dir or "").strip().strip('"')
        if not text:
            raise ValueError("no data folder given")
        target = Path(os.path.expanduser(text))
        if not target.is_absolute():
            raise ValueError(f"{text} is not a full path (it must start at a "
                             "drive or /, for example D:\\LocaLM-data)")
        target = Path(_plain_abs(target))
        reason = _unsafe_data_dir(str(target), root, allow_link=True)
        if reason:
            raise ValueError(f"{target} cannot be the data folder ({reason}); "
                             "choose a folder of its own")
    existed = target.exists() or target.is_symlink()
    if existed and not target.is_dir():
        raise ValueError(f"{target} is a file, not a folder")

    marker = read_marker(target) if existed else None
    parents: list = []
    if marker is None:
        if existed:
            names = sorted(e.name for e in os.scandir(target))
            marker = {"created": False, "installs": [],
                      "preexisting": [n for n in names
                                      if n not in (DATA_MARKER, DATA_MARKER + ".tmp")]}
        else:
            p = target.parent
            while not p.exists() and p != p.parent:
                parents.append(str(p))
                p = p.parent
            parents.reverse()
            marker = {"created": True, "installs": [], "preexisting": []}
    if not any(_same(p, root) for p in marker["installs"]):
        marker["installs"].append(str(root))
    target.mkdir(parents=True, exist_ok=True)
    _write_marker(target, marker)
    if portable:
        if cfg.exists():
            cfg.unlink()
    else:
        cfg.write_text(str(target) + "\n", encoding="utf-8")
    record(root, data_dir=str(target), data_created=marker["created"],
           data_preexisting=marker["preexisting"], data_parents_created=parents,
           home_cfg="" if portable else str(cfg))
    return target


# --------------------------------------------------------------------------- #
#  Guards                                                                      #
# --------------------------------------------------------------------------- #

def _unsafe_data_dir(path: str, repo: Path, *, allow_link: bool = False) -> Optional[str]:
    """A reason string if *path* is too dangerous for ``rm -rf``, else None.
    Refuses empty/relative paths, symlinks (unless *allow_link*), the
    filesystem or drive root, $HOME, the repo, and any ancestor of the repo."""
    if not path:
        return "empty path"
    p = Path(path)
    if not p.is_absolute():
        return "not an absolute path"
    if not allow_link and (p.is_symlink() or _is_junction(p)):
        return "is a symlink"
    try:
        rp = p.resolve()
    except OSError:
        return "cannot resolve path"
    if rp == rp.parent:                       # the filesystem root or a Windows drive root
        return "filesystem/drive root"
    try:
        if rp == Path.home().resolve():
            return "user home directory"
    except Exception:
        return "cannot resolve home directory"
    repo_r = repo.resolve()
    if rp == repo_r:
        return "repository root"
    if repo_r.is_relative_to(rp):             # rp is an ancestor of the repo
        return "an ancestor of the install directory"
    return None


def _is_junction(p: Path) -> bool:
    fn = getattr(os.path, "isjunction", None)
    if fn is not None:
        try:
            return bool(fn(p))
        except OSError:
            return False
    if sys.platform != "win32":
        return False
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return getattr(st, "st_reparse_tag", 0) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _is_link(p: Path) -> bool:
    return p.is_symlink() or _is_junction(p)


def _inside(child, parent) -> bool:
    try:
        c = os.path.normcase(os.path.abspath(str(child)))
        par = os.path.normcase(os.path.abspath(str(parent)))
    except (TypeError, ValueError):
        return False
    return c == par or c.startswith(par.rstrip("\\/") + os.sep)


def _is_our_venv(venv: Path) -> bool:
    return ((venv / ".localm-venv").is_file()
            or (venv / "Scripts" / "localm.exe").is_file()
            or (venv / "bin" / "localm").exists())


# --------------------------------------------------------------------------- #
#  Removing things                                                             #
# --------------------------------------------------------------------------- #

def _rmtree(path: Path) -> None:
    """``shutil.rmtree`` that also removes read-only files (a git checkout's
    object files on Windows). Links inside are removed, never followed."""
    def _retry(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry)
    else:  # pragma: no cover - older interpreters
        shutil.rmtree(path, onerror=lambda f, p, e: _retry(f, p, e[1]))


def _remove_entry(p: Path) -> None:
    """Remove a file, a link (never its target) or a folder."""
    if _is_link(p):
        if p.is_dir() and _is_junction(p):
            os.rmdir(p)
        else:
            p.unlink()
    elif p.is_dir():
        _rmtree(p)
    else:
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            p.unlink()


def _tree_size(path: Path, max_entries: int = 500_000) -> Tuple[int, int, bool]:
    """(bytes, files, complete) for *path*, without following links."""
    total = files = seen = 0
    stack = [str(path)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                seen += 1
                if seen > max_entries:
                    return total, files, False
                try:
                    if e.is_dir(follow_symlinks=False):
                        if not _is_junction(Path(e.path)):
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                        files += 1
                except OSError:
                    continue
    return total, files, True


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover


# --------------------------------------------------------------------------- #
#  Running processes                                                           #
# --------------------------------------------------------------------------- #

def _win_processes() -> Optional[list]:
    """[(pid, ppid, exe or None, start)] from a toolhelp snapshot."""
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.restype = wintypes.BOOL
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    k32.Process32NextW.restype = wintypes.BOOL
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.restype = wintypes.BOOL
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    invalid = ctypes.c_void_p(-1).value
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    if not snap or snap == invalid:
        return None
    rows = []
    try:
        pe = PE()
        pe.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            rows.append((int(pe.th32ProcessID), int(pe.th32ParentProcessID)))
            ok = k32.Process32NextW(snap, ctypes.byref(pe))
    finally:
        k32.CloseHandle(snap)

    out = []
    for pid, ppid in rows:
        exe = None
        start = None
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            try:
                buf = ctypes.create_unicode_buffer(32768)
                size = wintypes.DWORD(len(buf))
                if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    exe = buf.value
                times = [wintypes.FILETIME() for _ in range(4)]
                if k32.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
                    start = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            finally:
                k32.CloseHandle(h)
        out.append((pid, ppid, exe, start))
    return out


def _proc_processes() -> Optional[list]:
    """[(pid, ppid, exe or None, start, argv0 or None)] from /proc (Linux).
    argv0 is the path the program was started by; for a venv's symlinked
    ``bin/python`` it is the venv path while exe is the base interpreter."""
    out = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                raw = fh.read().decode("utf-8", "replace")
            fields = raw[raw.rindex(")") + 2:].split()
            ppid, start = int(fields[1]), int(fields[19])
        except (OSError, ValueError, IndexError):
            continue
        if fields[0] in ("Z", "X"):                 # exited, not yet reaped
            continue
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            exe = None
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv0 = fh.read().split(b"\0", 1)[0].decode("utf-8", "replace") or None
        except OSError:
            argv0 = None
        out.append((pid, ppid, exe, start, argv0))
    return out


def _ps_processes() -> Optional[list]:
    """[(pid, ppid, exe or None, None)] from ps (macOS and other POSIX)."""
    try:
        res = subprocess.run(["ps", "-axo", "pid=,ppid=,comm="],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    out = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        exe = parts[2] if len(parts) > 2 and os.path.isabs(parts[2]) else None
        out.append((int(parts[0]), int(parts[1]), exe, None))
    return out


def list_processes() -> Optional[list]:
    """[(pid, ppid, exe or None, start or None[, argv0])] for every visible
    process, or None when this platform cannot be enumerated."""
    try:
        if sys.platform == "win32":
            return _win_processes()
        if os.path.isdir("/proc"):
            return _proc_processes()
        return _ps_processes()
    except Exception:
        return None


def _ancestors(procs: list, pid: int) -> set:
    parent = {p[0]: p[1] for p in procs}
    out = set()
    cur = pid
    while cur and cur not in out:
        out.add(cur)
        cur = parent.get(cur, 0)
    return out


def running_from(dirs: Iterable, *, procs: Optional[list] = None,
                 self_pid: Optional[int] = None) -> Optional[list]:
    """Processes running a program from inside any of *dirs* (by executable
    path, or on Linux the absolute program path they were started with),
    excluding this process and its ancestors. Programs they started from
    elsewhere, such as a web browser, are not included.

    Returns [(pid, exe)], or None when processes cannot be listed on this
    platform."""
    procs = list_processes() if procs is None else procs
    if procs is None:
        return None
    me = os.getpid() if self_pid is None else self_pid
    skip = _ancestors(procs, me)
    roots = [d for d in (str(x) for x in dirs) if d]

    def programs(p) -> list:
        out = [p[2]] if p[2] else []
        if len(p) > 4 and p[4] and os.path.isabs(p[4]):
            out.append(p[4])
        return out

    return [(p[0], p[2]) for p in procs
            if p[0] not in skip and any(_inside(x, d) for x in programs(p) for d in roots)]


def _stop(pid: int, timeout: float = 8.0) -> bool:
    """Terminate *pid* and wait for it to exit. True when it is gone."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x0001 | 0x00100000, False, pid)  # TERMINATE | SYNCHRONIZE
        if not h:
            return not _alive(pid)
        try:
            k32.TerminateProcess(h, 1)
            return k32.WaitForSingleObject(h, int(timeout * 1000)) == 0
        finally:
            k32.CloseHandle(h)
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    time.sleep(0.3)
    return not _alive(pid)


def _alive(pid: int) -> bool:
    """Whether *pid* is still running. An exited process its parent has not
    reaped yet (a zombie) counts as gone."""
    if sys.platform == "win32":
        procs = list_processes()
        return bool(procs) and any(p[0] == pid for p in procs)
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                raw = fh.read().decode("utf-8", "replace")
            return raw[raw.rindex(")") + 2:].split()[0] not in ("Z", "X")
        except (OSError, ValueError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        res = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10)
        return not res.stdout.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


# --------------------------------------------------------------------------- #
#  Planning                                                                    #
# --------------------------------------------------------------------------- #

class _Item:
    __slots__ = ("path", "kind", "status", "reason")

    def __init__(self, path, kind, status, reason=""):
        self.path, self.kind, self.status, self.reason = str(path), kind, status, reason


def _data_dirs(root: Path, m: Optional[dict]) -> list:
    """[(folder, mode, preexisting, parents_created, recorded, others)] - every
    data folder this install used. mode is 'owned' (all LocaLM's), 'entries'
    (LocaLM's entries that are not in *preexisting*) or 'legacy' (nothing
    records what was there before). *others* lists the other existing install
    folders that use it, from its marker."""
    out: list = []
    seen: list = []

    def add(path, recorded, parents=()):
        if not path or any(_same(path, s) for s in seen):
            return
        seen.append(path)
        marker = read_marker(path)
        others = [p for p in (marker or {}).get("installs", [])
                  if not _same(p, root) and _live_install(p)]
        if _same(path, root / PORTABLE_HOME):
            out.append((path, "owned", None, [], recorded, others))
        elif marker is not None:
            owned = marker["created"]
            out.append((path, "owned" if owned else "entries", marker["preexisting"],
                        list(parents) if owned else [], recorded, others))
        elif m and _same(path, m.get("data_dir", "")) \
                and isinstance(m.get("data_preexisting"), list):
            owned = bool(m.get("data_created")) and (Path(path) / DATA_MARKER).is_file()
            out.append((path, "owned" if owned else "entries", m["data_preexisting"],
                        list(parents) if owned else [], recorded, others))
        else:
            out.append((path, "legacy", None, [], recorded, others))

    if m and m.get("data_dir"):
        add(m["data_dir"], True, m.get("data_parents_created") or [])
    for p in (m or {}).get("previous_data_dirs") or []:
        add(p, True)
    portable = root / PORTABLE_HOME
    if portable.is_dir():
        add(str(portable), False)
    line = _cfg_data_dir(root)
    if line and os.path.isabs(os.path.expanduser(line)):
        add(_plain_abs(line), False)
    return out


def _read_cfg_line(path: Path) -> str:
    """The first line of a ``localm-home.cfg``: UTF-8 (with or without a BOM),
    or on Windows, when that fails, the console (OEM) code page that cmd.exe's
    ``echo`` writes. Raises OSError or ValueError when it cannot be read."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if sys.platform != "win32":
            raise
        import ctypes
        text = raw.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def _cfg_data_dir(root: Path) -> str:
    """The data folder ``localm-home.cfg`` names, or "" when there is none or
    it cannot be read (_plan reports the unreadable case)."""
    cfg = root / HOME_CFG_NAME
    if not cfg.is_file():
        return ""
    try:
        return _read_cfg_line(cfg)
    except (OSError, ValueError, LookupError):
        return ""


def _plan(root: Path, m: Optional[dict], *, purge_data: bool,
          defer_runtime: bool) -> Tuple[List[_Item], list]:
    items: List[_Item] = []
    data_info = []
    in_use = [p for p in {sys.executable, getattr(sys, "_base_executable", ""),
                          sys.prefix, sys.base_prefix} if p]

    def runtime(path: Path, kind: str, recorded: bool):
        name = path.name
        fixed = _same(path, root / name) and name in DEFERRABLE
        if not path.exists():
            if recorded:
                items.append(_Item(path, kind, "gone"))
            return
        reason = _unsafe_data_dir(str(path), root) if kind == "runtime" else None
        if reason:
            items.append(_Item(path, kind, "refuse", reason))
            return
        busy = any(_inside(p, path) for p in in_use)
        if (defer_runtime or busy) and fixed:
            items.append(_Item(path, kind, "defer" if recorded else "warn-defer",
                               "" if recorded else "not in the install record"))
        elif busy:
            items.append(_Item(path, kind, "refuse",
                               "in use by this uninstaller - remove it after it exits"))
        else:
            items.append(_Item(path, kind, "remove" if recorded else "warn",
                               "" if recorded else "not in the install record"))

    # The environment.
    venv = root / ".venv"
    recorded_venv = m.get("venv") if m else ""
    if recorded_venv and not _same(recorded_venv, venv):
        items.append(_Item(recorded_venv, "venv", "refuse",
                           "recorded outside this folder"))
    if venv.exists():
        if _is_our_venv(venv) or recorded_venv:
            runtime(venv, "venv", True)
        else:
            items.append(_Item(venv, "venv", "keep",
                               "not a LocaLM environment (no setup marker)"))

    # Portable Python tooling.
    for key, name in RUNTIME_DIRS.items():
        rec = m.get(key, "") if m else ""
        if rec:
            p = Path(rec)
            if m.get("runtime_contained"):
                if _same(p, root / name) or _inside(p, root):
                    runtime(p, "runtime", True)
                else:
                    items.append(_Item(p, "runtime", "refuse",
                                       "recorded outside this folder"))
            else:
                items.append(_Item(p, "runtime", "keep",
                                   "shared runtime kept (used by other clones)"))
        elif (root / name).is_dir():
            runtime(root / name, "runtime", False)

    # Native runtime files.
    lib_dir = Path(m["lib_dir"]) if m and m.get("lib_dir") else root.joinpath(*_DEFAULT_LIB)
    recorded_names = list((m or {}).get("lib_entries") or (m or {}).get("binaries") or [])
    lib_ok = _inside(lib_dir, root) and not _same(lib_dir, root)
    for name in recorded_names:
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            items.append(_Item(name, "lib", "refuse", "suspicious name in the install record"))
            continue
        if not (m and m.get("lib_dir")):
            items.append(_Item(name, "lib", "refuse", "no runtime folder recorded"))
            continue
        if not lib_ok:
            items.append(_Item(lib_dir / name, "lib", "refuse",
                               "runtime folder recorded outside this folder"))
            continue
        items.append(_Item(lib_dir / name, "lib", "remove"))
    if lib_ok and lib_dir.is_dir():
        for name in _lib_entries(lib_dir):
            if name not in recorded_names:
                items.append(_Item(lib_dir / name, "lib", "warn",
                                   "in LocaLM's runtime folder but not in the install record"
                                   if m else "no install record - not certain we created this"))

    # Single files: the data-folder pointer, shortcuts, launcher entries.
    cfg = root / HOME_CFG_NAME
    rec_cfg = (m or {}).get("home_cfg", "")
    if rec_cfg and not _same(rec_cfg, cfg):
        items.append(_Item(rec_cfg, "file", "refuse", "not this clone's localm-home.cfg"))
    elif rec_cfg:
        items.append(_Item(cfg, "file", "remove"))
    elif cfg.exists():
        items.append(_Item(cfg, "file", "warn", "not in the install record"))
    shortcut = (m or {}).get("shortcut", "")
    if shortcut:
        if Path(shortcut).name.lower() in _SHORTCUT_NAMES:
            items.append(_Item(shortcut, "file", "remove"))
        else:
            items.append(_Item(shortcut, "file", "refuse", "not a LocaLM shortcut"))
    for f in (m or {}).get("files") or []:
        ok = (Path(f).name.lower() in _SHORTCUT_NAMES
              or (_inside(f, root) and not _same(f, root)))
        items.append(_Item(f, "file", "remove" if ok else "refuse",
                           "" if ok else "outside this folder and not a LocaLM shortcut"))
    desktop = root / _LAUNCHER_DESKTOP
    if desktop.is_file() and not any(_same(i.path, desktop) for i in items):
        items.append(_Item(desktop, "file", "warn", "not in the install record"))

    # The user's saved data.
    cfg = root / HOME_CFG_NAME
    if cfg.is_file():
        try:
            _read_cfg_line(cfg)
        except (OSError, ValueError, LookupError) as e:
            items.append(_Item(cfg, "note", "note",
                               f"cannot read it ({e}); the data folder it names "
                               "was not checked"))
    if purge_data:
        items.append(_Item("Your web browser", "note", "note",
                           "it may keep copies of recent chats until you clear its "
                           "site data for the address LocaLM ran on"))
    for d, mode, pre, parents, recorded, others in _data_dirs(root, m):
        dp = Path(d)
        if not dp.exists():
            if recorded:
                items.append(_Item(d, "data", "gone"))
            continue
        size, count, complete = _tree_size(dp)
        info = {"path": d, "mode": mode, "bytes": size, "files": count,
                "complete": complete, "delete": purge_data, "entries": [],
                "kept_entries": []}
        data_info.append(info)
        if not purge_data:
            items.append(_Item(d, "data", "keep", "your saved data"))
            continue
        reason = _unsafe_data_dir(d, root)
        if not reason and others:
            reason = ("also used by the LocaLM install in " + ", ".join(others)
                      + " - uninstall that one too, or delete it by hand")
        if reason:
            items.append(_Item(d, "data", "refuse", reason))
            info["delete"] = False
            continue
        if mode == "owned":
            items.append(_Item(d, "data", "remove"))
            for par in sorted(parents, key=len, reverse=True):
                items.append(_Item(par, "empty-dir", "remove"))
            continue
        foreign = set(pre or [])
        try:
            names = sorted(e.name for e in os.scandir(dp))
        except OSError as e:
            items.append(_Item(d, "data", "refuse", f"cannot read it: {e}"))
            info["delete"] = False
            continue
        unsure = []
        for name in names:
            if not is_data_entry(name) or name in foreign:
                info["kept_entries"].append(name)
            elif mode == "legacy" and name in _LEGACY_KEPT:
                info["kept_entries"].append(name)
                unsure.append(name)
            else:
                items.append(_Item(dp / name, "data-entry", "remove"))
                info["entries"].append(name)
        if unsure:
            items.append(_Item(d, "data", "keep",
                               "LocaLM cannot tell whether these were already yours, so "
                               "they are kept; delete them by hand if they are LocaLM's: "
                               + ", ".join(unsure)))
        rest = len(info["kept_entries"]) - len(unsure)
        if rest:
            why = ("was already in this folder before LocaLM was installed"
                   if pre is not None else "not one of LocaLM's own files")
            items.append(_Item(d, "data", "keep", f"{rest} item(s) kept: {why}"))

    # A stale lock left by an interrupted runtime download.
    lock = lib_dir.with_name(lib_dir.name + ".setup.lock")
    if lib_ok and lock.is_dir():
        items.append(_Item(lock, "lib", "warn", "left by an interrupted runtime download"))

    # uv's own entries that point into this folder: PATH, shell files, receipt.
    uv_item = next((i for i in items if _same(i.path, root / ".uv")), None)
    if uv_item is None:
        follow = "remove" if not (root / ".uv").exists() else None
    else:
        follow = {"remove": "remove", "defer": "remove", "gone": "remove",
                  "warn": "warn", "warn-defer": "warn"}.get(uv_item.status)
    handled = (m.get("path_dir", "") if m and m.get("path_modified") else "")
    for it in _pointers_into(root):
        if it.kind == "command":
            if not (m and m.get("command_shim") and _same(m["command_shim"], it.path)):
                items.append(it)
            continue
        if it.kind == "path-entry" and handled and _same(it.path, handled):
            continue
        if it.kind != "path-entry" and follow is None:
            continue
        if it.kind != "path-entry":
            it.status = follow
        items.append(it)

    # Things kept on purpose, reported so nothing setup did is silent.
    if m and m.get("uv_shared_installed"):
        where = shutil.which("uv") or "your user profile"
        items.append(_Item(where, "report", "keep",
                           "uv was installed by LocaLM setup; other programs may use it"))
    if m and m.get("venv") and not m.get("runtime_contained"):
        items.append(_Item("uv's Python and download cache in your user profile",
                           "report", "keep",
                           "shared setup - other programs and LocaLM folders may use them"))
    if purge_data:
        for p in _webview_profiles(root):
            items.append(_Item(p, "report", "keep",
                               "the app window's saved sign-in and copies of recent chats; "
                               "other pywebview apps share this folder, so delete it by "
                               "hand if nothing else uses it"))
        for d in data_info:
            for proj in _coder_project_dirs(Path(d["path"])):
                items.append(_Item(proj, "report", "keep",
                                   "LocaLM coder notes inside one of your projects"))
    return items, data_info


def _pointers_into(root: Path) -> List[_Item]:
    """Entries outside this folder that exist only to point into it: user
    PATH entries (Windows), the lines Astral's uv installer adds to shell
    startup files for a uv in ``<root>/.uv``, and its install receipt."""
    out: List[_Item] = []
    if sys.platform == "win32":
        try:
            from localm import globalcmd
            value, _ = globalcmd._win_read_user_path()
        except Exception:
            value = ""
        for entry in value.split(os.pathsep):
            e = entry.strip().strip('"')
            x = os.path.expandvars(e)
            if e and _is_full_path(x) and _inside(x, root):
                out.append(_Item(e, "path-entry", "remove", "points into this folder"))
    else:
        for rc in _rc_files():
            if _uv_lines_in(rc, root):
                out.append(_Item(rc, "rc-line", "remove",
                                 "a line that loads the uv in this folder"))
        shim = Path.home() / ".local" / "bin" / "localm"
        if shim.is_symlink() and _shim_points_into(shim, root):
            out.append(_Item(shim, "command", "remove",
                             "the localm command, which points into this folder"))
    for receipt in _uv_receipts():
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prefix = data.get("install_prefix") if isinstance(data, dict) else ""
        if isinstance(prefix, str) and prefix and _inside(prefix, root):
            out.append(_Item(receipt, "file", "remove",
                             "uv's install receipt for the uv in this folder"))
    return out


def _is_full_path(p: str) -> bool:
    """Whether *p* names a location on its own: on Windows a drive or UNC path
    (not ``\\x``, ``x`` or one with an undefined ``%VAR%`` left in it),
    elsewhere a path starting with ``/``."""
    if not p or "%" in p:
        return False
    if os.sep == "\\":
        drive, rest = os.path.splitdrive(p)
        return bool(drive) and rest.startswith(("\\", "/"))
    return os.path.isabs(p)


def _shim_points_into(shim: Path, root: Path) -> bool:
    """Whether the command *shim* belongs to the install at *root*: a symlink
    whose target is inside it, or a file inside it. True when it is gone."""
    if shim.is_symlink():
        try:
            target = os.readlink(shim)
        except OSError:
            return False
        if not os.path.isabs(target):
            target = os.path.join(str(shim.parent), target)
        return _inside(target, root)
    return not shim.exists() or _inside(shim, root)


_UV_RC_NAMES = (".profile", ".bashrc", ".bash_profile", ".bash_login", ".zshrc",
                ".zshenv", os.path.join(".config", "fish", "conf.d", "uv.env.fish"))


def _rc_files() -> List[Path]:
    home = Path.home()
    return [home / n for n in _UV_RC_NAMES if (home / n).is_file()]


def _uv_line_forms(root: Path) -> set:
    uvdir = root / ".uv"
    exprs = {uvdir.as_posix(), str(uvdir)}
    try:
        exprs.add("$HOME/" + uvdir.relative_to(Path.home()).as_posix())
    except ValueError:
        pass
    forms = set()
    for x in exprs:
        for script in ("env", "env.fish"):
            forms.add(f'. "{x}/{script}"')
            forms.add(f'source "{x}/{script}"')
    return forms


def _uv_lines_in(rc: Path, root: Path) -> int:
    try:
        text = rc.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return 0
    forms = _uv_line_forms(root)
    return sum(1 for line in text.splitlines() if line.strip() in forms)


def _strip_uv_lines(rc: Path, root: Path) -> int:
    """Remove the uv-loading lines for ``<root>/.uv`` from *rc*, leaving every
    other byte as it was. A symlinked *rc* is edited through the link (the
    file it points to is rewritten, the link stays), and the file keeps its
    permissions. Deletes uv's own fish snippet when nothing else is left in it
    and it is a plain file. Returns the number of lines removed."""
    raw = rc.read_bytes().decode("utf-8")
    forms = _uv_line_forms(root)
    lines = raw.splitlines(keepends=True)
    kept = [ln for ln in lines if ln.strip() not in forms]
    removed = len(lines) - len(kept)
    if not removed:
        return 0
    text = "".join(kept)
    if rc.name == "uv.env.fish" and not text.strip() and not rc.is_symlink():
        rc.unlink()
        return removed
    real = Path(os.path.realpath(rc))
    tmp = real.with_name(real.name + ".localm-tmp")
    tmp.write_bytes(text.encode("utf-8"))
    try:
        shutil.copymode(real, tmp)
        os.replace(tmp, real)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return removed


def _uv_receipts() -> List[Path]:
    dirs = []
    for base in (os.environ.get("XDG_CONFIG_HOME", ""), os.environ.get("LOCALAPPDATA", ""),
                 str(Path.home() / ".config")):
        if base:
            dirs.append(Path(base) / "uv")
    out = []
    for d in dirs:
        f = d / "uv-receipt.json"
        if f.is_file() and not any(_same(f, o) for o in out):
            out.append(f)
    return out


def _webview_profiles(root: Path) -> List[str]:
    """pywebview's default profile folders (shared by every pywebview app),
    where the app window keeps its cookies and page storage, when this install
    has the app-window package and the folder exists."""
    venv = root / ".venv"
    if not (any(venv.glob("Lib/site-packages/webview"))
            or any(venv.glob("lib/python*/site-packages/webview"))):
        return []
    if sys.platform == "win32":
        cands = [os.path.join(os.environ.get("APPDATA", ""), "pywebview"),
                 os.path.join(os.environ.get("USERPROFILE", ""), "pywebview")]
    elif sys.platform.startswith("linux"):
        cands = [str(Path.home() / ".pywebview")]
    else:
        cands = []
    return [c for c in cands if c and os.path.isdir(c)]


def _coder_project_dirs(data_dir: Path) -> List[str]:
    """``<project>/.localcoder`` for each project in the data folder's coder
    project list that has one."""
    try:
        entries = json.loads((data_dir / "coder-projects.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for e in entries if isinstance(entries, list) else []:
        path = e.get("path") if isinstance(e, dict) else None
        if not isinstance(path, str) or path.startswith(("\\\\", "//")):
            continue
        d = os.path.join(path, ".localcoder")
        if os.path.isdir(d):
            out.append(d)
    return out


# --------------------------------------------------------------------------- #
#  Uninstall                                                                   #
# --------------------------------------------------------------------------- #

def uninstall(root, *, purge_data=False, dry_run=False, force=False,
              stop_running=False, defer_runtime=False, log=print) -> dict:
    """Plan (and unless *dry_run*, execute) the uninstall of the clone at *root*.

    Item statuses: remove (recorded), warn (removed only with *force*), keep,
    refuse (never removed, even with *force*), defer (left for the caller to
    delete after this process exits). With *defer_runtime* the in-clone Python
    runtime folders are always deferred and their names written to the
    pending file.

    When LocaLM is running from this folder, nothing is touched unless
    *stop_running*, which stops those processes first.

    The report's ``exit`` is one of the EXIT_* codes."""
    root = Path(root).resolve()
    report = {"removed": [], "skipped": [], "warned": [], "refused": [],
              "failed": [], "deferred": [], "running": [], "stopped": [], "notes": [],
              "data": [], "venv": "", "ok": True, "no_manifest": False,
              "dry_run": bool(dry_run), "exit": EXIT_OK}

    m = load(root)
    if m is not None and (m.get("schema") or 0) > SCHEMA_VERSION:
        log(f"[uninstall] Manifest schema {m.get('schema')!r} is newer than this "
            "installer understands - aborting for safety. Nothing removed.")
        report["ok"] = False
        report["exit"] = EXIT_ABORTED
        return report
    if m is None:
        report["no_manifest"] = True
    else:
        m = _upgrade(m, root)
    report["venv"] = str(root / ".venv")

    items, data_info = _plan(root, m, purge_data=purge_data,
                             defer_runtime=defer_runtime)
    report["data"] = data_info

    busy_dirs = [root / ".venv"] + [root / n for n in RUNTIME_DIRS.values()]
    for info in data_info:
        if info["delete"] and info["mode"] == "owned":
            busy_dirs.append(Path(info["path"]))
        elif info["delete"]:
            busy_dirs.extend(Path(info["path"]) / e for e in info["entries"])
    running = running_from(busy_dirs)
    report["running"] = running or []
    if running is None:
        report["skipped"].append(("running LocaLM processes",
                                  "could not list processes on this system"))

    if dry_run:
        for it in items:
            _classify(report, it, force=force, dry_run=True, root=root)
        _global_command(report, m, root, dry_run=True)
        return _finish(report, root, m, items, dry_run=True)

    if report["running"]:
        if not stop_running:
            report["ok"] = False
            report["exit"] = EXIT_RUNNING
            return report
        if not _stop_all(report, busy_dirs):
            report["ok"] = False
            report["exit"] = EXIT_FAILED
            return report

    for it in items:
        _classify(report, it, force=force, dry_run=False, root=root)
    _global_command(report, m, root, dry_run=False)
    return _finish(report, root, m, items, dry_run=False)


def _stop_all(report: dict, dirs: list, rounds: int = 3) -> bool:
    """Stop every process running from *dirs*, parents first, rescanning for
    anything a stopped process started meanwhile. False when one survives."""
    pending = list(report["running"])
    for _ in range(rounds):
        for pid, exe in pending:
            label = f"PID {pid}" + (f" {exe}" if exe else "")
            if _stop(pid):
                if label not in report["stopped"]:
                    report["stopped"].append(label)
            else:
                report["failed"].append((label, "could not stop it - close LocaLM and try again"))
                return False
        time.sleep(0.5)
        pending = running_from(dirs) or []
        if not pending:
            return True
    for pid, exe in pending:
        report["failed"].append((f"PID {pid}" + (f" {exe}" if exe else ""),
                                 "keeps restarting - close LocaLM and try again"))
    return False


def _global_command(report: dict, m: Optional[dict], root: Path, *, dry_run: bool) -> None:
    """Remove the optional global `localm` command through globalcmd. A
    recorded command that now points to another install is kept."""
    if m is None or not (m.get("command_shim") or m.get("path_modified")):
        return
    shim = m.get("command_shim", "")
    if shim and Path(shim).name.lower() not in _SHIM_NAMES:
        report["refused"].append((shim, "not a localm command shim"))
        return
    if shim and not _shim_points_into(Path(shim), root):
        report["skipped"].append((shim, "points to another LocaLM install - kept"))
        shim = ""
    path_dir = m.get("path_dir", "") if m.get("path_modified") else ""
    if dry_run:
        if shim:
            report["removed"].append(shim)
        if path_dir:
            report["removed"].append(f"PATH entry {path_dir}")
        return
    try:
        from localm import globalcmd
        gc = globalcmd.uninstall_command(path_dir, shim)
    except Exception as e:
        report["failed"].append((shim or path_dir,
                                 f"could not remove the global command: {e}"))
        return
    report["removed"].extend(gc.get("removed", []))
    for n in gc.get("notes", []):
        if n.startswith("could not"):
            report["failed"].append((shim or path_dir, n))
        else:
            report["skipped"].append((n, "global command"))


def _classify(report: dict, it: _Item, *, force: bool, dry_run: bool,
              root: Path) -> None:
    st = it.status
    if st == "gone":
        report["skipped"].append((it.path, "already gone"))
        return
    if st == "keep":
        report["skipped"].append((it.path, it.reason))
        return
    if st == "refuse":
        report["refused"].append((it.path, it.reason))
        return
    if st == "note":
        report["notes"].append((it.path, it.reason))
        return
    if st in ("warn", "warn-defer"):
        report["warned"].append((it.path, it.reason))
        if not force:
            return
        st = "defer" if st == "warn-defer" else "remove"
        if dry_run:
            return
    if st == "defer":
        report["deferred"].append(it.path)
        return
    if it.kind == "path-entry":
        label = f"PATH entry {it.path}"
        if dry_run:
            report["removed"].append(label)
            return
        try:
            from localm import globalcmd
            changed = globalcmd._win_path_remove(it.path)
        except Exception as e:
            report["failed"].append((label, f"could not update PATH: {e}"))
            return
        if changed:
            report["removed"].append(label)
        else:
            report["skipped"].append((label, "already gone"))
        return
    if it.kind == "rc-line":
        label = f"uv line in {it.path}"
        if dry_run:
            report["removed"].append(label)
            return
        try:
            n = _strip_uv_lines(Path(it.path), root)
        except (OSError, ValueError) as e:
            report["failed"].append((label, f"could not edit it: {e}"))
            return
        if n:
            report["removed"].append(label)
        else:
            report["skipped"].append((label, "already gone"))
        return
    p = Path(it.path)
    if not (p.exists() or _is_link(p)):
        report["skipped"].append((it.path, "already gone"))
        return
    if dry_run:
        report["removed"].append(it.path)
        return
    try:
        if it.kind == "empty-dir":
            try:
                p.rmdir()
            except OSError:
                report["skipped"].append((it.path, "not empty - kept"))
                return
        elif it.kind in ("data", "runtime", "venv", "data-entry", "lib"):
            _remove_entry(p)
        elif p.is_dir() and not _is_link(p):
            report["refused"].append((it.path, "expected a file, found a folder"))
            return
        else:
            _remove_entry(p)
        report["removed"].append(it.path)
    except OSError as e:
        report["failed"].append((it.path, f"could not remove: {e}"))


def _finish(report: dict, root: Path, m: Optional[dict], items: List[_Item],
            *, dry_run: bool) -> dict:
    asked_refused = any(it.status == "refuse" and it.kind in ("data", "data-entry", "venv", "runtime", "lib", "file")
                        for it in items)
    if report["failed"]:
        report["ok"] = False
        report["exit"] = EXIT_FAILED
    elif asked_refused:
        report["exit"] = EXIT_PARTIAL
    if dry_run:
        _sweep_pycache(report, root, dry_run=True)
        return report
    if report["failed"]:
        # Keep the manifest and the runtime; write no pending file.
        report["deferred"] = []
        try:
            pending_path(root).unlink()
        except OSError:
            pass
        return report
    _sweep_pycache(report, root, dry_run=False)
    names = [Path(p).name for p in report["deferred"]
             if Path(p).name in DEFERRABLE and _same(p, root / Path(p).name)]
    if names:
        keep = {"schema": SCHEMA_VERSION}
        if ".venv" in names:
            keep["venv"] = str(root / ".venv")
        for key, name in RUNTIME_DIRS.items():
            if name in names:
                keep[key] = str(root / name)
                keep["runtime_contained"] = True
        try:
            _write(root, keep)
            pending_path(root).write_text("\n".join(names) + "\n", encoding="ascii")
        except OSError as e:
            report["failed"].append((str(pending_path(root)), f"could not write: {e}"))
            report["ok"] = False
            report["exit"] = EXIT_FAILED
        return report
    for p in (manifest_path(root), pending_path(root)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            report["failed"].append((str(p), f"could not remove: {e}"))
            report["ok"] = False
            report["exit"] = EXIT_FAILED
    return report


def _pycache_dirs(root: Path, keep: Iterable[str] = ()) -> List[Path]:
    """The ``__pycache__`` folders under *root*, outside ``.git``, the runtime
    folders uninstall removes whole, the folders in *keep*, and links."""
    kept = [Path(k) for k in keep]
    out: List[Path] = []
    for dirpath, dirnames, _files in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in DEFERRABLE and d != ".git"]
        dirnames[:] = [d for d in dirnames if not _is_link(here / d)
                       and not any(_same(here / d, k) for k in kept)]
        if "__pycache__" in dirnames:
            out.append(here / "__pycache__")
            dirnames.remove("__pycache__")
    return out


def _sweep_pycache(report: dict, root: Path, *, dry_run: bool) -> None:
    """Remove the Python bytecode caches running LocaLM from this folder
    created, leaving any inside a data folder that is kept."""
    keep = [d["path"] for d in report["data"] if not d["delete"]]
    dirs = _pycache_dirs(root, keep)
    if not dirs:
        return
    left = 0
    if not dry_run:
        for d in dirs:
            try:
                _rmtree(d)
            except OSError:
                left += 1
    label = f"{root}{os.sep}...{os.sep}__pycache__  ({len(dirs)} Python bytecode cache folders)"
    if left:
        report["skipped"].append((label, f"{left} could not be removed - they are "
                                         "only caches and can be deleted by hand"))
    else:
        report["removed"].append(label)


def finish_pending(root) -> Tuple[list, list]:
    """Remove the folders named in the pending file (allowlisted names only),
    then the pending file and the manifest when all are gone. For callers that
    run after the uninstaller's process has exited. Returns (removed, left)."""
    root = Path(root).resolve()
    try:
        names = pending_path(root).read_text(encoding="ascii").split()
    except (OSError, ValueError):
        return [], []
    removed, left = [], []
    for name in names:
        if name not in DEFERRABLE:
            continue
        p = root / name
        if not p.exists():
            continue
        try:
            _rmtree(p)
            removed.append(str(p))
        except OSError:
            left.append(str(p))
    if not left:
        for p in (pending_path(root), manifest_path(root)):
            try:
                p.unlink()
            except OSError:
                pass
    return removed, left


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #

def format_report(rep: dict) -> List[str]:
    """The report as the lines setup prints."""
    dry = rep.get("dry_run")
    out: List[str] = []

    def section(title, rows):
        if rows:
            out.append(f"  {title}")
            out.extend(f"    {r}" for r in rows)

    running = [f"PID {pid}  {exe or '(started by LocaLM)'}" for pid, exe in rep.get("running", [])]
    if dry:
        section("LocaLM is running from this folder - it will be stopped first:", running)
    section("Stopped:", rep.get("stopped", []))
    whole = [d["path"] for d in rep.get("data", []) if d["delete"] and d["mode"] == "owned"]
    entries = [os.path.join(d["path"], e) for d in rep.get("data", []) if d["delete"]
               for e in d["entries"]]
    removed = [x for x in rep.get("removed", [])
               if not any(_inside(x, d) for d in whole)
               and not any(_same(x, e) for e in entries)]
    if dry:
        removed = list(rep.get("deferred", [])) + removed
    section("Will be removed:" if dry else "Removed:", removed)
    if rep.get("data"):
        failed = [x for x, _ in rep.get("failed", [])]
        refused = [x for x, _ in rep.get("refused", [])]
        out.append("  Your saved data (chats, settings, downloaded models, generated images):")
        for d in rep["data"]:
            size = human_size(d["bytes"]) + ("" if d["complete"] else "+")
            what = f"{d['path']}  ({size}, {d['files']:,} files)"
            if not d["delete"] and any(_same(x, d["path"]) for x in refused):
                out.append(f"    NOT DELETED (see REFUSED below): {what}")
            elif not d["delete"]:
                out.append(f"    KEPT: {what}")
            elif d["mode"] == "owned":
                if dry:
                    out.append(f"    WILL BE DELETED: {what}")
                elif any(_same(x, d["path"]) for x in failed):
                    out.append(f"    NOT DELETED (see below): {what}")
                else:
                    out.append(f"    DELETED: {what}")
            else:
                out.append(f"    In {what}:")
                bad = [e for e in d["entries"]
                       if any(_same(x, os.path.join(d["path"], e)) for x in failed)]
                done = [e for e in d["entries"] if e not in bad]
                if done:
                    out.append(f"      {'will be deleted' if dry else 'deleted'}: "
                               + ", ".join(done))
                if bad:
                    out.append("      NOT deleted (see below): " + ", ".join(bad))
                if d["kept_entries"]:
                    out.append("      kept (see Kept below): " + ", ".join(d["kept_entries"]))
    kept = [f"{x}  ({why})" for x, why in rep.get("skipped", [])
            if why not in ("already gone", "your saved data")]
    section("Kept:", kept)
    section("Not in the install record - removed only if you continue:",
            [f"{x}  ({why})" for x, why in rep.get("warned", [])])
    section("REFUSED - never removed:", [f"{x}  ({why})" for x, why in rep.get("refused", [])])
    section("Could not remove:", [f"{x}  ({why})" for x, why in rep.get("failed", [])])
    section("Also note:", [f"{x}  ({why})" for x, why in rep.get("notes", [])])
    if rep.get("no_manifest"):
        out.append("  [!] No install record (.localm-install.json) was found, so the")
        out.append("      items above are LocaLM's usual locations, not a record.")
    if not out:
        out.append("  Nothing to remove.")
    return out


def _print_report(rep: dict) -> None:
    for line in format_report(rep):
        print(line)


# --------------------------------------------------------------------------- #
#  CLI: invoked by setup.sh / setup.bat (python -m localm.install_manifest)    #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="localm.install_manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="merge what setup created into the manifest")
    r.add_argument("--root", default=".")
    r.add_argument("--venv", default="")
    r.add_argument("--lib-dir", default="")
    r.add_argument("--home-cfg", default="")
    r.add_argument("--data-dir", default="")
    r.add_argument("--data-created", action="store_true")
    r.add_argument("--shortcut", default="")
    r.add_argument("--stamp", default="")
    r.add_argument("--runtime-contained", action="store_true")
    r.add_argument("--python-dir", default="")
    r.add_argument("--cache-dir", default="")
    r.add_argument("--uv-dir", default="")
    r.add_argument("--uv-shared-installed", action="store_true")
    r.add_argument("--path-dir", default="")
    r.add_argument("--command-shim", default="")
    r.add_argument("--path-modified", action="store_true")
    r.add_argument("--file", action="append", default=[],
                   help="another file setup created (repeatable)")

    d = sub.add_parser("prepare-data", help="create the data folder and record it")
    d.add_argument("--root", default=".")
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--portable", action="store_true")
    g.add_argument("--data-dir", default="")
    g.add_argument("--keep-current", action="store_true",
                   help="prepare the data folder this install already uses")

    c = sub.add_parser("current-data", help="print the data folder this install uses "
                                            "(exit 1 when there is none)")
    c.add_argument("--root", default=".")

    u = sub.add_parser("uninstall", help="remove what setup created")
    u.add_argument("--root", default=".")
    u.add_argument("--purge-data", action="store_true",
                   help="also delete the saved data (chats, settings, models)")
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--force", action="store_true",
                   help="also remove items with no install record (after the "
                        "caller has warned the user); never overrides the "
                        "catastrophic-path guard")
    u.add_argument("--stop-running", action="store_true",
                   help="stop LocaLM processes running from this folder first")
    u.add_argument("--defer-runtime", action="store_true",
                   help="leave the in-clone Python runtime for the calling "
                        f"script to delete, named in {PENDING_NAME}")

    f = sub.add_parser("finish", help="remove what a previous uninstall deferred")
    f.add_argument("--root", default=".")

    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.cmd == "current-data":
        current = current_data_dir(args.root)
        if not current:
            return 1
        print(current)
        return 0
    if args.cmd == "prepare-data" and args.keep_current:
        current = current_data_dir(args.root)
        if not current:
            print("  [!] No data folder is set up in this folder yet.")
            return 1
        portable = _same(current, Path(args.root).resolve() / PORTABLE_HOME)
        args.portable, args.data_dir = portable, ("" if portable else current)
    if args.cmd == "record":
        try:
            p = record(args.root, venv=args.venv, lib_dir=args.lib_dir,
                       home_cfg=args.home_cfg, data_dir=args.data_dir,
                       data_created=args.data_created, shortcut=args.shortcut,
                       stamp=args.stamp, runtime_contained=args.runtime_contained,
                       python_dir=args.python_dir, cache_dir=args.cache_dir,
                       uv_dir=args.uv_dir, uv_shared_installed=args.uv_shared_installed,
                       path_dir=args.path_dir, command_shim=args.command_shim,
                       path_modified=args.path_modified, files=args.file)
        except (OSError, ValueError) as e:
            print(f"[install] could not record the install: {e}", file=sys.stderr)
            return 1
        print(f"[install] recorded manifest at {p}")
        return 0
    if args.cmd == "prepare-data":
        try:
            target = prepare_data(args.root, data_dir=args.data_dir,
                                  portable=args.portable)
        except (OSError, ValueError) as e:
            print(f"  [!] Cannot use that data folder: {e}")
            return 1
        print(f"  Data directory: {target}")
        return 0
    if args.cmd == "finish":
        removed, left = finish_pending(args.root)
        for p in removed:
            print(f"  Removed: {p}")
        for p in left:
            print(f"  [!] Could not remove {p} - close any LocaLM window and delete it by hand.")
        return 1 if left else 0
    rep = uninstall(args.root, purge_data=args.purge_data, dry_run=args.dry_run,
                    force=args.force, stop_running=args.stop_running,
                    defer_runtime=args.defer_runtime)
    _print_report(rep)
    if rep["exit"] == EXIT_RUNNING:
        print("  [!] LocaLM is running from this folder. Close it (or let setup stop")
        print("      it) and run uninstall again. Nothing was changed.")
    return rep["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
