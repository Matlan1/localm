# SPDX-License-Identifier: AGPL-3.0-or-later
"""File identities, paths that cannot leave a folder, and writes that are on disk
when the call returns."""

from __future__ import annotations

import errno
import importlib
import json
import os
import shutil
import stat
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Optional, TypeVar

from . import shared

#: Seconds the moves of one run wait, in all, for files another program holds.
HELD_WAIT_S = 30.0
_RETRY_S = 0.1
# Seconds a write waits for a reader of the same file to close it.
_REPLACE_WAIT_S = 10.0
#: Seconds a read of a record or plan is tried again while it fails.
READ_WAIT_S = 10.0
_NAME_SURROGATE = 0x20000000
# Without it, opening a file for reading and writing on Windows removes a Ctrl-Z at its end.
_BINARY = getattr(os, "O_BINARY", 0)
PYCACHE = "__pycache__"
#: What the name of a copy that is not complete yet ends in.
PART = ".localm-part"
_T = TypeVar("_T")


class Unsafe(Exception):
    """A path that is not one of this install's own files."""


class Held(Exception):
    """A file or folder that another program holds, or that this account may not
    use. The step that met it can be made again.

    *path* is the file, *reason* what the system says and *holders* the programs that
    hold it, ``[{"pid", "name"}]``, as far as the system tells; ``str()`` names them
    all."""

    def __init__(self, path, reason) -> None:
        self.path, self.reason, self.holders = str(path), str(reason), _holders(path)
        by = ", ".join(f"{h['name']} (pid {h['pid']})" for h in self.holders)
        super().__init__(f"{path} cannot be used ({reason}){by and ', held by ' + by}")


class Blocked(Held):
    """A file that cannot be read now; nothing is known about what it holds."""


def _holders(path) -> list:
    try:
        if sys.platform == "win32":
            return shared.apply_update.rm_holders([path])
        if sys.platform == "linux":
            return _lockers(os.stat(path).st_ino)
    except OSError:
        return []
    return []


def _lockers(inode: int) -> list:
    """The programs that have the file *inode* locked, from ``/proc/locks``."""
    found = []
    for line in Path("/proc/locks").read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) > 5 and parts[1] != "->" and parts[5].rpartition(":")[2] == str(inode):
            name = Path(f"/proc/{parts[4]}/comm").read_text(encoding="utf-8").strip()
            found.append({"pid": int(parts[4]), "name": name})
    return found


#: Asked about a file that is still held: whether to go on waiting for it.
Ask = Callable[[Held], bool]


class Wait:
    """How long the moves of one run wait, in all, for held files. Then *ask*
    decides, until it says no, whether they wait as long again."""

    def __init__(self, seconds: Optional[float] = None, ask: Optional[Ask] = None) -> None:
        self.seconds = HELD_WAIT_S if seconds is None else seconds
        self.ask = ask
        self._deadline: Optional[float] = None

    def again(self, held: Held) -> bool:
        """Whether to wait on for the file of *held*."""
        if self.ask is None or not self.ask(held):
            self.ask = None
            return False
        self._deadline = None
        return True

    def until(self) -> float:
        """The time (``time.monotonic``) at which waiting ends; the first call
        starts the wait."""
        if self._deadline is None:
            self._deadline = time.monotonic() + self.seconds
        return self._deadline


def patient(wait: Wait, path, action: Callable[[], _T], why: str = "") -> _T:
    """*action*'s result, tried again until *wait* ends while the file *path* is
    held (Windows) or locked. Raises :class:`Held`, with *why* for what the system
    says, when it still is, or may not be used."""
    while True:
        try:
            return action()
        except (PermissionError, BlockingIOError) as e:
            may_end = os.name == "nt" or isinstance(e, BlockingIOError)
            if not may_end or time.monotonic() >= wait.until():
                held = Held(path, why or e.strerror or e)
                if not may_end or not wait.again(held):
                    raise held from e
            time.sleep(_RETRY_S)


def long_path(path) -> Path:
    """*path* absolute, in the form that reaches a file deeper than 260 characters
    on Windows."""
    return shared.updater_components._long(Path(path))


def real_folder(path) -> Path:
    """The folder or file *path* leads to, with links followed, in :func:`long_path`
    form."""
    return long_path(os.path.realpath(path))


def short_path(path: Path) -> str:
    """*path* (from :func:`long_path`) in its usual form."""
    text = str(path)
    if os.name == "nt" and text.startswith("\\\\?\\"):
        rest = text[4:]
        return "\\\\" + rest[4:] if rest.upper().startswith("UNC\\") else rest
    return text


# ------------------------------------------------------------------- paths

def clean(rel) -> str:
    """*rel*, when it is a relative POSIX path of plain file names. Raises
    :class:`Unsafe` for anything else: absolute, with a drive, a ``..`` or an empty
    part or a control character, and on Windows a stream, a part ending in a dot or
    a space or a device name."""
    package = shared.update_package
    ok = rel == package.NAME or package._safe_path(rel)
    if ok and os.name == "nt":
        ok = not any(":" in part or part[-1] in ". " or PureWindowsPath(part).is_reserved()
                     for part in rel.split("/"))
    if not ok or any(ord(c) < 32 for c in rel):
        raise Unsafe(f"{str(rel)[:120]!r} is not a path inside the install")
    return rel


def lstat(path) -> Optional[os.stat_result]:
    """``os.lstat(path)``, or None when there is nothing at *path*."""
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def spelled(path: Path) -> bool:
    """Whether the entry at *path*, which exists, is spelled exactly like that, on a
    system where another case would find it too. True elsewhere."""
    if sys.platform not in ("win32", "darwin"):
        return True
    return _nfc(path.name) in {_nfc(n) for n in os.listdir(path.parent)}


def is_link(st: os.stat_result) -> bool:
    """Whether *st* (from ``lstat``) is a symbolic link, a junction or another
    entry that stands for a different path."""
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_reparse_tag", 0) & _NAME_SURROGATE)


# --------------------------------------------------------------- identities

def content_identity(path, wait: Optional[Wait] = None) -> dict:
    """``{"sha256", "size", "mtime_ns"}`` of the regular file at *path*. Raises
    :class:`Unsafe` for anything that is not one, :class:`Held` when it cannot be
    opened within *wait*."""
    before = os.lstat(path)
    if is_link(before) or not stat.S_ISREG(before.st_mode):
        raise Unsafe(f"{path} is not a regular file")
    found = patient(wait or Wait(), path, lambda: shared.update_package._file_entry(Path(path)))
    return {**found, "mtime_ns": before.st_mtime_ns}


def quick_identity(path) -> dict:
    """``{"size", "mtime_ns"}`` of the file at *path*, or ``{"link"}`` with its
    target for a link."""
    st = os.lstat(path)
    if is_link(st):
        return {"link": os.readlink(path)}
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def with_copy(path, identity: dict, wait: Optional[Wait] = None) -> dict:
    """*identity* (from :func:`quick_identity`) with the checksum of the file at
    *path* as ``copy``, by which :func:`same` knows the same content under another
    time. Raises like :func:`content_identity`."""
    if set(identity) != {"size", "mtime_ns"}:
        return identity
    return {**identity, "copy": content_identity(path, wait)["sha256"]}


def same(path, identity: dict, wait: Optional[Wait] = None) -> bool:
    """Whether the file at *path* has *identity*. The content decides where the
    identity states a ``sha256``, else size and time or the target of a link, and
    with a ``copy`` the content when only the time differs. False when there is no
    file. Raises :class:`Held` when it cannot be read within *wait*."""
    try:
        have = quick_identity(path)
        if "sha256" not in identity and have == {k: identity[k] for k in identity if k != "copy"}:
            return True
        checksum = identity.get("sha256") or identity.get("copy")
        if checksum is None or have.get("size") != identity["size"]:
            return False
        return content_identity(path, wait)["sha256"] == checksum
    except (FileNotFoundError, NotADirectoryError, Unsafe):
        return False


def unchanged(path, identity: dict) -> bool:
    """Whether the file at *path* still has the size and time (or the link target)
    of *identity*."""
    try:
        have = quick_identity(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return all(have.get(k) == identity[k] for k in ("size", "mtime_ns", "link") if k in identity)


# ------------------------------------------------------------------- writes

def sync_folder(folder) -> None:
    """Flush the entries of *folder* to disk (nothing to do on Windows)."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_disk() -> None:
    """Flush every pending write to disk where the system has a call for it."""
    if hasattr(os, "sync"):
        os.sync()


def _flush(path) -> None:
    fd = os.open(path, os.O_RDWR | _BINARY if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flush_files(paths: Iterable, wait: Optional[Wait] = None) -> None:
    """Flush the content of each regular file of *paths* to disk. A read-only file
    on Windows, which cannot be flushed, is left as it is. Raises :class:`Held` when
    one cannot be opened within *wait*, OSError otherwise."""
    wait = wait or Wait()

    def one(path):
        st = os.lstat(path)
        if is_link(st) or (os.name == "nt" and not st.st_mode & stat.S_IWRITE):
            return None
        try:
            _flush(path)
        except PermissionError:
            return path
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        held = [path for path in pool.map(one, paths) if path is not None]
    for path in held:
        patient(wait, path, partial(_flush, path))


def replace_through(src: Path, dst: Path) -> None:
    """Rename *src* over *dst*; the rename is on disk when this returns. Raises
    OSError."""
    if os.name == "nt":
        import ctypes
        replace_existing, write_through = 0x1, 0x8
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        if not kernel32.MoveFileExW(ctypes.c_wchar_p(str(src)), ctypes.c_wchar_p(str(dst)),
                                    replace_existing | write_through):
            code = ctypes.get_last_error()  # type: ignore[attr-defined]
            raise ctypes.WinError(code)  # type: ignore[attr-defined]
    else:
        os.replace(src, dst)
        sync_folder(dst.parent)
        if src.parent != dst.parent:
            sync_folder(src.parent)


def write_json(path: Path, obj) -> None:
    """Write *obj* to *path* so that after a power cut the file holds either its
    old content or all of the new one, and is on disk when this returns. Raises
    :class:`Held` or OSError."""
    tmp = path.with_name(path.name + ".tmp")
    wait = Wait(_REPLACE_WAIT_S)

    def write() -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
    patient(wait, tmp, write)
    patient(wait, path, lambda: replace_through(tmp, path))


def _writable(path: Path) -> None:
    st = lstat(path)
    if os.name == "nt" and st is not None and not is_link(st) and not st.st_mode & stat.S_IWRITE:
        os.chmod(path, stat.S_IWRITE)


def unlink(path: Path) -> None:
    """Remove the file at *path* if it is there. Raises :class:`Held` or OSError."""
    try:
        _writable(path)
        patient(Wait(_REPLACE_WAIT_S), path, lambda: os.unlink(path))
    except FileNotFoundError:
        return
    sync_folder(path.parent)


def remove_through(path: Path) -> None:
    """Remove the file at *path* if it is there; that it is gone is on disk when this
    returns. Raises :class:`Held` or OSError."""
    gone = path.with_name(path.name + ".gone")
    if lstat(path) is not None:
        patient(Wait(_REPLACE_WAIT_S), path, lambda: replace_through(path, gone))
    unlink(gone)


def read_file(path: Path, limit: int) -> Optional[bytes]:
    """What the regular file *path* of at most *limit* bytes holds, or None when
    there is no file. Raises ValueError when it is anything else, and
    :class:`Blocked` when it cannot be read for a while."""
    st = lstat(path)
    if st is None:
        return None
    if is_link(st) or not stat.S_ISREG(st.st_mode) or st.st_size > limit:
        raise ValueError(f"{path.name} is not a file of the update")
    deadline = time.monotonic() + READ_WAIT_S
    while True:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            if time.monotonic() >= deadline:
                raise Blocked(path, e) from e
            time.sleep(_RETRY_S)


def read_json(path: Path, limit: int):
    """The JSON value in the file *path* (:func:`read_file`), or None when there is
    no file. Raises like :func:`read_file`, ValueError also for anything but JSON."""
    data = read_file(path, limit)
    if data is None:
        return None
    try:
        return shared.update_package._strict_json(data.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"{path.name} is damaged: {e}") from None


# -------------------------------------------------------------------- locks

def open_lock(path: Path) -> int:
    """A file descriptor of the lock file *path*, made when it is not there. Raises
    :class:`Unsafe` when it is a link, OSError otherwise."""
    st = lstat(path)
    if st is not None and is_link(st):
        raise Unsafe(f"{path} is a link")
    return os.open(path, os.O_RDWR | os.O_CREAT | _BINARY | getattr(os, "O_NOFOLLOW", 0), 0o666)


def lock_file(fd: int, wait: Optional[Wait] = None) -> None:
    """Lock *fd* until it is closed. Raises PermissionError or BlockingIOError when
    another holder has it until *wait* ends."""
    while True:
        try:
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (PermissionError, BlockingIOError):
            if wait is None or time.monotonic() >= wait.until():
                raise
            time.sleep(_RETRY_S)


# -------------------------------------------------------------------- moves

def move(src: Path, dst: Path, wait: Wait, *, through: bool = False) -> None:
    """Rename the file or folder *src* to *dst*, where nothing may be yet, creating the
    folders above *dst*; to another volume it is copied, compared and then deleted.
    *through*: the rename is on disk when this returns. Raises :class:`Held`
    for a file that cannot be moved, FileExistsError when there is something at
    *dst*, OSError otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if lstat(dst) is not None and spelled(dst):
        raise FileExistsError(f"{dst} is already there")
    _rename(replace_through if through else os.rename, src, dst, wait)


def put_over(src: Path, dst: Path, wait: Wait) -> None:
    """Rename the file *src* over the file at *dst*, which is never absent in between,
    also from another volume (:func:`move`); a read-only *dst* is replaced too. Raises
    :class:`Held` or OSError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _writable(dst)
    _rename(os.replace, src, dst, wait)


def _rename(call, src: Path, dst: Path, wait: Wait) -> None:
    try:
        patient(wait, dst if call is os.replace and lstat(dst) else src, lambda: call(src, dst))
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        _move_across(src, dst, wait)


def _move_across(src: Path, dst: Path, wait: Wait) -> None:
    """Move *src* to *dst* on another volume."""
    st = os.lstat(src)
    if not stat.S_ISDIR(st.st_mode) or is_link(st):
        copy_whole(src, dst, wait)
        _compare(src, dst, wait)
        unlink(src)
        return
    part = dst.with_name(dst.name + PART)
    if not remove_tree(part):
        raise FileExistsError(f"{part} cannot be removed")
    for folder, dirs, names in os.walk(src):
        here = Path(folder)
        links = [d for d in dirs if is_link(os.lstat(here / d))]
        dirs[:] = [d for d in dirs if d not in links]
        (part / here.relative_to(src)).mkdir(parents=True, exist_ok=True)
        for name in [*names, *links]:
            copy = part / here.relative_to(src) / name
            copy_whole(here / name, copy, wait)
            _compare(here / name, copy, wait)
    patient(wait, part, lambda: replace_through(part, dst))
    if not remove_tree(src):
        raise OSError(f"{src} was copied to {dst} and cannot be deleted")


def _compare(src: Path, copy: Path, wait: Wait) -> None:
    link = is_link(os.lstat(src))
    if (os.readlink(src) != os.readlink(copy) if link else
            not same(copy, {"sha256": content_identity(src, wait)["sha256"],
                            "size": os.lstat(src).st_size}, wait)):
        raise OSError(f"{copy} is not the same as {src}")


def copy_whole(src: Path, dst: Path, wait: Wait) -> None:
    """Copy the file or link *src*, with its mode and times, to *dst*, replacing it;
    *dst* is complete and on disk as soon as it exists. Raises :class:`Held` or OSError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + PART)
    unlink(tmp)
    patient(wait, src, lambda: shutil.copyfile(src, tmp, follow_symlinks=False))
    link = is_link(os.lstat(tmp))
    if os.name != "nt":
        shutil.copystat(src, tmp, follow_symlinks=False)
    if not link:
        patient(wait, tmp, lambda: _flush(tmp))
    if os.name == "nt" and not link:   # a read-only file cannot be flushed there
        shutil.copystat(src, tmp)
    patient(wait, dst, lambda: replace_through(tmp, dst))


def clear(folder: Path, wait: Wait) -> None:
    """Remove *folder* when it is empty. Raises :class:`Held` or OSError."""
    if not os.listdir(folder):
        patient(wait, folder, lambda: os.rmdir(folder))


def prune(folder: Path, stop: Path) -> None:
    """Remove *folder* and the folders above it, up to *stop* (kept), while each
    holds nothing, or nothing but compiled Python files."""
    while folder != stop and stop in folder.parents:
        st = lstat(folder)
        if st is not None:
            try:
                if is_link(st):
                    return
                if os.listdir(folder) == [PYCACHE]:
                    remove_tree(folder / PYCACHE)
                os.rmdir(folder)
            except OSError:
                return
        folder = folder.parent


def remove_tree(path: Path) -> bool:
    """Delete the folder *path*, never following a link out of it; what cannot be
    deleted (a library a program still has loaded) stays. Returns whether it is
    gone."""
    def again(func, name, _exc) -> None:
        try:
            if os.name == "nt" and not is_link(os.lstat(name)):
                os.chmod(name, stat.S_IWRITE)
                func(name)
        except OSError:
            return

    st = lstat(path)
    if st is None:
        return True
    if is_link(st):
        return False
    shutil.rmtree(path, onexc=again)
    return lstat(path) is None
