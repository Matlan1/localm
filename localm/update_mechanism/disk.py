# SPDX-License-Identifier: AGPL-3.0-or-later
"""File identities, paths that cannot leave a folder, and writes that are on disk
when the call returns."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Optional

from . import shared

#: Seconds the moves of one run wait, in all, for files another program holds.
HELD_WAIT_S = 30.0
_RETRY_S = 0.1
# Seconds a write waits for a reader of the same file to close it.
_REPLACE_WAIT_S = 10.0
_NAME_SURROGATE = 0x20000000
PYCACHE = "__pycache__"


class Unsafe(Exception):
    """A path that is not one of this install's own files."""


class Held(Exception):
    """A file that another program holds, or that may not be moved.

    *path* is the file; ``str()`` names it and, where the system says so, the
    programs that hold it."""

    def __init__(self, path, reason) -> None:
        self.path = str(path)
        super().__init__(f"{path} cannot be moved ({reason}){_holders(path)}")


def _holders(path) -> str:
    if sys.platform != "win32":
        return ""
    try:
        found = shared.apply_update.rm_holders([path])
    except OSError:
        return ""
    if not found:
        return ""
    return ", held by " + ", ".join(f"{h['name']} (pid {h['pid']})" for h in found)


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

def content_identity(path) -> dict:
    """``{"sha256", "size", "mtime_ns"}`` of the regular file at *path*. Raises
    :class:`Unsafe` for anything that is not one."""
    before = os.lstat(path)
    if is_link(before) or not stat.S_ISREG(before.st_mode):
        raise Unsafe(f"{path} is not a regular file")
    return {**shared.update_package._file_entry(Path(path)), "mtime_ns": before.st_mtime_ns}


def quick_identity(path) -> dict:
    """``{"size", "mtime_ns"}`` of the file at *path*, or ``{"link"}`` with its
    target for a link."""
    st = os.lstat(path)
    if is_link(st):
        return {"link": os.readlink(path)}
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def same(path, identity: dict) -> bool:
    """Whether the file at *path* has *identity*. The content decides where the
    identity states a checksum, else size and time, or the target of a link. False
    when there is no file."""
    try:
        have = quick_identity(path)
        if "sha256" not in identity:
            return have == identity
        if have.get("size") != identity["size"]:
            return False
        return content_identity(path)["sha256"] == identity["sha256"]
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
    fd = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flush_files(paths: Iterable) -> None:
    """Flush the content of each regular file of *paths* to disk. A read-only file
    on Windows, which cannot be flushed, is left as it is. Raises OSError."""
    def one(path) -> None:
        st = os.lstat(path)
        if is_link(st) or (os.name == "nt" and not st.st_mode & stat.S_IWRITE):
            return
        _flush(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(one, paths):
            pass


def _until(deadline: Callable[[], float], action: Callable[[], object]) -> None:
    """Run *action*, on Windows again until the time *deadline* returns while the
    file it needs is held."""
    while True:
        try:
            action()
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline():
                raise
            time.sleep(_RETRY_S)


def write_json(path: Path, obj) -> None:
    """Write *obj* to *path* so that after a power cut the file holds either its
    old content or all of the new one, and is on disk when this returns. Raises
    OSError."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    _until(Wait(_REPLACE_WAIT_S).until, lambda: os.replace(tmp, path))
    sync_folder(path.parent)


def unlink(path: Path) -> None:
    """Remove the file at *path* if it is there. Raises OSError."""
    try:
        _until(Wait(_REPLACE_WAIT_S).until, lambda: os.unlink(path))
    except FileNotFoundError:
        return
    sync_folder(path.parent)


def read_json(path: Path, limit: int):
    """The JSON value in the regular file *path* of at most *limit* bytes, or None
    when there is no file. Raises ValueError when it is anything else."""
    st = lstat(path)
    if st is None:
        return None
    if is_link(st) or not stat.S_ISREG(st.st_mode) or st.st_size > limit:
        raise ValueError(f"{path.name} is not a file of the update")
    try:
        return shared.update_package._strict_json(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{path.name} cannot be read: {e}") from None


# -------------------------------------------------------------------- locks

def open_lock(path: Path) -> int:
    """A file descriptor of the lock file *path*, made when it is not there. Raises
    :class:`Unsafe` when it is a link, OSError otherwise."""
    st = lstat(path)
    if st is not None and is_link(st):
        raise Unsafe(f"{path} is a link")
    return os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o666)


def lock_file(fd: int) -> None:
    """Lock *fd* until it is closed. Raises PermissionError or BlockingIOError when
    another holder has it."""
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


# -------------------------------------------------------------------- moves

class Wait:
    """How long the moves of one run wait, in all, for held files."""

    def __init__(self, seconds: Optional[float] = None) -> None:
        self.seconds = HELD_WAIT_S if seconds is None else seconds
        self._deadline: Optional[float] = None

    def until(self) -> float:
        """The time (``time.monotonic``) at which waiting ends; the first call
        starts the wait."""
        if self._deadline is None:
            self._deadline = time.monotonic() + self.seconds
        return self._deadline


def move(src: Path, dst: Path, wait: Wait) -> None:
    """Rename *src* to *dst*, where nothing may be yet, creating the folders above
    *dst*. Raises :class:`Held` for a file that cannot be moved, FileExistsError
    when there is something at *dst*, OSError otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if lstat(dst) is not None and spelled(dst):
        raise FileExistsError(f"{dst} is already there")
    _rename(os.rename, src, dst, wait)


def put_over(src: Path, dst: Path, wait: Wait) -> None:
    """Rename *src* over the file at *dst*, which is never absent in between.
    Raises :class:`Held` or OSError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _rename(os.replace, src, dst, wait)


def _rename(call, src: Path, dst: Path, wait: Wait) -> None:
    try:
        _until(wait.until, lambda: call(src, dst))
    except PermissionError as e:
        raise Held(dst if call is os.replace and lstat(dst) else src, e.strerror or e) from e


def copy_whole(src: Path, dst: Path, wait: Wait) -> None:
    """Copy *src*, with its mode and times, to *dst*, replacing it; *dst* is
    complete and on disk as soon as it exists. Raises :class:`Held` or OSError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        _until(wait.until, lambda: shutil.copy2(src, tmp, follow_symlinks=False))
    except PermissionError as e:
        raise Held(src, e.strerror or e) from e
    flush_files([tmp])
    os.replace(tmp, dst)
    sync_folder(dst.parent)


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
