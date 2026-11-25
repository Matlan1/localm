# SPDX-License-Identifier: AGPL-3.0-or-later
"""The small files of the data folder an older localm may not read once a newer one
wrote them, and its plugins: copied before the new version starts, and put back when
the update is undone and they changed.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import disk, mover
from .mover import Refused, Unreadable

#: The files and folders that are copied, of ``updater._DOWNGRADE_DATA``.
FILES = ("config.json", "registry.json", "coder-hook-trust.json", "auth.json",
         "auth.kdf.json", "sessions.json")
FOLDERS = ("plugins",)
_FOLDER, _LIST = "data", "data.json"
_MAX_LIST = 4 * 1024 * 1024
#: A file larger than this is not copied, nor a folder with more than these.
MAX_BYTES = 16 * 1024 * 1024
MAX_FOLDER_BYTES, MAX_FOLDER_FILES = 64 * 1024 * 1024, 5000


def _located(root, home) -> str:
    whole, inner = disk.real_folder(root), disk.real_folder(home)
    if whole in inner.parents:
        return inner.relative_to(whole).as_posix()
    return disk.short_path(inner)


def _regular(path: Path):
    st = disk.lstat(path)
    return None if st is None or disk.is_link(st) or not os.path.isfile(path) else st


def _files_in(folder: Path) -> list:
    """Every regular file below *folder*, as POSIX paths, compiled Python files
    excepted."""
    found = []
    for here, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d != disk.PYCACHE
                   and not disk.is_link(os.lstat(Path(here) / d))]
        found += [(Path(here) / n).relative_to(folder).as_posix() for n in names
                  if _regular(Path(here) / n) is not None]
    return sorted(found)


def _wanted(home: Path) -> tuple:
    """``({path in the data folder: whether it is there}, [folders listed whole])``."""
    files = {name: _regular(home / name) is not None for name in FILES
             if (st := _regular(home / name)) is None or st.st_size <= MAX_BYTES}
    whole = []
    for folder in FOLDERS:
        inner = _files_in(home / folder)
        if (len(inner) <= MAX_FOLDER_FILES
                and sum(os.lstat(home / folder / f).st_size for f in inner) <= MAX_FOLDER_BYTES):
            whole.append(folder)
            files.update({f"{folder}/{f}": True for f in inner})
    return files, whole


def keep(root, update_id: str, home) -> None:
    """Copy :data:`FILES` and :data:`FOLDERS` of the data folder *home* into the folder
    of the update *update_id*, readable by the user alone, and list them there; the list
    is written last. Raises OSError."""
    work, home = mover.work_folder(root, update_id), disk.real_folder(home)
    folder = work / _FOLDER
    if not disk.remove_tree(folder):
        raise OSError(f"{folder} cannot be removed")
    os.mkdir(folder, 0o700)
    files: dict = {}
    modes: dict = {}
    names, whole = _wanted(home)
    for name, there in names.items():
        files[name] = None
        if there:
            copy = folder / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            modes[name] = os.lstat(home / name).st_mode & 0o777
            _write_over(home / name, copy, 0o600)
            files[name] = disk.content_identity(copy)
    disk.flush_files([folder / name for name, identity in files.items() if identity])
    disk.write_json(work / _LIST, {"home": _located(root, home), "files": files,
                                   "modes": modes, "folders": whole})


def _allowed(name) -> bool:
    return name in FILES or disk.clean(name).split("/", 1)[0] in FOLDERS


def _read(root, update_id: str):
    try:
        raw = disk.read_json(mover.work_folder(root, update_id) / _LIST, _MAX_LIST)
    except ValueError as e:
        raise Unreadable(str(e)) from None
    if raw is None:
        return None
    try:
        ok = (isinstance(raw, dict) and isinstance(raw.get("home"), str)
              and isinstance(raw.get("files"), dict) and all(map(_allowed, raw["files"]))
              and isinstance(raw.get("modes"), dict)
              and all(type(m) is int and 0 <= m <= 0o777 for m in raw["modes"].values())
              and isinstance(raw.get("folders"), list) and set(raw["folders"]) <= set(FOLDERS))
    except (disk.Unsafe, TypeError):
        ok = False
    if not ok:
        raise Unreadable(f"{_LIST} does not list the copied data files")
    try:
        identities = {n: mover._identity(i) for n, i in raw["files"].items()}
    except (ValueError, TypeError, AttributeError) as e:
        raise Unreadable(f"{_LIST} does not list the copied data files: {e}") from None
    home = disk.real_folder(Path(root) / raw["home"])
    return home, identities, raw["modes"], raw["folders"]


def back(root, update_id: str) -> list:
    """Put the files :func:`keep` copied back where they changed since: the changed one,
    and one that came since, go to ``kept/<id>/data`` first. Does nothing when nothing
    was copied or the data folder is not there. Safe to repeat and to interrupt. Returns
    one line per file that could not be put back. Raises :class:`Unreadable`."""
    kept = _read(root, update_id)
    if kept is None or disk.lstat(kept[0]) is None:
        return []
    home, files, modes, whole = kept
    folder = mover.work_folder(root, update_id) / _FOLDER
    for name in whole:
        files.update({f"{name}/{f}": None for f in _files_in(home / name)
                      if f"{name}/{f}" not in files})
    errors = []
    for name, identity in files.items():
        live = home / name
        try:
            if identity is None:
                if _regular(live) is not None:
                    disk.move(live, _aside(root, update_id, name), disk.Wait())
                    disk.prune(live.parent, home)
            elif not disk.same(live, identity):
                if not disk.same(folder / name, identity):
                    raise Refused(f"the copy of {name} is damaged")
                if _regular(live) is not None:
                    disk.copy_whole(live, _aside(root, update_id, name), disk.Wait())
                live.parent.mkdir(parents=True, exist_ok=True)
                _write_over(folder / name, live, modes.get(name, 0o600))
                if not disk.same(live, identity):
                    raise Refused(f"{live} is not what it was before the update")
        except (OSError, disk.Held, disk.Unsafe, Refused) as e:
            errors.append(f"{name} of the data folder: {e}")
    return errors


def _aside(root, update_id: str, name: str) -> Path:
    folder = mover.update_folder(root) / "kept"
    for part in (update_id, _FOLDER):
        folder = mover.plain_folder(folder) / part
    folder.parent.mkdir(parents=True, exist_ok=True)
    if disk.lstat(folder) is None:
        os.mkdir(folder, 0o700)
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    target, n = folder / name, 0
    while disk.lstat(target) is not None:
        n += 1
        target = folder / f"{name}.{n}"
    return target


def _same_access(source: Path, target: Path) -> None:
    """On Windows, give the file *target* the permissions of *source*, where the file
    system has any; those of its folder apply until then, and when that fails."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    which, protected, inherits = 0x4, 0x80000000, 0x20000000
    size = wintypes.DWORD()
    api.GetFileSecurityW(str(source), which, None, 0, ctypes.byref(size))
    found = ctypes.create_string_buffer(size.value or 1)
    if size.value and api.GetFileSecurityW(str(source), which, found, size, ctypes.byref(size)):
        control = int.from_bytes(found.raw[2:4], "little")
        api.SetFileSecurityW(str(target), wintypes.DWORD(
            which | (protected if control & 0x1000 else inherits)), found)


def _write_over(source: Path, live: Path, mode: int) -> None:
    """Write the content of *source* into the file *live*, which keeps its permissions;
    it is made with *mode*, on Windows with the permissions of *source*, when it is not
    there."""
    data, there = source.read_bytes(), disk.lstat(live) is not None
    fd = os.open(live, os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not there:
            _same_access(source, live)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.ftruncate(fd, len(data))
        if not there and os.name != "nt":
            os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)
    st = source.stat()
    os.utime(live, ns=(st.st_atime_ns, st.st_mtime_ns))
