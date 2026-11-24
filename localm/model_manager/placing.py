# SPDX-License-Identifier: AGPL-3.0-or-later
"""Giving files their names in the models folder, and taking names away.

A file gets a new name only where that name is free (:func:`put_new`), or where it
still belongs to the exact file that is to be replaced (:func:`put_over`); several
files are replaced together with a journal (:func:`place_files`). A name is deleted
only when another one was seen to hold the file (:func:`drop_name`). See
test_model_files_are_named_only_through_placing.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import secrets
import shutil
import stat as _stat
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..debuglog import logger

# Where :func:`place_files` keeps what it replaces, under the folder it works in.
REPLACED = Path(".cache") / "huggingface" / "localm-replaced"
_JOURNAL = "journal.json"


# --------------------------------------------------------------- identity

def _device_and_number(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


def file_id(path: Path) -> tuple[int, int] | None:
    """The device and file number of *path*, or None when it does not exist or its
    volume gives no usable number (zero, or the same one for the folder it is in)."""
    path = Path(path)
    try:
        number = _device_and_number(path)
        if not number[1]:
            return None
        if path.parent != path and _device_and_number(path.parent) == number:
            return None
    except (OSError, ValueError):
        return None
    return number


def same_file(a: Path, b: Path) -> bool:
    """True when *a* and *b* are one existing file or folder, under whatever names.
    On a volume without file numbers only equal resolved paths count."""
    ida, idb = file_id(a), file_id(b)
    if ida is not None and idb is not None:
        return ida == idb
    try:
        return (os.path.lexists(a) and os.path.lexists(b)
                and os.path.normcase(str(Path(a).resolve()))
                == os.path.normcase(str(Path(b).resolve())))
    except OSError:
        return False


def same_entry(a: Path, b: Path) -> bool:
    """True when *a* and *b* are the same name in the same folder, whatever path leads
    to that folder. Names that differ only in case count when they are one file."""
    if os.path.normcase(str(a)) == os.path.normcase(str(b)):
        return True
    if not same_file(a.parent, b.parent):
        return False
    if os.path.normcase(a.name) == os.path.normcase(b.name):
        return True
    return a.name.casefold() == b.name.casefold() and same_file(a, b)


def stamp(path: Path) -> tuple | None:
    """What identifies the file *path* and its content as it is now: file number,
    size and modification time. None when it does not exist."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    number = file_id(path) or (0, 0)
    if _stat.S_ISDIR(st.st_mode):
        return (*number, 0, 0)
    return (*number, st.st_size, st.st_mtime_ns)


# --------------------------------------------------------------- new names

def _exists(dest: Path) -> FileExistsError:
    return FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dest))


def _rename_exclusive(src: Path, dest: Path) -> bool:
    """Rename with the system call that refuses an existing *dest*. False when the
    system or the file system has none."""
    if sys.platform == "win32":
        os.rename(src, dest)
        return True
    name, args = (("renamex_np", (os.fsencode(src), os.fsencode(dest), 4))
                  if sys.platform == "darwin" else
                  ("renameat2", (-100, os.fsencode(src), -100, os.fsencode(dest), 1)))
    call = getattr(ctypes.CDLL(None, use_errno=True), name, None)
    if call is None:
        return False
    if call(*args) == 0:
        return True
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
        return False
    raise OSError(code, os.strerror(code), str(src), None, str(dest))


def rename_new(src: Path, dest: Path) -> None:
    """Rename *src* to *dest*, raising FileExistsError when *dest* exists."""
    try:
        if _rename_exclusive(src, dest):
            return
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise _exists(dest) from e
        raise
    if os.path.lexists(dest):
        raise _exists(dest)
    if Path(src).is_dir() and not Path(src).is_symlink():
        os.rename(src, dest)
        return
    with open(dest, "xb"):
        pass
    mine = stamp(dest)
    try:
        put_over(src, dest, mine)
    except BaseException:
        if os.path.lexists(src) and stamp(dest) == mine:
            os.unlink(dest)
        raise


def second_name(src: Path, dest: Path) -> None:
    """Make *dest* a second name (hard link) of the file *src*. Raises FileExistsError
    when *dest* exists, OSError where links do not work."""
    os.link(src, dest)


def put_new(src: Path, dest: Path) -> None:
    """Give the file *src* the name *dest* instead of its own. Never replaces: raises
    FileExistsError, with nothing changed, when *dest* exists."""
    try:
        second_name(src, dest)
    except FileExistsError:
        raise _exists(dest) from None
    except OSError as e:
        logger.debug("cannot link %s to %s (%s); renaming it", src, dest, e)
        rename_new(src, dest)
        return
    try:
        os.unlink(src)
    except OSError as e:
        logger.warning("%s is now %s, but the old name could not be removed: %s", src, dest, e)


def put_over(src: Path, dest: Path, old: tuple | None) -> None:
    """Put the file *src* in place of *dest*, which must still be the file *old* (its
    :func:`stamp`). Raises FileExistsError, with nothing changed, when it is not."""
    if old is None or stamp(dest) != old:
        raise _exists(dest)
    os.replace(src, dest)


def put(src: Path, dest: Path, old: tuple | None) -> None:
    """:func:`put_over` the file *old*, or :func:`put_new` when there was none."""
    if old is None:
        put_new(src, dest)
    else:
        put_over(src, dest, old)


def put_first_free(src: Path, names: Iterable[Path]) -> Path:
    """Give *src* the first of *names* that is free and return it. Raises
    FileExistsError when none is."""
    last: Optional[Path] = None
    for dest in names:
        last = dest
        try:
            put_new(src, dest)
            return dest
        except FileExistsError:
            continue
    raise _exists(last if last is not None else src)


# --------------------------------------------------------------- several files

def place_files(files: dict[str, Path], base: Path, old: dict[str, tuple | None]) -> None:
    """Move the files *files* (name under *base* to path) to their names, last one
    first. *old* gives the :func:`stamp` of each file to be replaced; a name without
    one must be free. When one cannot be placed, everything goes back and the error is
    raised. What is replaced is kept, with a journal, under :data:`REPLACED` until all
    are placed, so a process that is killed meanwhile is put right by
    :func:`put_back`."""
    if len(files) == 1:
        (name, src), = files.items()
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        os.utime(src)
        put(src, base / name, old.get(name))
        return
    root = base / REPLACED / secrets.token_hex(8)
    for src in files.values():
        os.utime(src)
    if any(old.get(name) is not None for name in files):
        root.mkdir(parents=True)
        (root / _JOURNAL).write_text(json.dumps({
            name: {"new": list(stamp(src) or ()), "from": str(src)}
            for name, src in files.items()}), encoding="utf-8")
    try:
        for name in reversed(list(files)):
            final = base / name
            final.parent.mkdir(parents=True, exist_ok=True)
            if old.get(name) is None:
                put_new(files[name], final)
            else:
                _replace_one(files[name], final, old[name], root / "old" / name)
    except BaseException:
        if root.is_dir():
            undo(root, base)
        else:
            for name, src in files.items():
                if not os.path.lexists(src):
                    put_new(base / name, src)
        raise
    shutil.rmtree(root, ignore_errors=True)


def _replace_one(src: Path, final: Path, old: tuple | None, aside: Path) -> None:
    if stamp(final) != old:
        raise _exists(final)
    aside.parent.mkdir(parents=True, exist_ok=True)
    try:
        second_name(final, aside)
    except OSError as e:
        logger.debug("cannot link %s (%s); renaming it", final, e)
        rename_new(final, aside)
        put_new(src, final)
        return
    if not same_file(final, aside):
        raise _exists(final)
    os.replace(src, final)


def undo(root: Path, base: Path) -> bool:
    """Put back what an unfinished :func:`place_files` in *base* replaced, as the
    journal in *root* says, and give the new files their earlier names. Returns False,
    with *root* kept, when a file could not be put back."""
    try:
        journal = json.loads((root / _JOURNAL).read_text(encoding="utf-8"))
        rows = {str(n): (tuple(v["new"]), Path(v["from"])) for n, v in journal.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
        logger.warning("cannot read %s (%s); %s is left as it is", root / _JOURNAL, e, root)
        return not any(p.is_file() for p in (root / "old").rglob("*"))
    whole = True
    for name, (new, origin) in rows.items():
        final, aside = base / name, root / "old" / name
        try:
            _undo_one(final, aside, new, origin)
        except OSError as e:
            logger.error("could not put %s back to %s: %s", aside, final, e)
            whole = False
    if whole:
        shutil.rmtree(root, ignore_errors=True)
    return whole


def _undo_one(final: Path, aside: Path, new: tuple, origin: Path) -> None:
    if os.path.lexists(final) and stamp(final) == new:
        origin.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(aside):
            put_new(final, origin)
            return
        try:
            second_name(final, origin)
        except FileExistsError:
            pass
        except OSError:
            put_new(final, origin)
    if not os.path.lexists(aside):
        return
    if not os.path.lexists(final):
        put_new(aside, final)
    elif same_file(aside, final):
        os.unlink(aside)
    elif stamp(final) == new:
        put_over(aside, final, new)
    else:
        raise _exists(final)


def unfinished(base: Path) -> list[tuple[Path, list[str]]]:
    """The journals of unfinished :func:`place_files` calls in *base*, as ``(root, file
    names)``; an unreadable one has no names."""
    out = []
    try:
        roots = sorted(p for p in (base / REPLACED).iterdir() if p.is_dir())
    except OSError:
        return []
    for root in roots:
        try:
            names = [str(n) for n in json.loads((root / _JOURNAL).read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError, RecursionError):
            names = []
        out.append((root, names))
    return out


def put_back(base: Path, names: Iterable[str] | None = None) -> bool:
    """:func:`undo` every unfinished replacement in *base* (with *names*: those that
    touch one of them). The caller holds the locks of these names, so none is running.
    Returns False when something could not be put back."""
    wanted = None if names is None else set(names)
    return all([undo(root, base) for root, files in unfinished(base)
                if wanted is None or wanted & set(files)])


# --------------------------------------------------------------- deleting a name

def drop_name(victim: Path, keeper: Path, kept: Callable[[Path], bool],
              records: Optional[Path] = None, *, copy: bool = False) -> None:
    """Delete the name *victim* of a file, when *kept* (given the name *victim* has
    meanwhile) shows that *keeper* still holds the file without it. *victim* is
    renamed first, so when both paths reach one name, *keeper* is gone at that moment
    and *victim* is renamed back. A read-only file is deleted too, and *keeper* keeps
    its attributes. Raises OSError with *victim* left.

    With *records* a note of it is kept in that folder meanwhile, so
    :func:`finish_dropped` completes or undoes it when the process is killed. *copy*
    says *keeper* is a copy, not another name of the file."""
    temp = victim.with_name(f".localm-moving-{victim.name[:80]}.{secrets.token_hex(4)}")
    note = _note_drop(records, victim, temp, keeper, copy) if records is not None else None
    try:
        rename_new(victim, temp)
        try:
            if not kept(temp):
                raise OSError(f"{keeper} does not hold the file")
            _unlink_read_only_too(temp, keeper)
        except BaseException:
            if os.path.lexists(temp) and not os.path.lexists(victim):
                rename_new(temp, victim)
            raise
    finally:
        if note is not None:
            if not os.path.lexists(temp):
                note.unlink(missing_ok=True)
            _ACTIVE.discard(note.name)


# The notes of drops this process is in the middle of.
_ACTIVE: set[str] = set()


def _unlink_read_only_too(path: Path, keeper: Path) -> None:
    try:
        path.unlink()
        return
    except PermissionError:
        mode = path.stat().st_mode
        if mode & _stat.S_IWRITE:
            raise
    shared = same_file(path, keeper)
    os.chmod(path, mode | _stat.S_IWRITE)
    try:
        path.unlink()
    finally:
        if os.path.lexists(path):
            os.chmod(path, mode)
        elif shared:
            os.chmod(keeper, mode)


def _note_drop(records: Path, victim: Path, temp: Path, keeper: Path, copy: bool) -> Path:
    from localm.instances import process_start_identity
    records.mkdir(parents=True, exist_ok=True)
    note = records / f"{secrets.token_hex(8)}.json"
    _ACTIVE.add(note.name)
    note.write_text(json.dumps({
        "victim": str(victim.absolute()), "temp": str(temp.absolute()),
        "keeper": str(keeper.absolute()), "copy": copy,
        "keeper_stamp": list(stamp(keeper) or ()), "pid": os.getpid(),
        "start": process_start_identity(os.getpid())}), encoding="utf-8")
    return note


def finish_dropped(records: Path) -> None:
    """Complete or undo every :func:`drop_name` of a process that is gone, by the notes
    in *records*: the renamed file is deleted when its keeper still holds it, else gets
    its name back. One that can be neither is left, with its note. Never raises."""
    try:
        notes = sorted(records.glob("*.json"))
    except OSError:
        return
    for note in notes:
        if note.name in _ACTIVE:
            continue
        try:
            rec = json.loads(note.read_text(encoding="utf-8"))
            victim, temp, keeper = (Path(rec[k]) for k in ("victim", "temp", "keeper"))
            if not _owner_gone(rec):
                continue
            if os.path.lexists(temp):
                held = (stamp(keeper) is not None
                        and list(stamp(keeper) or ()) == rec.get("keeper_stamp")
                        and not same_file(temp, keeper)) if rec.get("copy") else (
                            same_file(temp, keeper))
                if held:
                    _unlink_read_only_too(temp, keeper)
                else:
                    rename_new(temp, victim)
                logger.info("finished an interrupted move of %s", victim)
            elif not temp.parent.is_dir():
                continue
            note.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
            logger.warning("could not finish the interrupted move noted in %s: %s", note, e)


def _owner_gone(rec: dict) -> bool:
    from localm.instances import pid_alive, process_start_identity
    pid = int(rec["pid"])
    if pid == os.getpid():
        return True
    if not pid_alive(pid):
        return True
    start = process_start_identity(pid)
    return start is not None and rec.get("start") is not None and start != rec["start"]
