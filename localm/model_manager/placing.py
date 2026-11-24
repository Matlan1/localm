# SPDX-License-Identifier: AGPL-3.0-or-later
"""Giving files their names in the models folder, and taking names away.

A file gets a new name only where that name is free (:func:`put_new`), or where it
still belongs to the exact file that is to be replaced (:func:`put_over`); several
files are replaced together (:func:`place_files`). A name is deleted only when another
one was seen to hold the file (:func:`drop_name`). See
test_model_files_are_named_only_through_placing.

What an interrupted step needs to be put right is recorded in a folder of the data
directory. Nothing below the models folder is read for that: a model can ship any file.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import secrets
import stat as _stat
import sys
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from ..debuglog import logger

# Under a destination folder: where a single file of a repository, or a whole one, is
# downloaded to and verified before its files get their names.
FILE_STAGING = Path(".cache") / "localm-staging"
REPO_STAGING = Path(".cache") / "huggingface" / "download" / "localm-staging"


# --------------------------------------------------------------- reserved names

def reserved(rel: str) -> bool:
    """True when *rel*, a ``/``-separated path below a model folder, is where localm or
    huggingface_hub keep data of their own: in ``.cache``, or under a name starting with
    ``.localm-``. A file a model ships there is never downloaded or copied in."""
    parts = [p.rstrip(". ") for p in rel.replace("\\", "/").casefold().split("/")]
    return parts[0] == ".cache" or any(p.startswith(".localm-") for p in parts)


def reserved_in_models(name: str) -> bool:
    """True when a file *name* directly in the models folder would be taken for data of
    localm: a :func:`reserved` one, a partial download or its record, or the record of
    where a projector came from."""
    return reserved(name) or name.casefold().endswith(
        (".part", ".part.json", ".incomplete", ".owner", ".origin.json"))


_SAID: set[str] = set()


def _say_once(text: str) -> None:
    if text not in _SAID:
        _SAID.add(text)
        logger.warning("%s", text)


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
    if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
        return False
    raise OSError(code, os.strerror(code), str(src), None, str(dest))


def put_new(src: Path, dest: Path) -> None:
    """Give the file or folder *src* the name *dest* instead of its own. Raises
    FileExistsError, with nothing changed, when *dest* exists. Where no system call
    renames without replacing, a file is linked and its old name removed; where links do
    not work either, *dest* is looked at just before the rename.

    Returns only when the old name is gone. An old name that cannot be removed is raised
    as the error, and the new one is taken away again."""
    try:
        if _rename_exclusive(src, dest):
            return
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise _exists(dest) from e
        raise
    if os.path.lexists(dest):
        raise _exists(dest)
    if os.path.islink(src) or os.path.isdir(src):
        os.rename(src, dest)
        return
    try:
        second_name(src, dest)
    except FileExistsError:
        raise _exists(dest) from None
    except OSError as e:
        logger.debug("cannot link %s to %s (%s); renaming it", src, dest, e)
        os.rename(src, dest)
        return
    try:
        os.unlink(src)
    except BaseException:
        if two_names(src, dest):
            os.unlink(dest)
        raise


def second_name(src: Path, dest: Path) -> None:
    """Make *dest* a second name (hard link) of the file *src*. Raises FileExistsError
    when *dest* exists, OSError where links do not work."""
    os.link(src, dest)


def two_names(a: Path, b: Path) -> bool:
    """True when *a* and *b* are names of one regular file (no symbolic link)."""
    try:
        at = os.lstat(a), os.lstat(b)
    except (OSError, ValueError):
        return False
    return (all(_stat.S_ISREG(st.st_mode) for st in at) and os.path.samestat(*at)
            and at[0].st_nlink >= 2 and file_id(a) is not None)


def drop_second_name(temp: Path, dest: Path) -> None:
    """Remove the temporary name *temp* when it and *dest* are names of one file, as a
    :func:`put_new` that was cut off leaves it where it has to link."""
    if two_names(temp, dest):
        try:
            os.unlink(temp)
        except OSError as e:
            logger.debug("could not remove %s, another name of %s: %s", temp, dest, e)


def only_name(path: Path) -> bool:
    """True when *path* is a regular file that has no other name. No other partial
    download is opened for writing."""
    try:
        st = os.lstat(path)
    except (OSError, ValueError):
        return False
    return _stat.S_ISREG(st.st_mode) and st.st_nlink <= 1


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

_PLACING, _DONE = "placing", "done"


class NotPutBack(OSError):
    """Files could not be placed, and what they replaced could not be put back either.
    The old files are kept, and :func:`put_back` tries again."""


class Unfinished(NamedTuple):
    """The record of a :func:`place_files` call that runs or was cut off: the folder in
    the models folder it works in (``""``: the models folder itself; None: the record
    cannot be read now), the names it places there, whether it is still placing, the
    process that places them (None: unknown, or it has given up) and the record's text."""
    record: Path
    folder: Optional[str]
    names: list[str]
    placing: bool
    owner: Optional[dict] = None
    text: str = ""


# The tokens of the place_files calls that run in this process.
_RUNNING: set[str] = set()


def place_files(files: dict[str, Path], base: Path, old: dict[str, tuple | None],
                models: Optional[Path] = None, records: Optional[Path] = None) -> None:
    """Move the files *files* (name under *base* to path) to their names, last one
    first. *old* gives the :func:`stamp` of each file to be replaced; a name without
    one must be free.

    With *records* (*base* is the models folder *models* or a folder in it, and every
    path is in a staging folder of *base*) they are placed together: what is replaced is
    moved aside, and a record in the folder *records*, changed in one step from
    "placing" to "done", says what to put back. When a file cannot be placed everything
    goes back and the error is raised, or :class:`NotPutBack`; a process that is killed
    meanwhile is put right by :func:`put_back`. Without *records* each file is replaced
    on its own."""
    for src in files.values():
        os.utime(src)
    if records is None or models is None or len(files) == 1 or not any(
            old.get(name) is not None for name in files):
        _place_each(files, base, old)
        return
    from localm import instances

    from ..storekit import atomic_write
    rows = [{"name": name, "had": old.get(name) is not None, "from": _below(src, base),
             "new": list(_size_and_time(src) or ())} for name, src in files.items()]
    folder = _below(base, models)
    if _parse(models, folder, rows) is None:
        raise OSError(errno.EINVAL, "not files localm can replace together", str(base))
    token = secrets.token_hex(16)
    kept = _kept(base, token)
    record = records / f"{token}.json"

    records.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "space": instances.pid_space_id(),
             "host": instances.host_id(), "local": instances.path_is_local(records),
             "start": instances.process_start_identity(os.getpid())}

    def _write(phase: str, owner: Optional[dict]) -> None:
        atomic_write(record, json.dumps({"phase": phase, "folder": folder, "files": rows,
                                         "id": (file_id(base) or (0, None))[1],
                                         "owner": owner}))

    def _settle() -> bool:
        try:
            text = _read(record)
            if text is None or _recover(record, text, models):
                return True
            if json.loads(text)["phase"] == _PLACING:
                _write(_PLACING, None)
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            logger.warning("could not settle %s: %s", record, e)
        return False

    _RUNNING.add(token)
    try:
        _write(_PLACING, owner)
        os.mkdir(kept)
        for i, name in reversed(list(enumerate(files))):
            final, was = base / name, old.get(name)
            final.parent.mkdir(parents=True, exist_ok=True)
            if was is not None:
                at = os.lstat(final)
                if stamp(final) != was:
                    raise _exists(final)
                put_new(final, kept / str(i))
                moved = os.lstat(kept / str(i))
                if (moved.st_size, moved.st_mtime_ns) != (at.st_size, at.st_mtime_ns):
                    raise _exists(final)
            put_new(files[name], final)
        _write(_DONE, owner)
    except OSError as e:
        if not _settle():
            raise NotPutBack(f"{e}. The files that were there are kept in {kept} and go back "
                             "with the next scan or pull that can move them") from e
        raise
    except BaseException:
        _settle()
        raise
    else:
        _settle()
    finally:
        _RUNNING.discard(token)


def _place_each(files: dict[str, Path], base: Path, old: dict[str, tuple | None]) -> None:
    added: list[str] = []
    try:
        for name in reversed(list(files)):
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            put(files[name], base / name, old.get(name))
            if old.get(name) is None:
                added.append(name)
    except BaseException:
        for name in added:
            try:
                put_new(base / name, files[name])
            except OSError as e:
                logger.warning("could not move %s back to %s: %s", base / name, files[name], e)
        raise


def _size_and_time(path: Path) -> tuple | None:
    now = stamp(path)
    return None if now is None else now[2:]


def _kept(base: Path, token: str) -> Path:
    return base / f".localm-old-{token}"


def _below(path: Path, base: Path) -> Optional[str]:
    """*path* relative to the folder *base*, ``/``-separated and ``""`` for *base*
    itself, or None when it is not below it."""
    try:
        rel = os.path.relpath(*(str(p).removeprefix("\\\\?\\") for p in (path, base)))
    except ValueError:
        return None
    parts = Path(rel).parts
    return None if ".." in parts or os.path.isabs(rel) else "/".join(parts)


def _inside(base: Path, name: object) -> Optional[Path]:
    """The file *name* (``/``-separated) below *base*, or None when it is not a plain
    relative name, or its folder is not in *base* once links are followed."""
    if not isinstance(name, str) or re.search(r"[\\:\0]", name) or any(
            p in ("", ".", "..") for p in name.split("/")):
        return None
    path = base.joinpath(*name.split("/"))
    return path if path.parent.resolve().is_relative_to(base.resolve()) else None


def _parse(models: Path, folder: object, rows: object) -> Optional[tuple]:
    """``(base, rows)`` of a record, each row ``(final, staged, had, new)``, or None when
    a part of it is not what :func:`place_files` writes: the folder is the models folder
    or directly in it, every name is inside that, and every staged file is in one of its
    staging folders."""
    from .pending_pulls import _as_extended_path
    if not isinstance(folder, str) or "/" in folder or not isinstance(rows, list):
        return None
    base = _inside(models, folder) if folder else models
    if base is None:
        return None
    out = []
    for row in rows:
        final, staged = _inside(base, row["name"]), _inside(base, row["from"])
        if (final is None or staged is None or reserved(row["name"])
                or not any(staged.is_relative_to(base / d) for d in (FILE_STAGING, REPO_STAGING))
                or len(row["new"]) != 2 or not isinstance(row["had"], bool)):
            return None
        out.append((final, _as_extended_path(staged, 255), row["had"], tuple(row["new"])))
    return base, out


def _read(record: Path) -> Optional[str]:
    """The text of *record*, or None when it is gone. A moment in which Windows refuses
    it, because it is being replaced, is waited out; raises OSError when it stays
    unreadable."""
    from .. import storekit
    for attempt in range(storekit._REPLACE_RETRIES):
        try:
            return record.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError as e:
            if (attempt == storekit._REPLACE_RETRIES - 1
                    or not storekit._is_transient_permission_error(e)):
                raise
            storekit._transient_backoff(attempt)
    return None


def _recover(record: Path, text: str, models: Path) -> bool:
    """Put right what the :func:`place_files` call of *record* left, by its content
    *text*: while it was placing, every old file gets its name back and the new files go
    back to staging; when it was done, the old files are deleted. Returns False when an
    old file could not be put back, or the record is not *text* any more; then nothing is
    deleted and the record stays."""
    try:
        rec = json.loads(text)
        parsed = _parse(models, rec["folder"], rec["files"])
        if parsed is None or rec["phase"] not in (_PLACING, _DONE):
            raise ValueError("it is not a record localm writes")
        now = _read(record)
        if now != text:
            return now is None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
        _say_unusable(record, e)
        return False
    (base, rows), done = parsed, rec["phase"] == _DONE
    if os.path.lexists(base) and not base.is_dir() or not models.is_dir():
        _say_once(f"cannot put back the files noted in {record}: {base} cannot be reached")
        return False
    kept = _kept(base, record.stem)
    if not os.path.lexists(kept) and (not base.is_dir() or rec.get("id") not in (
            None, (file_id(base) or (0, None))[1])):
        _say_once(f"the folder {base} was moved or deleted while its files were being replaced; "
                  f"the earlier files are in {kept.name} in that folder. Delete {record} when "
                  "they are not needed")
        return True
    whole = True
    for i, (final, staged, had, new) in enumerate(rows):
        aside = kept / str(i)
        try:
            if done:
                if os.path.lexists(aside):
                    _unlink_read_only_too(aside, final)
            else:
                _undo_one(final, aside, staged, had, new)
        except OSError as e:
            whole = False
            if not done:
                _say_once(f"{final} was being replaced when that was interrupted, and could not "
                          f"be put back ({e}). Every file was kept"
                          + (f", the earlier one is {aside}" if had else "") +
                          ". Close what uses these files, or move a file away that was put there "
                          "since")
    if whole:
        try:
            if os.path.lexists(kept):
                os.rmdir(kept)
            record.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("could not remove %s or %s yet: %s", kept, record, e)
    return whole or done


def _say_unusable(record: Path, e: Exception) -> None:
    _say_once(f"cannot use {record}, the record of an interrupted replacement ({e}). Nothing "
              f"was changed; kept files are in folders named .localm-old-{record.stem}")


def _undo_one(final: Path, aside: Path, staged: Path, had: bool, new: tuple) -> None:
    """Put back one file of a replacement that was cut off; every step leaves a state
    this goes on from. Raises OSError, with nothing deleted, when it cannot."""
    if had and not os.path.lexists(aside):
        return
    if not had and _size_and_time(final) != new:
        return
    if had and two_names(aside, final):
        os.unlink(aside)
        return
    if os.path.lexists(final):
        if _size_and_time(final) != new:
            raise _exists(final)
        if two_names(final, staged):
            os.unlink(final)
        else:
            staged.parent.mkdir(parents=True, exist_ok=True)
            put_new(final, staged)
    if had:
        put_new(aside, final)


def unfinished(records: Path) -> list[Unfinished]:
    """The records in the folder *records*; see :class:`Unfinished`."""
    out = []
    try:
        paths = sorted(p for p in records.glob("*.json") if re.fullmatch("[0-9a-f]{32}", p.stem))
    except OSError:
        return []
    for path in paths:
        try:
            text = _read(path)
            if text is None:
                continue
            rec = json.loads(text)
            out.append(Unfinished(path, str(rec["folder"]), [str(f["name"]) for f in rec["files"]],
                                  rec["phase"] == _PLACING, rec.get("owner"), text))
        except OSError:
            out.append(Unfinished(path, None, [], True))
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
            _say_unusable(path, e)
    return out


def put_back(models: Path, records: Path, folder: str,
             names: Optional[Iterable[str]] = None) -> bool:
    """Put right every :func:`place_files` call in *folder* (see :class:`Unfinished`)
    that was cut off; with *names*, those that place one of them. The caller holds the
    locks of these names. Returns False when an old file could not be put back, when such
    a call still runs, or when a record cannot be read, so it is unknown whose it is; then
    that record's files are untouched, and the caller places and deletes nothing there."""
    found = unfinished(records)
    if any(u.folder is None for u in found):
        _say_once(f"a record of an interrupted replacement in {records} cannot be read")
        return False
    return all([not u.placing if _runs(u) else _recover(u.record, u.text, models)
                for u in matching(found, folder, names)])


def matching(found: list[Unfinished], folder: str,
             names: Optional[Iterable[str]] = None) -> list[Unfinished]:
    """Those of *found* in *folder*; with *names*, those that place one of them."""
    wanted = None if names is None else {os.path.normcase(n) for n in names}
    return [u for u in found if u.folder is not None and u.folder.casefold() == folder.casefold()
            and (wanted is None or wanted & {os.path.normcase(n) for n in u.names})]


def _runs(u: Unfinished) -> bool:
    """True when the :func:`place_files` call of *u* runs in this process, or in another
    one that its record names and that is alive."""
    from localm import instances

    from .pull import _same_pid_space
    if u.record.stem in _RUNNING:
        return True
    if not isinstance(u.owner, dict) or not _same_pid_space(u.owner, u.record.parent):
        return False
    pid = u.owner.get("pid")
    return (isinstance(pid, int) and pid > 0 and pid != os.getpid() and instances.pid_alive(pid)
            and not instances.start_identity_differs(
                u.owner.get("start"), instances.process_start_identity(pid)))


def being_replaced(models: Path, records: Path) -> set[str]:
    """The paths (``os.path.normcase``) whose file a replacement that runs or was cut off
    may have moved aside."""
    return {os.path.normcase(str(base / n)) for u in unfinished(records) if u.placing
            for base in [models / (u.folder or "")]
            if os.path.lexists(_kept(base, u.record.stem)) for n in u.names}


# --------------------------------------------------------------- deleting a name

def drop_name(victim: Path, keeper: Path, *, copy: bool = False) -> None:
    """Delete the name *victim* of a file that *keeper* holds as well: as another name of
    the same file in another folder (two paths that reach one name are never that), or
    with *copy* as a file of its own, which the caller has verified. A read-only file is
    deleted too, and *keeper* keeps its attributes. Raises OSError, with *victim* left,
    when *keeper* does not hold it."""
    folders = file_id(victim.parent), file_id(keeper.parent)
    if not (keeper.is_file() and not same_file(victim, keeper) if copy else
            two_names(victim, keeper) and None not in folders and folders[0] != folders[1]):
        raise OSError(f"{keeper} does not hold the file")
    _unlink_read_only_too(victim, keeper)


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
