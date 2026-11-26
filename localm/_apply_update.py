# SPDX-License-Identifier: AGPL-3.0-or-later
"""File primitives of an update: verify and extract the build zip, the parts of the
install an update never replaces, and the restore of the backup an update by an older
localm made (:func:`rollback`, :func:`backup_names`, :func:`backup_problem`).

An older localm installs this build with its own updater and keeps a backup of what it
replaced; its health watchdog runs this build's ``scripts/rollback_update.py``, which
loads this file BY FILE PATH and puts that backup back with :func:`rollback`, so this
module imports only the standard library at import time. ``localm update --rollback``
(``updater.rollback_last``) uses the same functions.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Top-level entries an update never replaces: the venv, the data dir, version
# control, and agent/local-only trees.
NEVER_TOUCH = frozenset({
    ".venv", "venv", ".git", ".github", "home", ".localcoder", "issues", "qa",
    "dev-notes", "scratch", "node_modules", ".claude", "__pycache__",
    "localm-home.cfg", ".localm-venv", ".localm-run", ".pytest_cache", ".ruff_cache",
})

# Install-root-relative POSIX sub-tree paths that survive a swap, at finer
# granularity than a NEVER_TOUCH top-level name. The native llama.cpp binaries
# under the runtime wheel's lib/ are local install state: a whole-tree replace
# of runtime/ would delete them and swap in the empty scaffold. The rest of
# runtime/ still updates normally.
PRESERVE_WITHIN = ("runtime/localm_llama_runtime/lib",)

# The provisioned llama.cpp runtime, relative to the install root.
RUNTIME_LIB = PRESERVE_WITHIN[0]

# In the runtime dir: the previous build an update kept because this release's
# build did not work there. Not part of the runtime itself (see _tree_signature).
RUNTIME_KEPT_RECORD = ".localm-runtime-kept.json"

#: The extras the dependency step of an update installs.
DEPS_EXTRAS = ("coder", "voice", "monitor")

# The file the backup of an update by an older localm holds:
# ``{"names": the swap set, "files": every file it copied}``.
BACKUP_RECORD = ".localm-backup.json"

# Top-level names every localm build has, so no update can have added them.
_IN_EVERY_BUILD = ("VERSION", "pyproject.toml", "localm")

# Files up to this size are compared by content in :func:`runtime_differs`; larger
# ones by size.
_SMALL_FILE = 64 * 1024

#: Install-root-relative POSIX paths of every file a build ships with the execute
#: permission (git mode 100755). See test_executables_match_the_tracked_file_modes.
EXECUTABLES = (
    "docker/entrypoint.sh", "docker/healthcheck.sh", "docker/smoke-test.sh", "install.sh",
    "localm-launcher.sh", "localm.sh", "report-issue.sh", "rollback.sh", "setup-gui.sh",
    "setup.sh",
)


def _add_exec(path) -> bool:
    """Give *path* the execute permission wherever it has the read permission (what
    ``chmod +x`` does under the usual umask). Returns True when the mode changed.
    Raises OSError."""
    mode = stat.S_IMODE(os.stat(path).st_mode)
    want = mode | ((mode & 0o444) >> 2)
    if want == mode:
        return False
    os.chmod(path, want)
    return True


def restore_executables(root) -> list:
    """Give each :data:`EXECUTABLES` file present in the install at *root* its
    execute permission. Returns the paths it changed. Does nothing on Windows.
    Raises OSError."""
    if os.name == "nt":
        return []
    root = Path(root)
    return [rel for rel in EXECUTABLES
            if (root / rel).is_file() and _add_exec(root / rel)]


def _within_preserved(rel_posix: str) -> bool:
    """True if install-root-relative POSIX path *rel_posix* is a PRESERVE_WITHIN
    sub-tree or lies inside one."""
    return any(rel_posix == p or rel_posix.startswith(p + "/") for p in PRESERVE_WITHIN)


def _name_has_preserved(name: str) -> bool:
    """True if top-level *name* contains a PRESERVE_WITHIN sub-tree, so its swap must
    merge around that sub-tree instead of a blunt rmtree + copytree."""
    return any(p == name or p.startswith(name + "/") for p in PRESERVE_WITHIN)


def _copy_into(src, dst, name) -> None:
    """Copy every entry under *src* into *dst* (create dirs, overwrite files), SKIPPING
    any PRESERVE_WITHIN sub-tree. A merge, not a replace: *dst* may already exist and
    its preserved sub-trees are left exactly as they are (never read, written, or
    deleted, so a locked provisioned binary is never disturbed). *name* is the
    top-level entry, which builds install-root-relative paths for the preserve test."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for s in src.rglob("*"):
        rel = f"{name}/{s.relative_to(src).as_posix()}"
        if _within_preserved(rel):
            continue
        d = dst / s.relative_to(src)
        if s.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


def _prune(dst, name) -> list:
    """Remove entries under *dst*, deepest first, SKIPPING PRESERVE_WITHIN sub-trees
    (and never deleting a non-empty dir, which protects the ancestors of a preserved
    sub-tree): everything not preserved goes (:func:`rollback`, before it restores the
    backup).

    Returns a list of human-readable strings, one per FILE that was meant to be removed
    but could NOT be (its unlink raised - e.g. a Windows AV lock on a freshly written
    file). Such a file is left behind STALE, so the caller must SURFACE it rather than
    report success over it: ``rollback()`` folds these into its RuntimeError. A dir that will not rmdir is NOT reported: the only
    dirs attempted are already-empty ones, and a non-empty (or locked) dir left behind
    strands no functional state."""
    dst = Path(dst)
    errors = []
    for d in sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = f"{name}/{d.relative_to(dst).as_posix()}"
        if _within_preserved(rel):
            continue
        try:
            if d.is_dir():
                # rmdir only an already-empty dir; a non-empty one still holds a
                # preserved sub-tree or its ancestors and is kept.
                if not any(d.iterdir()):
                    d.rmdir()
            else:
                d.unlink()
        except FileNotFoundError:
            continue   # already gone (a race) is the desired end state, not a failure
        except OSError as e:
            if d.is_dir():
                # A failed empty-dir rmdir strands nothing; keep pruning.
                continue
            # A file we could not remove stays behind stale, so report it.
            errors.append(f"remove {rel}: {e}")
    return errors


def verify_zip(zip_path) -> None:
    """Raise ValueError unless *zip_path* is a real zip that looks like a localm
    build (contains ``VERSION`` and ``pyproject.toml``, possibly under one wrapper
    dir). A safety gate before anything is swapped."""
    zp = Path(zip_path)
    if not zp.is_file():
        raise ValueError("update archive not found")
    if not zipfile.is_zipfile(zp):
        raise ValueError("update archive is not a zip")
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n and not n.endswith("/")]
    # Accept either flat (VERSION at root) or a single wrapper dir (repo-sha/VERSION).
    wanted = ("VERSION", "pyproject.toml")
    flat = all(any(n == w for n in names) for w in wanted)
    roots = {n.split("/", 1)[0] for n in names if "/" in n}
    wrapped = len(roots) == 1 and all(
        any(n == f"{next(iter(roots))}/{w}" for n in names) for w in wanted)
    if not (flat or wrapped):
        raise ValueError("archive does not look like a localm build "
                         "(no VERSION + pyproject.toml)")


def _unsafe_member(name: str) -> bool:
    """True if a zip member name would escape the extraction root: an absolute path,
    a Windows drive (``C:``), or any ``..`` traversal component. We reject these
    OUTRIGHT rather than rely on extractall's silent sanitization, which would still
    drop a sanitized name (``../evil`` -> ``evil``) at the staging root, where it
    would count as a file of the build."""
    norm = (name or "").replace("\\", "/")
    if not norm:
        return False
    if norm.startswith("/"):
        return True
    if len(norm) >= 2 and norm[1] == ":":   # drive letter (C:...)
        return True
    return ".." in norm.split("/")


def _unsafe_swap_name(name) -> bool:
    """True if *name* is not usable as a top-level swap/rollback entry.

    Stricter than :func:`_unsafe_member`. A swap name must be exactly ONE
    component living directly inside the install, because every caller does
    ``installed / name`` and then rmtree/unlink/copy on the result.
    ``_unsafe_member`` only answers "would this ESCAPE the extraction root",
    which is a narrower question, and three shapes slip past it:

    * ``""`` and ``"."`` do not escape - they COLLAPSE. ``Path(install) / ""``
      is the install dir itself, so ``shutil.rmtree`` on it deletes the entire
      installation.
    * A NESTED name (``a/b``) reaches a path the swap set never describes.
    * A ``NEVER_TOUCH`` name is by definition not part of any swap: an update
      never replaces one, so a manifest naming one is poisoned by construction. This
      matters most on a PORTABLE install, where ``home`` sits INSIDE the install
      root - rolling back an entry named ``home`` would delete the user's models,
      chat history and auth keys, and the backup does not contain them to restore.

    Used on the ``<home>/updates/applied_names.json`` manifest, which is an
    ordinary file in the data dir and so is attacker-writable given any write
    primitive there.
    """
    if not isinstance(name, str):
        return True
    if _unsafe_member(name):
        return True
    parts = [p for p in name.replace("\\", "/").strip().split("/") if p]
    if len(parts) != 1 or parts[0] in (".", ".."):
        return True
    return parts[0] in NEVER_TOUCH


def extract(zip_path, staging) -> Path:
    """Extract *zip_path* into *staging* (cleared first) and return the source root -
    descending into a single wrapper directory if the archive has one.

    REJECTS any member whose path would escape the staging dir (absolute path, drive
    letter, or ``..`` traversal), so a crafted build cannot write outside staging or
    plant top-level debris via a sanitized name.

    Outside Windows, a file the archive records with an execute permission gets it
    (:func:`_add_exec`); no other permission bit is taken from the archive."""
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if _unsafe_member(member):
                raise ValueError(f"unsafe path in update archive: {member!r}")
        z.extractall(staging)
        if os.name != "nt":
            for info in z.infolist():
                if not info.is_dir() and (info.external_attr >> 16) & 0o111:
                    _add_exec(staging / info.filename)
    if not (staging / "VERSION").exists():
        entries = list(staging.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
    return staging


def _backup_record(backup_dir):
    """The record :data:`BACKUP_RECORD` in *backup_dir*, or None when it has none (a
    backup made by an older build). Raises OSError or ValueError when it cannot be
    read."""
    try:
        text = (Path(backup_dir) / BACKUP_RECORD).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
    except RecursionError as e:
        raise ValueError(str(e)) from None
    if not isinstance(record, dict) or not all(
            isinstance(record.get(k), list) and all(isinstance(x, str) for x in record[k])
            for k in ("names", "files")):
        raise ValueError(f"{BACKUP_RECORD} is not a backup record")
    return record


def backup_names(backup_dir) -> Optional[list]:
    """The swap set recorded inside *backup_dir*, or None when it has no readable
    record. It describes this backup whatever ``applied_names.json`` says, because a
    rename moves both at once."""
    try:
        record = _backup_record(backup_dir)
    except (OSError, ValueError):
        return None
    return record["names"] if record else None


def backup_problem(backup_dir, names) -> Optional[str]:
    """Why *backup_dir* does not hold what a rollback needs, or None when it does.
    Removes nothing.

    With a record: every file the record lists must be there. A swap name absent from
    the backup is not a problem: it was new in the update. A record that cannot be read
    is a problem. Without a record (a backup made by an older build): any of
    :data:`_IN_EVERY_BUILD` among *names* (the manifest of that update) must be there."""
    import os
    backup_dir = Path(backup_dir)
    try:
        record = _backup_record(backup_dir)
    except (OSError, ValueError) as e:
        return f"its list of files cannot be read: {e}"
    if record is None:
        missing = [n for n in _IN_EVERY_BUILD
                   if n in names and not os.path.exists(backup_dir / n)]
    else:
        missing = [f for f in record["files"] if not os.path.isfile(backup_dir / f)]
    if missing:
        return f"{len(missing)} expected item(s) are missing, first {missing[0]}"
    return None


def rollback(backup_dir, installed, names) -> None:
    """Undo a swap: remove the swapped *names* from the install, then restore
    whatever was backed up. A name that was NEW (absent from the backup) is therefore
    removed and not restored - the correct pre-apply state. A name containing a
    PRESERVE_WITHIN sub-tree is unwound around that sub-tree (its scaffold is pruned
    and restored, the provisioned binaries are left in place), so rollback also never
    trips over a locked binary.

    When the update re-provisioned the native runtime, its snapshot
    (:func:`runtime_snapshot_dir`) is restored too (:func:`restore_runtime`).

    Raises RuntimeError listing any remove or restore operation that FAILED, so a
    failed rollback is NEVER reported as a success. The backup dir is left intact for
    manual recovery. Best-effort: it attempts every name even if one fails, and a
    folder with a file that cannot be removed still gets every backed-up file that can
    be written, then it reports the collected failures. Running it again once the file
    is free finishes the job. A backed-up folder not among *names* still in the install
    is such a failure, and is left as it is."""
    backup_dir, installed = Path(backup_dir), Path(installed)
    errors = []
    removal_failed = set()
    for name in names:
        # The names can come from <home>/updates/applied_names.json, a plain file
        # in the data dir, so a poisoned entry would reach `installed / name` and
        # then rmtree/unlink. Rejected and recorded as an error, never skipped
        # silently. Checked here because it is the choke point both callers reach.
        if _unsafe_swap_name(name):
            errors.append(f"refused unsafe name from the update manifest: {name!r}")
            continue
        dst = installed / name
        try:
            if dst.exists():
                if dst.is_dir():
                    if _name_has_preserved(name):
                        # Strip the scaffold, keep provisioned binaries. A file that
                        # cannot be removed is folded in as a rollback failure.
                        errors.extend(_prune(dst, name))
                    else:
                        shutil.rmtree(dst)
                else:
                    dst.unlink()
        except OSError as e:
            errors.append(f"remove {name}: {e}")
            removal_failed.add(name)
    if backup_dir.exists():
        for entry in backup_dir.iterdir():
            if entry.name == BACKUP_RECORD:
                continue
            dst = installed / entry.name
            try:
                if entry.is_dir():
                    if _name_has_preserved(entry.name):
                        _copy_into(entry, dst, entry.name)   # dst still holds the preserved sub-tree
                    else:
                        # Merges only over a folder whose removal failed: any other
                        # existing folder fails here, since it was never removed.
                        shutil.copytree(entry, dst, dirs_exist_ok=entry.name in removal_failed)
                else:
                    shutil.copy2(entry, dst)
            except OSError as e:
                errors.append(f"restore {entry.name}: {e}")
    snapshot = runtime_snapshot_dir(backup_dir)
    if snapshot.is_dir():
        errors.extend(restore_runtime(snapshot, installed / RUNTIME_LIB))
    if errors:
        raise RuntimeError(
            f"rollback incomplete (backup kept at {backup_dir}): " + "; ".join(errors[:6]))


def post_swap_command(klass: str, backend: Optional[str] = None,
                      uv: Optional[str] = None) -> Optional[list]:
    """The extra command (argv list) an update class needs after the file swap, or
    None for a pure ``reboot``. ``deps`` installs localm and its dependencies plus
    :data:`DEPS_EXTRAS` from the release's ``uv.lock`` into ``.venv`` with the uv
    at *uv* (required for that class; see :func:`localm.uv_tool.find_uv` and
    :func:`localm.pyinstall.sync_command`), run from the install root; ``runtime``
    re-provisions the native binaries; ``setup`` is handled by the user (no
    in-process command)."""
    if klass == "deps":
        if not uv:
            raise ValueError("the deps update class needs the path of the uv to run")
        from localm import pyinstall
        return pyinstall.sync_command(uv, DEPS_EXTRAS)
    if klass == "runtime":
        # Re-invoke through the current interpreter: a bare "localm" argv[0]
        # resolves back to the LocaLM.exe launcher itself on Windows.
        # --force: setup-llama's "already provisioned" guard would otherwise
        # short-circuit and re-provision nothing. A runtime in use is still
        # refused by _clear_target_or_refuse before anything is deleted.
        # --yes: this runs as a detached subprocess with nothing on stdin, and
        # --force reaches a click.confirm() that would otherwise hang.
        # --keep-choice: a user's recorded choice of a backend that could not
        # be used here survives the re-provision of the installed one.
        return [sys.executable, "-m", "localm", "setup-llama",
                "--backend", backend or "vulkan", "--force", "--yes", "--keep-choice"]
    return None


# ------------------------- the running process's code ------------------------

def _norm(path) -> str:
    import os
    return os.path.normcase(str(Path(path).resolve()))


def pin_imports(code_dir, live_dir):
    """Make every later import of a not-yet-imported ``localm`` submodule in
    this process load it from *code_dir* (a copy of the code this process
    runs) instead of *live_dir* (the install's ``localm`` package, about to
    hold another version's files): each imported package whose ``__path__``
    entry lies under *live_dir* gets the same entry under *code_dir*. Returns
    a callable that puts every changed ``__path__`` back. Changes nothing for
    packages imported from elsewhere."""
    import importlib
    import os
    live, code = _norm(live_dir), Path(code_dir)
    changed = []
    for name, mod in list(sys.modules.items()):
        if not (name == "localm" or name.startswith("localm.")):
            continue
        paths = getattr(mod, "__path__", None)
        if not isinstance(paths, list):
            continue
        new = []
        for entry in paths:
            norm = _norm(entry)
            if norm == live or norm.startswith(live + os.sep):
                new.append(str(code / os.path.relpath(norm, live)) if norm != live
                           else str(code))
            else:
                new.append(entry)
        if new != paths:
            changed.append((mod, list(paths)))
            paths[:] = new
    importlib.invalidate_caches()

    def unpin() -> None:
        for mod, old in changed:
            mod.__path__[:] = old
        importlib.invalidate_caches()

    return unpin


def running_code_dir() -> Path:
    """The folder this process imports ``localm`` submodules from (the first
    entry of ``localm.__path__``)."""
    import localm
    return Path(list(localm.__path__)[0])


def keep_running_code(installed, dest):
    """Before the ``localm`` package of the install at *installed* is replaced
    while this process keeps running, keep a copy of it in ``<dest>/localm``
    (hard links for large files) and pin this process's imports to it
    (:func:`pin_imports`) when this process imports from that package. Returns
    the undo callable (a no-op when nothing was pinned). Raises OSError when
    the copy cannot be made."""
    live = Path(installed) / "localm"
    if _norm(running_code_dir()) != _norm(live):
        return lambda: None
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    try:
        _clone_tree(live, dest / "localm")
    except OSError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return pin_imports(dest / "localm", live)


# ------------------------- native runtime snapshot ---------------------------

def runtime_snapshot_dir(backup_dir) -> Path:
    """Where the update whose backup is *backup_dir* keeps the native runtime it
    replaced: a sibling ``<backup_dir>-runtime``."""
    backup_dir = Path(backup_dir)
    return backup_dir.with_name(backup_dir.name + "-runtime")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst*, or copy it when the link cannot be made (another
    volume, a file system without hard links). Files up to ``_SMALL_FILE`` (the
    markers and choice record, which are rewritten in place) are always copied, so
    a write to one side never changes the other."""
    if src.stat().st_size > _SMALL_FILE:
        try:
            import os
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _clone_tree(src: Path, dst: Path) -> None:
    """Recreate every directory and file under *src* in *dst* (which must not
    exist) by :func:`_link_or_copy`. Raises OSError."""
    dst.mkdir(parents=True)
    for s in sorted(src.rglob("*")):
        d = dst / s.relative_to(src)
        if s.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(s, d)


def snapshot_runtime(lib, dest) -> None:
    """Keep the runtime at *lib* in *dest* (replaced first) as hard links where
    possible, so a re-provision can be undone. Raises OSError; *dest* is removed
    again when the snapshot cannot be completed."""
    lib, dest = Path(lib), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    try:
        _clone_tree(lib, dest)
    except OSError:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _tree_signature(root: Path) -> dict:
    """Relative path -> content (small files) or size, for every file under *root*
    except a :data:`RUNTIME_KEPT_RECORD`."""
    sig = {}
    for f in root.rglob("*"):
        if f.is_file() and f.name != RUNTIME_KEPT_RECORD:
            st = f.stat()
            key = f.relative_to(root).as_posix()
            sig[key] = f.read_bytes() if st.st_size <= _SMALL_FILE else st.st_size
    return sig


def runtime_differs(lib, snapshot) -> bool:
    """Whether the runtime at *lib* differs from *snapshot*: a file added, removed,
    resized, or a small file (the markers) with other content. True when either
    side cannot be read."""
    try:
        return _tree_signature(Path(lib)) != _tree_signature(Path(snapshot))
    except OSError:
        return True


def mapped_by_others(root) -> dict:
    """Linux: ``{pid: [paths]}`` for every other process that maps a file under
    *root* (``/proc/<pid>/maps``), the way a loaded shared library shows. Empty
    elsewhere, or when ``/proc`` cannot be read."""
    import os
    found = {}
    proc = Path("/proc")
    if not sys.platform.startswith("linux") or not proc.is_dir():
        return found
    try:
        prefix = str(Path(root).resolve()) + "/"
        entries = list(proc.iterdir())
    except OSError:
        return found
    for d in entries:
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            maps = (d / "maps").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        paths = sorted({line.split(None, 5)[5].strip() for line in maps.splitlines()
                        if len(line.split(None, 5)) == 6
                        and line.split(None, 5)[5].startswith(prefix)})
        if paths:
            found[int(d.name)] = [Path(p) for p in paths]
    return found


def held_files(root) -> list:
    """Files under *root* another process holds so they cannot be replaced
    safely. Windows: probed with ``open(..., "r+b")`` (writes nothing), which it
    refuses for a loaded DLL. Linux: files another process maps
    (:func:`mapped_by_others`), since replacing a library under a live process
    mixes builds in it. A file that cannot be probed for another reason counts
    as not held."""
    held = []
    root = Path(root)
    if not root.is_dir():
        return held
    if sys.platform != "win32":
        for paths in mapped_by_others(root).values():
            held.extend(p for p in paths if p not in held)
        return held
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        try:
            with open(f, "r+b"):
                pass
        except PermissionError:
            held.append(f)
        except OSError:
            continue
    return held


def rm_holders(files: list) -> list:
    """Processes holding *files*, from the Windows Restart Manager:
    ``[{"pid", "name"}]``. Empty when it names none or cannot be asked."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes
    rm = ctypes.WinDLL("rstrtmgr")

    class _UniqueProcess(ctypes.Structure):
        _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]

    class _ProcessInfo(ctypes.Structure):
        _fields_ = [("Process", _UniqueProcess), ("strAppName", ctypes.c_wchar * 256),
                    ("strServiceShortName", ctypes.c_wchar * 64),
                    ("ApplicationType", ctypes.c_int), ("AppStatus", wintypes.ULONG),
                    ("TSSessionId", wintypes.DWORD), ("bRestartable", wintypes.BOOL)]

    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)
    if rm.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return []
    try:
        names = (ctypes.c_wchar_p * len(files))(*[str(f) for f in files])
        if rm.RmRegisterResources(session, len(files), names, 0, None, 0, None) != 0:
            return []
        needed, count, reasons = wintypes.UINT(0), wintypes.UINT(0), wintypes.DWORD(0)
        rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None,
                          ctypes.byref(reasons))
        if rc not in (0, 234) or needed.value == 0:
            return []
        infos = (_ProcessInfo * needed.value)()
        count = wintypes.UINT(needed.value)
        if rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), infos,
                        ctypes.byref(reasons)) != 0:
            return []
        return [{"pid": int(i.Process.dwProcessId), "name": i.strAppName}
                for i in infos[:count.value]]
    finally:
        rm.RmEndSession(session)


def restore_runtime(snapshot, lib) -> list:
    """Put the runtime kept in *snapshot* back at *lib*. Returns the failures as
    strings (empty on success, or when *lib* already equals the snapshot).

    Refuses before changing anything when a file under *lib* is held by another
    process, so the installed runtime is never left half replaced by this call."""
    snapshot, lib = Path(snapshot), Path(lib)
    if not runtime_differs(lib, snapshot):
        return []
    held = held_files(lib)
    if held:
        return [f"native runtime not restored: {len(held)} file(s) under {lib} are in use "
                f"(first: {held[0].name}); close every localm process and run the same "
                "rollback again, or re-provision it with `localm setup-llama --force`"]
    errors = []
    if lib.exists():
        for entry in sorted(lib.iterdir()):
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except OSError as e:
                errors.append(f"native runtime: remove {entry.name}: {e}")
    if errors:
        return errors
    try:
        lib.mkdir(parents=True, exist_ok=True)
        for s in sorted(snapshot.rglob("*")):
            d = lib / s.relative_to(snapshot)
            if s.is_dir():
                d.mkdir(parents=True, exist_ok=True)
            else:
                d.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(s, d)
    except OSError as e:
        errors.append(f"native runtime: restore from {snapshot}: {e}")
    return errors
