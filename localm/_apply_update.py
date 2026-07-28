# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a downloaded localm update: verify + extract the build zip, then swap the
source tree into the install with a backup, restoring that backup if the swap (or
the post-swap deps/runtime step run by ``updater.apply``) fails partway.

Runs ONLY from an explicit user action (``localm update`` / the GUI "Update now"
button), NEVER automatically. The swap runs in-process inside ``updater.apply()``
and the caller restarts afterwards (the CLI tells the user; the server re-execs).
LM-DA-011: the server's automatic restart (``_do_restart`` in
``localm/inference/http_server.py``) spawns a DETACHED helper process right
before re-exec'ing (``updater.spawn_health_watchdog()`` ->
``scripts/update_watchdog.py``) that polls the relaunched build's own
``/whoami`` for the applied VERSION and automatically rolls back - via the
standalone ``scripts/rollback_update.py``, loaded by file path so it works even
if the new build's ``localm`` package will not import - if it never comes up
healthy within a bounded window. Manual recovery (``localm update --rollback``,
or ``rollback.bat``/``rollback.sh`` in the install root) remains available for
a build too broken for even the watchdog's own spawn to matter.

The file primitives here are pure and unit-tested; ``updater.apply()`` wires them
to download, signature verification, and the post-swap step.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Top-level entries an update NEVER replaces: the venv, the data dir, version
# control, and agent/local-only trees. The build zip should not contain these
# anyway, but excluding them defensively means a swap can never clobber the user's
# models/config/sessions or the .venv even if a zip is mispackaged.
NEVER_TOUCH = frozenset({
    ".venv", "venv", ".git", ".github", "home", ".localcoder", "issues", "qa",
    "dev-notes", "scratch", "node_modules", ".claude", "__pycache__",
    "localm-home.cfg", ".localm-venv", ".pytest_cache", ".ruff_cache",
})

# Sub-tree paths (install-root-relative, POSIX) that are LOCAL INSTALL STATE living
# INSIDE an otherwise-shipped top-level tree, and so must survive a swap exactly like
# NEVER_TOUCH protects whole top-level entries - but at finer granularity than a
# top-level name.
#
# The native llama.cpp binaries that ``localm setup-llama`` provisions into the
# runtime wheel's ``lib/`` are the motivating case. They are gitignored, so a release
# build.zip carries only the empty scaffold; a blunt whole-tree replace of ``runtime/``
# would therefore DELETE the provisioned .dll/.so and swap in the empty scaffold,
# breaking inference until the user manually re-runs ``localm setup-llama``. Worse,
# while localm is RUNNING the loaded DLLs are file-locked on Windows, so deleting them
# mid-swap raises WinError 32 and can strand a half-applied tree (the rollback then
# also trips on the same lock). Leaving these binaries untouched fixes both: they
# persist across an update, and the swap never trips over a locked file. The REST of
# ``runtime/`` (its scaffold source) still updates normally. This is the sub-tree
# analogue of ``.venv`` being NEVER_TOUCH: setup creates it on a fresh install, and an
# in-place update must not clobber it.
PRESERVE_WITHIN = ("runtime/localm_llama_runtime/lib",)


def _debug_warn(msg, *args) -> None:
    """Best-effort WARNING via localm's logger. This module is stdlib-only and can be
    loaded BY FILE PATH (``scripts/rollback_update.py``) with NO importable localm
    package, so the logger import is lazy and guarded: a logging failure must never
    break or mask a swap. The only caller (``apply_files``) runs on the forward-apply
    path where the full package is importable; the standalone rollback path never
    reaches it."""
    try:
        from localm.debuglog import logger
        logger.warning(msg, *args)
    except Exception:
        pass


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
    top-level entry, used to build install-root-relative paths for the preserve test."""
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


def _prune(dst, name, *, keep_src=None) -> list:
    """Remove entries under *dst*, deepest first, SKIPPING PRESERVE_WITHIN sub-trees
    (and never deleting a non-empty dir, which protects the ancestors of a preserved
    sub-tree). With *keep_src*, remove only entries absent under keep_src (prune what
    the new build dropped - used by apply); without it, remove everything not preserved
    (used by rollback before restoring the backup).

    Returns a list of human-readable strings, one per FILE that was meant to be removed
    but could NOT be (its unlink raised - e.g. a Windows AV lock on a freshly written
    file). Such a file is left behind STALE, so the caller must SURFACE it rather than
    report success over it (we do not hide problems): ``rollback()`` folds these into
    its RuntimeError, ``apply_files()`` logs them. A dir that will not rmdir is NOT
    reported: the only dirs we attempt to remove are already-empty ones, and a
    non-empty (or locked) dir left behind strands no functional state - keeping it is
    the benign ancestor-protection case, not a failure."""
    dst = Path(dst)
    keep_src = Path(keep_src) if keep_src is not None else None
    errors = []
    for d in sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = f"{name}/{d.relative_to(dst).as_posix()}"
        if _within_preserved(rel):
            continue
        if keep_src is not None and (keep_src / d.relative_to(dst)).exists():
            continue
        try:
            if d.is_dir():
                # rmdir ONLY an already-empty dir; a non-empty one is deliberately kept
                # (it still holds a preserved sub-tree or its ancestors). This is the
                # documented ancestor-protection case, not a failure.
                if not any(d.iterdir()):
                    d.rmdir()
            else:
                d.unlink()
        except FileNotFoundError:
            continue   # already gone (a race) is the desired end state, not a failure
        except OSError as e:
            if d.is_dir():
                # An empty-dir rmdir that still failed (lock/permissions) is benign for
                # pruning - a leftover empty dir strands nothing - so keep swallowing it.
                continue
            # A FILE we intended to remove but could not stays behind stale: surface it
            # so apply()/rollback() never report success over a stale file.
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
    drop a sanitized name (``../evil`` -> ``evil``) at the staging root where
    swap_entries() would then pick it up and copy it into the install."""
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

    Stricter than :func:`_unsafe_member`, and for a different reason. A swap name
    must be exactly ONE component living directly inside the install, because
    every caller does ``installed / name`` and then rmtree/unlink/copy on the
    result. ``_unsafe_member`` only answers "would this ESCAPE the extraction
    root", which is a narrower question, and three shapes slip past it:

    * ``""`` and ``"."`` do not escape - they COLLAPSE. ``Path(install) / ""``
      is the install dir itself (verified on 3.12), so ``shutil.rmtree`` on it
      deletes the entire installation. Escape is not the only way to be lethal.
    * A NESTED name (``a/b``) reaches a path the swap set never describes.
    * A ``NEVER_TOUCH`` name is by definition not part of any swap: swap_entries
      excludes them, so a manifest naming one is poisoned by construction. This
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
    plant top-level debris via a sanitized name."""
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if _unsafe_member(member):
                raise ValueError(f"unsafe path in update archive: {member!r}")
        z.extractall(staging)
    if not (staging / "VERSION").exists():
        entries = list(staging.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
    return staging


def swap_entries(staged_root) -> list:
    """Top-level names in the staged build to copy into the install (everything
    except the NEVER_TOUCH set). Sorted for determinism."""
    return sorted(p.name for p in Path(staged_root).iterdir()
                  if p.name not in NEVER_TOUCH)


def backup(installed, names, backup_dir) -> None:
    """Copy the install's current version of each *name* into *backup_dir* (cleared
    first), so a failed apply can be rolled back. Names absent from the install are
    skipped (recorded by their absence).

    A name containing a PRESERVE_WITHIN sub-tree is backed up WITHOUT that sub-tree:
    the provisioned binaries are never modified by the swap, so there is nothing to
    restore, and skipping them avoids copying large (and, while running, locked)
    files on every update."""
    installed, backup_dir = Path(installed), Path(backup_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)
    for name in names:
        src = installed / name
        if not src.exists():
            continue
        dst = backup_dir / name
        if src.is_dir():
            if _name_has_preserved(name):
                _copy_into(src, dst, name)
            else:
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def apply_files(staged_root, installed, names) -> None:
    """Replace each *name* in the install with the staged version (whole-tree replace
    so upstream deletions within a replaced dir take effect).

    A name that contains a PRESERVE_WITHIN sub-tree (e.g. ``runtime/`` holding the
    provisioned native binaries) is MERGED instead of blunt-replaced: the new scaffold
    is copied in and files the new build dropped are pruned, but the preserved sub-tree
    is never read, written, or deleted - so provisioned binaries persist and a locked
    DLL never blocks the swap."""
    staged_root, installed = Path(staged_root), Path(installed)
    for name in names:
        src = staged_root / name
        if not src.exists():
            continue
        dst = installed / name
        if _name_has_preserved(name):
            _copy_into(src, dst, name)
            prune_errs = _prune(dst, name, keep_src=src)
            if prune_errs:
                # A stale file the new build DROPPED could not be removed after the merge
                # (e.g. a Windows AV lock on a freshly written file). It is left behind;
                # that is non-fatal for the forward apply (an extra file, not a broken
                # tree), so we LOG it rather than escalate a best-effort prune into a
                # failed update - but we never swallow it silently (do not hide problems).
                _debug_warn("update apply: could not remove %d stale file(s) under %r "
                            "after merge: %s", len(prune_errs), name,
                            "; ".join(prune_errs[:6]))
            continue
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def rollback(backup_dir, installed, names) -> None:
    """Undo a swap: remove the swapped *names* from the install, then restore
    whatever was backed up. A name that was NEW (absent from the backup) is therefore
    removed and not restored - the correct pre-apply state. A name containing a
    PRESERVE_WITHIN sub-tree is unwound around that sub-tree (its scaffold is pruned
    and restored, the provisioned binaries are left in place), so rollback also never
    trips over a locked binary.

    Raises RuntimeError listing any restore operation that FAILED, so a failed
    rollback is NEVER silently reported as a success (we do not hide problems). The
    backup dir is left intact for manual recovery. Best-effort: it attempts every
    name even if one fails, then reports the collected failures."""
    backup_dir, installed = Path(backup_dir), Path(installed)
    errors = []
    for name in names:
        # The names can come from <home>/updates/applied_names.json, which is a
        # plain file in the data dir: until here it was only type-checked as
        # list[str], so a poisoned entry (a "../.." traversal, or a
        # drive-qualified absolute name) reached `installed / name` and then
        # rmtree/unlink. This module already
        # rejects exactly those shapes for zip members in _unsafe_member; the
        # manifest path simply omitted the check. _unsafe_swap_name adds the
        # collapse/nesting/NEVER_TOUCH cases that a pure escape test misses (see
        # its docstring). Reject, do NOT silently skip:
        # the entry is recorded as an error so the RuntimeError below reports it
        # and the backup is kept (we do not hide problems).
        #
        # Checked HERE rather than in each caller because this is the choke point
        # both reach - scripts/rollback_update.py (stdlib-only, loads this module
        # by file path) and localm.updater.rollback - so a future caller cannot
        # bypass it.
        if _unsafe_swap_name(name):
            errors.append(f"refused unsafe name from the update manifest: {name!r}")
            continue
        dst = installed / name
        try:
            if dst.exists():
                if dst.is_dir():
                    if _name_has_preserved(name):
                        # Strip the scaffold, keep provisioned binaries. Any file it
                        # cannot remove is a real rollback failure - fold it in so the
                        # RuntimeError below reports it (never a stale file under a
                        # silent rolled_back:True).
                        errors.extend(_prune(dst, name))
                    else:
                        shutil.rmtree(dst)
                else:
                    dst.unlink()
        except OSError as e:
            errors.append(f"remove {name}: {e}")
    if backup_dir.exists():
        for entry in backup_dir.iterdir():
            dst = installed / entry.name
            try:
                if entry.is_dir():
                    if _name_has_preserved(entry.name):
                        _copy_into(entry, dst, entry.name)   # dst still holds the preserved sub-tree
                    else:
                        shutil.copytree(entry, dst)
                else:
                    shutil.copy2(entry, dst)
            except OSError as e:
                errors.append(f"restore {entry.name}: {e}")
    if errors:
        raise RuntimeError(
            f"rollback incomplete (backup kept at {backup_dir}): " + "; ".join(errors[:6]))


def swap_with_backup(staged_root, installed, backup_dir) -> list:
    """Back up, then swap the staged source into the install. Returns the list of
    swapped names (for a later rollback). Raises (after restoring) if the swap fails
    partway - we never leave a half-applied tree."""
    names = swap_entries(staged_root)
    backup(installed, names, backup_dir)
    try:
        apply_files(staged_root, installed, names)
    except Exception as swap_err:
        try:
            rollback(backup_dir, installed, names)
        except Exception as rb:
            # Both the swap AND the recovery failed - surface both; never pretend the
            # tree is intact. The backup dir is kept for manual restore.
            raise RuntimeError(
                f"update swap failed ({swap_err}) AND rollback also failed ({rb}); "
                f"the install may be inconsistent - restore from {backup_dir}") from swap_err
        raise
    return names


def post_swap_command(klass: str, backend: Optional[str] = None) -> Optional[list]:
    """The extra command (argv list) an update class needs after the file swap, or
    None for a pure ``reboot``. ``deps`` reinstalls editable; ``runtime`` re-provisions
    the native binaries; ``setup`` is handled by the user (no in-process command)."""
    if klass == "deps":
        return ["uv", "pip", "install", "-p", ".venv", "-e", ".[coder,voice,monitor]"]
    if klass == "runtime":
        # A bare "localm" argv[0] resolves back to the launcher exe itself when running
        # as the default native LocaLM.exe build (Windows same-directory command
        # resolution), which then mis-invokes and rolls the whole update back. Always
        # re-invoke through the current interpreter, matching every other
        # self-invocation site (setup_llama.py, applaunch.py, http_server.py, ...).
        return [sys.executable, "-m", "localm", "setup-llama", "--backend", backend or "vulkan"]
    return None
