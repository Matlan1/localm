# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a downloaded localm update: verify + extract the build zip, then swap the source tree into the install with a backup, restoring that backup if the swap (or the post-swap deps/runtime step run by ``updater.apply``) fails partway."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Top-level entries an update never replaces: the venv, the data dir, version
# control, and agent/local-only trees.
NEVER_TOUCH = frozenset({
    ".venv", "venv", ".git", ".github", "home", ".localcoder", "issues", "qa",
    "dev-notes", "scratch", "node_modules", ".claude", "__pycache__",
    "localm-home.cfg", ".localm-venv", ".pytest_cache", ".ruff_cache",
})

# Install-root-relative POSIX sub-tree paths that survive a swap, at finer
# granularity than a NEVER_TOUCH top-level name. The native llama.cpp binaries
# under the runtime wheel's lib/ are local install state: a whole-tree replace
# of runtime/ would delete them and swap in the empty scaffold. The rest of
# runtime/ still updates normally.
PRESERVE_WITHIN = ("runtime/localm_llama_runtime/lib",)


def _debug_warn(msg, *args) -> None:
    """Best-effort WARNING via localm's logger."""
    try:
        from localm.debuglog import logger
        logger.warning(msg, *args)
    except Exception:
        pass


def _within_preserved(rel_posix: str) -> bool:
    """True if install-root-relative POSIX path *rel_posix* is a PRESERVE_WITHIN sub-tree or lies inside one."""
    return any(rel_posix == p or rel_posix.startswith(p + "/") for p in PRESERVE_WITHIN)


def _name_has_preserved(name: str) -> bool:
    """True if top-level *name* contains a PRESERVE_WITHIN sub-tree, so its swap must merge around that sub-tree instead of a blunt rmtree + copytree."""
    return any(p == name or p.startswith(name + "/") for p in PRESERVE_WITHIN)


def _copy_into(src, dst, name) -> None:
    """Copy every entry under *src* into *dst* (create dirs, overwrite files), SKIPPING any PRESERVE_WITHIN sub-tree."""
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
    """Remove entries under *dst*, deepest first, SKIPPING PRESERVE_WITHIN sub-trees (and never deleting a non-empty dir, which protects the ancestors of a preserved sub-tree)."""
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
    """Raise ValueError unless *zip_path* is a real zip that looks like a localm build (contains ``VERSION`` and ``pyproject.toml``, possibly under one wrapper dir)."""
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
    """True if a zip member name would escape the extraction root: an absolute path, a Windows drive (``C:``), or any ``..`` traversal component."""
    norm = (name or "").replace("\\", "/")
    if not norm:
        return False
    if norm.startswith("/"):
        return True
    if len(norm) >= 2 and norm[1] == ":":   # drive letter (C:...)
        return True
    return ".." in norm.split("/")


def _unsafe_swap_name(name) -> bool:
    """True if *name* is not usable as a top-level swap/rollback entry."""
    if not isinstance(name, str):
        return True
    if _unsafe_member(name):
        return True
    parts = [p for p in name.replace("\\", "/").strip().split("/") if p]
    if len(parts) != 1 or parts[0] in (".", ".."):
        return True
    return parts[0] in NEVER_TOUCH


def extract(zip_path, staging) -> Path:
    """Extract *zip_path* into *staging* (cleared first) and return the source root - descending into a single wrapper directory if the archive has one."""
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
    """Top-level names in the staged build to copy into the install (everything except the NEVER_TOUCH set)."""
    return sorted(p.name for p in Path(staged_root).iterdir()
                  if p.name not in NEVER_TOUCH)


def backup(installed, names, backup_dir) -> None:
    """Copy the install's current version of each *name* into *backup_dir* (cleared first), so a failed apply can be rolled back."""
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
    """Replace each *name* in the install with the staged version (whole-tree replace so upstream deletions within a replaced dir take effect)."""
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
                # A stale file the new build dropped could not be removed. Logged
                # rather than failing the update: an extra file, not a broken tree.
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
    """Undo a swap: remove the swapped *names* from the install, then restore whatever was backed up."""
    backup_dir, installed = Path(backup_dir), Path(installed)
    errors = []
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
    """Back up, then swap the staged source into the install."""
    names = swap_entries(staged_root)
    backup(installed, names, backup_dir)
    try:
        apply_files(staged_root, installed, names)
    except Exception as swap_err:
        try:
            rollback(backup_dir, installed, names)
        except Exception as rb:
            # Both the swap and the recovery failed; the backup dir is kept.
            raise RuntimeError(
                f"update swap failed ({swap_err}) AND rollback also failed ({rb}); "
                f"the install may be inconsistent - restore from {backup_dir}") from swap_err
        raise
    return names


def post_swap_command(klass: str, backend: Optional[str] = None) -> Optional[list]:
    """The extra command (argv list) an update class needs after the file swap, or None for a pure ``reboot``. ``deps`` reinstalls editable; ``runtime`` re-provisions the native binaries; ``setup`` is handled by the user (no in-process command)."""
    if klass == "deps":
        return ["uv", "pip", "install", "-p", ".venv", "-e", ".[coder,voice,monitor]"]
    if klass == "runtime":
        # Re-invoke through the current interpreter: a bare "localm" argv[0]
        # resolves back to the LocaLM.exe launcher itself on Windows.
        # --force: setup-llama's "already provisioned" guard would otherwise
        # short-circuit and re-provision nothing. A runtime in use is still
        # refused by _clear_target_or_refuse before anything is deleted.
        # --yes: this runs as a detached subprocess with nothing on stdin, and
        # --force reaches a click.confirm() that would otherwise hang.
        return [sys.executable, "-m", "localm", "setup-llama",
                "--backend", backend or "vulkan", "--force", "--yes"]
    return None
