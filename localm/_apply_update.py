# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a downloaded localm update: verify + extract the build zip, then swap the
source tree into the install with a backup, restoring that backup if the swap (or
the post-swap deps/runtime step run by ``updater.apply``) fails partway.

Runs ONLY from an explicit user action (``localm update`` / the GUI "Update now"
button), NEVER automatically. The swap runs in-process inside ``updater.apply()``
and the caller restarts afterwards (the CLI tells the user; the server re-execs).
KNOWN GAP (LM-DA-011): there is no detached helper process, no post-relaunch
health check, and no automatic rollback for a build that swaps cleanly but
misbehaves after the restart. Recovery from such a build is ``localm update
--rollback`` - which lives inside the NEW build, so a build too broken to start
means restoring the kept backup dir by hand. The health-checked detached
relauncher must be built before the updater serves real releases.

The file primitives here are pure and unit-tested; ``updater.apply()`` wires them
to download, signature verification, and the post-swap step.
"""

from __future__ import annotations

import shutil
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
    skipped (recorded by their absence)."""
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
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def apply_files(staged_root, installed, names) -> None:
    """Replace each *name* in the install with the staged version (whole-tree replace
    so upstream deletions within a replaced dir take effect)."""
    staged_root, installed = Path(staged_root), Path(installed)
    for name in names:
        src = staged_root / name
        if not src.exists():
            continue
        dst = installed / name
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
    removed and not restored - the correct pre-apply state.

    Raises RuntimeError listing any restore operation that FAILED, so a failed
    rollback is NEVER silently reported as a success (we do not hide problems). The
    backup dir is left intact for manual recovery. Best-effort: it attempts every
    name even if one fails, then reports the collected failures."""
    backup_dir, installed = Path(backup_dir), Path(installed)
    errors = []
    for name in names:
        dst = installed / name
        try:
            if dst.exists():
                if dst.is_dir():
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
        return ["localm", "setup-llama", "--backend", backend or "vulkan"]
    return None
