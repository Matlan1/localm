# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a downloaded localm update: verify + extract the build zip, swap the source
tree (with a backup), run any deps/runtime step, then relaunch - rolling back if the
relaunched build fails its health check.

Runs ONLY from an explicit user action (``localm update`` / the GUI "Update now"
button), NEVER automatically, and the file swap runs in a DETACHED helper process
after the parent server has exited - so no live process is importing half-swapped
files. Self-modifying, so every apply backs up first and restores on failure: no
apply is reported as success unless the new VERSION is live afterwards.

The file primitives here are pure and unit-tested; ``main()`` wires them to the real
process orchestration (stop is implicit via the parent's exit; relaunch + health
check are integration-level).
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


def extract(zip_path, staging) -> Path:
    """Extract *zip_path* into *staging* (cleared first) and return the source root -
    descending into a single wrapper directory if the archive has one."""
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
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
    removed and not restored - the correct pre-apply state."""
    backup_dir, installed = Path(backup_dir), Path(installed)
    for name in names:
        dst = installed / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    if not backup_dir.exists():
        return
    for entry in backup_dir.iterdir():
        dst = installed / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)


def swap_with_backup(staged_root, installed, backup_dir) -> list:
    """Back up, then swap the staged source into the install. Returns the list of
    swapped names (for a later rollback). Raises (after restoring) if the swap fails
    partway - we never leave a half-applied tree."""
    names = swap_entries(staged_root)
    backup(installed, names, backup_dir)
    try:
        apply_files(staged_root, installed, names)
    except Exception:
        rollback(backup_dir, installed, names)
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
