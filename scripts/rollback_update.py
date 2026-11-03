#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone rollback: restore the PREVIOUS localm build after a bad update, even
when the updated build is too broken to run ``localm update --rollback`` itself.

``localm update`` keeps the pre-update version under ``<data-home>/updates/backup``
and records the full swap set in ``<data-home>/updates/applied_names.json``. This
script restores that backup into the install using NOTHING but the standard library
and the tiny, stdlib-only ``localm._apply_update`` primitives, which it loads BY FILE
PATH so a broken ``localm`` package cannot stop the rollback. Run it via
``rollback.bat`` / ``rollback.sh`` in the install root, or
``python scripts/rollback_update.py``.

It NEVER reports success it did not achieve: if there is no backup, the restore
helper cannot load, or a restore step fails, it says so, keeps the backup intact,
and exits non-zero, pointing at the backup dir for a manual copy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# The install root is the parent of this script's scripts/ dir, derived from __file__.
INSTALL_ROOT = Path(__file__).resolve().parent.parent


def _read_cfg_line(marker: Path) -> str:
    """The first line of a ``localm-home.cfg``: UTF-8 (with or without a BOM), or
    on Windows, when that fails, the console (OEM) code page that cmd.exe's ``echo``
    writes. Raises OSError, ValueError or LookupError when it cannot be read."""
    raw = marker.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if sys.platform != "win32":
            raise
        import ctypes
        text = raw.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}")
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def _detect_home(install_root: Path) -> Path:
    """The localm data dir, resolved WITHOUT importing localm.config (which may live in
    the broken build). Mirrors config._detect_home: LOCALM_HOME, else a portable
    ``localm-home.cfg`` / ``./home`` in the install, else the contained ``./home``."""
    env = os.environ.get("LOCALM_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if (install_root / "pyproject.toml").is_file():
        marker = install_root / "localm-home.cfg"
        if marker.is_file():
            try:
                line = _read_cfg_line(marker)
                if line:
                    return Path(line).expanduser()
            except (OSError, ValueError, LookupError):
                pass   # unreadable marker: fall through to ./home
        portable = install_root / "home"
        if portable.is_dir():
            return portable
    return install_root / "home"


def _load_helper(install_root: Path, name: str, code_root=None):
    """Load ``localm/<name>.py`` BY FILE PATH (bypassing the ``localm`` package
    __init__), so the rollback works even if the rest of localm is broken: from
    *code_root* (the code of the build that applied the update) when given and it
    has the file, else from *install_root*. Returns the module, or None when it
    cannot be loaded."""
    roots = [Path(code_root)] if code_root else []
    for root in dict.fromkeys(roots + [install_root]):
        path = root / "localm" / f"{name}.py"
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"_localm_{name}_standalone", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            return None   # a corrupt helper takes the manual-recovery path
    return None


def _load_apply_update(install_root: Path, code_root=None):
    """``localm/_apply_update.py`` (:func:`_load_helper`), or None (the caller then
    prints manual guidance)."""
    return _load_helper(install_root, "_apply_update", code_root)


def _restore_venv(install_root: Path, updates_dir: Path, code_root=None):
    """Put back the venv the update kept before its dependency step (the ``venv``
    entry of ``<updates>/backup-components``): every package that step changed and
    localm's own install record, unless that copy belongs to another backup than
    the one being restored. Returns ``(errors, undo)``: the failures (empty when
    there is nothing to restore or it was restored) and a callable that puts the
    venv back as it was before this call (None when nothing was changed)."""
    snap = updates_dir / "backup-components"
    state_file = snap / "state.json"
    if not state_file.exists():
        return [], None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        info = (state.get("components") or {}).get("venv")
    except (OSError, ValueError, AttributeError) as e:
        return [f"the record of what the update replaced ({state_file}) cannot be read: "
                f"{e}"], None
    if not info:
        return [], None
    try:
        held = (updates_dir / "backup" / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        held = None
    if info.get("backup_version") and held and info["backup_version"] != held:
        return [], None
    uc = _load_helper(install_root, "updater_components", code_root)
    if uc is None or not hasattr(uc, "venv_changes"):
        return [f"could not load the venv restore helper; the copy kept at "
                f"{info.get('venv_copy')} holds the previous packages"], None
    copy = uc._own_copy(info, snap)
    if copy is None:
        return [f"{info.get('venv_copy')} is not a venv copy this update kept"], None
    venv = install_root / ".venv"
    only = info.get("dists") if isinstance(info.get("dists"), list) else None
    torch_copy = uc._own_copy((state.get("components") or {}).get("pytorch"), snap)
    changed = uc.venv_changes(copy, venv, own=True)
    if changed is None:
        return [f"could not read the packages of {venv} or of {copy}"], None
    if only is not None:
        changed = [n for n in changed if n in only]
    if not changed and torch_copy is None:
        return [], None
    now = snap.with_name(snap.name + "-before-rollback")
    try:
        uc._clone_venv(venv, uc._venv_copy_path(venv, now, "deps"), now)
    except OSError as e:
        return [f"could not keep a copy of the current packages: {e}"], None
    keep = uc._venv_copy_path(venv, now, "deps")

    def undo():
        return uc._restore_venv(keep, venv, own=True)
    undo.keep = keep
    # The component step's PyTorch copy, then the dependency step's: reverse order.
    errors = uc._restore_venv(torch_copy, venv) if torch_copy is not None else []
    if not errors:
        changed = uc.venv_changes(copy, venv, own=True) or []
        if only is not None:
            changed = [n for n in changed if n in only]
        errors = uc._restore_venv(copy, venv, own=True, only=changed)
    if errors:
        back = undo()
        import shutil
        shutil.rmtree(keep, ignore_errors=True)
        return errors + back, None
    return [], undo


def _read_names(updates_dir: Path, backup_dir: Path) -> list:
    """The full set of top-level entries to unwind: the recorded swap manifest (which
    includes brand-new entries the update ADDED, absent from the backup), else the
    backup-dir listing for an older backup written before the manifest existed."""
    manifest = updates_dir / "applied_names.json"
    if manifest.is_file():
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and all(isinstance(x, str) for x in loaded):
                return loaded
        except (OSError, ValueError):
            pass
        # The manifest exists but is unreadable or malformed: roll back from the backup
        # dir instead, whose listing lacks the top-level entries the update added.
        print("Warning: applied_names.json exists but is unreadable; new files added by "
              "the update will not be removed by this rollback.", file=sys.stderr)
    return [p.name for p in backup_dir.iterdir()]


def main(argv=None, *, install_root: Path = INSTALL_ROOT, code_root=None) -> int:
    """Run the rollback. *code_root*, when given, is the localm tree of the build
    that applied the update (its update backup); the restore helpers are loaded
    from it before the install's own."""
    ap = argparse.ArgumentParser(
        description="Restore the previous localm build after a bad update.")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="roll back without the confirmation prompt")
    args = ap.parse_args(argv)

    home = _detect_home(install_root)
    updates_dir = home / "updates"
    backup_dir = updates_dir / "backup"
    if not backup_dir.is_dir() or not any(backup_dir.iterdir()):
        print(f"No update backup to roll back to (looked in {backup_dir}).\n"
              "Nothing to restore: either no update has been applied, or the data "
              "directory was not found. If your data lives elsewhere, set LOCALM_HOME "
              "and re-run.", file=sys.stderr)
        return 1

    names = _read_names(updates_dir, backup_dir)
    print(f"Roll back localm at {install_root}")
    print(f"  restoring the previous version from {backup_dir}")
    if not args.yes and sys.stdin.isatty():
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not rolled back.")
            return 1

    au = _load_apply_update(install_root, code_root)
    if au is None:
        print("Could not load the rollback helper (localm/_apply_update.py). The backup "
              f"is intact at {backup_dir}: to recover by hand, copy each entry there back "
              f"into {install_root}, overwriting the current files.", file=sys.stderr)
        return 2
    errors, undo = _restore_venv(install_root, updates_dir, code_root)
    if errors:
        print("Rollback FAILED: the previous packages could not be put back: "
              + "; ".join(errors) + f"\nNothing else was changed; the backup is kept at "
              f"{backup_dir}.", file=sys.stderr)
        return 2
    try:
        au.rollback(backup_dir, install_root, names)
    except Exception as e:
        back = undo() if undo else []
        packages = ("the packages were put back as they were" if not back else
                    "putting the packages back ALSO failed: " + "; ".join(back))
        print(f"Rollback FAILED: {e}\n{packages.capitalize()}. The backup is kept at "
              f"{backup_dir} for a manual restore; nothing was reported recovered that "
              "was not.", file=sys.stderr)
        return 2
    if undo:
        import shutil
        shutil.rmtree(undo.keep, ignore_errors=True)
    print("Rolled back to the previous version and its packages. Start localm again to "
          "use it; Settings > Updates shows any other part that still differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
