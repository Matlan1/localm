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
                line = marker.read_text(encoding="utf-8").strip()
                if line:
                    return Path(line).expanduser()
            except OSError:
                pass   # unreadable marker: fall through to ./home
        portable = install_root / "home"
        if portable.is_dir():
            return portable
    return install_root / "home"


def _load_apply_update(install_root: Path):
    """Load ``localm/_apply_update.py`` BY FILE PATH (bypassing the ``localm`` package
    __init__), so the rollback works even if the rest of localm is broken. Returns the
    module, or None when it cannot be loaded (the caller then prints manual guidance)."""
    path = install_root / "localm" / "_apply_update.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_localm_apply_update_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None   # a corrupt helper takes the manual-recovery path


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


def main(argv=None, *, install_root: Path = INSTALL_ROOT) -> int:
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

    au = _load_apply_update(install_root)
    if au is None:
        print("Could not load the rollback helper (localm/_apply_update.py). The backup "
              f"is intact at {backup_dir}: to recover by hand, copy each entry there back "
              f"into {install_root}, overwriting the current files.", file=sys.stderr)
        return 2
    try:
        au.rollback(backup_dir, install_root, names)
    except Exception as e:
        print(f"Rollback FAILED: {e}\nThe backup is kept at {backup_dir} for a manual "
              "restore; nothing was reported recovered that was not.", file=sys.stderr)
        return 2
    print("Rolled back to the previous version. Start localm again to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
