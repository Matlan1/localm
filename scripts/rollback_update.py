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

It restores the code, the venv and the native runtime. ComfyUI and the uv inside
localm take localm's own modules to restore, so it leaves them, says so and records
it; the next start of localm brings them to the pins of the version it restored.

It NEVER reports success it did not achieve: if there is no backup, the backup is
incomplete, the restore helper cannot load, or a restore step fails, it says so, keeps
the backup intact, and exits non-zero, pointing at the backup dir for a manual copy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# The install root is the parent of this script's scripts/ dir, derived from __file__.
INSTALL_ROOT = Path(__file__).resolve().parent.parent
# The last update results ``localm.updater_runtime`` and ``localm.updater_components``
# read, in the updates folder.
_RUNTIME_STATUS = "runtime-update.json"
_COMPONENTS_STATUS = "components-update.json"
# The parts an update's snapshot can hold that this script does not restore.
_PARTS_LEFT = (("comfyui", "ComfyUI"), ("uv", "the uv inside localm"))


def _detect_home(install_root: Path) -> Path:
    """The localm data dir, resolved WITHOUT importing localm.config (which may live in
    the broken build). Mirrors config._detect_home: LOCALM_HOME, else a portable
    ``localm-home.cfg`` (read by ``localm/_home_cfg.py``, loaded by file path; a
    relative line is relative to the install) / ``./home`` in the install, else the
    contained ``./home``. A marker that cannot be read, or a reader that cannot be
    loaded, is reported on stderr and the contained ``./home`` is used."""
    env = os.environ.get("LOCALM_HOME", "").strip()
    if env:
        return Path(env).expanduser().absolute()
    if (install_root / "pyproject.toml").is_file():
        marker = install_root / "localm-home.cfg"
        if marker.is_file():
            reader = _load_helper(install_root, "_home_cfg")
            if reader is None:
                print(f"  Cannot read {marker}: localm/_home_cfg.py could not be "
                      "loaded; using the folder home in the install.", file=sys.stderr)
            else:
                try:
                    line = reader.read_line(marker)
                    if line:
                        return reader.data_dir(line, install_root)
                except (OSError, ValueError, LookupError) as e:
                    print(f"  Cannot read {marker} ({e}); using the folder home in "
                          "the install.", file=sys.stderr)
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


def _held_version(updates_dir: Path):
    """The version the update backup holds, or None when it cannot be read."""
    try:
        return (updates_dir / "backup" / "VERSION").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None


def _parts_left(updates_dir: Path) -> list:
    """The names of the parts the update kept a copy of that this script does not
    restore, unless they were kept for another backup than the one being restored.
    Empty when the record cannot be read (:func:`_restore_venv` reports that)."""
    try:
        state = json.loads((updates_dir / "backup-components" / "state.json")
                           .read_text(encoding="utf-8"))
        comps, kept_for = state.get("components") or {}, state.get("backup_version")
    except (OSError, ValueError, AttributeError):
        return []
    held = _held_version(updates_dir)
    if kept_for and held and kept_for != held:
        return []
    return [label for name, label in _PARTS_LEFT if name in comps]


def _left_note(left: list) -> str:
    """The sentence naming the parts *left* behind."""
    return (" and ".join(left) + " were not put back; if the update changed them, the next "
            "start of localm with an internet connection returns them to this version's pins")


def _record_rollback(updates_dir: Path, install_root: Path, left: list) -> None:
    """Record the rollback as the last runtime and component update result, the way
    ``localm.updater._record_rollback`` does, so Settings > Updates and ``localm
    doctor`` do not go on reporting what the rolled-back update did. A file that
    cannot be written is reported on stderr and does not fail the rollback."""
    try:
        version = (install_root / "VERSION").read_text(encoding="utf-8").strip() or None
    except (OSError, ValueError):
        version = None
    note = f"the update was rolled back to localm {version or '(unknown version)'}"
    if left:
        note += "; " + _left_note(left)
    common = {"state": "rolled_back", "note": note, "error": None, "version": version,
              "at": time.time()}
    for name, fields in ((_RUNTIME_STATUS, {"backend": None, "from": None, "to": None,
                                            "holders": []}),
                         (_COMPONENTS_STATUS, {"components": [],
                                               "root": str(install_root.resolve())})):
        try:
            (updates_dir / name).write_text(json.dumps({**common, **fields}), encoding="utf-8")
        except OSError as e:
            print(f"Warning: could not record the rollback in {updates_dir / name}: {e}; "
                  "Settings > Updates may still show what the update did.", file=sys.stderr)


def _restore_venv(install_root: Path, updates_dir: Path, code_root=None):
    """Put back the packages the update changed, kept in ``<updates>/backup-components``:
    what its PyTorch step changed, then what its dependency step changed (localm's own
    install record included), unless they were kept for another backup than the one
    being restored. Returns ``(errors, undo)``: the failures (empty when there is
    nothing to restore or it was restored) and a callable that puts the venv back as it
    was before this call (None when nothing was changed)."""
    snap = updates_dir / "backup-components"
    state_file = snap / "state.json"
    if not state_file.exists():
        return [], None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        comps = state.get("components") or {}
    except (OSError, ValueError, AttributeError) as e:
        return [f"the record of what the update replaced ({state_file}) cannot be read: "
                f"{e}"], None
    deps, torch = comps.get("venv"), comps.get("pytorch")
    if not deps and not torch:
        return [], None
    held = _held_version(updates_dir)
    if state.get("backup_version") and held and state["backup_version"] != held:
        return [], None
    uc = _load_helper(install_root, "updater_components", code_root)
    if uc is None or not hasattr(uc, "venv_changes"):
        return ["could not load the package restore helper; the copies kept beside the "
                "venv hold the previous packages"], None
    venv = install_root / ".venv"
    steps = []
    for info, own in ((torch, False), (deps, True)):
        if not info:
            continue
        copy = uc._own_copy(info, snap)
        if copy is None:
            return [f"{info.get('venv_copy')} is not a venv copy this update kept"], None
        steps.append((copy, own, info.get("dists") if isinstance(info.get("dists"), list)
                      else None))
    now = snap.with_name(snap.name + "-before-rollback")
    keep = uc._venv_copy_path(venv, now, "deps")
    try:
        uc._clone_venv(venv, keep, now)
    except Exception as e:
        return [f"could not keep a copy of the current packages: {e}"], None

    def undo():
        return uc._restore_venv(keep, venv, own=True)
    undo.keep = uc._long(keep)
    errors, restored = [], False
    try:
        for copy, own, only in steps:
            changed = uc.venv_changes(copy, venv, own=own)
            if changed is None:
                errors = [f"could not read the packages of {venv} or of {copy}"]
                break
            if only is not None:
                changed = [n for n in changed if n in only]
            if changed:
                restored = True
                errors = uc._restore_venv(copy, venv, own=own, only=changed)
                if errors:
                    break
    except Exception as e:
        errors = [f"putting the packages back stopped with an error: {e}"]
    if errors or not restored:
        try:
            back = undo() if errors else []
        except Exception as e:
            back = [f"putting the current packages back stopped with an error: {e}"]
        import shutil
        shutil.rmtree(undo.keep, ignore_errors=True)
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
        except (OSError, ValueError, RecursionError):
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

    print(f"Roll back localm at {install_root}")
    print(f"  restoring the previous version from {backup_dir}")
    au = _load_apply_update(install_root, code_root)
    if au is None:
        print("Could not load the rollback helper (localm/_apply_update.py). The backup "
              f"is intact at {backup_dir}: to recover by hand, copy each entry there back "
              f"into {install_root}, overwriting the current files.", file=sys.stderr)
        return 2
    # A helper from a build before these checks has neither function.
    names = au.backup_names(backup_dir) if hasattr(au, "backup_names") else None
    if names is None:
        names = _read_names(updates_dir, backup_dir)
    problem = au.backup_problem(backup_dir, names) if hasattr(au, "backup_problem") else None
    if problem:
        print(f"Rollback refused: the backup is incomplete: {problem} in {backup_dir}.\n"
              "Nothing was changed; the next update replaces it.", file=sys.stderr)
        return 2
    if not args.yes and sys.stdin.isatty():
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not rolled back.")
            return 1
    try:
        errors, undo = _restore_venv(install_root, updates_dir, code_root)
    except Exception as e:
        errors, undo = [f"the package restore helper stopped with an error: {e}"], None
    if errors:
        print("Rollback FAILED: the previous packages could not be put back: "
              + "; ".join(errors) + f"\nNothing else was changed; the backup is kept at "
              f"{backup_dir}.", file=sys.stderr)
        return 2
    try:
        au.rollback(backup_dir, install_root, names)
    except Exception as e:
        try:
            back = undo() if undo else []
        except Exception as undo_error:
            back = [f"the package restore helper stopped with an error: {undo_error}"]
        packages = ("the packages were put back as they were" if not back else
                    "putting the packages back ALSO failed: " + "; ".join(back))
        print(f"Rollback FAILED: {e}\n{packages[0].upper()}{packages[1:]}. The backup is kept at "
              f"{backup_dir} for a manual restore; nothing was reported recovered that "
              "was not.", file=sys.stderr)
        return 2
    if undo:
        import shutil
        shutil.rmtree(undo.keep, ignore_errors=True)
    left = _parts_left(updates_dir)
    _record_rollback(updates_dir, install_root, left)
    print("Rolled back to the previous version" + (" and its packages" if undo else "")
          + ". Start localm again to use it; Settings > Updates shows any other part that "
            "still differs.")
    if left:
        note = _left_note(left)
        print(f"{note[0].upper()}{note[1:]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
