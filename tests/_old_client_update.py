# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the updater of an older localm (0.3.3 to 0.3.4b4) leaves behind when it installs
a build with its own code: the backup of the replaced top-level entries with its list
(``_apply_update.BACKUP_RECORD``), the copy of the venv and the components it may change
(``updater_components.snapshot_dir``), the swapped files, ``applied_names.json``, the
dependency step and the pending runtime and component updates. This build restores such
a backup (``updater.rollback_last``, ``scripts/rollback_update.py``) and carries out the
steps it left (``updater_runtime.finish_pending``, ``updater_components.finish_pending``),
so tests make one with :func:`old_client_update`."""

import json
import shutil
import subprocess
from pathlib import Path

from localm import _apply_update as au
from localm import updater, updater_components as uc, updater_runtime


def _deps(root: Path) -> frozenset:
    data = updater._read_toml(root / "pyproject.toml")
    proj = data.get("project", {}) if isinstance(data, dict) else {}
    deps = set(proj.get("dependencies", []) or [])
    for extra, items in (proj.get("optional-dependencies", {}) or {}).items():
        deps.update(f"{extra}:{it}" for it in items or [])
    return frozenset(deps)


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def old_client_update(inst, data: bytes, *, runner=None, defer: bool = False,
                      say=None) -> dict:
    """Install the build zip *data* into the install *inst* the way an older localm's
    updater does: back up, swap, run the dependency step through *runner* (``cmd ->
    exit status``, default ``subprocess.run``) when the dependencies changed, then record
    the pending runtime and component updates and, unless *defer*, carry them out.
    Returns ``{version, backup, names, runtime, components}``. Raises RuntimeError when
    the dependency step fails (the older updater rolls back then; tests of this build
    do not need that)."""
    inst = Path(inst)
    run = runner or (lambda c: subprocess.run(c, cwd=str(inst)).returncode)
    updir = updater._updates_dir()
    backup_dir = updater._backup_dir(create=True)
    work = updir / "runs" / "old-client"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    zip_path = work / "build.zip"
    zip_path.write_bytes(data)
    root = au.extract(zip_path, work / "staging")
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    names = sorted(p.name for p in root.iterdir() if p.name not in au.NEVER_TOUCH)
    rt_plan = updater_runtime.plan(root, inst)
    comp_plans = uc.plan(root, inst, deep=False)
    deps_changed = _deps(root) != _deps(inst)

    shutil.rmtree(backup_dir, ignore_errors=True)
    backup_dir.mkdir(parents=True)
    for name in names:
        if (inst / name).exists():
            _copy(inst / name, backup_dir / name)
    files = sorted(p.relative_to(backup_dir).as_posix() for p in backup_dir.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    (backup_dir / au.BACKUP_RECORD).write_text(
        json.dumps({"names": names, "files": files}), encoding="utf-8")
    for name in names:
        dst = inst / name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        _copy(root / name, dst)
    shutil.rmtree(work, ignore_errors=True)
    (updir / "applied_names.json").write_text(json.dumps(names), encoding="utf-8")

    snap = uc.snapshot_dir(backup_dir)
    if deps_changed:
        uc.snapshot([uc.DEPS], snap, root=inst, for_backup=uc.backup_version(backup_dir))
        cmd = au.post_swap_command("deps", uv=updater._uv_for_deps(inst))
        rc = int(run(cmd))
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} exited {rc}")
        uc.record_changes(snap, uc.DEPS, inst)
    runtime = components = None
    if rt_plan["reprovision"]:
        updater_runtime.write_pending(rt_plan, root=inst, version=version)
        if not defer:
            runtime = updater_runtime.finish_pending(root=inst, runner=runner,
                                                     env=updater._deps_env(inst), say=say)
    if uc.needs_finish(comp_plans):
        uc.write_pending(comp_plans, root=inst, version=version)
        if not defer:
            components = uc.finish_pending(root=inst, runner=runner,
                                           env=updater._deps_env(inst), say=say)
    return {"version": version, "backup": str(backup_dir), "names": names,
            "runtime": runtime, "components": components}
