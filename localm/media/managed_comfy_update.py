# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: STAGE S4 `localm comfy update`.

The pin is a single CONSTANT (``COMFYUI_PINNED_COMMIT`` in managed_comfy_fresh.py).
It advances ONLY when a maintainer bumps that constant to a new, localm-tested
ComfyUI commit and ships it; a user then runs ``localm comfy update``, which moves
their managed checkout to the (new) pinned commit and RE-APPLIES the localm patch
set. It NEVER auto-advances - update only ever targets the shipped pin (or an
explicit ``--commit`` for an advanced/test override).

Safety: update records the current commit first and, on ANY failure, ROLLS BACK -
it puts every source file it overwrote or removed back from a backup and re-applies
the patch set, so a failed update leaves the working prior install exactly as it
was, not a broken in-between state. The new source comes as a GitHub source archive
(:mod:`localm.media.source_archive`), with no git program; the venv is not touched
by an update, so there is nothing to restore there.

Requirements are NOT reinstalled by default: a partial pip upgrade cannot be rolled
back exactly, so update stays within the guaranteeable source rollback. When the pinned
commit changes ComfyUI's requirements.txt, update SAYS SO and points at
``--reinstall-requirements`` (opt-in); passing it reinstalls into the existing venv,
and a failure there still rolls the source back and reports the failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from localm.config import load_config
from localm.debuglog import logger
from localm.media import managed_comfy as mc
from localm.media.comfy_patches import apply_patches
from localm.media.managed_comfy import _LOCK_OWNER as _LOCK_OWNER
from localm.media.managed_comfy import _acquire_update_lock, _release_update_lock
from localm.media.managed_comfy import _update_lock_path as _update_lock_path
from localm.media import source_archive
from localm.media.managed_comfy_fresh import (
    COMFYUI_PINNED_COMMIT, COMFYUI_PINNED_VERSION, COMFYUI_REPO)
from localm.media.managed_comfy_provision import (
    MARKER_FILENAME, ProgressCb, ProvisionResult, _emit, _run, _tail)


# --------------------------------------------------------------------------- #
#  Single-flight: only ONE process may mutate the managed checkout at a time.  #
# --------------------------------------------------------------------------- #
# The lock is an atomic mkdir at a sibling path of the checkout, shared with
# remove_managed_comfy and taken by every localm process that mutates the
# checkout (the GUI spawns `python -m localm comfy update` as a child process).
# The helpers live in managed_comfy and are imported above; _update_lock_path
# and _LOCK_OWNER are re-exported from here.


def _rev_parse_head(root: Path) -> Optional[str]:
    """The commit of the managed source tree at *root*
    (:func:`localm.media.source_archive.tree_commit`), or None when it has no
    source record and is not a git checkout (e.g. installed via the non-git
    copytree fallback). None is the signal that a pin update is not possible -
    we never fake one."""
    return source_archive.tree_commit(root)


def _requirements_changed(root: Path, new_tree: Path) -> bool:
    """True when requirements.txt of the tree at *new_tree* differs from the one
    at *root*; an unreadable file counts as changed, so update warns rather than
    skipping a real dep bump."""
    try:
        return (root / "requirements.txt").read_bytes() != (new_tree / "requirements.txt").read_bytes()
    except OSError:
        return True


def _staging(root: Path, what: str) -> Path:
    """A fresh, absent folder beside *root* for *what* (a download or a backup)."""
    path = root.parent / f".{root.name}-{what}-{os.getpid()}"
    if path.exists():
        mc.rmtree_robust(path)
    return path


def _new_backup(root: Path) -> Path:
    """An absent folder beside *root* for a backup of its files. A backup an
    earlier failed update kept is left alone: the next free numbered name is
    used instead."""
    base = root.parent / f".{root.name}-backup-{os.getpid()}"
    path, n = base, 1
    while os.path.lexists(source_archive.long_path(path)):
        path, n = base.with_name(f"{base.name}-{n}"), n + 1
    return path


def _replace_source(root: Path, repo: str, commit: str, changes: list,
                    on_progress: ProgressCb) -> tuple:
    """Make the source tree at *root* that of *commit* of *repo*: fetch it as an
    archive beside *root*, then :func:`source_archive.apply_tree` it over the
    files *root*'s tree holds. Appends the :class:`source_archive.TreeChange` to
    *changes*. Returns ``(ok, detail, deps_changed)``. On failure *root* is
    unchanged, except when a file could not be put back: the failed change is
    then appended to *changes* too, so a rollback of *changes* tries it again and
    reports what still could not be put back."""
    old_files = source_archive.tree_files(root)
    if old_files is None:
        return False, f"the files of the source tree at {root} could not be read", False
    new = _staging(root, "download")
    try:
        try:
            source_archive.fetch_tree(repo, commit, new,
                                      on_progress=lambda m: _emit(on_progress, m))
        except (source_archive.ArchiveError, OSError) as e:
            return False, str(e), False
        deps_changed = _requirements_changed(root, new)
        try:
            changes.append(source_archive.apply_tree(new, root, old_files,
                                                     _new_backup(root)))
        except source_archive.TreeRollbackError as e:
            changes.append(e.change)
            return False, f"could not write the new source into {root}: {e.cause}", False
        except OSError as e:
            return False, f"could not write the new source into {root}: {e}", False
        return True, "", deps_changed
    finally:
        if new.exists():
            mc.rmtree_robust(new)


def _update_marker(root: Path, target_commit: str, target_version: Optional[str],
                   prev_commit: str, patch_outcomes) -> None:
    """Record the update in the managed dir's marker (provenance; not load-bearing).
    Merges into any existing marker so the original source (fresh/copy) is preserved."""
    marker_path = root / MARKER_FILENAME
    data = {}
    try:
        if marker_path.is_file():
            data = json.loads(marker_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
    except (OSError, ValueError, RecursionError):
        data = {}
    data.update({
        "stage": "S4",
        "commit": target_commit,
        "comfyui_version": (target_version if target_version is not None
                            else (COMFYUI_PINNED_VERSION
                                  if target_commit == COMFYUI_PINNED_COMMIT else None)),
        "updated_from_commit": prev_commit,
        "localm_patches": {o.name: o.status for o in patch_outcomes},
    })
    try:
        marker_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logger.debug("could not write managed-comfy update marker: %s", e)


def _sync_custom_nodes(root: Path, on_progress: ProgressCb, say, changes: list) -> tuple:
    """Move every installed pinned custom node to its pinned commit, appending
    each :class:`source_archive.TreeChange` to *changes*. Returns
    ``(ok, detail)``."""
    from localm.media.managed_comfy_fresh import required_custom_nodes
    for node in required_custom_nodes().nodes:
        d = root / "custom_nodes" / node.name
        head = _rev_parse_head(d) if d.is_dir() else None
        if head is None or head == node.commit:
            continue
        say(f"Moving custom node {node.name} to {node.commit[:12]} (from {head[:12]}) ...")
        ok, detail, _ = _replace_source(d, node.repo, node.commit, changes, on_progress)
        if not ok:
            return False, f"custom node {node.name}: {detail}"
    return True, ""


def _sync_pinned_torch(venv_python: Path, on_progress: ProgressCb, say) -> tuple:
    """Install the PyTorch this machine's managed ComfyUI uses
    (``comfy_torch_spec``) when the managed venv does not have the exact
    versions it pins. A spec without exact versions changes nothing. Returns
    ``(ok, detail)``."""
    from localm import updater_components as uc
    from localm.media.managed_comfy_fresh import comfy_torch_install_args, comfy_torch_spec
    args = comfy_torch_install_args(comfy_torch_spec())
    pinned = [r for r in uc._requirements_in(args) if "==" in r]
    site = uc.site_packages(venv_python.parent.parent)
    if not pinned or site is None:
        return True, ""
    missing = uc.unsatisfied(pinned, uc.dists(site))
    if not missing:
        return True, ""
    say("Installing the pinned PyTorch into the managed venv (" + ", ".join(missing) + ") ...")
    ok, out = _run([str(venv_python), "-m", "pip", "install", "--disable-pip-version-check",
                    *args], on_progress=on_progress, timeout=7200)
    return ok, ("" if ok else _tail(out))


def update_managed_comfy(cfg: Optional[dict] = None, *, on_progress: ProgressCb = None,
                         comfyui_repo: Optional[str] = None,
                         target_commit: Optional[str] = None,
                         target_version: Optional[str] = None,
                         reinstall_requirements: bool = False) -> ProvisionResult:
    """Advance localm's managed ComfyUI to the pinned commit and re-apply the localm
    patch set, rolling back safely on any failure.

    ``comfyui_repo`` / ``target_commit`` / ``target_version`` are injection points for
    an offline test; production callers pass none and get the shipped pin. Returns an
    honest ProvisionResult: ``status`` is "updated" on success, "noop" when already at
    the pin (patches re-verified), else an error status with the prior install intact."""
    cfg = cfg if cfg is not None else load_config()
    repo = comfyui_repo or COMFYUI_REPO
    target = target_commit or COMFYUI_PINNED_COMMIT
    paths = mc.managed_comfy_paths()
    root = paths.root
    log: list = []

    def _say(line: str) -> None:
        log.append(line)
        _emit(on_progress, line)

    def _result(ok: bool, status: str, message: str) -> ProvisionResult:
        return ProvisionResult(ok=ok, status=status, managed_root=root,
                               log="\n".join(log), message=message,
                               commit=target if ok else None)

    if not mc.is_managed_comfy_installed():
        return _result(False, "absent",
                       "No managed ComfyUI is installed. Run 'localm comfy setup' first.")

    prev_commit = _rev_parse_head(root)
    if prev_commit is None:
        return _result(False, "error",
                       "The managed ComfyUI has no recorded source commit (it was installed "
                       "via the non-git copy fallback), so a pinned-version update is not "
                       "possible. Reinstall it: 'localm comfy remove' then "
                       "'localm comfy setup'.")

    # Everything from here on MUTATES the working tree (source replace, patch, and
    # the rollback's own restore+patch). Take the single-flight lock first; a second
    # update is refused honestly rather than interleaved. Read-only refusals above
    # (not installed, no recorded commit) never take it.
    acquired, busy = _acquire_update_lock()
    if not acquired:
        return _result(False, "busy", busy)
    changes: list = []

    def _rollback(reason: str) -> ProvisionResult:
        """Put back every source file the update overwrote or removed (ComfyUI and
        its custom nodes) and re-apply the patch set, then report *reason*. If a
        file cannot be put back - OR the localm patch set cannot be re-applied on
        the restored source - say so rather than report a clean tree."""
        note = ""
        failed_files = []
        kept = []
        for change in reversed(changes):
            failed = change.rollback()
            if failed:
                failed_files += failed
                kept.append(source_archive.plain_path(change.backup))
        if not failed_files:
            failed = [o for o in apply_patches(root) if not o.ok]
            if failed:
                detail = "; ".join(f"{o.name}: {o.detail}" for o in failed)
                note = (f" The localm patch set could not be re-applied on the restored "
                        f"install ({detail}); run 'localm comfy update' again or reinstall "
                        "it with 'localm comfy remove' then 'localm comfy setup'.")
                _emit(on_progress, f"Rolled back to the previous ComfyUI "
                      f"({prev_commit[:12]}), but re-applying localm's patches failed.")
            elif changes:
                _emit(on_progress,
                      f"Rolled back to the previous ComfyUI ({prev_commit[:12]}).")
            for change in changes:
                change.discard()
        else:
            note = (f" The rollback to {prev_commit[:12]} ALSO failed ({len(failed_files)} "
                    f"file(s) could not be put back, e.g. {failed_files[0]}); the managed "
                    "ComfyUI may be in a mixed state - reinstall it with 'localm comfy "
                    "remove' then 'localm comfy setup'. The original files are kept in "
                    f"{', '.join(kept)}.")
        return _result(False, "error", reason + note)

    try:
        # 1) Move to the target pin: fetch its source archive beside the install and
        #    write its files over the current ones (venv/models/custom_nodes/outputs
        #    are not source files and are kept). Nothing changes before the archive
        #    is downloaded and verified.
        deps_changed = False
        if target == prev_commit:
            _say(f"Already at the pinned ComfyUI ({target[:12]}); re-verifying patches ...")
        else:
            _say(f"Fetching ComfyUI {target[:12]} (from {prev_commit[:12]}) from {repo} ...")
            ok, detail, deps_changed = _replace_source(root, repo, target, changes,
                                                       on_progress)
            if not ok:
                return _rollback(f"Could not update ComfyUI to {target[:12]}: {detail}")

        # 2) Optional dependency reinstall (opt-in: a partial pip upgrade is not exactly
        #    rollback-able, so it is off by default and only the source rollback is
        #    guaranteed).
        if reinstall_requirements:
            ok, detail = _sync_pinned_torch(paths.venv_python, on_progress, _say)
            if not ok:
                return _rollback(f"Installing the pinned PyTorch failed: {detail}")
            ok, detail = _sync_custom_nodes(root, on_progress, _say, changes)
            if not ok:
                return _rollback(f"Updating a pinned custom node failed: {detail}")
            req = root / "requirements.txt"
            if req.is_file():
                _say("Reinstalling ComfyUI requirements into the managed venv ...")
                ok, out = _run([str(paths.venv_python), "-m", "pip", "install",
                                "--disable-pip-version-check", "-r", str(req)],
                               on_progress=on_progress, timeout=3600)
                if not ok:
                    return _rollback(
                        f"Reinstalling ComfyUI requirements failed: {_tail(out)}")

        # 3) Re-apply the localm patch set at the new pin. A FAILED patch (could not be
        #    applied safely) rolls back; a SKIPPED patch (upstream already fixed it) is
        #    fine and expected as the pin advances past the bug.
        outcomes = apply_patches(root)
        failed = [o for o in outcomes if not o.ok]
        if failed:
            detail = "; ".join(f"{o.name}: {o.detail}" for o in failed)
            return _rollback(f"Re-applying localm patches failed: {detail}")
        for o in outcomes:
            _say(f"patch {o.name}: {o.status} ({o.detail})")

        # 4) Prove it still reads as installed, or roll back.
        if not mc.is_managed_comfy_installed():
            return _rollback("After the update the managed ComfyUI does not read as "
                             "installed (main.py or venv missing at the new pin).")

        # 5) Record provenance.
        _update_marker(root, target, target_version, prev_commit, outcomes)
        for change in changes:
            change.discard()

        if target == prev_commit:
            msg = (f"localm's managed ComfyUI is already at the pinned ComfyUI "
                   f"({target[:12]}); localm patches re-verified.")
            status = "noop"
        else:
            msg = (f"Updated localm's managed ComfyUI to ComfyUI {target[:12]} "
                   f"and re-applied localm's patches.")
            status = "updated"
        if deps_changed and not reinstall_requirements:
            msg += (" ComfyUI's requirements.txt changed at this pin; if ComfyUI "
                    "misbehaves, re-run with --reinstall-requirements.")
        return _result(True, status, msg)
    except Exception as e:  # last resort: roll back, never a half-claimed success
        logger.debug("update_managed_comfy failed", exc_info=True)
        return _rollback(f"Update failed: {e}")
    finally:
        # Released on EVERY path - success, error, rollback, and the rollback-also-
        # failed path above. A lock left behind by a crash is self-healing (the next
        # caller finds our dead pid and reclaims it), but leaking one on an ordinary
        # error would wedge every later update until someone deleted a folder by hand.
        _release_update_lock()
