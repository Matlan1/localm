# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: STAGE S4 `localm comfy update`.

The pin is a single CONSTANT (``COMFYUI_PINNED_COMMIT`` in managed_comfy_fresh.py).
It advances ONLY when a maintainer bumps that constant to a new, localm-tested
ComfyUI commit and ships it; a user then runs ``localm comfy update``, which moves
their managed checkout to the (new) pinned commit and RE-APPLIES the localm patch
set. It NEVER auto-advances - update only ever targets the shipped pin (or an
explicit ``--commit`` for an advanced/test override).

Safety: the new source, with the localm patch set applied, and the Python packages it
needs are fetched into a folder of their own inside the managed ComfyUI. Only files that
differ are then moved in, under a record (``localm.update_mechanism.steps``): a failure
puts every file back at once, and an update whose process ends is undone by the next
update or the next use of the managed ComfyUI. The new source comes as a GitHub source
archive (:mod:`localm.media.source_archive`), with no git program.

Requirements are NOT reinstalled by default. When the pinned commit changes ComfyUI's
requirements.txt, update SAYS SO and points at ``--reinstall-requirements`` (opt-in). With
it, pip's own dry run names what it would install; that is fetched first, the packages it
replaces are moved aside, and pip installs the fetched wheels with no network.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from localm.config import load_config
from localm.debuglog import logger
from localm.media import comfy_patches
from localm.media import managed_comfy as mc
from localm.media.comfy_patches import apply_patches
from localm.media.managed_comfy import _LOCK_OWNER as _LOCK_OWNER
from localm.media.managed_comfy import _acquire_update_lock, _release_update_lock
from localm.media.managed_comfy import _update_lock_path as _update_lock_path
from localm.media import source_archive
from localm.media.managed_comfy_fresh import (
    COMFYUI_PINNED_COMMIT, COMFYUI_PINNED_VERSION, COMFYUI_REPO)
from localm.media.managed_comfy_provision import (
    MARKER_FILENAME, ProgressCb, ProvisionResult, _emit, _run)


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


class PrepareError(Exception):
    """The update could not be prepared; the managed ComfyUI is as it was."""


@dataclass
class Prepared:
    """An update of the managed ComfyUI, ready to be moved in. *removed*: files of the
    current source the new one does not have. *packages*: whether Python packages are
    staged."""
    target: str
    previous: str
    removed: list = field(default_factory=list)
    packages: bool = False
    outcomes: list = field(default_factory=list)
    deps_changed: bool = False


def _stage_tree(installed: Path, staged: Path, repo: str, commit: str,
                on_progress: ProgressCb) -> list:
    """Fetch *commit* of *repo* into *staged*. Returns the files of the tree at
    *installed* that it does not have. Raises :class:`PrepareError`."""
    old_files = source_archive.tree_files(installed)
    if old_files is None:
        raise PrepareError(f"the files of the source tree at {installed} could not be read")
    try:
        source_archive.fetch_tree(repo, commit, staged,
                                  on_progress=lambda m: _emit(on_progress, m))
        head = source_archive.git_head(installed)
        rec = source_archive.read_record(staged)
        if head is not None and rec is not None:
            source_archive.write_record(staged, rec.get("repo") or "", rec["commit"],
                                        rec["files"], git_head=head)
    except (source_archive.ArchiveError, OSError) as e:
        raise PrepareError(str(e)) from e
    new_files = set(source_archive.tree_files(staged) or [])
    return [f for f in old_files if f not in new_files]


def _stage_patches(root: Path, staged: Path) -> list:
    """Apply the localm patch set to the source in *staged*, with a copy of each
    installed file it changes that *staged* does not have. Returns the outcomes.
    Raises OSError."""
    for patch in comfy_patches.PATCHES:
        installed, copy = root / patch.target, staged / patch.target
        if installed.is_file() and not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(installed, copy)
    return apply_patches(staged)


def prepare(place, root: Path, *, repo: str, target: str, version: Optional[str],
            reinstall: bool, on_progress: ProgressCb, say) -> Prepared:
    """Prepare the update of the managed ComfyUI at *root* to *target* in *place*
    (``update_mechanism.steps.Place`` of *root*), with its pinned custom nodes and its
    Python packages when *reinstall*. Changes nothing of the installed one. Raises
    :class:`PrepareError`."""
    from localm.update_mechanism import disk, mover
    previous = _rev_parse_head(root)
    if previous is None:
        raise PrepareError("The managed ComfyUI has no recorded source commit")
    out = Prepared(target, previous)
    staged = place.new(mover.TREE)
    installed = disk.real_folder(root)
    try:
        if target != previous:
            say(f"Fetching ComfyUI {target[:12]} (from {previous[:12]}) from {repo} ...")
            staged.rmdir()
            out.removed += _stage_tree(installed, staged, repo, target, on_progress)
            out.deps_changed = _requirements_changed(installed, staged)
        else:
            say(f"Already at the pinned ComfyUI ({target[:12]}); re-verifying patches ...")
        if reinstall:
            _stage_nodes(root, installed, staged, out, on_progress, say)
        out.outcomes = _stage_patches(installed, staged)
        failed = [o for o in out.outcomes if not o.ok]
        if failed:
            detail = "; ".join(f"{o.name}: {o.detail}" for o in failed)
            raise PrepareError(f"Re-applying localm patches failed: {detail}")
        if reinstall:
            from localm.media import managed_comfy_packages
            requirements = (staged if (staged / "requirements.txt").is_file()
                            else installed) / "requirements.txt"
            out.packages = managed_comfy_packages.prepare(
                place, mc._venv_python_path(root), requirements, on_progress=on_progress,
                say=say, run=_run)
        _update_marker(staged, target, version, previous, out.outcomes, installed=installed)
    except OSError as e:
        raise PrepareError(f"Could not prepare the update: {e}") from e
    return out


def _stage_nodes(root: Path, installed: Path, staged: Path, out: Prepared,
                 on_progress: ProgressCb, say) -> None:
    """Stage every installed pinned custom node that is not at its pinned commit.
    Raises :class:`PrepareError`."""
    from localm.media.managed_comfy_fresh import required_custom_nodes
    for node in required_custom_nodes().nodes:
        rel = f"custom_nodes/{node.name}"
        d = root / "custom_nodes" / node.name
        head = _rev_parse_head(d) if d.is_dir() else None
        if head is None or head == node.commit:
            continue
        say(f"Moving custom node {node.name} to {node.commit[:12]} (from {head[:12]}) ...")
        try:
            gone = _stage_tree(installed / "custom_nodes" / node.name, staged / rel,
                               node.repo, node.commit, on_progress)
        except PrepareError as e:
            raise PrepareError(f"Updating a pinned custom node failed: custom node "
                               f"{node.name}: {e}") from e
        out.removed += [f"{rel}/{f}" for f in gone]


def stages(root: Path, update_id: str, removed: list, packages: bool) -> list:
    """The calls that move in what :func:`prepare` prepared for *update_id*, for
    ``update_mechanism.steps.change``."""
    from localm.update_mechanism import mover, steps, steps_packages
    calls: list = [lambda: steps.move_in(root, update_id, mover.TREE, remove=removed)]
    if packages:
        calls.append(lambda: steps_packages.run_pip(root, update_id))
    return calls


def package_indexes() -> list:
    """The index options of the PyTorch this machine's managed ComfyUI uses."""
    from localm.media.managed_comfy_fresh import comfy_torch_install_args, comfy_torch_spec
    from localm.update_mechanism import steps_packages
    return steps_packages.index_args([comfy_torch_install_args(comfy_torch_spec())])


def _reinstall(root: Path, update_id: str):
    """For ``update_mechanism.steps.change``: installs packages of the managed venv
    again that a way back names."""
    def again(named: list) -> None:
        from localm import uv_tool
        from localm.update_mechanism import job, mover, steps_packages
        record = job.read(root)
        uv = uv_tool.own_uv(uv_tool.install_root())
        if uv is None:
            raise mover.Refused("localm has no uv to install them with")
        steps_packages.reinstall(root, record.id if record else update_id, named[0]["step"],
                                 [(p["name"], p["version"]) for p in named], uv,
                                 env=dict(os.environ), indexes=package_indexes())
    return again


def why_unusable(packages: bool) -> Optional[str]:
    """None when the managed ComfyUI reads as installed and, after a change of its
    packages, its Python imports PyTorch; else what is wrong."""
    if not mc.is_managed_comfy_installed():
        return ("After the update the managed ComfyUI does not read as installed (main.py "
                "or venv missing at the new pin).")
    if not packages:
        return None
    from localm.media import managed_comfy_packages
    return managed_comfy_packages.why_broken(mc.managed_comfy_paths().venv_python, run=_run)


def _update_marker(root: Path, target_commit: str, target_version: Optional[str],
                   prev_commit: str, patch_outcomes, installed: Optional[Path] = None) -> None:
    """Record the update in the marker in *root* (provenance; not load-bearing). Merges
    into the marker in *installed* (default *root*) so the original source (fresh/copy)
    is preserved."""
    marker_path = root / MARKER_FILENAME
    data = {}
    out_path, marker_path = marker_path, (installed or root) / MARKER_FILENAME
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
        out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logger.debug("could not write managed-comfy update marker: %s", e)


def update_managed_comfy(cfg: Optional[dict] = None, *, on_progress: ProgressCb = None,
                         comfyui_repo: Optional[str] = None,
                         target_commit: Optional[str] = None,
                         target_version: Optional[str] = None,
                         reinstall_requirements: bool = False) -> ProvisionResult:
    """Advance localm's managed ComfyUI to the pinned commit with the localm patch set
    applied; a failure leaves the previous one.

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
                       "The managed ComfyUI has no recorded source commit (its source "
                       "record is missing or unreadable), so a pinned-version update is "
                       "not possible. Reinstall it: 'localm comfy remove' then "
                       "'localm comfy setup'.")

    # Everything from here on MUTATES the working tree. Take the single-flight lock first; a second
    # update is refused honestly rather than interleaved. Read-only refusals above
    # (not installed, no recorded commit) never take it.
    acquired, busy = _acquire_update_lock()
    if not acquired:
        return _result(False, "busy", busy)
    from localm.update_mechanism import steps
    try:
        unsettled = mc.settle_managed_comfy()
        if unsettled:
            return _result(False, "error", "An interrupted update of the managed ComfyUI could "
                           f"not be undone: {unsettled} If that cannot be mended, reinstall it: "
                           "'localm comfy remove' then 'localm comfy setup'.")
        with steps.Work(root) as work:
            try:
                prepared = prepare(work, root, repo=repo, target=target,
                                   version=target_version, reinstall=reinstall_requirements,
                                   on_progress=on_progress, say=_say)
            except PrepareError as e:
                why = str(e) if target == prev_commit else (
                    f"Could not update ComfyUI to {target[:12]}: {e}")
                return _result(False, "error", why)
            for o in prepared.outcomes:
                _say(f"patch {o.name}: {o.status} ({o.detail})")
            if prepared.packages:
                _say("Reinstalling ComfyUI requirements into the managed venv ...")
            try:
                steps.change(work, stages(root, work.id, prepared.removed, prepared.packages),
                             old=prev_commit, new=target,
                             prove=lambda: why_unusable(prepared.packages),
                             reinstall=_reinstall(root, work.id))
            except (steps.Failed, *steps.STEP_ERRORS) as e:
                note = ""
                if getattr(e, "errors", None):
                    note = (" Files could not be put back; the managed ComfyUI may be in a "
                            "mixed state until the next update or use of it puts them back.")
                else:
                    _emit(on_progress, f"Rolled back to the previous ComfyUI "
                                       f"({prev_commit[:12]}).")
                return _result(False, "error",
                               f"Could not update ComfyUI to {target[:12]}: {e}{note}")

        if target == prev_commit:
            msg = (f"localm's managed ComfyUI is already at the pinned ComfyUI "
                   f"({target[:12]}); localm patches re-verified.")
            status = "noop"
        else:
            msg = (f"Updated localm's managed ComfyUI to ComfyUI {target[:12]} "
                   f"and re-applied localm's patches.")
            status = "updated"
        if prepared.deps_changed and not reinstall_requirements:
            msg += (" ComfyUI's requirements.txt changed at this pin; if ComfyUI "
                    "misbehaves, re-run with --reinstall-requirements.")
        return _result(True, status, msg)
    except Exception as e:
        logger.debug("update_managed_comfy failed", exc_info=True)
        return _result(False, "error", f"Update failed: {e}")
    finally:
        # Released on EVERY path - success and error. A lock left behind by a crash is
        # self-healing (the next caller finds our dead pid and reclaims it), but leaking
        # one on an ordinary error would wedge every later update until someone deleted
        # a folder by hand.
        _release_update_lock()
