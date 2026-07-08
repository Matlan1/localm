# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: STAGE S4 `localm comfy update` (design decision 7).

The pin is a single CONSTANT (``COMFYUI_PINNED_COMMIT`` in managed_comfy_fresh.py).
It advances ONLY deliberately: a maintainer bumps that constant to a new, localm-
tested ComfyUI commit and ships it; a user then runs ``localm comfy update``, which
moves their managed checkout to the (new) pinned commit and RE-APPLIES the localm
patch set. It NEVER auto-advances - update only ever targets the shipped pin (or an
explicit ``--commit`` for an advanced/test override).

Safety (AGENTS.md rule 5: no facade, no half-updated install): update records the
current commit first and, on ANY failure, ROLLS BACK - it returns the managed source
to that prior commit and re-applies the patch set, so a failed update leaves the
working prior install exactly as it was, not a broken in-between state. The rollback
is git-based (the managed ComfyUI is a git checkout), so it is cheap and exact; the
venv is not touched by an update, so there is nothing to restore there.

Requirements are NOT reinstalled by default: a partial pip upgrade cannot be rolled
back exactly, so update stays within the guaranteeable git rollback. When the pinned
commit changes ComfyUI's requirements.txt, update SAYS SO and points at
``--reinstall-requirements`` (opt-in); passing it reinstalls into the existing venv,
and a failure there still rolls the source back and reports honestly.

Design + locked decisions: dev-notes/DESIGN-localm-managed-comfyui-2026-07-08.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from localm.config import load_config
from localm.debuglog import logger
from localm.media import managed_comfy as mc
from localm.media.comfy_patches import apply_patches
from localm.media.managed_comfy_fresh import (
    COMFYUI_PINNED_COMMIT, COMFYUI_PINNED_VERSION, COMFYUI_REPO)
from localm.media.managed_comfy_provision import (
    MARKER_FILENAME, ProgressCb, ProvisionResult, _emit, _run, _tail)


def _rev_parse_head(root: Path) -> Optional[str]:
    """The managed checkout's HEAD sha, or None when *root* is not a git checkout
    (e.g. installed via the non-git copytree fallback). None is the signal that a
    pin update is not possible - we never fake one."""
    ok, out = _run(["git", "-C", str(root), "rev-parse", "HEAD"], timeout=30)
    return out.strip() if ok else None


def _requirements_changed(root: Path, prev: str, target: str) -> bool:
    """True when requirements.txt differs between *prev* and *target*. ``git diff
    --quiet`` exits 0 for no change, non-zero for a change; a git error (unknown here)
    is treated as 'changed' so we warn rather than silently skip a real dep bump."""
    ok, _out = _run(["git", "-C", str(root), "diff", "--quiet", prev, target,
                     "--", "requirements.txt"], timeout=60)
    return not ok


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
    except (OSError, ValueError):
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
                       "The managed ComfyUI has no git history (it was installed via "
                       "the non-git copy fallback), so a pinned-version update is not "
                       "possible. Reinstall it: 'localm comfy remove' then "
                       "'localm comfy setup'.")

    def _rollback(reason: str) -> ProvisionResult:
        """Return the managed source to prev_commit and re-apply the patch set, then
        report *reason*. Rolls back the git source exactly; if even the rollback
        checkout fails, say so rather than pretend the tree is clean (rule 5)."""
        note = ""
        ok, out = _run(["git", "-C", str(root), "checkout", "--force", "--quiet",
                        prev_commit], on_progress=on_progress, timeout=300)
        if ok:
            apply_patches(root)  # restore the prior patch set on the restored source
            _emit(on_progress, f"Rolled back to the previous ComfyUI ({prev_commit[:12]}).")
        else:
            note = (f" The rollback to {prev_commit[:12]} ALSO failed ({_tail(out)}); the "
                    "managed ComfyUI may be in a mixed state - reinstall it with "
                    "'localm comfy remove' then 'localm comfy setup'.")
        return _result(False, "error", reason + note)

    try:
        # 1) Make the target commit available (a pin can be newer than the clone; for
        #    the S2 copy path 'origin' is the user's local dir, so fetch the canonical
        #    ComfyUI repo explicitly). A fetch failure is not yet destructive.
        _say(f"Fetching ComfyUI updates from {repo} ...")
        ok, out = _run(["git", "-C", str(root), "fetch", "--quiet", repo],
                       on_progress=on_progress, timeout=1800)
        if not ok:
            return _rollback(f"Could not fetch ComfyUI updates: {_tail(out)}")

        # Heads-up (before we move) when the pin changes ComfyUI's dependencies.
        deps_changed = _requirements_changed(root, prev_commit, target)

        # 2) Move to the target pin. --force discards our patch edits to tracked files
        #    (they are re-applied below); untracked venv/models/custom_nodes are kept.
        if target == prev_commit:
            _say(f"Already at the pinned ComfyUI ({target[:12]}); re-verifying patches ...")
        else:
            _say(f"Checking out ComfyUI {target[:12]} (from {prev_commit[:12]}) ...")
        ok, out = _run(["git", "-C", str(root), "checkout", "--force", "--quiet", target],
                       on_progress=on_progress, timeout=300)
        if not ok:
            return _rollback(f"Could not check out ComfyUI {target[:12]}: {_tail(out)}")

        # 3) Optional dependency reinstall (opt-in: a partial pip upgrade is not exactly
        #    rollback-able, so it is off by default and only the source rollback is
        #    guaranteed).
        if reinstall_requirements:
            req = root / "requirements.txt"
            if req.is_file():
                _say("Reinstalling ComfyUI requirements into the managed venv ...")
                ok, out = _run([str(paths.venv_python), "-m", "pip", "install",
                                "--disable-pip-version-check", "-r", str(req)],
                               on_progress=on_progress, timeout=3600)
                if not ok:
                    return _rollback(
                        f"Reinstalling ComfyUI requirements failed: {_tail(out)}")

        # 4) Re-apply the localm patch set at the new pin. A FAILED patch (could not be
        #    applied safely) rolls back; a SKIPPED patch (upstream already fixed it) is
        #    fine and expected as the pin advances past the bug.
        outcomes = apply_patches(root)
        failed = [o for o in outcomes if not o.ok]
        if failed:
            detail = "; ".join(f"{o.name}: {o.detail}" for o in failed)
            return _rollback(f"Re-applying localm patches failed: {detail}")
        for o in outcomes:
            _say(f"patch {o.name}: {o.status} ({o.detail})")

        # 5) Prove it still reads as installed, or roll back.
        if not mc.is_managed_comfy_installed():
            return _rollback("After the update the managed ComfyUI does not read as "
                             "installed (main.py or venv missing at the new pin).")

        # 6) Record provenance.
        _update_marker(root, target, target_version, prev_commit, outcomes)

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
