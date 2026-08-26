# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-managed ComfyUI: STAGE S4 `localm comfy update`.

The pin is a single CONSTANT (``COMFYUI_PINNED_COMMIT`` in managed_comfy_fresh.py).
It advances ONLY when a maintainer bumps that constant to a new, localm-tested
ComfyUI commit and ships it; a user then runs ``localm comfy update``, which moves
their managed checkout to the (new) pinned commit and RE-APPLIES the localm patch
set. It NEVER auto-advances - update only ever targets the shipped pin (or an
explicit ``--commit`` for an advanced/test override).

Safety: update records the current commit first and, on ANY failure, ROLLS BACK -
it returns the managed source to that prior commit and re-applies the patch set, so
a failed update leaves the working prior install exactly as it was, not a broken
in-between state. The rollback is git-based (the managed ComfyUI is a git
checkout), so it is cheap and exact; the venv is not touched by an update, so there
is nothing to restore there.

Requirements are NOT reinstalled by default: a partial pip upgrade cannot be rolled
back exactly, so update stays within the guaranteeable git rollback. When the pinned
commit changes ComfyUI's requirements.txt, update SAYS SO and points at
``--reinstall-requirements`` (opt-in); passing it reinstalls into the existing venv,
and a failure there still rolls the source back and reports the failure.
"""

from __future__ import annotations

import json
import os
import shutil
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


# --------------------------------------------------------------------------- #
#  Single-flight: only ONE update may mutate the managed checkout at a time.   #
# --------------------------------------------------------------------------- #
# An update is a long MUTATION of one working tree: fetch, checkout to a new commit,
# re-apply the patch set, and on failure roll the checkout back and re-apply the OLD
# patch set. Two of those interleaved is a corrupted checkout, or one call's
# rollback restoring over the other's in-flight update.
#
# It MUST be cross-process, not a threading.Lock: the GUI route does not call this
# function in-process at all, it spawns `python -m localm comfy update` as a CHILD
# PROCESS, so the two contenders are separate interpreters. (managed_comfy._remove_lock
# is a threading.Lock and is correct for ITS job, which is in-process only - it
# would guard nothing here.)
#
# mkdir is ATOMIC; stat-then-create is not - two callers can both observe "free" in
# the gap and both proceed.
_LOCK_OWNER = "owner.json"


def _update_lock_path() -> Path:
    """Where the update lock lives: a SIBLING of the managed checkout, never inside
    it, so the update's own ``git checkout --force`` can never disturb the lock that
    is protecting it."""
    root = mc.managed_comfy_paths().root
    return root.parent / (root.name + ".update.lock")


def _lock_holder_pid(lock: Path) -> Optional[int]:
    """The pid recorded in the lock, or None when it cannot be read."""
    try:
        data = json.loads((lock / _LOCK_OWNER).read_text(encoding="utf-8"))
        pid = data.get("pid") if isinstance(data, dict) else None
        return pid if isinstance(pid, int) else None
    except (OSError, ValueError):
        return None


def _acquire_update_lock() -> tuple:
    """Take the update lock. Returns ``(True, "")`` or ``(False, <honest reason>)``.

    FAILS FAST rather than waiting: an update takes minutes, so a caller blocked on
    the lock would be indistinguishable from a hang.

    Staleness is judged by PID LIVENESS, never by elapsed time. The operation is
    unbounded (a fetch over a slow link, a dependency reinstall), so any fixed
    timeout would eventually reclaim a LIVE holder's lock and produce the exact
    concurrent mutation this exists to prevent. ``pid_alive`` is conservative -
    when it genuinely cannot tell it returns True - so an uncertain answer keeps
    the lock rather than stealing it."""
    from localm.instances import pid_alive
    lock = _update_lock_path()
    for attempt in (1, 2):
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(str(lock))                      # ATOMIC: creates or raises
        except FileExistsError:
            pid = _lock_holder_pid(lock)
            if pid is not None and not pid_alive(pid):
                # The holder is provably gone (a crash, a killed process). Reclaim
                # once, then retry the atomic create - never assume the retry wins,
                # another caller may have taken it in between.
                logger.debug("reclaiming stale comfy-update lock from dead pid %s", pid)
                try:
                    shutil.rmtree(str(lock))
                except OSError as e:
                    return False, (f"An earlier ComfyUI update left a lock at {lock} "
                                   f"that could not be cleared ({e}). Remove that "
                                   "folder and try again.")
                if attempt == 1:
                    continue
                return False, "Another ComfyUI update is already running."
            if pid is None:
                # Cannot tell who holds it: do NOT steal (that is how two updates end
                # up interleaved). Say exactly how to clear it by hand instead.
                return False, (f"A ComfyUI update lock exists at {lock} but its owner "
                               "could not be read. If no update is running, remove "
                               "that folder and try again.")
            return False, (f"Another ComfyUI update is already running (process {pid}). "
                           "Wait for it to finish, then try again.")
        except OSError as e:
            return False, f"Could not take the ComfyUI update lock at {lock}: {e}"
        # Won the create. Record the owner IMMEDIATELY so a contender can judge us.
        try:
            (lock / _LOCK_OWNER).write_text(
                json.dumps({"pid": os.getpid()}), encoding="utf-8")
        except OSError as e:
            logger.debug("could not write comfy-update lock owner: %s", e)
        return True, ""
    return False, "Another ComfyUI update is already running."


def _release_update_lock() -> None:
    """Drop the lock. Never raises: a failure to clean up must not turn a SUCCESSFUL
    update into a reported failure - and a leftover lock is self-healing anyway, since
    the next caller finds our dead pid and reclaims it."""
    try:
        shutil.rmtree(str(_update_lock_path()))
    except OSError as e:
        logger.debug("could not remove comfy-update lock: %s", e)


def _rev_parse_head(root: Path) -> Optional[str]:
    """The managed checkout's HEAD sha, or None when *root* is not a git checkout
    (e.g. installed via the non-git copytree fallback). None is the signal that a
    pin update is not possible - we never fake one."""
    ok, out = _run(["git", "-C", str(root), "rev-parse", "HEAD"], timeout=30)
    return out.strip() if ok else None


def _requirements_changed(root: Path, prev: str, target: str) -> bool:
    """True when requirements.txt differs between *prev* and *target*. ``git diff
    --quiet`` exits 0 for no change, non-zero for a change; a git error is treated
    as 'changed', so update warns rather than skipping a real dep bump."""
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

    # Everything from here on MUTATES the working tree (fetch, checkout, patch, and
    # the rollback's own checkout+patch). Take the single-flight lock first; a second
    # update is refused honestly rather than interleaved. Read-only refusals above
    # (not installed, no git history) never take it.
    acquired, busy = _acquire_update_lock()
    if not acquired:
        return _result(False, "busy", busy)

    def _rollback(reason: str) -> ProvisionResult:
        """Return the managed source to prev_commit and re-apply the patch set, then
        report *reason*. Rolls back the git source exactly; if even the rollback
        checkout fails - OR the checkout succeeds but the localm patch set cannot be
        re-applied on the restored source - say so rather than report a clean
        tree."""
        note = ""
        ok, out = _run(["git", "-C", str(root), "checkout", "--force", "--quiet",
                        prev_commit], on_progress=on_progress, timeout=300)
        if ok:
            # Re-apply the prior patch set on the restored source. The --force checkout
            # discarded our patch edits (they are re-applied here), so a FAILED re-apply
            # leaves the install UNPATCHED (the __func__ fix gone, ACE-Step music broken)
            # while the tree otherwise reads as the prior install. Surface it in the note
            # instead of reporting a clean rollback.
            failed = [o for o in apply_patches(root) if not o.ok]
            if failed:
                detail = "; ".join(f"{o.name}: {o.detail}" for o in failed)
                note = (f" The localm patch set could not be re-applied on the restored "
                        f"install ({detail}); run 'localm comfy update' again or reinstall "
                        "it with 'localm comfy remove' then 'localm comfy setup'.")
                _emit(on_progress, f"Rolled back to the previous ComfyUI "
                      f"({prev_commit[:12]}), but re-applying localm's patches failed.")
            else:
                _emit(on_progress,
                      f"Rolled back to the previous ComfyUI ({prev_commit[:12]}).")
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
        # ASK FOR THE COMMIT BY NAME. A bare `git fetch <url>` uses the default
        # refspec, which brings the remote's current branch tip and nothing else -
        # no tags, and no guarantee the PINNED commit is in the resulting pack.
        # Such a fetch reports success and the checkout then fails mid-update with
        #
        #     fatal: unable to read tree (fe4195f7f427...)
        #
        # i.e. git has the commit object but not the tree it points at. GitHub
        # allows fetching an exact sha (uploadpack.allowReachableSHA1InWant),
        # which is both precise and smaller than pulling every branch.
        ok, out = _run(["git", "-C", str(root), "fetch", "--quiet", repo, target],
                       on_progress=on_progress, timeout=1800)
        if not ok:
            # A server that refuses a by-sha fetch is a normal configuration, not
            # an error - fall back to everything, tags included, so a pin that
            # lives on a tag or a non-default branch still resolves.
            ok, out2 = _run(["git", "-C", str(root), "fetch", "--quiet", "--tags",
                             repo, "+refs/heads/*:refs/remotes/comfy-upstream/*"],
                            on_progress=on_progress, timeout=1800)
            out = out + "\n" + out2
        if not ok:
            return _rollback(f"Could not fetch ComfyUI updates: {_tail(out)}")

        # PROVE the target is completely present BEFORE anything destructive runs.
        # `checkout --force` discards our patch edits on its way to failing, so
        # discovering an incomplete object graph from the checkout itself means
        # the working tree has already been disturbed. Resolving the commit's own
        # tree is the cheap check that the pack really arrived.
        ok, out = _run(["git", "-C", str(root), "rev-parse", "--quiet", "--verify",
                        f"{target}^{{tree}}"], on_progress=None, timeout=60)
        if not ok:
            return _rollback(
                f"ComfyUI {target[:12]} did not download completely - its contents "
                f"are missing from the local clone, so the update was not started. "
                f"Your ComfyUI is untouched. Re-run the update to try again."
            )

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
    finally:
        # Released on EVERY path - success, error, rollback, and the rollback-also-
        # failed path above. A lock left behind by a crash is self-healing (the next
        # caller finds our dead pid and reclaims it), but leaking one on an ordinary
        # error would wedge every later update until someone deleted a folder by hand.
        _release_update_lock()
