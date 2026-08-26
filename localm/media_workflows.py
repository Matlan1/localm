# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin workflow management for the media plugins (image / music / video).

A user can upload ComfyUI workflows (in ComfyUI: Save -> API format) and pick
which one a plugin uses, straight from the plugin's page. Files live under
``home_dir()/workflows/<media>/``; the selected file is recorded per-plugin in
``config["plugins"][<media>]["workflow"]``.

The generators resolve the active workflow in this order, so an existing setup
keeps working untouched:
  1. the SELECTED file in ``home_dir()/workflows/<media>/`` (this module), then
  2. the legacy personal override committed next to the generator
     (flux_workflow.json / ace_workflow_local.json / wan_workflow_local.json), then
  3. the committed example template.

Tier 2 is superseded: a self-update whole-tree-replaces the ``localm/`` package
dir, so an override left there is destroyed on update. ``migrate_legacy_override``
(called once from each media plugin's ``register()``) moves any such file up into
tier 1 - which lives under the updater's NEVER_TOUCH ``home/`` and survives -
selecting it so the user's active workflow is unchanged.

Backend-agnostic: this only reads/writes files + the config marker. Path-safety
is enforced with ``localm.pathsafe`` so an uploaded/selected name can never escape
the per-media directory.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

MEDIA_TYPES = ("image", "music", "video")

# Per-media mutual exclusion for the four workflow routes (list/upload/select/
# delete). The route bodies run on a threadpool worker, so two requests for the
# same media can interleave, and every check-then-act sequence here needs
# serialising. One lock per media: image/music/video are independent directory
# trees and independent config blocks. Acquired inside the threadpool closure,
# so waiting on it never blocks the event loop.
_media_locks: dict[str, threading.Lock] = {}
_media_locks_guard = threading.Lock()


def _lock_for(media: str) -> threading.Lock:
    with _media_locks_guard:
        lock = _media_locks.get(media)
        if lock is None:
            lock = _media_locks[media] = threading.Lock()
        return lock


# Budget for run_in_threadpool_bounded() in the four routes below. Module-level
# so a test can monkeypatch it down.
#
# _WORKFLOW_OWN_WORK_TIMEOUT_S is the ceiling for one holder's own work.
# _WORKFLOW_RMW_TIMEOUT_S is what the routes pass, and is 2x that: all four
# routes share one _lock_for(media) lock acquired inside the bounded closure, so
# a request's clock also covers the time it waits behind another holder.
_WORKFLOW_OWN_WORK_TIMEOUT_S = 30.0
_WORKFLOW_RMW_TIMEOUT_S = 2 * _WORKFLOW_OWN_WORK_TIMEOUT_S

# Distinct from None, which is a legitimate VALUE for `active` (no workflow
# selected). Using None as both "caller did not pass this" and "resolved to no
# selection" makes list_workflows re-resolve selected_name() a second time
# whenever nothing is selected, defeating the single-config-load point of
# passing `active` through at all.
_UNSET = object()


def workflows_dir(media: str) -> Path:
    """The per-media workflows directory (created on first write)."""
    from localm.config import home_dir
    return home_dir() / "workflows" / media


def _confined(media: str, name: str) -> Path:
    """A path-safe handle inside the media dir. Raises ValueError on a
    traversal/invalid name.

    confined_name() itself raises fastapi.HTTPException (pathsafe.py documents it
    as being for HTTP call sites only), but this module is framework-free (see
    the module docstring) and has a non-HTTP caller (the CLI), so the exception
    is normalized here. Every function in this module documents "Raises
    ValueError" for its own bad-name case, and that covers confinement failures
    too. make_workflow_router's routes catch ValueError and convert it to
    HTTPException(400, str(e)) themselves, so the HTTP response (400, "Invalid
    file name") is unchanged."""
    from fastapi import HTTPException
    from localm.pathsafe import confined_name
    try:
        return confined_name(workflows_dir(media), name)
    except HTTPException as e:
        raise ValueError(str(e.detail)) from e


def selected_name(media: str) -> Optional[str]:
    """The filename the user selected for *media*, or None."""
    from localm.config import load_config
    blk = (load_config().get("plugins") or {}).get(media) or {}
    name = blk.get("workflow")
    return name if isinstance(name, str) and name else None


def active_workflow_path(media: str) -> Optional[Path]:
    """The selected workflow FILE for *media*, or None when none is selected (or
    the selected file has gone). The generators fall back to their own
    legacy/example template when this is None, so selection is purely additive."""
    name = selected_name(media)
    if not name:
        return None
    try:
        p = _confined(media, name)
    except Exception:
        return None
    return p if p.is_file() else None


def is_workflow_json(data) -> bool:
    """A ComfyUI API-format workflow is a non-empty dict of node-id -> object,
    where at least one node carries a ``class_type`` (what ComfyUI's Save (API
    format) emits). This rejects a random JSON file pasted in by mistake."""
    if not isinstance(data, dict) or not data:
        return False
    return any(isinstance(v, dict) and "class_type" in v for v in data.values())


def list_workflows(media: str, *, active=_UNSET) -> list:
    """Every uploaded workflow for *media*, newest first, with active/default
    flags so the page can show which one is in use.

    *active*: pass an already-resolved ``selected_name(media)`` (None is a
    valid value here - "nothing selected") to skip this function's own config
    load - see ``_list_and_selected`` below, the shape every caller that needs
    both actually wants. Omit it (the default, the _UNSET sentinel) to resolve
    it internally, unchanged for a caller that only wants the list."""
    d = workflows_dir(media)
    if active is _UNSET:
        active = selected_name(media)
    items = []
    if d.is_dir():
        files = [p for p in d.glob("*.json") if p.is_file()]
        for p in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):
            # Defense in depth on top of the caller-side per-media lock
            # (_lock_for): a file can still vanish between the is_file()
            # check above and here from something the lock does not cover -
            # a user manually deleting it on disk, an external tool, another
            # localm process sharing this LOCALM_HOME. Skip it rather than
            # crash the whole listing over one file that is simply gone by
            # the time we get to it; it will not appear in the NEXT listing
            # either, so nothing is silently hidden, just not resurrected.
            try:
                items.append({"name": p.name, "is_active": p.name == active,
                              "size_bytes": p.stat().st_size,
                              "mtime": p.stat().st_mtime})
            except FileNotFoundError:
                continue
    return items


def _list_and_selected(media: str) -> tuple:
    """(list_workflows(media), selected_name(media)) from ONE config load.
    Every route in make_workflow_router needs both together; calling them
    independently loads config.json TWICE per request - real cost on Windows,
    where a config read can hit the documented ~1s antivirus/indexer retry
    (config.py's _replace_atomic doc)."""
    active = selected_name(media)
    return list_workflows(media, active=active), active


def save_workflow(media: str, name: str, content: bytes) -> str:
    """Validate + store an uploaded workflow. Returns the stored filename.

    Raises ValueError when the body is not JSON or not a ComfyUI API-format
    workflow, so a bad upload is rejected with a clear reason rather than silently
    accepted and then failing every generation."""
    from localm.storekit import atomic_write
    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"not valid JSON: {e}")
    if not is_workflow_json(data):
        raise ValueError(
            "not a ComfyUI API-format workflow (expected a dict of nodes, each "
            "with a class_type). In ComfyUI use Save (API format), not the "
            "regular Save.")
    # Validate the RAW name first (rejects "", "..", separators) BEFORE adding the
    # .json suffix, so ".." cannot become the legit filename "...json".
    _confined(media, name)
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    p = _confined(media, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Re-serialize (normalized) so we never persist arbitrary bytes from upload.
    # atomic_write (temp file + os.replace) so a concurrent reader - the model-slots
    # probe and every generator's workflow_path().read_text(), none of which take a
    # lock - never observes a half-written file.
    atomic_write(p, json.dumps(data, indent=2, ensure_ascii=False))
    return p.name


def select_workflow(media: str, name: Optional[str]) -> Optional[str]:
    """Set the active workflow for *media* (None/"" clears it -> fall back to the
    legacy/example template). Returns the selected name or None. Raises ValueError
    when *name* does not exist."""
    from localm.config import update_config
    chosen: Optional[str] = None
    if name:
        p = _confined(media, name)
        if not p.is_file():
            raise ValueError(f"no such workflow: {name}")
        chosen = p.name

    def _mut(cfg: dict) -> None:
        plugins = cfg.get("plugins")
        if not isinstance(plugins, dict):
            plugins = cfg["plugins"] = {}
        blk = plugins.get(media)
        if not isinstance(blk, dict):
            blk = plugins[media] = {}
        if chosen:
            blk["workflow"] = chosen
        else:
            blk.pop("workflow", None)

    update_config(_mut)
    return chosen


def delete_workflow(media: str, name: str) -> None:
    """Delete an uploaded workflow. Refuses to delete the ACTIVE one (select
    another first) so a generation is never left pointing at a missing file.
    Raises ValueError when it does not exist or is active."""
    p = _confined(media, name)
    if not p.is_file():
        raise ValueError(f"no such workflow: {name}")
    if selected_name(media) == p.name:
        raise ValueError("cannot delete the active workflow; select another first")
    p.unlink()


# ---------------------------------------------------------------------------
#  Legacy override migration
# ---------------------------------------------------------------------------
#
# Before the per-plugin workflow picker existed, a user customised a media plugin
# by committing a personal workflow next to its generator, INSIDE the localm/
# package dir. A self-update whole-tree-replaces that dir (localm/_apply_update.py
# swaps the "localm" entry: rmtree + copytree), so an override left at one of
# these legacy paths is DESTROYED on update. The primary location - the SELECTED
# file under home_dir()/workflows/<media>/ - lives under the updater's NEVER_TOUCH
# "home/" and survives, so we move any legacy override there once, at startup.
_LEGACY_OVERRIDES = {
    "image": Path(__file__).parent / "image_gen" / "flux_workflow.json",
    "music": Path(__file__).parent / "music_gen" / "ace_workflow_local.json",
    "video": Path(__file__).parent / "video_gen" / "wan_workflow_local.json",
}


def _dedup_name(dest_dir: Path, name: str) -> Path:
    """A path inside *dest_dir* whose name does not collide with an existing file
    (``x.json`` -> ``x-1.json`` -> ``x-2.json`` ...), so migrating a legacy
    override can never clobber a differently-authored workflow the user already
    saved under the same name."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = dest_dir / name
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}-{i}{suffix}"
        i += 1
    return candidate


def _existing_copy(dest_dir: Path, legacy: Path) -> Optional[Path]:
    """An existing workflow in *dest_dir* whose bytes already match *legacy* (this
    override was migrated on an earlier run, possibly under a dedup name), or None.
    Scanning by CONTENT - not just the base name - means a persistently
    unremovable in-package original cannot spawn a fresh dedup copy every startup."""
    if not dest_dir.is_dir():
        return None
    for p in sorted(dest_dir.glob("*.json")):
        try:
            if p.is_file() and filecmp.cmp(legacy, p, shallow=False):
                return p
        except OSError:
            continue
    return None


def _finalize_migration(media: str, legacy: Path, dest: Path) -> tuple:
    """Keep the migrated *dest* the active workflow, then remove the in-package
    *legacy* original. Returns ``(ok, note)``.

    Auto-select is gated on ``active_workflow_path`` (the EFFECTIVE resolution),
    not merely on a selection marker: when no valid workflow is active the legacy
    file WAS the active one (resolution is selected -> legacy -> example, and a
    selection whose file has gone falls through to the legacy), so the moved copy
    must become the selection or generation would silently drop to the example.

    Fail-safe: the original is removed ONLY once the override is safely active
    from home, so a failed select/remove never silently deactivates or loses the
    user's workflow."""
    if active_workflow_path(media) is None:
        try:
            select_workflow(media, dest.name)
        except Exception as e:
            return False, (f"{media}: saved legacy {legacy.name} to workflows/"
                           f"{media}/{dest.name} but could not select it ({e}); "
                           f"left the in-package copy in place")
    try:
        legacy.unlink()
    except OSError as e:
        return False, (f"{media}: migrated legacy {legacy.name} to workflows/"
                       f"{media}/{dest.name} but could not remove the in-package "
                       f"copy ({e}); a later update will")
    return True, f"{media}: migrated legacy {legacy.name} to workflows/{media}/{dest.name}"


def _migrate_one(media: str, legacy: Path) -> Optional[tuple]:
    """Move one media's legacy in-package override into home/workflows/<media>/.
    Returns None when there is nothing to migrate, else ``(ok, note)``. Pure with
    respect to the configured home (the caller injects the source path), so it is
    unit-testable without touching the real repo checkout."""
    if not legacy.is_file():
        return None
    dest_dir = workflows_dir(media)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Already migrated (identical bytes already under home, base name or a dedup
    # name)? Retire the in-package duplicate instead of copying it again.
    existing = _existing_copy(dest_dir, legacy)
    if existing is not None:
        return _finalize_migration(media, legacy, existing)
    # A DIFFERENT workflow already owns that name - keep both under a fresh name.
    dest = dest_dir / legacy.name
    if dest.exists():
        dest = _dedup_name(dest_dir, legacy.name)
    shutil.copy2(legacy, dest)          # home copy is safe before we touch the original
    return _finalize_migration(media, legacy, dest)


def migrate_legacy_override(media: str) -> Optional[str]:
    """Startup entry point (called from each media plugin's ``register()``): move
    *media*'s legacy in-package workflow override to the update-surviving
    home/workflows location, once. A no-op when there is nothing to migrate.

    Skipped under the automated test suite - in-process via ``"pytest" in
    sys.modules``, and inside a localm SUBPROCESS a test spawns (where pytest is
    absent) via the ``LOCALM_SKIP_LEGACY_WORKFLOW_MIGRATION`` flag conftest sets
    and children inherit. The in-package sources are the real repository checkout,
    NOT the test's tmp home, so migrating there would move a developer's personal
    workflow out of their working tree. (The migration logic itself is exercised
    directly via ``_migrate_one`` on temp paths.) Logs and returns the outcome
    note (None when nothing happened)."""
    if "pytest" in sys.modules or os.environ.get("LOCALM_SKIP_LEGACY_WORKFLOW_MIGRATION"):
        return None
    legacy = _LEGACY_OVERRIDES.get(media)
    if legacy is None:
        return None
    try:
        result = _migrate_one(media, legacy)
    except Exception as e:
        result = (False, f"{media}: legacy workflow migration failed ({e}); "
                         f"left it in place")
    if result is None:
        return None
    ok, note = result
    from localm.debuglog import logger
    (logger.info if ok else logger.warning)("workflow migration: %s", note)
    return note


def make_workflow_router(media: str):
    """Build the list / upload / select / delete routes for one media plugin.

    Mounted by each media plugin's register() via its own host, so the routes are
    auto-scoped to that plugin's capability. fastapi is imported lazily here so
    the generators (which import this module) stay framework-free. Upload takes
    the already-parsed workflow JSON as a body field, so there is no multipart /
    python-multipart dependency."""
    from fastapi import APIRouter, HTTPException

    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded,
    )

    router = APIRouter()

    # Off the event loop, all four routes: list_workflows/selected_name do
    # synchronous filesystem I/O (a directory glob + a stat per file, a
    # config.json read) that inline blocks the WHOLE server for the duration -
    # worst on GET, which the GUI polls on its own timer, so it is a repeating
    # stall for as long as the page is open. Same shape as the already-offloaded
    # comfy-model-slots route.
    #
    # Every route ALSO acquires _lock_for(media) around its whole body, inside
    # the threadpool closure (so it costs the worker thread, never the event
    # loop) - see the module-level comment on _lock_for: offloading alone removes
    # the free serialization these check-then-act sequences depend on, and this
    # restores it.
    #
    # Bounded at _WORKFLOW_RMW_TIMEOUT_S - see that constant's own comment for
    # why a client-side timeout here is still safe against corruption.

    @router.get(f"/api/{media}/workflows")
    async def _list_workflows():
        def _do() -> tuple:
            with _lock_for(media):
                return _list_and_selected(media)

        try:
            workflows, selected = await run_in_threadpool_bounded(
                _do, timeout=_WORKFLOW_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Listing {media} workflows timed out: {e}")
        return {"workflows": workflows, "selected": selected}

    @router.post(f"/api/{media}/workflows")
    async def _upload_workflow(body: dict):
        wf = body.get("workflow")
        if not isinstance(wf, dict):
            raise HTTPException(400, "missing 'workflow' object (the ComfyUI "
                                     "API-format JSON)")
        name = str(body.get("name") or "workflow.json")

        def _do() -> dict:
            with _lock_for(media):
                saved = save_workflow(media, name, json.dumps(wf).encode("utf-8"))
                if body.get("activate"):
                    select_workflow(media, saved)
                workflows, selected = _list_and_selected(media)
                return {"name": saved, "workflows": workflows, "selected": selected}

        try:
            return await run_in_threadpool_bounded(_do, timeout=_WORKFLOW_RMW_TIMEOUT_S)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Uploading the {media} workflow timed out: {e}")

    @router.post(f"/api/{media}/workflows/select")
    async def _select_workflow(body: dict):
        def _do() -> dict:
            with _lock_for(media):
                sel = select_workflow(media, body.get("name"))
                return {"selected": sel, "workflows": list_workflows(media, active=sel)}

        try:
            return await run_in_threadpool_bounded(_do, timeout=_WORKFLOW_RMW_TIMEOUT_S)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Selecting the {media} workflow timed out: {e}")

    @router.delete(f"/api/{media}/workflows/{{name}}")
    async def _delete_workflow(name: str):
        def _do() -> dict:
            with _lock_for(media):
                delete_workflow(media, name)
                workflows, selected = _list_and_selected(media)
                return {"workflows": workflows, "selected": selected}

        try:
            return await run_in_threadpool_bounded(_do, timeout=_WORKFLOW_RMW_TIMEOUT_S)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Deleting the {media} workflow timed out: {e}")

    return router
