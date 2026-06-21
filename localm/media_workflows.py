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

Backend-agnostic: this only reads/writes files + the config marker. Path-safety
is enforced with ``localm.pathsafe`` so an uploaded/selected name can never escape
the per-media directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MEDIA_TYPES = ("image", "music", "video")


def workflows_dir(media: str) -> Path:
    """The per-media workflows directory (created on first write)."""
    from localm.config import home_dir
    return home_dir() / "workflows" / media


def _confined(media: str, name: str) -> Path:
    """A path-safe handle inside the media dir (raises on a traversal name)."""
    from localm.pathsafe import confined_name
    return confined_name(workflows_dir(media), name)


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


def list_workflows(media: str) -> list:
    """Every uploaded workflow for *media*, newest first, with active/default
    flags so the page can show which one is in use."""
    d = workflows_dir(media)
    active = selected_name(media)
    items = []
    if d.is_dir():
        files = [p for p in d.glob("*.json") if p.is_file()]
        for p in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):
            items.append({"name": p.name, "is_active": p.name == active,
                          "size_bytes": p.stat().st_size,
                          "mtime": p.stat().st_mtime})
    return items


def save_workflow(media: str, name: str, content: bytes) -> str:
    """Validate + store an uploaded workflow. Returns the stored filename.

    Raises ValueError when the body is not JSON or not a ComfyUI API-format
    workflow, so a bad upload is rejected with a clear reason rather than silently
    accepted and then failing every generation."""
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
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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


def make_workflow_router(media: str):
    """Build the list / upload / select / delete routes for one media plugin.

    Mounted by each media plugin's register() via its own host, so the routes are
    auto-scoped to that plugin's capability. fastapi is imported lazily here so
    the generators (which import this module) stay framework-free. Upload takes
    the already-parsed workflow JSON as a body field, so there is no multipart /
    python-multipart dependency."""
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.get(f"/api/{media}/workflows")
    async def _list_workflows():
        return {"workflows": list_workflows(media), "selected": selected_name(media)}

    @router.post(f"/api/{media}/workflows")
    async def _upload_workflow(body: dict):
        wf = body.get("workflow")
        if not isinstance(wf, dict):
            raise HTTPException(400, "missing 'workflow' object (the ComfyUI "
                                     "API-format JSON)")
        name = str(body.get("name") or "workflow.json")
        try:
            saved = save_workflow(media, name, json.dumps(wf).encode("utf-8"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if body.get("activate"):
            select_workflow(media, saved)
        return {"name": saved, "workflows": list_workflows(media),
                "selected": selected_name(media)}

    @router.post(f"/api/{media}/workflows/select")
    async def _select_workflow(body: dict):
        try:
            sel = select_workflow(media, body.get("name"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"selected": sel, "workflows": list_workflows(media)}

    @router.delete(f"/api/{media}/workflows/{{name}}")
    async def _delete_workflow(name: str):
        try:
            delete_workflow(media, name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"workflows": list_workflows(media), "selected": selected_name(media)}

    return router
