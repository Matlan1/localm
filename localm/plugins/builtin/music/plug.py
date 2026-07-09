# SPDX-License-Identifier: AGPL-3.0-or-later
"""Music plugin: ComfyUI ACE-Step music generation + a library for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``music`` capability):
  POST   /api/music                       - generate a track (background job)
  GET    /api/music/history               - generated tracks, newest first
  GET    /api/music/file/{name}           - serve a generated track
  DELETE /api/music/file/{name}           - delete a track (+ sidecar)
  POST   /api/music/file/{name}/move      - move a track to a folder

Generation runs as a background job streamed through the kernel's /api/jobs/*
SSE endpoint. REQUIRES the GUI: ``attach_gui`` must have been called on the app
(it publishes ``request.app.state.jobs`` / ``.self_url``); when it has not, the
generate route returns a clear 503. The backend is selected per-plugin (default
ComfyUI ACE-Step) and reads this plugin's own config (see backend.py). Ships
DISABLED by default.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from localm.inference.http_server import principal_id
from localm.media import gallery
from localm.pathsafe import confined_file
from . import backend as _backend

_router = APIRouter()


class MusicRequest(BaseModel):
    tags: str                         # style prompt: genre, mood, instruments
    lyrics: str | None = None         # None/empty = instrumental
    duration_seconds: float = 120.0   # arbitrary track length
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    lyrics_strength: float | None = None


class MoveFileRequest(BaseModel):
    dest: str                         # destination directory on this machine


def _music_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "gui_music"


def _music_path(name: str) -> Path:
    return confined_file(_music_dir(), name, "track")


@_router.post("/api/music")
async def music(req: MusicRequest, request: Request):
    if not req.tags.strip():
        raise HTTPException(400, "Empty style tags")
    if req.duration_seconds <= 0 or req.duration_seconds > 3600:
        raise HTTPException(400, "Duration must be between 1 and 3600 seconds")

    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Music generation needs the localm GUI server "
                                 "(the background job manager is unavailable).")
    self_url = getattr(request.app.state, "self_url", "")

    music_dir = _music_dir()
    music_dir.mkdir(parents=True, exist_ok=True)
    out_path = music_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.flac"

    from localm.config import load_config
    s = _backend.settings(load_config())
    owner = principal_id(request)

    def _generate(job):
        from localm.audit import SessionMode, effective_mode
        # Privacy mode forces no on-disk traces: it suppresses the sidecar AND
        # forces ComfyUI's own output copy to be deleted, regardless of the
        # opt-in delete_outputs setting.
        is_privacy = effective_mode("server") == SessionMode.PRIVACY
        if s.get("warning"):
            job.push({"type": "line", "text": s["warning"]})
        ok, msg = _backend.ensure_available(
            s, on_progress=lambda t: job.push({"type": "line", "text": t}))
        job.push({"type": "line", "text": msg})
        if not ok:
            return False
        from localm.vram import decide_media_swap, unload_chat_for_media
        swap = decide_media_swap(s)
        gen_swap = False
        if swap:
            if not unload_chat_for_media(job, self_url, "music"):
                gen_swap = True
        else:
            job.push({"type": "line", "text":
                      "Both models fit in VRAM - keeping the chat model loaded "
                      "(no swap)."})
        job.push({"type": "line", "text":
                  f"Submitting ACE-Step workflow to the music backend "
                  f"({req.duration_seconds:.0f}s track)..."})
        kwargs = {}
        if req.seed is not None:
            kwargs["seed"] = req.seed
        if req.steps is not None:
            kwargs["steps"] = req.steps
        if req.cfg is not None:
            kwargs["cfg"] = req.cfg
        if req.lyrics_strength is not None:
            kwargs["lyrics_strength"] = req.lyrics_strength
        ok, message = _backend.generate(
            s, req.tags, out_path,
            self_url=self_url,
            write_sidecar=not is_privacy,
            # Delete ComfyUI's own copy when the user opted in OR privacy mode
            # forces no traces.
            delete_outputs=bool(s.get("delete_outputs")) or is_privacy,
            on_progress=lambda t: job.push({"type": "line", "text": t}),
            lyrics=req.lyrics,
            duration_seconds=req.duration_seconds,
            swap=gen_swap,
            cancel_check=lambda: job.cancel_requested,
            **kwargs,
        )
        job.push({"type": "line", "text": message})
        if ok:
            job.result = out_path.name
            gallery.stamp_owner("music", out_path.name, owner)
        # Restore VRAM on EVERY exit path once we have unloaded the chat model -
        # success, failure, OR cancel. Mirrors image/plug.py: the old code reloaded
        # only on success, so a failed or cancelled generation left the chat model
        # unloaded AND the music backend resident in VRAM (GPU hang).
        # reload_chat_after_media frees the backend's VRAM first, then reloads
        # the chat model.
        if swap:
            from localm.vram import reload_chat_after_media
            reload_chat_after_media(job, self_url, s, _backend, "music")
        return ok

    job = jobs.start_fn("music", _generate, result_path=out_path.name,
                        owner=owner)
    return {"job_id": job.id}


@_router.get("/api/music/file/{name}")
async def music_file(name: str, request: Request):
    gallery.owned_or_404(request, "music", name)
    path = _music_path(name)
    media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/flac"
    return FileResponse(str(path), media_type=media)


@_router.delete("/api/music/file/{name}")
async def music_delete(name: str, request: Request):
    gallery.owned_or_404(request, "music", name)
    path = _music_path(name)
    sidecar = path.with_suffix(path.suffix + ".json")
    path.unlink()
    if sidecar.is_file():
        sidecar.unlink()
    gallery.forget_owner("music", name)
    return {"status": "deleted", "name": name}


@_router.post("/api/music/file/{name}/move")
async def music_move(name: str, req: MoveFileRequest, request: Request):
    gallery.owned_or_404(request, "music", name)
    path = _music_path(name)
    dest_dir = Path(req.dest).expanduser()
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Cannot create destination: {e}")
    if not dest_dir.is_dir():
        raise HTTPException(400, f"Not a directory: {req.dest}")
    target = dest_dir / path.name
    if target.exists():
        raise HTTPException(409, f"Already exists: {target}")
    shutil.move(str(path), str(target))
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.is_file():
        shutil.move(str(sidecar), str(dest_dir / sidecar.name))
    gallery.forget_owner("music", name)     # left the gallery dir
    return {"status": "moved", "path": str(target)}


@_router.get("/api/music/history")
async def music_history(request: Request):
    """Generated tracks, newest first, with their sidecar metadata - filtered to
    the caller's own (an admin/owner sees all; unowned/legacy entries stay
    visible to everyone, matching gallery.owned_or_404)."""
    music_dir = _music_dir()
    items = []
    if music_dir.is_dir():
        files = [p for p in music_dir.iterdir()
                 if p.suffix.lower() in (".flac", ".mp3", ".wav", ".ogg")]
        for p in sorted(files, key=lambda f: f.stat().st_mtime,
                        reverse=True)[:100]:
            meta = {}
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    pass
            items.append({"name": p.name, "meta": meta,
                          "path": str(p),
                          "size_bytes": p.stat().st_size,
                          "mtime": p.stat().st_mtime})
    allowed = set(gallery.owned_names(request, "music", [it["name"] for it in items]))
    return {"tracks": [it for it in items if it["name"] in allowed]}


def register(host) -> None:
    host.mount_router(_router)
    # Workflow management (list/upload/select/delete) for the Music page, scoped
    # to this plugin's capability.
    from localm.media_workflows import make_workflow_router, migrate_legacy_override
    host.mount_router(make_workflow_router("music"))
    # One-time: rescue any legacy personal override left INSIDE the package
    # (localm/music_gen/ace_workflow_local.json) into home/workflows, which
    # survives a self-update (the localm/ dir is whole-tree-replaced). No-op when absent.
    migrate_legacy_override("music")

    # Register model roles
    from localm.plugins.contract import ModelRoleDescriptor
    host.register_model_role(ModelRoleDescriptor("music-unet", "Diffusion model (UNet)", "diffusion-unet"))
    host.register_model_role(ModelRoleDescriptor("music-clip", "Text encoder (CLIP)", "text-encoder", required=False))
    host.register_model_role(ModelRoleDescriptor("music-vae", "VAE", "vae", required=False))

    # "On app start" readiness check (one of the 5 trigger points ComfyUI
    # status gets checked at - see comfy_client.py's readiness-cache
    # docstring): fire-and-forget, does not block plugin registration or
    # attempt to launch ComfyUI.
    from localm.media.comfy_client import warm_comfy_status_async
    warm_comfy_status_async()


def unregister() -> None:
    pass
