# SPDX-License-Identifier: AGPL-3.0-or-later
"""Video plugin: ComfyUI Wan short-video generation + a library for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``video`` capability):
  POST   /api/video                       - generate a clip (background job)
  GET    /api/video/history               - generated clips, newest first
  GET    /api/video/file/{name}           - serve a generated clip
  DELETE /api/video/file/{name}           - delete a clip (+ sidecar)
  POST   /api/video/file/{name}/move      - move a clip to a folder

Generation runs as a background job streamed through the kernel's /api/jobs/*
SSE endpoint. REQUIRES the GUI: ``attach_gui`` must have been called on the app
(it publishes ``request.app.state.jobs`` / ``.self_url``); when it has not, the
generate route returns a clear 503. The backend is selected per-plugin (default
ComfyUI Wan) and reads this plugin's own config (see backend.py). Ships DISABLED
by default.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from localm.inference.http_server import principal_id
from localm.media import gallery
from localm.media import paths as media_paths
from localm.pathsafe import confined_file
from . import backend as _backend

_router = APIRouter()


class VideoRequest(BaseModel):
    prompt: str                       # scene description (motion verbs matter)
    negative_prompt: str | None = None
    seconds: float = 5.0
    fps: int = 24
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None
    input_image: str | None = None    # path on this machine (image-to-video)
    # {node_id: {input_name: value}} - see comfy_client.workflow_model_slots /
    # apply_model_overrides. Picked from the Workflow panel's model dropdowns.
    model_overrides: dict[str, dict[str, str]] | None = None


class MoveFileRequest(BaseModel):
    dest: str                         # destination directory on this machine


def _video_dir() -> Path:
    return media_paths.gallery_dir(media_paths.VIDEO_DIR_NAME)


def _video_path(name: str) -> Path:
    return confined_file(_video_dir(), name, "clip")


@_router.post("/api/video")
async def video(req: VideoRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    if req.seconds <= 0 or req.seconds > 20:
        raise HTTPException(400, "Duration must be between 1 and 20 seconds")
    if req.fps <= 0 or req.fps > 60:
        raise HTTPException(400, "FPS must be between 1 and 60")
    input_image = None
    if req.input_image:
        input_image = media_paths.confined_input_image(req.input_image)

    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Video generation needs the localm GUI server "
                                 "(the background job manager is unavailable).")
    self_url = getattr(request.app.state, "self_url", "")

    video_dir = _video_dir()
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.mp4"

    from localm.config import load_config
    _cfg = load_config()
    s = _backend.settings(_cfg)
    owner = principal_id(request)

    def _generate(job):
        from localm.audit import SessionMode, effective_mode
        if s.get("warning"):
            job.push({"type": "line", "text": s["warning"]})
        ok, msg = _backend.ensure_available(
            s, on_progress=lambda t: job.push({"type": "line", "text": t}))
        job.push({"type": "line", "text": msg})
        if not ok:
            return False
        from localm.vram import (decide_media_swap, media_single_device_shortfall,
                                 unload_chat_for_media)
        from localm.media.comfy_client import resolve_media_placement
        # Per-component GPU placement (opt-in) plus the user-facing notice, in one shared
        # helper (image/music/video share this preamble). placement is applied inside
        # generate_video; notice is what to tell the user.
        placement, notice = resolve_media_placement(_cfg, s["api_url"])
        if notice:
            job.push({"type": "line", "text": notice})
        swap = decide_media_swap(s)
        # REG-532: the gate reads COMBINED free VRAM across a configured GPU split, but
        # each media model component loads WHOLE onto ONE card (localm ORDERS the cards
        # via --default-device, never masks - the model still lands on one), so a job that
        # "fits" in 2x4 GB combined can still OOM on one 4 GB card. When the chosen card
        # cannot hold it, swap anyway: unloading the chat model frees VRAM on every split
        # card, including that one.
        shortfall = media_single_device_shortfall(s)
        if shortfall and not swap:
            swap = True
            job.push({"type": "line", "text":
                      f"GPU {shortfall['index']} has "
                      f"{shortfall['free'] / 1024 ** 3:.1f} GB free, but this job "
                      f"needs {shortfall['needed'] / 1024 ** 3:.1f} GB on a single "
                      "card - unloading the chat model first."})
        gen_swap = False
        if swap:
            if not unload_chat_for_media(job, self_url, "video"):
                gen_swap = True
        else:
            job.push({"type": "line", "text":
                      "Both models fit in VRAM - keeping the chat model loaded "
                      "(no swap)."})
        job.push({"type": "line", "text":
                  f"Submitting Wan workflow to the video backend "
                  f"({req.seconds:.0f}s clip - video is slow, be patient)..."})
        kwargs = {}
        for field in ("negative_prompt", "seconds", "fps", "width",
                      "height", "seed", "steps", "cfg"):
            value = getattr(req, field)
            if value is not None:
                kwargs[field] = value
        if req.model_overrides:
            kwargs["model_overrides"] = req.model_overrides
        is_privacy = effective_mode("server") == SessionMode.PRIVACY
        # privacy mode forces deletion of ComfyUI's own output copy: no traces
        # left anywhere, regardless of the configured delete_outputs preference.
        delete_outputs = bool(s.get("delete_outputs")) or is_privacy
        ok, message = _backend.generate(
            s, req.prompt, out_path,
            self_url=self_url,
            # privacy mode: the prompt never touches disk
            write_sidecar=not is_privacy,
            on_progress=lambda t: job.push({"type": "line", "text": t}),
            input_image=input_image,
            swap=gen_swap,
            delete_outputs=delete_outputs,
            cancel_check=lambda: job.cancel_requested,
            placement=placement,
            **kwargs,
        )
        job.push({"type": "line", "text": message})
        if ok:
            job.result = out_path.name
            gallery.stamp_owner("video", out_path.name, owner)
        # Restore VRAM on EVERY exit path once we have unloaded the chat model -
        # success, failure, OR cancel. The old code reloaded only on success, so
        # a failed or cancelled video gen left the chat model unloaded AND the Wan
        # backend resident in VRAM (a GPU hang). reload_chat_after_media frees the
        # backend's VRAM first, then reloads the chat model, so it is the right
        # restore on the error and cancel paths too. Mirrors image/plug.py.
        if swap:
            from localm.vram import reload_chat_after_media
            reload_chat_after_media(job, self_url, s, _backend, "video")
        return ok

    job = jobs.start_fn("video", _generate, result_path=out_path.name,
                        owner=owner)
    return {"job_id": job.id}


@_router.get("/api/video/file/{name}",
             dependencies=[Depends(gallery.require_owner("video"))])
async def video_file(name: str):
    path = _video_path(name)
    media = {".mp4": "video/mp4", ".webm": "video/webm",
             ".gif": "image/gif"}.get(path.suffix.lower(),
                                      "application/octet-stream")
    return FileResponse(str(path), media_type=media)


@_router.delete("/api/video/file/{name}",
                dependencies=[Depends(gallery.require_owner("video"))])
async def video_delete(name: str):
    path = _video_path(name)
    sidecar = path.with_suffix(path.suffix + ".json")
    path.unlink()
    if sidecar.is_file():
        sidecar.unlink()
    gallery.forget_owner("video", name)
    return {"status": "deleted", "name": name}


@_router.post("/api/video/file/{name}/move",
              dependencies=[Depends(gallery.require_owner("video"))])
async def video_move(name: str, req: MoveFileRequest, request: Request):
    """Move a generated clip (and its sidecar) to a folder on this machine.

    The destination is checked BEFORE the mkdir: require_owner proves artifact
    ownership, not authority over the host filesystem."""
    path = _video_path(name)
    dest_dir = media_paths.confined_move_dest(request, req.dest)
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
    gallery.forget_owner("video", name)     # left the gallery dir
    return {"status": "moved", "path": str(target)}


@_router.get("/api/video/history")
async def video_history(request: Request):
    """Generated clips, newest first, with their sidecar metadata - filtered to
    the caller's own (an admin/owner sees all; unowned/legacy entries stay
    visible to everyone, matching gallery.require_owner)."""
    video_dir = _video_dir()
    items = []
    if video_dir.is_dir():
        files = [p for p in video_dir.iterdir()
                 if p.suffix.lower() in (".mp4", ".webm", ".gif")]
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
    allowed = set(gallery.owned_names(request, "video", [it["name"] for it in items]))
    return {"videos": [it for it in items if it["name"] in allowed]}


@_router.get("/api/video/comfy-models")
async def video_comfy_models():
    """Model-file slots the active video workflow exposes (for the Workflow
    panel's model-picker dropdowns), resolved against the live ComfyUI. Honest
    about unreachability (rule 5) - never a silently-empty picker.

    Resolution is a blocking urlopen of ComfyUI's multi-MB /object_info (10s
    timeout), so it runs OFF the event loop: inline it stalled every concurrent
    request server-wide while ComfyUI was slow (REG-638)."""
    from fastapi.concurrency import run_in_threadpool

    from localm.config import load_config
    s = _backend.settings(load_config())
    slots = await run_in_threadpool(_backend._comfy_model_slots, s)
    if slots is None:
        return {"reachable": False, "api_url": s["api_url"], "slots": [],
                "message": "ComfyUI is not running - launch it to see available models."}
    return {"reachable": True, "api_url": s["api_url"], "slots": slots}


@_router.post("/api/video/comfy-launch")
async def video_comfy_launch():
    """Start (or confirm) ComfyUI is up for the video plugin, without running a
    generation - backs the Workflow panel's "Launch ComfyUI" button."""
    from fastapi.concurrency import run_in_threadpool

    from localm.config import load_config
    s = _backend.settings(load_config())
    ok, message = await run_in_threadpool(_backend.ensure_available, s)
    return {"ok": ok, "message": message, "api_url": s["api_url"]}


def register(host) -> None:
    host.mount_router(_router)
    # Workflow management (list/upload/select/delete) for the Video page, scoped
    # to this plugin's capability.
    from localm.media_workflows import make_workflow_router, migrate_legacy_override
    host.mount_router(make_workflow_router("video"))
    # One-time: rescue any legacy personal override left INSIDE the package
    # (localm/video_gen/wan_workflow_local.json) into home/workflows, which
    # survives a self-update (the localm/ dir is whole-tree-replaced). No-op when absent.
    migrate_legacy_override("video")

    # Register model roles
    from localm.plugins.contract import ModelRoleDescriptor
    host.register_model_role(ModelRoleDescriptor("video-unet", "Diffusion model (UNet)", "diffusion-unet"))
    host.register_model_role(ModelRoleDescriptor("video-clip", "Text encoder (CLIP)", "text-encoder", required=False))

    # "On app start" readiness check (one of the 5 trigger points ComfyUI
    # status gets checked at - see comfy_client.py's readiness-cache
    # docstring): fire-and-forget, does not block plugin registration or
    # attempt to launch ComfyUI.
    from localm.media.comfy_client import warm_comfy_status_async
    warm_comfy_status_async()


def unregister() -> None:
    pass
