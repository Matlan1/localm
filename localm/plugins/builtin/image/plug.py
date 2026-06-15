"""Image plugin: ComfyUI image generation + a gallery for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``image`` capability):
  POST   /api/imagine                       - generate an image (background job)
  GET    /api/imagine/history               - generated images, newest first
  GET    /api/imagine/file/{name}           - serve a generated image
  DELETE /api/imagine/file/{name}           - delete an image (+ sidecar)
  POST   /api/imagine/file/{name}/move      - move an image to a folder
  POST   /api/imagine/file/{name}/rename    - rename an image in place

Generation runs as a background job streamed through the kernel's /api/jobs/*
SSE endpoint. REQUIRES the GUI: ``attach_gui`` must have been called on the app
(it publishes ``request.app.state.jobs`` / ``.self_url``); when it has not, the
generate route returns a clear 503 rather than failing obscurely. The backend is
selected per-plugin (default ComfyUI) and reads this plugin's own config -
see backend.py. Ships DISABLED by default.
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

from localm.pathsafe import confined_file, confined_name
from . import backend as _backend

_router = APIRouter()


class ImagineRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    guidance: float | None = None
    input_image: str | None = None    # path on this machine (img2img)
    denoise: float | None = None


class MoveFileRequest(BaseModel):
    dest: str                         # destination directory on this machine


class RenameFileRequest(BaseModel):
    new_name: str                     # new basename (extension kept)


def _images_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "gui_images"


def _image_path(name: str) -> Path:
    return confined_file(_images_dir(), name, "image")


def _reload_llm(job, self_url: str, s: dict) -> None:
    """Hand VRAM back: ask the backend to drop its models, then reload the chat
    model so the next reply is instant. Skipped when reload-after-generate is off."""
    if not s["reload_after"]:
        job.push({"type": "line", "text":
                  "Keeping the image backend loaded (reload is off) - the chat "
                  "model reloads on the next message."})
        return
    if not _backend.free_vram(s):
        job.push({"type": "line", "text":
                  "The image backend kept its models in VRAM - the chat model "
                  "will reload on the next message instead."})
        return
    job.push({"type": "line", "text": "Reloading the chat model..."})
    try:
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        _rq.post(f"{self_url}/models/load", headers=headers, timeout=300)
        job.push({"type": "line", "text": "Chat model ready."})
    except Exception as e:
        job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})


@_router.post("/api/imagine")
async def imagine(req: ImagineRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    input_image = None
    if req.input_image:
        input_image = Path(req.input_image).expanduser()
        if not input_image.is_file():
            raise HTTPException(400, f"Input image not found: {req.input_image}")

    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Image generation needs the localm GUI server "
                                 "(the background job manager is unavailable).")
    self_url = getattr(request.app.state, "self_url", "")

    images_dir = _images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.png"
    from localm.config import load_config
    s = _backend.settings(load_config())

    def _generate(job):
        from localm.audit import SessionMode, effective_mode
        if s.get("warning"):
            job.push({"type": "line", "text": s["warning"]})
        ok, msg = _backend.ensure_available(
            s, on_progress=lambda t: job.push({"type": "line", "text": t}))
        job.push({"type": "line", "text": msg})
        if not ok:
            return False
        job.push({"type": "line", "text": "Submitting workflow to the image backend..."})
        ok, message = _backend.generate(
            s, req.prompt, out_path,
            self_url=self_url,
            # privacy mode: the prompt never touches disk
            write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
            guidance=req.guidance,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            input_image=input_image,
            denoise=req.denoise,
        )
        job.push({"type": "line", "text": message})
        if ok:
            job.result = out_path.name
            _reload_llm(job, self_url, s)
        return ok

    job = jobs.start_fn("imagine", _generate, result_path=out_path.name)
    return {"job_id": job.id}


@_router.get("/api/imagine/file/{name}")
async def imagine_file(name: str):
    return FileResponse(str(_image_path(name)), media_type="image/png")


@_router.delete("/api/imagine/file/{name}")
async def imagine_delete(name: str):
    path = _image_path(name)
    sidecar = path.with_suffix(path.suffix + ".json")
    path.unlink()
    if sidecar.is_file():
        sidecar.unlink()
    return {"status": "deleted", "name": name}


@_router.post("/api/imagine/file/{name}/move")
async def imagine_move(name: str, req: MoveFileRequest):
    """Move a generated image (and its metadata sidecar) to a folder on this
    machine - e.g. into a project or pictures directory."""
    path = _image_path(name)
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
    return {"status": "moved", "path": str(target)}


@_router.post("/api/imagine/file/{name}/rename")
async def imagine_rename(name: str, req: RenameFileRequest):
    """Rename a generated image (and its metadata sidecar) in place."""
    path = _image_path(name)
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(400, "Empty name")
    if not new_name.lower().endswith(path.suffix.lower()):
        new_name += path.suffix          # keep the extension
    target = confined_name(_images_dir(), new_name)
    if target.exists():
        raise HTTPException(409, f"Already exists: {new_name}")
    path.rename(target)
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.is_file():
        sidecar.rename(target.with_suffix(target.suffix + ".json"))
    return {"status": "renamed", "name": target.name}


@_router.get("/api/imagine/history")
async def imagine_history():
    """Generated images, newest first, with their sidecar metadata."""
    images_dir = _images_dir()
    items = []
    if images_dir.is_dir():
        for p in sorted(images_dir.glob("*.png"),
                        key=lambda f: f.stat().st_mtime, reverse=True)[:100]:
            meta = {}
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    pass
            items.append({"name": p.name, "meta": meta,
                          "path": str(p),
                          "mtime": p.stat().st_mtime})
    return {"images": items}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
