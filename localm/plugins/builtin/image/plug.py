# SPDX-License-Identifier: AGPL-3.0-or-later
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

from localm.inference.http_server import principal_id
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


def _allowed_input_roots() -> list[Path]:
    """Roots an ``input_image`` (img2img source) may live under.

    img2img uploads the file into ComfyUI's input/ dir, so an unconfined path is
    an arbitrary-local-file read primitive (any readable file gets copied into
    ComfyUI). Confine it to localm's own data dir plus the project checkout when
    running from source - the only locations a user legitimately drops a source
    image into for this local tool."""
    from localm.config import home_dir
    roots: list[Path] = []
    try:
        roots.append(home_dir().resolve())
    except OSError:
        pass
    # When running from a source checkout, the repo root is a legitimate place
    # to keep a reference image (e.g. an examples/ asset). Skip it for installed
    # copies (no pyproject.toml), which keeps the surface to the data dir only.
    repo_root = Path(__file__).resolve().parents[4]
    if (repo_root / "pyproject.toml").is_file():
        roots.append(repo_root)
    return roots


def _confined_input_image(raw: str) -> Path:
    """Resolve and confine an ``input_image`` to an allowed root.

    Raises HTTPException(400) when the path escapes every allowed root or does
    not point at an existing file. Symlinks are resolved first, so a link inside
    an allowed root that targets outside it is still rejected."""
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid input image path")
    roots = _allowed_input_roots()
    if not any(resolved == r or r in resolved.parents for r in roots):
        raise HTTPException(
            400, "Input image must be inside the localm data directory")
    if not resolved.is_file():
        raise HTTPException(400, f"Input image not found: {raw}")
    return resolved


def _images_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "gui_images"


def _image_path(name: str) -> Path:
    return confined_file(_images_dir(), name, "image")


def _unload_chat(job, self_url: str) -> bool:
    """Unload the chat model BEFORE the image model loads, so it gets the VRAM.

    Uses the same bearer-token + TLS handling as the reload path: the
    ``/v1/models/unload`` endpoint needs the models-write scope, so an
    unauthenticated call is rejected and the chat model stays resident - the
    image model then loads on top of it and hangs the GPU driver. Logs the
    outcome (and the VRAM freed) so a failure is visible instead of silent.
    Returns True when the server confirmed the chat model is unloaded."""
    job.push({"type": "line", "text": "Freeing VRAM: unloading the chat model..."})
    try:
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        from localm import tls as _tls
        resp = _rq.post(f"{self_url}/models/unload", headers=headers, timeout=300,
                        verify=_tls.requests_verify(self_url))
        if not resp.ok:
            job.push({"type": "line", "text":
                      f"Could not unload the chat model (HTTP {resp.status_code}) - "
                      "the image backend may run low on VRAM."})
            return False
        data = {}
        try:
            data = resp.json()
        except Exception:
            pass
        if data.get("status") == "already_unloaded":
            job.push({"type": "line", "text":
                      "No chat model was loaded - VRAM already free."})
            return True
        before, after = data.get("vram_before_bytes"), data.get("vram_after_bytes")
        if data.get("vram_freed") and before is not None and after is not None:
            gb = max(0.0, (after - before) / 1024 ** 3)
            job.push({"type": "line", "text":
                      f"Chat model unloaded - freed {gb:.1f} GB of VRAM."})
        elif data.get("vram_freed") is False:
            job.push({"type": "line", "text":
                      "Chat model unloaded, but VRAM has not dropped yet - continuing."})
        else:
            job.push({"type": "line", "text": "Chat model unloaded."})
        return True
    except Exception as e:
        job.push({"type": "line", "text":
                  f"Could not unload the chat model ({e}) - "
                  "the image backend may run low on VRAM."})
        return False


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
        from localm import tls as _tls
        _rq.post(f"{self_url}/models/load", headers=headers, timeout=300,
                 verify=_tls.requests_verify(self_url))
        job.push({"type": "line", "text": "Chat model ready."})
    except Exception as e:
        job.push({"type": "line", "text": f"Reload deferred to the next message ({e})."})


@_router.post("/api/imagine")
async def imagine(req: ImagineRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    input_image = None
    if req.input_image:
        input_image = _confined_input_image(req.input_image)

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
        from localm.vram import decide_media_swap
        swap = decide_media_swap(s)
        gen_swap = False
        if swap:
            # Unload here (authenticated + logged) so the chat model is actually
            # gone before FLUX loads; fall back to the backend's own unload only
            # if this one could not reach the server.
            if not _unload_chat(job, self_url):
                gen_swap = True
        else:
            job.push({"type": "line", "text":
                      "Both models fit in VRAM - keeping the chat model loaded "
                      "(no swap)."})
        job.push({"type": "line", "text": "Submitting workflow to the image backend..."})
        is_privacy = effective_mode("server") == SessionMode.PRIVACY
        # privacy mode forces deletion of ComfyUI's own output copy: no traces
        # left anywhere, regardless of the configured delete_outputs preference.
        delete_outputs = bool(s.get("delete_outputs")) or is_privacy
        ok, message = _backend.generate(
            s, req.prompt, out_path,
            self_url=self_url,
            # privacy mode: the prompt never touches disk
            write_sidecar=not is_privacy,
            guidance=req.guidance,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            input_image=input_image,
            denoise=req.denoise,
            swap=gen_swap,
            delete_outputs=delete_outputs,
            cancel_check=lambda: job.cancel_requested,
        )
        job.push({"type": "line", "text": message})
        if ok:
            job.result = out_path.name
        # Restore VRAM on EVERY exit path once we have unloaded the chat model -
        # success, failure, OR cancel. The old code reloaded only on success, so
        # hitting Stop mid-generation left the chat model unloaded AND ComfyUI's
        # model resident in VRAM (the reported "fails unloading the image model,
        # loading chat - not sure it even tried"). _reload_llm frees the backend's
        # VRAM first, then reloads the chat model, so it is the right restore on
        # both the cancel and the error paths too.
        if swap:
            _reload_llm(job, self_url, s)
        return ok

    job = jobs.start_fn("imagine", _generate, result_path=out_path.name,
                        owner=principal_id(request))
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
    # Workflow management (list/upload/select/delete) for the Image page, scoped
    # to this plugin's capability.
    from localm.media_workflows import make_workflow_router
    host.mount_router(make_workflow_router("image"))

    # Register model roles
    from localm.plugins.contract import ModelRoleDescriptor
    host.register_model_role(ModelRoleDescriptor("image-unet", "Diffusion model (UNet)", "diffusion-unet"))
    host.register_model_role(ModelRoleDescriptor("image-clip1", "Text encoder 1 (CLIP-L)", "text-encoder", required=False))
    host.register_model_role(ModelRoleDescriptor("image-clip2", "Text encoder 2 (T5/CLIP-G)", "text-encoder", required=False))
    host.register_model_role(ModelRoleDescriptor("image-vae", "VAE", "vae", required=False))
    host.register_model_role(ModelRoleDescriptor("image-lora", "LoRA", "lora", required=False))


def unregister() -> None:
    pass
