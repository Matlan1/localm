# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image plugin: ComfyUI image generation + a gallery for the chat surface.

Routes (mounted by the engine, auto-scoped to the ``image`` capability):
  POST   /api/imagine                       - generate an image (background job)
  POST   /v1/images/generations             - OpenAI-compatible: generate and
                                              return the image(s) in the response
  GET    /api/imagine/history               - generated images, newest first
  GET    /api/imagine/file/{name}           - serve a generated image
  DELETE /api/imagine/file/{name}           - delete an image (+ sidecar)
  POST   /api/imagine/file/{name}/move      - move an image to a folder
  POST   /api/imagine/file/{name}/rename    - rename an image in place

Generation runs as a background job streamed through the kernel's /api/jobs/*
SSE endpoint. It does not require the GUI: the job registry is created by
``attach_engine``, so a headless ``localm serve`` can generate too.
The one thing it needs is this server's OWN address, for the chat/media
VRAM handover; ``resolve_self_url`` derives that from the advertised bind
coordinates when the GUI never published ``.self_url``, and the generate route
503s with that specific reason if it genuinely cannot be determined. The backend
is selected per-plugin (default ComfyUI) and reads this plugin's own config -
see backend.py. Ships DISABLED by default.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from localm.image_gen.comfy import is_safe_lora_name
from localm.inference.http_server import _resolve_disconnect_poll, principal_id
from localm.media import gallery
from localm.media import paths as media_paths
from localm.pathsafe import confined_file, confined_name
from localm.selfclient import resolve_self_url
from . import backend as _backend

_router = APIRouter()


class ImagineRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    guidance: float | None = Field(None, allow_inf_nan=False)
    cfg: float | None = Field(None, allow_inf_nan=False)          # negative-prompt CFG scale; no effect without one
    input_image: str | None = None    # path on this machine (img2img)
    denoise: float | None = Field(None, allow_inf_nan=False)
    # {node_id: {input_name: value}} - see comfy_client.workflow_model_slots /
    # apply_model_overrides. Picked from the Workflow panel's model dropdowns.
    model_overrides: dict[str, dict[str, str]] | None = None
    lora_name: str | None = None              # from the live LoRA picker, or None
    lora_strength_model: float | None = Field(None, allow_inf_nan=False)  # None keeps generate_image()'s default (1.0)
    lora_strength_clip: float | None = Field(None, allow_inf_nan=False)   # None keeps generate_image()'s default (0.5)


class MoveFileRequest(BaseModel):
    dest: str                         # destination directory on this machine


class RenameFileRequest(BaseModel):
    new_name: str                     # new basename (extension kept)


def _images_dir() -> Path:
    return media_paths.gallery_dir(media_paths.IMAGE_DIR_NAME)


def _image_path(name: str) -> Path:
    return confined_file(_images_dir(), name, "image")


def _validate_lora_name(raw: str) -> str:
    """HTTP-layer wrapper over ``comfy.is_safe_lora_name`` - a 400 up front,
    before this route's VRAM-swap/background-job dance ever starts, rather
    than a job that fails partway through with the same message. That shared
    predicate (not a route-local copy) is also enforced again inside
    ``_build_image_workflow`` itself, so the coder agent's ``generate_image``
    tool and any other caller that reaches ``comfy.generate_image`` directly -
    bypassing this route entirely - cannot skip the check either."""
    name = raw.strip()
    if not is_safe_lora_name(name):
        raise HTTPException(400, "Invalid LoRA name")
    return name


def _make_generate(req, *, input_image, lora_name, out_path, owner, self_url,
                   instance_token, width=None, height=None, in_gallery=True):
    """The background-job body for one image generation of ``req`` into
    ``out_path``: backend availability, VRAM handover, generation, and the
    chat-model reload on every exit path. ``width``/``height`` are forwarded to
    the backend only when both are set. ``in_gallery=False`` skips recording
    ``out_path`` as a gallery artifact owned by ``owner``."""
    def _generate(job):
        from localm.audit import SessionMode, effective_mode
        from localm.config import load_config
        # Resolved here, in the job's own worker thread, not the route above.
        _cfg = load_config()
        s = _backend.settings(_cfg)
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
        # generate_image; notice is what to tell the user.
        placement, notice = resolve_media_placement(_cfg, s["api_url"])
        if notice:
            job.push({"type": "line", "text": notice})
        swap = decide_media_swap(s)
        # The gate above reads COMBINED free VRAM across a configured GPU split,
        # but each media model component loads WHOLE onto ONE card (localm ORDERS the
        # cards via --default-device, never masks - the model still lands on one). When
        # the card actually chosen cannot hold it, swap anyway: unloading the chat model
        # frees VRAM on every split card, including that one.
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
            # Unload here (authenticated + logged) so the chat model is actually
            # gone before FLUX loads; fall back to the backend's own unload only
            # if this one could not reach the server.
            if not unload_chat_for_media(job, self_url, "image", instance_token):
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
        size_kwargs = ({"width": width, "height": height}
                       if width is not None and height is not None else {})
        ok, message = _backend.generate(
            s, req.prompt, out_path,
            self_url=self_url,
            instance_token=instance_token,
            # privacy mode: no sidecar (the image's own metadata is stripped
            # in every mode)
            write_sidecar=not is_privacy,
            guidance=req.guidance,
            cfg=req.cfg,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            input_image=input_image,
            denoise=req.denoise,
            model_overrides=req.model_overrides,
            lora_name=lora_name,
            lora_strength_model=req.lora_strength_model,
            lora_strength_clip=req.lora_strength_clip,
            swap=gen_swap,
            delete_outputs=delete_outputs,
            cancel_check=lambda: job.cancel_requested,
            placement=placement,
            on_progress=lambda t: job.push({"type": "line", "text": t}),
            **size_kwargs,
        )
        job.push({"type": "line", "text": message})
        if ok:
            job.result = out_path.name
            if in_gallery:
                gallery.stamp_owner("image", out_path.name, owner)
        # Mark the outcome BEFORE the VRAM handover below, which is best-effort
        # cleanup that can itself raise (e.g. a non-comfy backend's free_vram())
        # and must never turn a successful generation into a reported failure.
        job.mark_outcome("done" if ok else "failed")
        # Restore VRAM on EVERY exit path once the chat model has been unloaded -
        # success, failure, OR cancel. reload_chat_after_media frees the
        # backend's VRAM first, then reloads the chat model.
        if swap:
            from localm.vram import reload_chat_after_media
            reload_chat_after_media(job, self_url, s, _backend, "image", instance_token)
        return ok
    return _generate


@_router.post("/api/imagine")
async def imagine(req: ImagineRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    from localm.debuglog import logger
    logger.info("imagine: %d-char prompt", len(req.prompt))
    input_image = None
    if req.input_image:
        input_image = media_paths.confined_input_image(req.input_image)
    lora_name = _validate_lora_name(req.lora_name) if req.lora_name else None

    # Any app built through attach_engine has a background-job registry, so
    # reaching this branch means the router was mounted on an app that never ran
    # it. That is a construction error, answered with a clean 503 rather than an
    # unguarded AttributeError and an opaque 500.
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Image generation needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    # A headless server may not know its OWN address, which the VRAM handover
    # below needs (self_request raises on an empty base_url). Gate on that and
    # say so.
    self_url = resolve_self_url(request.app)
    if not self_url:
        raise HTTPException(503, "Image generation needs this server's own "
                                 "address to free VRAM first, and it could not "
                                 "be determined.")
    # Threaded through to the VRAM handover below so it authenticates on a
    # keyless server too - see selfclient.self_request's docstring.
    instance_token = getattr(request.app.state, "instance_token", None)

    images_dir = _images_dir()
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.png"
    owner = principal_id(request)

    job = jobs.start_fn(
        "imagine",
        _make_generate(req, input_image=input_image, lora_name=lora_name,
                       out_path=out_path, owner=owner, self_url=self_url,
                       instance_token=instance_token),
        result_path=out_path.name, owner=owner)
    return {"job_id": job.id}


MAX_IMAGES_PER_REQUEST = 4
MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 2048
MAX_IMAGE_PIXELS = 2048 * 2048
MAX_PROMPT_CHARS = 32000
_SIZE_RE = re.compile(r"^(\d{1,5})x(\d{1,5})$")


class ImageGenerationRequest(BaseModel):
    """Body of ``POST /v1/images/generations``. ``model``, ``quality``,
    ``style`` and ``user`` are accepted and ignored: the image model is the one
    the active workflow loads. Unknown fields are ignored."""
    prompt: str
    model: str | None = None
    n: int = Field(1, ge=1, le=MAX_IMAGES_PER_REQUEST)
    size: str | None = None
    response_format: str | None = None
    output_format: str | None = None


def parse_image_size(size: str | None) -> tuple[int, int] | None:
    """``(width, height)`` for a ``WIDTHxHEIGHT`` size, or None for "auto" or no
    size (the workflow's own size). Raises ``HTTPException`` (400) for a size
    outside ``MIN_IMAGE_SIDE``..``MAX_IMAGE_SIDE`` per side, off a multiple of 8,
    or over ``MAX_IMAGE_PIXELS`` in total."""
    if size is None or size.strip().lower() in ("", "auto"):
        return None
    m = _SIZE_RE.match(size.strip().lower())
    if not m:
        raise HTTPException(
            400, f"size must be 'auto' or WIDTHxHEIGHT such as '1024x1024' (got {size!r}).")
    w, h = int(m.group(1)), int(m.group(2))
    if not (MIN_IMAGE_SIDE <= w <= MAX_IMAGE_SIDE and MIN_IMAGE_SIDE <= h <= MAX_IMAGE_SIDE):
        raise HTTPException(
            400, f"size sides must be between {MIN_IMAGE_SIDE} and {MAX_IMAGE_SIDE} pixels (got {size!r}).")
    if w % 8 or h % 8:
        raise HTTPException(400, f"size sides must be multiples of 8 (got {size!r}).")
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(
            400, f"size is too large: at most {MAX_IMAGE_PIXELS} pixels in total (got {size!r}).")
    return w, h


def _is_private_session() -> bool:
    from localm.audit import SessionMode, effective_mode
    return effective_mode("server") == SessionMode.PRIVACY


CANCEL_SETTLE_SECONDS = 30.0
JOB_POLL_SECONDS = 1.0


async def _await_job(job, request: Request) -> tuple[str, str]:
    """Wait for a background job to end and return ``(status, last_line)``.

    When the client goes away while it runs, cancels the job, waits up to
    ``CANCEL_SETTLE_SECONDS`` for it to finish writing and stop, then raises
    ``asyncio.CancelledError``."""
    queue = job.subscribe()
    poll = _resolve_disconnect_poll(request)
    last_line = ""
    deadline = None
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=JOB_POLL_SECONDS)
            except TimeoutError:
                if deadline is None:
                    if poll is not None and await poll():
                        job.cancel()
                        deadline = time.monotonic() + CANCEL_SETTLE_SECONDS
                elif time.monotonic() >= deadline:
                    raise asyncio.CancelledError() from None
                continue
            kind = event.get("type")
            if kind == "line":
                last_line = str(event.get("text", ""))
            elif kind == "end":
                if deadline is not None:
                    raise asyncio.CancelledError()
                return str(event.get("status", "failed")), last_line
    finally:
        job.unsubscribe(queue)


def _remove_tree_when_idle(path: Path, jobs: list) -> None:
    """Remove a private generation directory once every job in ``jobs`` has
    stopped writing: at once when they all have, otherwise from a background
    thread that waits for them (at most ``CANCEL_SETTLE_SECONDS`` * 20) and logs a
    warning if the directory still cannot be removed."""
    def _idle() -> bool:
        return all(getattr(j, "finished_at", None) is not None for j in jobs)

    if _idle():
        _remove_tree(path)
        return

    def _later() -> None:
        deadline = time.monotonic() + CANCEL_SETTLE_SECONDS * 20
        while not _idle() and time.monotonic() < deadline:
            time.sleep(max(JOB_POLL_SECONDS / 2, 0.01))
        try:
            _remove_tree(path)
        except HTTPException:
            pass

    threading.Thread(target=_later, name="localm-image-cleanup", daemon=True).start()


def _remove_tree(path: Path) -> None:
    """Delete a private generation directory, failing loudly if any file stays."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("could not remove the private image directory %s: %s", path, e)
        raise HTTPException(
            500, "The generated image could not be removed from disk after it was "
                 "read, so it was not returned (privacy mode keeps no image on disk)."
        ) from e


@_router.post("/v1/images/generations")
async def create_image(req: ImageGenerationRequest, request: Request):
    """OpenAI-compatible image generation: runs the same job as ``/api/imagine``
    and returns the finished image(s) in the response.

    Outside privacy mode each image is kept in the gallery (with its sidecar).
    ``response_format`` defaults to ``url`` when an API key is configured (the
    URL is the gallery file route, fetched with the same key) and to
    ``b64_json`` when none is, where ``url`` is a 400. In privacy mode nothing
    is kept: the image is generated into a private directory, returned as
    ``b64_json`` (the default there; ``url`` is a 400) and the directory is
    deleted before the response is sent."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Empty prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"prompt is too long (max {MAX_PROMPT_CHARS} characters).")
    size = parse_image_size(req.size)
    if req.output_format is not None and req.output_format.strip().lower() != "png":
        raise HTTPException(400, "output_format must be 'png': this server writes PNG images.")
    private = _is_private_session()
    from localm.auth import any_key_configured
    url_servable = not private and any_key_configured()
    fmt = (req.response_format or ("url" if url_servable else "b64_json")).strip().lower()
    if fmt not in ("url", "b64_json"):
        raise HTTPException(400, "response_format must be 'url' or 'b64_json' "
                                 f"(got {req.response_format!r}).")
    if fmt == "url" and private:
        raise HTTPException(
            400, "response_format 'url' needs the image kept on disk, and session mode "
                 "'privacy' keeps none: request 'b64_json' instead.")
    if fmt == "url" and not url_servable:
        raise HTTPException(
            400, "response_format 'url' needs an API key: the gallery file route cannot "
                 "be fetched without one while no key is configured. Request 'b64_json', "
                 "or create a key with 'localm key create'.")

    from localm.debuglog import logger
    logger.info("v1 images: %d-char prompt, n=%d", len(prompt), req.n)
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "Image generation needs this server's "
                                 "background job registry, which is "
                                 "unavailable.")
    self_url = resolve_self_url(request.app)
    if not self_url:
        raise HTTPException(503, "Image generation needs this server's own "
                                 "address to free VRAM first, and it could not "
                                 "be determined.")
    instance_token = getattr(request.app.state, "instance_token", None)
    owner = principal_id(request)
    gen_req = ImagineRequest(prompt=prompt)
    width, height = size if size else (None, None)

    private_dir = None
    started_jobs = []
    items = []
    try:
        if private:
            from localm.config import temp_dir
            private_dir = Path(tempfile.mkdtemp(prefix="localm-img-", dir=temp_dir()))
            out_dir = private_dir
        else:
            out_dir = _images_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
        base_url = str(request.base_url).rstrip("/")
        for _ in range(req.n):
            out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.png"
            job = jobs.start_fn(
                "imagine",
                _make_generate(gen_req, input_image=None, lora_name=None,
                               out_path=out_path, owner=owner, self_url=self_url,
                               instance_token=instance_token, width=width,
                               height=height, in_gallery=not private),
                result_path=out_path.name, owner=owner)
            started_jobs.append(job)
            status, last_line = await _await_job(job, request)
            if status == "cancelled":
                raise HTTPException(409, "The image generation was cancelled.")
            if status != "done" or not out_path.is_file():
                raise HTTPException(
                    502, f"Image generation failed: {last_line or 'no reason was reported'}")
            if fmt == "b64_json":
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, out_path.read_bytes)
                items.append({"b64_json": base64.b64encode(data).decode("ascii")})
            else:
                items.append({"url": f"{base_url}/api/imagine/file/{out_path.name}"})
    finally:
        if private_dir is not None:
            _remove_tree_when_idle(private_dir, started_jobs)
    return {"created": int(time.time()), "data": items}


@_router.get("/api/imagine/file/{name}",
             dependencies=[Depends(gallery.require_owner("image"))])
async def imagine_file(name: str):
    return FileResponse(str(_image_path(name)), media_type="image/png")


@_router.delete("/api/imagine/file/{name}",
                dependencies=[Depends(gallery.require_owner("image"))])
async def imagine_delete(name: str):
    path = _image_path(name)
    sidecar = path.with_suffix(path.suffix + ".json")
    path.unlink()
    if sidecar.is_file():
        sidecar.unlink()
    gallery.forget_owner("image", name)
    return {"status": "deleted", "name": name}


@_router.post("/api/imagine/file/{name}/move",
              dependencies=[Depends(gallery.require_owner("image"))])
async def imagine_move(name: str, req: MoveFileRequest, request: Request):
    """Move a generated image (and its metadata sidecar) to a folder on this
    machine - e.g. into a project or pictures directory.

    The destination is checked BEFORE the mkdir: require_owner proves artifact
    ownership, not authority over the host filesystem."""
    path = _image_path(name)
    dest_dir = media_paths.confined_move_dest(request, req.dest)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Cannot create destination: {e}") from e
    if not dest_dir.is_dir():
        raise HTTPException(400, f"Not a directory: {req.dest}")
    target = dest_dir / path.name
    if target.exists():
        raise HTTPException(409, f"Already exists: {target}")
    shutil.move(str(path), str(target))
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.is_file():
        shutil.move(str(sidecar), str(dest_dir / sidecar.name))
    gallery.forget_owner("image", name)     # left the gallery dir
    return {"status": "moved", "path": str(target)}


@_router.post("/api/imagine/file/{name}/rename",
              dependencies=[Depends(gallery.require_owner("image"))])
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
    gallery.rename_owner("image", name, target.name)
    return {"status": "renamed", "name": target.name}


@_router.get("/api/imagine/history")
async def imagine_history(request: Request):
    """Generated images, newest first, with their sidecar metadata - filtered to
    the caller's own (an admin/owner sees all; unowned/legacy entries stay
    visible to everyone, matching gallery.require_owner)."""
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
    allowed = set(gallery.owned_names(request, "image", [it["name"] for it in items]))
    return {"images": [it for it in items if it["name"] in allowed]}


@_router.get("/api/imagine/comfy-models")
async def imagine_comfy_models(request: Request):
    """Model-file slots the active image workflow exposes (for the Workflow
    panel's model-picker dropdowns), plus the LoRA files ComfyUI currently has
    installed (for the generation form's LoRA picker), resolved against the
    live ComfyUI. Reports unreachability rather than returning a
    silently-empty picker.

    LoRAs are enumerated separately from ``slots``: a LoraLoader node is not
    normally present in the active workflow JSON (the plugin injects one at
    generation time only when a LoRA is requested - see comfy.py's
    ``_build_image_workflow``), so the ``workflow_model_slots`` node walk that
    builds ``slots`` would never surface it.

    Each slot also carries the localm ``model_type`` its loader node holds, the
    declared role it fills (``role_id``/``role_label`` from
    ``host.register_model_role``), and ``installed`` - decided by the SAME rule
    preflight uses to call a model missing, so the picker cannot call a slot fine
    that generation then refuses. ``roles`` reports every declared role including
    ones this workflow has no slot for, and ``registry_models`` lists this box's
    own registered component models by type. Both are answered from the registry,
    so they are returned even when ComfyUI is unreachable: "we could not ask
    ComfyUI" is a different answer from "you have nothing".

    The slot/LoRA resolution is a blocking urlopen of ComfyUI's /object_info
    (commonly several MB, 10s timeout), so it runs OFF the event loop, the same
    way the /comfy-launch route below offloads its own slow call.

    Bounded at a bit over comfy_object_info's own 10s urlopen timeout, so this
    only fires for a call genuinely stuck beyond that (a wedged native call, not
    an ordinary slow-ComfyUI load). That one budget also covers the registry
    read the role join needs, a small local JSON, and that read sits inside the
    SAME offload so it cannot land back on the event loop. settings() gets its
    own offload below: it can reach sanitize_comfy_url's blocking DNS lookup, a
    second way onto the event loop distinct from the /object_info fetch."""
    from localm.config import load_config
    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded,
    )
    from localm.plugins.media_roles import plugin_model_roles
    # Read in the request, not in the worker: this walks the plugin manager's
    # in-memory descriptors and does no I/O.
    roles = plugin_model_roles(request.app, "image")
    try:
        s = await run_in_threadpool_bounded(_backend.settings, load_config(), timeout=20.0)
        resolved = await run_in_threadpool_bounded(
            _backend._comfy_model_roles, s, roles, timeout=20.0)
        loras = await run_in_threadpool_bounded(_backend._comfy_lora_options, s, timeout=20.0)
    except ThreadCallTimeout as e:
        raise HTTPException(504, f"Reading ComfyUI's model list timed out: {e}") from e
    out = {"api_url": s["api_url"], "loras": loras or [], **resolved}
    if not resolved["reachable"]:
        out["message"] = "ComfyUI is not running - launch it to see available models."
    return out


@_router.post("/api/imagine/comfy-launch")
async def imagine_comfy_launch():
    """Start (or confirm) ComfyUI is up for the image plugin, without running a
    generation - backs the Workflow panel's "Launch ComfyUI" button. Runs the
    same ensure_available() path a real generation uses, off the event loop
    since a cold ComfyUI start can take minutes. settings() itself is also
    offloaded (a separate, short budget): it can reach sanitize_comfy_url's
    blocking DNS lookup.

    Bounded at the SAME comfy_launch_timeout ensure_comfy itself will honour
    (comfy_launch_wait_seconds), plus a buffer, never an independent value: a
    smaller bound would abort a launch still progressing under a larger
    user-configured timeout."""
    from localm.config import load_config
    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded,
    )
    from localm.media.comfy_client import comfy_launch_wait_seconds
    cfg = load_config()
    budget = comfy_launch_wait_seconds(cfg) + 30.0
    try:
        s = await run_in_threadpool_bounded(_backend.settings, cfg, timeout=20.0)
        ok, message = await run_in_threadpool_bounded(
            _backend.ensure_available, s, timeout=budget)
    except ThreadCallTimeout as e:
        raise HTTPException(504, f"Launching ComfyUI timed out: {e}") from e
    return {"ok": ok, "message": message, "api_url": s["api_url"]}


def register(host) -> None:
    host.mount_router(_router)
    # Workflow management (list/upload/select/delete) for the Image page, scoped
    # to this plugin's capability.
    from localm.media_workflows import make_workflow_router, migrate_legacy_override
    host.mount_router(make_workflow_router("image"))
    # One-time: rescue any legacy personal override left INSIDE the package
    # (localm/image_gen/flux_workflow.json) into home/workflows, which survives a
    # self-update (the localm/ dir is whole-tree-replaced). No-op when absent.
    migrate_legacy_override("image")

    # Register model roles
    from localm.plugins.contract import ModelRoleDescriptor
    host.register_model_role(ModelRoleDescriptor("image-unet", "Diffusion model (UNet)", "diffusion-unet"))
    host.register_model_role(ModelRoleDescriptor("image-clip1", "Text encoder 1 (CLIP-L)", "text-encoder", required=False))
    host.register_model_role(ModelRoleDescriptor("image-clip2", "Text encoder 2 (T5/CLIP-G)", "text-encoder", required=False))
    host.register_model_role(ModelRoleDescriptor("image-vae", "VAE", "vae", required=False))
    host.register_model_role(ModelRoleDescriptor("image-lora", "LoRA", "lora", required=False))

    # "On app start" readiness check: fire-and-forget, does not block plugin
    # registration and does not attempt to launch ComfyUI.
    from localm.media.comfy_client import warm_comfy_status_async
    warm_comfy_status_async()


def unregister() -> None:
    pass
