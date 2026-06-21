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


def _unload_chat(job, self_url: str) -> bool:
    """Unload the chat model BEFORE the music model loads, so it gets the VRAM.

    Uses the same bearer-token + TLS handling as the reload path: the
    ``/v1/models/unload`` endpoint needs the models-write scope, so an
    unauthenticated call is rejected and the chat model stays resident - the
    music model then loads on top of it and hangs the GPU driver. Logs the
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
                      "the music backend may run low on VRAM."})
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
                  "the music backend may run low on VRAM."})
        return False


def _reload_llm(job, self_url: str, s: dict) -> None:
    """Hand VRAM back: ask the backend to drop its models, then reload the chat
    model. Skipped when reload-after-generate is off."""
    if not s["reload_after"]:
        job.push({"type": "line", "text":
                  "Keeping the music backend loaded (reload is off) - the chat "
                  "model reloads on the next message."})
        return
    if not _backend.free_vram(s):
        job.push({"type": "line", "text":
                  "The music backend kept its models in VRAM - the chat model "
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
            if not _unload_chat(job, self_url):
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
            write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
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
            if swap:
                _reload_llm(job, self_url, s)
        return ok

    job = jobs.start_fn("music", _generate, result_path=out_path.name)
    return {"job_id": job.id}


@_router.get("/api/music/file/{name}")
async def music_file(name: str):
    path = _music_path(name)
    media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/flac"
    return FileResponse(str(path), media_type=media)


@_router.delete("/api/music/file/{name}")
async def music_delete(name: str):
    path = _music_path(name)
    sidecar = path.with_suffix(path.suffix + ".json")
    path.unlink()
    if sidecar.is_file():
        sidecar.unlink()
    return {"status": "deleted", "name": name}


@_router.post("/api/music/file/{name}/move")
async def music_move(name: str, req: MoveFileRequest):
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
    return {"status": "moved", "path": str(target)}


@_router.get("/api/music/history")
async def music_history():
    """Generated tracks, newest first, with their sidecar metadata."""
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
    return {"tracks": items}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
