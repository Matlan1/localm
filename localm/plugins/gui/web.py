"""
GUI web layer — API routes and static file serving, attached to the
existing localm FastAPI inference app.

Routes (all under /api, bearer-protected when LOCALM_API_KEY is set):
  GET    /api/models                       registry + active model
  POST   /api/models/load                  switch the active engine
  POST   /api/coder/sessions               start a coder session
  GET    /api/coder/sessions/{id}/events   SSE event stream
  POST   /api/coder/sessions/{id}/message  send a task / chat message
  POST   /api/coder/sessions/{id}/confirm  answer a pending confirmation
  POST   /api/coder/sessions/{id}/stop     interrupt the current task
  DELETE /api/coder/sessions/{id}          terminate the session

The static frontend is mounted at / and must be attached AFTER all API
routes so it doesn't shadow them.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from localm.inference.http_server import _require_auth
from .sessions import CoderSession, SessionManager

STATIC_DIR = Path(__file__).parent / "static"

# SSE keepalive interval — must beat proxy/browser idle timeouts
_KEEPALIVE_S = 15


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class LoadModelRequest(BaseModel):
    model: str


class CreateSessionRequest(BaseModel):
    cwd: str
    auto_approve: bool = False
    max_turns: int = 40
    mode: str | None = None           # None = config coder_mode/mode, else privacy
    model: str | None = None          # switch active engine when given
    scope: str | None = None          # glob restricting file-access tools
    temperature: float | None = None
    max_tokens: int | None = None


class PullRequest(BaseModel):
    spec: str
    name: str | None = None


class RemoveModelRequest(BaseModel):
    model: str


class AliasRequest(BaseModel):
    model: str
    alias: str


class ImagineRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    guidance: float | None = None
    input_image: str | None = None    # path on this machine (img2img)
    denoise: float | None = None


class MoveImageRequest(BaseModel):
    dest: str                         # destination directory on this machine


class MusicRequest(BaseModel):
    tags: str                         # style prompt: genre, mood, instruments…
    lyrics: str | None = None         # None/empty = instrumental
    duration_seconds: float = 120.0   # arbitrary track length
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    lyrics_strength: float | None = None


class VideoRequest(BaseModel):
    prompt: str                       # scene description (motion verbs matter)
    negative_prompt: str | None = None
    seconds: float = 5.0              # snapped to Wan's 4k+1 frame rule
    fps: int = 24
    width: int | None = None          # template default 832 (multiple of 16)
    height: int | None = None         # template default 480
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    input_image: str | None = None    # path on this machine (image-to-video)


class MessageRequest(BaseModel):
    text: str


class ConfirmRequest(BaseModel):
    confirm_id: str
    approved: bool


class ConversationUpsert(BaseModel):
    title: str = "Untitled"
    updated_at: float = 0
    pinned: bool = False
    folder: str | None = None
    branches: list = []           # parked message-branch tails (fork points)
    messages: list = []


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 8000


class RagCreateRequest(BaseModel):
    name: str


class RagAddRequest(BaseModel):
    paths: list[str]
    embed: bool = True            # try embeddings; degrades to lexical-only


class RagQueryRequest(BaseModel):
    query: str
    k: int = 4


class RagRemoveDocRequest(BaseModel):
    path: str


class RagExtractRequest(BaseModel):
    filename: str
    content_b64: str              # in-memory extraction — no disk writes
    max_chars: int = 24_000


class PromptUpsert(BaseModel):
    system: str = ""
    params: dict = {}             # sampling defaults (temperature, top_p, …)


class MemoryUpdate(BaseModel):
    text: str


class MemoryAppend(BaseModel):
    text: str


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str | None = None


# ------------------------------------------------------------------ #
#  Attach                                                             #
# ------------------------------------------------------------------ #

def attach_gui(
    app: FastAPI,
    *,
    self_url: str,
    switch_model,
    active_model,
) -> SessionManager:
    """
    Add GUI routes and static serving to *app*.

    Parameters
    ----------
    self_url:
        Base URL of this server's own /v1 API — coder agents talk to the
        model through it (e.g. ``http://127.0.0.1:8642/v1``).
    switch_model:
        ``Callable[[str], Awaitable[None]]`` — swaps the active engine.
    active_model:
        ``Callable[[], str]`` — name of the currently served model.
    """
    manager = SessionManager()

    # -------------------------- models ---------------------------- #

    @app.get("/api/models", dependencies=[Depends(_require_auth)])
    async def gui_models():
        from localm.config import load_registry
        registry = load_registry()
        current = active_model()
        models = []
        for name, entry in sorted(registry.items()):
            path = Path(entry.get("path", ""))
            size = None
            try:
                if path.is_file():
                    size = path.stat().st_size
            except OSError:
                pass
            models.append({
                "name": name,
                "source": entry.get("source", ""),
                "size_bytes": size,
                "active": name == current,
            })
        return {"models": models, "active": current}

    @app.post("/api/models/load", dependencies=[Depends(_require_auth)])
    async def gui_load_model(req: LoadModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            return {"status": "already_active", "model": req.model}
        try:
            await switch_model(req.model)
        except Exception as e:
            raise HTTPException(500, f"Failed to load {req.model}: {e}")
        return {"status": "loaded", "model": req.model}

    # -------------------------- coder ----------------------------- #

    @app.get("/api/coder/sessions", dependencies=[Depends(_require_auth)])
    async def list_sessions():
        return {"sessions": manager.list(), "active_model": active_model()}

    @app.post("/api/coder/sessions", dependencies=[Depends(_require_auth)])
    async def create_session(req: CreateSessionRequest):
        cwd = Path(req.cwd).expanduser()
        if not cwd.is_dir():
            raise HTTPException(400, f"Not a directory: {req.cwd}")

        # Per-session model: the local GPU serves one engine at a time, so a
        # different model means switching the active engine for everyone.
        if req.model and req.model != active_model():
            from localm.config import load_registry
            if req.model not in load_registry():
                raise HTTPException(404, f"Model not registered: {req.model}")
            try:
                await switch_model(req.model)
            except Exception as e:
                raise HTTPException(500, f"Failed to load {req.model}: {e}")

        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend(
            self_url,
            model=active_model(),
            api_key=os.environ.get("LOCALM_API_KEY") or "localm",
        )

        gen_kwargs = {}
        if req.temperature is not None:
            gen_kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            gen_kwargs["max_tokens"] = req.max_tokens

        from localm.audit import effective_mode
        session_mode = req.mode or effective_mode("coder").value

        loop = asyncio.get_running_loop()
        # Agent construction scans the project (map build) — keep it off the loop
        session = await loop.run_in_executor(None, lambda: CoderSession(
            cwd.resolve(),
            backend,
            auto_approve=req.auto_approve,
            max_turns=req.max_turns,
            mode=session_mode,
            scope=req.scope,
            **gen_kwargs,
        ))
        manager.create(session)
        return session.info()

    def _get_session(session_id: str) -> CoderSession:
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(404, f"No such session: {session_id}")
        return session

    @app.get("/api/coder/sessions/{session_id}/events",
             dependencies=[Depends(_require_auth)])
    async def session_events(session_id: str, replay: bool = False):
        """SSE event stream. ``?replay=true`` first re-sends the session's
        event history (so a reloaded page rebuilds its feed), then goes live."""
        session = _get_session(session_id)
        loop = asyncio.get_running_loop()

        async def _stream():
            if replay:
                # Snapshot history, then drop anything still queued — those
                # events are part of the snapshot and must not arrive twice.
                snapshot = list(session.history)
                while True:
                    try:
                        session.events.get_nowait()
                    except queue.Empty:
                        break
                for event in snapshot:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'replay_done'})}\n\n"
            while True:
                try:
                    event = await loop.run_in_executor(
                        None, session.events.get, True, _KEEPALIVE_S)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "closed":
                    return

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/coder/sessions/{session_id}/undo",
              dependencies=[Depends(_require_auth)])
    async def session_undo(session_id: str):
        session = _get_session(session_id)
        summary = session.undo()
        if summary is None:
            raise HTTPException(409, "Nothing to undo (or agent is busy)")
        return {"status": "undone", "summary": summary}

    @app.post("/api/coder/sessions/{session_id}/compact",
              dependencies=[Depends(_require_auth)])
    async def session_compact(session_id: str):
        session = _get_session(session_id)
        loop = asyncio.get_running_loop()
        compacted = await loop.run_in_executor(None, session.compact)
        if not compacted:
            raise HTTPException(409, "Nothing to compact (or agent is busy)")
        return {"status": "compacted"}

    @app.get("/api/coder/sessions/{session_id}/log",
             dependencies=[Depends(_require_auth)])
    async def session_log(session_id: str):
        """Parsed JSONL audit log (log/full modes only)."""
        session = _get_session(session_id)
        path = session.audit_log_path()
        if path is None or not Path(path).is_file():
            raise HTTPException(404, "No audit log for this session "
                                     "(privacy mode keeps nothing)")
        entries = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"path": str(path), "entries": entries}

    @app.post("/api/coder/sessions/{session_id}/message",
              dependencies=[Depends(_require_auth)])
    async def session_message(session_id: str, req: MessageRequest):
        session = _get_session(session_id)
        if not req.text.strip():
            raise HTTPException(400, "Empty message")
        if not session.send_message(req.text):
            raise HTTPException(409, "Agent is busy with a previous task")
        return {"status": "started"}

    @app.post("/api/coder/sessions/{session_id}/confirm",
              dependencies=[Depends(_require_auth)])
    async def session_confirm(session_id: str, req: ConfirmRequest):
        session = _get_session(session_id)
        if not session.answer_confirm(req.confirm_id, req.approved):
            raise HTTPException(409, "No matching pending confirmation")
        return {"status": "answered", "approved": req.approved}

    @app.post("/api/coder/sessions/{session_id}/stop",
              dependencies=[Depends(_require_auth)])
    async def session_stop(session_id: str):
        session = _get_session(session_id)
        session.stop()
        return {"status": "stopping"}

    @app.delete("/api/coder/sessions/{session_id}",
                dependencies=[Depends(_require_auth)])
    async def session_delete(session_id: str):
        if manager.remove(session_id) is None:
            raise HTTPException(404, f"No such session: {session_id}")
        return {"status": "closed"}

    # ----------------------- model ops + jobs --------------------- #

    from .jobs import JobManager
    jobs = JobManager()

    @app.post("/api/models/pull", dependencies=[Depends(_require_auth)])
    async def model_pull(req: PullRequest):
        if not req.spec.strip():
            raise HTTPException(400, "Empty model spec")
        args = ["pull", req.spec]
        if req.name:
            args += ["--name", req.name]
        job = jobs.start_cli("pull", args)
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(_require_auth)])
    async def model_remove(req: RemoveModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model — switch first")
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"])
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(_require_auth)])
    async def model_alias(req: AliasRequest):
        from localm.config import load_registry
        registry = load_registry()
        if req.model not in registry:
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.alias in registry:
            raise HTTPException(409, f"Name already taken: {req.alias}")
        from localm.model_manager import alias_model
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, alias_model, req.model, req.alias)
        except Exception as e:
            raise HTTPException(400, f"Alias failed: {e}")
        return {"status": "aliased", "model": req.model, "alias": req.alias}

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(_require_auth)])
    async def job_events(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        loop = asyncio.get_running_loop()

        async def _stream():
            while True:
                try:
                    event = await loop.run_in_executor(
                        None, job.events.get, True, _KEEPALIVE_S)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "end":
                    return

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(_require_auth)])
    async def job_cancel(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        job.cancel()
        return {"status": "cancelling"}

    # ------------------ ComfyUI generation helpers ---------------- #
    # Shared by image (/api/imagine) and music (/api/music) generation.

    def _ensure_comfy(job) -> bool:
        """ComfyUI reachable? If not, launch it when configured, else
        tell the user exactly what to do."""
        import shlex
        import subprocess
        import sys as _sys
        import time as _t
        from localm.config import load_config
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        api_url = default_api_url()
        if _comfy_alive(api_url):
            return True
        launch_cmd = load_config().get("comfy_launch_cmd")
        if not launch_cmd:
            job.push({"type": "line", "text":
                      f"ComfyUI is not running at {api_url}. Start it "
                      "(your ComfyUI/Stability Matrix launcher) and retry, "
                      "or set a launch command so localm can start it:  "
                      "localm config comfy_launch_cmd \"D:\\path\\to\\comfyui.bat\""})
            return False
        job.push({"type": "line",
                  "text": f"ComfyUI not running — launching: {launch_cmd}"})
        # The command is the user's own config value (their launcher
        # script). cmd /c handles .bat files; shlex covers POSIX.
        if _sys.platform == "win32":
            argv = ["cmd", "/c", launch_cmd]
        else:
            argv = shlex.split(launch_cmd)
        try:
            subprocess.Popen(argv,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as e:
            job.push({"type": "line", "text": f"Launch failed: {e}"})
            return False
        deadline = _t.monotonic() + 180
        while _t.monotonic() < deadline:
            if _comfy_alive(api_url):
                job.push({"type": "line", "text": "ComfyUI is up."})
                return True
            _t.sleep(2)
        job.push({"type": "line",
                  "text": "ComfyUI did not come up within 3 minutes."})
        return False

    def _reload_llm(job) -> None:
        """Hand VRAM back: ask ComfyUI to drop its models, then reload
        the chat model so the next reply is instant.  Skipped when the
        reload_llm_after_imagine setting is off (e.g. batch generating)."""
        from localm.config import load_config
        if not load_config().get("reload_llm_after_imagine", True):
            job.push({"type": "line", "text":
                      "Keeping ComfyUI loaded (reload_llm_after_imagine is "
                      "off) — the chat model reloads on the next message."})
            return
        from localm.image_gen.comfy import free_comfy_vram
        if not free_comfy_vram():
            job.push({"type": "line", "text":
                      "ComfyUI kept its models in VRAM (no /free support) — "
                      "the chat model will reload on the next message instead."})
            return
        job.push({"type": "line", "text": "Reloading the chat model…"})
        try:
            import requests as _rq
            headers = {}
            key = os.environ.get("LOCALM_API_KEY")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            _rq.post(f"{self_url}/models/load", headers=headers, timeout=300)
            job.push({"type": "line", "text": "Chat model ready."})
        except Exception as e:
            job.push({"type": "line", "text":
                      f"Reload deferred to the next message ({e})."})

    # ----------------------- image generation --------------------- #

    from localm.config import home_dir
    images_dir = home_dir() / "gui_images"

    @app.post("/api/imagine", dependencies=[Depends(_require_auth)])
    async def imagine(req: ImagineRequest):
        if not req.prompt.strip():
            raise HTTPException(400, "Empty prompt")
        input_image = None
        if req.input_image:
            input_image = Path(req.input_image).expanduser()
            if not input_image.is_file():
                raise HTTPException(400, f"Input image not found: {req.input_image}")

        images_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        out_path = images_dir / f"{_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.png"

        def _generate(job):
            from localm.audit import SessionMode, effective_mode
            from localm.image_gen.comfy import generate_image
            if not _ensure_comfy(job):
                return False
            job.push({"type": "line", "text": "Submitting workflow to ComfyUI…"})
            ok, message = generate_image(
                req.prompt,
                out_path,
                guidance=req.guidance,
                negative_prompt=req.negative_prompt,
                seed=req.seed,
                input_image=input_image,
                denoise=req.denoise,
                localm_url=self_url,
                # privacy mode: the prompt never touches disk
                write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
            )
            job.push({"type": "line", "text": message})
            if ok:
                job.result = out_path.name
                _reload_llm(job)
            return ok

        job = jobs.start_fn("imagine", _generate, result_path=out_path.name)
        return {"job_id": job.id}

    def _confined_file(base: Path, name: str, kind: str) -> Path:
        """Resolve *name* and guarantee it stays directly inside *base*.

        Blocklisting separators is not enough on Windows: a drive-relative
        name like ``C:evil`` joins to ``C:evil`` (outside *base*), and an
        absolute name replaces the join entirely.  We verify the *resolved*
        path's parent is *base* and the basename is unchanged, which also
        rejects ``..``, nested subpaths, and ``con``/device names."""
        if name != Path(name).name or name in ("", ".", ".."):
            raise HTTPException(400, "Invalid file name")
        try:
            resolved = (base / name).resolve()
        except (OSError, ValueError):
            raise HTTPException(400, "Invalid file name")
        if resolved.parent != base.resolve() or resolved.name != name:
            raise HTTPException(400, "Invalid file name")
        if not resolved.is_file():
            raise HTTPException(404, f"No such {kind}")
        return resolved

    def _image_path(name: str) -> Path:
        return _confined_file(images_dir, name, "image")

    @app.get("/api/imagine/file/{name}", dependencies=[Depends(_require_auth)])
    async def imagine_file(name: str):
        from fastapi.responses import FileResponse
        return FileResponse(str(_image_path(name)), media_type="image/png")

    @app.delete("/api/imagine/file/{name}", dependencies=[Depends(_require_auth)])
    async def imagine_delete(name: str):
        path = _image_path(name)
        sidecar = path.with_suffix(path.suffix + ".json")
        path.unlink()
        if sidecar.is_file():
            sidecar.unlink()
        return {"status": "deleted", "name": name}

    @app.post("/api/imagine/file/{name}/move",
              dependencies=[Depends(_require_auth)])
    async def imagine_move(name: str, req: MoveImageRequest):
        """Move a generated image (and its metadata sidecar) to a folder on
        this machine — e.g. into a project or pictures directory."""
        import shutil
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

    @app.get("/api/imagine/history", dependencies=[Depends(_require_auth)])
    async def imagine_history():
        """Generated images, newest first, with their sidecar metadata."""
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

    # ----------------------- music generation --------------------- #

    music_dir = home_dir() / "gui_music"

    @app.post("/api/music", dependencies=[Depends(_require_auth)])
    async def music(req: MusicRequest):
        if not req.tags.strip():
            raise HTTPException(400, "Empty style tags")
        if req.duration_seconds <= 0 or req.duration_seconds > 3600:
            raise HTTPException(400, "Duration must be between 1 and 3600 seconds")

        music_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        out_path = music_dir / f"{_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.flac"

        def _generate(job):
            from localm.audit import SessionMode, effective_mode
            from localm.music_gen import generate_music
            if not _ensure_comfy(job):
                return False
            job.push({"type": "line", "text":
                      f"Submitting ACE-Step workflow to ComfyUI "
                      f"({req.duration_seconds:.0f}s track)…"})
            kwargs = {}
            if req.seed is not None:
                kwargs["seed"] = req.seed
            if req.steps is not None:
                kwargs["steps"] = req.steps
            if req.cfg is not None:
                kwargs["cfg"] = req.cfg
            if req.lyrics_strength is not None:
                kwargs["lyrics_strength"] = req.lyrics_strength
            ok, message = generate_music(
                req.tags,
                out_path,
                lyrics=req.lyrics,
                duration_seconds=req.duration_seconds,
                localm_url=self_url,
                on_progress=lambda t: job.push({"type": "line", "text": t}),
                write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
                **kwargs,
            )
            job.push({"type": "line", "text": message})
            if ok:
                job.result = out_path.name
                _reload_llm(job)
            return ok

        job = jobs.start_fn("music", _generate, result_path=out_path.name)
        return {"job_id": job.id}

    def _music_path(name: str) -> Path:
        return _confined_file(music_dir, name, "track")

    @app.get("/api/music/file/{name}", dependencies=[Depends(_require_auth)])
    async def music_file(name: str):
        from fastapi.responses import FileResponse
        path = _music_path(name)
        media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/flac"
        return FileResponse(str(path), media_type=media)

    @app.delete("/api/music/file/{name}", dependencies=[Depends(_require_auth)])
    async def music_delete(name: str):
        path = _music_path(name)
        sidecar = path.with_suffix(path.suffix + ".json")
        path.unlink()
        if sidecar.is_file():
            sidecar.unlink()
        return {"status": "deleted", "name": name}

    @app.post("/api/music/file/{name}/move",
              dependencies=[Depends(_require_auth)])
    async def music_move(name: str, req: MoveImageRequest):
        import shutil
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

    @app.get("/api/music/history", dependencies=[Depends(_require_auth)])
    async def music_history():
        """Generated tracks, newest first, with their sidecar metadata."""
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

    # ----------------------- video generation --------------------- #

    video_dir = home_dir() / "gui_video"

    @app.post("/api/video", dependencies=[Depends(_require_auth)])
    async def video(req: VideoRequest):
        if not req.prompt.strip():
            raise HTTPException(400, "Empty prompt")
        if req.seconds <= 0 or req.seconds > 20:
            raise HTTPException(400, "Duration must be between 1 and 20 seconds")
        if req.fps <= 0 or req.fps > 60:
            raise HTTPException(400, "FPS must be between 1 and 60")
        input_image = None
        if req.input_image:
            input_image = Path(req.input_image).expanduser()
            if not input_image.is_file():
                raise HTTPException(400, f"Input image not found: {req.input_image}")

        video_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        out_path = video_dir / f"{_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}.mp4"

        def _generate(job):
            from localm.audit import SessionMode, effective_mode
            from localm.video_gen import generate_video
            if not _ensure_comfy(job):
                return False
            job.push({"type": "line", "text":
                      f"Submitting Wan workflow to ComfyUI "
                      f"({req.seconds:.0f}s clip — video is slow, be patient)…"})
            kwargs = {}
            for field in ("negative_prompt", "seconds", "fps", "width",
                          "height", "seed", "steps", "cfg"):
                value = getattr(req, field)
                if value is not None:
                    kwargs[field] = value
            ok, message = generate_video(
                req.prompt,
                out_path,
                input_image=input_image,
                localm_url=self_url,
                on_progress=lambda t: job.push({"type": "line", "text": t}),
                write_sidecar=effective_mode("server") != SessionMode.PRIVACY,
                **kwargs,
            )
            job.push({"type": "line", "text": message})
            if ok:
                job.result = out_path.name
                _reload_llm(job)
            return ok

        job = jobs.start_fn("video", _generate, result_path=out_path.name)
        return {"job_id": job.id}

    def _video_path(name: str) -> Path:
        return _confined_file(video_dir, name, "clip")

    @app.get("/api/video/file/{name}", dependencies=[Depends(_require_auth)])
    async def video_file(name: str):
        from fastapi.responses import FileResponse
        path = _video_path(name)
        media = {".mp4": "video/mp4", ".webm": "video/webm",
                 ".gif": "image/gif"}.get(path.suffix.lower(),
                                          "application/octet-stream")
        return FileResponse(str(path), media_type=media)

    @app.delete("/api/video/file/{name}", dependencies=[Depends(_require_auth)])
    async def video_delete(name: str):
        path = _video_path(name)
        sidecar = path.with_suffix(path.suffix + ".json")
        path.unlink()
        if sidecar.is_file():
            sidecar.unlink()
        return {"status": "deleted", "name": name}

    @app.post("/api/video/file/{name}/move",
              dependencies=[Depends(_require_auth)])
    async def video_move(name: str, req: MoveImageRequest):
        import shutil
        path = _video_path(name)
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

    @app.get("/api/video/history", dependencies=[Depends(_require_auth)])
    async def video_history():
        """Generated clips, newest first, with their sidecar metadata."""
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
        return {"videos": items}

    # ------------------------ model discovery --------------------- #
    # Search HuggingFace for GGUF models and show per-quant "fits your
    # VRAM" badges. User-initiated prelude to a pull (docs/network.md);
    # net_mode=off blocks it like everything else.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF unreachable
        return 422              # bad repo / no GGUF files

    @app.get("/api/discover/search", dependencies=[Depends(_require_auth)])
    async def discover_search(q: str = "", limit: int = 20):
        from localm.discover import DiscoverError, hf_search, vram_info
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: hf_search(q, limit=limit))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        return {"query": q, "results": results, "vram": vram_info()}

    @app.get("/api/discover/files", dependencies=[Depends(_require_auth)])
    async def discover_files(repo: str):
        from localm.discover import (DiscoverError, fit_label, hf_gguf_files,
                                     vram_info)
        loop = asyncio.get_running_loop()
        try:
            files = await loop.run_in_executor(
                None, lambda: hf_gguf_files(repo))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        vram = vram_info()
        total = vram.get("total")
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
        return {"repo": repo.strip().strip("/"), "files": files, "vram": vram}

    # ------------------- chat conversation store ------------------ #
    # Server-side persistence for GUI chat conversations so they survive
    # browser reloads, profile wipes, and other devices on the LAN.
    # Strictly gated on the chat surface's session mode: in privacy mode
    # (the default) nothing is readable or writable here and the GUI keeps
    # conversations in memory only.

    import re as _re

    chats_dir = home_dir() / "chats"
    _CONV_ID = _re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    _CONV_MAX_BYTES = 16 * 1024 * 1024   # data-URI images make these large

    def _chat_persist_enabled() -> bool:
        from localm.audit import SessionMode, effective_mode
        return effective_mode("chat") != SessionMode.PRIVACY

    def _conv_path(conv_id: str) -> Path:
        if not _CONV_ID.match(conv_id):
            raise HTTPException(400, "Invalid conversation id")
        return chats_dir / f"{conv_id}.json"

    @app.get("/api/conversations", dependencies=[Depends(_require_auth)])
    async def conversations_list():
        if not _chat_persist_enabled():
            return {"enabled": False, "conversations": []}
        items = []
        if chats_dir.is_dir():
            for p in chats_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    data["id"] = p.stem
                    items.append(data)
                except Exception:
                    continue   # corrupt file — skip, never block the list
        items.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
        return {"enabled": True, "conversations": items[:200]}

    @app.put("/api/conversations/{conv_id}",
             dependencies=[Depends(_require_auth)])
    async def conversation_upsert(conv_id: str, req: ConversationUpsert):
        if not _chat_persist_enabled():
            raise HTTPException(
                403, "Chat persistence is off (privacy mode). "
                     "Set mode/chat_mode to 'log' or 'full' to enable it.")
        path = _conv_path(conv_id)
        payload = json.dumps(
            {"id": conv_id, "title": req.title,
             "updated_at": req.updated_at,
             "pinned": req.pinned, "folder": req.folder,
             "branches": req.branches,
             "messages": req.messages},
            ensure_ascii=False)
        if len(payload.encode("utf-8")) > _CONV_MAX_BYTES:
            raise HTTPException(413, "Conversation too large to persist")
        chats_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        return {"status": "saved", "id": conv_id}

    @app.delete("/api/conversations/{conv_id}",
                dependencies=[Depends(_require_auth)])
    async def conversation_delete(conv_id: str):
        if not _chat_persist_enabled():
            raise HTTPException(403, "Chat persistence is off (privacy mode)")
        path = _conv_path(conv_id)
        if path.is_file():
            path.unlink()
            return {"status": "deleted", "id": conv_id}
        return {"status": "absent", "id": conv_id}

    # -------------------------- web access ------------------------ #
    # Search and fetch for the chat surface, enforced by localm.netpolicy
    # (net_mode, net_allow/net_deny, SSRF guard). Callers are either the
    # user's explicit /web command or the per-conversation web-access toggle
    # — both are direct consent, so "ask" mode does not re-prompt here; only
    # "off" blocks. Domain rules and the private-address guard always apply.

    @app.post("/api/web/search", dependencies=[Depends(_require_auth)])
    async def web_search_endpoint(req: WebSearchRequest):
        from localm.netpolicy import NetworkPolicyError, web_search
        if not req.query.strip():
            raise HTTPException(400, "Empty query")
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: web_search(req.query, max_results=req.max_results))
        except NetworkPolicyError as e:
            raise HTTPException(403, str(e))
        except Exception as e:
            raise HTTPException(502, f"Search failed: {e}")
        return {"query": req.query, "results": results}

    @app.post("/api/web/fetch", dependencies=[Depends(_require_auth)])
    async def web_fetch_endpoint(req: WebFetchRequest):
        from localm.netpolicy import NetworkPolicyError, fetch_text
        loop = asyncio.get_running_loop()
        try:
            final_url, text = await loop.run_in_executor(
                None, lambda: fetch_text(req.url))
        except NetworkPolicyError as e:
            raise HTTPException(403, str(e))
        except Exception as e:
            raise HTTPException(502, f"Fetch failed: {e}")
        max_chars = max(500, min(req.max_chars, 60_000))
        return {"url": final_url,
                "text": text[:max_chars],
                "truncated": len(text) > max_chars}

    # ----------------------------- voice --------------------------- #
    # Whisper STT for the mic button. Audio is decoded in memory and never
    # written to disk, so privacy mode stays trace-free. TTS needs no
    # endpoint — the browser's speechSynthesis is offline by construction.

    @app.post("/api/voice/transcribe", dependencies=[Depends(_require_auth)])
    async def voice_transcribe(req: TranscribeRequest):
        import base64
        from localm.voice import VoiceError, transcribe_bytes
        try:
            data = base64.b64decode(req.audio_b64, validate=True)
        except Exception:
            raise HTTPException(400, "audio_b64 is not valid base64")
        if len(data) > 25_000_000:
            raise HTTPException(413, "Recording too large (max 25 MB)")
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                None, lambda: transcribe_bytes(data, language=req.language))
        except VoiceError as e:
            status = 501 if "faster-whisper" in str(e) else 422
            raise HTTPException(status, str(e))
        return {"text": text}

    # ----------------------- assistant memory --------------------- #
    # ChatGPT-style persistent memory for chat: a plain markdown file the
    # user can read and edit, injected into the system prompt when the
    # drawer toggle is on. LOCALCODER.md is the coder-side analogue.
    #
    # Privacy semantics: privacy mode means "no new traces", not amnesia —
    # READING memory written by earlier non-privacy sessions is allowed,
    # but WRITES (which would persist conversation-derived facts) return
    # 403 while privacy is active.

    memory_file = home_dir() / "chat-memory.md"
    _MEMORY_MAX = 64_000   # characters — keep injection bounded

    def _memory_writable() -> bool:
        from localm.audit import SessionMode, effective_mode
        return effective_mode("chat") != SessionMode.PRIVACY

    def _read_memory() -> str:
        if memory_file.is_file():
            try:
                return memory_file.read_text(encoding="utf-8")
            except OSError:
                return ""
        return ""

    def _write_memory(text: str) -> None:
        text = text.strip()
        if len(text) > _MEMORY_MAX:
            raise HTTPException(413, "Memory file too large (max 64k chars)")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not text:
            memory_file.unlink(missing_ok=True)
            return
        tmp = memory_file.with_name(memory_file.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(memory_file)

    @app.get("/api/memory", dependencies=[Depends(_require_auth)])
    async def memory_get():
        return {"text": _read_memory(), "writable": _memory_writable(),
                "path": str(memory_file)}

    @app.put("/api/memory", dependencies=[Depends(_require_auth)])
    async def memory_put(req: MemoryUpdate):
        if not _memory_writable():
            raise HTTPException(
                403, "Memory writes are off in privacy mode (no new traces). "
                     "Set mode/chat_mode to 'log' or 'full' to enable them.")
        _write_memory(req.text)
        return {"status": "saved", "chars": len(req.text.strip())}

    @app.post("/api/memory/append", dependencies=[Depends(_require_auth)])
    async def memory_append(req: MemoryAppend):
        if not _memory_writable():
            raise HTTPException(
                403, "Memory writes are off in privacy mode (no new traces)")
        fact = req.text.strip()
        if not fact:
            raise HTTPException(400, "Nothing to remember")
        current = _read_memory().strip()
        line = fact if fact.startswith("-") else f"- {fact}"
        _write_memory((current + "\n" + line) if current else line)
        return {"status": "appended"}

    # ------------------------ prompt library ---------------------- #
    # Named personas: a system prompt plus sampling defaults, applied from
    # the chat parameters drawer. Explicit user assets (like knowledge
    # collections), stored in <data dir>/prompts.json in every session mode.

    prompts_file = home_dir() / "prompts.json"

    def _check_prompt_name(name: str) -> str:
        name = (name or "").strip()
        if not name or len(name) > 64 or any(c in name for c in "\n\r\t"):
            raise HTTPException(
                400, "Persona names must be 1-64 characters on one line")
        return name

    def _load_prompts() -> dict:
        if prompts_file.is_file():
            try:
                return json.loads(prompts_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_prompts(data: dict) -> None:
        prompts_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = prompts_file.with_name(prompts_file.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(prompts_file)

    @app.get("/api/prompts", dependencies=[Depends(_require_auth)])
    async def prompts_list():
        data = _load_prompts()
        return {"prompts": [
            {"name": name, **entry} for name, entry in sorted(data.items())
        ]}

    @app.put("/api/prompts/{name}", dependencies=[Depends(_require_auth)])
    async def prompt_upsert(name: str, req: PromptUpsert):
        name = _check_prompt_name(name)
        data = _load_prompts()
        data[name] = {"system": req.system, "params": req.params}
        _save_prompts(data)
        return {"status": "saved", "name": name}

    @app.delete("/api/prompts/{name}", dependencies=[Depends(_require_auth)])
    async def prompt_delete(name: str):
        name = _check_prompt_name(name)
        data = _load_prompts()
        if name not in data:
            raise HTTPException(404, f"No such persona: {name}")
        del data[name]
        _save_prompts(data)
        return {"status": "deleted", "name": name}

    # --------------------------- knowledge ------------------------ #
    # Document collections for retrieval-augmented chat (localm.rag).
    # Collections are explicit user data — indexing writes to <data dir>/rag/
    # in every session mode, like generated images. /api/rag/extract is the
    # exception: it converts an uploaded attachment to text entirely in
    # memory, so privacy-mode chats can use documents without leaving traces.

    def _self_embed(texts: list) -> list:
        """Embed via this server's own /v1/embeddings — the endpoint holds
        the inference semaphore, so indexing never races a chat reply.
        Raises when the backend has no embedding support (GGUF ctypes
        binding); callers degrade to lexical-only."""
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = _rq.post(f"{self_url}/embeddings",
                     json={"input": texts, "model": active_model() or "localm"},
                     headers=headers, timeout=600)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def _get_collection(name: str):
        from localm.rag import Collection
        try:
            coll = Collection(name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not coll.exists():
            raise HTTPException(404, f"No such collection: {name}")
        return coll

    @app.get("/api/rag/collections", dependencies=[Depends(_require_auth)])
    async def rag_collections():
        from localm.rag import Collection, collection_names
        return {"collections": [Collection(n).stats()
                                for n in collection_names()]}

    @app.post("/api/rag/collections", dependencies=[Depends(_require_auth)])
    async def rag_create(req: RagCreateRequest):
        from localm.rag import Collection
        try:
            coll = Collection(req.name.strip())
        except ValueError as e:
            raise HTTPException(400, str(e))
        if coll.exists():
            raise HTTPException(409, f"Collection already exists: {coll.name}")
        coll.create()
        return coll.stats()

    @app.get("/api/rag/collections/{name}",
             dependencies=[Depends(_require_auth)])
    async def rag_detail(name: str):
        coll = _get_collection(name)
        return {**coll.stats(), "docs": coll.docs()}

    @app.delete("/api/rag/collections/{name}",
                dependencies=[Depends(_require_auth)])
    async def rag_delete(name: str):
        from localm.rag import delete_collection
        try:
            if not delete_collection(name):
                raise HTTPException(404, f"No such collection: {name}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"status": "deleted", "name": name}

    @app.post("/api/rag/collections/{name}/add",
              dependencies=[Depends(_require_auth)])
    async def rag_add(name: str, req: RagAddRequest):
        coll = _get_collection(name)
        paths = [Path(p).expanduser() for p in req.paths if p.strip()]
        if not paths:
            raise HTTPException(400, "No paths given")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise HTTPException(400, f"Not found: {', '.join(missing[:5])}")
        embed = req.embed

        def _index(job):
            embed_fn = _self_embed if embed else None
            result = coll.add_paths(
                paths, embed_fn=embed_fn,
                on_progress=lambda t: job.push({"type": "line", "text": t}))
            summary = (f"done: {result['added']} added, "
                       f"{result['updated']} updated, "
                       f"{result['skipped']} unchanged, "
                       f"{len(result['failed'])} failed — "
                       f"{result['chunks']} chunks total")
            job.push({"type": "line", "text": summary})
            for f in result["failed"][:10]:
                job.push({"type": "line",
                          "text": f"  failed: {f['path']}: {f['error']}"})
            return True

        job = jobs.start_fn("rag-index", _index)
        return {"job_id": job.id}

    @app.post("/api/rag/collections/{name}/query",
              dependencies=[Depends(_require_auth)])
    async def rag_query(name: str, req: RagQueryRequest):
        coll = _get_collection(name)
        if not req.query.strip():
            raise HTTPException(400, "Empty query")
        k = max(1, min(req.k, 20))
        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(
            None, lambda: coll.query(req.query, k=k, embed_fn=_self_embed))
        return {"collection": name, "query": req.query, "hits": hits}

    @app.post("/api/rag/collections/{name}/remove-doc",
              dependencies=[Depends(_require_auth)])
    async def rag_remove_doc(name: str, req: RagRemoveDocRequest):
        coll = _get_collection(name)
        if not coll.remove_doc(req.path):
            raise HTTPException(404, f"Not in this collection: {req.path}")
        return {"status": "removed", "path": req.path}

    @app.post("/api/rag/extract", dependencies=[Depends(_require_auth)])
    async def rag_extract(req: RagExtractRequest):
        """Uploaded chat attachment → plain text, entirely in memory."""
        import base64
        from localm.rag import ExtractError, extract_bytes
        try:
            data = base64.b64decode(req.content_b64, validate=True)
        except Exception:
            raise HTTPException(400, "content_b64 is not valid base64")
        if len(data) > 30_000_000:
            raise HTTPException(413, "Attachment too large (max 30 MB)")
        try:
            text = extract_bytes(data, req.filename)
        except ExtractError as e:
            raise HTTPException(422, str(e))
        max_chars = max(500, min(req.max_chars, 200_000))
        return {"filename": req.filename,
                "text": text[:max_chars],
                "chars": len(text),
                "truncated": len(text) > max_chars}

    # --------------------- coder session history ------------------ #
    # Read-only browser for past audit logs (~/.localm/sessions/*.jsonl,
    # written in log/full modes). Live sessions have /log; this lists what
    # earlier sessions — including ones from before a server restart — left
    # behind. Privacy mode writes no logs, so the list is simply empty.

    @app.get("/api/coder/history", dependencies=[Depends(_require_auth)])
    async def coder_history():
        from localm import audit as _audit
        from localm.audit import SessionMode, effective_mode
        sessions_dir = _audit._SESSIONS_DIR
        items = []
        if sessions_dir.is_dir():
            for p in sorted(sessions_dir.glob("*.jsonl"),
                            key=lambda f: f.stat().st_mtime, reverse=True)[:100]:
                items.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                })
        return {"enabled": effective_mode("coder") != SessionMode.PRIVACY,
                "logs": items}

    @app.get("/api/coder/history/{name}",
             dependencies=[Depends(_require_auth)])
    async def coder_history_entries(name: str):
        from localm import audit as _audit
        if not name.endswith(".jsonl"):
            raise HTTPException(400, "Invalid log name")
        path = _confined_file(_audit._SESSIONS_DIR, name, "session log")
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"path": str(path), "entries": entries}

    # ------------------------- static ----------------------------- #
    # Mounted last: API routes above take precedence over the SPA files.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="gui")

    return manager
