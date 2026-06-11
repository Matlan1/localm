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
    mode: str = "privacy"
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


class MessageRequest(BaseModel):
    text: str


class ConfirmRequest(BaseModel):
    confirm_id: str
    approved: bool


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

        loop = asyncio.get_running_loop()
        # Agent construction scans the project (map build) — keep it off the loop
        session = await loop.run_in_executor(None, lambda: CoderSession(
            cwd.resolve(),
            backend,
            auto_approve=req.auto_approve,
            max_turns=req.max_turns,
            mode=req.mode,
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

    # ----------------------- image generation --------------------- #

    images_dir = Path.home() / ".localm" / "gui_images"

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
            the chat model so the next reply is instant."""
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

        def _generate(job):
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
            )
            job.push({"type": "line", "text": message})
            if ok:
                job.result = out_path.name
                _reload_llm(job)
            return ok

        job = jobs.start_fn("imagine", _generate, result_path=out_path.name)
        return {"job_id": job.id}

    @app.get("/api/imagine/file/{name}", dependencies=[Depends(_require_auth)])
    async def imagine_file(name: str):
        from fastapi.responses import FileResponse
        # Confine to the GUI images directory — no path components allowed
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "Invalid file name")
        path = images_dir / name
        if not path.is_file():
            raise HTTPException(404, "No such image")
        return FileResponse(str(path), media_type="image/png")

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
                              "mtime": p.stat().st_mtime})
        return {"images": items}

    # ------------------------- static ----------------------------- #
    # Mounted last: API routes above take precedence over the SPA files.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="gui")

    return manager
