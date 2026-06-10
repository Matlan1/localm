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
    temperature: float | None = None
    max_tokens: int | None = None


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

    @app.post("/api/coder/sessions", dependencies=[Depends(_require_auth)])
    async def create_session(req: CreateSessionRequest):
        cwd = Path(req.cwd).expanduser()
        if not cwd.is_dir():
            raise HTTPException(400, f"Not a directory: {req.cwd}")

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
            **gen_kwargs,
        ))
        manager.create(session)
        return {
            "id": session.id,
            "cwd": str(session.cwd),
            "model": active_model(),
            "auto_approve": req.auto_approve,
        }

    def _get_session(session_id: str) -> CoderSession:
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(404, f"No such session: {session_id}")
        return session

    @app.get("/api/coder/sessions/{session_id}/events",
             dependencies=[Depends(_require_auth)])
    async def session_events(session_id: str):
        session = _get_session(session_id)
        loop = asyncio.get_running_loop()

        async def _stream():
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

    # ------------------------- static ----------------------------- #
    # Mounted last: API routes above take precedence over the SPA files.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="gui")

    return manager
