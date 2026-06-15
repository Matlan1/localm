"""
GUI web layer - API routes and static file serving, attached to the
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
from localm.pathsafe import confined_file as _confined_file
from .sessions import CoderSession, SessionManager

STATIC_DIR = Path(__file__).parent / "static"

# SSE keepalive interval - must beat proxy/browser idle timeouts
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
    dry_run: bool = False             # destructive tools report but don't run
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


class MessageRequest(BaseModel):
    text: str


class ConfirmRequest(BaseModel):
    confirm_id: str
    approved: bool
    always_allow: bool = False        # approve + whitelist the tool this session


class ConversationUpsert(BaseModel):
    title: str = "Untitled"
    updated_at: float = 0
    pinned: bool = False
    folder: str | None = None
    branches: list = []           # parked message-branch tails (fork points)
    messages: list = []


class PromptUpsert(BaseModel):
    system: str = ""
    params: dict = {}             # sampling defaults (temperature, top_p, …)


class MemoryUpdate(BaseModel):
    text: str


class MemoryAppend(BaseModel):
    text: str


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
        Base URL of this server's own /v1 API - coder agents talk to the
        model through it (e.g. ``http://127.0.0.1:8642/v1``).
    switch_model:
        ``Callable[[str], Awaitable[None]]`` - swaps the active engine.
    active_model:
        ``Callable[[], str]`` - name of the currently served model.
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
        # Agent construction scans the project (map build) - keep it off the loop
        session = await loop.run_in_executor(None, lambda: CoderSession(
            cwd.resolve(),
            backend,
            auto_approve=req.auto_approve,
            max_turns=req.max_turns,
            mode=session_mode,
            scope=req.scope,
            dry_run=req.dry_run,
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
                # Snapshot history, then drop anything still queued - those
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
        status = session.send_message(req.text)
        if status == "closed":
            raise HTTPException(409, "Session is closed")
        # "started" begins a task; "queued" steers the running one - the text
        # is injected into the conversation at the next turn boundary.
        return {"status": status}

    @app.post("/api/coder/sessions/{session_id}/confirm",
              dependencies=[Depends(_require_auth)])
    async def session_confirm(session_id: str, req: ConfirmRequest):
        session = _get_session(session_id)
        if not session.answer_confirm(req.confirm_id, req.approved,
                                      always_allow=req.always_allow):
            raise HTTPException(409, "No matching pending confirmation")
        return {"status": "answered", "approved": req.approved,
                "always_allow": req.approved and req.always_allow}

    @app.get("/api/coder/sessions/{session_id}/files",
             dependencies=[Depends(_require_auth)])
    async def session_files(session_id: str):
        """Files the agent has changed this session, with change counts."""
        session = _get_session(session_id)
        return {"files": session.changed_files()}

    @app.get("/api/coder/sessions/{session_id}/files/diff",
             dependencies=[Depends(_require_auth)])
    async def session_files_diff(session_id: str, path: str = ""):
        """Cumulative unified diff of session changes (?path= for one file).

        Diffs only files the agent's tracker recorded - arbitrary paths
        cannot be read through this endpoint."""
        session = _get_session(session_id)
        loop = asyncio.get_running_loop()
        diff = await loop.run_in_executor(
            None, session.session_diff, path or None)
        if path and not diff:
            raise HTTPException(404, f"'{path}' was not changed this session")
        return {"diff": diff}

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

    @app.get("/api/fs/dirs", dependencies=[Depends(_require_auth)])
    async def fs_dirs(path: str = ""):
        """Subdirectories of *path*, for the coder setup directory picker.

        An empty path lists drive roots on Windows (filesystem root
        elsewhere). Only directory names leave the server - no file
        names or contents. The GUI is localhost + bearer-auth, and the
        coder agent this picker feeds can read those directories anyway.
        """
        if not path:
            if os.name == "nt":
                import string
                roots = [f"{letter}:\\" for letter in string.ascii_uppercase
                         if Path(f"{letter}:\\").is_dir()]
                return {"path": "", "parent": None, "dirs": roots}
            path = "/"
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a directory: {path}")
        p = p.resolve()
        dirs = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                try:
                    if child.is_dir() and not child.name.startswith("."):
                        dirs.append(child.name)
                except OSError:
                    continue   # broken junction / reparse point
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {path}")
        at_root = p.parent == p
        return {"path": str(p),
                "parent": "" if at_root else str(p.parent),
                "dirs": dirs}

    # ----------------------- model ops + jobs --------------------- #

    from .jobs import JobManager
    jobs = JobManager()

    # Shared services that converted plugins (rag/image/music/video) reach via
    # request.app.state: the background-job manager, this server's own /v1 base
    # URL (for self-embedding), and the active-model accessor. The /api/jobs/*
    # SSE endpoints stay here in the kernel GUI so every plugin's jobs stream
    # through one manager.
    app.state.jobs = jobs
    app.state.self_url = self_url
    app.state.active_model = active_model

    @app.post("/api/models/pull", dependencies=[Depends(_require_auth)])
    async def model_pull(req: PullRequest):
        if not req.spec.strip():
            raise HTTPException(400, "Empty model spec")
        args = ["pull", req.spec]
        if req.name:
            args += ["--name", req.name]
        # Stream structured download progress; suppress huggingface_hub's own
        # tqdm bars (their \r output doesn't line-stream cleanly).
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        })
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(_require_auth)])
    async def model_remove(req: RemoveModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
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

    # Media generation (image /api/imagine*, music /api/music*, video /api/video*)
    # moved to standalone builtin plugins (localm/plugins/builtin/{image,music,
    # video}) in Phase 3; each ships disabled by default and reads its own
    # per-plugin backend config (the shared ComfyUI launch/reload helpers moved
    # into those plugins' backends). _confined_file lives in localm.pathsafe
    # (imported at module top) - still used below by coder_history.

    from localm.config import home_dir

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
                    continue   # corrupt file - skip, never block the list
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

    # Web search and fetch (/api/web/*) moved to the builtin "web" plugin
    # (localm/plugins/builtin/web) in Phase 3; it ships disabled by default.

    # ----------------------- assistant memory --------------------- #
    # ChatGPT-style persistent memory for chat: a plain markdown file the
    # user can read and edit, injected into the system prompt when the
    # drawer toggle is on. LOCALCODER.md is the coder-side analogue.
    #
    # Privacy semantics: privacy mode means "no new traces", not amnesia -
    # READING memory written by earlier non-privacy sessions is allowed,
    # but WRITES (which would persist conversation-derived facts) return
    # 403 while privacy is active.

    memory_file = home_dir() / "chat-memory.md"
    _MEMORY_MAX = 64_000   # characters - keep injection bounded

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

    # Knowledge / RAG (/api/rag/*) moved to the builtin "rag" plugin
    # (localm/plugins/builtin/rag) in Phase 3; it ships disabled by default and
    # reaches the shared job manager + self-embed URL via request.app.state.

    # --------------------- coder session history ------------------ #
    # Read-only browser for past audit logs (~/.localm/sessions/*.jsonl,
    # written in log/full modes). Live sessions have /log; this lists what
    # earlier sessions - including ones from before a server restart - left
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
