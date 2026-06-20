# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder plugin: the offline AI coding agent on the GUI surface.

Routes (mounted by the engine, auto-scoped to the ``coder`` capability):
  GET    /api/coder/sessions                    - list live sessions + active model
  POST   /api/coder/sessions                    - start a session (optional model switch)
  GET    /api/coder/sessions/{id}/events        - SSE event stream (?replay=true)
  POST   /api/coder/sessions/{id}/message       - send a task / steering message
  POST   /api/coder/sessions/{id}/confirm       - answer a pending confirmation
  POST   /api/coder/sessions/{id}/undo          - revert the last file write
  POST   /api/coder/sessions/{id}/compact       - summarise old history
  GET    /api/coder/sessions/{id}/log           - parsed JSONL audit log (log/full)
  GET    /api/coder/sessions/{id}/files         - files changed this session
  GET    /api/coder/sessions/{id}/files/diff    - cumulative session diff
  POST   /api/coder/sessions/{id}/stop          - interrupt the current task
  DELETE /api/coder/sessions/{id}               - terminate the session
  GET    /api/coder/history                     - browse past session audit logs
  GET    /api/coder/history/{name}              - read one past audit log

The agentic coder runs shell commands and writes files, so the engine gates every
route above on the ``coder`` capability scope. Live sessions need the kernel GUI's
shared services (``request.app.state.coder_sessions`` / ``.switch_model`` /
``.self_url`` / ``.active_model``), published by ``attach_gui``; the routes degrade
to 503 when those are absent (headless / no GUI). The read-only history endpoints
read ``~/.localm/sessions/*.jsonl`` directly and work without the GUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from localm.pathsafe import confined_file as _confined_file
from localm.plugins.coder.sessions import CoderSession

_router = APIRouter()

# SSE keepalive interval - must beat proxy/browser idle timeouts.
_KEEPALIVE_S = 15


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

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


class MessageRequest(BaseModel):
    text: str


class ConfirmRequest(BaseModel):
    confirm_id: str
    approved: bool
    always_allow: bool = False        # approve + whitelist the tool this session


# ------------------------------------------------------------------ #
#  Shared-service access (published by the GUI's attach_gui)          #
# ------------------------------------------------------------------ #

def _sessions(request: Request):
    """The live SessionManager, or 503 when the GUI server isn't running."""
    mgr = getattr(request.app.state, "coder_sessions", None)
    if mgr is None:
        raise HTTPException(503, "Coder sessions need the localm GUI server "
                                 "(run `localm gui`).")
    return mgr


def _principal_from_request(request: Request) -> tuple[bool, str | None]:
    """Identify the caller for session isolation: returns ``(is_owner, principal)``.

    The OWNER (an ADMIN/owner key, or open-mode loopback) is ``(True, None)`` and
    may touch every session. A minted, non-owner scoped key is ``(False, <hash of
    its bearer>)`` and may touch only the sessions IT created. The principal is a
    SHA-256 of the presented bearer, so it identifies the key without storing it."""
    from localm import scopes as S
    from localm.auth import any_key_configured, verify
    from localm.inference.http_server import _bearer_token
    if not any_key_configured():
        return True, None                       # open mode = loopback owner
    bearer = _bearer_token(request)
    held = verify(bearer) if bearer else None
    if held is not None and S.ADMIN in held:
        return True, None                       # the owner key
    import hashlib
    digest = hashlib.sha256((bearer or "").encode("utf-8")).hexdigest()
    return False, digest


def _get_session(request: Request, session_id: str):
    session = _sessions(request).get(session_id)
    if session is None:
        raise HTTPException(404, f"No such session: {session_id}")
    is_owner, principal = _principal_from_request(request)
    # Isolation: a scoped caller may only touch the sessions IT created - it must
    # not read or steer the owner's full-capability sessions (which keep run_shell).
    # 404 (not 403) so a scoped key cannot even probe which session ids exist.
    if not is_owner and session.principal != principal:
        raise HTTPException(404, f"No such session: {session_id}")
    return session


# ------------------------------------------------------------------ #
#  Session lifecycle                                                  #
# ------------------------------------------------------------------ #

@_router.get("/api/coder/sessions")
async def list_sessions(request: Request):
    mgr = _sessions(request)
    is_owner, principal = _principal_from_request(request)
    return {"sessions": mgr.list(principal=principal, is_owner=is_owner),
            "active_model": request.app.state.active_model()}


@_router.post("/api/coder/sessions")
async def create_session(req: CreateSessionRequest, request: Request):
    mgr = _sessions(request)
    active_model = request.app.state.active_model
    self_url = request.app.state.self_url

    # Safe-to-share scoping. The OWNER (an ADMIN/owner key, or open-mode loopback)
    # gets the full coder: any directory, every tool. A MINTED, non-owner coder-
    # scoped key gets a RESTRICTED session - locked to read + confined-edit tools
    # (no run_shell/run_tests/git-hooks/network/sub-agents, all RCE or exfil
    # vectors), forced into the project root, and isolated to its own sessions. So
    # a key you hand out can read and edit this project but cannot execute
    # anything; you review and run. (run_shell is cwd-independent, and write_file +
    # run_tests/git-hooks is RCE, so disabling the executing tools - not just
    # confining the cwd - is the real containment.)
    is_owner, principal = _principal_from_request(request)
    restricted = not is_owner

    if restricted:
        # Force the session into the instance's project root, ignoring req.cwd, so
        # a scoped key cannot point the (confined) file tools at arbitrary paths.
        root = getattr(request.app.state, "root_dir", None) or str(Path.cwd())
        cwd = Path(root).expanduser().resolve()
        if not cwd.is_dir():
            raise HTTPException(400, f"Project root is not a directory: {cwd}")
    else:
        cwd = Path(req.cwd).expanduser()
        if not cwd.is_dir():
            raise HTTPException(400, f"Not a directory: {req.cwd}")
        cwd = cwd.resolve()

    # A per-session model switch changes the one shared engine for EVERYONE, so a
    # scoped key must not trigger it (DoS / interfering with the owner's session).
    if req.model and req.model != active_model():
        if restricted:
            raise HTTPException(
                403, "Switching models needs the owner key; a scoped key uses the "
                "active model.")
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        switch_model = getattr(request.app.state, "switch_model", None)
        if switch_model is None:
            raise HTTPException(503, "Model switching needs the localm GUI server.")
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
        cwd,
        backend,
        auto_approve=req.auto_approve,
        max_turns=req.max_turns,
        mode=session_mode,
        scope=req.scope,
        dry_run=req.dry_run,
        restricted=restricted,
        **gen_kwargs,
    ))
    session.principal = principal      # who owns this session (None = the owner)
    mgr.create(session)
    return session.info()


@_router.get("/api/coder/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request, replay: bool = False):
    """SSE event stream. ``?replay=true`` first re-sends the session's
    event history (so a reloaded page rebuilds its feed), then goes live."""
    session = _get_session(request, session_id)
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


@_router.post("/api/coder/sessions/{session_id}/undo")
async def session_undo(session_id: str, request: Request):
    session = _get_session(request, session_id)
    summary = session.undo()
    if summary is None:
        raise HTTPException(409, "Nothing to undo (or agent is busy)")
    return {"status": "undone", "summary": summary}


@_router.post("/api/coder/sessions/{session_id}/compact")
async def session_compact(session_id: str, request: Request):
    session = _get_session(request, session_id)
    loop = asyncio.get_running_loop()
    compacted = await loop.run_in_executor(None, session.compact)
    if not compacted:
        raise HTTPException(409, "Nothing to compact (or agent is busy)")
    return {"status": "compacted"}


@_router.get("/api/coder/sessions/{session_id}/log")
async def session_log(session_id: str, request: Request):
    """Parsed JSONL audit log (log/full modes only)."""
    session = _get_session(request, session_id)
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


@_router.post("/api/coder/sessions/{session_id}/message")
async def session_message(session_id: str, req: MessageRequest, request: Request):
    session = _get_session(request, session_id)
    if not req.text.strip():
        raise HTTPException(400, "Empty message")
    status = session.send_message(req.text)
    if status == "closed":
        raise HTTPException(409, "Session is closed")
    # "started" begins a task; "queued" steers the running one - the text
    # is injected into the conversation at the next turn boundary.
    return {"status": status}


@_router.post("/api/coder/sessions/{session_id}/confirm")
async def session_confirm(session_id: str, req: ConfirmRequest, request: Request):
    session = _get_session(request, session_id)
    if not session.answer_confirm(req.confirm_id, req.approved,
                                  always_allow=req.always_allow):
        raise HTTPException(409, "No matching pending confirmation")
    return {"status": "answered", "approved": req.approved,
            "always_allow": req.approved and req.always_allow}


@_router.get("/api/coder/sessions/{session_id}/files")
async def session_files(session_id: str, request: Request):
    """Files the agent has changed this session, with change counts."""
    session = _get_session(request, session_id)
    return {"files": session.changed_files()}


@_router.get("/api/coder/sessions/{session_id}/files/diff")
async def session_files_diff(session_id: str, request: Request, path: str = ""):
    """Cumulative unified diff of session changes (?path= for one file).

    Diffs only files the agent's tracker recorded - arbitrary paths
    cannot be read through this endpoint."""
    session = _get_session(request, session_id)
    loop = asyncio.get_running_loop()
    diff = await loop.run_in_executor(
        None, session.session_diff, path or None)
    if path and not diff:
        raise HTTPException(404, f"'{path}' was not changed this session")
    return {"diff": diff}


@_router.post("/api/coder/sessions/{session_id}/stop")
async def session_stop(session_id: str, request: Request):
    session = _get_session(request, session_id)
    session.stop()
    return {"status": "stopping"}


@_router.delete("/api/coder/sessions/{session_id}")
async def session_delete(session_id: str, request: Request):
    if _sessions(request).remove(session_id) is None:
        raise HTTPException(404, f"No such session: {session_id}")
    return {"status": "closed"}


# ------------------------------------------------------------------ #
#  Session history (read-only; works without the GUI services)        #
# ------------------------------------------------------------------ #
# Browser for past audit logs (~/.localm/sessions/*.jsonl, written in log/full
# modes). Live sessions have /log; this lists what earlier sessions - including
# ones from before a server restart - left behind. Privacy mode writes no logs,
# so the list is simply empty.

@_router.get("/api/coder/history")
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


@_router.get("/api/coder/history/{name}")
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


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
