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
from fastapi.responses import FileResponse, StreamingResponse
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
    resume: bool = False              # restore this cwd's saved conversation (CODER-2)
    custom_instructions: str | None = None   # extra system-prompt guidance (rec#584)


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
    from localm.inference.http_server import _request_token
    if not any_key_configured():
        return True, None                       # open mode = loopback owner
    # The browser GUI authenticates with the HttpOnly session cookie, not an
    # Authorization header, so resolve BOTH (header wins, else the cookie) - the
    # same source the main auth uses. Reading only the header made a cookie-authed
    # owner (the GUI) look like a non-owner, so it never saw its own coder sessions
    # (issue A1).
    token, _src = _request_token(request)
    held = verify(token) if token else None
    if held is not None and S.ADMIN in held:
        return True, None                       # the owner key
    import hashlib
    digest = hashlib.sha256((token or "").encode("utf-8")).hexdigest()
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
    # A key carrying the privileged coder:full scope (owner-only to mint) gets the
    # UNRESTRICTED coder, same as the owner; a plain coder-scoped key stays
    # restricted (read + confined edit, no shell).
    from localm import scopes as _S
    from localm.inference.http_server import caller_scopes as _caller_scopes
    held = _caller_scopes(request) or set()
    restricted = not (is_owner or _S.CODER_FULL in held)

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
        localm_server=True,   # self-connection: grammar sampling available
    )

    gen_kwargs = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_kwargs["max_tokens"] = req.max_tokens

    from localm.audit import effective_mode
    # Pass the session's project dir so a per-project .localcoder/config.toml mode
    # is honored by the GUI coder, not just the global coder_mode (REC-CODER-MODE-TOML).
    session_mode = req.mode or effective_mode("coder", cwd=cwd).value

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
        custom_instructions=req.custom_instructions,
        **gen_kwargs,
    ))
    session.principal = principal      # who owns this session (None = the owner)
    mgr.create(session)
    # Optional resume (CODER-2): restore this cwd's saved conversation into the
    # new session. Owner / coder:full only - a restricted scoped session must not
    # load the owner's prior conversation. The checkpoint read runs off the loop.
    resumed = False
    if req.resume and not restricted:
        resumed = await loop.run_in_executor(None, session.resume_from_checkpoint)
    return {**session.info(), "resumed": resumed}


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


@_router.get("/api/coder/sessions/{session_id}/files/download")
async def session_file_download(session_id: str, request: Request, path: str = ""):
    """Download one file the agent created/changed this session.

    For pulling coder output onto a phone (or any client). Restricted to files
    the session's change tracker recorded AND confined to the session root, so
    it is NOT an arbitrary-file-read primitive - an untracked path, an escaping
    path, or a since-deleted file is refused."""
    session = _get_session(request, session_id)
    if not path:
        raise HTTPException(400, "path is required")
    tracked = {f["path"] for f in session.changed_files()}
    if path not in tracked:
        raise HTTPException(404, f"'{path}' was not changed this session")
    try:
        root = Path(session.cwd).resolve()
        abs_path = (root / path).resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid path")
    if abs_path != root and root not in abs_path.parents:
        raise HTTPException(400, "Path escapes the session root")
    if not abs_path.is_file():
        raise HTTPException(404, f"'{path}' no longer exists on disk")
    return FileResponse(str(abs_path), filename=abs_path.name,
                        media_type="application/octet-stream")


@_router.post("/api/coder/sessions/{session_id}/stop")
async def session_stop(session_id: str, request: Request):
    session = _get_session(request, session_id)
    session.stop()
    return {"status": "stopping"}


@_router.delete("/api/coder/sessions/{session_id}")
async def session_delete(session_id: str, request: Request):
    _get_session(request, session_id)   # principal check: 404 unless it is the caller's
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
async def coder_history(request: Request):
    # Past session logs are the OWNER's audit trail (other coder sessions' commands
    # and file contents). They are not tagged per-key, so a scoped/shared key sees
    # none of them - only the owner browses history.
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        # "authorized": False lets the GUI say "sign in as the owner" instead of
        # mislabelling this as privacy mode - the two used to be indistinguishable
        # (issue A1).
        return {"enabled": False, "authorized": False, "logs": []}
    from localm import audit as _audit
    from localm.audit import SessionMode, effective_mode
    sessions_dir = _audit._SESSIONS_DIR
    items = []
    if sessions_dir.is_dir():
        # Coder sessions are the ONLY ones labelled "localcoder"
        # (AuditLog filename = <ts>_<pid>_<label>.jsonl). Regular GUI chat goes
        # through the HTTP server as "_server.jsonl" and the CLI chat REPL as
        # "_chat.jsonl", and all three share this directory - so glob the coder
        # label only, or coder history lists chat sessions too (coder-history-chat).
        for p in sorted(sessions_dir.glob("*_localcoder.jsonl"),
                        key=lambda f: f.stat().st_mtime, reverse=True)[:100]:
            items.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    return {"enabled": effective_mode("coder") != SessionMode.PRIVACY,
            "authorized": True, "logs": items}


@_router.get("/api/coder/history/{name}")
async def coder_history_entries(name: str, request: Request):
    # Reading a past log would expose the owner's coder transcript (commands, file
    # contents); a scoped/shared key may not (the logs are not per-key tagged).
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        raise HTTPException(404, "No such log")
    from localm import audit as _audit
    # Only coder logs are listed by coder_history; reading a chat/server log
    # ("_server.jsonl"/"_chat.jsonl") through the coder endpoint makes no sense
    # (coder-history-chat). Path traversal is still blocked by _confined_file.
    if not name.endswith("_localcoder.jsonl"):
        raise HTTPException(400, "Invalid log name")
    path = _confined_file(_audit._SESSIONS_DIR, name, "session log")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"path": str(path), "entries": entries}


@_router.get("/api/coder/resumable")
async def coder_resumable(request: Request, cwd: str = ""):
    """Is there a saved conversation to resume for *cwd*? (CODER-2)

    Owner-only: resuming restores the OWNER's prior conversation, so a scoped /
    shared key is never told one exists (and create_session also refuses resume
    for a restricted session). Returns ``{"resumable": false}`` when there is
    nothing to resume or the caller is not the owner."""
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        return {"resumable": False}
    if not cwd.strip():
        raise HTTPException(400, "cwd is required")
    try:
        p = Path(cwd).expanduser()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid cwd")
    if not p.is_dir():
        return {"resumable": False}
    from localm.plugins.coder.agent import checkpoint_info
    info = checkpoint_info(p.resolve())
    if not info:
        return {"resumable": False}
    return {"resumable": True, "cwd": str(p.resolve()), **info}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
