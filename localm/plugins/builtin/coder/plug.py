# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder plugin: the offline AI coding agent on the GUI surface.

Routes (mounted by the engine, auto-scoped to the ``coder`` capability):
  GET    /api/coder/sessions                    - list live sessions + active model
  POST   /api/coder/sessions                    - start a session (optional model switch)
  GET    /api/coder/sessions/{id}/events        - SSE event stream (?replay=true)
  POST   /api/coder/sessions/{id}/message       - send a task / steering message
  POST   /api/coder/sessions/{id}/estimate      - plan a task without running it
  POST   /api/coder/sessions/{id}/confirm       - answer a pending confirmation
  POST   /api/coder/sessions/{id}/undo          - revert the last file write
  POST   /api/coder/sessions/{id}/compact       - summarise old history
  POST   /api/coder/sessions/{id}/model         - repoint this session's model
  POST   /api/coder/sessions/{id}/settings      - approve / scope / verify, live
  POST   /api/coder/sessions/{id}/cwd           - move the session to another dir
  GET    /api/coder/sessions/{id}/memory        - the project-memory file
  POST   /api/coder/sessions/{id}/memory        - append a memory bullet
  POST   /api/coder/sessions/{id}/memory/forget - drop matching bullets
  GET    /api/coder/sessions/{id}/background    - this session's background jobs
  GET    /api/coder/sessions/{id}/log           - parsed JSONL audit log (log/full)
  GET    /api/coder/sessions/{id}/result        - last finished task, as JSON
  GET    /api/coder/sessions/{id}/files         - files changed this session
  GET    /api/coder/sessions/{id}/files/diff    - cumulative session diff
  GET    /api/coder/sessions/{id}/patch         - patch-mode diff (non-consuming)
  GET    /api/coder/sessions/{id}/patch/download- the same diff as a .patch file
  POST   /api/coder/sessions/{id}/stop          - interrupt the current task
  DELETE /api/coder/sessions/{id}               - terminate the session
  GET    /api/coder/history                     - browse past session audit logs
  GET    /api/coder/history/{name}              - read one past audit log
  GET    /api/coder/episodes                    - stored lessons for a project

The REPL's own session controls map onto these routes: /approve, /scope and
/verify are the settings route, /cd is the cwd route, /remember + /forget +
/memory are the three memory routes, and /bg is the background route.

Six CLI options have a web form here: --estimate (the estimate route),
--patch-mode (the patch_mode field + the two patch routes), --native-tools (the
native_tools field, with the effective value reported back), --output-format
json (the result route), --episodes (the episodes route), and --until (unified
onto the verify/auto_verify oracle, whose retry cap is verify_max_retries).

The agentic coder runs shell commands and writes files, so the engine gates every
route above on the ``coder`` capability scope. Live sessions need the kernel GUI's
shared services (``request.app.state.coder_sessions`` / ``.switch_model`` /
``.self_url`` / ``.active_model``), published by ``attach_gui``; the routes degrade
to 503 when those are absent (headless / no GUI). The read-only history endpoints
read ``<data dir>/sessions/*.jsonl`` directly and work without the GUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from localm.pathsafe import confined_file as _confined_file
from localm.pathsafe import is_unc_or_device_path as _is_unc_or_device_path
from localm.plugins.coder.sessions import CoderSession, SessionUnavailable
from localm.executor import get_plugin_executor

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
    # Auto-approve file writes but STILL prompt before shell execution. Only
    # meaningful with auto_approve, which is what it carves an exception out of.
    interactive_confirm: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # Pins the sampler's RNG, so the same seed, model, prompt and settings reproduce
    # the same output. Forwarded as a plain generation kwarg.
    seed: int | None = None
    resume: bool = False              # restore this cwd's saved conversation
    # WHICH past conversation to restore, when several are saved for this cwd. None
    # plus resume restores the most recent; an id continues that particular session.
    resume_checkpoint_id: str | None = None
    custom_instructions: str | None = None   # extra system-prompt guidance
    # Exit-code oracle: a command the harness runs before a turn that changed files
    # may finish. None plus auto_verify uses the project's detected check. Ignored
    # for a restricted (scoped-key) session, which has no execution.
    verify: str | None = None
    auto_verify: bool = True
    # How many fix attempts that oracle gets before it reports failure, bounded to
    # 1-50 so a request cannot pin the shared engine on an unbounded retry loop.
    # None keeps the Agent's own default.
    verify_max_retries: int | None = Field(None, ge=1, le=50)
    # Capture every file write as a unified diff and touch nothing on disk. The
    # accumulated patch is read back from GET .../patch and saved with
    # .../patch/download.
    patch_mode: bool = False
    # Ask for the OpenAI-compatible native tools protocol. Wired straight to the
    # backend; whether the connected server honours it is reported back by
    # create_session.
    native_tools: bool = False


class MessageRequest(BaseModel):
    text: str


class EstimateRequest(BaseModel):
    text: str


class EpisodeTargetRequest(BaseModel):
    """Which project's lessons an episode WRITE operation applies to.

    A body rather than a query parameter, unlike the two read routes: these are
    state-changing, so they use unsafe methods, which is what the CSRF check
    applies to.
    """
    cwd: str


class SetModelRequest(BaseModel):
    model: str


class SessionSettingsRequest(BaseModel):
    """Live changes to a running session (the REPL's /approve, /scope, /verify).

    Every field is optional, and ABSENT is not the same as NULL: a field the
    caller did not send is left alone, a field sent as null is CLEARED. That
    distinction is read off ``model_fields_set``, so one PATCH-shaped call can
    say "turn scope off" without restating the verify command.
    """
    auto_approve: bool | None = None
    scope: str | None = None
    # A command to run, or null for no exit-code check at all. Mutually exclusive
    # with auto_verify below: sending both is refused rather than resolved by an
    # ordering nobody can see.
    verify: str | None = None
    # True re-detects the project's own check (the REPL's `/verify auto`).
    auto_verify: bool | None = None


class SessionCwdRequest(BaseModel):
    cwd: str


class MemoryRequest(BaseModel):
    text: str


class MemoryForgetRequest(BaseModel):
    """Which bullets to drop. A body rather than a query parameter for the same
    reason as EpisodeTargetRequest: this destroys user content, so it must be an
    unsafe method, and an unsafe method is what the CSRF check applies to."""
    pattern: str


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
    from localm.auth import any_key_configured
    from localm.inference.http_server import caller_scopes, principal_id
    if not any_key_configured():
        return True, None                       # open mode = loopback owner
    # The browser GUI authenticates with the HttpOnly session cookie (an opaque
    # session id), not an Authorization header. caller_scopes/principal_id translate
    # a session id to its scope snapshot and owning-key hash, so a cookie session and
    # the same key as a bearer map to one principal; a raw verify() fails on a sid.
    held = caller_scopes(request)
    if held is not None and S.ADMIN in held:
        return True, None                       # the owner key / owner session
    return False, principal_id(request)


def _get_session(request: Request, session_id: str):
    session = _sessions(request).get(session_id)
    if session is None:
        raise HTTPException(404, f"No such session: {session_id}")
    is_owner, principal = _principal_from_request(request)
    # Isolation: a scoped caller may only touch the sessions it created, not the
    # owner's full-capability sessions (which keep run_shell). 404 rather than 403,
    # so a scoped key cannot probe which session ids exist.
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
    # gets the full coder: any directory, every tool. A minted, non-owner
    # coder-scoped key gets a RESTRICTED session: read plus confined-edit tools only
    # (no run_shell/run_tests/git-hooks/network/sub-agents), forced into the project
    # root, and isolated to its own sessions.
    is_owner, principal = _principal_from_request(request)
    # A key carrying the privileged coder:full scope (owner-only to mint) gets the
    # unrestricted coder, same as the owner; a plain coder-scoped key stays
    # restricted (read plus confined edit, no shell).
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
        # req.cwd is client-supplied: refuse UNC/device syntax unconditionally,
        # BEFORE cwd.is_dir()/.resolve() below ever run. Checked on the EXPANDED
        # string, so a value whose configured home directory is itself a UNC path
        # cannot expand into a UNC string past the check; expanduser() is pure
        # string/env-var work with no syscall. The restricted branch above ignores
        # req.cwd entirely and uses root_dir, so it needs no guard of its own.
        cwd = Path(req.cwd).expanduser()
        if _is_unc_or_device_path(str(cwd)):
            raise HTTPException(
                400, "'cwd' must be a local directory path, not a UNC or device path.")
        if not cwd.is_dir():
            raise HTTPException(400, f"Not a directory: {req.cwd}")
        cwd = cwd.resolve()

    # A per-session model switch changes the one shared engine for everyone, so a
    # scoped key must not trigger it.
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
        native_tools=req.native_tools,
    )
    # The request is wired to the backend for real; whether the SERVER honours it is
    # reported back to the caller. A localm self-connection does not implement the
    # OpenAI tools API (its ChatRequest declares no tools/tool_choice, so the fields
    # are dropped), and the session runs on localm's own grammar-constrained
    # tool-call convention instead.
    notes: list[str] = []
    if req.native_tools and not backend.supports_native_tools:
        notes.append(
            "native_tools was not applied: this server does not implement the "
            "OpenAI tools API. The session uses localm's own tool-call "
            "convention (grammar-constrained where the loaded model supports "
            "it).")

    gen_kwargs = {}
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_kwargs["max_tokens"] = req.max_tokens
    if req.seed is not None:
        gen_kwargs["seed"] = req.seed

    # Omitted leaves the Agent's own default as the single source of truth, matching
    # how temperature/max_tokens above are handled.
    verify_kwargs = {}
    if req.verify_max_retries is not None:
        verify_kwargs["verify_max_retries"] = req.verify_max_retries

    from localm.audit import effective_mode
    # Pass the session's project dir so a per-project .localcoder/config.toml mode is
    # honored by the GUI coder, not just the global coder_mode.
    session_mode = req.mode or effective_mode("coder", cwd=cwd).value

    loop = asyncio.get_running_loop()
    # Agent construction scans the project (map build) - keep it off the loop
    session = await loop.run_in_executor(get_plugin_executor(), lambda: CoderSession(
        cwd,
        backend,
        auto_approve=req.auto_approve,
        max_turns=req.max_turns,
        mode=session_mode,
        scope=req.scope,
        dry_run=req.dry_run,
        interactive_confirm=req.interactive_confirm,
        patch_mode=req.patch_mode,
        restricted=restricted,
        custom_instructions=req.custom_instructions,
        verify=req.verify,
        auto_verify=req.auto_verify,
        **verify_kwargs,
        **gen_kwargs,
    ))
    session.principal = principal      # who owns this session (None = the owner)
    mgr.create(session)
    # Optional resume: restore this cwd's saved conversation into the new session.
    # Owner / coder:full only. The checkpoint read runs off the loop.
    resumed = False
    if req.resume and not restricted:
        resumed = await loop.run_in_executor(
            get_plugin_executor(), session.resume_from_checkpoint,
            req.resume_checkpoint_id)
    return {**session.info(), "resumed": resumed, "notes": notes}


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
                    get_plugin_executor(), session.events.get, True, _KEEPALIVE_S)
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
    compacted = await loop.run_in_executor(get_plugin_executor(), session.compact)
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


@_router.post("/api/coder/sessions/{session_id}/estimate")
async def session_estimate(session_id: str, req: EstimateRequest, request: Request):
    """Plan a task without running it: the web form of the CLI's --estimate.

    One planning turn, zero tool calls, and NOTHING added to the conversation
    (see ``coder.estimate.estimate_task``). Refused while the session is busy.

    The busy/closed decision is the SESSION's, taken under its own lock in
    ``run_estimate``, not a check made here.

    The plan is pushed into the session feed as well as returned, so every open
    tab sees it - the same contract every other session event has."""
    session = _get_session(request, session_id)
    if not req.text.strip():
        raise HTTPException(400, "Empty task")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            get_plugin_executor(), session.run_estimate, req.text)
    except SessionUnavailable as e:
        # The session refused the claim. A dedicated type, never a broad
        # RuntimeError, so a model that raises RuntimeError is not reported as a busy
        # session.
        raise HTTPException(
            409, "Session is closed" if e.reason == "closed"
            else "Session is busy - estimate before starting a task")
    except Exception as e:                                     # noqa: BLE001
        raise HTTPException(502, f"Estimate failed: {type(e).__name__}: {e}")
    return result


@_router.get("/api/coder/sessions/{session_id}/result")
async def session_result(session_id: str, request: Request):
    """The last finished task's machine-readable result.

    The web equivalent of the CLI's ``--output-format json``: the same
    ``{ok, response, turns, total_tokens}`` payload (plus this surface's
    ``verify_state`` and ``changed_files``), readable by a client that is not
    holding the SSE stream open. 404 while no task has finished yet - an empty
    body would be indistinguishable from a task that produced nothing."""
    session = _get_session(request, session_id)
    if session.last_result is None:
        raise HTTPException(404, "No finished task in this session yet")
    return session.last_result


@_router.get("/api/coder/sessions/{session_id}/patch")
async def session_patch(session_id: str, request: Request):
    """The unified diff a patch-mode session has captured instead of writing.

    Reading NEVER consumes it (``session.current_patch()``, not
    ``agent.flush_patch()``): a reloaded tab, a retry or a second reader sees
    the same patch. 409 when the session is not in patch mode, where an empty
    diff would mean "everything was written to disk"."""
    session = _get_session(request, session_id)
    if not session.patch_mode:
        raise HTTPException(
            409, "This session is not in patch mode - its writes went to disk. "
                 "Start a session with patch_mode=true to capture them instead.")
    loop = asyncio.get_running_loop()
    patch = await loop.run_in_executor(get_plugin_executor(), session.current_patch)
    return {"patch": patch, "empty": not patch}


@_router.get("/api/coder/sessions/{session_id}/patch/download")
async def session_patch_download(session_id: str, request: Request):
    """The same patch as a .patch attachment - the web form of the CLI's
    ``--patch-mode FILE``, where the file lands on the CLIENT's disk.

    Streamed from memory, never through a server-side temp file."""
    session = _get_session(request, session_id)
    if not session.patch_mode:
        raise HTTPException(
            409, "This session is not in patch mode - its writes went to disk.")
    loop = asyncio.get_running_loop()
    patch = await loop.run_in_executor(get_plugin_executor(), session.current_patch)
    if not patch:
        raise HTTPException(404, "Nothing captured yet - the agent has not written "
                                 "anything in this session")
    return Response(
        content=patch,
        media_type="text/x-patch",
        headers={"Content-Disposition":
                 f'attachment; filename="coder-session-{session.id}.patch"'},
    )


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
        get_plugin_executor(), session.session_diff, path or None)
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


@_router.post("/api/coder/sessions/{session_id}/model")
async def session_set_model(session_id: str, req: SetModelRequest, request: Request):
    """Repoint an existing session's pinned model.

    Same trust model as create_session's optional model switch above: a
    per-session model change repoints the ONE shared engine for EVERYONE, so a
    scoped key must not trigger it. Mirrored here rather than factored out,
    since the "did the caller actually ask for a change" gate differs (there it
    is optional and compared against active_model(); here it is the whole point
    of the call)."""
    session = _get_session(request, session_id)
    if session.busy:
        # Fail fast, before the expensive, globally-visible switch_model call below.
        # session.set_model()'s own lock-protected busy check is the authoritative
        # gate against racing a message.
        raise HTTPException(409, "Session is busy; cannot switch models mid-task")

    active_model = request.app.state.active_model
    is_owner, _ = _principal_from_request(request)
    from localm import scopes as _S
    from localm.inference.http_server import caller_scopes as _caller_scopes
    held = _caller_scopes(request) or set()
    restricted = not (is_owner or _S.CODER_FULL in held)

    if req.model != active_model():
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
            res = await switch_model(req.model)
        except Exception as e:
            raise HTTPException(500, f"Failed to load {req.model}: {e}")
        # switch_model preempts an in-flight load of a DIFFERENT model (single-slot,
        # http_server.switch_engine's preempt=True default), so a newer switch
        # elsewhere can abandon this one mid-load and reports that by returning
        # {"status": "superseded"} rather than raising.
        if isinstance(res, dict) and res.get("status") == "superseded":
            raise HTTPException(
                503, f"Model load was superseded by a newer request: {res.get('by')}")

    if not session.set_model(req.model):
        raise HTTPException(409, "Session is busy; cannot switch models mid-task")
    return session.info()


@_router.post("/api/coder/sessions/{session_id}/settings")
async def session_settings(session_id: str, req: SessionSettingsRequest,
                           request: Request):
    """Change auto-approve, scope or verification on a LIVE session.

    NOT BUSY-GUARDED, unlike the model route: revoking auto-approve and
    tightening a scope must work while the agent is mid-run. These are
    attribute writes read at the next tool dispatch, not state a running turn
    can tear.
    """
    session = _get_session(request, session_id)
    if session.closed:
        raise HTTPException(409, "Session is closed")
    sent = req.model_fields_set
    if "verify" in sent and req.auto_verify:
        raise HTTPException(
            400, "Send either verify (a specific command) or auto_verify "
                 "(re-detect the project's check), not both.")

    changed: list[str] = []
    if "auto_approve" in sent and req.auto_approve is not None:
        session.set_auto_approve(req.auto_approve)
        changed.append("auto_approve")
    if "scope" in sent:
        session.set_scope(req.scope)
        changed.append("scope")
    if "verify" in sent or req.auto_verify:
        try:
            session.set_verify(req.verify, detect=bool(req.auto_verify))
        except SessionUnavailable as e:
            raise HTTPException(409, str(e))
        changed.append("verify")
    # info() reports the EFFECTIVE state, so the caller reads back what actually took
    # rather than an echo of what it asked for.
    return {**session.info(), "changed": changed}


@_router.post("/api/coder/sessions/{session_id}/cwd")
async def session_set_cwd(session_id: str, req: SessionCwdRequest,
                          request: Request):
    """Move a live session to another project directory (the REPL's /cd).

    REFUSED for a restricted (scoped-key) session: create_session forces such a
    session into the instance's project root and ignores the cwd it was given.
    Refused on the SESSION's restriction rather than the caller's, so an owner
    cannot move a shared key's session out of the root either.

    Busy-guarded where /settings is not: this rebuilds the project map and the
    system prompt, so applying it under a running turn would change the prompt
    out from under a request already in flight.
    """
    session = _get_session(request, session_id)
    if session.closed:
        raise HTTPException(409, "Session is closed")
    if session.restricted:
        raise HTTPException(
            403, "A shared-key session is confined to the project root and "
                 "cannot change directory.")
    # Client-supplied path: refuse UNC/device syntax unconditionally, on the EXPANDED
    # string, BEFORE is_dir()/resolve() ever run.
    cwd = Path(req.cwd).expanduser()
    if _is_unc_or_device_path(str(cwd)):
        raise HTTPException(
            400, "'cwd' must be a local directory path, not a UNC or device path.")
    if not cwd.is_dir():
        raise HTTPException(400, f"Not a directory: {req.cwd}")
    cwd = cwd.resolve()

    # A project that declared itself private does not get a transcript.
    from localm.plugins.coder.privacy import refuse_move_into_stricter_project
    refusal = refuse_move_into_stricter_project(session.mode, cwd)
    if refusal:
        raise HTTPException(409, refusal)

    # Rebuilding the project map scans the tree, so it runs in the executor.
    loop = asyncio.get_running_loop()
    moved = await loop.run_in_executor(
        get_plugin_executor(), session.set_cwd, cwd)
    if not moved:
        raise HTTPException(409, "Session is busy; cannot change directory "
                                 "mid-task")
    return session.info()


@_router.get("/api/coder/sessions/{session_id}/memory")
async def session_memory(session_id: str, request: Request):
    """The project-memory file (LOCALCODER.md) this session injects.

    NOT owner-gated, unlike history and episodes: those are the owner's own
    records held in the localm home directory, while this is a file in the
    session's own project directory that a restricted session's confined
    write_file/edit_file can already read and rewrite (SAFE_RESTRICTED_TOOLS).
    """
    session = _get_session(request, session_id)
    return session.memory()


@_router.post("/api/coder/sessions/{session_id}/memory")
async def session_remember(session_id: str, req: MemoryRequest, request: Request):
    """Append a bullet to the project memory (the REPL's /remember).

    Goes through the agent, so the system prompt is rebuilt immediately. An
    edit made by asking the agent to rewrite the file does not call
    reload_memory and only takes effect next session.
    """
    session = _get_session(request, session_id)
    if session.closed:
        raise HTTPException(409, "Session is closed")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    try:
        return session.remember(text)
    except OSError as e:
        raise HTTPException(500, f"Could not write the memory file: {e}")


@_router.post("/api/coder/sessions/{session_id}/memory/forget")
async def session_forget(session_id: str, req: MemoryForgetRequest,
                         request: Request):
    """Drop memory bullets matching a substring (the REPL's /forget).

    ``removed`` and ``had_file`` come back alongside the new memory so a caller
    can tell "there is no memory file" from "no entry matched"; both leave the
    memory unchanged.
    """
    session = _get_session(request, session_id)
    if session.closed:
        raise HTTPException(409, "Session is closed")
    pattern = req.pattern.strip()
    if not pattern:
        raise HTTPException(400, "pattern is required")
    try:
        return session.forget(pattern)
    except OSError as e:
        raise HTTPException(500, f"Could not rewrite the memory file: {e}")


@_router.get("/api/coder/sessions/{session_id}/background")
async def session_background(session_id: str, request: Request):
    """Background jobs THIS session started (the REPL's /bg).

    An owner GUI session starts background shell jobs and sub-agents through
    run_shell_background / spawn_agent_background.

    Scoped to this session's own jobs, never the whole registry: get_registry()
    is process-wide, and a GUI server runs many sessions in one process, so an
    unfiltered list would show one session another's work - and job labels are
    full command lines. ``supported`` is reported separately because a
    restricted session has no background tools at all, so "none yet" must not
    look the same as "never".
    """
    session = _get_session(request, session_id)
    return {**session.background(), "supported": not session.restricted}


@_router.delete("/api/coder/sessions/{session_id}")
async def session_delete(session_id: str, request: Request):
    _get_session(request, session_id)   # principal check: 404 unless it is the caller's
    # remove() -> CoderSession.close() -> agent.close() is BLOCKING work: a
    # checkpoint write, the audit close, and, for a session that ran run_shell,
    # _detect_shell_changes(), which shells out to `git status --porcelain`
    # (timeout 10) plus up to two `git diff`s (timeout 15 each). It runs off the
    # loop, bounded a little above that ~40s worst case.
    #
    # SessionManager.remove() pops session_id from its dict BEFORE close() runs, so
    # an abandoned close() cannot race a second DELETE of the same id: the retry
    # sees the id gone and 404s instead of double-closing. The checkpoint file
    # close() writes is keyed to this session's own unique id.
    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded,
    )
    try:
        removed = await run_in_threadpool_bounded(
            _sessions(request).remove, session_id, timeout=60.0)
    except ThreadCallTimeout as e:
        raise HTTPException(504, f"Closing the session timed out: {e}")
    if removed is None:
        raise HTTPException(404, f"No such session: {session_id}")
    return {"status": "closed"}


# ------------------------------------------------------------------ #
#  Session history (read-only; works without the GUI services)        #
# ------------------------------------------------------------------ #
# Browser for past audit logs (<data dir>/sessions/*.jsonl, written in log/full
# modes). Live sessions have /log; this lists what earlier sessions left behind,
# including ones from before a server restart. Privacy mode writes no logs, so the
# list is empty.

@_router.get("/api/coder/history")
async def coder_history(request: Request):
    # Past session logs are the OWNER's audit trail (other coder sessions' commands
    # and file contents) and are not tagged per-key, so a scoped/shared key sees none
    # of them.
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        # "authorized": False lets the GUI say "sign in as the owner" instead of
        # mislabelling this as privacy mode.
        return {"enabled": False, "authorized": False, "logs": []}
    from localm import audit as _audit
    from localm.audit import SessionMode, effective_mode
    sessions_dir = _audit._SESSIONS_DIR
    items = []
    if sessions_dir.is_dir():
        # Coder sessions are the ONLY ones labelled "localcoder" (AuditLog filename =
        # <ts>_<pid>_<label>.jsonl). GUI chat writes "_server.jsonl" and the CLI chat
        # REPL "_chat.jsonl" into the same directory, so glob the coder label only.
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
    # contents); a scoped/shared key may not, as the logs are not per-key tagged.
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        raise HTTPException(404, "No such log")
    from localm import audit as _audit
    # Only coder logs are listed by coder_history, so a chat/server log
    # ("_server.jsonl"/"_chat.jsonl") is refused here. Path traversal is still blocked
    # by _confined_file.
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
    """Is there a saved conversation to resume for *cwd*?

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
    # `cwd` is a raw HTTP query parameter, and this is a GET route with no CSRF check
    # on it (CSRF only applies to unsafe methods): refuse UNC/device syntax
    # unconditionally, on the EXPANDED string, BEFORE p.is_dir()/.resolve() run.
    if _is_unc_or_device_path(str(p)):
        raise HTTPException(
            400, "'cwd' must be a local directory path, not a UNC or device path.")
    if not p.is_dir():
        return {"resumable": False}
    from localm.plugins.coder.agent import checkpoint_info
    info = checkpoint_info(p.resolve())
    if not info:
        return {"resumable": False}
    return {"resumable": True, "cwd": str(p.resolve()), **info}


# A permanent statement, not an empty-state message: the list below is incomplete by
# construction for anyone who uses privacy mode.
_PRIVACY_NOTE = "Privacy-mode sessions are never recorded and never appear here."


def _dormant_for(path_str: str) -> list:
    """Past conversations saved for one project, newest first.

    Checkpoints are keyed on a digest of the project path and live under the
    data dir, NOT inside the project, so this still answers for a project
    directory that has been moved or deleted.

    Privacy-mode sessions cannot appear: that mode writes no checkpoint at all
    (see Agent.save_checkpoint), so nothing here filters them out.
    """
    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    try:
        return list_checkpoints(Path(path_str))
    except Exception:
        # One unreadable project must not blank the whole listing; the rest of the
        # response is still returned.
        return []


@_router.get("/api/coder/dormant")
async def coder_dormant(request: Request, cwd: str = ""):
    """Past coder conversations, grouped by project, resumable by id.

    Owner-only for the same reason /api/coder/resumable is: a session title is
    the user's own words about their own work, so a scoped or shared key is
    never shown one.

    *cwd* is optional and only decides which group is marked ``current``; the
    listing spans every remembered project either way.
    """
    is_owner, _ = _principal_from_request(request)
    if not is_owner:
        return {"projects": [], "privacy_note": _PRIVACY_NOTE}

    current = ""
    if cwd.strip():
        try:
            p = Path(cwd).expanduser()
        except (OSError, ValueError, RuntimeError):
            raise HTTPException(400, "Invalid cwd")
        # Same guard, in the same order, as the resumable probe above: refuse
        # UNC/device syntax on the EXPANDED string before any filesystem call.
        if _is_unc_or_device_path(str(p)):
            raise HTTPException(
                400, "'cwd' must be a local directory path, not a UNC or device path.")
        try:
            current = str(p.resolve())
        except (OSError, ValueError, RuntimeError):
            raise HTTPException(400, "Invalid cwd")

    def _collect() -> list:
        from localm.plugins.coder.projects import list_projects
        rows, seen = [], set()
        if current:
            rows.append({"path": current, "name": Path(current).name or current,
                         "available": Path(current).is_dir(), "current": True,
                         "sessions": _dormant_for(current)})
            seen.add(os.path.normcase(current))
        for entry in list_projects():
            path = str(entry.get("path") or "")
            if not path or os.path.normcase(path) in seen:
                continue
            seen.add(os.path.normcase(path))
            rows.append({"path": path,
                         "name": entry.get("name") or path,
                         "available": bool(entry.get("available")),
                         "current": False,
                         "sessions": _dormant_for(path)})
        return rows

    # Off the event loop: this globs and parses a JSON file per checkpoint per
    # project, which is unbounded filesystem work.
    loop = asyncio.get_running_loop()
    projects = await loop.run_in_executor(get_plugin_executor(), _collect)
    return {"projects": projects, "privacy_note": _PRIVACY_NOTE}


# ------------------------------------------------------------------ #
#  Episodic memory (the CLI's --episodes family)                      #
# ------------------------------------------------------------------ #

def _is_owner(request: Request) -> bool:
    """Owner-only is the gate for EVERY episode operation, read or write.

    A restricted (scoped-key) session is excluded from episodic memory entirely:
    it neither recalls a lesson nor writes one.
    """
    is_owner, _ = _principal_from_request(request)
    return is_owner


def _episode_root(cwd: str) -> Path:
    """Validate a caller-supplied project path and return the key to file under.

    ONE helper shared by all five episode routes.

    NO is_dir() check: lessons live under the localm data dir keyed by the
    RESOLVED project path, so a directory that has been moved or removed still
    has an entry.

    resolve() is the ONLY filesystem touch on a client-supplied string, and it
    is required: the CLI derives the same key by resolving the same way. UNC and
    device syntax is refused LEXICALLY, before that call - a GET carries no CSRF
    check, and a UNC string reaching the filesystem is an SMB dial.
    """
    if not cwd.strip():
        raise HTTPException(400, "cwd is required")
    try:
        p = Path(cwd).expanduser()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(400, "Invalid cwd")
    if _is_unc_or_device_path(str(p)):
        raise HTTPException(
            400, "'cwd' must be a local directory path, not a UNC or device path.")
    return p.resolve()


async def _episode_op(fn):
    """Run a store operation off the event loop, turning a failure into a 5xx
    that NAMES it rather than a clean-looking empty result. Every episode route
    goes through this so no operation can quietly half-happen."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(get_plugin_executor(), fn)
    except HTTPException:
        raise
    except Exception as e:                                     # noqa: BLE001
        raise HTTPException(
            500, f"Episode store operation failed: {type(e).__name__}: {e}")


@_router.get("/api/coder/episodes")
async def coder_episodes(request: Request, cwd: str = ""):
    """The episodic-memory lessons stored for *cwd*: the CLI's ``--episodes``.

    OWNER-ONLY, like ``/api/coder/resumable``: a restricted (scoped-key) session
    is excluded from episodic memory entirely - it neither recalls a lesson nor
    writes one. A non-owner gets an empty list rather than a 403.

    Read-only. Forgetting, restoring and consolidating a lesson are separate CLI
    flags with their own destructive semantics and are not exposed here.

    Lessons live under the localm data dir, never inside the project, so the
    ``cwd`` here is only the KEY they are filed under; nothing in the project
    tree is read, and a directory that no longer exists still has its lessons."""
    if not _is_owner(request):
        return {"episodes": [], "cwd": None}
    root = _episode_root(cwd)

    def _read():
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(root)
        return [
            {"id": e.id, "outcome": e.outcome, "lesson": e.lesson,
             "summary": e.summary, "task": e.task, "turns": e.turns,
             "ts": e.ts, "merged": e.merged, "files": list(e.files)}
            for e in store.all()
        ]

    loop = asyncio.get_running_loop()
    try:
        episodes = await loop.run_in_executor(get_plugin_executor(), _read)
    except Exception as e:                                     # noqa: BLE001
        # An unreadable store is NOT an empty one, so it is reported as an error
        # rather than as an empty list.
        raise HTTPException(
            500, f"Could not read the episode store for {root}: "
                 f"{type(e).__name__}: {e}")
    return {"episodes": episodes, "cwd": str(root)}


@_router.get("/api/coder/episodes/archive")
async def coder_episodes_archive(request: Request, cwd: str = ""):
    """Lessons this project has DROPPED and can get back: ``--episodes-archive``.

    An unreadable archive is NOT an empty one: it answers 503 rather than a 200
    with an empty list. The CLI reports the same condition by exiting non-zero.
    """
    if not _is_owner(request):
        return {"archived": [], "cwd": None}
    root = _episode_root(cwd)

    def _read():
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(root)
        rows = store.forgotten()
        if not rows and not store.last_forgotten_ok:
            raise HTTPException(
                503, "The episode archive exists but could not be read, so this "
                     "list would be INCOMPLETE. It may be locked by another "
                     "process - try again.")
        return rows

    rows = await _episode_op(_read)
    return {"archived": rows, "cwd": str(root)}


@_router.post("/api/coder/episodes/{episode_id}/forget")
async def coder_episode_forget(episode_id: str, request: Request,
                               req: EpisodeTargetRequest):
    """Drop ONE lesson from recall: ``--forget-episode``.

    Reversible: the record is archived first. If the archiving half failed, the
    lesson is still gone from recall, and that is reported as a caveat on the
    outcome rather than swallowed.
    """
    if not _is_owner(request):
        raise HTTPException(404, "No such episode")
    root = _episode_root(req.cwd)

    def _forget():
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(root)
        if not store.forget(episode_id):
            raise HTTPException(404, f"No episode with id {episode_id}")
        return store.last_archive_ok

    archived = await _episode_op(_forget)
    return {
        "forgotten": episode_id,
        "recoverable": archived,
        # Stated plainly rather than left to be inferred from the boolean: without
        # the archive copy, this forget is not undoable.
        "warning": None if archived else
        "The lesson was dropped from recall, but the archive could not be "
        "written - so this one cannot be restored.",
    }


@_router.post("/api/coder/episodes/{episode_id}/restore")
async def coder_episode_restore(episode_id: str, request: Request,
                                req: EpisodeTargetRequest):
    """Put an archived lesson back into recall: ``--restore-episode``.

    Carries the CLI's two caveats, both describing a restore that SUCCEEDED: the
    archive may not have been updated (so the lesson is live AND still listed as
    forgotten), and a store at its cap may have evicted it again immediately.
    """
    if not _is_owner(request):
        raise HTTPException(404, "No such episode")
    root = _episode_root(req.cwd)

    def _restore():
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(root)
        ep = store.restore(episode_id)
        if ep is None:
            if not store.last_forgotten_ok:
                # The archive EXISTS but could not be read, so "no such id" cannot be
                # claimed - the episode may be in there.
                raise HTTPException(
                    503, "The episode archive could not be read, so this id "
                         "could not be looked up. Nothing was changed - try "
                         "again.")
            raise HTTPException(404, f"No archived episode with id {episode_id}")
        return {
            "restored": ep.id,
            "lesson": ep.lesson or ep.summary,
            "archive_updated": store.last_restore_archive_ok,
            "evicted_again": any(e.id == ep.id for e in store.last_evicted),
        }

    out = await _episode_op(_restore)
    notes = []
    if not out["archive_updated"]:
        notes.append("The lesson is live again, but the archive could not be "
                     "updated, so it is also still listed as forgotten.")
    if out["evicted_again"]:
        notes.append("The store is at its episode cap and this lesson ranked "
                     "lowest, so it was dropped again immediately. Forget one "
                     "you no longer need first.")
    out["notes"] = notes
    return out


@_router.delete("/api/coder/episodes")
async def coder_episodes_clear(request: Request, req: EpisodeTargetRequest):
    """Erase ALL episodic memory for a project, archive included:
    ``--forget-episodes``.

    NOT reversible; the archive is erased too, so no lesson text survives in a
    sidecar. The counts are read BEFORE the erase so the response can say what
    was destroyed.
    """
    if not _is_owner(request):
        raise HTTPException(403, "Owner only")
    root = _episode_root(req.cwd)

    def _clear():
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(root)
        live = len(store.all())
        archived = len(store.forgotten())
        store.clear()
        # Read back rather than trusting the unlink: this is the one episode
        # operation with no undo, so success is confirmed against the store on disk.
        after = EpisodeStore(root)
        remaining = len(after.all()) + len(after.forgotten())
        if remaining:
            raise HTTPException(
                500, f"Erase did not fully complete: {remaining} record(s) "
                     "remain, so this is NOT reported as cleared.")
        return {"erased": live, "erased_archived": archived}

    return await _episode_op(_clear)


@_router.post("/api/coder/episodes/consolidate")
async def coder_episodes_consolidate(request: Request, req: EpisodeTargetRequest):
    """Ask the model to merge related lessons into one: ``--consolidate-episodes``.

    OPT-IN and manual only, never automatic. Every input is archived, so a merge
    is reversible with restore.

    Reports what it DID (groups, merged, replaced, archived, skipped), and a
    group whose merge came back unusable is counted as skipped and left alone.
    """
    if not _is_owner(request):
        raise HTTPException(403, "Owner only")
    root = _episode_root(req.cwd)
    self_url = getattr(request.app.state, "self_url", None)
    active_model = getattr(request.app.state, "active_model", None)
    if not self_url or active_model is None:
        raise HTTPException(503, "Consolidation needs the localm GUI server "
                                 "(run `localm gui`).")

    def _consolidate():
        from localm.plugins.coder.backends.http import HTTPBackend
        from localm.plugins.coder.episodes import EpisodeStore, consolidate
        from localm.textnorm import strip_think
        backend = HTTPBackend(
            self_url,
            model=active_model(),
            api_key=os.environ.get("LOCALM_API_KEY") or "localm",
            localm_server=True,
        )

        def _complete(prompt: str) -> str:
            return strip_think(
                backend.chat([{"role": "user", "content": prompt}],
                             max_tokens=1024) or "")

        return consolidate(EpisodeStore(root), complete=_complete)

    return await _episode_op(_consolidate)


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
