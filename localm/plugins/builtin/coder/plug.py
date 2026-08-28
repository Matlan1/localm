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

The REPL's own session controls have a web form here for the same reason
(parity audit O6): /approve, /scope and /verify are the settings route, /cd is
the cwd route, /remember + /forget + /memory are the three memory routes, /bg is
and /bg is the background route. Until now a GUI session could not revoke its
own auto-approve, and the workaround suggested for the others (start again with
resume) does not exist in privacy mode, which is the DEFAULT on both surfaces.

Six options that were CLI-only have a web form here, per the standing
CLI/GUI parity rule: --estimate (the estimate route), --patch-mode (the
patch_mode field + the two patch routes), --native-tools (the native_tools
field, with the effective value reported back), --output-format json (the result
route), --episodes (the episodes route), and --until (unified onto the existing
verify/auto_verify oracle, whose retry cap verify_max_retries now exposes).

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
    # The CLI's --interactive-confirm: auto-approve file writes but STILL prompt
    # before shell execution. Only meaningful with auto_approve, which is what it
    # carves an exception out of; on its own it changes nothing, because every
    # destructive tool already prompts.
    interactive_confirm: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # Pins the sampler's RNG so the same seed, model, prompt and settings
    # reproduce the same output (the CLI's --seed). A plain generation kwarg
    # alongside the two above, forwarded the same way.
    seed: int | None = None
    resume: bool = False              # restore this cwd's saved conversation
    # WHICH past conversation to restore, when several are saved for this cwd.
    # None + resume -> the most recent, unchanged. An id (from
    # /api/coder/dormant) continues that particular session instead.
    resume_checkpoint_id: str | None = None
    custom_instructions: str | None = None   # extra system-prompt guidance
    # Exit-code oracle: a command the HARNESS runs before a turn that changed
    # files may finish. None + auto_verify -> the project's detected check.
    # Ignored for a restricted (scoped-key) session, which has no execution.
    verify: str | None = None
    auto_verify: bool = True
    # How many fix attempts that oracle gets before it reports failure - the web
    # equivalent of the CLI's --goal-max-iters, and bounded the same way (1-50)
    # so a request cannot pin the shared engine on an unbounded retry loop. None
    # keeps the Agent's own default.
    verify_max_retries: int | None = Field(None, ge=1, le=50)
    # Capture every file write as a unified diff and touch nothing on disk (the
    # CLI's --patch-mode). The accumulated patch is read back from
    # GET .../patch and saved with .../patch/download - a browser has no output
    # FILE for the CLI's argument to name.
    patch_mode: bool = False
    # Ask for the OpenAI-compatible native tools protocol (the CLI's
    # --native-tools). Wired straight to the backend; whether the connected
    # server can honour it is reported back rather than assumed - see
    # create_session.
    native_tools: bool = False
    # WHICH model server answers this session, the web form of the CLI's
    # --online / --anthropic / --url. Per session, never global config: a
    # global default is how a later project's source reaches a cloud provider
    # without anyone deciding, and that failure is silent.
    #   local (default) - this localm. Offline, grammar-constrained tool calls.
    #   url             - any OpenAI-compatible endpoint. On this machine
    #                     (Ollama, LM Studio, vLLM) or off it; _resolve_backend
    #                     classifies which, because the privacy consequence
    #                     differs even though it is one field to the user.
    #   openai / anthropic - the provider APIs. Off-machine, and it costs money.
    backend: str = "local"
    backend_url: str | None = None      # required for backend="url"
    backend_model: str | None = None    # model id at the far end; blank = provider default
    # Per-session credential for an off-machine backend, held in memory for the
    # life of the session and NEVER written to disk (that is a later stage, and
    # it needs its own review as a new secret at rest). Blank falls back to
    # OPENAI_API_KEY / ANTHROPIC_API_KEY, which is how the CLI already resolves
    # them. Never echoed back by session.info().
    backend_api_key: str | None = None


class MessageRequest(BaseModel):
    text: str


class EstimateRequest(BaseModel):
    text: str


class EpisodeTargetRequest(BaseModel):
    """Which project's lessons an episode WRITE operation applies to.

    A body rather than a query parameter, unlike the two read routes: these are
    state-changing, so they must be unsafe methods, and an unsafe method is what
    the CSRF check applies to. A destructive operation reachable by a URL alone
    is one someone can be walked into.
    """
    cwd: str


class SetModelRequest(BaseModel):
    model: str


class SessionSettingsRequest(BaseModel):
    """Live changes to a running session (the REPL's /approve, /scope, /verify).

    Every field is optional, and ABSENT is not the same as NULL: a field the
    caller did not send is left alone, a field sent as null is CLEARED. That
    distinction is read off ``model_fields_set``, so one PATCH-shaped call can
    say "turn scope off" without also having to restate the verify command, and
    "leave scope as it is" without a magic sentinel string.
    """
    auto_approve: bool | None = None
    scope: str | None = None
    # A command to run, or null for no exit-code check at all. Mutually
    # exclusive with auto_verify below: sending both asks for a specific command
    # AND for re-detection, which cannot both be honoured, so it is refused
    # rather than resolved by an ordering nobody can see.
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
    # The browser GUI authenticates with the HttpOnly session cookie (now an opaque
    # session id), not an Authorization header. Resolve identity through the SAME
    # central helpers the main auth uses so a cookie session and the same key as a
    # bearer map to one principal: caller_scopes/principal_id translate a session id
    # to its scope snapshot and owning-key hash (a raw verify() would fail on a sid,
    # making a cookie-authed owner look like a non-owner).
    held = caller_scopes(request)
    if held is not None and S.ADMIN in held:
        return True, None                       # the owner key / owner session
    return False, principal_id(request)


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
#  Coder LLM backend selection                                       #
# ------------------------------------------------------------------ #
# A GUI coder session was hardwired to this localm; the terminal has had
# --online / --anthropic / --url since the beginning. This closes that parity
# gap. Everything below is PER SESSION.

_BACKEND_MODES = ("local", "url", "openai", "anthropic")

# Fixed provider bases. Constants rather than user input, so only "url" is a
# GUI-supplied destination - but all three still pass netpolicy below, because
# net_mode=off means the user disabled network access, and that must hold for a
# provider we picked just as much as for one they typed.
_PROVIDER_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}
_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _url_leaves_machine(url: str) -> bool:
    """True when *url* points somewhere other than this machine.

    Uses ``bindhost.is_loopback_host``, the canonical classifier, rather than a
    private set of literals. It differs from ``reviewer.py``'s
    ``_LOOPBACK_HOSTS`` in two places, both in the SAFE direction here:
    ``127.0.0.2`` (the whole 127.0.0.0/8 block) is correctly local, and
    ``0.0.0.0`` / an empty host are NOT treated as local. A wildcard bind
    address is not a destination and a missing host is malformed, so calling
    either "on this machine" would grant the quiet path to a string nobody
    validated. Unparseable input answers True, because the only safe answer to
    "might this leave the machine" is yes.
    """
    from urllib.parse import urlparse
    try:
        from localm.bindhost import is_loopback_host
        return not is_loopback_host((urlparse(url).hostname or "").lower())
    except Exception:
        return True


def _resolve_backend(req: "CreateSessionRequest", *, self_url: str,
                     model_name: str, restricted: bool, session_mode: str):
    """Build this session's LLM backend and describe it honestly.

    Returns ``(backend, descriptor, notes)``. The descriptor is what
    ``session.info()`` reports, so a UI can show WHILE THE SESSION RUNS that it
    is talking to a remote model rather than only at the moment it was picked -
    a hint shown once at selection does not meet "obvious while it is on".

    BLOCKING: netpolicy resolves DNS. Call this off the event loop.

    The descriptor carries no credential, and neither does ``info()``: a secret
    that round-trips to a client is a secret in a browser history, a proxy log
    and a screenshot.
    """
    mode = (req.backend or "local").strip().lower()
    if mode not in _BACKEND_MODES:
        raise HTTPException(
            400, f"Unknown backend {mode!r}. Choose one of: "
                 f"{', '.join(_BACKEND_MODES)}.")

    notes: list[str] = []

    if mode == "local":
        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend(
            self_url,
            model=model_name,
            api_key=os.environ.get("LOCALM_API_KEY") or "localm",
            localm_server=True,   # self-connection: grammar sampling available
            native_tools=req.native_tools,
        )
        return backend, {"backend": "local", "leaves_machine": False,
                         "target": "this localm", "model": model_name}, notes

    # ---- everything past here is a NON-default backend ------------------- #

    # Owner-only, matching coder_reviewer's admin_only=True. A minted, non-owner
    # key must not be able to point the coder anywhere: that is an exfil channel
    # for the project's source and a billing channel for someone else's account.
    if restricted:
        raise HTTPException(
            403, "Choosing a model server needs the owner key; a scoped key "
                 "uses this localm.")

    if mode == "url":
        base = (req.backend_url or "").strip()
        if not base:
            raise HTTPException(400, "backend='url' needs backend_url.")
        if not (base.startswith("http://") or base.startswith("https://")):
            raise HTTPException(
                400, "backend_url must start with http:// or https://.")
    else:
        base = _PROVIDER_BASE[mode]

    from localm.netpolicy import NetworkPolicyError, check_url, check_url_shape

    # SHAPE FIRST, ALWAYS, AND BEFORE THE CLASSIFICATION BELOW DEPENDS ON IT.
    # urlparse and the HTTP client disagree about backslashes and control
    # characters in the authority, so until that guard has run, "is this host on
    # this machine" is a question about a destination the client may not
    # actually dial. Running it first is what makes _url_leaves_machine's answer
    # safe to branch on.
    try:
        check_url_shape(base)
    except NetworkPolicyError as e:
        raise HTTPException(400, str(e))

    leaves = _url_leaves_machine(base)

    # DESTINATION policy applies only to a destination that actually leaves the
    # machine. This is NOT the whole of check_url for every URL: check_url's
    # public-address arm refuses loopback by default, so guarding every base URL
    # with it makes "point the coder at my own Ollama" - the single most likely
    # use of this field - fail out of the box, with a message telling the user to
    # set net_allow_private, a GLOBAL setting that
    # would weaken the SSRF guard for genuine web fetches too.
    #
    # The calibration matches what the rest of localm already does. netpolicy
    # guards URLs that arrive from UNTRUSTED CONTENT (a model-supplied media
    # URL, a redirect chain during a model pull). An OWNER-CONFIGURED service
    # endpoint does not go through it at all: comfy_api_url and the
    # coder_reviewer URL both point wherever the owner says. This field is the
    # same kind of thing, and it is already owner-gated above.
    #
    # An off-machine destination is different, and there check_url is exactly
    # right: net_mode=off then means what it says, the deny list applies, and a
    # LAN address is refused unless net_allow_private is set - whose name
    # describes precisely that choice, so the refusal is actionable rather than
    # a setting about something else.
    if leaves:
        try:
            check_url(base)
        except NetworkPolicyError as e:
            raise HTTPException(403, f"Network policy refused {base}: {e}")

    # Privacy mode REFUSES an off-machine model. Not a checkbox, not a fallback:
    # localm already answers this question this way for memory (fully off in
    # privacy mode) and for the coder reviewer (skipped in privacy mode), and a
    # third behaviour for the same question would be drift. The gate lives in a
    # leaf module so this call site does not re-derive it - see remotegate.
    #
    # Refusing rather than quietly using the local model is the load-bearing
    # half: substituting a different model than the one the user picked is the
    # silent-override failure, and a session that came back "created" while
    # ignoring the choice is indistinguishable from one that honoured it.
    from localm import remotegate
    if leaves and not remotegate.remote_allowed_for_mode(session_mode):
        raise HTTPException(403, remotegate.refusal_message("Off-machine models"))

    api_key = (req.backend_api_key or "").strip()
    env_var = _PROVIDER_ENV.get(mode)
    if not api_key and env_var:
        api_key = os.environ.get(env_var, "").strip()
    if not api_key and mode in ("openai", "anthropic"):
        # Refuse now, naming the cause, rather than building a session that 401s
        # on its first message - by which point the failure reads as the provider
        # being down rather than a key never having been supplied.
        raise HTTPException(
            400, f"{mode} needs an API key. Enter one for this session, or set "
                 f"{env_var} before starting localm.")

    from localm.plugins.coder.backends.http import HTTPBackend
    target_model = (req.backend_model or "").strip() or model_name
    backend = HTTPBackend(
        base,
        model=target_model,
        # A local OpenAI-compatible server (Ollama, LM Studio, llama.cpp)
        # usually ignores the bearer entirely; "localm" is the placeholder the
        # backend's own docstring names for that case.
        api_key=api_key or "localm",
        anthropic=(mode == "anthropic"),
        # localm_server stays FALSE for every mode here, including a URL that
        # happens to point at another localm. That costs grammar-constrained
        # tool calls in that one case, and it is the conservative choice the
        # backend already documents: an unknown third-party server can 400 on
        # grammar kwargs it does not understand.
        native_tools=req.native_tools,
    )
    if leaves:
        notes.append(
            "This session sends your prompts and the file contents it reads to "
            f"{base}. They leave this machine.")
    # Said out loud rather than left for the user to infer from a capability
    # they cannot see, the same way native_tools is.
    if not backend.supports_grammar:
        notes.append(
            "Grammar-constrained tool calls are off for this backend; the "
            "session uses the text tool-call convention instead.")
    return backend, {"backend": mode, "leaves_machine": leaves,
                     "target": base, "model": target_model}, notes


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
        # req.cwd is client-supplied (HTTP request body) - refuse UNC/device
        # syntax unconditionally, BEFORE cwd.is_dir()/.resolve() below ever
        # run. Checked on the EXPANDED string, not the raw one: expanduser()
        # is pure string/env-var work (no syscall), so it is safe to run
        # first, and a `~` value whose configured home directory is itself a
        # UNC path (a real roaming-profile configuration) would otherwise
        # expand INTO a UNC string after a pre-expansion check had already
        # cleared it - the check must see the same string that reaches
        # is_dir()/resolve(). Same guard as run_coder_task in
        # mcpserver/server.py. This is the branch the OWNER (or open-mode's
        # default-owner caller) and coder:full keys take - the `restricted`
        # branch above already
        # ignores req.cwd entirely and uses root_dir instead, so it needs no
        # guard of its own; the MORE-trusted branch is the one that actually
        # touches the string, not the less-trusted one.
        cwd = Path(req.cwd).expanduser()
        if _is_unc_or_device_path(str(cwd)):
            raise HTTPException(
                400, "'cwd' must be a local directory path, not a UNC or device path.")
        if not cwd.is_dir():
            raise HTTPException(400, f"Not a directory: {req.cwd}")
        cwd = cwd.resolve()

    # A resume of a directory that already has a RESUMED session running here
    # must join that one, not start a second: the GUI's "past sessions" rail
    # is a snapshot (not live), so a row for an already-resumed checkpoint
    # stays clickable, and repeat clicks would otherwise each spawn their own
    # CoderSession with its own open events stream that nothing ever tears
    # down. Scoped to an existing session that was ITSELF opened via resume -
    # a resume request must still be free to load a saved checkpoint into a
    # cwd that only has a plain (non-resumed) session running, which is a
    # different, legitimate request the join would otherwise silently eat.
    # Gated on `not restricted` for the same reason the resume-from-
    # checkpoint call below is (line ~629): a restricted session never
    # actually resumes history, so it has no "same conversation" to rejoin.
    if req.resume and not restricted:
        existing = mgr.find_by_cwd(cwd, principal=principal)
        if existing is not None and existing.opened_via_resume:
            return {**existing.info(), "resumed": False,
                    "notes": ["Already open - joined the session already "
                              "running for this folder instead of starting "
                              "another."]}

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

    from localm.audit import effective_mode, mode_at_least_as_private, parse_mode
    # Pass the session's project dir so a per-project .localcoder/config.toml mode
    # is honored by the GUI coder, not just the global coder_mode.
    # Resolved BEFORE the backend, not after: the privacy gate needs it to decide
    # whether this session may talk to an off-machine model at all.
    floor_mode = effective_mode("coder", cwd=cwd)
    if req.mode:
        try:
            requested_mode = parse_mode(req.mode)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # A scoped key must not force a less-private mode than the project's
        # configured floor - the same shape as the req.model gate above (a
        # restricted key cannot switch models either).
        if restricted and not mode_at_least_as_private(requested_mode, floor_mode):
            raise HTTPException(
                403, "Requesting a less private mode needs the owner key; a "
                "scoped key uses the project's configured mode.")
    session_mode = req.mode or floor_mode.value

    loop = asyncio.get_running_loop()
    # WHICH model server answers this session. Off the event loop:
    # netpolicy resolves DNS for a non-local backend, and a blocking resolve in
    # an async handler stalls every request this server is serving, not just
    # this one.
    backend, backend_info, notes = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: _resolve_backend(req, self_url=self_url,
                                 model_name=active_model(),
                                 restricted=restricted,
                                 session_mode=session_mode))
    # The request is wired to the backend for real; whether the SERVER honours
    # it is a separate question and the caller is told the answer rather than
    # left to assume. A localm self-connection does NOT implement the OpenAI
    # tools API (its ChatRequest declares no tools/tool_choice, so the fields
    # are dropped), and the session runs on localm's own grammar-constrained
    # tool-call convention instead - which is the equivalent guarantee, not a
    # downgrade. Nothing errors either way; saying nothing is what would make an
    # ignored option indistinguishable from an applied one.
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

    # Omitted -> the Agent's own default stays the single source of truth,
    # matching how temperature/max_tokens above are handled rather than
    # duplicating the number here.
    verify_kwargs = {}
    if req.verify_max_retries is not None:
        verify_kwargs["verify_max_retries"] = req.verify_max_retries

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
        backend_info=backend_info,
        **verify_kwargs,
        **gen_kwargs,
    ))
    session.principal = principal      # who owns this session (None = the owner)
    # Whether THIS session is itself a resume, for the join-guard above: it
    # must be able to tell "a resume request landed on an already-resumed
    # session" (join it) apart from "a resume request landed on a plain,
    # never-resumed session" (load the checkpoint into a genuinely new one
    # instead, unaffected by the plain session already running).
    session.opened_via_resume = bool(req.resume and not restricted)
    mgr.create(session)
    # Optional resume: restore this cwd's saved conversation into the new
    # session. Owner / coder:full only - a restricted scoped session must not
    # load the owner's prior conversation. The checkpoint read runs off the loop.
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
    (see ``coder.estimate.estimate_task`` for why that matters more here than in
    the CLI, which exits immediately afterwards). Refused while the session is
    busy: an estimate is a pre-flight on work you have not started, and running
    one against the shared engine mid-task would both queue behind the running
    turn and read as if it described it.

    The busy/closed decision is the SESSION's, taken under its own lock in
    ``run_estimate``, not a check made here - reading ``session.busy`` from the
    route and acting on it afterwards is check-then-act with a real window, and
    this route is a SECOND trigger for a backend that until now had exactly one.

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
        # The session refused the claim. A DEDICATED type, never a broad
        # RuntimeError: a model that raises RuntimeError would otherwise be
        # reported as a busy session - a fault hidden behind a status that
        # invites a retry.
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
    ``agent.flush_patch()``): a reloaded tab, a retry or a second reader must
    see the same patch, and a drain-on-read would look identical the first time
    and be empty ever after. 409 when the session is not in patch mode, because
    an empty diff there means "everything was written to disk", which is the
    opposite of what this endpoint reports."""
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
    ``--patch-mode FILE``, where the file lands on the CLIENT's disk because
    that is the only disk a browser can write to.

    Streamed from memory, never through a server-side temp file: the whole point
    of patch mode is that this content did not touch a disk."""
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

    Without this route a session's model is fixed forever at creation time
    (CoderSession.set_model's docstring in sessions.py has the full story of
    why that is a bug, not just a limitation) - the GUI's model switcher had no
    way to reach an already-running coder session at all, so it kept sending
    the ORIGINAL model on every request and could reload it right back into
    VRAM after the user switched away from it.

    Same trust model as create_session's optional model switch above: a
    per-session model change repoints the ONE shared engine for EVERYONE, so a
    scoped key must not trigger it (DoS / interfering with the owner's
    session) - mirrored here rather than factored out, since the "did the
    caller actually ask for a change" gate differs (there it is optional and
    compared against active_model(); here it is the whole point of the call)."""
    session = _get_session(request, session_id)
    if session.busy:
        # Fail fast, before the (expensive, globally-visible) switch_model call
        # below - session.set_model()'s own lock-protected busy check is the
        # authoritative gate against the case where this races a message.
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
        # switch_model preempts an in-flight load of a DIFFERENT model (single-
        # slot, http_server.switch_engine's preempt=True default) - a newer
        # switch elsewhere can abandon THIS one mid-load, and it reports that
        # by returning {"status": "superseded"} rather than raising. Reporting
        # 200 here would tell the caller its switch happened when it did not
        # (get_engine() itself guards the identical case at http_server.py).
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

    The REPL has had /approve, /scope and /verify since the beginning; the web
    surface had none of them, and the workaround on record (start again with
    resume) needs a checkpoint, which privacy mode never writes - and privacy is
    the default on both surfaces. So for a default GUI session there was no
    route to any of this at all.

    DELIBERATELY NOT BUSY-GUARDED, unlike the model route. The safety-relevant
    half of this is REVOKING auto-approve, and the moment a user reaches for
    that is precisely the moment the agent is mid-run doing something they want
    stopped. A 409 there would refuse the control in the only case it exists
    for. Tightening a scope is the same shape. These are attribute writes read
    at the next tool dispatch, not state a running turn can tear.
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
    # info() reports the EFFECTIVE state, so the caller reads back what actually
    # took rather than an echo of what it asked for - a re-detect that found no
    # check is the case that must not look like a check was set.
    return {**session.info(), "changed": changed}


@_router.post("/api/coder/sessions/{session_id}/cwd")
async def session_set_cwd(session_id: str, req: SessionCwdRequest,
                          request: Request):
    """Move a live session to another project directory (the REPL's /cd).

    Unreachable from the web surface in EVERY mode until now, including the
    resume workaround, because a checkpoint is looked up by cwd - so there was
    no shape of request that could move a conversation to another project.

    REFUSED for a restricted (scoped-key) session, and that is the whole
    containment: create_session forces such a session into the instance's
    project root and ignores the cwd it was given, so a route that moved it
    afterwards would hand back exactly what was taken away. Refused on the
    SESSION's restriction rather than the caller's, so an owner cannot move a
    shared key's session out of the root either.

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
    # Client-supplied path: refuse UNC/device syntax unconditionally, on the
    # EXPANDED string, BEFORE is_dir()/resolve() ever run - the same guard and
    # the same reasoning as create_session's cwd check.
    cwd = Path(req.cwd).expanduser()
    if _is_unc_or_device_path(str(cwd)):
        raise HTTPException(
            400, "'cwd' must be a local directory path, not a UNC or device path.")
    if not cwd.is_dir():
        raise HTTPException(400, f"Not a directory: {req.cwd}")
    cwd = cwd.resolve()

    # A project that declared itself private does not get a transcript. The
    # reasoning lives with the helper, which the REPL's /cd shares.
    from localm.plugins.coder.privacy import refuse_move_into_stricter_project
    refusal = refuse_move_into_stricter_project(session.mode, cwd)
    if refusal:
        raise HTTPException(409, refusal)

    # Rebuilding the project map scans the tree - the same reason create_session
    # builds its Agent in the executor rather than on the loop.
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

    NOT owner-gated, unlike history and episodes, and the difference is real
    rather than an oversight: those are the owner's own records held in the
    localm home directory, while this is a file in the session's own project
    directory that a restricted session's confined write_file/edit_file can
    already read and rewrite (SAFE_RESTRICTED_TOOLS). Gating it would refuse
    through the front door what stays open at the side, which is theatre.
    """
    session = _get_session(request, session_id)
    return session.memory()


@_router.post("/api/coder/sessions/{session_id}/memory")
async def session_remember(session_id: str, req: MemoryRequest, request: Request):
    """Append a bullet to the project memory (the REPL's /remember).

    The RELOAD is the point, not the write. A GUI session loads and injects
    LOCALCODER.md but had no way to change it, and asking the agent to edit the
    file does not call reload_memory - so an edit made that way sat in the file
    without reaching the running session, taking effect only next session. This
    goes through the agent, so the system prompt is rebuilt now.
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
    can tell "there is no memory file" from "no entry matched" - both leave the
    memory unchanged, and they call for different next steps.
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

    An owner GUI session can start background shell jobs and sub-agents through
    run_shell_background / spawn_agent_background, and had nowhere to enumerate
    them - it could only poll an id the model happened to mention.

    Scoped to this session's own jobs, never the whole registry: get_registry()
    is process-wide, and a GUI server runs many sessions in one process, so an
    unfiltered list would show one session another's work - and job labels are
    full command lines, so it would also read the owner's commands out to a
    scoped key. ``supported`` is reported because a restricted session has no
    background tools at all, and "none yet" must not look the same as "never".
    """
    session = _get_session(request, session_id)
    return {**session.background(), "supported": not session.restricted}


@_router.delete("/api/coder/sessions/{session_id}")
async def session_delete(session_id: str, request: Request):
    _get_session(request, session_id)   # principal check: 404 unless it is the caller's
    # remove() -> CoderSession.close() -> agent.close() is BLOCKING work: a
    # checkpoint write, the audit close, and - for a session that ran run_shell -
    # _detect_shell_changes(), which shells out to `git status --porcelain`
    # (timeout 10) plus up to two `git diff`s (timeout 15 each). Inline on this
    # async handler that froze the whole event loop, and every concurrent request
    # with it, for the git duration. Offload it, matching the pattern the other
    # potentially-slow routes here already use.
    #
    # Bounded at a bit over the roughly 40s worst-case git
    # budget above. Safe against corruption: SessionManager.remove() pops
    # session_id from its dict BEFORE close() runs (sessions.py), so an
    # abandoned close() cannot race a second DELETE of the SAME id - the
    # retry would see the id already gone and 404 immediately rather than
    # double-closing; the checkpoint file close() writes is keyed to this
    # session's own unique id, never shared with another session.
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
        # mislabelling this as privacy mode, which would be indistinguishable
        # from it.
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
    # `cwd` is a raw HTTP query parameter, and this is a GET route with no CSRF
    # check on it (CSRF only applies to unsafe methods) - refuse UNC/device
    # syntax unconditionally, BEFORE p.is_dir()/.resolve() below ever run.
    # Checked on the EXPANDED string (see create_session's cwd guard for why:
    # expanduser() is pure string/env-var work, no syscall, so it is safe to
    # run first and check what actually reaches is_dir()/resolve()).
    if _is_unc_or_device_path(str(p)):
        raise HTTPException(
            400, "'cwd' must be a local directory path, not a UNC or device path.")
    if not p.is_dir():
        return {"resumable": False}
    from localm.plugins.coder.agent import checkpoint_info
    info = checkpoint_info(p.resolve())
    if not info:
        return {"resumable": False}
    if info.get("unreadable"):
        return {"resumable": False, "unreadable": True}
    return {"resumable": True, "cwd": str(p.resolve()), **info}


# A permanent statement, not an empty-state message. The list below is
# INCOMPLETE by construction for anyone who uses privacy mode, and a surface
# that only admits that when it happens to be empty reads as an excuse for a
# short list rather than as a property of the feature.
_PRIVACY_NOTE = "Privacy-mode sessions are never recorded and never appear here."


def _dormant_for(path_str: str) -> list:
    """Past conversations saved for one project, newest first.

    Checkpoints are keyed on a digest of the project path and live under the
    data dir, NOT inside the project - so this still answers for a project
    directory that has been moved or deleted, which is exactly when a user most
    wants their conversation back.

    Privacy-mode sessions cannot appear: that mode writes no checkpoint at all
    (see Agent.save_checkpoint), so the omission is structural here rather than
    a filter this route could get wrong.
    """
    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    try:
        return list_checkpoints(Path(path_str))
    except Exception:
        # One unreadable project must not blank the whole listing; the rest of
        # the response is still true and useful.
        return []


@_router.get("/api/coder/dormant")
async def coder_dormant(request: Request, cwd: str = ""):
    """Past coder conversations, grouped by project, resumable by id.

    Owner-only for the same reason /api/coder/resumable is: a session title is
    the user's own words about their own work, so a scoped or shared key is
    never shown one.

    *cwd* is optional and only decides which group is marked ``current`` - the
    listing spans every remembered project either way, which is the point: a
    past session is reachable without first typing its project path back into
    the form.
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
    # project, which is unbounded filesystem work and would otherwise stall
    # every other request while it runs.
    loop = asyncio.get_running_loop()
    projects = await loop.run_in_executor(get_plugin_executor(), _collect)
    return {"projects": projects, "privacy_note": _PRIVACY_NOTE}


# ------------------------------------------------------------------ #
#  Episodic memory (the CLI's --episodes family)                      #
# ------------------------------------------------------------------ #

def _is_owner(request: Request) -> bool:
    """Owner-only is the gate for EVERY episode operation, read or write.

    Lessons are the owner's own record of their own projects, and a restricted
    (scoped-key) session is excluded from episodic memory entirely - it neither
    recalls a lesson nor writes one - so nothing here can belong to a shared key.
    """
    is_owner, _ = _principal_from_request(request)
    return is_owner


def _episode_root(cwd: str) -> Path:
    """Validate a caller-supplied project path and return the key to file under.

    ONE helper rather than a copy per route: five near-identical guards is five
    chances for one to drift, and the one that drifts is the one that stops
    refusing UNC.

    NO is_dir() check. Lessons live under the localm data dir keyed by the
    RESOLVED project path, so a directory that has been moved or removed still
    has an entry and stays reachable here.

    That leaves resolve() as the ONLY filesystem touch on a client-supplied
    string, and resolve() is required: the CLI derives the very same key by
    resolving the same way, so the two surfaces would otherwise disagree about
    which project they are looking at. UNC and device syntax is refused
    LEXICALLY, before that call - a GET carries no CSRF check, and a UNC string
    reaching the filesystem is an SMB dial.
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

    OWNER-ONLY, for the same reason ``/api/coder/resumable`` is: these are the
    owner's own past sessions on their own projects, and a restricted (scoped-
    key) session is excluded from episodic memory entirely - it neither recalls
    a lesson nor writes one - so there is nothing here that belongs to a shared
    key. A non-owner gets an empty list rather than a 403, matching resumable:
    the answer carries no information either way.

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
        # An unreadable store is NOT an empty one, and reporting it as empty
        # would be the exact "clean negative for a step that failed" the CLI's
        # own archive path refuses to do (see cli/_main.py's --episodes-archive).
        raise HTTPException(
            500, f"Could not read the episode store for {root}: "
                 f"{type(e).__name__}: {e}")
    return {"episodes": episodes, "cwd": str(root)}


@_router.get("/api/coder/episodes/archive")
async def coder_episodes_archive(request: Request, cwd: str = ""):
    """Lessons this project has DROPPED and can get back: ``--episodes-archive``.

    An unreadable archive is NOT an empty one, and that difference is the whole
    point of this endpoint: the lesson you are looking for may be sitting in
    there, recoverable, while a 200 with an empty list would tell you it is gone.
    The CLI refuses that collapse by exiting non-zero; here it is a 503, because
    the condition is transient (another process holding the file) and a retry is
    the right next move.
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

    Reversible - the record is archived first - so this is a plain POST rather
    than something the UI has to frighten anyone about. If the
    archiving half failed, the lesson is still gone from recall, so that is
    reported as a caveat on a real outcome rather than swallowed.
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
        # Said plainly rather than left to be inferred from a boolean: without
        # the archive copy this particular forget is NOT undoable, and a user who
        # believed otherwise would find that out at the worst possible moment.
        "warning": None if archived else
        "The lesson was dropped from recall, but the archive could not be "
        "written - so this one cannot be restored.",
    }


@_router.post("/api/coder/episodes/{episode_id}/restore")
async def coder_episode_restore(episode_id: str, request: Request,
                                req: EpisodeTargetRequest):
    """Put an archived lesson back into recall: ``--restore-episode``.

    Carries the CLI's two caveats, because both describe a restore that
    SUCCEEDED and is still not what the user pictured: the archive may not have
    been updated (so the lesson is live AND still listed as forgotten), and a
    store at its cap may have evicted it again immediately.
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
                # The archive EXISTS but could not be read, so "no such id" is a
                # claim we cannot make - the episode may well be in there.
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

    NOT reversible, and the archive goes too: "cleared episodic memory" while the
    lesson text still sat in a sidecar would be a privacy claim that is not true.
    The counts are read BEFORE the erase so the response can
    say what was actually destroyed - afterwards there is nothing left to count.
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
        # Read back rather than trusting the unlink. This is the one episode
        # operation with no undo: a partial erase leaves lesson text on disk
        # under a privacy promise that was not kept, so it is never reported
        # as success.
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

    OPT-IN and manual only, never automatic - a local model rewriting stored
    memory is exactly where one bad merge poisons every future run. Every input
    is archived, so a merge it gets wrong is reversible with restore.

    Reports what it DID (groups, merged, replaced, archived, skipped) rather than
    mutating silently, and a group whose merge came back unusable is counted as
    skipped and left alone.
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
        # Pinned to this localm. This HTTPBackend does NOT follow a session's
        # backend selection, unlike the one in create_session.
        # The session ESTIMATE route builds no backend at all - it calls
        # session.run_estimate and therefore already uses the session's own
        # backend, selection included.
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
