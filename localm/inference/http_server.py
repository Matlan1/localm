# SPDX-License-Identifier: AGPL-3.0-or-later
"""
OpenAI-compatible HTTP inference server built with FastAPI + uvicorn.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (streaming + non-streaming, multimodal-capable)

Start programmatically:
    from localm.inference.http_server import serve
    serve(engine, host="127.0.0.1", port=8642)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)
from fastapi.security import HTTPBearer

from localm import scopes
from localm.inference.backends.base import ModelLoadCancelled
from localm.inference.chat_pipeline import ChatPipeline
from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatResponse,
    FullChoice, Message, UsageInfo, make_chunk_id,
)

# Global engine reference set by serve()
_engine: Engine | None = None

# Inference serialisation - only one request runs inference at a time.
# Additional requests queue behind this semaphore.
_inference_sem: asyncio.Semaphore | None = None

# Preemptive model switching (see switch_engine). `_switch_desired` is the model
# name of the MOST RECENT switch request; `_switch_loading` is the model whose
# load is currently in flight; `_switch_cancel` aborts that in-flight load. These
# are read/written only on the event-loop thread inside switch_engine, except the
# cancel event, which the native load-progress callback reads from the loader
# thread (threading.Event is thread-safe).
_switch_desired: Optional[str] = None
_switch_loading: Optional[str] = None
_switch_cancel: Optional["threading.Event"] = None


async def switch_engine(name: str, make_engine, *, on_active=None) -> dict:
    """Swap the shared ``_engine`` to *name*, PREEMPTING any in-flight or queued
    load so the user's latest selection wins immediately.

    Selecting a new model while a previous selection is still loading used to make
    the server finish loading the abandoned model before even starting the new one
    (both queued behind the inference semaphore). Here, a newer request instead:

      * records itself as the desired model, and
      * aborts the in-flight load mid-flight (its native model load stops and
        returns, instead of running to completion) when it targets a DIFFERENT
        model, and
      * drops a queued-but-not-yet-started switch whose target is now stale.

    Still serialised on ``_inference_sem`` so a switch never races a generation
    onto the GPU. ``make_engine(name)`` builds (does NOT load) the target Engine;
    ``on_active(name)`` optionally syncs external active-model state when *name*
    becomes active.

    Returns a status dict:
      ``{"status": "loaded", "model": name}``          - this call loaded it,
      ``{"status": "already_active", "model": name}``  - it was already loaded,
      ``{"status": "superseded", "model": name, "by": <newer>}`` - abandoned; a
        newer selection owns the load (NOT an error).
    """
    global _engine, _switch_desired, _switch_loading, _switch_cancel
    _switch_desired = name
    # A newer target arrived: abort the in-flight load, unless it is already
    # loading THIS same model (then let it finish rather than restart it).
    if _switch_cancel is not None and _switch_loading != name:
        _switch_cancel.set()

    loop = asyncio.get_running_loop()
    async with _inference_sem:
        # Coalesce: an even-newer request may have superseded us while we waited
        # for the semaphore - don't load a stale target; let the newest win.
        if _switch_desired != name:
            return {"status": "superseded", "model": name, "by": _switch_desired}
        if _engine is not None and _engine.display_name == name and _engine.loaded:
            if on_active is not None:
                on_active(name)
            return {"status": "already_active", "model": name}

        new_engine = make_engine(name)
        cancel = threading.Event()
        _switch_cancel = cancel
        _switch_loading = name
        new_engine.set_load_cancel(cancel)   # make the native load abortable
        try:
            old = _engine
            if old is not None and old.loaded:
                await loop.run_in_executor(None, old.unload)
            await loop.run_in_executor(None, new_engine.load)
        except ModelLoadCancelled:
            # A newer switch aborted us mid-load. Report superseded, not an error.
            return {"status": "superseded", "model": name, "by": _switch_desired}
        finally:
            # Clear the in-flight markers only if they are still OURS (a later
            # switch may already have installed its own before we got here).
            if _switch_cancel is cancel:
                _switch_cancel = None
                _switch_loading = None
        _engine = new_engine
        if on_active is not None:
            on_active(name)
        return {"status": "loaded", "model": name}

# Monotonic timestamp of the last inference request, for the optional idle-unload
# loop (config "idle_unload_seconds"). Touched at the start of each inference
# endpoint, like Ollama's keep_alive (measured from the last request).
_last_activity: float = time.monotonic()


def _touch_activity() -> None:
    """Record that an inference request just arrived (resets the idle timer)."""
    global _last_activity
    _last_activity = time.monotonic()


def _sanitize_client_context(raw) -> dict:
    """Reduce an untrusted GUI ``client`` payload to a safe, bounded dict for a bug
    report: only known string fields (capped) plus a capped list of console-error
    strings. Anything else is dropped. Returns {} for non-dict / empty input."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for field in ("userAgent", "page", "viewport", "appVersion"):
        val = raw.get(field)
        if isinstance(val, (str, int, float)) and str(val).strip():
            out[field] = str(val)[:500]
    console = raw.get("console")
    if isinstance(console, list):
        errs = [str(e)[:1000] for e in console if isinstance(e, (str, int, float))]
        if errs:
            out["console"] = errs[-40:]
    return out


def _idle_unload_ttl() -> int:
    """Configured idle-unload TTL in seconds (0 = disabled), read live so a
    Settings change applies without a restart. A bad value falls back to 0."""
    try:
        from localm.config import load_config
        return max(0, int(load_config().get("idle_unload_seconds", 0) or 0))
    except (TypeError, ValueError):
        return 0


async def _idle_unload_once(ttl: int) -> bool:
    """One idle check: unload the model if it has been idle for >= ttl seconds.
    Returns True if it unloaded. Does NO sleeping (the loop owns cadence), so the
    decision is unit-testable without waiting.

    The unload runs UNDER the inference semaphore so it can never free the native
    context mid-decode (that crashes the GPU driver), and the idle time is
    re-checked inside the lock so a request that arrived while we waited for the
    lock cancels the unload. The next inference reloads the model lazily."""
    if (ttl <= 0 or _engine is None or not _engine.loaded
            or _inference_sem is None):
        return False
    if (time.monotonic() - _last_activity) < ttl:
        return False
    loop = asyncio.get_running_loop()
    async with _inference_sem:
        # Re-check under the lock: a request may have touched activity (or the
        # model may already be gone) while we waited for the semaphore.
        if not (_engine is not None and _engine.loaded
                and (time.monotonic() - _last_activity) >= ttl):
            return False
        idle_s = int(time.monotonic() - _last_activity)
        await loop.run_in_executor(None, _engine.unload)
        from localm.debuglog import logger as _dbg
        _dbg.info("idle-unload: freed %s after %ds idle (ttl=%ds); it reloads "
                  "on the next request", _engine.display_name, idle_s, ttl)
        return True


async def _idle_unload_loop() -> None:
    """Free the model from VRAM after `idle_unload_seconds` of no inference.

    Opt-in (default 0 = disabled). Runs as a lifespan background task; the actual
    decision lives in `_idle_unload_once`. A transient error is logged (RULE 5:
    surface, do not swallow) instead of killing the loop."""
    while True:
        ttl = _idle_unload_ttl()
        if ttl <= 0:
            # Disabled: poll occasionally so enabling it at runtime takes effect.
            await asyncio.sleep(30)
            continue
        # Check within the TTL, but not too hot and not too slow.
        await asyncio.sleep(max(5, min(ttl, 30)))
        try:
            await _idle_unload_once(ttl)
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.warning("idle-unload check failed (continuing)", exc_info=True)

# Optional bearer-token auth - enabled when LOCALM_API_KEY is set.
_bearer_scheme = HTTPBearer(auto_error=False)

# S2: the browser GUI authenticates with an HttpOnly session cookie whose value is
# an OPAQUE session id (localm.sessions), NOT the API key - the page JS can never
# read it, and rolling the key does not invalidate it. Cookie-sourced auth on a
# state-changing method must additionally carry a CSRF token in this header; the
# token is an HMAC DERIVED from the session (csrf_token_for / _csrf_ok), fetched by
# the client from GET /api/session, so it is always in lockstep with the session and
# cannot desync (there is NO separate CSRF cookie). The Authorization-header path
# (CLI / SDK / coder) cannot be forged cross-site and is therefore CSRF-exempt.
SESSION_COOKIE = "localm_session"
CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Hard cap on any request body, rejected (413) from the Content-Length BEFORE the
# body is buffered or parsed - so a large base64 upload cannot be materialized in
# memory ahead of a route's own size checks (a decode-time OOM DoS on
# /api/rag/upload + /extract, CWE-400). 160 MB comfortably fits the largest
# legitimate upload (100 MB decoded ~= 133 MB base64 + the JSON wrapper) while
# rejecting anything larger up front. Nothing else the server accepts is remotely
# this big. Read at request time so a test can monkeypatch it.
MAX_REQUEST_BODY_BYTES = 160_000_000
# SEAMLESS: the session cookie PERSISTS so the user stays signed in across a browser
# or PWA restart, instead of being a session cookie the browser drops on close
# (which made the key gate - and its "Install certificate" step - reappear every
# time, the "install a new cert / re-enter the key every restart" report). Browsers
# clamp cookie lifetime to ~400 days, so we ask for that ceiling; the escape hatch is
# /api/session/logout (Settings: leave the key blank and Save).
SESSION_MAX_AGE = 400 * 24 * 3600  # ~400 days (the browser cap)


def _bearer_token(request) -> Optional[str]:
    """Extract a presented bearer token from the raw Authorization header (used
    by the origin/management middleware and the request-aware auth core)."""
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def _request_token(request) -> tuple[Optional[str], str]:
    """Resolve the presented key and where it came from. The Authorization
    header wins (programmatic clients); otherwise the HttpOnly ``localm_session``
    cookie (the browser GUI). Returns ``(token, source)`` with *source* one of
    ``"header"`` / ``"cookie"`` / ``"none"``."""
    header = _bearer_token(request)
    if header:
        return header, "header"
    cookie = (request.cookies.get(SESSION_COOKIE) or "").strip()
    if cookie:
        return cookie, "cookie"
    return None, "none"


def _principal_from_token(token, source):
    """Resolve a presented credential to ``(scopes, key_hash, fs_access)`` or None.

    A ``header`` token is a raw API key -> ``auth.verify()``. A ``cookie`` token is
    now an OPAQUE SESSION ID -> the server-side session store (``localm.sessions``),
    which returns the scope / owning-key / fs-access SNAPSHOT taken at login. So a
    cookie session stays valid across an owner-key roll (the reported bug), and the
    durable key never has to live in the cookie. ``key_hash`` is the sha256 of the
    key that minted the session, so ``principal_id`` over a cookie matches the same
    key presented as a bearer (job ownership parity)."""
    if not token:
        return None
    if source == "cookie":
        from localm import sessions
        rec = sessions.lookup(token)
        if rec is None:
            return None
        held = set(rec.get("scopes", []))
        if scopes.ADMIN not in held:
            # A SCOPED-key session lives only as long as its underlying key does:
            # re-validate the owning key against the live keystore every request, so
            # revoking or expiring the key cuts the session off too (parity with the
            # bearer path, where verify() re-checks the keystore each request). An
            # owner/ADMIN session is intentionally exempt - it is decoupled from the
            # key VALUE so an owner-key ROLL does not log the owner out (the S1 fix).
            from localm.auth import key_hash_live
            if not key_hash_live(rec.get("key_hash")):
                return None
        return (held, rec.get("key_hash"), rec.get("fs_access", "none"))
    from localm.auth import _hash_key, fs_access_for, verify
    held = verify(token)
    if held is None:
        return None
    fs = "host" if scopes.ADMIN in held else fs_access_for(token, "none")
    return held, _hash_key(token), fs


def _csrf_secret(request) -> str:
    """The per-process CSRF secret for this app, created lazily if a standalone
    mount (a bare attach_gui in tests) never ran create_app's setup."""
    st = request.app.state
    sec = getattr(st, "csrf_secret", None)
    if not sec:
        sec = secrets.token_urlsafe(32)
        st.csrf_secret = sec
    return sec


def csrf_token_for(request, sid: str) -> str:
    """The CSRF token for a session id: a deterministic HMAC of the id under the
    per-process secret. Derived from the session, so it is available exactly when
    the session is (delivered to the client via GET /api/session and the shell
    <meta>) and can never desync from it. Empty for an empty sid."""
    if not sid:
        return ""
    return hmac.new(_csrf_secret(request).encode("utf-8"),
                    sid.encode("utf-8"), hashlib.sha256).hexdigest()


def _csrf_ok(request) -> bool:
    """CSRF check for a cookie-authenticated unsafe request: the ``X-CSRF-Token``
    header must equal the token DERIVED from the session cookie (signed with the
    per-process secret). No separate CSRF cookie exists to be cleared or to desync,
    so a valid session always has a usable token. A cross-site page can neither read
    the HttpOnly session cookie nor the token (SOP), and cannot set a non-simple
    header; SameSite=Strict and the same-origin guard remain in force."""
    header = request.headers.get(CSRF_HEADER, "")
    sid = (request.cookies.get(SESSION_COOKIE) or "").strip()
    if not header or not sid:
        return False
    return hmac.compare_digest(header, csrf_token_for(request, sid))


def _enforce_request(request: Request, scope: Optional[str]) -> None:
    """Shared auth core (request-aware). Open/dev mode when no key is configured
    anywhere (unless LOCALM_REQUIRE_AUTH forces fail-closed). Otherwise a valid
    key is required - from the Authorization header OR the session cookie - and
    *scope* (None = 'any valid key') must be granted (the owner key implies every
    scope). Cookie-sourced auth on an unsafe method additionally requires a valid
    CSRF token (an HMAC derived from the session)."""
    from localm.auth import any_key_configured, require_auth_enabled
    if not any_key_configured():
        if require_auth_enabled():
            raise HTTPException(
                status_code=401,
                detail="Auth required but no API key configured "
                       "(set one via the launcher or LOCALM_API_KEY)")
        return  # open/dev mode
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    held = prin[0]
    if source == "cookie" and request.method not in _SAFE_METHODS:
        if not _csrf_ok(request):
            raise HTTPException(
                status_code=403,
                detail="Missing or invalid CSRF token for a cookie-"
                       "authenticated state change.")
    if scope is not None and not scopes.grants(held, scope):
        raise HTTPException(status_code=403,
                            detail=f"Key lacks required scope: {scope}")


def _require_auth(request: Request) -> None:
    """Require any valid API key (no specific scope)."""
    _enforce_request(request, None)


def require_scope(scope: str):
    """FastAPI dependency factory: require a key whose scopes grant *scope*.
    Use as ``dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))]``."""
    def dep(request: Request) -> None:
        _enforce_request(request, scope)
    return dep


def caller_scopes(request: Request) -> Optional[set]:
    """The scope set the presented key grants (the owner key -> {ADMIN}), or
    None in open mode / when no valid key is presented. Routes use this to make
    authorisation decisions that depend on *who* the caller is (e.g. only an
    owner/ADMIN principal may mint keys carrying privileged scopes)."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    return prin[0] if prin else None


def principal_id(request: Request) -> Optional[str]:
    """A stable, opaque per-key identity for the CURRENT caller, or None in open
    mode / when no token is presented. It is the same SHA-256 the keystore stores
    (never the plaintext key), so it identifies the key WITHOUT exposing it, and
    is identical whether the key arrives via the Authorization header or the
    session cookie. Used to bind a background job to the key that created it
    (KEY-SCOPE-2), so only that key (or an admin/owner) may stream or cancel it."""
    from localm import scopes
    from localm.auth import any_key_configured
    if not any_key_configured():
        return None
    token, source = _request_token(request)
    if not token or not token.strip():
        return None
    if source == "cookie":
        from localm import sessions
        rec = sessions.lookup(token)
        if rec:
            return rec.get("key_hash")
        return None
    prin = _principal_from_token(token, source)
    if prin is not None:
        held, key_hash, _ = prin
        return key_hash
    return None


def job_owner_ok(request: Request, job_owner: Optional[str]) -> bool:
    """Whether the caller may stream/cancel a job created by *job_owner*. A job
    with NO recorded owner (created in open mode) is unrestricted; an admin/owner
    key may reach any job; otherwise the caller's principal must match the
    creator's. Pairs with principal_id() stamped at job creation."""
    if job_owner is None:
        return True
    held = caller_scopes(request)
    if held is not None and scopes.ADMIN in held:
        return True
    return principal_id(request) == job_owner


def effective_fs_access(request: Request) -> str:
    """The caller's effective reach into the SERVER HOST filesystem: "host" (the
    whole disk), "shared" (confined to owner-designated shared roots), or "none"
    (no host FS - device upload only).

    Open mode (loopback owner) and the owner/ADMIN key always resolve to "host";
    any other valid key uses its stored fs_access level (default "none" for a
    legacy key); no valid key -> "none". Filesystem reach is a per-credential dial
    kept deliberately INDEPENDENT of ownership, so an owner can pair one of their
    own devices with a lower-reach key."""
    from localm.auth import any_key_configured
    if not any_key_configured():
        return "host"                       # open/dev mode = loopback owner
    token, source = _request_token(request)
    prin = _principal_from_token(token, source)
    if prin is None:
        return "none"                       # keys configured, none/invalid presented
    held, _key_hash, fs = prin
    if scopes.ADMIN in held:
        return "host"                       # owner key / owner session
    return fs                               # bearer key or session fs-access snapshot


def require_fs_host(request: Request) -> None:
    """FastAPI dependency: require a caller with FULL host filesystem access
    (owner / open mode / a key explicitly granted fs_access=host). Gates the host
    file/folder browser so a merely config-reading key can no longer enumerate the
    server's disk."""
    _enforce_request(request, None)         # a valid key (or open mode) first
    if effective_fs_access(request) != "host":
        raise HTTPException(
            status_code=403,
            detail="This key does not have host filesystem access")


def _is_loopback_host(host: str) -> bool:
    """True for a loopback bind host (127.0.0.0/8, ::1, localhost).

    Decisions that must distinguish a local-only server from a network-exposed
    one key off the CONFIGURED bind host, never the request peer: portmux relays
    every connection through an internal loopback socket, so the peer always looks
    like 127.0.0.1 even for a genuinely remote client (see portmux.py, and
    gui/web.py which makes the same choice for open-mode key seeding)."""
    import ipaddress
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ------------------------------------------------------------------ #
#  Surface mounting (H6 phase 5: on-demand GUI on a running instance)  #
# ------------------------------------------------------------------ #

def mount_gui_surface(app) -> bool:
    """Add the GUI surface (its /api routes + the SPA static mount) to a running
    ``api``-mode app, in place. Idempotent: returns False if a GUI is already
    mounted (a ``full`` instance, or a second call), True if it mounted now.

    Safe at runtime because ``attach_gui`` only appends routes + a ``/`` catch-all
    mount and sets ``app.state`` services - it adds NO middleware (Starlette reads
    ``app.router.routes`` per request, so appended routes take effect immediately;
    only new middleware would need a stack rebuild). The engine + inference
    semaphore are this instance's own (it already loaded the model for /v1), so no
    second model load happens; ``switch_model`` swaps the shared ``_engine`` under
    ``_inference_sem`` exactly as the GUI launcher does."""
    global _engine
    if getattr(app.state, "gui_mounted", False):
        return False

    scheme = getattr(app.state, "instance_scheme", "http")
    port = getattr(app.state, "instance_port", None)
    if not port:
        # advertise() sets instance_port before uvicorn accepts connections, so a
        # real request can never reach here without it; guard anyway so a manual
        # app build fails loudly instead of dialling "http://127.0.0.1:None/v1".
        raise HTTPException(500, "Instance not fully started (no bind port); "
                            "cannot mount the GUI surface yet.")
    self_url = f"{scheme}://127.0.0.1:{port}/v1"

    def active_model() -> str:
        return _engine.display_name if _engine is not None else ""

    def _build_engine(name: str) -> Engine:
        from localm.config import load_registry
        from localm.model_manager import get_model_info, get_model_mmproj
        info = get_model_info(name)
        if info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = info
        # VIS-1: a GUI/registry switch must not drop vision. Carry the model's
        # mmproj (a registry-recorded one, else a sibling projector auto-detected
        # next to the GGUF) into the new Engine, the same way the CLI --mmproj flag
        # does - otherwise switching models silently loses image support.
        mmproj = get_model_mmproj(name)
        return Engine(
            str(m_path),
            display_name=name if name in load_registry() else m_hint,
            mmproj_path=mmproj,
        )

    async def switch_model(name: str) -> dict:
        # Preemptive switch: a newer selection aborts an in-flight load rather
        # than waiting for the abandoned model to finish (see switch_engine).
        return await switch_engine(name, _build_engine)

    from localm.plugins.gui.web import attach_gui
    # Claim the mount BEFORE attaching so a re-entrant (or, after some future
    # refactor, a concurrent) call cannot double-register the GUI routes; roll the
    # flag back if the attach itself fails. Today mount_gui_surface + attach_gui
    # run fully synchronously inside the request handler, so the event loop never
    # interleaves two of them - this just makes the invariant explicit and robust.
    app.state.gui_mounted = True
    try:
        manager = attach_gui(
            app, self_url=self_url, switch_model=switch_model, active_model=active_model)
    except Exception:
        app.state.gui_mounted = False
        raise
    # attach_gui re-affirms app.state.gui_mounted; reflect the surface change in
    # discovery so /whoami and the registry report this is now a full instance.
    app.state.coder_sessions = manager
    app.state.instance_mode = "full"
    app.openapi_schema = None   # force the schema to include the new routes
    try:
        from localm import instances
        from localm.config import home_dir
        instances.set_mode(home_dir(), getattr(app.state, "instance_id", ""), "full")
    except Exception as e:
        # The mount already succeeded; this is best-effort registry sync (so
        # discovery advertises "full" not "api"), not fatal - but now visible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("registry mode not updated to full: %s", e)
    return True


# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def _do_shutdown() -> None:
    """SRV-4: the actual stop sequence. Unload the model FIRST so the native
    context is freed cleanly (a hard exit while it is loaded segfaults during
    teardown), clear the crash marker so this intentional stop is not reported as
    a crash, then exit the process so the stop is guaranteed (Ctrl+C sometimes
    does nothing). Separated from the route so it can be tested without exiting."""
    try:
        if _engine is not None:
            _engine.unload()
    except Exception:
        pass
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard()
    except Exception:
        pass
    import os
    os._exit(0)


def _request_shutdown(delay: float = 0.25) -> None:
    """Run _do_shutdown shortly after returning, so the 200 response flushes to
    the client before the process exits."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_shutdown()

    threading.Thread(target=_run, daemon=True).start()


def _restart_argv() -> list:
    """The command line to re-launch this server. Always ``python -m localm <args>``
    - the canonical entry the codebase uses - so a restart works regardless of how
    the server was originally started (a console-script .exe, ``-m``, or a script
    path, any of which can make ``sys.argv[0]`` un-re-runnable by the interpreter)."""
    import sys
    return [sys.executable, "-m", "localm", *sys.argv[1:]]


def _do_restart() -> None:
    """R18: restart this server IN PLACE. Unload the model FIRST (clean native
    teardown, like _do_shutdown - a hard re-exec while it is loaded can segfault),
    clear the crash marker so this intentional restart is not reported as a crash,
    then re-exec the same command line so the server comes back on the same port.
    os.execv replaces the process image and does not return on success. Separated
    from the route so it can be tested without actually re-execing."""
    try:
        if _engine is not None:
            _engine.unload()
    except Exception:
        pass
    try:
        from localm import bugreport
        bugreport.disarm_crash_guard()
    except Exception:
        pass

    try:
        global _audit
        if _audit is not None and hasattr(_audit, "close"):
            _audit.close()
    except Exception:
        pass

    try:
        from localm.debuglog import dump_ring_buffer, flush_log_handlers, recent_activity
        # Privacy mode opts out of ALL automatic disk traces, so skip the
        # crash-recovery breadcrumb dumps (ring buffer + pre_restart.log): they are
        # session-derived INFO breadcrumbs written without the user asking. If the
        # mode cannot be resolved, fail toward privacy (skip) - matching audit.py's
        # fail-safe-to-privacy default - rather than write a trace the user may have
        # opted out of.
        try:
            from localm.audit import SessionMode, effective_mode
            _privacy = effective_mode("server") == SessionMode.PRIVACY
        except Exception:
            _privacy = True
        if not _privacy:
            dump_ring_buffer()
            # Also write a clear text log for the bug reporter to ingest directly,
            # in case the JSON buffer fails to load back into memory.
            from localm.config import home_dir
            pre_log = home_dir() / "logs" / "pre_restart.log"
            pre_log.parent.mkdir(parents=True, exist_ok=True)
            pre_log.write_text("\n".join(recent_activity()), encoding="utf-8")
        # Flush all log handlers before os.execv so no buffered lines are lost
        # (Task 1: log durability / save-bug). This flushes already-open handlers;
        # it creates no new trace file, so it is safe in privacy mode too.
        flush_log_handlers()
    except Exception:
        pass

    import os
    import sys

    # NEW-G: no inheritable fd (the debug-log FileHandler, the uvicorn listening
    # socket) must survive into the re-exec'd image, where the freshly loaded
    # ggml/llama runtime warns "Failed to close child file descriptors at 3/4".
    # os.execv does NOT close fds; marking them non-inheritable drops them at exec
    # (POSIX O_CLOEXEC / Windows handle inheritance). Best-effort per fd; stdin/
    # stdout/stderr (0-2) are left inheritable so the new process keeps the console.
    try:
        max_fd = 4096
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if isinstance(soft, int) and 0 < soft < max_fd:
                max_fd = soft
        except Exception:
            pass  # no resource module (Windows) - the 4096 default is plenty
        for fd in range(3, max_fd):
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass  # not an open fd
    except Exception:
        pass  # never let fd hygiene block the restart

    os.execv(sys.executable, _restart_argv())


def _request_restart(delay: float = 0.25) -> None:
    """Run _do_restart shortly after returning, so the 200 response flushes to the
    client before the process re-execs (mirrors _request_shutdown)."""
    import threading
    import time as _t

    def _run():
        _t.sleep(delay)
        _do_restart()

    threading.Thread(target=_run, daemon=True).start()


def create_app(engine: Engine, *, api_landing: bool = False) -> FastAPI:
    global _engine, _inference_sem
    _engine = engine

    # Inference serialisation semaphore. Created eagerly so the app works even
    # when the lifespan does not run (tests, or being mounted inside another
    # app); on Python 3.10+ a Semaphore binds to the event loop lazily on first
    # use, so constructing it here without a running loop is safe. The lifespan
    # re-affirms it for the real uvicorn server.
    _inference_sem = asyncio.Semaphore(1)

    # Session-persistence mode for this server (privacy → no traces).
    # One audit log / transcript covers the server lifetime; GUI chat and
    # any API client traffic flow through /v1/chat/completions and land here.
    from localm.audit import effective_mode, make_audit_log, make_transcript
    _mode = effective_mode("server")
    _audit = make_audit_log(_mode, label="server")
    _transcript = make_transcript(_mode, label="server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem
        # Semaphore created inside the running event loop - Python 3.10+ safe
        _inference_sem = asyncio.Semaphore(1)
        # Prune expired browser sessions once at startup so an install that rarely
        # mints new sessions does not accumulate stale rows (create() only prunes
        # opportunistically). Best-effort: a sweep failure must never block startup.
        try:
            from localm import sessions as _sessions
            _sessions.sweep()
        except Exception:
            from localm.debuglog import logger as _dbg
            _dbg.debug("session sweep at startup failed (non-fatal)", exc_info=True)
        # SRV-3: route an uncaught asyncio task exception through the bug reporter
        # instead of a silent "Task exception was never retrieved". Skipped under
        # pytest so the test runner keeps its own loop handling.
        if "pytest" not in sys.modules:
            try:
                from localm import bugreport
                bugreport.install_asyncio_handler(asyncio.get_running_loop())
            except Exception:
                pass
        # Plugins register() before this loop existed, so loop-dependent plugin
        # work (the jobs scheduler) is queued on the manager; run it now that
        # the loop is up. Without this, no scheduled job ever fired on a stock
        # server start (memory-audit 2026-07-02, critical C2). attach_engine
        # runs after create_app, so the manager is resolved at lifespan time.
        _pm = getattr(app.state, "plugin_manager", None)
        if _pm is not None:
            _pm.run_startup_callbacks()
        # Optional idle-unload background task (config "idle_unload_seconds"); it
        # is a cheap no-op while disabled. Cancelled on shutdown so it never
        # outlives the app.
        idle_task = asyncio.create_task(_idle_unload_loop())
        try:
            yield
        finally:
            idle_task.cancel()
            try:
                await idle_task
            except asyncio.CancelledError:
                pass
            _audit.close()

    app = FastAPI(
        title="localm inference server",
        version="0.1.1",
        lifespan=lifespan,
    )

    # SRV-2: one backstop so an unexpected error in ANY route returns a
    # consistent JSON 500 and is logged, instead of leaking a traceback or a
    # bare body. Starlette already keeps a Python exception from killing the
    # server; this standardises the response shape and the logging so a single
    # failing request is a clean 500, never a crash and never an info leak. (A
    # native fault - a C-extension segfault - cannot be caught in-process; those
    # are prevented at the source, e.g. voice audio is decoded/validated before
    # the native path, and surfaced via the crash marker on restart.)
    @app.exception_handler(Exception)
    async def _unhandled_error(request, exc):  # noqa: ANN001 - framework signature
        from localm.debuglog import logger as _dbg
        _dbg.exception("unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error"})

    # Chat-pipeline hooks: plugins register inlet/stream/outlet transforms that
    # run on every /v1/chat/completions turn. Created here so it exists before
    # plugins load (attach_engine, below) and stays reachable as
    # request.app.state.chat_pipeline. A pipeline with no hooks is a no-op.
    app.state.chat_pipeline = ChatPipeline()

    # Per-process "shell token" (H5): in open mode the management routes require
    # this token, which the loopback GUI shell injects into the SPA (web.py
    # _gui_index). It gates the no-Origin local-client path that bearer auth /
    # the Origin guard alone do not cover. Per-process so it dies on restart;
    # never persisted.
    app.state.shell_token = secrets.token_urlsafe(32)

    # Per-process CSRF secret. The CSRF token is a deterministic HMAC of the
    # session id (below), so it is provably present exactly when the session is and
    # CANNOT desync from it (the old design used a SEPARATE readable cookie that a
    # client reset could clear while the HttpOnly session survived, 403-ing every
    # write). Per-process so it dies on restart (the client re-fetches the token
    # from /api/session on the next boot); never persisted.
    app.state.csrf_secret = secrets.token_urlsafe(32)

    # api-mode landing: a bare `localm serve` has no GUI shell, so GET / would
    # 404. Redirect it to the auto-generated API docs. Only on the api path so
    # it never collides with the GUI's own "/" handler + StaticFiles catch-all.
    if api_landing:
        @app.get("/", include_in_schema=False)
        async def _api_root() -> RedirectResponse:
            return RedirectResponse(url="/docs", status_code=307)

    # Debug mode: log every request with timing to the debug log file
    from localm.debuglog import debug_enabled, logger as _dbg
    if debug_enabled():
        @app.middleware("http")
        async def _log_requests(request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            _dbg.debug(
                "%s %s -> %d (%.0f ms)",
                request.method, request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
            )
            return response

    # CORS: localhost-only by default. A wildcard here would let ANY website
    # the user visits call this API from browser JS and read the responses
    # (drive-by GPU use, response exfiltration, /v1/models/unload abuse).
    # Override with config "cors_origins": ["https://app.example"] or "*".
    from localm.config import load_config
    cors_cfg = load_config().get("cors_origins")
    cors_kwargs: dict
    if cors_cfg == "*":
        cors_kwargs = {"allow_origins": ["*"]}
    elif isinstance(cors_cfg, list) and cors_cfg:
        cors_kwargs = {"allow_origins": cors_cfg}
    else:
        cors_kwargs = {
            "allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        }
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["*"],
        allow_headers=["*"],
        **cors_kwargs,
    )

    # CSRF / drive-by guard. The default CORS policy admits ANY localhost:PORT
    # origin, so without this a malicious local web page (a dev server, an npm
    # postinstall server) could drive state-changing endpoints from the user's
    # browser - mint a key, flip require_auth, install a plugin, load/unload the
    # model, browse the filesystem, OR index-and-read arbitrary files through a
    # plugin data route like /api/rag, or drive the coder agent via /api/coder -
    # even on a keyless install (open-mode scope collapse). We require every
    # unsafe-method request to be same-origin (or an explicitly configured CORS
    # origin), EXCEPT the OpenAI-compatible inference API, which is deliberately
    # left cross-origin callable so a local app can use it. Guarding by default
    # (allowlist, not denylist) means a new plugin route is protected the moment
    # it is added, without anyone remembering to register it here. Non-browser
    # clients (CLI / SDK) send no Origin; "cors_origins": "*" opts out entirely.
    _UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    _CROSS_ORIGIN_OK = (
        "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
        # Surface management (phase 5 on-demand GUI mount) is driven by a local
        # process (the attaching `localm gui`), not the browser shell: it sends
        # no Origin and has no shell_token. The route does its OWN strict auth
        # (this instance's attach token, or an owner API key), which - not the
        # same-origin gate - is the real credential, so it is exempt here. A
        # cross-origin browser page still cannot set the Authorization header
        # without a secret it cannot read, so the exemption adds no CSRF surface.
        "/v1/surfaces/",
    )
    _cors_allowlist = cors_cfg if isinstance(cors_cfg, list) else []
    _cors_wildcard = cors_cfg == "*"

    @app.middleware("http")
    async def _origin_guard(request, call_next):
        if (request.method in _UNSAFE_METHODS
                and not request.url.path.startswith(_CROSS_ORIGIN_OK)):
            # Same-origin / CORS-allowlist check. "cors_origins": "*" opts OUT
            # of THIS check only (an explicit "any origin may call me") - it
            # must NOT also waive the open-mode shell-token gate below, which is
            # a separate credential requirement, not a same-origin check
            # (AUD-CORSWILD).
            if not _cors_wildcard:
                origin = request.headers.get("origin")
                allowlisted = bool(origin) and origin in _cors_allowlist
                if origin:
                    host = request.headers.get("host", "")
                    same_origin = origin.split("://", 1)[-1] == host
                    if not (same_origin or allowlisted):
                        return JSONResponse(
                            status_code=403,
                            content={"detail": "Cross-origin request refused "
                                     "(only same-origin requests or a configured "
                                     "'cors_origins' may use this endpoint)."},
                        )
            # H5: open-mode management gate. With no key configured, management
            # routes still require the per-process shell token (injected into the
            # loopback GUI shell), so a no-Origin local client - curl, a script -
            # can no longer mint a key, flip config, install a plugin, load a
            # model, or browse the filesystem unauthenticated. Protected mode (a
            # key exists) is handled by bearer auth on the route; a require_auth-
            # without-key install 503s at the route. The token is required even
            # for an allowlisted CORS origin (F2): an Origin header is forgeable
            # by a non-browser client, so it is not a management credential - a
            # configured external origin must use an API key for state changes.
        is_unsafe = request.method in _UNSAFE_METHODS
        is_metadata_get = (
            request.method == "GET"
            and (request.url.path.startswith("/api/") or request.url.path.startswith("/v1/"))
            and request.url.path != "/api/session"
            and not request.url.path.startswith("/v1/models")
        )
        if (is_unsafe or is_metadata_get) and not request.url.path.startswith(_CROSS_ORIGIN_OK):
            from localm.auth import any_key_configured, require_auth_enabled
            if not any_key_configured() and not require_auth_enabled():
                token = getattr(request.app.state, "shell_token", None)
                presented = _bearer_token(request)
                if not (token and presented
                        and hmac.compare_digest(presented, token)):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Open-mode management requires the "
                                 "localm GUI shell on this machine, or an API key "
                                 "(run 'localm key generate')."},
                    )
        return await call_next(request)

    @app.middleware("http")
    async def _limit_body_size(request, call_next):
        # Reject an over-large body from its Content-Length, BEFORE Starlette reads
        # and parses it, so a giant base64 upload cannot be buffered into memory
        # ahead of a route's own caps (the /api/rag/upload + /extract decode-time
        # OOM DoS). Read the module global at call time so it stays monkeypatchable.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                too_big = int(cl) > MAX_REQUEST_BODY_BYTES
            except ValueError:
                too_big = False
            if too_big:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large."})
        return await call_next(request)

    # Security response headers (R41 defense-in-depth). The user-content render
    # path is already XSS-safe via DOMPurify (see dev-notes/SECURITY-xss-render-
    # review-2026-06-23.md); the one gap that review found was the absence of a
    # Content-Security-Policy backstop on the GUI shell. nosniff is enforced
    # everywhere (stops a response being MIME-sniffed into executable HTML). The
    # CSP ships in REPORT-ONLY mode: it never blocks, so it cannot break the GUI
    # (the inline shell-token script, the TTS CDN/HF fetches, the sandboxed
    # artifact iframe, workers), but it documents the intended policy and surfaces
    # any violation for the maintainer to resolve before a later coordinated flip
    # to an enforcing policy (which needs a nonce on the inline shell script).
    _CSP_REPORT_ONLY = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://huggingface.co https://*.hf.co "
        "https://cdn.jsdelivr.net; "
        "worker-src 'self' blob:; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )

    @app.middleware("http")
    async def _security_headers(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault(
            "Content-Security-Policy-Report-Only", _CSP_REPORT_ONLY)
        return resp

    # API-surface disclosure guard. FastAPI's built-in docs (/docs, /redoc,
    # /openapi.json) enumerate every route plus its request/response schema. That
    # is fine on a loopback bind (the local owner), but on a NETWORK bind it hands
    # an UNAUTHENTICATED remote client a full map of the attack surface - every
    # endpoint stays scope-gated, so it grants no access, but it is needless
    # disclosure and inconsistent with how /whoami hides root_dir off-loopback.
    # Serve the docs only on a loopback bind, keyed on the CONFIGURED bind host
    # (never the peer - portmux makes the peer always look loopback). 404, not
    # 403, so the endpoints simply do not exist off-loopback and reveal nothing.
    # bind_host is unset in tests / a standalone mount -> default loopback ->
    # docs stay available there, matching prior behaviour.
    _DOCS_PATHS = frozenset(
        {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"})

    @app.middleware("http")
    async def _docs_loopback_only(request, call_next):
        if request.url.path in _DOCS_PATHS:
            host = getattr(request.app.state, "bind_host", "127.0.0.1")
            if not _is_loopback_host(host):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return await call_next(request)

    # ---------------------------------------------------------------- #
    #  Route groups (extracted to localm/inference/routes/*.py)          #
    # ---------------------------------------------------------------- #
    # The engine + inference semaphore are module globals read live by the route
    # modules (via `import localm.inference.http_server as _hs`), so a model swap
    # that reassigns them is seen there. Only these session-scoped objects are
    # create_app locals, so they travel to the route groups on ctx. Registration
    # order is irrelevant: FastAPI matches exact path templates by method, and no
    # group has a same-method literal-vs-param path collision with another.
    from types import SimpleNamespace
    ctx = SimpleNamespace(audit=_audit, transcript=_transcript, mode=_mode)
    from localm.inference.routes import models as _routes_models
    _routes_models.register(app, ctx)
    from localm.inference.routes import system as _routes_system
    _routes_system.register(app, ctx)
    from localm.inference.routes import session as _routes_session
    _routes_session.register(app, ctx)
    from localm.inference.routes import plugins as _routes_plugins
    _routes_plugins.register(app, ctx)
    from localm.inference.routes import config as _routes_config
    _routes_config.register(app, ctx)
    from localm.inference.routes import keys as _routes_keys
    _routes_keys.register(app, ctx)
    from localm.inference.routes import admin as _routes_admin
    _routes_admin.register(app, ctx)
    from localm.inference.routes import chat as _routes_chat
    _routes_chat.register(app, ctx)

    # ---- plugin engine: load enabled plugins + management API ------- #
    # Wrapped so a plugin-engine failure can never stop the server starting.
    try:
        from localm.plugins.engine import attach_engine
        attach_engine(app, _engine)
    except Exception as e:
        # Server must still start, but make the loss visible: WARNING (not a buried
        # debug line) plus a sentinel so plugin_manager being unset is diagnosable.
        from localm.debuglog import logger as _dbg
        _dbg.warning("plugins unavailable: %s", e)
        _dbg.exception("plugin engine attach failed")
        app.state.plugin_engine_error = str(e)

    return app


# ------------------------------------------------------------------ #
#  Performance metric helpers                                          #
# ------------------------------------------------------------------ #

def _engine_finish_reason(engine) -> str:
    """Why the last generation ended - "stop" unless the backend reported a
    real string (mocks and minimal engines without the attribute count as stop)."""
    fr = getattr(engine, "last_finish_reason", "stop")
    return fr if isinstance(fr, str) else "stop"


def _ttft_ms(gen_start: float, first_token_at: Optional[float]) -> Optional[float]:
    """Time to first token in milliseconds, or None if nothing was generated."""
    if first_token_at is None:
        return None
    return round((first_token_at - gen_start) * 1000, 1)


def _tokens_per_sec(completion_tokens: int, elapsed: float) -> Optional[float]:
    """Generation throughput, or None when not measurable."""
    if not completion_tokens or elapsed <= 0:
        return None
    return round(completion_tokens / elapsed, 2)


# ------------------------------------------------------------------ #
#  SSE streaming                                                       #
# ------------------------------------------------------------------ #

def _last_user_text(messages: list) -> str:
    """Text of the most recent user message (for the audit trail)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in (content or [])
                            if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _messages_prompt_text(messages: list) -> str:
    """Flatten chat messages to plain text for prompt token counting (text parts
    of multimodal content included; non-text parts ignored)."""
    return " ".join(
        m.get("content") if isinstance(m.get("content"), str)
        else " ".join(p.get("text", "") for p in (m.get("content") or [])
                      if p.get("type") == "text")
        for m in messages
    )


def _audit_exchange(audit, transcript, messages: list, reply: str) -> None:
    """Record one chat exchange (log/full modes; no-op log in privacy)."""
    if audit is None:
        return
    try:
        user_text = _last_user_text(messages)
        audit.user(user_text)
        audit.llm(reply)
        if transcript is not None:
            transcript.exchange(user_text, reply)
    except Exception as e:
        # In log/full mode a failed write silently drops the record; surface it
        # so the gap is discoverable instead of invisible.
        from localm.debuglog import logger as _dbg
        _dbg.warning("audit/transcript write failed: %s; this exchange was not recorded", e)
        pass  # auditing must never break serving


def _reason_sse(content: str, reasoning: str,
                model_id: str, chunk_id: str, ts: int) -> list:
    """SSE ``data:`` lines for a (content, reasoning) split (H4). Reasoning is
    emitted before content (it precedes the answer); empty parts produce
    nothing, so an ordinary content-only token yields exactly one chunk."""
    from localm.inference.protocol import ChatChunk, ChoiceDelta, StreamChoice
    out = []
    for field, value in (("reasoning_content", reasoning), ("content", content)):
        if not value:
            continue
        chunk = ChatChunk(
            id=chunk_id, created=ts, model=model_id,
            choices=[StreamChoice(delta=ChoiceDelta(**{field: value}))],
        )
        out.append(f"data: {chunk.model_dump_json()}\n\n")
    return out


async def _stream_sse(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    from localm.inference.protocol import ChoiceDelta, StreamChoice
    from localm.inference.textnorm import ThinkSplitter

    chunk_id = make_chunk_id()
    ts = int(time.time())
    think = ThinkSplitter()   # route <think> reasoning into delta.reasoning_content (H4)

    prompt_tokens = engine.count_messages_tokens(messages)

    # Context Limit Handling: Trigger compact_messages if we are dangerously close to the limit.
    # We reserve a buffer of 2048 tokens for compaction overhead and response generation.
    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            new_messages, changed = compact_messages(messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = engine.count_messages_tokens(messages)

    # Role announcement
    role_chunk = ChatChunk(
        id=chunk_id,
        created=ts,
        model=model_id,
        choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    # Run blocking generator in executor so we don't block the event loop
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def _generate():
        try:
            for token in engine.chat_stream(messages, **gen_kwargs):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Log it (with full traceback, to the debug log) and surface it to the
            # client - a silent thread death looks like an empty reply. We deliberately
            # do NOT traceback.print_exc() here: _dbg.exception already records the
            # trace, and an expected condition (e.g. the conversation outgrew n_ctx_max)
            # should reach the user as a clean message, not a raw console traceback.
            # Printing it was also the historical WinError-6 crash source on Windows.
            from localm.debuglog import logger as _dbg
            _dbg.exception("generation thread failed")
            loop.call_soon_threadsafe(
                token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, _DONE)

    import threading

    # Serialise inference - only one request runs at a time
    async with sem:
        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        gen_error: Exception | None = None
        while True:
            token = await token_queue.get()
            if token is _DONE:
                break
            if isinstance(token, Exception):
                gen_error = token
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            # Stream hook transforms the piece before it is recorded and sent,
            # so usage reflects exactly what the client receives.
            if pipeline is not None and ctx is not None and pipeline.has("stream"):
                token = pipeline.run_stream(token, ctx)
            completion_parts.append(token)
            for data in _reason_sse(*think.feed(token), model_id, chunk_id, ts):
                yield data

        t.join()
        gen_elapsed = time.perf_counter() - gen_start
        # Release any tail held back while disambiguating a partial <think> tag.
        for data in _reason_sse(*think.flush(), model_id, chunk_id, ts):
            yield data

    if gen_error is not None:
        err_chunk = ChatChunk.token(
            f"\n[inference error: {gen_error}]", model_id, chunk_id, ts)
        yield f"data: {err_chunk.model_dump_json()}\n\n"

    streamed = "".join(completion_parts)
    # Outlet runs after every chunk has been sent, so it cannot alter the live
    # stream (a stream hook does that). Here it only shapes the recorded reply
    # (audit / transcript / side-effects); usage stays tied to what was streamed.
    reply = streamed
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        reply = await pipeline.run_outlet(streamed, messages, ctx)

    _audit_exchange(audit, transcript, messages, reply)

    # Count tokens on the streamed text - what the client actually received
    completion_tokens = engine.count_tokens(streamed)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(completion_tokens, gen_elapsed),
        context_capacity=engine.context_capacity(),
    )
    # Honesty: a generation that errored mid-stream must not report a clean
    # "stop" on the terminal frame. The error text was already streamed as a
    # visible chunk above, but a PROGRAMMATIC client keys off finish_reason and
    # would otherwise read the failure as a successful completion. Mark it
    # "error" so the failure is machine-detectable, not only human-readable.
    finish_reason = "error" if gen_error is not None else _engine_finish_reason(engine)
    done = ChatChunk.done(model_id, chunk_id, ts, usage=usage,
                          finish_reason=finish_reason)
    yield f"data: {done.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------ #
#  SSE streaming for /v1/completions                                   #
# ------------------------------------------------------------------ #

async def _stream_sse_completion(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
) -> AsyncIterator[str]:
    chunk_id = make_chunk_id()
    ts = int(time.time())
    # *messages* arrive already inlet-transformed; count tokens on what
    # inference sees (matches the chat path).
    prompt_tokens = engine.count_tokens(_messages_prompt_text(messages))

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()

    def _generate():
        try:
            for token in engine.chat_stream(messages, **gen_kwargs):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        except Exception as e:
            # Surface an inference failure to the client instead of letting this
            # daemon thread die - an uncaught thread death fires a crash report and
            # looks to the user like an empty reply. _dbg.exception logs the full
            # trace to the debug log (same contract as the chat-completions path).
            from localm.debuglog import logger as _dbg
            _dbg.exception("completion generation thread failed")
            loop.call_soon_threadsafe(token_queue.put_nowait, RuntimeError(str(e)))
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

    import threading

    async with sem:
        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        gen_error: Exception | None = None
        while True:
            token = await token_queue.get()
            if token is None:
                break
            if isinstance(token, Exception):
                gen_error = token
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            # Stream hook transforms each piece before it is recorded and sent,
            # so usage and the audit trail reflect what the client receives.
            if pipeline is not None and ctx is not None and pipeline.has("stream"):
                token = pipeline.run_stream(token, ctx)
            completion_parts.append(token)
            chunk = {
                "id": chunk_id, "object": "text_completion.chunk",
                "created": ts, "model": model_id,
                "choices": [{"text": token, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        t.join()
        gen_elapsed = time.perf_counter() - gen_start

    if gen_error is not None:
        err = {
            "id": chunk_id, "object": "text_completion.chunk",
            "created": ts, "model": model_id,
            "choices": [{"text": f"\n[inference error: {gen_error}]",
                         "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    streamed = "".join(completion_parts)
    # Outlet shapes only the recorded reply (the live stream already went out);
    # then record the exchange (audit + transcript), exactly like chat.
    reply = streamed
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        reply = await pipeline.run_outlet(streamed, messages, ctx)
    _audit_exchange(audit, transcript, messages, reply)

    completion_tokens = engine.count_tokens(streamed)
    # Honesty (mirrors the chat path): a mid-stream error is reported as "error",
    # not "stop", so a client keying off finish_reason detects the failure even
    # though the error text was already streamed as a visible chunk.
    done = {
        "id": chunk_id, "object": "text_completion.chunk",
        "created": ts, "model": model_id,
        "choices": [{"text": "", "index": 0,
                     "finish_reason": ("error" if gen_error is not None else "stop")}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_ms": _ttft_ms(gen_start, first_token_at),
            "tokens_per_sec": _tokens_per_sec(completion_tokens, gen_elapsed),
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------ #
#  Non-streaming completion                                            #
# ------------------------------------------------------------------ #

async def _complete(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    audit=None,
    transcript=None,
    pipeline=None,
    ctx=None,
    **gen_kwargs,
):
    loop = asyncio.get_running_loop()

    prompt_tokens = engine.count_messages_tokens(messages)

    capacity = engine.context_capacity()
    if capacity is not None and len(messages) > 3:
        buffer = max(2048, int(capacity * 0.10))
        if capacity - prompt_tokens < buffer:
            from localm.inference.compact import compact_messages
            def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
            new_messages, changed = compact_messages(messages, _gen_for_compact)
            if changed:
                messages = list(new_messages)
                prompt_tokens = engine.count_messages_tokens(messages)
    def _run():
        return "".join(engine.chat_stream(messages, **gen_kwargs))

    # Serialise inference - only one request runs at a time
    async with sem:
        gen_start = time.perf_counter()
        text = await loop.run_in_executor(None, _run)
        gen_elapsed = time.perf_counter() - gen_start

    # Outlet fully controls the returned content in the non-streaming path.
    if pipeline is not None and ctx is not None and pipeline.has("outlet"):
        text = await pipeline.run_outlet(text, messages, ctx)

    _audit_exchange(audit, transcript, messages, text)

    # Split the model's <think> reasoning out of the visible answer into a
    # separate field (H4), so API clients get clean content (token count stays on
    # the full generated text - reasoning was still generated).
    from localm.inference.textnorm import split_think
    answer, reasoning = split_think(text)

    completion_tokens = engine.count_tokens(text)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        tokens_per_sec=_tokens_per_sec(completion_tokens, gen_elapsed),
    )

    response = ChatResponse(
        id=make_chunk_id(),
        created=int(time.time()),
        model=model_id,
        choices=[
            FullChoice(
                message=Message(role="assistant", content=answer,
                                reasoning_content=reasoning or None),
                finish_reason=_engine_finish_reason(engine),
            )
        ],
        usage=usage,
    )
    return JSONResponse(response.model_dump())


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _protocol_messages_to_dicts(messages: List[Message]) -> list:
    """Convert Pydantic Message objects to plain dicts for backends."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
        else:
            parts = []
            for part in msg.content:
                if hasattr(part, "text"):
                    parts.append({"type": "text", "text": part.text})
                elif hasattr(part, "image_url"):
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url},
                    })
                elif hasattr(part, "input_audio"):
                    parts.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": part.input_audio.data,
                            "format": part.input_audio.format,
                        },
                    })
            result.append({"role": msg.role, "content": parts})
    return result


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8642,
          ssl_certfile: Optional[str] = None,
          ssl_keyfile: Optional[str] = None,
          project: Optional[str] = None,
          isolated: bool = False) -> None:
    """Start the server - blocks until Ctrl+C.

    When ``ssl_certfile`` / ``ssl_keyfile`` are given (built-in TLS, NET-1), the
    server speaks HTTPS on this port; a plain-HTTP request to it then fails the
    TLS handshake (effectively refused) rather than crossing the network in
    cleartext.

    Advertises itself in the instance registry (H6 phase 3/4) as an ``api``
    surface so a future launch can discover and attach to it; ``isolated`` keeps
    it invisible to discovery.
    """
    from localm import instances, portmux
    from localm.config import home_dir

    app = create_app(engine, api_landing=True)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    scheme = "https" if ssl_certfile else "http"

    # SRV-CTRLC: no custom Win32 Ctrl+C handler here. A previous one resolved the
    # loop with get_event_loop_policy().get_event_loop() on the control-handler OS
    # thread - which is NOT the serving loop (portmux's asyncio.run makes a fresh
    # loop with no set_event_loop), so loop.stop() never fired - yet it returned
    # True, consuming the event and DEFEATING uvicorn's own SIGINT/KeyboardInterrupt
    # shutdown too. Net effect: it ATE Ctrl+C and the server hung. Removing it lets
    # Ctrl+C flow through uvicorn's standard signal handling (KeyboardInterrupt is
    # caught in portmux.run_server), which portmux's SRV-6 loop-wakeup task keeps
    # responsive on Windows. Verified live (Ctrl+Break) that the server now stops
    # instead of no-opping; a graceful vs abrupt human Ctrl+C is uvicorn's own path.

    with instances.advertise(app, home_dir(), host=host, port=port, mode="api",
                             scheme=scheme, project=project, isolated=isolated):
        # On a TLS bind, also catch a plain-http request on the same port with an
        # https redirect (issue 8); plain binds are a direct uvicorn.run. SRV-5:
        # in debug mode uvicorn logs at "info" so the console shows requests.
        from localm.debuglog import uvicorn_log_level
        portmux.run_server(app, host=host, port=port,
                           log_level=uvicorn_log_level(),
                           ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)

