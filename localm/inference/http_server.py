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
import hmac
import json
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPBearer

from localm import scopes
from localm.inference.backends.base import (
    messages_contain_image,
)
from localm.inference.chat_pipeline import ChatHookContext, ChatPipeline
from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatRequest, ChatResponse, CompletionRequest, EmbeddingRequest,
    FullChoice, Message, UsageInfo, make_chunk_id,
)

# Global engine reference set by serve()
_engine: Engine | None = None

# Inference serialisation - only one request runs inference at a time.
# Additional requests queue behind this semaphore.
_inference_sem: asyncio.Semaphore | None = None

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

# S2: the browser GUI authenticates with an HttpOnly session cookie (the API key
# the page JS can no longer read) instead of localStorage. Cookie-sourced auth on
# a state-changing method must echo the readable CSRF cookie in this header
# (double-submit); the Authorization-header path (CLI / SDK / coder) cannot be
# forged cross-site and is therefore CSRF-exempt.
SESSION_COOKIE = "localm_session"
CSRF_COOKIE = "localm_csrf"
CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# SEAMLESS: the auth cookies PERSIST so the user stays signed in across a browser
# or PWA restart, instead of being session cookies the browser drops on close
# (which made the key gate - and its "Install certificate" step - reappear every
# time, the "install a new cert / re-enter the key every restart" report).
# Browsers clamp cookie lifetime to ~400 days, so we ask for that ceiling; the
# escape hatch is the existing /api/session/logout (Settings: leave the key blank
# and Save). Both cookies MUST share this lifetime, or the readable CSRF token
# would expire before the session and bounce the user mid-use.
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


def _csrf_double_submit_ok(request) -> bool:
    """Double-submit CSRF check: the ``X-CSRF-Token`` header must equal the
    readable ``localm_csrf`` cookie (both set at login). A cross-site page can
    neither read the cookie to forge the header nor (with SameSite=Strict) get
    the browser to send the cookie at all - defence in depth."""
    header = request.headers.get(CSRF_HEADER, "")
    cookie = request.cookies.get(CSRF_COOKIE, "")
    return bool(header and cookie and hmac.compare_digest(header, cookie))


def _enforce_request(request: Request, scope: Optional[str]) -> None:
    """Shared auth core (request-aware). Open/dev mode when no key is configured
    anywhere (unless LOCALM_REQUIRE_AUTH forces fail-closed). Otherwise a valid
    key is required - from the Authorization header OR the session cookie - and
    *scope* (None = 'any valid key') must be granted (the owner key implies every
    scope). Cookie-sourced auth on an unsafe method additionally requires a valid
    double-submit CSRF token."""
    from localm.auth import any_key_configured, require_auth_enabled, verify
    if not any_key_configured():
        if require_auth_enabled():
            raise HTTPException(
                status_code=401,
                detail="Auth required but no API key configured "
                       "(set one via the launcher or LOCALM_API_KEY)")
        return  # open/dev mode
    token, source = _request_token(request)
    held = verify(token) if token else None
    if held is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if source == "cookie" and request.method not in _SAFE_METHODS:
        if not _csrf_double_submit_ok(request):
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
    from localm.auth import any_key_configured, verify
    if not any_key_configured():
        return None
    token, _ = _request_token(request)
    return verify(token) if token else None


def principal_id(request: Request) -> Optional[str]:
    """A stable, opaque per-key identity for the CURRENT caller, or None in open
    mode / when no token is presented. It is the same SHA-256 the keystore stores
    (never the plaintext key), so it identifies the key WITHOUT exposing it, and
    is identical whether the key arrives via the Authorization header or the
    session cookie. Used to bind a background job to the key that created it
    (KEY-SCOPE-2), so only that key (or an admin/owner) may stream or cancel it."""
    from localm.auth import _hash_key
    token, _ = _request_token(request)
    if not token or not token.strip():
        return None
    return _hash_key(token.strip())


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

    async def switch_model(name: str) -> None:
        global _engine
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
        new_engine = Engine(
            str(m_path),
            display_name=name if name in load_registry() else m_hint,
            mmproj_path=mmproj,
        )
        loop = asyncio.get_running_loop()
        async with _inference_sem:
            old = _engine
            if old is not None and old.loaded:
                await loop.run_in_executor(None, old.unload)
            await loop.run_in_executor(None, new_engine.load)
            _engine = new_engine

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
        dump_ring_buffer()
        # Also write a clear text log for the bug reporter to ingest directly,
        # in case the JSON buffer fails to load back into memory.
        from localm.config import home_dir
        pre_log = home_dir() / "logs" / "pre_restart.log"
        pre_log.parent.mkdir(parents=True, exist_ok=True)
        pre_log.write_text("\n".join(recent_activity()), encoding="utf-8")
        # Flush all log handlers before os.execv so no buffered lines are lost
        # (Task 1: log durability / save-bug).
        flush_log_handlers()
    except Exception:
        pass

    import os
    import sys
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
        # SRV-3: route an uncaught asyncio task exception through the bug reporter
        # instead of a silent "Task exception was never retrieved". Skipped under
        # pytest so the test runner keeps its own loop handling.
        if "pytest" not in sys.modules:
            try:
                from localm import bugreport
                bugreport.install_asyncio_handler(asyncio.get_running_loop())
            except Exception:
                pass
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
        version="0.1.0",
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
        if (request.method in _UNSAFE_METHODS and not _cors_wildcard
                and not request.url.path.startswith(_CROSS_ORIGIN_OK)):
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

    # ---------------------------------------------------------------- #
    #  Health                                                            #
    # ---------------------------------------------------------------- #

    @app.get("/health")
    async def health():
        if _engine is None:
            raise HTTPException(503, "No engine initialised")
        return {
            "status": "ok",
            "model":  _engine.display_name,
            "loaded": _engine.loaded,
        }

    # ---------------------------------------------------------------- #
    #  Session auth (S2): HttpOnly cookie + CSRF for the browser GUI    #
    # ---------------------------------------------------------------- #

    @app.post("/api/session", include_in_schema=False)
    async def session_login(request: Request, response: Response):
        """Exchange the API key for an HttpOnly session cookie so the browser
        GUI never has to hold the key in JS-readable localStorage. The key is
        read from the JSON body ``{"key": ...}`` or an Authorization: Bearer
        header, verified, and on success set as the HttpOnly ``localm_session``
        cookie plus a readable ``localm_csrf`` cookie (double-submit CSRF). 401
        on a bad key; 400 in open mode (nothing to log into)."""
        from localm.auth import any_key_configured, verify
        presented: Optional[str] = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                presented = (body.get("key") or "").strip() or None
        except Exception:
            presented = None
        if not presented:
            presented = _bearer_token(request)
        if not any_key_configured():
            raise HTTPException(
                400, "This server runs in open mode (no API key configured); "
                "there is nothing to log into.")
        held = verify(presented) if presented else None
        if held is None:
            raise HTTPException(401, "Invalid API key")
        secure = request.url.scheme == "https"
        csrf = secrets.token_urlsafe(32)
        response.set_cookie(SESSION_COOKIE, presented, httponly=True,
                            secure=secure, samesite="strict", path="/",
                            max_age=SESSION_MAX_AGE)
        response.set_cookie(CSRF_COOKIE, csrf, httponly=False,
                            secure=secure, samesite="strict", path="/",
                            max_age=SESSION_MAX_AGE)
        return {"authed": True, "scopes": sorted(held)}

    @app.post("/api/session/logout", include_in_schema=False)
    async def session_logout(request: Request, response: Response):
        """Clear the session + CSRF cookies (sign the browser out).

        A POST from the cookie-authenticated browser GUI is subject to the
        same double-submit CSRF check as every other state-changing endpoint
        (Task 3: CSRF-post-clear). A bearer-header caller (CLI / SDK) is
        CSRF-exempt because it cannot be driven cross-site."""
        token, source = _request_token(request)
        if source == "cookie" and not _csrf_double_submit_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Include the localm_csrf "
                "cookie value in the X-CSRF-Token header.")
        secure = request.url.scheme == "https"
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=secure, samesite="strict")
        response.delete_cookie(CSRF_COOKIE, path="/", httponly=False, secure=secure, samesite="strict")
        return {"authed": False}

    @app.post("/api/auth/key/clear",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))],
              include_in_schema=False)
    async def clear_owner_key(request: Request, response: Response):
        """Delete the server-side owner key (auth.key) and immediately
        invalidate the caller's session cookie.

        CSRF-protected: a cookie-authenticated caller must echo the
        localm_csrf cookie value in X-CSRF-Token (double-submit pattern).
        After this call any_key_configured() returns False (assuming no
        LOCALM_API_KEY env var and an empty keystore), and all subsequent
        web UI requests that rely on the session cookie will fail auth and
        be redirected to the key gate by the client (Task 3: CSRF-post-clear).
        """
        _, source = _request_token(request)
        if source == "cookie" and not _csrf_double_submit_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Include the localm_csrf "
                "cookie value in the X-CSRF-Token header.")
        from localm.auth import clear_api_key
        clear_api_key()
        secure = request.url.scheme == "https"
        # Invalidate the session cookie immediately so the browser is forced
        # back to the key gate on the next navigation (the old cookie value
        # no longer matches any key). Also clear the CSRF token.
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True,
                               secure=secure, samesite="strict")
        response.delete_cookie(CSRF_COOKIE, path="/", httponly=False,
                               secure=secure, samesite="strict")
        from localm.debuglog import logger as _dbg
        _dbg.info("owner API key cleared via /api/auth/key/clear; session invalidated")
        return {"cleared": True}

    @app.get("/api/session", include_in_schema=False)
    async def session_state(request: Request):
        """Report whether the caller is authenticated, so the GUI can show or
        hide its key gate without ever reading the key. *authed* reflects the
        presented cookie/header; *required* is True when a key is configured."""
        from localm.auth import (any_key_configured, require_auth_enabled,
                                  verify)
        configured = any_key_configured()
        token, _ = _request_token(request)
        held = verify(token) if (configured and token) else None
        return {"authed": held is not None,
                "scopes": sorted(held) if held else [],
                "required": configured or require_auth_enabled()}

    # ---------------------------------------------------------------- #
    #  Instance identity (H6 server-rework, phase 3)                    #
    # ---------------------------------------------------------------- #

    @app.get("/whoami", include_in_schema=False)
    async def whoami(request: Request):
        """Identity handshake for instance discovery: confirms this really is a
        localm server and which instance/project it serves. Unauthenticated like
        /health; never returns the attach token or pid. Fields are set on
        app.state by the surface's advertise() wrapper (None before startup
        wiring, e.g. an app mounted standalone in tests).

        root_dir is an absolute host path that can carry the OS username, so it is
        disclosed only on a loopback bind and omitted over the network. Discovery
        matches root_dir from the 0600 registry file (not from /whoami), so this
        omission breaks nothing (security review 2026-06-20)."""
        from localm import instances
        st = request.app.state
        root = getattr(st, "root_dir", None)
        if root is not None:
            bind = getattr(st, "bind_host", "127.0.0.1")
            loopback = bind in ("127.0.0.1", "localhost", "::1")
            if not loopback:
                try:
                    import ipaddress
                    loopback = ipaddress.ip_address(bind).is_loopback
                except ValueError:
                    loopback = False
            if not loopback:
                root = None
        return instances.whoami_payload(
            instance_id=getattr(st, "instance_id", None),
            root_dir=root,
            mode=getattr(st, "instance_mode", None),
        )

    # ---------------------------------------------------------------- #
    #  Surface management: on-demand GUI mount (H6 phase 5)              #
    # ---------------------------------------------------------------- #

    @app.post("/v1/surfaces/gui", include_in_schema=False)
    async def mount_gui(request: Request):
        """Mount the GUI surface on this running instance (the phase-5 on-demand
        mount). An ``api``-mode instance (``localm serve``) serves only /v1; a
        later ``localm gui`` in the same dir calls this to add the GUI + coder
        live - one process, no second model load - then opens it.

        Mounting the GUI exposes the coder agent (shell + file edits), so this is
        an OWNER-level action. Authorized by EITHER this instance's attach token
        (the local same-user secret in the 0600 ``run/`` file, which the
        attaching process reads) OR an API key granting ADMIN. It is exempt from
        the same-origin guard (a local non-browser caller has no Origin); this
        token/key check is the gate. Idempotent: a full instance returns
        already_mounted."""
        presented = _bearer_token(request)
        st = request.app.state
        inst_token = getattr(st, "instance_token", None)
        ok = bool(presented and inst_token
                  and hmac.compare_digest(presented, inst_token))
        if not ok and presented:
            # Fall back to an owner/ADMIN API key (protected mode).
            from localm.auth import any_key_configured, verify
            if any_key_configured():
                held = verify(presented)
                ok = held is not None and scopes.grants(held, scopes.ADMIN)
        if not ok:
            raise HTTPException(
                403, "Surface management requires this instance's attach token "
                "or an owner API key.")
        mounted = mount_gui_surface(request.app)
        return {"status": "mounted" if mounted else "already_mounted",
                "mode": "full"}

    # ---------------------------------------------------------------- #
    #  Built-in TLS: CA download (NET-1)                                 #
    # ---------------------------------------------------------------- #

    @app.get("/localm-ca.crt", include_in_schema=False)
    async def localm_ca_cert():
        """Serve localm's local CA certificate so a browser or phone can trust
        the built-in TLS once - removing the warning and enabling PWA install.
        Deliberately public and unauthenticated: a CA *certificate* carries no
        secret (the CA private key never leaves ``<home>/tls``), and the client
        needs it before it can present a key. 404 when this install has no CA
        (e.g. a loopback / plain-HTTP run that never generated one)."""
        from localm import tls
        from localm.config import home_dir
        ca = tls.ca_cert_path(home_dir())
        if not ca.is_file():
            raise HTTPException(404, "No localm CA on this server (TLS not enabled).")
        return Response(
            content=ca.read_bytes(),
            media_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": 'attachment; filename="localm-ca.crt"'},
        )

    # ---------------------------------------------------------------- #
    #  Models list                                                       #
    # ---------------------------------------------------------------- #

    @app.get("/v1/models",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def list_models():
        # The GUI can run with no engine yet (fresh install, empty registry -
        # the user adds a model from the Models page). Report an empty list.
        if _engine is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [
                {
                    "id":       _engine.display_name,
                    "object":   "model",
                    "created":  int(time.time()),
                    "owned_by": "localm",
                    "loaded":   _engine.loaded,
                }
            ],
        }

    @app.get("/v1/models/{model_id}",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def model_detail(model_id: str):
        """Registry metadata for one model: path, source, size, hash, aliases."""
        from localm.config import load_registry
        registry = load_registry()
        entry = registry.get(model_id)
        if entry is None:
            raise HTTPException(404, f"Model not registered: {model_id}")
        path = entry.get("path", "")
        p = Path(path)
        size = None
        try:
            if p.is_file():
                size = p.stat().st_size
            elif p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except OSError:
            pass
        aliases = sorted(
            n for n, e in registry.items()
            if e.get("path") == path and n != model_id
        )
        return {
            "id": model_id,
            "object": "model",
            "owned_by": "localm",
            "path": Path(path).name if path else "",  # basename only; never leak the absolute path
            "source": entry.get("source", ""),
            "sha256": entry.get("sha256"),
            "size_bytes": size,
            "aliases": aliases,
            "active": _engine is not None and _engine.display_name == model_id,
            "loaded": _engine is not None
                      and _engine.display_name == model_id and _engine.loaded,
        }

    # ---------------------------------------------------------------- #
    #  Plugins                                                           #
    # ---------------------------------------------------------------- #

    @app.get("/v1/plugins", dependencies=[Depends(require_scope(scopes.PLUGINS_READ))])
    async def list_plugins():
        from localm.plugins.loader import discover_errors, discover_plugins
        return {
            "plugins": [
                {
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "entry": m.entry,
                    "path": str(m.path),
                    "tool_exports": m.tool_exports,
                }
                for m in discover_plugins()
            ],
            "errors": discover_errors(),
        }

    @app.post("/v1/server/shutdown",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_shutdown_ep():
        """SRV-4: stop this server cleanly (owner / config-write scope). A direct
        method to shut down so the user is not left force-closing the window
        (which segfaults) or relying on a Ctrl+C that sometimes does nothing. The
        model is unloaded before exit. (A Settings button calls this - Lane E.)"""
        _request_shutdown()
        return {"stopping": True}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """R18: restart this server in place (owner / config-write scope). The model
        is unloaded first, then the process re-execs the same command line and comes
        back on the same port - a Settings button calls this, and the GUI's reconnect
        overlay auto-reconnects when the fresh process is up."""
        _request_restart()
        return {"restarting": True}

    @app.post("/api/bug-report",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def file_bug_report_ep(body: dict):
        """R47: file a bug report from the GUI. The CLI has `localm bug-report`, but
        the GUI had no manual trigger. Saves an editable markdown report (a safe
        environment snapshot plus the user's note - never keys/config/chat data;
        with ``include_log`` the home-scrubbed tail of the current run's log) and
        returns its path so the GUI can point the user at the file to edit/send.
        Owner / config-write scoped and same-origin gated like the other management
        routes (a report can carry local diagnostics)."""
        from localm import bugreport
        # The GUI "Report a bug" button sends ``description``; ``message`` is
        # accepted as an alias so the documented payload and CLI-shaped callers
        # both work. Single canonical endpoint (the GUI router does not duplicate
        # it - that would shadow this one and drop the user's text + log flag).
        description = (body.get("description") or body.get("message") or "").strip()
        if not description:
            raise HTTPException(400, "Please describe the problem before sending.")
        # Optional browser context the GUI attaches (user agent, page, viewport,
        # recent JS console errors). Untrusted client input: take only known
        # fields, coerce to strings, and cap sizes so a crafted payload cannot
        # bloat the report. It is rendered as plain text (markdown code fence),
        # never executed.
        client = _sanitize_client_context(body.get("client"))
        path = bugreport.save_user_report(
            description, include_log=bool(body.get("include_log")), client=client)
        if path is None:
            # A failed save must not report success (we do not hide problems).
            raise HTTPException(500, "Could not save the bug report to disk.")
        return {"saved": True, "filename": path.name, "path": str(path),
                "maintainer": bugreport.MAINTAINER_EMAIL}

    @app.post("/v1/plugins/install", dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_ep(body: dict):
        from pathlib import Path as _P
        from localm.plugins.loader import PluginError, install_plugin
        source = body.get("source", "")
        if not source:
            raise HTTPException(400, "Missing 'source' (local directory path)")
        try:
            manifest = install_plugin(_P(source), force=bool(body.get("force")))
        except PluginError as e:
            raise HTTPException(400, str(e))
        return {"status": "installed", "name": manifest.name,
                "version": manifest.version}

    @app.delete("/v1/plugins/{name}", dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def remove_plugin_ep(name: str):
        from localm.plugins.loader import PluginError, remove_plugin
        try:
            existed = remove_plugin(name)
        except PluginError as e:
            raise HTTPException(400, str(e))
        if not existed:
            raise HTTPException(404, f"Plugin not installed: {name}")
        return {"status": "removed", "name": name}

    # ---------------------------------------------------------------- #
    #  Config                                                            #
    # ---------------------------------------------------------------- #

    @app.get("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config():
        from localm.config import load_config
        from localm.audit import effective_mode
        cfg = load_config()
        # Read-only extras for the frontend (skipped by the settings form).
        # The server mode is fixed at startup (the audit log is opened then);
        # the coder default is resolved per new session.
        cfg["effective_mode"] = _mode.value
        cfg["effective_coder_mode"] = effective_mode("coder").value
        # Resolved context ceiling (VRAM-derived when ctx_auto) - the GUI
        # bases its compaction threshold on this, not the static config.
        eff_ctx = getattr(_engine, "effective_ctx_max", None) if _engine else None
        cfg["effective_ctx_max"] = eff_ctx if isinstance(eff_ctx, int) else None
        return cfg

    @app.get("/v1/config/schema",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config_schema():
        """The typed settings schema (widget/label/help/group/owner/options/
        min/max) with each non-secret field's CURRENT value injected as its
        `default`, so the GUI can render the right control pre-filled. Secret
        fields never carry a value (schema_json omits secret defaults)."""
        from localm.config import load_config
        from localm.settings_schema import schema_json
        return {"fields": schema_json(values=load_config())}

    @app.patch("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def patch_config(body: dict):
        """Update known config keys and persist. Unknown keys are rejected.

        The read-only extras the GET handler injects (effective_mode etc.) are
        dropped first, so a client that round-trips the whole config object is
        not rejected for echoing back values it never edited."""
        from localm.config import load_config, save_config
        from localm.settings_schema import validate_update
        readonly = {"effective_mode", "effective_coder_mode", "effective_ctx_max"}
        body = {k: v for k, v in body.items() if k not in readonly}
        try:
            validated = validate_update(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # SEC-3: refuse to enable require_auth while no API key exists. Doing so
        # is a one-way self-lockout: the very next keyless request 401s and the
        # GUI sends no Bearer, so the toggle could never be undone from the GUI.
        # Only block ENABLING it (turning it off or unrelated edits are fine);
        # the auth-state check belongs here, not in the static schema validator.
        if validated.get("require_auth") is True:
            from localm.auth import any_key_configured
            if not any_key_configured():
                raise HTTPException(
                    400,
                    "Cannot enable require_auth while no API key is configured: "
                    "this would lock you out. Set an owner key (the launcher or "
                    "LOCALM_API_KEY) or create a named key first, then enable it.")
        cfg = load_config()
        cfg.update(validated)
        save_config(cfg)
        return cfg

    # ---------------------------------------------------------------- #
    #  Per-plugin media config (image / music / video)                   #
    # ---------------------------------------------------------------- #

    @app.get("/v1/media/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_media_config():
        """Per-plugin media (ComfyUI) config for image/music/video, each with its
        editable fields and RESOLVED values (the per-plugin block value, else the
        shared global comfy_* fallback). The GUI 'Media' section renders one
        subsection per plugin so the three are configured independently."""
        from localm.config import load_config
        from localm.settings_schema import MEDIA_PLUGINS, media_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        labels = {"image": "Image", "music": "Music", "video": "Video"}
        out = []
        for name in MEDIA_PLUGINS:
            block = plugins.get(name) if isinstance(plugins.get(name), dict) else {}
            out.append({"plugin": name, "label": labels[name],
                        "fields": media_schema_json(name, block, cfg)})
        return {"plugins": out}

    @app.post("/v1/media/config/{name}",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_media_config(name: str, body: dict):
        """Save ONE media plugin's own config block, deep-merged so the other
        fields and the other plugins are untouched. A blank field clears that
        plugin's override (it falls back to the shared global default)."""
        from localm.config import load_config, update_config
        from localm.settings_schema import (MEDIA_PLUGINS, media_schema_json,
                                             validate_media_block)
        if name not in MEDIA_PLUGINS:
            raise HTTPException(404, f"unknown media plugin: {name}")
        try:
            merge = validate_media_block(name, body or {})
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _deep_merge(dst: dict, src: dict) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_merge(dst[k], v)
                else:
                    dst[k] = v

        def _mutate(cfg: dict) -> None:
            plugins = cfg.get("plugins")
            if not isinstance(plugins, dict):
                plugins = cfg["plugins"] = {}
            block = plugins.get(name)
            if not isinstance(block, dict):
                block = plugins[name] = {}
            _deep_merge(block, merge)

        update_config(_mutate)
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        return {"plugin": name, "fields": media_schema_json(name, block, cfg)}

    @app.get("/v1/comfy/status", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_comfy_status():
        """Returns the alive status of the ComfyUI server."""
        from localm.image_gen.comfy import _comfy_alive, _comfy_api_url
        alive = _comfy_alive(_comfy_api_url(), timeout=1.0)
        return {"alive": alive}

    # ---------------------------------------------------------------- #
    #  API keys - scoped keystore (auth.json)                            #
    # ---------------------------------------------------------------- #

    @app.get("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def list_keys_ep(caller: Optional[set] = Depends(caller_scopes)):
        from localm.auth import list_keys
        from localm.config import load_config
        # is_owner lets the GUI hide owner-only (privileged) scope choices from a
        # mere keys:admin device; presets ride along so a keys:admin key WITHOUT
        # config:read can still see the quick-select bundles.
        is_owner = caller is not None and scopes.ADMIN in caller
        presets = load_config().get("key_presets", [])
        return {"keys": list_keys(), "is_owner": is_owner, "presets": presets}

    @app.post("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def create_key_ep(body: dict, caller: Optional[set] = Depends(caller_scopes)):
        from localm.auth import create_key
        scope_list = body.get("scopes", [])
        if not isinstance(scope_list, list):
            raise HTTPException(400, "'scopes' must be a list of scope strings")
        # Prefer a server-computed deadline from a relative TTL (expires_in seconds)
        # so a key's lifetime never depends on the client's clock; fall back to an
        # absolute epoch (expires) for raw API clients that want a specific deadline.
        expires = body.get("expires")
        expires_in = body.get("expires_in")
        if expires_in is not None:
            try:
                expires = time.time() + float(expires_in)
            except (TypeError, ValueError):
                raise HTTPException(400, "'expires_in' must be seconds (a number)")
        elif expires is not None:
            try:
                expires = float(expires)
            except (TypeError, ValueError):
                raise HTTPException(400, "'expires' must be an epoch number or null")
        # Only an owner/ADMIN caller may grant privileged scopes; a key holding
        # merely keys:admin must not be able to mint itself owner-equivalent
        # access. In open mode (caller is None) privileged grants are refused.
        is_owner = caller is not None and scopes.ADMIN in caller
        try:
            created = create_key(body.get("name", ""), scope_list,
                                 allow_privileged=is_owner, expires=expires)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        # The plaintext key is returned exactly once - it is never recoverable.
        return created

    @app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def revoke_key_ep(key_id: str):
        from localm.auth import revoke_key
        if not revoke_key(key_id):
            raise HTTPException(404, f"No such key: {key_id}")
        return {"status": "revoked", "id": key_id}

    # ---------------------------------------------------------------- #
    #  Model lifecycle - unload / load                                   #
    # ---------------------------------------------------------------- #

    @app.post("/v1/models/unload", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def unload_model():
        """
        Release the model from GPU/CPU memory.

        Call this before starting a VRAM-intensive task (e.g. ComfyUI FLUX
        generation) so the GPU memory is fully available.  The next call to
        /v1/chat/completions will reload the model automatically.
        """
        if _engine is None:
            raise HTTPException(503, "No engine initialised")
        if not _engine.loaded:
            return {"status": "already_unloaded", "model": _engine.display_name}
        loop = asyncio.get_running_loop()
        # Under the inference semaphore: freeing the native context while a
        # generation is mid-decode crashes the GPU driver (access violation
        # in the HIP runtime). Unload must wait its turn.
        from localm.discover import vram_info
        from localm.vram import wait_for_vram_release

        def _free():
            return vram_info().get("free")

        async with _inference_sem:
            before = _free()
            await loop.run_in_executor(None, _engine.unload)
            # The native context free is deferred: do NOT return until VRAM has
            # actually dropped, or a caller (e.g. ComfyUI media gen) loads its
            # model on top of the not-yet-freed weights, exceeds total VRAM and
            # hangs the GPU driver (TDR). Best-effort: a no-op when VRAM is
            # unmeasurable (before is None) or already freed.
            released, after = await loop.run_in_executor(
                None, lambda: wait_for_vram_release(_free, before_bytes=before))
        result = {"status": "unloaded", "model": _engine.display_name}
        if before is not None:
            result.update(vram_freed=released,
                          vram_before_bytes=before, vram_after_bytes=after)
        return result

    @app.post("/v1/models/load", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def load_model():
        """
        Explicitly reload the model into memory.

        Normally you don't need this - /v1/chat/completions reloads
        automatically if the model was unloaded.  Use this endpoint if you
        want to pre-warm the model before the first inference request.
        """
        if _engine is None:
            raise HTTPException(503, "No engine initialised")
        if _engine.loaded:
            return {"status": "already_loaded", "model": _engine.display_name}
        loop = asyncio.get_running_loop()
        async with _inference_sem:
            await loop.run_in_executor(None, _engine.load)
        return {"status": "loaded", "model": _engine.display_name}

    # ---------------------------------------------------------------- #
    #  Chat completions                                                  #
    # ---------------------------------------------------------------- #

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_auth)])
    async def chat_completions(req: ChatRequest, request: Request):
        if _engine is None:
            raise HTTPException(503, "No model loaded")
        _touch_activity()

        # Convert pydantic Messages to plain dicts for the backend
        messages = _protocol_messages_to_dicts(req.messages)

        # Chat-pipeline hooks. The inlet runs here (so token counting and
        # inference see the transformed messages); the per-request context is
        # built whenever a pipeline is present so the inlet, stream, and outlet
        # of this turn share one ctx. A pipeline with no hooks costs nothing.
        pipeline = getattr(request.app.state, "chat_pipeline", None)
        ctx = None
        if pipeline is not None:
            ctx = ChatHookContext(
                model_id=req.model, stream=req.stream,
                request_id=make_chunk_id(),
            )
            if pipeline.has("inlet"):
                messages = await pipeline.run_inlet(messages, ctx)

        # Reject image input on a text-only model with a clear 400 instead of
        # silently dropping the picture. For GGUF (always text-only) this is
        # known immediately; for an unloaded HF model multimodal support is
        # only known after loading, so load first before deciding.
        if messages_contain_image(messages) and not _engine.supports_images:
            if not _engine.loaded and _engine.can_be_multimodal:
                loop = asyncio.get_running_loop()
                async with _inference_sem:
                    await loop.run_in_executor(None, _engine.load)
            if not _engine.supports_images:
                # Capability-aware, install-specific guidance: route to a vision
                # model this install actually has, instead of a flat dead-end.
                # supports_images is False here, so if an mmproj_path is set on the
                # backend the projector FAILED to load (a working one would have made
                # supports_images True) - report that honest cause, not "text-only"
                # and not the stale "GGUF vision is not implemented".
                from localm.model_manager import vision_input_guidance
                mmproj_failed = bool(
                    getattr(getattr(_engine, "_backend", None), "mmproj_path", None))
                raise HTTPException(400, vision_input_guidance(mmproj_failed=mmproj_failed))

        gen_kwargs = dict(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            grammar=req.grammar,
            seed=req.seed,
        )
        # Strip None so Engine uses its config defaults
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if req.stream:
            return StreamingResponse(
                _stream_sse(_engine, messages, req.model, _inference_sem,
                            audit=_audit, transcript=_transcript,
                            pipeline=pipeline, ctx=ctx, **gen_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _complete(_engine, messages, req.model, _inference_sem,
                                   audit=_audit, transcript=_transcript,
                                   pipeline=pipeline, ctx=ctx, **gen_kwargs)

    # ---------------------------------------------------------------- #
    #  Embeddings  (/v1/embeddings)                                     #
    # ---------------------------------------------------------------- #

    @app.post("/v1/embeddings", dependencies=[Depends(_require_auth)])
    async def embeddings(req: EmbeddingRequest):
        if _engine is None:
            raise HTTPException(503, "No model loaded")
        _touch_activity()

        # Honor the OpenAI encoding_format contract. "float" returns plain JSON
        # arrays; "base64" returns each vector as a base64-encoded little-endian
        # float32 buffer (FAC-9). Anything else is rejected up front rather than
        # silently downgraded to float, which would mislead the client.
        fmt = (req.encoding_format or "float").lower()
        if fmt not in ("float", "base64"):
            raise HTTPException(
                400,
                f"Unsupported encoding_format {req.encoding_format!r}: "
                "expected 'float' or 'base64'.")

        texts = [req.input] if isinstance(req.input, str) else req.input

        loop = asyncio.get_running_loop()
        try:
            async with _inference_sem:
                vecs = await loop.run_in_executor(None, lambda: _engine.embed(texts))
        except NotImplementedError as e:
            raise HTTPException(422, str(e))

        def _encode(vec):
            if fmt == "base64":
                import base64
                import struct
                buf = struct.pack("<%df" % len(vec), *(float(x) for x in vec))
                return base64.b64encode(buf).decode("ascii")
            return vec

        total_tokens = sum(_engine.count_tokens(t) for t in texts)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": _encode(vec)}
                for i, vec in enumerate(vecs)
            ],
            "model": req.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }

    # ---------------------------------------------------------------- #
    #  Raw text completions  (/v1/completions)                          #
    # ---------------------------------------------------------------- #

    @app.post("/v1/completions", dependencies=[Depends(_require_auth)])
    async def completions(req: CompletionRequest, request: Request):
        if _engine is None:
            raise HTTPException(503, "No model loaded")
        _touch_activity()

        # Wrap the prompt as a single user message so raw completions flow through
        # the SAME chat-pipeline hooks and audit/transcript as
        # /v1/chat/completions (B17). Otherwise this is a parallel, unguarded,
        # unaudited path to the model: plugin safety/transform inlet/stream/outlet
        # hooks never fire and nothing is recorded.
        messages = [{"role": "user", "content": req.prompt}]

        pipeline = getattr(request.app.state, "chat_pipeline", None)
        ctx = None
        if pipeline is not None:
            ctx = ChatHookContext(
                model_id=req.model, stream=req.stream,
                request_id=make_chunk_id(),
            )
            if pipeline.has("inlet"):
                messages = await pipeline.run_inlet(messages, ctx)

        gen_kwargs = dict(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            grammar=req.grammar,
            seed=req.seed,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if req.stream:
            return StreamingResponse(
                _stream_sse_completion(_engine, messages, req.model, _inference_sem,
                                       audit=_audit, transcript=_transcript,
                                       pipeline=pipeline, ctx=ctx, **gen_kwargs),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        loop = asyncio.get_running_loop()
        # Count tokens on the (possibly inlet-transformed) messages - what
        # inference actually sees, matching the chat path.
        prompt_tokens = _engine.count_tokens(_messages_prompt_text(messages))

        def _run():
            return "".join(_engine.chat_stream(messages, **gen_kwargs))

        async with _inference_sem:
            text = await loop.run_in_executor(None, _run)

        # Outlet fully controls the returned content in the non-streaming path;
        # then record the exchange (audit + transcript), exactly like chat.
        if pipeline is not None and ctx is not None and pipeline.has("outlet"):
            text = await pipeline.run_outlet(text, messages, ctx)
        _audit_exchange(_audit, _transcript, messages, text)

        completion_tokens = _engine.count_tokens(text)
        ts  = int(time.time())
        cid = make_chunk_id()
        return {
            "id": cid,
            "object": "text_completion",
            "created": ts,
            "model": req.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

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

    # Exact prompt token count from the backend tokenizer
    prompt_text = " ".join(
        m.get("content") if isinstance(m.get("content"), str)
        else " ".join(p.get("text", "") for p in (m.get("content") or [])
                      if p.get("type") == "text")
        for m in messages
    )
    prompt_tokens = engine.count_tokens(prompt_text)

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
                prompt_text = " ".join(
                    m.get("content") if isinstance(m.get("content"), str)
                    else " ".join(p.get("text", "") for p in (m.get("content") or [])
                                  if p.get("type") == "text")
                    for m in messages
                )
                prompt_tokens = engine.count_tokens(prompt_text)

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
    done = ChatChunk.done(model_id, chunk_id, ts, usage=usage,
                          finish_reason=_engine_finish_reason(engine))
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
    done = {
        "id": chunk_id, "object": "text_completion.chunk",
        "created": ts, "model": model_id,
        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
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

    # Exact prompt token count before running inference
    prompt_text = " ".join(
        m.get("content") if isinstance(m.get("content"), str)
        else " ".join(p.get("text", "") for p in (m.get("content") or [])
                      if p.get("type") == "text")
        for m in messages
    )
    prompt_tokens = engine.count_tokens(prompt_text)

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
                prompt_text = " ".join(
                    m.get("content") if isinstance(m.get("content"), str)
                    else " ".join(p.get("text", "") for p in (m.get("content") or [])
                                  if p.get("type") == "text")
                    for m in messages
                )
                prompt_tokens = engine.count_tokens(prompt_text)
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

    # Task 2: Ctrl+C (Windows SIGINT). On Windows, signal.signal(SIGINT) is
    # unreliable in async code (the Python signal handler runs on the main thread
    # only at bytecode boundaries, and uvicorn's event loop often does not give it
    # a chance). The Win32 SetConsoleCtrlHandler fires on a dedicated OS thread
    # that the event loop does not block. We catch CTRL_C_EVENT (0) and
    # CTRL_BREAK_EVENT (1) there and tell the running loop to stop, which causes
    # uvicorn to wind down cleanly (model unload in _do_shutdown runs from the
    # route or via atexit). No new third-party dependencies needed - ctypes is
    # in the stdlib.
    if sys.platform == "win32":
        import ctypes
        import threading

        _ctrl_stop_event = threading.Event()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
        def _ctrl_handler(ctrl_type):
            # CTRL_C_EVENT = 0, CTRL_BREAK_EVENT = 1
            if ctrl_type in (0, 1):
                _ctrl_stop_event.set()
                try:
                    # Find the running asyncio loop and ask it to stop from the
                    # control-handler thread (which is NOT the event loop thread).
                    import asyncio
                    loop = asyncio.get_event_loop_policy().get_event_loop()
                    if loop is not None and loop.is_running():
                        loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
                return True   # handled; suppress the default Ctrl+C handler
            return False      # pass through other control events

        ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)

    with instances.advertise(app, home_dir(), host=host, port=port, mode="api",
                             scheme=scheme, project=project, isolated=isolated):
        # On a TLS bind, also catch a plain-http request on the same port with an
        # https redirect (issue 8); plain binds are a direct uvicorn.run. SRV-5:
        # in debug mode uvicorn logs at "info" so the console shows requests.
        from localm.debuglog import uvicorn_log_level
        portmux.run_server(app, host=host, port=port,
                           log_level=uvicorn_log_level(),
                           ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)

