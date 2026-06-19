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
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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

# Optional bearer-token auth - enabled when LOCALM_API_KEY is set.
_bearer_scheme = HTTPBearer(auto_error=False)


def _bearer_token(request) -> Optional[str]:
    """Extract a presented bearer token from the raw Authorization header (used
    by the origin/management middleware, which has the request but not the
    parsed HTTPBearer credentials)."""
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def _enforce_scope(
    credentials: Optional[HTTPAuthorizationCredentials],
    scope: Optional[str],
) -> None:
    """Shared auth core. Open/dev mode when no key is configured anywhere
    (unless LOCALM_REQUIRE_AUTH forces fail-closed). Otherwise a valid key is
    required; *scope* None means 'any valid key', else the key's scopes must
    grant *scope* (the owner key implies every scope)."""
    from localm.auth import any_key_configured, require_auth_enabled, verify
    if not any_key_configured():
        if require_auth_enabled():
            raise HTTPException(
                status_code=503,
                detail="Auth required but no API key configured "
                       "(set one via the launcher or LOCALM_API_KEY)")
        return  # open/dev mode
    held = verify(credentials.credentials) if credentials else None
    if held is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if scope is not None and not scopes.grants(held, scope):
        raise HTTPException(status_code=403,
                            detail=f"Key lacks required scope: {scope}")


def _require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """Require any valid API key (no specific scope)."""
    _enforce_scope(credentials, None)


def require_scope(scope: str):
    """FastAPI dependency factory: require a key whose scopes grant *scope*.
    Use as ``dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))]``."""
    def dep(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> None:
        _enforce_scope(credentials, scope)
    return dep


def caller_scopes(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[set]:
    """The scope set the presented key grants (the owner key -> {ADMIN}), or
    None in open mode / when no valid key is presented. Routes use this to make
    authorisation decisions that depend on *who* the caller is (e.g. only an
    owner/ADMIN principal may mint keys carrying privileged scopes)."""
    from localm.auth import any_key_configured, verify
    if not any_key_configured():
        return None
    return verify(credentials.credentials) if credentials else None


# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def create_app(engine: Engine) -> FastAPI:
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
        yield
        _audit.close()

    app = FastAPI(
        title="localm inference server",
        version="0.1.0",
        lifespan=lifespan,
    )

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
        # is a one-way self-lockout: the very next keyless request 503s and the
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
    #  API keys - scoped keystore (auth.json)                            #
    # ---------------------------------------------------------------- #

    @app.get("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def list_keys_ep():
        from localm.auth import list_keys
        return {"keys": list_keys()}

    @app.post("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def create_key_ep(body: dict, caller: Optional[set] = Depends(caller_scopes)):
        from localm.auth import create_key
        scope_list = body.get("scopes", [])
        if not isinstance(scope_list, list):
            raise HTTPException(400, "'scopes' must be a list of scope strings")
        # Only an owner/ADMIN caller may grant privileged scopes; a key holding
        # merely keys:admin must not be able to mint itself owner-equivalent
        # access. In open mode (caller is None) privileged grants are refused.
        is_owner = caller is not None and scopes.ADMIN in caller
        try:
            created = create_key(body.get("name", ""), scope_list,
                                 allow_privileged=is_owner)
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
                from localm.model_manager import vision_input_guidance
                raise HTTPException(400, vision_input_guidance())

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
    async def completions(req: CompletionRequest):
        if _engine is None:
            raise HTTPException(503, "No model loaded")

        # Wrap the prompt as a single user message so the chat backend handles it
        messages = [{"role": "user", "content": req.prompt}]

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
                _stream_sse_completion(_engine, req.prompt, req.model, _inference_sem, **gen_kwargs),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        loop = asyncio.get_running_loop()
        prompt_tokens = _engine.count_tokens(req.prompt)

        def _run():
            return "".join(_engine.chat_stream(messages, **gen_kwargs))

        async with _inference_sem:
            text = await loop.run_in_executor(None, _run)

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
    except Exception:
        from localm.debuglog import logger as _dbg
        _dbg.exception("plugin engine attach failed")

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
    except Exception:
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
            # Log it (debug log included) and surface it to the client -
            # a silent thread death looks like an empty reply.
            from localm.debuglog import logger as _dbg
            _dbg.exception("generation thread failed")
            import traceback
            traceback.print_exc()
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
    prompt: str,
    model_id: str,
    sem: asyncio.Semaphore,
    **gen_kwargs,
) -> AsyncIterator[str]:
    messages = [{"role": "user", "content": prompt}]
    chunk_id = make_chunk_id()
    ts = int(time.time())
    prompt_tokens = engine.count_tokens(prompt)

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _generate():
        try:
            for token in engine.chat_stream(messages, **gen_kwargs):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

    import threading

    async with sem:
        gen_start = time.perf_counter()
        first_token_at: float | None = None
        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        completion_parts: list[str] = []
        while True:
            token = await token_queue.get()
            if token is None:
                break
            if first_token_at is None:
                first_token_at = time.perf_counter()
            completion_parts.append(token)
            chunk = {
                "id": chunk_id, "object": "text_completion.chunk",
                "created": ts, "model": model_id,
                "choices": [{"text": token, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        t.join()
        gen_elapsed = time.perf_counter() - gen_start

    completion_tokens = engine.count_tokens("".join(completion_parts))
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
          ssl_keyfile: Optional[str] = None) -> None:
    """Start the server - blocks until Ctrl+C.

    When ``ssl_certfile`` / ``ssl_keyfile`` are given (built-in TLS, NET-1), the
    server speaks HTTPS on this port; a plain-HTTP request to it then fails the
    TLS handshake (effectively refused) rather than crossing the network in
    cleartext.
    """
    import uvicorn

    app = create_app(engine)
    # Record the bind host so routes that depend on it (open-mode seeding,
    # CA download) can reason about loopback vs network binds.
    app.state.bind_host = host
    uvicorn.run(app, host=host, port=port, log_level="warning",
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
