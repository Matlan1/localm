# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ollama-native routes: /api/tags, /api/ps, /api/show, /api/version,
/api/chat, /api/generate, /api/embed, /api/embeddings and /api/copy, plus
/api/pull, /api/push, /api/create, /api/delete and /api/blobs/{digest}, which
answer 501 with the localm equivalent.

Chat and generate translate the request to an OpenAI chat body and call the
registered /v1/chat/completions endpoint with a derived request, so capability
routing, peer routing, compaction, pins and audit behave exactly as on /v1.
Every route is an exact path on the same app and port; the wire translation
lives in ``localm.inference.ollama_protocol``."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import localm
import localm.inference.http_server as _hs
from localm import peer_routing, scopes
from localm.executor import get_plugin_executor
from localm.inference import ollama_protocol as P
from localm.inference.protocol import ChatRequest, EmbeddingRequest
from localm.inference.routes._chat_bridge import (
    EndpointMissing, chat_endpoint, collect_body, derived_request, validation_text,
)

_NDJSON = "application/x-ndjson"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")
_LISTED_MODEL_TYPES = ("llm", "embedding")


class OllamaRoute(APIRoute):
    """An APIRoute whose failures are rendered as ``{"error": "<message>"}``,
    the body Ollama clients read: a refused credential or scope, a request that
    fails validation (a 400, as in Ollama) and an :class:`OllamaError`. The
    wrapped handler covers everything FastAPI does for the route, dependencies
    and body validation included."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except P.OllamaError as exc:
                return _error(exc.status, exc.message)
            except StarletteHTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
                return _error(exc.status_code, detail, exc.headers)
            except RequestValidationError as exc:
                return _error(400, validation_text(exc.errors()))
        return handler


def _error(status: int, message: str,
           headers: Optional[Mapping[str, str]] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message}, headers=headers)


def _endpoint(app: FastAPI, path: str):
    try:
        return chat_endpoint(app, path)
    except EndpointMissing as exc:
        raise P.OllamaError(500, str(exc)) from None


def _file_facts(entry: dict) -> tuple[int, Optional[float], str]:
    """``(size_bytes, mtime, format)`` of a registry entry's weights, read from
    disk. Blocking: run it in an executor."""
    from localm.model_manager.registry import _entry_path
    epath = _entry_path(entry)
    if epath is None:
        return 0, None, ""
    path = Path(epath)
    try:
        if path.is_file():
            st = path.stat()
            return st.st_size, st.st_mtime, path.suffix.lstrip(".").lower() or "unknown"
        if path.is_dir():
            weights = [f for f in path.iterdir()
                       if f.is_file() and f.suffix.lower() in _WEIGHT_SUFFIXES]
            size = sum(f.stat().st_size for f in weights)
            fmt = "safetensors" if any(f.suffix.lower() == ".safetensors"
                                       for f in weights) else "hf"
            return size, path.stat().st_mtime, fmt
    except OSError:
        pass
    return 0, None, ""


def _listed_models(registry: dict) -> dict[str, dict]:
    """Registry entries Ollama clients may see: chat and embedding models, plus
    the startup model when it is not registered."""
    out = {name: entry for name, entry in registry.items()
           if isinstance(entry, dict)
           and entry.get("model_type", "llm") in _LISTED_MODEL_TYPES}
    default = _hs._default_model_name
    if default and default not in registry:
        startup = getattr(_hs._engine, "model_path", "")
        out[default] = {"path": str(startup) if startup else "", "source": "startup"}
    return out


def _entry_details(entry: dict, fmt: str) -> dict[str, Any]:
    arch = entry.get("architecture")
    return P.model_details(fmt=fmt, family=arch if isinstance(arch, str) else "")


def _tag_object(name: str, entry: dict, facts: tuple[int, Optional[float], str]) -> dict:
    size, mtime, fmt = facts
    return {
        "name": name,
        "model": name,
        "modified_at": P.iso_from_timestamp(mtime) if mtime is not None else None,
        "size": size,
        "digest": P.digest_of(entry),
        "details": _entry_details(entry, fmt),
    }


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    _require_auth = _hs._require_auth
    read_models = [Depends(require_scope(scopes.MODELS_READ))]
    write_models = [Depends(require_scope(scopes.MODELS_WRITE))]

    def route(path: str, *, methods: list, **kwargs):
        def register_route(fn):
            app.router.add_api_route(path, fn, methods=methods,
                                     route_class_override=OllamaRoute, **kwargs)
            return fn
        return register_route

    def _registry_names() -> set[str]:
        from localm.config import load_registry
        names = set(load_registry())
        if _hs._default_model_name:
            names.add(_hs._default_model_name)
        return names

    def _canonical(name: str) -> str:
        return P.resolve_model_name(name, _registry_names())

    def _note_ignored(plan: P.Plan, req: Any) -> None:
        ignored = plan.ignored_options + P.unknown_fields(req)
        if ignored:
            from localm.debuglog import logger as _dbg
            _dbg.debug("ollama request keys ignored: %s", ", ".join(ignored))

    # ---------------------------------------------------------------- #
    #  Reads                                                            #
    # ---------------------------------------------------------------- #

    @route("/api/version", methods=["GET"], dependencies=[Depends(_require_auth)])
    async def ollama_version():
        return {"version": localm.__version__}

    @route("/api/tags", methods=["GET"], dependencies=read_models)
    async def ollama_tags():
        from localm.config import load_registry
        listed = _listed_models(load_registry())

        def _collect() -> list[dict]:
            return [_tag_object(name, entry, _file_facts(entry))
                    for name, entry in sorted(listed.items())]

        loop = asyncio.get_running_loop()
        return {"models": await loop.run_in_executor(get_plugin_executor(), _collect)}

    @route("/api/ps", methods=["GET"], dependencies=read_models)
    async def ollama_ps():
        from localm.config import load_registry
        listed = _listed_models(load_registry())
        loaded = [(name, listed[name], engine)
                  for name, engine in list(_hs._engines.items())
                  if name in listed and engine.loaded]

        def _collect() -> list[dict]:
            out = []
            ttl = _hs._idle_unload_ttl()
            now = time.monotonic()
            for name, entry, engine in loaded:
                obj = _tag_object(name, entry, _file_facts(entry))
                obj.pop("modified_at")
                last = _hs._last_activity_per_model.get(name)
                obj["expires_at"] = P.expiry_iso(
                    ttl, None if last is None else now - last)
                capacity = engine.context_capacity()
                if isinstance(capacity, int) and capacity > 0:
                    obj["context_length"] = capacity
                out.append(obj)
            return out

        loop = asyncio.get_running_loop()
        return {"models": await loop.run_in_executor(get_plugin_executor(), _collect)}

    @route("/api/show", methods=["POST"], dependencies=read_models)
    async def ollama_show(req: P.OllamaShowRequest):
        from localm.config import load_registry
        from localm.model_manager.capabilities import model_capabilities
        name = P.request_model_name(req)
        registry = load_registry()
        listed = _listed_models(registry)
        name = P.resolve_model_name(name, listed)
        entry = listed.get(name)
        if entry is None:
            raise P.OllamaError(404, f"model '{name}' not found")

        def _collect() -> dict:
            facts = _file_facts(entry)
            caps = model_capabilities(name, reg=registry) if name in registry else {}
            capabilities = ["embedding"] if entry.get("model_type") == "embedding" \
                else ["completion", "tools"]
            if caps.get("vision"):
                capabilities.append("vision")
            if caps.get("reasoning"):
                capabilities.append("thinking")
            info: dict[str, Any] = {}
            arch = entry.get("architecture")
            if isinstance(arch, str) and arch:
                info["general.architecture"] = arch
                if isinstance(caps.get("context_length"), int):
                    info[f"{arch}.context_length"] = caps["context_length"]
            return {
                "modelfile": "",
                "parameters": "",
                "template": "",
                "details": _entry_details(entry, facts[2]),
                "model_info": info,
                "modified_at": (P.iso_from_timestamp(facts[1])
                                if facts[1] is not None else None),
                "capabilities": capabilities,
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_plugin_executor(), _collect)

    # ---------------------------------------------------------------- #
    #  Chat and generate                                                #
    # ---------------------------------------------------------------- #

    def _require_unload_allowed(request: Request) -> None:
        """The management gate for unloading a model: the models:write scope
        when keys exist; in open mode, the GUI shell token (or this instance's
        attach token) on a same-origin or origin-less request, as
        POST /v1/models/unload requires."""
        from localm.auth import any_key_configured, ct_equal, require_auth_enabled
        _hs._enforce_request(request, scopes.MODELS_WRITE)
        if any_key_configured() or require_auth_enabled():
            return
        presented = _hs._bearer_token(request)
        state = request.app.state
        token_ok = any(t and ct_equal(presented, t) for t in (
            getattr(state, "shell_token", None), getattr(state, "instance_token", None)))
        origin = request.headers.get("origin")
        foreign = bool(origin) and (
            origin.split("://", 1)[-1] != request.headers.get("host", ""))
        if not token_ok or foreign:
            raise P.OllamaError(
                403, "Unloading a model with no API key configured requires the "
                     "localm GUI shell on this machine, or an API key (run "
                     "'localm key generate').")

    async def _load_or_unload(request: Request, name: str, unload: bool,
                              kind: str) -> JSONResponse:
        resolved = _canonical(name)
        if unload:
            _require_unload_allowed(request)
            if resolved not in _registry_names():
                raise P.OllamaError(404, f"model '{name}' not found")
            result = await _hs.unload_one_model(resolved)
            if isinstance(result, dict) and result.get("status") == "confirm_required":
                raise P.OllamaError(
                    409, f"model '{name}' is in use; retry when its requests finish")
        elif peer_routing.get_route(resolved) is None:
            try:
                await _hs.get_engine(resolved)
            except HTTPException as exc:
                if exc.status_code == 404:
                    raise P.OllamaError(404, f"model '{name}' not found") from None
                raise
        return JSONResponse(P.reply_object(
            kind, name, done=True, reason="unload" if unload else "load"))

    async def _run_chat(request: Request, plan: P.Plan, kind: str, name: str):
        started = time.perf_counter()
        try:
            chat_req = ChatRequest(**plan.body)
        except ValidationError as exc:
            raise P.OllamaError(400, validation_text(exc.errors())) from None
        derived = derived_request(
            request, "/v1/chat/completions", json.dumps(plan.body).encode("utf-8"))
        inner = await _endpoint(request.app, "/v1/chat/completions")(chat_req, derived)
        return await _reply(inner, plan, kind, name, started)

    async def _reply(inner: Response, plan: P.Plan, kind: str, name: str,
                     started: float):
        body_iterator = getattr(inner, "body_iterator", None)
        if inner.status_code >= 400:
            raise P.OllamaError(inner.status_code, await _error_text(inner))
        if body_iterator is not None and plan.stream:
            headers = {k: v for k, v in inner.headers.items()
                       if k.lower().startswith("x-localm-")}
            headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            return StreamingResponse(
                P.ndjson_stream(
                    P.iter_sse_json(body_iterator), kind=kind, model=name,
                    want_thinking=plan.want_thinking, started=started),
                media_type=_NDJSON, headers=headers)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError):
            raise P.OllamaError(
                502, "the model route returned a reply that is not JSON") from None
        total_ns = int((time.perf_counter() - started) * 1_000_000_000)
        return JSONResponse(P.completion_to_reply(
            kind, data, name, want_thinking=plan.want_thinking, total_ns=total_ns))

    async def _error_text(inner: Response) -> str:
        body_iterator = getattr(inner, "body_iterator", None)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError):
            return raw.decode("utf-8", "replace")[:500] or "request failed"
        if isinstance(data, dict):
            return str(data.get("error") or data.get("detail") or "request failed")
        return "request failed"

    @route("/api/chat", methods=["POST"], dependencies=[Depends(_require_auth)])
    async def ollama_chat(req: P.OllamaChatRequest, request: Request):
        name = P.request_model_name(req)
        if not req.messages:
            return await _load_or_unload(
                request, name, P.keep_alive_is_zero(req.keep_alive), "chat")
        plan = P.plan_chat(req, _canonical(name))
        _note_ignored(plan, req)
        return await _run_chat(request, plan, "chat", name)

    @route("/api/generate", methods=["POST"], dependencies=[Depends(_require_auth)])
    async def ollama_generate(req: P.OllamaGenerateRequest, request: Request):
        name = P.request_model_name(req)
        if not req.prompt and not req.images:
            return await _load_or_unload(
                request, name, P.keep_alive_is_zero(req.keep_alive), "generate")
        plan = P.plan_generate(req, _canonical(name))
        _note_ignored(plan, req)
        return await _run_chat(request, plan, "generate", name)

    # ---------------------------------------------------------------- #
    #  Embeddings                                                       #
    # ---------------------------------------------------------------- #

    async def _embed(request: Request, req: P.OllamaEmbedRequest,
                     texts: list[str]) -> tuple[list[list[float]], int, int]:
        name = P.request_model_name(req)
        if req.dimensions is not None:
            raise P.OllamaError(400, "dimensions is not supported")
        started = time.perf_counter()
        result = await _endpoint(request.app, "/v1/embeddings")(
            EmbeddingRequest(model=_canonical(name), input=texts))
        vectors = [item["embedding"] for item in result["data"]]
        if len(vectors) != len(texts):
            raise P.OllamaError(
                502, f"the embedding model returned {len(vectors)} vectors for "
                     f"{len(texts)} inputs")
        tokens = int((result.get("usage") or {}).get("prompt_tokens") or 0)
        return vectors, tokens, int((time.perf_counter() - started) * 1_000_000_000)

    @route("/api/embed", methods=["POST"], dependencies=[Depends(_require_auth)])
    async def ollama_embed(req: P.OllamaEmbedRequest, request: Request):
        if req.input is None:
            raise P.OllamaError(400, "input is required")
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        name = P.request_model_name(req)
        if not texts:
            return {"model": name, "embeddings": []}
        vectors, tokens, total_ns = await _embed(request, req, texts)
        out = {"model": name, "embeddings": vectors, "total_duration": total_ns}
        if tokens > 0:
            out["prompt_eval_count"] = tokens
        return out

    @route("/api/embeddings", methods=["POST"], dependencies=[Depends(_require_auth)])
    async def ollama_embeddings(req: P.OllamaEmbedRequest, request: Request):
        if req.prompt is None:
            raise P.OllamaError(400, "prompt is required")
        vectors, _tokens, _ns = await _embed(request, req, [req.prompt])
        return {"embedding": vectors[0]}

    # ---------------------------------------------------------------- #
    #  Model management                                                 #
    # ---------------------------------------------------------------- #

    @route(P.COPY_PATH, methods=["POST"], dependencies=write_models)
    async def ollama_copy(req: P.OllamaCopyRequest):
        from localm.config import load_registry
        from localm.model_manager.registry import _sanitize_name, alias_model
        source, destination = (req.source or "").strip(), (req.destination or "").strip()
        if not source or not destination:
            raise P.OllamaError(400, "source and destination are required")
        registry = load_registry()
        source = P.resolve_model_name(source, registry)
        if source not in registry:
            raise P.OllamaError(404, f"model '{source}' not found")
        if _sanitize_name(destination) != destination:
            raise P.OllamaError(
                400, f"'{destination}' is not a usable model name; use letters, "
                     "digits, '.', '_' and '-'")
        if destination in registry:
            raise P.OllamaError(409, f"model '{destination}' already exists")
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            get_plugin_executor(), alias_model, source, destination)
        if not ok:
            raise P.OllamaError(500, f"could not register '{destination}'")
        return {"status": "success"}

    unsupported = {
        "/api/pull": "localm does not pull from the Ollama registry; download a "
                     "model with `localm pull <owner/repo:file.gguf>` or the GUI.",
        "/api/push": "localm has no model registry to push to.",
        "/api/create": "Modelfile creation is not supported; register a GGUF with "
                       "`localm pull <path or spec>`.",
        "/api/delete": "remove a model with `localm rm <model>` or the GUI.",
    }

    def _refusal(path: str):
        message = unsupported[path]

        async def _handler():
            raise P.OllamaError(501, message)
        return _handler

    for _path, _method in (("/api/pull", "POST"), ("/api/push", "POST"),
                           ("/api/create", "POST"), ("/api/delete", "DELETE")):
        app.router.add_api_route(
            _path, _refusal(_path), methods=[_method], dependencies=write_models,
            include_in_schema=False, route_class_override=OllamaRoute)

    @route(P.BLOBS_PREFIX + "{digest}", methods=["POST", "HEAD"],
                   dependencies=write_models, include_in_schema=False)
    async def ollama_blobs(digest: str):
        raise P.OllamaError(
            501, "blob upload is only used by /api/create, which is not supported")

