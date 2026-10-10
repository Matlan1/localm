# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anthropic Messages routes: ``POST /v1/messages`` and
``POST /v1/messages/count_tokens``.

``/v1/messages`` translates the request to an OpenAI chat body and calls the
registered ``/v1/chat/completions`` endpoint with a derived request, so
capability routing, peer routing, compaction, tool calling, pins and audit
behave exactly as on that route. Errors, including auth and validation
failures, are rendered in Anthropic's ``{"type": "error", "error": {...}}``
shape. The wire translation lives in ``localm.inference.anthropic_protocol``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import localm.inference.http_server as _hs
from localm.inference import anthropic_protocol as A
from localm.inference.ollama_protocol import iter_sse_json
from localm.inference.protocol import ChatRequest
from localm.inference.backends.base import PretokenizerUnsafeInputError
from localm.inference.routes._chat_bridge import (
    EndpointMissing, chat_endpoint, collect_body, derived_request, validation_text,
)
from localm.inference.tool_calling import (
    ToolsError, has_tool_history, parse_tool_choice, render_messages, validate_tools,
)


class AnthropicRoute(APIRoute):
    """An APIRoute whose failures are rendered as Anthropic error bodies: a
    refused credential, a request that fails validation (400) and an
    :class:`AnthropicError`."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except A.AnthropicError as exc:
                return _error(exc.status, exc.message)
            except StarletteHTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
                return _error(exc.status_code, detail, exc.headers)
            except RequestValidationError as exc:
                return _error(400, validation_text(exc.errors()))
        return handler


def _error(status: int, message: str, headers: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=A.error_body(status, message),
                        headers=headers)


def register(app: FastAPI, ctx) -> None:
    _require_auth = _hs._require_auth

    def route(path: str, **kwargs):
        def register_route(fn):
            app.router.add_api_route(path, fn, methods=["POST"],
                                     route_class_override=AnthropicRoute, **kwargs)
            return fn
        return register_route

    def _chat_request(body: dict[str, Any]) -> ChatRequest:
        try:
            return ChatRequest(**body)
        except ValidationError as exc:
            raise A.AnthropicError(400, validation_text(exc.errors())) from None

    async def _error_text(inner: Response) -> str:
        body_iterator = getattr(inner, "body_iterator", None)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except ValueError:
            return raw.decode("utf-8", "replace")[:500] or "request failed"
        if isinstance(data, dict):
            return str(data.get("detail") or data.get("error") or "request failed")
        return "request failed"

    @route("/v1/messages", dependencies=[Depends(_require_auth)])
    async def anthropic_messages(req: A.MessagesRequest, request: Request):
        want_thinking = A.thinking_enabled(req.thinking)
        body = A.plan_messages(req, req.model)
        chat_req = _chat_request(body)
        derived = derived_request(
            request, "/v1/chat/completions", json.dumps(body).encode("utf-8"))
        try:
            endpoint = chat_endpoint(request.app, "/v1/chat/completions")
        except EndpointMissing as exc:
            raise A.AnthropicError(500, str(exc)) from None
        inner = await endpoint(chat_req, derived)
        if inner.status_code >= 400:
            raise A.AnthropicError(inner.status_code, await _error_text(inner))
        model = req.model or ""
        body_iterator = getattr(inner, "body_iterator", None)
        if req.stream and body_iterator is not None:
            headers = {k: v for k, v in inner.headers.items()
                       if k.lower().startswith("x-localm-")}
            headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            return StreamingResponse(
                A.message_stream(iter_sse_json(body_iterator), model=model,
                                 want_thinking=want_thinking),
                media_type="text/event-stream", headers=headers)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except ValueError:
            raise A.AnthropicError(502, "the chat route returned a reply that is not JSON") from None
        headers = {k: v for k, v in inner.headers.items() if k.lower().startswith("x-localm-")}
        reply = A.message_from_completion(data, model or str(data.get("model") or ""),
                                          want_thinking)
        return JSONResponse(reply, headers=headers)

    @route("/v1/messages/count_tokens", dependencies=[Depends(_require_auth)])
    async def anthropic_count_tokens(req: A.CountTokensRequest, request: Request):
        body = A.plan_count(req, req.model)
        chat_req = _chat_request(body)
        messages = _hs._protocol_messages_to_dicts(chat_req.messages)
        try:
            tools = validate_tools(chat_req.tools)
            choice = parse_tool_choice(chat_req.tool_choice, tools)
        except ToolsError as exc:
            raise A.AnthropicError(400, str(exc)) from None
        if tools or has_tool_history(messages):
            messages = render_messages(messages, list(tools), choice)
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise A.AnthropicError(400, "model is required")
        engine = await _hs.get_engine(req.model)
        _hs._pin(engine)
        try:
            tokens = await asyncio.get_running_loop().run_in_executor(
                None, engine.count_messages_tokens, messages)
        except PretokenizerUnsafeInputError as exc:
            raise A.AnthropicError(400, str(exc)) from None
        finally:
            _hs._unpin(engine)
        return {"input_tokens": int(tokens)}
