# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI Responses route: ``POST /v1/responses``.

The request is translated to a chat body and the registered
``/v1/chat/completions`` endpoint is called with a derived request, so
capability routing, peer routing, compaction, tool calling, structured output
and audit behave as on that route. A stored response (``store`` not false) is
kept in this process's memory for ``previous_response_id``; a response the
store cannot keep is returned with ``store`` false. Errors, including
auth and validation failures, are rendered as OpenAI error bodies. The wire
translation and the store live in ``localm.inference.responses_protocol``."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import localm.inference.http_server as _hs
from localm.inference import responses_protocol as R
from localm.inference.ollama_protocol import iter_sse_json
from localm.inference.protocol import ChatRequest
from localm.inference.routes._chat_bridge import (
    EndpointMissing, chat_endpoint, collect_body, derived_request, validation_text,
)


class ResponsesRoute(APIRoute):
    """An APIRoute whose failures are rendered as OpenAI error bodies: a refused
    credential, a request that fails validation (400), a
    :class:`ResponsesError`, and an unexpected error (500, its traceback logged)."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except R.ResponsesError as exc:
                return _error(exc.status, exc.message, exc.param)
            except StarletteHTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
                return _error(exc.status_code, detail, headers=exc.headers)
            except RequestValidationError as exc:
                return _error(400, validation_text(exc.errors()))
            except Exception:
                from localm.debuglog import logger as _dbg
                _dbg.exception("unhandled error on %s", request.url.path)
                return _error(500, "Internal server error")
        return handler


def _error(status: int, message: str, param: Any = None, headers: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=R.error_body(status, message, param),
                        headers=headers)


STORE = R.ResponseStore()


def register(app: FastAPI, ctx) -> None:
    _require_auth = _hs._require_auth

    async def _error_text(inner: Response) -> str:
        body_iterator = getattr(inner, "body_iterator", None)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError):
            return raw.decode("utf-8", "replace")[:500] or "request failed"
        if isinstance(data, dict):
            return str(data.get("detail") or data.get("error") or "request failed")
        return "request failed"

    async def create_response(req: R.ResponsesRequest, request: Request):
        principal = _hs.principal_id(request)
        history: list[dict[str, Any]] = []
        if req.previous_response_id:
            stored = STORE.get(principal, req.previous_response_id)
            if stored is None:
                raise R.ResponsesError(
                    404, f"Previous response with id '{req.previous_response_id}' not found.",
                    "previous_response_id")
            history = stored
        ignored = R.ignored_fields(req)
        if ignored:
            from localm.debuglog import logger as _dbg
            _dbg.debug("responses request keys ignored: %s", ", ".join(ignored))
        body, conversation = R.plan_chat(req, history)
        try:
            chat_req = ChatRequest(**body)
        except ValidationError as exc:
            raise R.ResponsesError(400, validation_text(exc.errors())) from None
        derived = derived_request(
            request, "/v1/chat/completions", json.dumps(body).encode("utf-8"))
        try:
            endpoint = chat_endpoint(request.app, "/v1/chat/completions")
        except EndpointMissing as exc:
            raise R.ResponsesError(500, str(exc)) from None
        inner = await endpoint(chat_req, derived)
        if inner.status_code >= 400:
            raise R.ResponsesError(inner.status_code, await _error_text(inner))
        shell = R.Shell(req)

        def remember(final: dict[str, Any]) -> None:
            if req.store is False or final.get("status") == "failed":
                return
            kept = STORE.put(principal, final["id"],
                             conversation + R.output_messages(final["output"]))
            if not kept:
                final["store"] = False

        headers = {k: v for k, v in inner.headers.items() if k.lower().startswith("x-localm-")}
        body_iterator = getattr(inner, "body_iterator", None)
        if req.stream and body_iterator is not None:
            headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            return StreamingResponse(
                R.response_stream(shell, iter_sse_json(body_iterator), on_done=remember),
                media_type="text/event-stream", headers=headers)
        raw = (await collect_body(body_iterator)
               if body_iterator is not None else bytes(inner.body))
        try:
            data = json.loads(raw)
        except (ValueError, RecursionError):
            raise R.ResponsesError(502, "the chat route returned a reply that is not JSON") from None
        final = R.response_from_completion(shell, data)
        remember(final)
        return JSONResponse(final, headers=headers)

    app.router.add_api_route("/v1/responses", create_response, methods=["POST"],
                             dependencies=[Depends(_require_auth)],
                             route_class_override=ResponsesRoute)
