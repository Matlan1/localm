"""
OpenAI-compatible HTTP inference server built with FastAPI + uvicorn.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (streaming + non-streaming, multimodal-capable)

Start programmatically:
    from localm.inference.http_server import serve
    serve(engine, host="127.0.0.1", port=8080)
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatRequest, ChatResponse, FullChoice, Message,
    make_chunk_id,
)

# Global engine reference set by serve()
_engine: Engine | None = None


# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def create_app(engine: Engine) -> FastAPI:
    global _engine
    _engine = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield  # model is already loaded before we're called

    app = FastAPI(
        title="localm inference server",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- #
    #  Health                                                            #
    # ---------------------------------------------------------------- #

    @app.get("/health")
    async def health():
        if _engine is None or not _engine._backend.loaded:
            raise HTTPException(503, "Model not loaded")
        return {"status": "ok", "model": _engine.display_name}

    # ---------------------------------------------------------------- #
    #  Models list                                                       #
    # ---------------------------------------------------------------- #

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": _engine.display_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "localm",
                }
            ],
        }

    # ---------------------------------------------------------------- #
    #  Chat completions                                                  #
    # ---------------------------------------------------------------- #

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        if _engine is None:
            raise HTTPException(503, "No model loaded")

        # Convert pydantic Messages to plain dicts for the backend
        messages = _protocol_messages_to_dicts(req.messages)

        gen_kwargs = dict(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
        )
        # Strip None so Engine uses its config defaults
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if req.stream:
            return StreamingResponse(
                _stream_sse(_engine, messages, req.model, **gen_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _complete(_engine, messages, req.model, **gen_kwargs)

    return app


# ------------------------------------------------------------------ #
#  SSE streaming                                                       #
# ------------------------------------------------------------------ #

async def _stream_sse(
    engine: Engine,
    messages: list,
    model_id: str,
    **gen_kwargs,
) -> AsyncIterator[str]:
    chunk_id = make_chunk_id()
    ts = int(time.time())

    # Role announcement
    role_chunk = ChatChunk(
        id=chunk_id,
        created=ts,
        model=model_id,
        choices=[
            __import__("localm.inference.protocol", fromlist=["StreamChoice"]).StreamChoice(
                delta=__import__("localm.inference.protocol", fromlist=["ChoiceDelta"]).ChoiceDelta(role="assistant")
            )
        ],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    # Run blocking generator in executor so we don't block the event loop
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _generate():
        try:
            for token in engine.chat_stream(messages, **gen_kwargs):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

    import threading
    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    while True:
        token = await token_queue.get()
        if token is None:
            break
        chunk = ChatChunk.token(token, model_id, chunk_id, ts)
        yield f"data: {chunk.model_dump_json()}\n\n"

    done = ChatChunk.done(model_id, chunk_id, ts)
    yield f"data: {done.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    t.join()


# ------------------------------------------------------------------ #
#  Non-streaming completion                                            #
# ------------------------------------------------------------------ #

async def _complete(engine: Engine, messages: list, model_id: str, **gen_kwargs):
    loop = asyncio.get_running_loop()

    def _run():
        return "".join(engine.chat_stream(messages, **gen_kwargs))

    text = await loop.run_in_executor(None, _run)

    response = ChatResponse(
        id=make_chunk_id(),
        created=int(time.time()),
        model=model_id,
        choices=[
            FullChoice(
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
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

def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the server — blocks until Ctrl+C."""
    import uvicorn

    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="warning")
