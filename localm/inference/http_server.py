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
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from localm.inference.engine import Engine
from localm.inference.protocol import (
    ChatChunk, ChatRequest, ChatResponse, CompletionRequest, EmbeddingRequest,
    FullChoice, Message, UsageInfo, make_chunk_id,
)

# Global engine reference set by serve()
_engine: Engine | None = None

# Inference serialisation — only one request runs inference at a time.
# Additional requests queue behind this semaphore.
_inference_sem: asyncio.Semaphore | None = None

# Optional bearer-token auth — enabled when LOCALM_API_KEY is set.
_bearer_scheme = HTTPBearer(auto_error=False)


def _require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """Validate the Bearer token when LOCALM_API_KEY is set in the environment.

    If the env var is absent the server runs in open/dev mode (no auth check).
    If the env var is set, every request to a protected endpoint must supply a
    matching ``Authorization: Bearer <token>`` header.
    """
    api_key = os.environ.get("LOCALM_API_KEY")
    if not api_key:
        return  # no key configured — dev/local mode, skip auth
    if credentials is None or credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def create_app(engine: Engine) -> FastAPI:
    global _engine
    _engine = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _inference_sem
        # Semaphore created inside the running event loop — Python 3.10+ safe
        _inference_sem = asyncio.Semaphore(1)
        yield

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
        if _engine is None:
            raise HTTPException(503, "No engine initialised")
        return {
            "status": "ok",
            "model":  _engine.display_name,
            "loaded": _engine.loaded,
        }

    # ---------------------------------------------------------------- #
    #  Models list                                                       #
    # ---------------------------------------------------------------- #

    @app.get("/v1/models")
    async def list_models():
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

    # ---------------------------------------------------------------- #
    #  Model lifecycle — unload / load                                   #
    # ---------------------------------------------------------------- #

    @app.post("/v1/models/unload", dependencies=[Depends(_require_auth)])
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
        await loop.run_in_executor(None, _engine.unload)
        return {"status": "unloaded", "model": _engine.display_name}

    @app.post("/v1/models/load", dependencies=[Depends(_require_auth)])
    async def load_model():
        """
        Explicitly reload the model into memory.

        Normally you don't need this — /v1/chat/completions reloads
        automatically if the model was unloaded.  Use this endpoint if you
        want to pre-warm the model before the first inference request.
        """
        if _engine is None:
            raise HTTPException(503, "No engine initialised")
        if _engine.loaded:
            return {"status": "already_loaded", "model": _engine.display_name}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _engine.load)
        return {"status": "loaded", "model": _engine.display_name}

    # ---------------------------------------------------------------- #
    #  Chat completions                                                  #
    # ---------------------------------------------------------------- #

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_auth)])
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
            grammar=req.grammar,
            seed=req.seed,
        )
        # Strip None so Engine uses its config defaults
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if req.stream:
            return StreamingResponse(
                _stream_sse(_engine, messages, req.model, _inference_sem, **gen_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _complete(_engine, messages, req.model, _inference_sem, **gen_kwargs)

    # ---------------------------------------------------------------- #
    #  Embeddings  (/v1/embeddings)                                     #
    # ---------------------------------------------------------------- #

    @app.post("/v1/embeddings", dependencies=[Depends(_require_auth)])
    async def embeddings(req: EmbeddingRequest):
        if _engine is None:
            raise HTTPException(503, "No model loaded")

        texts = [req.input] if isinstance(req.input, str) else req.input

        loop = asyncio.get_running_loop()
        try:
            async with _inference_sem:
                vecs = await loop.run_in_executor(None, lambda: _engine.embed(texts))
        except NotImplementedError as e:
            raise HTTPException(422, str(e))

        total_tokens = sum(_engine.count_tokens(t) for t in texts)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
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

    return app


# ------------------------------------------------------------------ #
#  Performance metric helpers                                          #
# ------------------------------------------------------------------ #

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

async def _stream_sse(
    engine: Engine,
    messages: list,
    model_id: str,
    sem: asyncio.Semaphore,
    **gen_kwargs,
) -> AsyncIterator[str]:
    from localm.inference.protocol import ChoiceDelta, StreamChoice

    chunk_id = make_chunk_id()
    ts = int(time.time())

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
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _generate():
        try:
            for token in engine.chat_stream(messages, **gen_kwargs):
                loop.call_soon_threadsafe(token_queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

    import threading

    # Serialise inference — only one request runs at a time
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
            chunk = ChatChunk.token(token, model_id, chunk_id, ts)
            yield f"data: {chunk.model_dump_json()}\n\n"

        t.join()
        gen_elapsed = time.perf_counter() - gen_start

    # Count tokens on the full completion text — more accurate and efficient
    completion_tokens = engine.count_tokens("".join(completion_parts))

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        ttft_ms=_ttft_ms(gen_start, first_token_at),
        tokens_per_sec=_tokens_per_sec(completion_tokens, gen_elapsed),
    )
    done = ChatChunk.done(model_id, chunk_id, ts, usage=usage)
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

    # Serialise inference — only one request runs at a time
    async with sem:
        gen_start = time.perf_counter()
        text = await loop.run_in_executor(None, _run)
        gen_elapsed = time.perf_counter() - gen_start

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
                message=Message(role="assistant", content=text),
                finish_reason="stop",
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

def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the server — blocks until Ctrl+C."""
    import uvicorn

    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="warning")
