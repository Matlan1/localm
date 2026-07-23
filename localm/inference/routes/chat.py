# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference routes: chat completions, embeddings, and raw completions.

Extracted verbatim from create_app(); behavior unchanged. These are the heaviest
routes - they read the live engine and inference semaphore from the http_server
module globals, the session-scoped audit/transcript from ctx, and call the
streaming/completion helpers that stay on http_server (_stream_sse,
_stream_sse_completion, _complete, _audit_exchange, _messages_prompt_text,
_protocol_messages_to_dicts).

NOTE on the local ``ctx``: each handler builds its own per-request
``ChatHookContext`` named ``ctx`` (the chat-pipeline turn context). To keep those
bodies byte-for-byte identical, the session-scoped audit/transcript are unpacked
from the register ``ctx`` into module-style locals (``_audit`` / ``_transcript``)
once at the top of register(), before any handler body shadows ``ctx``.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import localm.inference.http_server as _hs
from localm.inference.backends.base import InvalidGrammarError, messages_contain_image
from localm.inference.chat_pipeline import ChatHookContext
from localm.inference.gbnf import check_grammar_structure
from localm.inference.protocol import (
    ChatRequest, CompletionRequest, EmbeddingRequest, make_chunk_id,
)


def register(app: FastAPI, ctx) -> None:
    # Unpack the session-scoped objects BEFORE the handler bodies (which each
    # rebind a local ``ctx`` for the chat-pipeline turn) so the bodies stay
    # verbatim and still reach the create_app audit/transcript.
    _audit = ctx.audit
    _transcript = ctx.transcript
    _require_auth = _hs._require_auth
    _touch_activity = _hs._touch_activity
    _protocol_messages_to_dicts = _hs._protocol_messages_to_dicts
    _stream_sse = _hs._stream_sse
    _stream_sse_completion = _hs._stream_sse_completion
    _complete = _hs._complete
    _generate_full = _hs._generate_full
    _messages_prompt_text = _hs._messages_prompt_text
    _audit_exchange = _hs._audit_exchange
    _pin_engine = _hs._pin_engine
    _memory_used_header = _hs._memory_used_header

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_auth)])
    async def chat_completions(req: ChatRequest, request: Request):
        if not req.model:
            raise HTTPException(400, "Model parameter is required and cannot be empty")
            
        engine = await _hs.get_engine(req.model)
        # Pin the engine the instant we own it - SYNCHRONOUSLY, before the inlet
        # or any other await - so a concurrent model load cannot evict it out from
        # under this in-flight request (AUDIT-CRIT-1). Released in the finally
        # below, or, for a streaming response, by _pin_engine at stream end.
        _hs._pin(engine)
        streaming_handoff = False
        try:
            _touch_activity(engine.display_name)

            # Convert pydantic Messages to plain dicts for the backend
            messages = _protocol_messages_to_dicts(req.messages)

            # Chat-pipeline hooks. The inlet runs here (so token counting and
            # inference see the transformed messages); the per-request context is
            # built whenever a pipeline is present so the inlet, stream, and outlet
            # of this turn share one ctx. A pipeline with no hooks costs nothing.
            pipeline = getattr(request.app.state, "chat_pipeline", None)
            ctx = None
            if pipeline is not None:
                from localm.inference.http_server import caller_scopes, principal_id
                ctx = ChatHookContext(
                    model_id=req.model, stream=req.stream,
                    request_id=make_chunk_id(),
                    principal=principal_id(request),
                    scopes=tuple(caller_scopes(request) or ()),
                )
                ctx.state["client_id"] = request.headers.get("x-client-id", "")
                if pipeline.has("inlet"):
                    messages = await pipeline.run_inlet(messages, ctx)

            sem = _hs._inference_sems.setdefault(engine.display_name, asyncio.Semaphore(1))

            # Reject image input on a text-only model with a clear 400 instead of
            # silently dropping the picture. For GGUF (always text-only) this is
            # known immediately; for an unloaded HF model multimodal support is
            # only known after loading, so load first before deciding.
            if messages_contain_image(messages) and not engine.supports_images:
                if not engine.loaded and engine.can_be_multimodal:
                    loop = asyncio.get_running_loop()
                    async with sem:
                        await loop.run_in_executor(None, engine.load)
                if not engine.supports_images:
                    # Capability-aware guidance: route to a vision model this install
                    # has. supports_images is False here, so an mmproj_path set on the
                    # backend means the projector FAILED to load (rule 5: report that
                    # honest cause, not "text-only").
                    from localm.model_manager import vision_input_guidance
                    mmproj_failed = bool(
                        getattr(getattr(engine, "_backend", None), "mmproj_path", None))
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
            if req.grammar_lazy:
                # Reject the half-formed request instead of silently degrading: a
                # lazy grammar without its trigger patterns can never engage.
                if not req.grammar or not req.grammar_triggers:
                    raise HTTPException(
                        400, "grammar_lazy requires both grammar and grammar_triggers")
                gen_kwargs["grammar_lazy"] = True
                gen_kwargs["grammar_triggers"] = req.grammar_triggers

            # Reject a malformed grammar with a clean 400 UP FRONT (before streaming
            # starts, so both the stream and non-stream paths get a real 4xx). A bad
            # grammar reaching generation NULL-derefs the native sampler, which the
            # GGUF backend catches by latching _grammar_unsupported - silently
            # stripping grammar from every LATER request too. Validating here keeps a
            # per-request user error from poisoning the feature for all clients.
            if req.grammar:
                try:
                    # LM-FZ-001: a pure-Python structural check FIRST, unconditionally
                    # (no RPC, so it also covers the RunnerBusy-deferred path below,
                    # which otherwise skips straight to a generation-time native
                    # call). A grammar built of thousands of unmatched "(" drove
                    # llama.cpp's native GBNF parser into a real stack overflow -
                    # this rejects that shape before any of it reaches the parser.
                    check_grammar_structure(req.grammar)
                    engine.validate_grammar(req.grammar)
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # Defense in depth: a crashed/faulted native worker surfaces
                    # here as a bare RuntimeError (not InvalidGrammarError), which
                    # would otherwise escape as an unhandled 500. The structural
                    # check above should catch the confirmed crash shape before
                    # this point; this is the safety net for whatever it does not.
                    raise HTTPException(400, f"Invalid grammar: {e}")

            if req.stream:
                # Ownership of the pin transfers to _pin_engine, which releases it
                # when the stream ends - do NOT unpin in the finally below.
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse(engine, messages, req.model, sem,
                                audit=_audit, transcript=_transcript,
                                pipeline=pipeline, ctx=ctx, **gen_kwargs)),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        # F11: "used N memories" + recall degrade reason (the inlet
                        # already ran above, so ctx.state is populated). No-op when
                        # memory did not run this turn.
                        **_memory_used_header(ctx),
                    },
                )
            resp = await _complete(engine, messages, req.model, sem,
                                   audit=_audit, transcript=_transcript,
                                   pipeline=pipeline, ctx=ctx,
                                   request=request, **gen_kwargs)
            for _hk, _hv in _memory_used_header(ctx).items():
                resp.headers[_hk] = _hv          # F11: same surface, non-streaming
            return resp
        finally:
            if not streaming_handoff:
                _hs._unpin(engine)

    @app.post("/v1/embeddings", dependencies=[Depends(_require_auth)])
    async def embeddings(req: EmbeddingRequest):
        if not req.model:
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        # If the requested model is registered as model_type="embedding", OR it is
        # exactly the configured ``embedding_model`` (the dedicated on-device
        # embedder: Qwen3-Embedding, bge-small, ...), route directly to
        # embed_texts() - no chat engine lookup needed or wanted. This is the
        # primary path when the user has an embedding model selected but no chat
        # model loaded: get_engine would 503, and every indexing job would
        # silently fall back to BM25 (AGENTS.md rule 5).
        #
        # The configured-name check is NOT redundant with the registry check:
        # `setup-embeddings` registers a freshly downloaded known-key model (the
        # default "bge-small-en-v1.5") under a PREFIXED alias
        # ("embedding-bge-small-en-v1.5", cli/maintenance.py) to avoid clobbering
        # a user's own naming, but leaves the `embedding_model` config value as the
        # raw known-key string. `_make_self_embed` (rag/plug.py) always sends that
        # raw config value as `model`, so a registry-name-only check never matches
        # the default flow and 404s here - which is worse than the pre-fix
        # behavior when a chat model WAS loaded (it used to work via the loaded
        # engine's can_embed=False fallback). Matching on the configured name
        # directly, independent of what alias (if any) the registry uses, is what
        # actually fixes the "no chat model loaded" case for every setup, not only
        # one where the user explicitly picked an embedding model "from your
        # models" (which happens to already store the exact registry name).
        try:
            from localm.config import load_config, load_registry
            _reg = load_registry()
            _entry = _reg.get((req.model or "").strip()) if _reg else None
            _emb_cfg_name = str(load_config().get("embedding_model") or "").strip()
        except Exception:
            _entry = None
            _emb_cfg_name = ""
        _is_registered_embedder = isinstance(_entry, dict) and _entry.get("model_type") == "embedding"
        _is_configured_embedder = bool(_emb_cfg_name) and (req.model or "").strip() == _emb_cfg_name
        if _is_registered_embedder or _is_configured_embedder:
            from localm.inference.embedder import embed_texts, last_error
            loop = asyncio.get_running_loop()
            texts_emb = [req.input] if isinstance(req.input, str) else req.input
            fmt_emb = (req.encoding_format or "float").lower()
            if fmt_emb not in ("float", "base64"):
                raise HTTPException(
                    400,
                    f"Unsupported encoding_format {req.encoding_format!r}: "
                    "expected 'float' or 'base64'.")
            try:
                vecs_emb = await loop.run_in_executor(None, lambda: embed_texts(texts_emb))
            except RuntimeError as e:
                # The isolated embedder worker can hard-crash MID-embed (a native
                # GPU-backend fault, e.g. a missing rocBLAS/Tensile kernel library -
                # see IsolatedEmbedder.embed's docstring: this is re-raised, never
                # silently swallowed, per rule 5). Left uncaught, this reached the
                # caller as a bare Starlette "Internal Server Error" with no detail
                # at all - useless for diagnosing a GPU/runtime fault, and exactly
                # what rag/plug.py's _make_self_embed then logged verbatim to every
                # indexing job ("embeddings unavailable (Internal server error)").
                # Surface the real cause as a clean, actionable 503 instead.
                raise HTTPException(503, f"Embedding failed: {e}")
            if vecs_emb is None:
                why = last_error() or "embedding model unavailable"
                raise HTTPException(422, f"Embedding model unavailable ({why}). "
                                    "Run 'localm setup-embeddings'.")

            def _enc_emb(vec):
                if fmt_emb == "base64":
                    import base64
                    import struct
                    buf = struct.pack("<%df" % len(vec), *(float(x) for x in vec))
                    return base64.b64encode(buf).decode("ascii")
                return vec

            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": _enc_emb(vec)}
                    for i, vec in enumerate(vecs_emb)
                ],
                "model": req.model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        # Resolve WITHOUT forcing a chat-model load. A GGUF backend
        # (can_embed=False) embeds via the dedicated small embedder, so loading the
        # multi-GB chat model (and, under VRAM pressure, evicting the active one) is
        # pure waste (AUDIT-MED-13). Only a can_embed backend (HF) needs a real load.
        engine = await _hs.get_engine(req.model, load=False)
        if getattr(getattr(engine, "_backend", None), "can_embed", True):
            engine = await _hs.get_engine(req.model)
        _touch_activity(engine.display_name)

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

        sem = _hs._inference_sems.setdefault(engine.display_name, asyncio.Semaphore(1))
        loop = asyncio.get_running_loop()
        if isinstance(getattr(engine, "active_requests", None), int):
            engine.active_requests += 1
        try:
            async with sem:
                vecs = await loop.run_in_executor(None, lambda: engine.embed(texts))
        except NotImplementedError as e:
            raise HTTPException(422, str(e))
        finally:
            if isinstance(getattr(engine, "active_requests", None), int):
                engine.active_requests = max(0, engine.active_requests - 1)

        def _encode(vec):
            if fmt == "base64":
                import base64
                import struct
                buf = struct.pack("<%df" % len(vec), *(float(x) for x in vec))
                return base64.b64encode(buf).decode("ascii")
            return vec

        # Off the event loop: count_tokens is a native tokenizer call PER input, and
        # `input` is only bounded by the 160 MB body cap (millions of short strings),
        # so summing it on the single-threaded loop would freeze every other request
        # for the whole batch - an event-loop-block DoS. Offload it (the embed above
        # already runs in an executor).
        total_tokens = await loop.run_in_executor(
            None, lambda: sum(engine.count_tokens(t) for t in texts))
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": _encode(vec)}
                for i, vec in enumerate(vecs)
            ],
            "model": req.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }

    @app.post("/v1/completions", dependencies=[Depends(_require_auth)])
    async def completions(req: CompletionRequest, request: Request):
        if not req.model:
            raise HTTPException(400, "Model parameter is required and cannot be empty")
            
        engine = await _hs.get_engine(req.model)
        # Pin synchronously the instant we own the engine (AUDIT-CRIT-1); released
        # in the finally below, or by _pin_engine at stream end for a streaming
        # response.
        _hs._pin(engine)
        streaming_handoff = False
        try:
            _touch_activity(engine.display_name)

            # Wrap the prompt as a single user message so raw completions flow through
            # the SAME chat-pipeline hooks and audit/transcript as
            # /v1/chat/completions (B17). Otherwise this is a parallel, unguarded,
            # unaudited path to the model: plugin safety/transform inlet/stream/outlet
            # hooks never fire and nothing is recorded.
            messages = [{"role": "user", "content": req.prompt}]

            pipeline = getattr(request.app.state, "chat_pipeline", None)
            ctx = None
            if pipeline is not None:
                from localm.inference.http_server import caller_scopes, principal_id
                ctx = ChatHookContext(
                    model_id=req.model, stream=req.stream,
                    request_id=make_chunk_id(),
                    principal=principal_id(request),
                    scopes=tuple(caller_scopes(request) or ()),
                )
                ctx.state["client_id"] = request.headers.get("x-client-id", "")
                if pipeline.has("inlet"):
                    messages = await pipeline.run_inlet(messages, ctx)

            sem = _hs._inference_sems.setdefault(engine.display_name, asyncio.Semaphore(1))

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
            if req.grammar_lazy:
                # Same contract as /v1/chat/completions: lazy needs its triggers.
                if not req.grammar or not req.grammar_triggers:
                    raise HTTPException(
                        400, "grammar_lazy requires both grammar and grammar_triggers")
                gen_kwargs["grammar_lazy"] = True
                gen_kwargs["grammar_triggers"] = req.grammar_triggers

            # Same up-front grammar validation as /v1/chat/completions: a malformed
            # grammar is a clean 400, never a native fault that latches the silent
            # _grammar_unsupported degrade for every later request.
            if req.grammar:
                try:
                    # LM-FZ-001: a pure-Python structural check FIRST, unconditionally
                    # (no RPC, so it also covers the RunnerBusy-deferred path below,
                    # which otherwise skips straight to a generation-time native
                    # call). A grammar built of thousands of unmatched "(" drove
                    # llama.cpp's native GBNF parser into a real stack overflow -
                    # this rejects that shape before any of it reaches the parser.
                    check_grammar_structure(req.grammar)
                    engine.validate_grammar(req.grammar)
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # Defense in depth: a crashed/faulted native worker surfaces
                    # here as a bare RuntimeError (not InvalidGrammarError), which
                    # would otherwise escape as an unhandled 500. The structural
                    # check above should catch the confirmed crash shape before
                    # this point; this is the safety net for whatever it does not.
                    raise HTTPException(400, f"Invalid grammar: {e}")

            if req.stream:
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse_completion(engine, messages, req.model, sem,
                                           audit=_audit, transcript=_transcript,
                                           pipeline=pipeline, ctx=ctx, **gen_kwargs)),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            # Count tokens on the (possibly inlet-transformed) messages - what
            # inference actually sees, matching the chat path.
            prompt_tokens = engine.count_tokens(_messages_prompt_text(messages))

            async with sem:
                # Cancelable on client disconnect (same as /v1/chat/completions'
                # non-streaming path): an aborted request releases the per-model
                # _inference_lock instead of generating to end-of-budget and
                # blocking the next request to this model.
                text = await _generate_full(engine, messages, request, **gen_kwargs)

            # Outlet fully controls the returned content in the non-streaming path;
            # then record the exchange (audit + transcript), exactly like chat.
            if pipeline is not None and ctx is not None and pipeline.has("outlet"):
                text = await pipeline.run_outlet(text, messages, ctx)
            _audit_exchange(_audit, _transcript, messages, text)

            completion_tokens = engine.count_tokens(text)
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
        finally:
            if not streaming_handoff:
                _hs._unpin(engine)
