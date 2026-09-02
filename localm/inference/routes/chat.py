# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inference routes: chat completions, embeddings, and raw completions.

These are the heaviest routes - they read the live engine and inference semaphore
from the http_server module globals, the session-scoped audit/transcript from
ctx, and call the streaming/completion helpers that stay on http_server
(_stream_sse, _stream_sse_completion, _complete, _audit_exchange,
_messages_prompt_text, _protocol_messages_to_dicts).

NOTE on the local ``ctx``: each handler builds its own per-request
``ChatHookContext`` named ``ctx`` (the chat-pipeline turn context), so the
session-scoped audit/transcript are unpacked from the register ``ctx`` into
module-style locals (``_audit`` / ``_transcript``) once at the top of
register(), before any handler body shadows ``ctx``.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import localm.inference.http_server as _hs
from localm.inference.backends.base import (
    EmbedBatchTooLargeError, GrammarUnsupportedError, InvalidGrammarError,
    PretokenizerUnsafeInputError, TriggerValidatorUnavailableError,
    messages_contain_image,
)
from localm.inference.chat_pipeline import ChatHookContext
from localm.inference.gbnf import check_grammar_structure, validate_trigger_patterns
from localm.inference.protocol import (
    ChatRequest, CompletionRequest, EmbeddingRequest, make_chunk_id,
)


def register(app: FastAPI, ctx) -> None:
    # Session-scoped objects, unpacked before the handler bodies (each of which
    # rebinds a local ``ctx`` for its own chat-pipeline turn).
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
        # An empty model means "no preference" and resolves the same way the None
        # default does. Still a 400 when there is nothing to resolve to.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        engine = await _hs.get_engine(req.model)
        # Report the model that actually answered when the request named none.
        # Both an omitted field (None) and an explicit "" are falsy and fall through
        # to engine.display_name; an explicit "localm" echoes back unchanged.
        reported_model = req.model or engine.display_name
        # Pin the engine synchronously, before the inlet or any other await, so a
        # concurrent model load cannot evict it mid-request. Released in the finally
        # below, or by _pin_engine at stream end for a streaming response.
        _hs._pin(engine)
        streaming_handoff = False
        try:
            _touch_activity(engine.display_name)

            # Convert pydantic Messages to plain dicts for the backend
            messages = _protocol_messages_to_dicts(req.messages)

            # Chat-pipeline hooks. The inlet runs here, so token counting and
            # inference see the transformed messages. The per-request context is
            # built whenever a pipeline is present, so the inlet, stream and outlet
            # of this turn share one ctx.
            pipeline = getattr(request.app.state, "chat_pipeline", None)
            ctx = None
            if pipeline is not None:
                from localm.inference.http_server import caller_scopes, principal_id
                ctx = ChatHookContext(
                    # reported_model, not req.model: a hook that reads model_id to
                    # decide behaviour must see the model actually in use. An
                    # explicit "localm" passes through unchanged.
                    model_id=reported_model, stream=req.stream,
                    request_id=make_chunk_id(),
                    principal=principal_id(request),
                    scopes=tuple(caller_scopes(request) or ()),
                )
                ctx.state["client_id"] = request.headers.get("x-client-id", "")
                if pipeline.has("inlet"):
                    messages = await pipeline.run_inlet(messages, ctx)

            sem = _hs._inference_sems.setdefault(engine.display_name, asyncio.Semaphore(1))

            # Reject image input on a text-only model with a 400 instead of dropping
            # the picture. GGUF is always text-only; an unloaded HF model's
            # multimodal support is known only after loading, so load first.
            if messages_contain_image(messages) and not engine.supports_images:
                if not engine.loaded and engine.can_be_multimodal:
                    loop = asyncio.get_running_loop()
                    async with sem:
                        await loop.run_in_executor(None, engine.load)
                if not engine.supports_images:
                    # supports_images is False here, so an mmproj_path set on the
                    # backend means the projector failed to load; the guidance names
                    # that cause rather than "text-only".
                    from localm.model_manager import vision_input_guidance
                    backend = getattr(engine, "_backend", None)
                    mmproj_failed = bool(getattr(backend, "mmproj_path", None))
                    active_model_path = getattr(backend, "model_path", None)
                    raise HTTPException(400, vision_input_guidance(
                        mmproj_failed=mmproj_failed,
                        active_model_path=active_model_path))

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
                # A lazy grammar without its trigger patterns can never engage.
                if not req.grammar or not req.grammar_triggers:
                    raise HTTPException(
                        400, "grammar_lazy requires both grammar and grammar_triggers")
                # A caller-supplied trigger pattern reaches native std::regex matching
                # against an uncapped, growing buffer on every token, so it is rejected
                # up front. run_in_executor, not a direct call: the probe can block
                # until its timeout and must not hold the event loop.
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, validate_trigger_patterns, req.grammar_triggers)
                except TriggerValidatorUnavailableError as e:
                    # Handled before the InvalidGrammarError arm: the pattern was
                    # never checked. Status comes from the shared table.
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar trigger: {e}")
                gen_kwargs["grammar_lazy"] = True
                gen_kwargs["grammar_triggers"] = req.grammar_triggers

            # Reject a malformed grammar with a 400 up front, before streaming starts,
            # so both the stream and non-stream paths get a real 4xx.
            if req.grammar:
                try:
                    # Pure-Python structural check first and unconditionally, with no
                    # RPC, so it also covers the RunnerBusy-deferred path below.
                    # Rejects a deeply unbalanced grammar before the native GBNF
                    # parser sees it.
                    check_grammar_structure(req.grammar)
                    # Off the event loop, for the same reason the trigger probe
                    # above is: validate_grammar's backend RPC waits on the
                    # isolated model worker, so a direct call here would freeze
                    # the single event loop for every concurrent request, not
                    # just this one, for as long as that worker takes to answer.
                    # The executor re-raises in this coroutine, so every arm
                    # below still catches exactly what it caught before.
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: engine.validate_grammar(
                            req.grammar, lazy=bool(req.grammar_lazy)))
                except GrammarUnsupportedError as e:
                    # The backend cannot apply a grammar at all. Separate from the
                    # InvalidGrammarError arm below, and above the `if req.stream:`
                    # branch so the streaming and non-streaming paths get the same
                    # status and reason.
                    raise HTTPException(400, str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # A bare RuntimeError means the isolated worker crashed, timed
                    # out, or returned something unexpected while checking the
                    # grammar. Reported as a worker fault; the message does not claim
                    # the model will reload.
                    raise HTTPException(
                        503, f"Grammar validation failed: the model worker "
                        f"faulted ({e}).")

            # Pre-dispatch context capacity guard: count prompt tokens on the
            # inlet-transformed messages, compact when approaching the ceiling, and
            # reject an oversized request with HTTP 413.
            loop = asyncio.get_running_loop()
            try:
                prompt_tokens = await loop.run_in_executor(
                    None, engine.count_messages_tokens, messages)
            except PretokenizerUnsafeInputError as e:
                # Counting tokenizes, so this is where the refusal surfaces on an
                # ordinary chat: before the generation call whose own handler
                # would otherwise be the one to report it.
                raise HTTPException(400, str(e))

            capacity = engine.context_capacity()
            if (isinstance(capacity, int) and capacity > 0
                    and isinstance(prompt_tokens, int) and len(messages) > 3):
                buffer = max(2048, int(capacity * 0.10))
                if capacity - prompt_tokens < buffer:
                    from localm.inference.compact import compact_messages
                    def _gen_for_compact(ms: list[dict], max_t: int) -> str:
                        return "".join(engine.chat_stream(ms, max_tokens=max_t, temperature=0.3))
                    new_messages, changed = await loop.run_in_executor(
                        None, compact_messages, messages, _gen_for_compact)
                    if changed:
                        messages = list(new_messages)
                        prompt_tokens = await loop.run_in_executor(
                            None, engine.count_messages_tokens, messages)

            if (isinstance(capacity, int) and capacity > 0
                    and isinstance(prompt_tokens, int) and prompt_tokens > capacity):
                raise HTTPException(
                    413,
                    f"Prompt ({prompt_tokens} tokens) exceeds the model's maximum "
                    f"context capacity ({capacity} tokens). Start a new chat, "
                    f"or raise it:  localm config n_ctx_max 32768  (or set ctx_auto "
                    f"true to size it from free VRAM).")

            if req.stream:
                # Ownership of the pin transfers to _pin_engine, which releases it
                # when the stream ends - do NOT unpin in the finally below.
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse(engine, messages, reported_model, sem,
                                audit=_audit, transcript=_transcript,
                                pipeline=pipeline, ctx=ctx, prompt_tokens=prompt_tokens, **gen_kwargs)),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        # "used N memories" plus the recall degrade reason; the inlet
                        # already ran, so ctx.state is populated. No-op when memory
                        # did not run this turn.
                        **_memory_used_header(ctx),
                    },
                )
            resp = await _complete(engine, messages, reported_model, sem,
                                   audit=_audit, transcript=_transcript,
                                   pipeline=pipeline, ctx=ctx,
                                   request=request, prompt_tokens=prompt_tokens, **gen_kwargs)
            for _hk, _hv in _memory_used_header(ctx).items():
                resp.headers[_hk] = _hv          # same surface, non-streaming
            return resp
        finally:
            if not streaming_handoff:
                _hs._unpin(engine)

    @app.post("/v1/embeddings", dependencies=[Depends(_require_auth)])
    async def embeddings(req: EmbeddingRequest):
        # An empty model means "no preference": resolved below to the configured
        # embedder when one exists, otherwise through the general get_engine()
        # resolution. Refused only when there is nothing to fall back to.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        # Route straight to embed_texts(), with no chat engine lookup, when the
        # requested model is registered as model_type="embedding" or is exactly the
        # configured ``embedding_model``. Both checks are needed: setup-embeddings
        # registers under a prefixed alias while the ``embedding_model`` config value
        # stays the raw known-key string, which is what _make_self_embed sends.
        try:
            from localm.config import load_config, load_registry
            _reg = load_registry()
            _emb_cfg_name = str(load_config().get("embedding_model") or "").strip()
        except Exception:
            _reg = None
            _emb_cfg_name = ""
        # An omitted model resolves to the configured embedder when one exists.
        # Falls through to the general get_engine() resolution below only when no
        # embedder is configured at all.
        resolved_model = req.model or (_emb_cfg_name or None)
        _entry = _reg.get((resolved_model or "").strip()) if _reg else None
        _is_registered_embedder = isinstance(_entry, dict) and _entry.get("model_type") == "embedding"
        _is_configured_embedder = bool(_emb_cfg_name) and (resolved_model or "").strip() == _emb_cfg_name
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
            except PretokenizerUnsafeInputError as e:
                # A permanent property of this text against this model, not a
                # transient worker condition, so 400 rather than the 503 below:
                # retrying the same request cannot succeed.
                raise HTTPException(400, str(e))
            except RuntimeError as e:
                # The isolated embedder worker can hard-crash mid-embed on a native
                # GPU-backend fault, which IsolatedEmbedder.embed re-raises. Reported
                # as a 503 carrying the cause.
                raise HTTPException(503, f"Embedding failed: {e}")
            if vecs_emb is None:
                # Off the event loop: last_error() takes embedder._LOCK, which
                # get_embedder holds across a spawn plus a native load.
                why = await loop.run_in_executor(None, last_error) \
                    or "embedding model unavailable"
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
                # resolved_model is truthy here: both branches leading into this
                # block require an explicit match against it. req.model is None on
                # the omitted-model-resolved-to-the-configured-embedder path.
                "model": resolved_model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        # Resolve without forcing a chat-model load: a GGUF backend (can_embed=False)
        # embeds via the dedicated small embedder, so only a can_embed backend needs
        # a real load. resolved_model equals req.model when the client named one, and
        # is None otherwise.
        engine = await _hs.get_engine(resolved_model, load=False)
        if getattr(getattr(engine, "_backend", None), "can_embed", True):
            engine = await _hs.get_engine(resolved_model)
        _touch_activity(engine.display_name)

        # encoding_format: "float" returns plain JSON arrays, "base64" returns each
        # vector as a base64-encoded little-endian float32 buffer. Anything else is
        # rejected up front rather than downgraded to float.
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
        except EmbedBatchTooLargeError as e:
            raise HTTPException(413, str(e))
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

        # Off the event loop: count_tokens is a native tokenizer call per input, and
        # ``input`` is bounded only by the 160 MB body cap.
        total_tokens = await loop.run_in_executor(
            None, lambda: sum(engine.count_tokens(t) for t in texts))
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": _encode(vec)}
                for i, vec in enumerate(vecs)
            ],
            # Report the model that actually answered when the request named none:
            # resolved_model is None in that case and falls through to
            # engine.display_name. An explicit "localm" echoes back unchanged.
            "model": resolved_model or engine.display_name,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }

    @app.post("/v1/completions", dependencies=[Depends(_require_auth)])
    async def completions(req: CompletionRequest, request: Request):
        # Same resolution as /v1/chat/completions: an empty model is "no preference",
        # refused only when there is nothing to fall back to.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        engine = await _hs.get_engine(req.model)
        # Report the model that actually answered, same as the chat route above.
        reported_model = req.model or engine.display_name
        # Pin synchronously the instant we own the engine; released in the finally
        # below, or by _pin_engine at stream end for a streaming response.
        _hs._pin(engine)
        streaming_handoff = False
        try:
            _touch_activity(engine.display_name)

            # Wrap the prompt as a single user message so raw completions flow through
            # the same chat-pipeline hooks and audit/transcript as
            # /v1/chat/completions.
            messages = [{"role": "user", "content": req.prompt}]

            pipeline = getattr(request.app.state, "chat_pipeline", None)
            ctx = None
            if pipeline is not None:
                from localm.inference.http_server import caller_scopes, principal_id
                ctx = ChatHookContext(
                    # reported_model, not req.model: a hook that reads model_id to
                    # decide behaviour must see the model actually in use. An
                    # explicit "localm" passes through unchanged.
                    model_id=reported_model, stream=req.stream,
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
                # Same up-front trigger-pattern validation as /v1/chat/completions,
                # in an executor so the probe cannot hold the event loop.
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, validate_trigger_patterns, req.grammar_triggers)
                except TriggerValidatorUnavailableError as e:
                    # Same arm order and same table-derived status as
                    # /v1/chat/completions: "could not check" is a 503, not a 400.
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar trigger: {e}")
                gen_kwargs["grammar_lazy"] = True
                gen_kwargs["grammar_triggers"] = req.grammar_triggers

            # Same up-front grammar validation as /v1/chat/completions: a malformed
            # grammar is a 400, never a native fault that latches the silent
            # _grammar_unsupported degrade for every later request.
            if req.grammar:
                try:
                    # Pure-Python structural check first and unconditionally, with no
                    # RPC, so it also covers the RunnerBusy-deferred path below.
                    # Rejects a deeply unbalanced grammar before the native GBNF
                    # parser sees it.
                    check_grammar_structure(req.grammar)
                    # Same off-the-event-loop offload as /v1/chat/completions,
                    # for the same reason - see that route.
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: engine.validate_grammar(
                            req.grammar, lazy=bool(req.grammar_lazy)))
                except GrammarUnsupportedError as e:
                    # Same capability refusal as /v1/chat/completions: the backend
                    # cannot apply a grammar at all.
                    raise HTTPException(400, str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # Same fault attribution as /v1/chat/completions, including not
                    # claiming the model will reload.
                    raise HTTPException(
                        503, f"Grammar validation failed: the model worker "
                        f"faulted ({e}).")

            # Count tokens on the (possibly inlet-transformed) messages, matching the
            # chat path. Off the event loop: count_tokens is a native tokenizer call.
            loop = asyncio.get_running_loop()
            try:
                prompt_tokens = await loop.run_in_executor(
                    None, engine.count_tokens, _messages_prompt_text(messages))
            except PretokenizerUnsafeInputError as e:
                raise HTTPException(400, str(e))

            capacity = engine.context_capacity()
            if (isinstance(capacity, int) and capacity > 0
                    and isinstance(prompt_tokens, int) and prompt_tokens > capacity):
                raise HTTPException(
                    413,
                    f"Prompt ({prompt_tokens} tokens) exceeds the model's maximum "
                    f"context capacity ({capacity} tokens). Start a new chat, "
                    f"or raise it:  localm config n_ctx_max 32768  (or set ctx_auto "
                    f"true to size it from free VRAM).")

            if req.stream:
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse_completion(engine, messages, reported_model, sem,
                                           audit=_audit, transcript=_transcript,
                                           pipeline=pipeline, ctx=ctx, prompt_tokens=prompt_tokens, **gen_kwargs)),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            gen_error: Exception | None = None
            async with sem:
                # Cancelable on client disconnect: an aborted request releases the
                # per-model _inference_lock instead of generating to end-of-budget.
                # Backend errors map through the same table _complete uses; anything
                # else falls through to the generic backstop.
                try:
                    text = await _generate_full(engine, messages, request, **gen_kwargs)
                except _hs._BACKEND_ERROR_TYPES as e:
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except RuntimeError as e:
                    # A generation failure: not enough free VRAM for this prompt, a
                    # conversation that outgrew n_ctx_max, or a native decode error.
                    # Catches only RuntimeError, not Exception, so a broken engine
                    # (AttributeError) and CancelledError still propagate.
                    from localm.debuglog import logger as _dbg
                    _dbg.exception("non-streaming completion generation failed")
                    gen_error = e
                    # The shared renderer also scrubs machine paths out of the
                    # text, which a model-load RuntimeError carries verbatim.
                    text = _hs.inference_error_text(e)

            # The outlet controls the returned content on the non-streaming path,
            # except for a failed generation, whose error surfaces verbatim. Then
            # record the exchange (audit + transcript), exactly like chat.
            if gen_error is None and pipeline is not None and ctx is not None and pipeline.has("outlet"):
                text = await pipeline.run_outlet(text, messages, ctx)
            _audit_exchange(_audit, _transcript, messages, text)

            completion_tokens = await loop.run_in_executor(None, engine.count_tokens, text)
            ts  = int(time.time())
            cid = make_chunk_id()
            return {
                "id": cid,
                "object": "text_completion",
                "created": ts,
                "model": reported_model,
                # "error", not "stop", when generation failed, matching the terminal
                # frame the streaming twin emits.
                "choices": [{"text": text, "index": 0,
                             "finish_reason": "error" if gen_error is not None else "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        finally:
            if not streaming_handoff:
                _hs._unpin(engine)
