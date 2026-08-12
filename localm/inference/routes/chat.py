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
from localm.inference.backends.base import (
    EmbedBatchTooLargeError, GrammarUnsupportedError, InvalidGrammarError,
    TriggerValidatorUnavailableError, messages_contain_image,
)
from localm.inference.chat_pipeline import ChatHookContext
from localm.inference.gbnf import check_grammar_structure, validate_trigger_patterns
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
        # An empty model means "no preference" - exactly what this field's own
        # default (None, see protocol.ChatRequest) means - so resolve it the
        # same way instead of refusing it up front.
        #
        # Refusing here made get_engine's recovery chain UNREACHABLE from this
        # route. get_engine resolves an unnamed request through
        # `_active_model_name or _default_model_name` and reloads the result
        # (http_server.py), and its 503 even documents "the transient window
        # during an active-model eviction/unload where _active_model_name was
        # just cleared" - but nothing empty ever got that far, because this line
        # ran first. So after vram.evict_chat_for_embedder cleared the active
        # pointer to free VRAM for the embedder, a turn that did not name the
        # model got an instant 400 and the evicted model was never reloaded:
        # chat stayed dead until the user loaded it by hand from the Models page.
        # Live-reproduced on a real server against a real GGUF: post-eviction the
        # unnamed turn 400'd in 4 ms with no reload, while the named turn
        # reloaded and answered in 9.7 s.
        #
        # Still a 400 when there is genuinely nothing to resolve (started with no
        # model and none ever loaded): that request really is unserveable and the
        # caller does have to name one. Only the recoverable case changes.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        engine = await _hs.get_engine(req.model)
        # Report the model that ACTUALLY answered when the request named none.
        # model_id is echoed straight into the response envelope. Both shapes of
        # "unnamed" - the field omitted (Optional default is None) and an
        # explicit "" - are falsy and fall through to engine.display_name. Only
        # an explicit "localm" is a real value the client sent, so it still
        # echoes "localm" unchanged; that sentinel behaviour is deliberate. (The
        # default here used to BE the string "localm", which made an omitted
        # field indistinguishable from an explicit "localm" and defeated this
        # exact fallback - see the model field's own comment in protocol.py.)
        reported_model = req.model or engine.display_name
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
                    # reported_model, not req.model: a chat hook that reads
                    # model_id to decide behaviour (chat/plug.py's thinking
                    # inlet does) must see the model actually in use. An
                    # unnamed request could not reach a hook at all before,
                    # so passing the raw empty string here would silently
                    # disable that handling on exactly the recovery path this
                    # change opened. "localm" still passes through unchanged.
                    model_id=reported_model, stream=req.stream,
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
                # Reject the half-formed request instead of silently degrading: a
                # lazy grammar without its trigger patterns can never engage.
                if not req.grammar or not req.grammar_triggers:
                    raise HTTPException(
                        400, "grammar_lazy requires both grammar and grammar_triggers")
                # GitHub #928/#933: a trigger pattern reaches native std::regex
                # matching against an uncapped, ever-growing buffer on every
                # token. localm's own pattern was catastrophically backtracking-
                # prone; a CALLER-supplied one (this field) is unvalidated caller
                # input reaching the identical path, so it gets the identical
                # up-front rejection before any of it reaches generation.
                #
                # run_in_executor, not a direct call: an unsafe pattern's probe
                # can legitimately take up to _TRIGGER_PROBE_TIMEOUT/_SPAWN_TIMEOUT
                # seconds to time out (see gbnf.py). Calling that synchronously
                # here would block THIS event loop for the whole wait, freezing
                # every OTHER concurrent request on this server for one caller's
                # bad pattern - exactly the class of failure this whole defense
                # exists to prevent, just moved from the native layer to this one.
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, validate_trigger_patterns, req.grammar_triggers)
                except TriggerValidatorUnavailableError as e:
                    # BEFORE the InvalidGrammarError arm, because it IS one. The
                    # pattern was never checked (the probe pool was saturated, or
                    # its daemon could not be reached), so 400 would blame the
                    # caller for a condition on this side of the wire that a
                    # retry can clear. Still a refusal: an unproven pattern never
                    # reaches the native sampler.
                    #
                    # The status comes FROM the shared table, not from a literal
                    # here. Writing 503 twice would make the table's documented
                    # arm ordering decorative at this call site: reordering it
                    # (which IS the hazard the table warns about, and which has
                    # its own test) would silently stop mattering here while the
                    # test still passed. Measured exactly that way round before
                    # this line derived its status.
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar trigger: {e}")
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
                except GrammarUnsupportedError as e:
                    # The backend cannot apply a grammar AT ALL (a HuggingFace
                    # model without the [grammar] extra). Refusing here is the
                    # whole point: this used to be a silent no-op, so the request
                    # ran to completion and returned unconstrained text that the
                    # caller had no way to tell from a grammar-conformant answer.
                    #
                    # NOT folded into the InvalidGrammarError arm below, even
                    # though both are 400: that message says "Invalid grammar",
                    # which would send the caller to fix a grammar that is
                    # perfectly good. Same wrong-thing-to-fix failure the worker
                    # -fault arm further down was written to correct.
                    #
                    # This sits ABOVE the `if req.stream:` branch, so the
                    # streaming and non-streaming paths get the identical status
                    # and the identical reason - the refusal happens before a
                    # single byte of either response is committed.
                    raise HTTPException(400, str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # A bare RuntimeError here (as opposed to InvalidGrammarError
                    # above) means the isolated worker process crashed, timed
                    # out, or returned something unexpected while checking the
                    # grammar (see ModelRunner._simple_request) - the grammar
                    # itself is not the problem. Labeling this "Invalid grammar"
                    # told the caller to fix the wrong thing, so report it as
                    # the worker fault it is (503, matching how /v1/embeddings
                    # reports the identical isolated-worker-crash shape below).
                    #
                    # Do NOT claim the model will reload: _simple_request has
                    # FOUR RuntimeError shapes and only two of them actually
                    # kill the worker (a confirmed-dead process, or the
                    # explicit shutdown() on timeout) - the other two (the
                    # worker's own generic `except Exception` catch replying
                    # untagged, or an unexpected response) leave it alive, so
                    # GgufBackend.loaded stays True and the SAME worker serves
                    # the next request. A prior version of this message
                    # promised a reload unconditionally; that was false for
                    # half the cases and has been removed rather than fixed
                    # with a check this file cannot make (which sub-case
                    # occurred is only known inside ModelRunner).
                    #
                    # This also does not fully close the #964 visibility gap:
                    # it helps any client that reads the response body, but
                    # the coder plugin's own HTTP client (localm/plugins/
                    # coder/backends/http.py's _raise_for_status) does not
                    # read the body except for 401/403, and 503 is in its
                    # retry set (400 was not) - so for that specific caller
                    # this trades an instant, mislabeled failure for up to
                    # ~30s of retries ending in the same detail-less message.
                    # That gap is in a file this session does not own; fixing
                    # it is tracked separately.
                    raise HTTPException(
                        503, f"Grammar validation failed: the model worker "
                        f"faulted ({e}).")

            if req.stream:
                # Ownership of the pin transfers to _pin_engine, which releases it
                # when the stream ends - do NOT unpin in the finally below.
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse(engine, messages, reported_model, sem,
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
            resp = await _complete(engine, messages, reported_model, sem,
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
        # An empty model means "no preference", exactly like /v1/chat/completions
        # and /v1/completions (see chat_completions above for the full
        # rationale) - refuse only when there is genuinely nothing to fall back
        # to. Unlike those two routes, "no preference" is resolved a few lines
        # below to the CONFIGURED embedder first when one exists (see
        # resolved_model's own comment) rather than straight to whatever chat
        # model happens to be active, because unlike chat, an embedding from
        # the wrong model is not an error - it is silently wrong. Only when no
        # embedder is configured at all does an unnamed request reach the
        # general get_engine() resolution further down, and even then
        # Engine.embed() itself falls back to the dedicated embedder (or
        # raises a clean NotImplementedError, caught below as a 422) when the
        # resolved engine cannot embed - so this gate relaxation cannot turn
        # into a silently-wrong 200 either way.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
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
            _emb_cfg_name = str(load_config().get("embedding_model") or "").strip()
        except Exception:
            _reg = None
            _emb_cfg_name = ""
        # An omitted model must resolve to the CONFIGURED embedder when one
        # exists, not to whatever chat model happens to be active. Unlike chat,
        # embeddings from different models are not comparable (the same hazard
        # NEW-RAG-DIM-NO-REEMBED tracks for a deliberate model switch) - and
        # "active model" is shared, mutable state that unrelated chat activity
        # can change between two otherwise-identical requests. Without this, an
        # unnamed request would fall through to the general get_engine()
        # resolution below, which is only safe by ACCIDENT: it happens to be
        # deterministic whenever the active model's backend cannot itself embed
        # (every GGUF, and any HF chat decoder - both always route through
        # Engine.embed()'s own dedicated-embedder fallback regardless of WHICH
        # such model is active), but is not deterministic if the active model
        # is itself an embedding-capable HF encoder. Falls through to that
        # general resolution only when no embedder is configured at all - at
        # that point there is nothing dedicated to prefer, and this is the same
        # best-effort "use whatever's loaded" fallback chat/completions use.
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
                # resolved_model is guaranteed truthy here (both branches above
                # that lead into this block require an explicit match against
                # it) - report it rather than req.model, which is None on the
                # omitted-model-resolved-to-the-configured-embedder path.
                "model": resolved_model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        # Resolve WITHOUT forcing a chat-model load. A GGUF backend
        # (can_embed=False) embeds via the dedicated small embedder, so loading the
        # multi-GB chat model (and, under VRAM pressure, evicting the active one) is
        # pure waste (AUDIT-MED-13). Only a can_embed backend (HF) needs a real load.
        # resolved_model, not req.model: identical when the client named a model
        # explicitly, but resolved_model is also correctly None here rather than
        # a leftover configured-embedder name (that case always took the branch
        # above and returned already - see resolved_model's own comment).
        engine = await _hs.get_engine(resolved_model, load=False)
        if getattr(getattr(engine, "_backend", None), "can_embed", True):
            engine = await _hs.get_engine(resolved_model)
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
            # Report the model that actually answered when the request named
            # none, same as /v1/chat/completions and /v1/completions - an
            # omitted model (resolved_model falls back to req.model here, and
            # is None in that case too) falls through to engine.display_name;
            # an explicit "localm" is a real value the client sent and still
            # echoes back unchanged.
            "model": resolved_model or engine.display_name,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }

    @app.post("/v1/completions", dependencies=[Depends(_require_auth)])
    async def completions(req: CompletionRequest, request: Request):
        # Same resolution as /v1/chat/completions above, for the same reason: an
        # empty model is "no preference", and refusing it here would leave this
        # route unable to recover an evicted model too. Fixing only the chat
        # route would leave the identical hole one endpoint over.
        if not req.model and not (_hs._active_model_name or _hs._default_model_name):
            raise HTTPException(400, "Model parameter is required and cannot be empty")

        engine = await _hs.get_engine(req.model)
        # Report the model that actually answered, same as the chat route above.
        reported_model = req.model or engine.display_name
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
                    # reported_model, not req.model: a chat hook that reads
                    # model_id to decide behaviour (chat/plug.py's thinking
                    # inlet does) must see the model actually in use. An
                    # unnamed request could not reach a hook at all before,
                    # so passing the raw empty string here would silently
                    # disable that handling on exactly the recovery path this
                    # change opened. "localm" still passes through unchanged.
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
                # same run_in_executor reasoning (a probe can block for real
                # seconds; a direct call would freeze this whole event loop for
                # every concurrent request, not just this one) - see that route.
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, validate_trigger_patterns, req.grammar_triggers)
                except TriggerValidatorUnavailableError as e:
                    # Same arm order, same reasoning, and same table-derived
                    # status as /v1/chat/completions: "could not check" is a 503,
                    # not a 400 blaming the pattern.
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar trigger: {e}")
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
                except GrammarUnsupportedError as e:
                    # Same capability refusal as /v1/chat/completions above, for
                    # the same reason: without it a grammar request against a
                    # backend that cannot apply one returns unconstrained text
                    # with a 200 and no signal. See that route for the full note.
                    raise HTTPException(400, str(e))
                except InvalidGrammarError as e:
                    raise HTTPException(400, f"Invalid grammar: {e}")
                except RuntimeError as e:
                    # Same fault-attribution fix as /v1/chat/completions above,
                    # including NOT claiming the model will reload (only two
                    # of _simple_request's four RuntimeError shapes actually
                    # kill the worker - see that comment for the full
                    # reasoning and the coder-plugin visibility caveat).
                    raise HTTPException(
                        503, f"Grammar validation failed: the model worker "
                        f"faulted ({e}).")

            if req.stream:
                streaming_handoff = True
                return StreamingResponse(
                    _pin_engine(engine, _stream_sse_completion(engine, messages, reported_model, sem,
                                           audit=_audit, transcript=_transcript,
                                           pipeline=pipeline, ctx=ctx, **gen_kwargs)),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            # Count tokens on the (possibly inlet-transformed) messages - what
            # inference actually sees, matching the chat path. Off the event
            # loop: count_tokens is a native tokenizer call, same reasoning as
            # the /v1/embeddings usage count above (this route's own
            # docstring-equivalent - a direct call here freezes every other
            # request for the duration of the native call).
            loop = asyncio.get_running_loop()
            prompt_tokens = await loop.run_in_executor(
                None, engine.count_tokens, _messages_prompt_text(messages))

            gen_error: Exception | None = None
            async with sem:
                # Cancelable on client disconnect (same as /v1/chat/completions'
                # non-streaming path): an aborted request releases the per-model
                # _inference_lock instead of generating to end-of-budget and
                # blocking the next request to this model.
                #
                # The same backend error contract as _complete, from the same
                # table, so the two non-streaming handlers cannot answer the
                # identical failure differently. This route had NO exception
                # handling here at all, so every backend refusal - an image this
                # model could not process, a grammar rejected at sampler-build
                # time - reached the generic backstop as an opaque 500 with the
                # reason discarded, while /v1/chat/completions' streaming twin
                # reported it. A bug that is not one of these still falls through
                # to that backstop, deliberately: see backend_error_status.
                try:
                    text = await _generate_full(engine, messages, request, **gen_kwargs)
                except _hs._BACKEND_ERROR_TYPES as e:
                    raise HTTPException(_hs.backend_error_status(e), str(e))
                except RuntimeError as e:
                    # A generation FAILURE (not enough free VRAM for this prompt,
                    # a conversation that outgrew n_ctx_max, a native decode
                    # error) - the LAST of the four generation paths to get this
                    # arm. The other three already render it inline: _complete
                    # for non-streaming chat, and both streaming legs via their
                    # gen_error handling. This one had no arm, so the reason was
                    # discarded by the generic backstop and the caller got
                    # {"detail": "Internal server error"} for a failure its OWN
                    # streaming twin reports in full.
                    #
                    # Catch ONLY RuntimeError, not Exception, for the same reason
                    # _complete does: a broken engine (a method-less mock ->
                    # AttributeError) is a real bug that must surface loudly
                    # rather than be dressed up as an "inference error" (rule 5),
                    # and CancelledError (client disconnect) must not be
                    # swallowed either.
                    from localm.debuglog import logger as _dbg
                    _dbg.exception("non-streaming completion generation failed")
                    gen_error = e
                    text = f"\n[inference error: {e}]"

            # Outlet fully controls the returned content in the non-streaming path
            # (but a failed generation surfaces its error verbatim, not reshaped by
            # the outlet - same carve-out as _complete);
            # then record the exchange (audit + transcript), exactly like chat.
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
                # "error", not "stop", when generation failed - the machine-
                # detectable half of the contract, and the whole reason a 200
                # carrying an error text is honest rather than a lie. Its own
                # streaming twin already marks its terminal frame this way
                # (http_server.py's _stream_sse_completion), so a client that
                # already handles the stream needs no change to handle this.
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
