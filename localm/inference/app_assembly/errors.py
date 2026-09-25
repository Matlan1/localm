# SPDX-License-Identifier: AGPL-3.0-or-later
"""The exception handlers ``create_app()`` installs: a generic, logged 500 for
anything unhandled, a debug-log seam in front of FastAPI's own HTTPException
handler, and a 422 formatter that stays serializable and bounded for any body."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Install the three handlers. The response of each is pinned by
    tests/test_create_app_characterization.py."""
    # One backstop so an unexpected error in ANY route returns a consistent
    # JSON 500 and is logged, instead of leaking a traceback or bare body. This
    # standardises the response shape and logging so a failing request is a clean
    # 500, never a crash or info leak. (A native fault - a C-extension segfault -
    # cannot be caught in-process; those are prevented at the source, e.g. voice
    # audio is validated before the native path, and surfaced via the crash marker.)
    @app.exception_handler(Exception)
    async def _unhandled_error(request, exc):  # noqa: ANN001 - framework signature
        from localm.debuglog import logger as _dbg
        _dbg.exception("unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error"})

    # A refusal carries its REASON in the HTTPException detail, and that detail
    # is logged next to the status so a refusal is self-diagnosing from the debug
    # log alone rather than leaving only a status and a timing.
    #
    # Gated on debug_enabled() like the request/timing lines it sits beside, NOT
    # on debug_content_enabled(): an HTTPException detail is server-authored
    # operational text (which validation refused, which capability is missing),
    # never chat content. Nothing here reads the request body. See
    # docs/privacy.md and the same debug_enabled() gate on the request log
    # (app_assembly/diagnostics.py).
    #
    # Registered for starlette's HTTPException (fastapi's subclasses it, and
    # fastapi registers the starlette class as its own key), then DELEGATED to
    # fastapi's own handler so the response - status, body shape, and any
    # WWW-Authenticate / Retry-After headers - stays byte-identical. This is a
    # logging seam, not a response change.
    from starlette.exceptions import HTTPException as _StarletteHTTPException

    @app.exception_handler(_StarletteHTTPException)
    async def _log_http_exception(request, exc):  # noqa: ANN001 - framework signature
        from fastapi.exception_handlers import http_exception_handler
        from localm.debuglog import debug_enabled, logger as _dbg
        if debug_enabled():
            # Truncated: a detail can be long by design (the VRAM-overflow 503
            # carries a multi-line "Options:" list), and the point here is to
            # name the cause, not to mirror the whole body into the log.
            detail = str(getattr(exc, "detail", "") or "")
            if len(detail) > 500:
                detail = detail[:500] + " ...[truncated]"
            _dbg.debug("%s %s refused %d: %s", request.method,
                       request.url.path, exc.status_code, detail)
        return await http_exception_handler(request, exc)

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):  # noqa: ANN001 - framework signature
        # A 422 body must stay serializable. pydantic records the offending value
        # under `input`; when a client sends a NON-FINITE number (NaN / Infinity),
        # Starlette's JSONResponse serializes the error with allow_nan=False and
        # CRASHES into a 500 - so a bad numeric param turned a clean 422 into an
        # unhandled 500 (live-confirmed: `top_k: NaN`, `seed: NaN`, and any float
        # field once allow_inf_nan=False rejects it). Replace non-finite floats in
        # the error detail so the 422 always renders. Same shape as FastAPI's
        # default handler for every other (finite) validation error.
        # The 422 body must also stay BOUNDED. pydantic records the offending
        # value under `input` verbatim, so a deeply-nested body puts that nesting
        # in the error object - and `jsonable_encoder` walks it recursively. A
        # ~2 KB body of `[[[[...]]]]` therefore raised RecursionError INSIDE this
        # handler, so FastAPI could not build a response at all and the documented
        # 422 became an opaque 500. Measured: a window around 961 to ~2900 levels
        # (shallower parses and validates cleanly; deeper is refused by the JSON
        # parser's own depth limit first), ~0.25 s of event-loop CPU per request,
        # and a 147x latency rise on unrelated requests under four connections.
        # Same failure the NaN note above describes, by a different route.
        import math

        from fastapi.encoders import jsonable_encoder

        _MAX_ERR_DEPTH = 20     # far past anything a real API request nests
        _ELIDED = "...[nested value elided]"

        def _depth_capped(v, depth: int = 0):
            """Prune the error object BEFORE `jsonable_encoder` ever sees it.

            The ORDER is the fix. Pruning afterwards cannot work, because the
            encoder is what recurses: it would blow the stack before any depth
            limit downstream of it got the chance to apply."""
            if depth >= _MAX_ERR_DEPTH:
                return _ELIDED
            if isinstance(v, dict):
                return {k: _depth_capped(x, depth + 1) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_depth_capped(x, depth + 1) for x in v]
            return v

        def _finite_safe(v, depth: int = 0):
            if depth >= _MAX_ERR_DEPTH:
                return _ELIDED
            if isinstance(v, float) and not math.isfinite(v):
                return repr(v)      # "nan" / "inf" / "-inf"
            if isinstance(v, dict):
                return {k: _finite_safe(x, depth + 1) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_finite_safe(x, depth + 1) for x in v]
            return v

        # _finite_safe runs AFTER the encoder: the encoder itself can PRODUCE a
        # non-finite float (a Decimal("NaN") becomes float("nan")), so running it
        # before would let a NaN reach the response and 500. Only the DEPTH prune
        # runs ahead of the encoder.
        try:
            safe = _finite_safe(jsonable_encoder(_depth_capped(exc.errors())))
        except RecursionError:
            # SAFETY NET, not the fix: it catches a shape the prune above cannot
            # reach (a deeply nested object that is not a dict/list/tuple, which
            # passes through untouched and the encoder then recurses into). The
            # prune stays: this catch still pays the full CPU cost of the
            # recursion. Warned rather than silenced, since reaching it means a
            # shape got past the prune.
            from localm.debuglog import logger as _dbg
            _dbg.warning("validation error for %s was too deeply nested to "
                         "encode even after depth-capping; returned a 422 "
                         "without the structured detail", request.url.path)
            safe = []

        def _field(err) -> str:
            # Drop the "body"/"query" container so the user sees the name they
            # actually typed ("max_tokens"), not pydantic's full path.
            parts = [str(p) for p in (err.get("loc") or ())
                     if p not in ("body", "query", "path", "header")]
            return ".".join(parts) or "request"

        def _one(err) -> str:
            name = _field(err)
            msg = (err.get("msg") or "is invalid").strip()
            # pydantic phrases these as "Input should be X", which reads as
            # "max_tokens input should be X" once the field name is prepended.
            # "max_tokens must be X" is the same information, in English.
            if msg.lower().startswith("input should be "):
                msg = "must be " + msg[len("input should be "):]
            elif msg[:1].isupper():
                msg = msg[0].lower() + msg[1:]
            # On a MISSING field pydantic reports the whole request body as
            # `input`, so echoing it back is noise, not evidence.
            got = ("" if err.get("type") == "missing" or "input" not in err
                   else f" (got {err.get('input')!r})")
            # max_tokens=0 gets its own message: it reads as a legitimate "no
            # limit" but collides with the engine's internal unlimited sentinel,
            # which would turn it into an unbounded generation.
            if name == "max_tokens" and err.get("input") in (0, "0"):
                return ("max_tokens must be 1 or more - 0 is not 'no limit'. "
                        "Omit max_tokens entirely to use the model's default")
            return f"{name} {msg}{got}"

        # `detail` is the human sentence, because every client in this repo does
        # `data.detail || r.statusText` and stringifying pydantic's error LIST
        # there is what produced the unreadable dump users were shown. The
        # structured form is preserved verbatim under `errors` for anything that
        # wants to parse it - nothing is lost, it just stops being the thing a
        # person reads.
        try:
            summary = "; ".join(_one(e) for e in (safe or []))
        except Exception:   # never let error FORMATTING turn a 422 into a 500
            summary = ""
        return JSONResponse(
            status_code=422,
            content={"detail": summary or "Request validation failed",
                     "errors": safe})
