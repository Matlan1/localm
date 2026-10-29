# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared HTTP domain-error-to-response translation.

``route_errors`` is the decorator every HTTP-surfaced feature uses to run a
domain call, catch its known exception types and translate them to an
HTTPException with a specific status (web/voice routes, GUI discover routes,
memory routes, MCP plugin-management handlers). ``format_localm_error`` is the
shared formatting idiom they both need.
"""

from __future__ import annotations

import functools
from typing import Callable

from fastapi import HTTPException

# A mapping value is either a plain HTTP status code (message = str(exc)), or a
# callable exc -> (status, message) for the rare case one exception type maps
# to different statuses/messages depending on its own state (e.g. VoiceError's
# "needs-faster-whisper" code, which is 501 vs 422 depending on e.code).
ErrorSpec = "int | Callable[[Exception], tuple[int, str]]"


def route_errors(mapping: dict) -> Callable:
    """Decorator for an async FastAPI route handler: catch any exception type
    in *mapping* and raise the corresponding ``HTTPException`` instead.

    ``mapping`` is a plain dict literal at the call site, e.g.
    ``@route_errors({NetworkPolicyError: 403, Exception: lambda e: (502, f"...: {e}")})``
    - there is no central registry to maintain. Exception types are matched via
    ``isinstance`` in the mapping's iteration order (dict order = declaration
    order), so put a subclass before its base class and any catch-all
    ``Exception`` entry last. An ``HTTPException`` already raised by the
    handler body (e.g. a 400 for bad input) always passes through untouched,
    even under a catch-all ``Exception`` entry.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as exc:
                for exc_type, spec in mapping.items():
                    if isinstance(exc, exc_type):
                        if callable(spec):
                            status, message = spec(exc)
                        else:
                            status, message = spec, str(exc)
                        raise HTTPException(status, message) from exc
                raise

        return wrapper

    return decorator


def format_localm_error(e) -> str:
    """The ``LocalmError``-to-text idiom repeated four times verbatim in
    ``inference/routes/admin.py`` (APP-LIFECYCLE-3): trims the trailing colon
    when *e.reason* is empty."""
    return f"{e.summary}: {e.reason}".strip().strip(":").strip()


UNDECODABLE_BODY_MESSAGE = (
    "The request body is labelled JSON but is not valid UTF-8 text, so it cannot "
    "be JSON. Send a JSON document, or the right Content-Type for other data.")


FASTAPI_BODY_PARSE_DETAIL = "There was an error parsing the body"


def is_undecodable_body(exc: Exception) -> bool:
    """True for the HTTP 400 FastAPI itself raises when a body labelled JSON
    cannot be decoded as text (its detail is FastAPI's body-parse text and its
    cause is the ``UnicodeDecodeError``). A route's own 400 chained from a
    ``UnicodeDecodeError`` is not matched."""
    return (getattr(exc, "status_code", None) == 400
            and getattr(exc, "detail", None) == FASTAPI_BODY_PARSE_DETAIL
            and isinstance(exc.__cause__, UnicodeDecodeError))


def json_invalid_text(error: dict) -> str:
    """One sentence for a pydantic ``json_invalid`` validation *error*: the
    decoder's reason and the character position."""
    reason = (error.get("ctx") or {}).get("error") or "it could not be parsed"
    position = next((p for p in error.get("loc", ()) if isinstance(p, int)), None)
    where = f" (at character {position})" if position is not None else ""
    return f"The request body is not valid JSON: {reason}{where}."
