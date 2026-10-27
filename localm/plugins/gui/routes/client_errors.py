# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI client-error route: ``POST /api/gui-error`` records an error the browser
showed or threw on the ``localm`` logger at WARNING, prefixed ``gui:``, so it
reaches the in-memory recent-activity buffer (``/api/activity``, the debug
window) and, in debug mode, the debug log file.

Every field is untrusted: the body is size-capped, each field is length-capped,
ANSI escape sequences, control characters, line and paragraph separators and
bidi overrides are removed, and home paths and credentials are scrubbed before
anything is logged. Reports are rate-limited per credential; the reports dropped
by the limit are counted in one line instead of being logged one by one.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from localm.inference.http_server import SESSION_COOKIE, _bearer_token, _require_auth

MAX_BODY_BYTES = 8192
MAX_MESSAGE_CHARS = 1000
MAX_SOURCE_CHARS = 64
MAX_PAGE_CHARS = 128
RATE_LIMIT_COUNT = 30
RATE_LIMIT_WINDOW_S = 60.0
MAX_TRACKED_CLIENTS = 256

# CSI (ESC [ ... final byte), OSC (ESC ] ... BEL or ESC \), and any other
# two-byte ESC sequence.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b.?")
# C0 and C1 controls, DEL, the Unicode line/paragraph separators, and the bidi
# embedding/override/isolate characters.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")
_SPACES_RE = re.compile(r" {2,}")


def clean_field(value, limit: int) -> str:
    """*value* as one log-safe line of at most *limit* characters: a non-string
    becomes "", ANSI sequences are removed, every control or separator
    character becomes a space, runs of spaces collapse, and an over-long result
    is cut with a trailing "...". Never raises."""
    if not isinstance(value, str):
        return ""
    text = _ANSI_RE.sub("", value[: limit * 4])
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _scrub(text: str) -> Optional[str]:
    """*text* with home paths, URL credentials, credential-named parameters,
    bearer tokens, API keys, email addresses and machine names redacted, or
    None when the scrubber itself fails (the caller then logs no text)."""
    try:
        from localm.bugreport.scrub import _scrub_secrets
        return _scrub_secrets(text)
    except Exception:
        return None


class _RateLimiter:
    """Fixed-window counter per client key. Thread-safe."""

    def __init__(self, count: int, window_s: float, max_clients: int) -> None:
        self._count = count
        self._window = window_s
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._state: dict[str, list] = {}

    def admit(self, key: str, now: float) -> tuple[bool, int, bool]:
        """Count one report from *key* at *now*. Returns ``(admitted,
        dropped_before, first_drop)``: *dropped_before* is the number of
        reports dropped in the previous window, reported once on the first
        admitted report after it; *first_drop* is True for the first report
        dropped in a window."""
        with self._lock:
            st = self._state.get(key)
            if st is None:
                if len(self._state) >= self._max_clients:
                    self._state.pop(next(iter(self._state)))
                st = [now, 0, 0]
                self._state[key] = st
            carried = 0
            if now - st[0] >= self._window:
                carried = st[2]
                st[0], st[1], st[2] = now, 0, 0
            if st[1] >= self._count:
                st[2] += 1
                return False, 0, st[2] == 1
            st[1] += 1
            return True, carried, False


def _client_key(request: Request) -> str:
    """A short digest of the credential the request presented (bearer token,
    X-API-Key or session cookie), or "anonymous" when it presented none."""
    token = (_bearer_token(request)
             or (request.headers.get("x-api-key") or "").strip()
             or (request.cookies.get(SESSION_COOKIE) or "").strip())
    if not token:
        return "anonymous"
    return hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16]


async def _read_capped(request: Request, limit: int) -> Optional[bytes]:
    """The request body, or None once it exceeds *limit* bytes."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                return None
        except ValueError:
            return None
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def register(app: FastAPI, ctx) -> None:
    limiter = _RateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_S, MAX_TRACKED_CLIENTS)
    app.state.gui_error_limiter = limiter

    @app.post("/api/gui-error", dependencies=[Depends(_require_auth)])
    async def gui_error(request: Request):
        """Log one GUI error report. Body: JSON ``{message, source, page,
        time}``; only ``message`` is required, ``time`` is accepted and not
        logged (the log line carries the server's own time).

        Returns 200 ``{"logged": true}``; 400 for a body that is not a JSON
        object or has no message; 413 above ``MAX_BODY_BYTES``; 429
        ``{"logged": false}`` while the caller is over the rate limit."""
        import json

        from localm.debuglog import logger as _dbg

        raw = await _read_capped(request, MAX_BODY_BYTES)
        if raw is None:
            return JSONResponse(status_code=413, content={"detail": "Error report too large."})
        try:
            body = json.loads(raw.decode("utf-8", "replace") or "null")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "Expected a JSON object."})
        message = clean_field(body.get("message"), MAX_MESSAGE_CHARS)
        if not message:
            return JSONResponse(status_code=400, content={"detail": "message is required."})
        source = clean_field(body.get("source"), MAX_SOURCE_CHARS) or "gui"
        page = clean_field(body.get("page"), MAX_PAGE_CHARS)

        admitted, dropped_before, first_drop = limiter.admit(
            _client_key(request), time.monotonic())
        if not admitted:
            if first_drop:
                _dbg.warning("gui: too many GUI error reports; dropping further "
                             "reports for up to %d s", int(RATE_LIMIT_WINDOW_S))
            return JSONResponse(status_code=429, content={"logged": False})
        if dropped_before:
            _dbg.warning("gui: %d GUI error report(s) were dropped by the rate limit",
                         dropped_before)

        where = f"[{source}{' ' + page if page else ''}]"
        line = _scrub(f"{where} {message}")
        if line is None:
            _dbg.warning("gui: %s error report not logged: the scrubber failed", source)
            return {"logged": False}
        _dbg.warning("gui: %s", line)
        return {"logged": True}
