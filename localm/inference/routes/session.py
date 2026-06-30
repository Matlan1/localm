# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session auth routes (S2): HttpOnly cookie + CSRF for the browser GUI.

Extracted verbatim from create_app(); behavior unchanged. The session cookie
names, max age, and the token/CSRF helpers live on the http_server module and are
referenced via ``_hs.`` so external importers still find them there.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, Response

import localm.inference.http_server as _hs
from localm import scopes


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    _bearer_token = _hs._bearer_token
    _request_token = _hs._request_token
    _csrf_double_submit_ok = _hs._csrf_double_submit_ok

    @app.post("/api/session", include_in_schema=False)
    async def session_login(request: Request, response: Response):
        """Exchange the API key for an HttpOnly session cookie so the browser
        GUI never has to hold the key in JS-readable localStorage. The key is
        read from the JSON body ``{"key": ...}`` or an Authorization: Bearer
        header, verified, and on success set as the HttpOnly ``localm_session``
        cookie plus a readable ``localm_csrf`` cookie (double-submit CSRF). 401
        on a bad key; 400 in open mode (nothing to log into)."""
        from localm.auth import any_key_configured, verify
        presented = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                presented = (body.get("key") or "").strip() or None
        except Exception:
            presented = None
        if not presented:
            presented = _bearer_token(request)
        if not any_key_configured():
            raise HTTPException(
                400, "This server runs in open mode (no API key configured); "
                "there is nothing to log into.")
        held = verify(presented) if presented else None
        if held is None:
            raise HTTPException(401, "Invalid API key")
        secure = request.url.scheme == "https"
        csrf = secrets.token_urlsafe(32)
        response.set_cookie(_hs.SESSION_COOKIE, presented, httponly=True,
                            secure=secure, samesite="strict", path="/",
                            max_age=_hs.SESSION_MAX_AGE)
        response.set_cookie(_hs.CSRF_COOKIE, csrf, httponly=False,
                            secure=secure, samesite="strict", path="/",
                            max_age=_hs.SESSION_MAX_AGE)
        return {"authed": True, "scopes": sorted(held)}

    @app.post("/api/session/logout", include_in_schema=False)
    async def session_logout(request: Request, response: Response):
        """Clear the session + CSRF cookies (sign the browser out).

        A POST from the cookie-authenticated browser GUI is subject to the
        same double-submit CSRF check as every other state-changing endpoint
        (Task 3: CSRF-post-clear). A bearer-header caller (CLI / SDK) is
        CSRF-exempt because it cannot be driven cross-site."""
        token, source = _request_token(request)
        if source == "cookie" and not _csrf_double_submit_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Include the localm_csrf "
                "cookie value in the X-CSRF-Token header.")
        secure = request.url.scheme == "https"
        response.delete_cookie(_hs.SESSION_COOKIE, path="/", httponly=True, secure=secure, samesite="strict")
        response.delete_cookie(_hs.CSRF_COOKIE, path="/", httponly=False, secure=secure, samesite="strict")
        return {"authed": False}

    @app.post("/api/auth/key/clear",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))],
              include_in_schema=False)
    async def clear_owner_key(request: Request, response: Response):
        """Delete the server-side owner key (auth.key) and immediately
        invalidate the caller's session cookie.

        CSRF-protected: a cookie-authenticated caller must echo the
        localm_csrf cookie value in X-CSRF-Token (double-submit pattern).
        After this call any_key_configured() returns False (assuming no
        LOCALM_API_KEY env var and an empty keystore), and all subsequent
        web UI requests that rely on the session cookie will fail auth and
        be redirected to the key gate by the client (Task 3: CSRF-post-clear).
        """
        _, source = _request_token(request)
        if source == "cookie" and not _csrf_double_submit_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Include the localm_csrf "
                "cookie value in the X-CSRF-Token header.")
        from localm.auth import clear_api_key
        clear_api_key()
        secure = request.url.scheme == "https"
        # Invalidate the session cookie immediately so the browser is forced
        # back to the key gate on the next navigation (the old cookie value
        # no longer matches any key). Also clear the CSRF token.
        response.delete_cookie(_hs.SESSION_COOKIE, path="/", httponly=True,
                               secure=secure, samesite="strict")
        response.delete_cookie(_hs.CSRF_COOKIE, path="/", httponly=False,
                               secure=secure, samesite="strict")
        from localm.debuglog import logger as _dbg
        _dbg.info("owner API key cleared via /api/auth/key/clear; session invalidated")
        return {"cleared": True}

    @app.get("/api/session", include_in_schema=False)
    async def session_state(request: Request):
        """Report whether the caller is authenticated, so the GUI can show or
        hide its key gate without ever reading the key. *authed* reflects the
        presented cookie/header; *required* is True when a key is configured."""
        from localm.auth import (any_key_configured, require_auth_enabled,
                                  verify)
        configured = any_key_configured()
        token, _ = _request_token(request)
        held = verify(token) if (configured and token) else None
        return {"authed": held is not None,
                "scopes": sorted(held) if held else [],
                "required": configured or require_auth_enabled()}
