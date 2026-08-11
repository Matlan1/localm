# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session auth routes: HttpOnly cookie + CSRF for the browser GUI.

Extracted verbatim from create_app(); behavior unchanged. The session cookie
names, max age, and the token/CSRF helpers live on the http_server module and are
referenced via ``_hs.`` so external importers still find them there.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request, Response

import localm.inference.http_server as _hs
from localm import scopes


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    _bearer_token = _hs._bearer_token
    _request_token = _hs._request_token
    _csrf_ok = _hs._csrf_ok

    @app.post("/api/session", include_in_schema=False)
    async def session_login(request: Request, response: Response):
        """Exchange the API key for an HttpOnly session cookie so the browser GUI
        never has to hold the key in JS-readable localStorage. The key is read from
        the JSON body ``{"key": ...}`` or an Authorization: Bearer header, verified,
        and on success set as the HttpOnly ``localm_session`` cookie (an opaque
        session id). The response body returns the ``csrf`` token (an HMAC of the
        session, NOT a cookie) for the client to send as ``X-CSRF-Token`` on writes.
        401 on a bad key; 400 in open mode (nothing to log into)."""
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
        # Mint an OPAQUE server-side session; the cookie carries the session id,
        # never the raw key (so rolling the key no longer logs the browser out and
        # the durable secret never sits in a cookie jar). The scope/identity/
        # fs-access snapshot is taken now so the session stays valid across a roll.
        from localm import scopes as S, sessions
        from localm.auth import _hash_key, _is_owner_key, fs_access_for
        fs = "host" if S.ADMIN in held else fs_access_for(presented, "none")
        # Record WHETHER THE OWNER KEY minted this session, while that is still
        # provable. key_hash freezes the key's VALUE, and an owner-key roll
        # deliberately leaves sessions alive, so afterwards the frozen hash matches
        # neither the new owner key nor any keystore entry - identical to a REVOKED
        # scoped key, which is why the owner's own scheduled jobs silently lost
        # shell (REG-509). _is_owner_key is a constant-time plaintext compare
        # against the live owner key: a POSITIVE proof that reads no keystore, so a
        # corrupt auth.json cannot flip it, and holding ADMIN cannot earn it.
        sid = sessions.create(scopes=held, key_hash=_hash_key(presented),
                              fs_access=fs,
                              owner_key_minted=_is_owner_key(presented))
        secure = request.url.scheme == "https"
        response.set_cookie(_hs.SESSION_COOKIE, sid, httponly=True,
                            secure=secure, samesite="strict", path="/",
                            max_age=_hs.SESSION_MAX_AGE)
        # CSRF token is DERIVED from the session (not a separate cookie that could
        # be cleared independently and desync); hand it back for the client to echo
        # as X-CSRF-Token on state-changing requests.
        return {"authed": True, "scopes": sorted(held),
                "csrf": _hs.csrf_token_for(request, sid)}

    @app.post("/api/session/logout", include_in_schema=False)
    async def session_logout(request: Request, response: Response):
        """Sign the browser out: revoke the session and clear its cookie.

        A POST from the cookie-authenticated browser GUI is subject to the same CSRF
        check as every other state-changing endpoint (the X-CSRF-Token header must
        match the token derived from the session). A bearer-header caller (CLI / SDK)
        is CSRF-exempt because it cannot be driven cross-site."""
        token, source = _request_token(request)
        if source == "cookie" and not _csrf_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Send the session's csrf token "
                "(from GET /api/session) in the X-CSRF-Token header.")
        # Real, server-side logout: drop the session row so the cookie value can
        # never be replayed (deleting the cookie alone left a valid server session).
        warnings: list[str] = []
        if source == "cookie" and token:
            from localm import sessions
            if sessions.revoke(token) is None:
                # The store write failed, so the session id is STILL VALID on the
                # server. Clearing the cookie below stops this browser using it,
                # but that is exactly the "deleting the cookie alone" state the
                # server-side revocation exists to improve on, so reporting a
                # clean sign-out here would be the rule-5 lie. sessions.revoke
                # has already warned to the local log with the reason.
                warnings.append(sessions.REVOKE_FAILURE_LABEL)
        secure = request.url.scheme == "https"
        response.delete_cookie(_hs.SESSION_COOKIE, path="/", httponly=True, secure=secure, samesite="strict")
        # "authed" stays False and is honest either way: this browser's cookie is
        # gone. The warning says the SERVER-side session was not dropped, which is
        # a different fact and the one a caller cannot otherwise discover.
        return {"authed": False, "warnings": warnings}

    @app.post("/api/auth/key/clear",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))],
              include_in_schema=False)
    async def clear_owner_key(request: Request, response: Response):
        """Delete the server-side owner key (auth.key) and immediately
        invalidate the caller's session cookie.

        CSRF-protected: a cookie-authenticated caller must send the session's
        derived token in X-CSRF-Token. After this call any_key_configured() returns
        False (assuming no LOCALM_API_KEY env var and an empty keystore), and all
        subsequent web UI requests that rely on the session cookie will fail auth and
        be redirected to the key gate by the client (Task 3: CSRF-post-clear).
        """
        _, source = _request_token(request)
        if source == "cookie" and not _csrf_ok(request):
            raise HTTPException(
                403,
                "Missing or invalid CSRF token. Send the session's csrf token "
                "(from GET /api/session) in the X-CSRF-Token header.")
        from localm.auth import clear_api_key
        failed = clear_api_key()
        # The key that minted every current session is gone; those sessions carry
        # their own ADMIN scope snapshot, so they MUST be revoked or a leftover
        # cookie would keep full access after the key was cleared (would defeat the
        # clear). Sign out everywhere - and READ THE RESULT: revoke_all returns None
        # when the store could not be written, which is a failed sign-out, not an
        # empty store. Discarding it is what made this route claim a completed clear
        # while every session stayed live.
        from localm import sessions
        revoked = sessions.revoke_all()
        secure = request.url.scheme == "https"
        # Invalidate the session cookie immediately so the browser is forced
        # back to the key gate on the next navigation (the old cookie value
        # no longer matches any key). Also clear the CSRF token.
        response.delete_cookie(_hs.SESSION_COOKIE, path="/", httponly=True,
                               secure=secure, samesite="strict")
        from localm.debuglog import logger as _dbg
        # Rule 5: a security step that failed must never report success. BOTH
        # halves of this route are such a step, and each can fail on its own:
        # a surviving auth.key/keystore still grants access, and a surviving
        # ADMIN session cookie still grants access. So "cleared" is true only
        # when BOTH completed. Previously the session half was not read at all,
        # and the comment here asserted "the sessions ARE revoked either way"
        # as established fact - it was an unmeasured premise, not a proof, and
        # it is deleted rather than worked around because a false invariant
        # comment is worse than no comment: it is what the docs were written
        # from. Reported honestly rather than raised - a 500 would imply
        # NOTHING had happened, when typically one half did.
        #
        # ONLY path-free labels go on the wire. clear_api_key also returns
        # "path" (an absolute filesystem path, which carries the account name -
        # rule 2) and "error" (raw OS exception text - py/stack-trace-exposure);
        # sessions.REVOKE_FAILURE_LABEL is path-free for the same reason. Those
        # other fields are for the LOCAL CLI and the local log only.
        #
        # Nothing is logged HERE for the credential half, on purpose - this is
        # not a silenced warning. clear_api_key already warns to THIS SAME
        # logger once per thing it could not remove (localm/auth.py, the OSError
        # handlers), and those lines carry the path and the OS error, so they
        # are strictly more informative than anything this route could add.
        # sessions.revoke_all warns for its own half on the same logger.
        warnings = [f["what"] for f in failed]
        if revoked is None:
            warnings.append(sessions.REVOKE_FAILURE_LABEL)
        if warnings:
            return {"cleared": False, "warnings": warnings}
        _dbg.info("owner API key cleared via /api/auth/key/clear; session invalidated")
        return {"cleared": True, "warnings": []}

    @app.get("/api/session", include_in_schema=False)
    async def session_state(request: Request):
        """Report whether the caller is authenticated, so the GUI can show or hide
        its key gate without ever reading the key. *authed* reflects the presented
        cookie/header; *required* is True when a key is configured. For a cookie
        session it also returns the ``csrf`` token derived from that session, so the
        client always has a token in lockstep with its session (it can never desync)
        and can refresh it after a server restart rotated the secret."""
        from localm.auth import any_key_configured, require_auth_enabled
        configured = any_key_configured()
        token, source = _request_token(request)
        prin = (_hs._principal_from_token(token, source)
                if (configured and token) else None)
        held = prin[0] if prin else None
        csrf = (_hs.csrf_token_for(request, token)
                if (held is not None and source == "cookie") else "")
        return {"authed": held is not None,
                "scopes": sorted(held) if held else [],
                "required": configured or require_auth_enabled(),
                "csrf": csrf}
