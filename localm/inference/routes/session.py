# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session auth routes: HttpOnly cookie + CSRF for the browser GUI.

The session cookie names, max age, and the token/CSRF helpers live on the
http_server module and are referenced via ``_hs.`` so external importers still
find them there.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request, Response

import localm.inference.http_server as _hs
from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback


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
        # never the raw key. The scope/identity/fs-access snapshot is taken now, so
        # the session stays valid across a key roll.
        from localm import scopes as S, sessions
        from localm.auth import _hash_key, _is_owner_key, fs_access_for, rag_roots_for
        fs = "host" if S.ADMIN in held else fs_access_for(presented, "none")
        # Same "owner is never confined by a per-credential field" shape as fs
        # above: an ADMIN session snapshots [] (unrestricted), never a stored
        # per-key list, exactly like effective_rag_roots resolves it live.
        rag_roots = [] if S.ADMIN in held else rag_roots_for(presented, [])
        # Record WHETHER THE OWNER KEY minted this session, while that is still
        # provable. key_hash freezes the key's VALUE, and an owner-key roll leaves
        # sessions alive, so afterwards the frozen hash matches neither the new
        # owner key nor any keystore entry - identical to a REVOKED scoped key.
        # _is_owner_key is a constant-time plaintext compare against the live owner
        # key: a POSITIVE proof that reads no keystore, so a corrupt auth.json
        # cannot flip it, and holding ADMIN cannot earn it.
        sid = sessions.create(scopes=held, key_hash=_hash_key(presented),
                              fs_access=fs, rag_roots=rag_roots,
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
        # never be replayed.
        warnings: list[str] = []
        if source == "cookie" and token:
            from localm import sessions
            if sessions.revoke(token) is None:
                # The store write failed, so the session id is STILL VALID on the
                # server. Clearing the cookie below stops this browser using it,
                # but the session survives, so this does not report a clean
                # sign-out. sessions.revoke has already logged the failure and
                # its cause locally.
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
        be redirected to the key gate by the client.
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
        # cookie would keep full access after the key was cleared. Sign out
        # everywhere - and READ THE RESULT: revoke_all returns None when the store
        # could not be written, which is a failed sign-out, not an empty store.
        from localm import sessions
        revoked = sessions.revoke_all()
        secure = request.url.scheme == "https"
        # Invalidate the session cookie immediately so the browser is forced
        # back to the key gate on the next navigation (the old cookie value
        # no longer matches any key). Also clear the CSRF token.
        response.delete_cookie(_hs.SESSION_COOKIE, path="/", httponly=True,
                               secure=secure, samesite="strict")
        from localm.debuglog import logger as _dbg
        # A security step that failed must never report success. BOTH halves of
        # this route are such a step, and each can fail on its own: a surviving
        # auth.key/keystore still grants access, and a surviving ADMIN session
        # cookie still grants access. So "cleared" is true only when BOTH
        # completed. Reported honestly rather than raised - a 500 would imply
        # NOTHING had happened, when typically one half did.
        #
        # ONLY path-free labels go on the wire. clear_api_key also returns
        # "path" (an absolute filesystem path, which carries the account name)
        # and "error" (raw OS exception text); sessions.REVOKE_FAILURE_LABEL is
        # path-free for the same reason. Those other fields are for the LOCAL
        # CLI and the local log only.
        #
        # Nothing is logged HERE for the credential half: clear_api_key already
        # warns to THIS SAME logger once per thing it could not remove
        # (localm/auth.py, the OSError handlers), and those lines carry the path
        # and the OS error. sessions.revoke_all warns for its own half on the
        # same logger.
        warnings = [f["what"] for f in failed]
        if revoked is None:
            warnings.append(sessions.REVOKE_FAILURE_LABEL)
        if warnings:
            return {"cleared": False, "warnings": warnings}
        _dbg.info("owner API key cleared via /api/auth/key/clear; session invalidated")
        return {"cleared": True, "warnings": []}

    @app.post("/api/auth/key/rotate",
              dependencies=[Depends(require_scope(scopes.ADMIN))],
              include_in_schema=False)
    async def rotate_owner_key(request: Request, response: Response):
        """Roll or set the owner key (``auth.key``) - the GUI form of
        ``localm key generate`` and ``localm key set <key>``.

        The body is optional: no body (or ``{}``) mints a fresh random key,
        ``{"key": "<value>"}`` persists that exact one. An empty or whitespace
        ``key`` GENERATES rather than clearing, so this route can never drop the
        server to open mode by accident - ``/api/auth/key/clear`` is the only way
        out of protected mode.

        The active key is returned so the caller can show it once. That discloses
        nothing new to THIS caller: an ADMIN principal can already read the owner
        key from ``GET /api/pairing/qr``.

        OWNER-GATED AND REMOTE-CAPABLE:

        * Requires ``scopes.ADMIN``, NOT the ``config:write`` its sibling clear
          route takes, and NOT ``keys:admin``. Clearing REMOVES a credential;
          setting INSTALLS one the caller chose, so a merely config:write or
          keys:admin holder POSTing ``{"key": "<a value I know>"}`` would promote
          itself to owner in a single call.
        * NOT loopback-only. Reaching this route already requires the owner
          credential, so it grants no authority the caller does not already hold.
          The backstop against a stolen owner cookie rolling the key is
          ``localm key recover``, run locally on the server machine.

        CSRF needs no check here: ``_enforce_request`` already requires a valid
        ``X-CSRF-Token`` for any cookie-sourced unsafe method.

        Browser sessions are NOT revoked, matching ``localm key generate`` and
        ``localm key set``: sessions are decoupled from the key value so a roll
        does not sign the GUI out. The command that DOES revoke is ``localm key
        recover``, the local compromise path.
        """
        from localm import auth
        payload = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}
        requested = payload.get("key")
        if requested is not None and not isinstance(requested, str):
            raise HTTPException(
                400, "'key' must be a string, or omitted to generate one.")
        # Capture BEFORE writing: True only on an otherwise-keyless install, which
        # is the open -> protected transition the lockout guard below exists for.
        was_open = not auth.any_key_configured()
        chose_own = bool(requested and requested.strip())
        if chose_own:
            try:
                auth.set_api_key(requested)
            except ValueError as e:
                # set_api_key refuses a key that is too short or uses characters an
                # HTTP Authorization header cannot carry. That is caller input, so
                # it is a 400, not a 500. The previous key is untouched on this path.
                raise HTTPException(400, str(e)) from e
            key = requested.strip()
        else:
            key = auth.regenerate_key()

        # READ THE KEY BACK rather than assuming the write decided the outcome.
        # Two different things can make a "successful" rotation a no-op on the
        # credential the server actually accepts, and they need different words:
        #
        #  1. LOCALM_API_KEY outranks the file (auth.get_api_key), so under that
        #     env var the new key is genuinely on disk and the server still accepts
        #     the OLD environment one. The CLI says the same thing through
        #     _note_env_override.
        #  2. Anything else that leaves the read-back disagreeing with what was just
        #     written is an unexplained failure, and is reported as one rather than
        #     blamed on an env var that is not set.
        import os as _os
        warnings: list[str] = []
        env_override = bool((_os.environ.get(auth.ENV_VAR) or "").strip())
        active = auth.get_api_key() == key
        if not active and env_override:
            warnings.append(
                f"{auth.ENV_VAR} is set in the server's environment and overrides "
                "the stored key, so the server still accepts the environment's key "
                "and NOT this new one. Unset it and restart to use this key.")
        elif not active:
            warnings.append(
                "the new key was written but could not be read back as the active "
                "key, so the previous credential may still be the live one")

        # Lockout guard, mirroring the first-key path in routes/keys.py. In open
        # mode the loopback GUI is trusted via the per-process shell token, which
        # the server STOPS honouring the instant a key exists - so setting the very
        # first key from the local GUI would orphan the browser that just did it.
        # On the open -> protected transition from a loopback bind, hand THIS
        # browser an opaque owner session (the id in the cookie, never the key).
        # Loopback + open-mode only: a network bind already required a key up
        # front, so this never fires there.
        if was_open and _is_loopback(getattr(app.state, "bind_host", "127.0.0.1")):
            from localm import sessions
            # owner_key_minted is PROVEN, not assumed: _is_owner_key re-compares
            # against the live value, so a write that somehow did not take reports
            # False instead of stamping a privilege nothing established.
            sid = sessions.create(scopes={scopes.ADMIN},
                                  key_hash=auth._hash_key(key), fs_access="host",
                                  owner_key_minted=auth._is_owner_key(key))
            secure = request.url.scheme == "https"
            response.set_cookie(_hs.SESSION_COOKIE, sid, httponly=True,
                                secure=secure, samesite="strict", path="/",
                                max_age=_hs.SESSION_MAX_AGE)

        from localm.debuglog import logger as _dbg
        # Never the key itself, and no filesystem path: this log is attached to bug
        # reports. Only that a rotation happened and whether it took effect.
        _dbg.info("owner API key %s via /api/auth/key/rotate (active=%s)",
                  "set" if chose_own else "rolled", active)
        # "rotated" is true only when the key is BOTH persisted and live, the same
        # both-halves-completed contract /api/auth/key/clear uses for "cleared" - a
        # caller reading only the status code must not conclude the credential
        # changed. The warnings are path-free and carry no OS exception text.
        return {"rotated": not warnings, "active": active, "key": key,
                "warnings": warnings}

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
