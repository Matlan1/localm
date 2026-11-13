# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoped API key routes (auth.json): list, create, revoke."""

from __future__ import annotations

import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Response

import localm.inference.http_server as _hs
from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback

#: Held across the last-key check and the revoke of one DELETE /v1/keys/{id}.
_REVOKE_LOCK = threading.Lock()


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def list_keys_ep(request: Request):
        from localm.auth import list_keys, owner_key_downgrade_warning
        from localm.config import load_config
        # is_owner lets the GUI hide owner-only (privileged) scope choices from a
        # mere keys:admin device; presets ride along so a keys:admin key WITHOUT
        # config:read can still see the quick-select bundles.
        is_owner = _hs.caller_is_admin_key(request)
        presets = load_config().get("key_presets", [])
        # Not owner-gated. In open mode caller is None and an is_owner gate
        # would hide this in exactly the case it matters. In protected mode via
        # a scoped keystore, a non-owner keys:admin caller can also reach this
        # field - the same exposure list_keys() already gives that caller class
        # on this same route, unconditionally.
        warning = owner_key_downgrade_warning()
        if warning is not None:
            warning = _hs.path_formatter(request)(warning)
        return {"keys": list_keys(), "is_owner": is_owner, "presets": presets,
                "owner_key_warning": warning}

    @app.post("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def create_key_ep(request: Request, response: Response, body: dict):
        from localm import auth
        from localm.auth import create_key
        from localm.config import ConfigUnreadable
        # Capture BEFORE minting: True only when this is the very first key on an
        # otherwise-keyless install (the open -> protected transition).
        was_open = not auth.any_key_configured()
        scope_list = body.get("scopes", [])
        if not isinstance(scope_list, list):
            raise HTTPException(400, "'scopes' must be a list of scope strings")
        # Prefer a server-computed deadline from a relative TTL (expires_in seconds)
        # so a key's lifetime never depends on the client's clock; fall back to an
        # absolute epoch (expires) for raw API clients that want a specific deadline.
        expires = body.get("expires")
        expires_in = body.get("expires_in")
        if expires_in is not None:
            try:
                expires = time.time() + float(expires_in)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "'expires_in' must be seconds (a number)") from exc
        elif expires is not None:
            try:
                expires = float(expires)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "'expires' must be an epoch number or null") from exc
        # Only an owner/ADMIN caller may grant privileged scopes. In open mode
        # privileged grants are refused.
        is_owner = _hs.caller_is_admin_key(request)
        # The same owner-only gate covers host-filesystem reach, refused rather
        # than silently downgraded to "none".
        fs_access = auth.norm_fs_access(body.get("fs_access"))
        if fs_access != "none" and not is_owner:
            raise HTTPException(
                403,
                f"Refusing to grant fs_access={fs_access!r}: only the owner key "
                "may grant a new key host-filesystem reach."
            )
        # The same owner-only gate covers RAG-indexing reach: a key-scoped
        # rag_roots list REPLACES the whitelist rather than narrowing it
        # (rag/store.py's confine_index_path). Refused rather than silently
        # downgraded to unrestricted.
        rag_roots_in = body.get("rag_roots")
        if rag_roots_in is not None and not isinstance(rag_roots_in, list):
            raise HTTPException(400, "'rag_roots' must be a list of folder paths")
        rag_roots = auth.norm_rag_roots(rag_roots_in)
        if rag_roots and not is_owner:
            raise HTTPException(
                403,
                "Refusing to set rag_roots: only the owner key may confine a "
                "new key's RAG-indexing reach to specific folders."
            )
        if not is_owner:
            beyond = [s for s in scopes.normalize(scope_list)
                      if not _hs.caller_grants(request, s)]
            if beyond:
                raise HTTPException(
                    403,
                    f"Refusing to grant scope(s) {', '.join(beyond)}: a key may "
                    "only mint keys with scopes it holds itself.")
        try:
            created = create_key(body.get("name", ""), scope_list,
                                 allow_privileged=is_owner, expires=expires,
                                 fs_access=fs_access, rag_roots=rag_roots or None)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        # Catch at grant: warn (do not block) when a granted scope unlocks a
        # plugin the host cannot serve yet (not installed / missing pip extras).
        mgr = getattr(app.state, "plugin_manager", None)
        if mgr is not None:
            try:
                warns = mgr.scope_deps_warnings(scope_list)
                warns += scopes.unknown_scope_warnings(
                    scope_list,
                    {(sp.scope or n) for n, sp in mgr.discover().items()})
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.warning("could not check plugin readiness for granted scopes: %s",
                             e, exc_info=True)
                warns = ["Could not check whether the plugins behind these scopes "
                         "are installed and ready; the key was created."]
            if warns:
                created = {**created, "warnings": warns}
        # Lockout guard: in open mode the loopback GUI is trusted via the
        # per-process shell token, which the server stops honouring the instant
        # auth turns on. On the FIRST key minted from a loopback bind, seed a
        # persistent owner key and hand THIS browser an owner session cookie
        # (future `localm gui` loads re-establish it; web.py). Loopback plus
        # open-mode only.
        owner = None
        if was_open and _is_loopback(getattr(app.state, "bind_host", "127.0.0.1")):
            try:
                owner, owner_created = auth.ensure_owner_key()
            except ConfigUnreadable as e:
                created = {**created, "owner_key_created": False, "warnings": [
                    *created.get("warnings", []),
                    f"No owner session was set up for this browser: {e}."]}
            else:
                created = {**created, "owner_key_created": owner_created}
        if owner is not None:
            # An OPAQUE owner session (id in the cookie, never the key), so this
            # browser keeps full access across a later key roll.
            from localm import sessions
            # owner_key_minted: see sessions.create. _is_owner_key re-compares
            # against the live value, so a seed that did not take reports False.
            from fastapi.concurrency import run_in_threadpool
            seeded: str = owner
            sid = await run_in_threadpool(
                lambda: sessions.create(scopes={scopes.ADMIN},
                                        key_hash=auth._hash_key(seeded), fs_access="host",
                                        owner_key_minted=auth._is_owner_key(seeded)))
            secure = request.url.scheme == "https"
            response.set_cookie(_hs.session_cookie_name(request), sid, httponly=True,
                                secure=secure, samesite="strict", path="/",
                                max_age=_hs.SESSION_MAX_AGE)
            # CSRF is derived from the session (client fetches it from
            # GET /api/session); no separate CSRF cookie to set.
        # The plaintext key is returned exactly once - it is never recoverable.
        return created

    @app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def revoke_key_ep(key_id: str, request: Request):
        """Revoke a named key and end what runs under it: its browser sessions,
        the coder sessions it created and its open streams. Answers 409 when the
        server is reachable from the network, must hold a key, and this is its
        last one."""
        from fastapi.concurrency import run_in_threadpool

        from localm import auth
        from localm.config import ConfigUnreadable
        needs_key = _hs.network_bind_needs_key(request.app)

        def _revoke():
            with _REVOKE_LOCK:
                if needs_key and auth.removal_leaves_no_key(revoke_id=key_id):
                    return None
                return auth.revoke_key(key_id)

        try:
            revoked = await run_in_threadpool(_revoke)
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        if revoked is None:
            raise HTTPException(409, _hs.LAST_KEY_NETWORK_DETAIL)
        if not revoked:
            raise HTTPException(404, f"No such key: {key_id}")
        return {"status": "revoked", "id": key_id}
