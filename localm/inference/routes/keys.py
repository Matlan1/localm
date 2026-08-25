# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoped API key routes (auth.json): list, create, revoke."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response

import localm.inference.http_server as _hs
from localm import scopes
from localm.bindhost import is_loopback_host as _is_loopback


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    caller_scopes = _hs.caller_scopes

    @app.get("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def list_keys_ep(caller: Optional[set] = Depends(caller_scopes)):
        from localm.auth import list_keys
        from localm.config import load_config
        # is_owner lets the GUI hide owner-only (privileged) scope choices from a
        # mere keys:admin device; presets ride along so a keys:admin key WITHOUT
        # config:read can still see the quick-select bundles.
        is_owner = caller is not None and scopes.ADMIN in caller
        presets = load_config().get("key_presets", [])
        return {"keys": list_keys(), "is_owner": is_owner, "presets": presets}

    @app.post("/v1/keys", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def create_key_ep(request: Request, response: Response, body: dict,
                            caller: Optional[set] = Depends(caller_scopes)):
        from localm import auth
        from localm.auth import create_key
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
            except (TypeError, ValueError):
                raise HTTPException(400, "'expires_in' must be seconds (a number)")
        elif expires is not None:
            try:
                expires = float(expires)
            except (TypeError, ValueError):
                raise HTTPException(400, "'expires' must be an epoch number or null")
        # Only an owner/ADMIN caller may grant privileged scopes; a key holding
        # merely keys:admin must not be able to mint itself owner-equivalent
        # access. In open mode (caller is None) privileged grants are refused.
        is_owner = caller is not None and scopes.ADMIN in caller
        # Same owner-only gate, extended to host-filesystem reach: a merely
        # keys:admin-scoped caller minting fs_access=host would hand the new
        # key access to the whole server disk, which is not a scope but is an
        # equivalent-severity grant. Refuse rather than silently downgrading to
        # "none" (AGENTS rule 5: a security step must never report success on a
        # request it did not actually honour).
        fs_access = auth.norm_fs_access(body.get("fs_access"))
        if fs_access != "none" and not is_owner:
            raise HTTPException(
                403,
                f"Refusing to grant fs_access={fs_access!r}: only the owner key "
                "may grant a new key host-filesystem reach."
            )
        # Same owner-only gate, extended to RAG-indexing reach: a key-scoped
        # rag_roots list REPLACES the whitelist rather than narrowing it
        # (rag/store.py's confine_index_path), so it can point a new key at any
        # folder on the host disk - including ones outside home/cwd/the
        # configured rag_allowed_roots a merely keys:admin caller's own keys are
        # bound to. That is host-filesystem reach, not a scope, so it gets the
        # same refusal as fs_access=host rather than silently downgrading to
        # unrestricted (AGENTS rule 5: a security step must never report success
        # on a request it did not actually honour).
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
        try:
            created = create_key(body.get("name", ""), scope_list,
                                 allow_privileged=is_owner, expires=expires,
                                 fs_access=fs_access, rag_roots=rag_roots or None)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Catch at grant: warn (do not block) when a granted scope unlocks a
        # plugin the host cannot serve yet (not installed / missing pip extras).
        mgr = getattr(app.state, "plugin_manager", None)
        if mgr is not None:
            try:
                warns = mgr.scope_deps_warnings(scope_list)
            except Exception:
                warns = []
            if warns:
                created = {**created, "warnings": warns}
        # Lockout guard: in open mode the loopback GUI is trusted via the
        # per-process shell token, which the server STOPS honouring the instant auth
        # turns on (a key now exists). Without this, minting your first key from the
        # local GUI would orphan the browser session and lock the owner out. So on the
        # FIRST key minted from a loopback bind, seed a persistent owner key and hand
        # THIS browser an owner session cookie (future `localm gui` loads re-establish
        # it; web.py). Loopback + open-mode only: a network bind already requires a key
        # up front, so this never fires there and grants no authority the local user
        # did not already hold via the shell token.
        if was_open and _is_loopback(getattr(app.state, "bind_host", "127.0.0.1")):
            owner = auth.get_api_key() or auth.regenerate_key()
            # Hand this browser an OPAQUE owner session (id in the cookie, never the
            # key), so it keeps full access and survives a later key roll - same
            # decoupled-session model as the login/auto-seed paths.
            from localm import sessions
            # owner_key_minted: see sessions.create. `owner` IS the owner key here
            # by construction (read back, or just seeded), but the proof is taken
            # rather than assumed - _is_owner_key re-compares against the live
            # value, so a seed that somehow did not take reports False instead of
            # stamping a privilege nothing established.
            sid = sessions.create(scopes={scopes.ADMIN},
                                  key_hash=auth._hash_key(owner), fs_access="host",
                                  owner_key_minted=auth._is_owner_key(owner))
            secure = request.url.scheme == "https"
            response.set_cookie(_hs.SESSION_COOKIE, sid, httponly=True,
                                secure=secure, samesite="strict", path="/",
                                max_age=_hs.SESSION_MAX_AGE)
            # CSRF is derived from the session (client fetches it from
            # GET /api/session); no separate CSRF cookie to set.
        # The plaintext key is returned exactly once - it is never recoverable.
        return created

    @app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def revoke_key_ep(key_id: str):
        from localm.auth import revoke_key
        if not revoke_key(key_id):
            raise HTTPException(404, f"No such key: {key_id}")
        return {"status": "revoked", "id": key_id}
