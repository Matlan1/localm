# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scoped API key routes (auth.json): list, create, revoke.

Extracted verbatim from create_app(); behavior unchanged.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes


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
    async def create_key_ep(body: dict, caller: Optional[set] = Depends(caller_scopes)):
        from localm.auth import create_key
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
        try:
            created = create_key(body.get("name", ""), scope_list,
                                 allow_privileged=is_owner, expires=expires)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        # The plaintext key is returned exactly once - it is never recoverable.
        return created

    @app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_scope(scopes.KEYS_ADMIN))])
    async def revoke_key_ep(key_id: str):
        from localm.auth import revoke_key
        if not revoke_key(key_id):
            raise HTTPException(404, f"No such key: {key_id}")
        return {"status": "revoked", "id": key_id}
