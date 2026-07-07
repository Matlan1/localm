# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-instance GPU/VRAM coordination route (see ``localm.gpu_registry``).

One endpoint: a sibling localm instance on this machine, itself out of local
eviction candidates, asks THIS instance to release its own VRAM. Deliberately
a SEPARATE auth code path from ``require_scope``/``MODELS_WRITE``: it accepts
ONLY this instance's own ``coordination_token`` (minted fresh at startup,
stored only in this instance's own 0600 gpu-registry entry) - never the real
API key, shell token, or instance attach token, so holding a real API key
alone does not grant this without ALSO knowing the token."""

from __future__ import annotations

import hmac

from fastapi import FastAPI, HTTPException, Request

import localm.inference.http_server as _hs

# A dedicated header (not Authorization - that is the API-key/bearer path,
# and this must stay a visibly separate mechanism). A JSON body field is also
# accepted as a fallback for a caller that prefers a plain POST body.
COORDINATION_TOKEN_HEADER = "X-LocalM-Coordination-Token"


async def _presented_token(request: Request) -> "str | None":
    header = request.headers.get(COORDINATION_TOKEN_HEADER)
    if header:
        return header
    try:
        body = await request.json()
    except Exception:
        return None
    if isinstance(body, dict):
        val = body.get("coordination_token")
        if isinstance(val, str) and val:
            return val
    return None


def register(app: FastAPI, ctx) -> None:
    @app.post("/v1/instances/cooperate-unload", include_in_schema=False)
    async def cooperate_unload(request: Request):
        """Unload THIS instance's currently-loaded model(s) - the one action a
        sibling instance's cooperation request can trigger. Authenticated
        ONLY by this instance's own coordination_token; reuses
        ``unload_all_models()`` (the same VRAM-release-wait behavior as
        ``POST /v1/models/unload``) rather than duplicating it."""
        presented = await _presented_token(request)
        coord = getattr(_hs, "_gpu_coord", None)
        expected = coord.get("token") if isinstance(coord, dict) else None
        if not expected or not presented or not hmac.compare_digest(str(presented), str(expected)):
            # Deliberately identical 403 whether coordination is simply not
            # enabled on this instance (no _gpu_coord) or a token was
            # presented and did not match - never confirm/deny which case it
            # is to an unauthenticated caller.
            raise HTTPException(
                403, "Invalid or missing coordination token.")
        result = await _hs.unload_all_models()
        # Minimal response: a sibling only needs to know whether VRAM was
        # released, not this instance's model names/VRAM numbers.
        return {"status": result.get("status", "unloaded")}
