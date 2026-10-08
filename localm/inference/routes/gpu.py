# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-instance GPU/VRAM coordination routes (see ``localm.gpu_registry``).

Three endpoints a sibling localm instance on this machine uses to talk to this one:

- ``GET /v1/instances/status``: this instance's live model / VRAM status.
- ``POST /v1/instances/cooperate-unload``: a sibling, itself out of local eviction
  candidates, asks THIS instance to release its own VRAM. A SEPARATE auth code path
  from ``require_scope``/``MODELS_WRITE``: the request is honoured only after the
  requester confirms, by calling this instance back on ``/v1/instances/vouch``,
  that it really sent it. An API key alone does not grant it.
- ``POST /v1/instances/vouch``: confirms (once) that THIS instance sent a given
  unload request to the asking instance.

All three exist only on an instance that coordinates (not ``--isolated``, not a bare
test app). Status and unload also require a loopback bind; vouch does not, because it
confirms only a request id this instance generated and sent. All three refuse any
request that carries an ``Origin`` header: a sibling instance is not a browser, so a
browser page cannot read the status or trigger the others."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request

import localm.inference.http_server as _hs

_REFUSED = "Cooperation request not verified."


def _from_a_sibling(request: Request) -> bool:
    """Whether the request can be from another localm instance rather than a
    browser page: it carries no ``Origin`` header."""
    return "origin" not in request.headers


def _coordinating(request: Request) -> bool:
    """Whether this instance coordinates, serves only loopback, and the request
    is not from a browser page."""
    return (bool(getattr(_hs, "_gpu_coord", None)) and _from_a_sibling(request)
            and _hs._is_loopback_host(
                getattr(request.app.state, "bind_host", "127.0.0.1")))


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def register(app: FastAPI, ctx) -> None:
    @app.get("/v1/instances/status", include_in_schema=False)
    async def instance_status(request: Request):
        """This instance's live coordination status (see ``_hs._gpu_status``)."""
        if not _coordinating(request):
            raise HTTPException(404, "Not Found")
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, _hs._gpu_status)
        if status is None:
            raise HTTPException(404, "Not Found")
        return status

    @app.post("/v1/instances/vouch", include_in_schema=False)
    async def vouch(request: Request):
        """Confirm that THIS instance sent unload request ``request_id`` to
        instance ``asker_instance_id``. One confirmation per request."""
        from localm import gpu_registry
        body = await _json_body(request)
        if (not getattr(_hs, "_gpu_coord", None) or not _from_a_sibling(request)
                or not gpu_registry.vouch_for(
                    body.get("request_id"), body.get("asker_instance_id"))):
            raise HTTPException(403, _REFUSED)
        return {"vouched": True}

    @app.post("/v1/instances/cooperate-unload", include_in_schema=False)
    async def cooperate_unload(request: Request):
        """Unload THIS instance's currently-loaded model(s) - the one action a
        sibling instance's cooperation request can trigger. Runs only after the
        named requester confirms the request (``gpu_registry.verify_requester``);
        reuses ``unload_all_models()`` (the same VRAM-release-wait behavior as
        ``POST /v1/models/unload``) rather than duplicating it."""
        from localm import gpu_registry
        body = await _json_body(request)
        coord = getattr(_hs, "_gpu_coord", None)
        verified = False
        if coord and _coordinating(request):
            loop = asyncio.get_running_loop()
            verified = await loop.run_in_executor(
                None, gpu_registry.verify_requester, body.get("requester"),
                body.get("request_id"), coord["instance_id"])
        if not verified:
            # An identical 403 whether coordination is not enabled on this
            # instance or the requester did not confirm, so the two cases are
            # indistinguishable to a caller.
            raise HTTPException(403, _REFUSED)
        result = await _hs.unload_all_models()
        # Minimal response: a sibling only needs to know whether VRAM was
        # released, not this instance's model names/VRAM numbers.
        return {"status": result.get("status", "unloaded")}
