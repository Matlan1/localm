# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-instance model routing: discover a live peer that already has a
model loaded, and let the caller accept or clear a route to it. See
``localm.peer_routing`` for the state and forwarding this wraps, and
``routes/chat.py`` for where an accepted route is actually used.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded

# Budget for run_in_threadpool_bounded() below. A peer lookup reads the
# machine-wide registry directory and then runs one /whoami probe per
# live-pid entry at gpu_registry.list_gpu_peers' own 0.7s timeout, so this
# sits well above the worst case for a realistic number of local instances.
_PEER_LOOKUP_TIMEOUT_S = 20.0


class PeerRouteAccept(BaseModel):
    """Body of ``POST /v1/models/{model_id}/peer-route`` - a JSON body, not
    query parameters, because ``api_key`` is a real credential and query
    strings end up in access logs."""
    instance_id: str
    api_key: str


def _resolve_requested_name(model_id: str) -> str:
    """Same resolution ``get_engine`` applies at the top of its own body: an
    empty or ``"localm"`` name resolves via ``_resolve_unnamed_model_name``."""
    name = (model_id or "").strip()
    if not name or name == "localm":
        name = _hs._resolve_unnamed_model_name() or ""
    return name


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/models/{model_id}/peer-offer",
             dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def peer_offer(model_id: str):
        """Whether a live sibling instance already has this model loaded, and
        if so, enough about it (never its coordination_token) for a client to
        decide whether to offer routing to the user."""
        from localm import peer_routing
        from localm.config import load_registry
        name = _resolve_requested_name(model_id)
        if not name:
            return {"available": False, "peer": None}
        self_id = (_hs._gpu_coord or {}).get("instance_id") if _hs._gpu_coord else None

        def _lookup():
            registry = load_registry()
            canonical, aliases = peer_routing.registry_name_and_aliases(registry, name)
            return peer_routing.find_offer(canonical, aliases, exclude_self_id=self_id)

        try:
            peer = await run_in_threadpool_bounded(
                _lookup, timeout=_PEER_LOOKUP_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Looking for a peer with '{name}' loaded timed out: {e}")
        if peer is None:
            return {"available": False, "peer": None}
        return {"available": True, "peer": {
            "instance_id": peer.get("instance_id"),
            "host": peer.get("host"),
            "port": peer.get("port"),
            "scheme": peer.get("scheme") or "http",
            "model": peer.get("model"),
        }}

    @app.post("/v1/models/{model_id}/peer-route",
              dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def accept_peer_route(model_id: str, body: PeerRouteAccept):
        """Accept an offer: re-verify the named peer is still live and still
        advertising a matching model, then store the route. Never trusts a
        possibly-stale client-supplied offer without re-checking."""
        from localm import peer_routing
        from localm.config import load_registry
        name = _resolve_requested_name(model_id)
        if not name:
            raise HTTPException(400, "No model specified or configured to route")
        instance_id = body.instance_id
        api_key = body.api_key.strip()
        if not instance_id or not api_key:
            raise HTTPException(400, "instance_id and api_key are required")
        self_id = (_hs._gpu_coord or {}).get("instance_id") if _hs._gpu_coord else None

        # Verification only. run_in_threadpool_bounded ABANDONS the awaiting
        # caller at its deadline while the worker thread runs to completion, so
        # a set_route() in here would install the route after the client had
        # already been told 504.
        # See test_a_timed_out_accept_never_installs_a_route.
        def _verify():
            registry = load_registry()
            canonical, aliases = peer_routing.registry_name_and_aliases(registry, name)
            peer = peer_routing.find_offer(canonical, aliases, exclude_self_id=self_id)
            if peer is None or peer.get("instance_id") != instance_id:
                return None
            return peer_routing.PeerRoute(
                model=name, instance_id=peer.get("instance_id"),
                host=peer.get("host"), port=int(peer.get("port")),
                scheme=peer.get("scheme") or "http", api_key=api_key)

        try:
            route = await run_in_threadpool_bounded(
                _verify, timeout=_PEER_LOOKUP_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Verifying peer '{instance_id}' timed out: {e}")
        if route is None:
            raise HTTPException(
                409, f"Peer instance '{instance_id}' is no longer live, no "
                f"longer has '{name}' loaded, or is not reachable at a "
                "loopback address over http or https; re-check available peers.")
        peer_routing.set_route(route)
        return {"status": "routed", "model": name, "peer": route.safe_dict()}

    @app.delete("/v1/models/{model_id}/peer-route",
               dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def clear_peer_route(model_id: str):
        """Clear an active route for *model_id*, if any. Idempotent: clearing
        an unrouted model is a no-op success, not an error."""
        from localm import peer_routing
        name = _resolve_requested_name(model_id)
        removed = peer_routing.clear_route(name) if name else None
        return {"status": "unrouted", "model": name, "was_routed": removed is not None}
