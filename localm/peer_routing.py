# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-instance model ROUTING: forward this instance's chat/completion
requests for one model name to a live sibling instance that already has that
model loaded, instead of loading a redundant local copy.

Builds on ``localm.gpu_registry`` (peer discovery, liveness + ``/whoami``
identity verification) but is the other half: gpu_registry only cooperates on
VRAM release, this module never touches that machinery and never reuses its
``coordination_token`` - a route is authenticated with the PEER's own real API
key, supplied by the user when they accept an offer, held in this process's
memory only (never persisted, never the same credential as
``coordination_token``).

Routing state (:data:`_ROUTES`) is process-local and in-memory: a restart
drops every route, and a route is established again explicitly by the user
accepting a fresh offer.

Forwarding is a raw reverse proxy for a fixed, caller-specified path (never a
path taken from the incoming request) - the request body and a minimal header
set go to the peer unchanged, the peer's response comes back unchanged. Chat-
pipeline hooks, audit/transcript logging, and per-request activity tracking
configured on THIS instance do not run for a forwarded request."""

from __future__ import annotations

import asyncio
import ipaddress
import threading
from dataclasses import dataclass
from typing import Optional

from localm.debuglog import logger

# Read/write timeout for a forwarded request: generous on the read side since
# a chat completion can stream for minutes, same-machine loopback so the
# connect side stays short.
_CONNECT_TIMEOUT = 5.0
_READ_TIMEOUT = 300.0


@dataclass
class PeerRoute:
    model: str
    instance_id: str
    host: str
    port: int
    scheme: str
    api_key: str

    def safe_dict(self) -> dict:
        """The subset of this route safe to hand back to a client - never
        ``api_key``."""
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "model": self.model,
        }


# model name -> PeerRoute. Process-local, in-memory, never persisted.
_ROUTES: dict = {}

# Guards every read and write of _ROUTES. Held only across dict operations,
# never across a network call or a registry read.
# See test_find_offer_does_not_hold_the_routes_lock and
# test_forward_does_not_hold_the_routes_lock_across_the_request.
_ROUTES_LOCK = threading.RLock()


def get_route(model_name: Optional[str]) -> Optional[PeerRoute]:
    if not model_name:
        return None
    with _ROUTES_LOCK:
        return _ROUTES.get(model_name)


def set_route(route: PeerRoute) -> None:
    with _ROUTES_LOCK:
        _ROUTES[route.model] = route


def clear_route(model_name: Optional[str]) -> Optional[PeerRoute]:
    """Remove and return the route for *model_name*, or None if there was
    none."""
    if not model_name:
        return None
    with _ROUTES_LOCK:
        return _ROUTES.pop(model_name, None)


def list_routes() -> dict:
    """Every active route, keyed by model name, as :meth:`PeerRoute.safe_dict`
    values - never includes an ``api_key``."""
    with _ROUTES_LOCK:
        return {name: route.safe_dict() for name, route in _ROUTES.items()}


# ------------------------------------------------------------------ #
#  Discovery / matching                                              #
# ------------------------------------------------------------------ #

def dial_host(host: Optional[str]) -> str:
    """The address this machine dials to reach a peer that registered *host*,
    via ``bindhost.self_connect_host`` (wildcards and ``localhost`` become a
    loopback literal, any other literal is returned as itself)."""
    from localm.bindhost import self_connect_host
    return self_connect_host(host)


# The only schemes a peer entry may name. gpu_registry.list_gpu_peers applies
# no whitelist of its own, and _peer_url interpolates the value straight into
# "{scheme}://{host}:{port}{path}", where any string containing "://" moves the
# authority off *host* entirely.
# See test_a_peer_whose_scheme_smuggles_an_authority_is_never_offered.
_ROUTABLE_SCHEMES = ("http", "https")


def is_routable_peer_host(host: Optional[str]) -> bool:
    """Whether *host* names an address this instance has identity-verified.

    True only when :func:`dial_host` yields a loopback IP literal. A name that
    does not parse as an IP address, and any value that is not a string, is
    False rather than raising.

    ``gpu_registry.list_gpu_peers`` runs its ``/whoami`` identity handshake
    with no ``bind_host``, so the address it verifies is exactly
    ``127.0.0.1``. This predicate accepts the whole loopback class, which is
    marginally wider: a peer bound only on ``::1`` has nothing answering on
    ``127.0.0.1`` and so fails that handshake and is never offered, so the
    extra width is unreachable without a forged registry entry. See
    test_a_peer_advertising_a_non_loopback_host_is_never_offered."""
    try:
        return ipaddress.ip_address(dial_host(host)).is_loopback
    except (ValueError, AttributeError, TypeError):
        return False


def is_routable_peer_endpoint(host: Optional[str], scheme: Optional[str]) -> bool:
    """Whether a peer registering *host* and *scheme* may be sent this
    instance's forwarded request and its bearer credential.

    Requires BOTH a verified-loopback *host* (:func:`is_routable_peer_host`)
    and a *scheme* in :data:`_ROUTABLE_SCHEMES`. Both fields come from the same
    untrusted registry entry, so pinning only the host leaves the credential's
    destination open through the other one."""
    return scheme in _ROUTABLE_SCHEMES and is_routable_peer_host(host)


def find_offer(canonical_name: str, aliases, *, exclude_self_id: Optional[str] = None) -> Optional[dict]:
    """A live, identity-verified peer (via ``gpu_registry.list_gpu_peers``)
    whose advertised ``model`` matches *canonical_name* or any of *aliases* -
    exact match first, then casefolded - or None.

    Matches by NAME only: it does not verify the underlying model files are
    identical, only that the peer's own chosen name for what it has loaded
    equals one this instance also uses for the same registry entry.

    A peer whose registered ``host``/``scheme`` fails
    :func:`is_routable_peer_endpoint` is skipped and logged at WARNING,
    however well it matches.

    Best-effort: any failure reading the registry is logged and yields None,
    never raised, matching every other public function in gpu_registry."""
    if not canonical_name:
        return None
    names = {canonical_name, *aliases}
    folded = {n.casefold() for n in names}
    try:
        from localm import gpu_registry
        peers = gpu_registry.list_gpu_peers(exclude_self_id=exclude_self_id)
    except Exception as e:
        logger.debug("peer_routing: peer lookup failed: %s", e)
        return None
    for peer in peers:
        model = peer.get("model")
        if not model:
            continue
        if model not in names and model.casefold() not in folded:
            continue
        if not is_routable_peer_endpoint(peer.get("host"), peer.get("scheme") or "http"):
            logger.warning(
                "peer_routing: peer %r advertises %r at unroutable endpoint "
                "scheme=%r host=%r; not offering it, because only a loopback "
                "address over http or https has had its occupant "
                "identity-verified",
                peer.get("instance_id"), model, peer.get("scheme"), peer.get("host"))
            continue
        return peer
    return None


def registry_name_and_aliases(registry: dict, model_name: str) -> tuple:
    """*model_name*'s canonical registry key and every other key sharing its
    ``path`` (its aliases) - the same grouping
    ``routes/models.py``'s ``model_detail`` uses to compute ``aliases``.

    ``model_name`` itself may already be an alias: this resolves via the
    entry's own ``path`` rather than assuming *model_name* is canonical, so
    the returned alias set is the same regardless of which alias was asked
    for. Returns (model_name, frozenset()) unchanged when *model_name* is not
    a dict entry in *registry* (e.g. the startup/default model, which is not
    always registered)."""
    entry = registry.get(model_name)
    if not isinstance(entry, dict):
        return model_name, frozenset()
    path = entry.get("path")
    aliases = frozenset(
        n for n, e in registry.items()
        if isinstance(e, dict) and e.get("path") == path and n != model_name
    )
    return model_name, aliases


# ------------------------------------------------------------------ #
#  Forwarding                                                        #
# ------------------------------------------------------------------ #

def _peer_url(route: PeerRoute, path: str) -> str:
    from localm.bindhost import self_connect_host, url_host
    host = url_host(self_connect_host(route.host))
    return f"{route.scheme}://{host}:{int(route.port)}{path}"


async def forward(route: PeerRoute, request, path: str):
    """Forward *request* to *route*'s peer at the fixed literal *path*
    (never a path taken from *request* itself) and stream the response back
    unchanged. Returns a ``fastapi.responses.StreamingResponse``.

    On a network failure reaching the peer (connection refused, timeout, DNS/
    TLS error), the route is CLEARED and ``HTTPException(502, ...)`` is
    raised naming the peer as unavailable - this failed request is not
    retried locally; the next request for this model name proceeds as an
    ordinary local load because the route is gone. Any HTTP response the peer
    DOES return (2xx or not) is passed through unchanged - that is a real
    answer from a live peer, not "peer unavailable".

    A *route* whose endpoint fails :func:`is_routable_peer_endpoint` is refused
    before the request body or the ``Authorization`` header is built: the route is
    CLEARED, a WARNING is logged, and ``HTTPException(502, ...)`` is raised.
    ``find_offer`` already refuses to offer such a peer, so this is the second
    of two checks and holds however the ``PeerRoute`` was constructed. See
    test_forward_refuses_a_non_loopback_route_without_sending_the_key."""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    if not is_routable_peer_endpoint(route.host, route.scheme):
        clear_route(route.model)
        logger.warning(
            "peer_routing: refusing to forward %r to unroutable endpoint "
            "scheme=%r host=%r (peer %r); clearing the route without sending "
            "the credential",
            route.model, route.scheme, route.host, route.instance_id)
        raise HTTPException(
            502, f"Route for '{route.model}' names peer endpoint "
            f"{route.scheme!r}://{route.host!r}, which this instance cannot "
            "identity-verify; the route has been cleared. Retry to load a "
            "local copy.")

    body = await request.body()
    fwd_headers = {"Authorization": f"Bearer {route.api_key}"}
    content_type = request.headers.get("content-type")
    if content_type:
        fwd_headers["Content-Type"] = content_type

    url = _peer_url(route, path)
    loop = asyncio.get_running_loop()

    import requests
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        verify = False

    def _send():
        return requests.post(
            url, data=body, headers=fwd_headers, stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), verify=verify)

    try:
        resp = await loop.run_in_executor(None, _send)
    except requests.RequestException as e:
        clear_route(route.model)
        logger.warning("peer_routing: forwarding '%s' to peer %s (%s:%s) "
                       "failed, clearing the route: %s",
                       route.model, route.instance_id, route.host, route.port, e)
        raise HTTPException(
            502, f"Peer instance at {route.host}:{route.port} became "
            f"unavailable while routing '{route.model}'; the route has been "
            "cleared. Retry to load a local copy, or re-offer routing.")

    def _iter_sync():
        return resp.iter_content(chunk_size=None)

    it = await loop.run_in_executor(None, _iter_sync)

    async def _body_iter():
        _sentinel = object()
        while True:
            chunk = await loop.run_in_executor(None, next, it, _sentinel)
            if chunk is _sentinel:
                break
            if chunk:
                yield chunk

    media_type = resp.headers.get("content-type", "application/json")
    return StreamingResponse(
        _body_iter(), status_code=resp.status_code, media_type=media_type)
