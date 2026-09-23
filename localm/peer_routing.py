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
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
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
    # The peer's own name for the model; forwarded requests name it this way.
    # None means the same name as *model*.
    peer_model: Optional[str] = None

    def safe_dict(self) -> dict:
        """The subset of this route safe to hand back to a client - never
        ``api_key``."""
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "model": self.model,
            "peer_model": self.peer_model or self.model,
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


def local_identity(registry: dict, model_name: str) -> dict:
    """*model_name*'s file identity in this instance's *registry*:
    ``{"path", "size", "sha256"}``, each None when unknown. Same shape the
    coordination registry advertises for a peer's loaded models."""
    ident = {"path": None, "size": None, "sha256": None}
    entry = registry.get(model_name) if isinstance(registry, dict) else None
    if not isinstance(entry, dict):
        return ident
    if isinstance(entry.get("sha256"), str) and entry["sha256"]:
        ident["sha256"] = entry["sha256"].lower()
    try:
        from localm.model_manager.registry import get_model_info
        info = get_model_info(model_name, reg=registry)
        if info is not None and info[0]:
            p = Path(info[0]).resolve()
            ident["path"] = str(p)
            if p.is_file():
                ident["size"] = p.stat().st_size
    except (OSError, ValueError) as e:
        logger.debug("peer_routing: could not resolve %s for its identity: %s",
                     model_name, e)
    return ident


def _same_file(a: dict, b: dict) -> Optional[bool]:
    """Whether identities *a* and *b* name the same model file: True, False, or
    None when either carries nothing to compare.

    Equal paths, or equal sha256 digests, are the same file. Two files with
    the same base name and the same byte size are treated as the same file.
    Anything else that can be compared is a different file."""
    pa, pb = a.get("path"), b.get("path")
    if pa and pb and os.path.normcase(pa) == os.path.normcase(pb):
        return True
    ha, hb = a.get("sha256"), b.get("sha256")
    if ha and hb:
        return ha.lower() == hb.lower()
    sa, sb = a.get("size"), b.get("size")
    if pa and pb and isinstance(sa, int) and isinstance(sb, int):
        return (os.path.basename(os.path.normcase(pa))
                == os.path.basename(os.path.normcase(pb)) and sa == sb)
    if pa and pb:
        return False
    return None


def _peer_loaded_models(peer: dict) -> list:
    """The peer entry's loaded models as ``{"name", "path", "size", "sha256"}``
    dicts. An entry that advertises only its active ``model`` yields that one
    name with no identity."""
    out = []
    listed = peer.get("models")
    if isinstance(listed, list):
        for m in listed:
            if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"]:
                out.append(m)
    if not out and isinstance(peer.get("model"), str) and peer["model"]:
        out.append({"name": peer["model"]})
    return out


def find_offer(canonical_name: str, aliases, *, exclude_self_id: Optional[str] = None,
               identity: Optional[dict] = None,
               instance_id: Optional[str] = None) -> Optional[dict]:
    """A live, identity-verified peer (via ``gpu_registry.list_gpu_peers``)
    with a loaded model that is the same model as *canonical_name*, or None.
    The returned dict is the peer's registry entry plus ``"matched_model"``,
    the peer's own name for that model.

    With *identity* (see :func:`local_identity`) a peer model is matched by
    FILE: the same path, the same sha256, or the same base name and byte size.
    When the two sides cannot be compared (either identity is empty) the match
    falls back to NAME: the peer's name equals *canonical_name* or any of
    *aliases*, exact first, then casefolded. A name match whose files differ
    is never a match. With *instance_id*, only that peer is considered.

    A peer whose registered ``host``/``scheme`` fails
    :func:`is_routable_peer_endpoint` is skipped and logged at WARNING,
    however well it matches.

    Best-effort: any failure reading the registry is logged and yields None,
    never raised, matching every other public function in gpu_registry."""
    if not canonical_name:
        return None
    names = {canonical_name, *aliases}
    folded = {n.casefold() for n in names}
    ident = identity or {}
    try:
        from localm import gpu_registry
        peers = gpu_registry.list_gpu_peers(exclude_self_id=exclude_self_id)
    except Exception as e:
        logger.debug("peer_routing: peer lookup failed: %s", e)
        return None
    for peer in peers:
        if instance_id is not None and peer.get("instance_id") != instance_id:
            continue
        matched = None
        for m in _peer_loaded_models(peer):
            same = _same_file(ident, m)
            if same is True:
                matched = m["name"]
                break
            if same is None and (m["name"] in names or m["name"].casefold() in folded):
                matched = m["name"]
                break
        if matched is None:
            continue
        model = matched
        if not is_routable_peer_endpoint(peer.get("host"), peer.get("scheme") or "http"):
            logger.warning(
                "peer_routing: peer %r advertises %r at unroutable endpoint "
                "scheme=%r host=%r; not offering it, because only a loopback "
                "address over http or https has had its occupant "
                "identity-verified",
                peer.get("instance_id"), model, peer.get("scheme"), peer.get("host"))
            continue
        return {**peer, "matched_model": matched}
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


def _auth_headers(api_key: Optional[str]) -> dict:
    """The Authorization header for *api_key*, or none for an empty key (a peer
    in open mode needs no credential)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def forward_body(route: PeerRoute, raw: bytes) -> bytes:
    """*raw*, a JSON request body, rewritten for *route*'s peer: ``model`` names
    the model the way the peer does, and ``pin_model`` is true so the peer
    answers with exactly that model. A body that is not a JSON object is
    returned unchanged."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(data, dict):
        return raw
    data["model"] = route.peer_model or route.model
    data["pin_model"] = True
    return json.dumps(data).encode("utf-8")


class PeerCredentialError(Exception):
    """The peer refused the credential offered for a route."""


def verify_peer_credential(peer: dict, api_key: Optional[str], *,
                           timeout: float = 5.0) -> None:
    """Check *api_key* (empty for none) against *peer* before any route using
    it is stored: the peer's ``GET /api/session`` must report the key valid, or
    no key required. A key with any scope passes, as it does for the peer's
    chat endpoint. A peer that does not answer ``/api/session`` is checked
    with an authenticated ``GET /v1/models`` instead.

    Raises :class:`PeerCredentialError` when the key is refused, and
    ``requests.RequestException`` when the peer cannot be reached. Refuses,
    without sending anything, a peer whose endpoint fails
    :func:`is_routable_peer_endpoint`."""
    scheme = peer.get("scheme") or "http"
    if not is_routable_peer_endpoint(peer.get("host"), scheme):
        raise PeerCredentialError(
            f"peer endpoint {scheme!r}://{peer.get('host')!r} is not a "
            "verified loopback address")
    probe = PeerRoute(model="", instance_id=str(peer.get("instance_id") or ""),
                      host=peer.get("host"), port=int(peer.get("port")),
                      scheme=scheme, api_key=api_key or "")
    url = _peer_url(probe, "/v1/models")
    import requests
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        verify = False
    resp = requests.get(_peer_url(probe, "/api/session"), headers=_auth_headers(api_key),
                        timeout=timeout, verify=verify)
    try:
        state = resp.json() if resp.status_code == 200 else None
    except ValueError:
        state = None
    if isinstance(state, dict) and "authed" in state and "required" in state:
        if state["required"] and not state["authed"]:
            raise PeerCredentialError(
                "the peer requires its own API key" if not api_key
                else "the peer rejected that API key")
        return
    resp = requests.get(url, headers=_auth_headers(api_key), timeout=timeout,
                        verify=verify)
    if resp.status_code in (401, 403):
        raise PeerCredentialError(
            "the peer requires its own API key" if not api_key
            else f"the peer rejected that API key (HTTP {resp.status_code})")
    if resp.status_code >= 400:
        raise PeerCredentialError(f"the peer answered HTTP {resp.status_code}")


async def forward(route: PeerRoute, request, path: str, *,
                  body: Optional[bytes] = None, headers: Optional[dict] = None):
    """Forward *request* to *route*'s peer at the fixed literal *path*
    (never a path taken from *request* itself) and stream the response back
    unchanged. Returns a ``fastapi.responses.StreamingResponse``. *body*, when
    given, is sent instead of the request's own body (see
    :func:`forward_body`). *headers* are added to the response returned to
    the client.

    A 401 or 403 from the peer means the credential behind the route no longer
    works (the peer's key was changed or removed): the route is CLEARED and
    ``HTTPException(502, ...)`` is raised saying so.

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

    if body is None:
        body = await request.body()
    fwd_headers = _auth_headers(route.api_key)
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

    if resp.status_code in (401, 403):
        resp.close()
        clear_route(route.model)
        logger.warning("peer_routing: peer %s (%s:%s) refused the credential for "
                       "'%s' (HTTP %s), clearing the route",
                       route.instance_id, route.host, route.port, route.model,
                       resp.status_code)
        raise HTTPException(
            502, f"Peer instance at {route.host}:{route.port} refused the API key "
            f"used for '{route.model}' (HTTP {resp.status_code}); the route has "
            "been cleared. Retry to load a local copy, or re-offer routing with "
            "that instance's current key.")

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
        _body_iter(), status_code=resp.status_code, media_type=media_type,
        headers=dict(headers or {}))
