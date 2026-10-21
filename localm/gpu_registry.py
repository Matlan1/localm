# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live detection of, and communication with, the other localm instances running on
this machine (multi-instance GPU/VRAM cooperation).

Nothing is written to disk. Running instances are found the way any program finds
a listening server: the operating system's table of listening TCP ports
(:mod:`localm.listeners`), narrowed to the range localm claims
(``config.PORT_RANGE``) plus :data:`EXTRA_PEER_PORTS`, then asked who they are (``GET /whoami``, must answer
``app == "localm"``) and what they hold (``GET /v1/instances/status``). When the
socket table cannot be read, every port in the range is probed instead.

A peer asks another instance to release its VRAM with
``POST /v1/instances/cooperate-unload``. There is no shared secret: the request
names the requester (its instance id, port and scheme) and carries a random
request id the requester remembers for :data:`REQUEST_TTL_S` seconds. The
receiving instance calls the requester back on ``POST /v1/instances/vouch`` with
that id and acts only if the listener on the named port belongs to the OS user
running the receiver and confirms it really sent it. A caller that never
received the id cannot get a confirmation, and another user's listener is never
asked.

``GET /v1/instances/status`` without a credential answers a redacted status (no
pid, no project directory, no model file identity). A reader gets the full status with ``POST
/v1/instances/status``, which is verified the same way as an unload request, or,
for an instance of its own install, with that instance's attach token.

Everything here is advisory and best-effort: every public function returns a safe
default rather than raising, failures are logged at debug, and nothing is
escalated into a harder failure than "no peer was found"."""

from __future__ import annotations

import os
import secrets
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from localm.debuglog import logger
from localm.instances import fetch_any_whoami, fetch_whoami

REQUEST_TTL_S = 30.0
#: Floor on the wait for a verified status answer, which includes the peer
#: calling this instance back before it answers.
VERIFIED_STATUS_TIMEOUT_S = 3.0
DETECTION_ENV = "LOCALM_PEER_DETECTION"
STATUS_PATH = "/v1/instances/status"
UNLOAD_PATH = "/v1/instances/cooperate-unload"
VOUCH_PATH = "/v1/instances/vouch"

# Dialled for a listener that accepts any address.
_LOOPBACK = "127.0.0.1"
_WILDCARDS = ("0.0.0.0", "::", "*", "")
_SCHEMES = ("http", "https")
_FALLBACK_PROBE_WORKERS = 100
_FALLBACK_PROBE_TIMEOUT = 0.5

# Ports outside localm's range that discovery also looks at: Ollama's default
# port, which docs/ollama-api.md tells users to serve on.
EXTRA_PEER_PORTS = (11434,)


# ------------------------------------------------------------------ #
#  This instance's own live status                                    #
# ------------------------------------------------------------------ #

_status_provider: Optional[Callable[[], Optional[dict]]] = None
_own_identity: Optional[dict] = None


def set_local_status_provider(provider: Optional[Callable[[], Optional[dict]]],
                              identity: Optional[dict] = None) -> None:
    """Register (or with None, clear) the callable that reports THIS instance's
    live coordination status: ``{instance_id, pid, port, host, scheme, root_dir,
    model, models, vram_estimate_bytes, gpu_index}``, or None when this instance does not
    coordinate (a plain test app or an ``--isolated`` run).

    *identity* is THIS instance's ``{instance_id, port, scheme, root_dir}``, read without
    calling *provider*: it names this instance when it asks a peer for its full
    status (:func:`fetch_status`). Cleared with the provider."""
    global _status_provider, _own_identity
    _status_provider = provider
    _own_identity = (dict(identity) if provider is not None and isinstance(identity, dict)
                     else None)


def own_identity() -> Optional[dict]:
    """THIS instance's ``{instance_id, port, scheme}`` as registered with
    :func:`set_local_status_provider`, or None when none is registered or it
    lacks an instance id or an int port."""
    ident = _own_identity
    if not ident or not isinstance(ident.get("instance_id"), str) or not ident["instance_id"]:
        return None
    port = ident.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        return None
    return ident


def own_status() -> Optional[dict]:
    """This instance's live coordination status from the registered provider, or
    None when none is registered, the provider reports none, or it raises."""
    provider = _status_provider
    if provider is None:
        return None
    try:
        status = provider()
    except Exception as e:
        logger.debug("gpu_registry: local status provider failed: %s", e)
        return None
    return status if isinstance(status, dict) else None


# ------------------------------------------------------------------ #
#  Detecting running instances                                        #
# ------------------------------------------------------------------ #

def _dial_address(address: str) -> Optional[str]:
    """The loopback literal to dial for a listener bound to *address*, or None
    when it listens on a non-loopback interface only."""
    if address in _WILDCARDS:
        return _LOOPBACK
    try:
        import ipaddress
        return address if ipaddress.ip_address(address).is_loopback else None
    except ValueError:
        return None


def _probe_range(lo: int, hi: int, extra: tuple = ()) -> list:
    """Every port in ``lo..hi``, and every port in *extra*, accepting a TCP
    connection on loopback, as ``(address, port)`` pairs. The fallback when the
    socket table is unreadable."""
    def _try(port: int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(_FALLBACK_PROBE_TIMEOUT)
        try:
            s.connect((_LOOPBACK, port))
            return (_LOOPBACK, port)
        except OSError:
            return None
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=_FALLBACK_PROBE_WORKERS) as ex:
        return [r for r in ex.map(_try, (*range(lo, hi + 1), *extra)) if r is not None]


def candidate_endpoints() -> list:
    """``(dial address, port)`` for every listening TCP port inside localm's claimed
    range, or in :data:`EXTRA_PEER_PORTS`, that this machine could dial on
    loopback, one per port. Empty when
    ``LOCALM_PEER_DETECTION=off``."""
    if os.environ.get(DETECTION_ENV, "").strip().lower() == "off":
        return []
    from localm.config import PORT_RANGE
    from localm.listeners import listening_endpoints
    lo, hi = PORT_RANGE
    endpoints = listening_endpoints()
    if endpoints is None:
        found = _probe_range(lo, hi, EXTRA_PEER_PORTS)
    else:
        found = [(a, p) for a, p in endpoints
                 if lo <= p <= hi or p in EXTRA_PEER_PORTS]
    chosen: dict = {}
    for address, port in found:
        dial = _dial_address(address)
        if dial is not None and port not in chosen:
            chosen[port] = dial
    return [(dial, port) for port, dial in sorted(chosen.items())]


def _status_object(r) -> Optional[dict]:
    """The JSON object of a 200 response *r*, or None."""
    if r is None or r.status_code != 200:
        return None
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _get_status(scheme: str, port: int, timeout: float, dial: str,
                token: Optional[str] = None) -> Optional[dict]:
    """``GET /v1/instances/status`` on *dial*:*port*, presenting *token* as a
    bearer credential when given."""
    import requests
    from localm.bindhost import url_host
    url = f"{scheme}://{url_host(dial)}:{int(port)}{STATUS_PATH}"
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        verify = False
    except Exception as e:
        logger.debug("gpu_registry: could not determine TLS verification for %s: %s",
                     url, e)
        return None
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        r = requests.get(url, timeout=timeout, verify=verify, allow_redirects=False,
                         headers=headers)
    except requests.RequestException:
        return None
    return _status_object(r)


def _post_verified_status(scheme: str, port: int, timeout: float, dial: str,
                          instance_id: str) -> Optional[dict]:
    """``POST /v1/instances/status`` naming this instance (:func:`own_identity`)
    and a fresh request id it confirms when the peer calls back, as
    :func:`request_cooperative_unload` does; waits at least
    :data:`VERIFIED_STATUS_TIMEOUT_S`. None when this instance has no identity,
    *instance_id* is this instance, or the peer does not answer a status
    object, and without sending anything when the listener on *port* is not
    this OS user's. The caller must not be this instance's event loop thread,
    which has to answer the callback."""
    me = own_identity()
    if me is None or me["instance_id"] == instance_id:
        return None
    url = _loopback_url(scheme, dial, port, STATUS_PATH)
    if url is None:
        return None
    from localm.listeners import port_owned_by_current_user
    if not port_owned_by_current_user(int(port), dial, fresh=True):
        return None
    request_id = secrets.token_urlsafe(24)
    _remember_request(request_id, instance_id, int(port), PURPOSE_STATUS)
    try:
        r = _post_json(url, {
            "requester": {"instance_id": me["instance_id"], "port": me["port"],
                          "scheme": me.get("scheme") or "http"},
            "request_id": request_id}, max(timeout, VERIFIED_STATUS_TIMEOUT_S))
    finally:
        _forget_request(request_id)
    return _status_object(r)


def _same_install_token(instance_id: str, port: int, dial: str) -> Optional[str]:
    """The attach token this install's instance registry records for
    *instance_id* at *port*, or None. None also when the listener on *port* is
    not this OS user's, so the token is never sent to another user's socket."""
    try:
        from localm.config import home_dir
        from localm.instances import list_entries
        entry = next((e for e in list_entries(home_dir())
                      if e.get("instance_id") == instance_id
                      and str(e.get("port")) == str(port)), None)
    except Exception as e:
        logger.debug("gpu_registry: could not read this install's instances: %s", e)
        return None
    token = entry.get("token") if entry else None
    if not isinstance(token, str) or not token:
        return None
    from localm.listeners import port_owned_by_current_user
    return token if port_owned_by_current_user(int(port), dial) else None


def fetch_status(scheme: str, port: int, timeout: float,
                 dial: str = _LOOPBACK, *,
                 instance_id: Optional[str] = None) -> Optional[dict]:
    """The live coordination status of the peer on *dial*:*port*, or None when it
    is unreachable, refuses, or answers something that is not a status object.

    With *instance_id* (the peer's, from its ``/whoami``) the full status is asked
    for first: by a verified ``POST`` when this instance coordinates
    (:func:`own_identity`), else with the attach token this install records for
    that instance at that port. Otherwise, or when the peer does not answer that,
    the plain ``GET`` answers, which a current peer redacts (no pid, no model file
    identity)."""
    if instance_id:
        full = _post_verified_status(scheme, port, timeout, dial, instance_id)
        if full is not None:
            return full
        token = _same_install_token(instance_id, port, dial)
        if token:
            full = _get_status(scheme, port, timeout, dial, token)
            if full is not None:
                return full
    return _get_status(scheme, port, timeout, dial)


def _probe_peer(endpoint: tuple, exclude_self_id: Optional[str],
                timeout: float) -> Optional[dict]:
    dial, port = endpoint
    for scheme in _SCHEMES:
        ident = fetch_any_whoami(scheme, port, timeout, dial)
        if ident is not None:
            break
    else:
        return None
    iid = ident.get("instance_id")
    me = own_identity()
    if not iid or iid == exclude_self_id or (me is not None and iid == me["instance_id"]):
        return None
    status = fetch_status(scheme, port, timeout, dial, instance_id=iid)
    if status is None or status.get("instance_id") != iid:
        return None
    try:
        if int(status.get("pid", -1)) == os.getpid():
            return None
    except (TypeError, ValueError):
        return None
    return {**status, "host": dial, "scheme": scheme, "port": port,
            "root_dir": status.get("root_dir") or ident.get("root_dir"),
            "mode": ident.get("mode"), "version": ident.get("version")}


def list_gpu_peers(*, exclude_self_id: Optional[str] = None,
                   timeout: float = 0.7) -> list:
    """The other localm instances running on this machine that coordinate GPU use,
    found live: each candidate port inside localm's range, or in
    :data:`EXTRA_PEER_PORTS`, must answer
    ``GET /whoami`` as localm and ``GET /v1/instances/status`` as the same
    instance. Excludes *exclude_self_id* and, unconditionally, THIS process.

    Each peer is the status object (``instance_id``, ``pid``, ``model``,
    ``models``, ``vram_estimate_bytes``, ``gpu_index``) plus ``host`` (the loopback
    address dialled), ``scheme``, ``port``, ``root_dir`` (from the full status,
    which only a verified sibling or a credential reads, else from an older
    peer's ``/whoami``; None when neither carries it) and the ``mode`` /
    ``version`` from its ``/whoami`` answer. Sorted by instance id.

    Best-effort and advisory: any failure yields the peers found so far."""
    try:
        endpoints = candidate_endpoints()
    except Exception as e:
        logger.debug("gpu_registry: could not list candidate ports: %s", e)
        return []
    if not endpoints:
        return []

    def _one(endpoint: tuple) -> Optional[dict]:
        try:
            return _probe_peer(endpoint, exclude_self_id, timeout)
        except Exception as e:
            logger.debug("gpu_registry: probing %s failed: %s", endpoint, e)
            return None

    with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as ex:
        peers = [p for p in ex.map(_one, endpoints) if p is not None]
    peers.sort(key=lambda p: str(p.get("instance_id") or ""))
    return peers


# ------------------------------------------------------------------ #
#  Asking a peer to release its VRAM, and vouching for the ask        #
# ------------------------------------------------------------------ #

PURPOSE_UNLOAD = "unload"
PURPOSE_STATUS = "status"

# request id -> (peer instance id, peer port, purpose, monotonic expiry). Guarded
# by _pending_lock.
_pending: dict = {}
_pending_lock = threading.Lock()


def _remember_request(request_id: str, peer_instance_id: str,
                      peer_port: Optional[int] = None,
                      purpose: Optional[str] = None) -> None:
    now = time.monotonic()
    with _pending_lock:
        for rid in [r for r, entry in _pending.items() if entry[-1] <= now]:
            _pending.pop(rid, None)
        _pending[request_id] = (peer_instance_id, peer_port, purpose,
                                now + REQUEST_TTL_S)


def _forget_request(request_id: str) -> None:
    with _pending_lock:
        _pending.pop(request_id, None)


def vouch_for(request_id: object, asker_instance_id: object,
              asker_port: object = None, purpose: object = None) -> bool:
    """Whether THIS instance really sent request *request_id* to the instance
    *asker_instance_id*, and it is still within :data:`REQUEST_TTL_S`.

    *asker_port* (the port the asker serves on) and *purpose*
    (:data:`PURPOSE_UNLOAD` or :data:`PURPOSE_STATUS`), when given, must match
    the port the request was sent to and what it asked for, wherever those were
    recorded; an asker that names neither is matched on the id alone. A
    malformed *asker_port* or *purpose* is refused. Confirms at most once: the id
    is consumed."""
    if not isinstance(request_id, str) or not isinstance(asker_instance_id, str):
        return False
    if asker_port is not None and (isinstance(asker_port, bool)
                                   or not isinstance(asker_port, int)):
        return False
    if purpose is not None and not isinstance(purpose, str):
        return False
    with _pending_lock:
        entry = _pending.get(request_id)
        if entry is None:
            return False
        peer_id, peer_port, sent_purpose, expiry = entry
        if peer_id != asker_instance_id or expiry <= time.monotonic():
            return False
        if asker_port is not None and peer_port is not None and asker_port != peer_port:
            return False
        if purpose is not None and sent_purpose is not None and purpose != sent_purpose:
            return False
        _pending.pop(request_id, None)
        return True


def _post_json(url: str, body: dict, timeout: float):
    """POST *body* as JSON to a loopback *url*; the response, or None on a
    transport error."""
    import requests
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        verify = False
    except Exception as e:
        logger.debug("gpu_registry: could not determine TLS verification for %s: %s",
                     url, e)
        return None
    try:
        return requests.post(url, json=body, timeout=timeout, verify=verify,
                             allow_redirects=False)
    except requests.RequestException as e:
        logger.debug("gpu_registry: POST %s failed: %s", url, e)
        return None


def _loopback_url(scheme: str, host: object, port: int, path: str) -> Optional[str]:
    """``scheme://host:port/path`` for a peer endpoint that is loopback over
    http or https, or None for anything else."""
    from localm.peer_routing import is_routable_peer_endpoint
    if not is_routable_peer_endpoint(host, scheme):
        return None
    from localm.bindhost import self_connect_host, url_host
    return f"{scheme}://{url_host(self_connect_host(host))}:{int(port)}{path}"


_REQUESTER_DIALS = (_LOOPBACK, "::1")


def _requester_dial(scheme: str, port: int, instance_id: str,
                    timeout: float) -> Optional[str]:
    """The loopback address (127.0.0.1, then ::1) on which *port* is listening
    as this OS user and answers ``/whoami`` as *instance_id*, or None."""
    from localm.listeners import port_owned_by_current_user
    for dial in _REQUESTER_DIALS:
        if (port_owned_by_current_user(port, dial, fresh=True)
                and fetch_whoami(scheme, port, instance_id, timeout, dial) is not None):
            return dial
    return None


def verify_requester(requester: object, request_id: object, self_instance_id: str,
                     *, timeout: float = 2.0, purpose: Optional[str] = None,
                     self_port: Optional[int] = None) -> bool:
    """Whether *requester* (``{instance_id, port, scheme}``, from a received unload
    or status request) is a live localm instance, listening on loopback as the OS
    user running this process, that confirms it sent *request_id* to this
    instance. The socket's owner is read again after the confirmation, so a port
    that changes hands during the exchange is refused.

    Dials loopback only (127.0.0.1, or ::1 for a requester bound there), sends
    nothing unless *requester* is well-formed, and never dials a listener that
    belongs to another OS user. *purpose* and *self_port* (the port this instance
    serves on) are sent with the confirmation request, so the requester confirms
    only a request of that purpose that it sent to that port (:func:`vouch_for`)."""
    if not isinstance(requester, dict) or not isinstance(request_id, str) or not request_id:
        return False
    iid, port, scheme = (requester.get("instance_id"), requester.get("port"),
                         requester.get("scheme") or "http")
    if (not isinstance(iid, str) or not iid or iid == self_instance_id
            or isinstance(port, bool) or not isinstance(port, int)
            or not 0 < port < 65536 or scheme not in _SCHEMES):
        return False
    dial = _requester_dial(scheme, port, iid, timeout)
    if dial is None:
        logger.debug("gpu_registry: requester %r on port %d is not a localm "
                     "instance of this OS user; not asking it to confirm", iid, port)
        return False
    url = _loopback_url(scheme, dial, port, VOUCH_PATH)
    if url is None:
        return False
    body: dict = {"request_id": request_id, "asker_instance_id": self_instance_id}
    if self_port is not None:
        body["asker_port"] = self_port
    if purpose is not None:
        body["purpose"] = purpose
    r = _post_json(url, body, timeout)
    if r is None or r.status_code != 200:
        return False
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return False
    if not (isinstance(data, dict) and data.get("vouched") is True):
        return False
    from localm.listeners import port_owned_by_current_user
    return port_owned_by_current_user(port, dial, fresh=True)


def request_cooperative_unload(peer: dict, *, timeout: float = 5.0) -> bool:
    """Ask a live *peer* (as returned by :func:`list_gpu_peers`) to release its own
    VRAM via ``POST /v1/instances/cooperate-unload``. The request names this
    instance and carries a fresh request id this instance will confirm when the
    peer calls back (see the module docstring).

    Refuses before sending anything unless the peer's ``host``/``scheme`` are
    loopback over http or https, the listener on its port is this OS user's,
    and this instance itself coordinates (:func:`own_status`).

    Advisory and best-effort: any failure (no own status, no port, network error,
    timeout, non-200, malformed body) returns False. A caller must treat False
    exactly like "no peer available"."""
    port = peer.get("port")
    peer_id = peer.get("instance_id")
    if not port or not isinstance(peer_id, str) or not peer_id:
        return False
    me = own_status()
    if not me or not me.get("instance_id") or not me.get("port"):
        return False
    scheme = peer.get("scheme") or "http"
    url = _loopback_url(scheme, peer.get("host"), port, UNLOAD_PATH)
    if url is None:
        logger.warning(
            "gpu_registry: refusing cooperative-unload to peer %r at unroutable "
            "endpoint scheme=%r host=%r; only a loopback address over http or "
            "https has had its occupant identity-verified",
            peer_id, scheme, peer.get("host"))
        return False
    from localm.bindhost import self_connect_host
    from localm.listeners import port_owned_by_current_user
    if not port_owned_by_current_user(int(port), self_connect_host(peer.get("host")),
                                      fresh=True):
        logger.warning(
            "gpu_registry: refusing cooperative-unload to peer %r on port %s: the "
            "listener there does not belong to this OS user", peer_id, port)
        return False
    request_id = secrets.token_urlsafe(24)
    _remember_request(request_id, peer_id, int(port), PURPOSE_UNLOAD)
    try:
        r = _post_json(url, {
            "requester": {"instance_id": me["instance_id"], "port": int(me["port"]),
                          "scheme": me.get("scheme") or "http"},
            "request_id": request_id}, timeout)
    finally:
        _forget_request(request_id)
    if r is None:
        return False
    if r.status_code != 200:
        logger.debug("gpu_registry: cooperate-unload to %s returned %s", url,
                     r.status_code)
        return False
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return False
    return isinstance(data, dict) and data.get("status") in ("unloaded", "already_unloaded")
