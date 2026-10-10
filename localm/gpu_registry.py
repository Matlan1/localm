# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live detection of, and communication with, the other localm instances running on
this machine (multi-instance GPU/VRAM cooperation).

Nothing is written to disk. Running instances are found the way any program finds
a listening server: the operating system's table of listening TCP ports
(:mod:`localm.listeners`), narrowed to the range localm claims
(``config.PORT_RANGE``), then asked who they are (``GET /whoami``, must answer
``app == "localm"``) and what they hold (``GET /v1/instances/status``). When the
socket table cannot be read, every port in the range is probed instead.

A peer asks another instance to release its VRAM with
``POST /v1/instances/cooperate-unload``. There is no shared secret: the request
names the requester (its instance id, port and scheme) and carries a random
request id the requester remembers for :data:`REQUEST_TTL_S` seconds. The
receiving instance calls the requester back on ``POST /v1/instances/vouch`` with
that id and acts only if the requester confirms it really sent it. A caller that
never received the id cannot get a confirmation.

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


# ------------------------------------------------------------------ #
#  This instance's own live status                                    #
# ------------------------------------------------------------------ #

_status_provider: Optional[Callable[[], Optional[dict]]] = None


def set_local_status_provider(provider: Optional[Callable[[], Optional[dict]]]) -> None:
    """Register (or with None, clear) the callable that reports THIS instance's
    live coordination status: ``{instance_id, pid, port, host, scheme, model,
    models, vram_estimate_bytes, gpu_index}``, or None when this instance does not
    coordinate (a plain test app or an ``--isolated`` run)."""
    global _status_provider
    _status_provider = provider


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


def _probe_range(lo: int, hi: int) -> list:
    """Every port in ``lo..hi`` accepting a TCP connection on loopback, as
    ``(address, port)`` pairs. The fallback when the socket table is unreadable."""
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
        return [r for r in ex.map(_try, range(lo, hi + 1)) if r is not None]


def candidate_endpoints() -> list:
    """``(dial address, port)`` for every listening TCP port inside localm's claimed
    range that this machine could dial on loopback, one per port. Empty when
    ``LOCALM_PEER_DETECTION=off``."""
    if os.environ.get(DETECTION_ENV, "").strip().lower() == "off":
        return []
    from localm.config import PORT_RANGE
    from localm.listeners import listening_endpoints
    lo, hi = PORT_RANGE
    endpoints = listening_endpoints()
    if endpoints is None:
        found = _probe_range(lo, hi)
    else:
        found = [(a, p) for a, p in endpoints if lo <= p <= hi]
    chosen: dict = {}
    for address, port in found:
        dial = _dial_address(address)
        if dial is not None and port not in chosen:
            chosen[port] = dial
    return [(dial, port) for port, dial in sorted(chosen.items())]


def fetch_status(scheme: str, port: int, timeout: float,
                 dial: str = _LOOPBACK) -> Optional[dict]:
    """``GET /v1/instances/status`` on *dial*:*port*: the peer's live coordination
    status, or None when it is unreachable, refuses, or answers something that is
    not a status object."""
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
    try:
        r = requests.get(url, timeout=timeout, verify=verify, allow_redirects=False)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


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
    if not iid or iid == exclude_self_id:
        return None
    status = fetch_status(scheme, port, timeout, dial)
    if status is None or status.get("instance_id") != iid:
        return None
    try:
        if int(status.get("pid", -1)) == os.getpid():
            return None
    except (TypeError, ValueError):
        return None
    return {**status, "host": dial, "scheme": scheme, "port": port,
            "root_dir": ident.get("root_dir"), "mode": ident.get("mode"),
            "version": ident.get("version")}


def list_gpu_peers(*, exclude_self_id: Optional[str] = None,
                   timeout: float = 0.7) -> list:
    """The other localm instances running on this machine that coordinate GPU use,
    found live: each candidate port inside localm's range must answer
    ``GET /whoami`` as localm and ``GET /v1/instances/status`` as the same
    instance. Excludes *exclude_self_id* and, unconditionally, THIS process.

    Each peer is the status object (``instance_id``, ``pid``, ``model``,
    ``models``, ``vram_estimate_bytes``, ``gpu_index``) plus ``host`` (the loopback
    address dialled), ``scheme``, ``port`` and the ``root_dir`` / ``mode`` /
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

# request id -> (peer instance id, monotonic expiry). Guarded by _pending_lock.
_pending: dict = {}
_pending_lock = threading.Lock()


def _remember_request(request_id: str, peer_instance_id: str) -> None:
    now = time.monotonic()
    with _pending_lock:
        for rid in [r for r, (_p, exp) in _pending.items() if exp <= now]:
            _pending.pop(rid, None)
        _pending[request_id] = (peer_instance_id, now + REQUEST_TTL_S)


def _forget_request(request_id: str) -> None:
    with _pending_lock:
        _pending.pop(request_id, None)


def vouch_for(request_id: object, asker_instance_id: object) -> bool:
    """Whether THIS instance really sent the unload request *request_id* to the
    instance *asker_instance_id*, and it is still within :data:`REQUEST_TTL_S`.
    Confirms at most once: the id is consumed."""
    if not isinstance(request_id, str) or not isinstance(asker_instance_id, str):
        return False
    with _pending_lock:
        entry = _pending.get(request_id)
        if entry is None:
            return False
        peer_id, expiry = entry
        if peer_id != asker_instance_id or expiry <= time.monotonic():
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


def verify_requester(requester: object, request_id: object, self_instance_id: str,
                     *, timeout: float = 2.0) -> bool:
    """Whether *requester* (``{instance_id, port, scheme}``, from a received unload
    request) is a live localm instance that confirms it sent *request_id* to this
    instance. Dials loopback only, and sends nothing unless *requester* is
    well-formed."""
    if not isinstance(requester, dict) or not isinstance(request_id, str) or not request_id:
        return False
    iid, port, scheme = (requester.get("instance_id"), requester.get("port"),
                         requester.get("scheme") or "http")
    if (not isinstance(iid, str) or not iid or iid == self_instance_id
            or isinstance(port, bool) or not isinstance(port, int)
            or not 0 < port < 65536 or scheme not in _SCHEMES):
        return False
    if fetch_whoami(scheme, port, iid, timeout) is None:
        return False
    url = _loopback_url(scheme, _LOOPBACK, port, VOUCH_PATH)
    if url is None:
        return False
    r = _post_json(url, {"request_id": request_id,
                         "asker_instance_id": self_instance_id}, timeout)
    if r is None or r.status_code != 200:
        return False
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return False
    return isinstance(data, dict) and data.get("vouched") is True


def request_cooperative_unload(peer: dict, *, timeout: float = 5.0) -> bool:
    """Ask a live *peer* (as returned by :func:`list_gpu_peers`) to release its own
    VRAM via ``POST /v1/instances/cooperate-unload``. The request names this
    instance and carries a fresh request id this instance will confirm when the
    peer calls back (see the module docstring).

    Refuses before sending anything unless the peer's ``host``/``scheme`` are
    loopback over http or https, and unless this instance itself coordinates
    (:func:`own_status`).

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
    request_id = secrets.token_urlsafe(24)
    _remember_request(request_id, peer_id)
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
