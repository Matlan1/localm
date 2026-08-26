# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-authenticated loopback HTTP client: call THIS server's own API.

One implementation of the auth/TLS setup every self-call needs: the
``Authorization: Bearer`` header (owner key, else the per-instance attach token
in open mode) and ``verify=localm.tls.requests_verify(url)`` against a loopback
``self_url``. Consumers include RAG's ``_make_self_embed`` /
``_make_self_classify`` / ``_make_self_describe_image``
(``plugins/builtin/rag/plug.py``) and the chat<->media VRAM swap's
``unload_chat_for_media`` / ``reload_chat_after_media`` (``vram.py``).
"""

from __future__ import annotations

from typing import Optional

import requests

from localm.bindhost import self_connect_host, url_host


def read_activity(scheme: str, port, instance_token: Optional[str] = None,
                  bind_host: Optional[str] = None) -> tuple:
    """Ask a running localm server what it is doing. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body (may hold an empty list)
      ``"unauthorized"`` - the server wants a key this client does not have
      ``"unsupported"``  - the server has no activity route (an older localm)
      ``"http"``         - some other HTTP status; payload is the code
      ``"unreachable"``  - could not connect; payload is a short reason

    Each non-ok state is distinct, so a caller can say WHICH one happened. An
    empty list is a real answer and is only ever returned under ``"ok"``.

    Shared by ``localm status`` and the MCP activity tool, which must answer
    this question identically.

    *instance_token*: a genuinely OPEN (keyless) server's open-mode middleware
    needs the caller to prove it is a local process, and no API key exists to
    send in that mode. The per-instance attach token from the 0600 registry file
    (``instances.attach_target``/``snapshot``) is that proof. Used ONLY when no
    API key is configured; a protected-mode server keeps using the real key.

    *bind_host* is the address that server BOUND (the instance registry records
    it). Omitted, this dials the IPv4 loopback; passed, an IPv6-bound server is
    dialled on an address it is actually listening on.
    """
    from localm import tls as _tls
    from localm.auth import resolve_bearer_headers

    url = f"{scheme}://{url_host(self_connect_host(bind_host))}:{port}/api/activity"
    headers = resolve_bearer_headers(instance_token)
    try:
        r = requests.get(url, headers=headers, timeout=5,
                         verify=_tls.requests_verify(url))
    except requests.exceptions.RequestException as e:
        return "unreachable", type(e).__name__
    if r.status_code in (401, 403):
        return "unauthorized", r.status_code
    if r.status_code == 404:
        return "unsupported", r.status_code
    if not r.ok:
        return "http", r.status_code
    try:
        return "ok", r.json()
    except ValueError:
        # A 200 whose body is not JSON means something other than localm answered.
        return "http", r.status_code


def resolve_self_url(app) -> Optional[str]:
    """This server's own ``/v1`` base URL, or None if it cannot be determined.

    ``app.state.self_url`` is published by ``attach_gui`` only, so under a
    headless ``localm serve`` the fallback rebuilds it from what
    ``instances.advertise()`` publishes (``instance_scheme`` /
    ``instance_port``, both set before uvicorn accepts connections).

    Returns None rather than "" when it cannot tell, so a caller can report
    that instead of handing an empty string to self_request().
    """
    url = getattr(app.state, "self_url", "") or ""
    if url:
        return url
    scheme = getattr(app.state, "instance_scheme", None)
    port = getattr(app.state, "instance_port", None)
    if scheme and port:
        from localm.bindhost import self_connect_host, url_host
        host = url_host(self_connect_host(getattr(app.state, "bind_host", None)))
        return f"{scheme}://{host}:{port}/v1"
    return None


def self_request(method: str, path: str, *, json: Optional[dict] = None,
                  timeout: float = 30, base_url: Optional[str] = None,
                  instance_token: Optional[str] = None) -> requests.Response:
    """Call this server's own API: ``method`` *path* against *base_url*, with
    the auth/TLS handling every self-call needs.

    Builds an ``Authorization: Bearer`` header from the ACTIVE owner key
    (``localm.auth.get_api_key`` - the ``LOCALM_API_KEY`` env var, else the
    persisted ``<home>/auth.key``) when one is configured.

    *instance_token*: in OPEN (keyless) mode there is no key to send, and the
    open-mode management gate (``_origin_guard``) requires proof of a local
    process for state-changing calls like ``/v1/models/unload``/``load``; an
    empty ``Authorization`` header 403s there. The per-instance attach token
    from the 0600 registry file is that proof. Used ONLY when no API key is
    configured, mirroring ``read_activity``.

    Resolves the TLS verify argument via ``localm.tls.requests_verify``, so a
    loopback HTTPS self-call trusts this install's own local CA.

    *base_url* is the caller's already-resolved self-URL (e.g.
    ``http://127.0.0.1:PORT/v1``) and is REQUIRED; there is no implicit default.
    Returns the raw ``requests.Response`` and never raises for a non-2xx status.
    """
    if not base_url:
        raise ValueError("self_request: base_url is required")
    from localm.auth import resolve_bearer_headers
    headers = resolve_bearer_headers(instance_token)
    from localm import tls as _tls
    url = f"{base_url}{path}"
    return requests.request(method, url, json=json, headers=headers,
                            timeout=timeout, verify=_tls.requests_verify(url))
