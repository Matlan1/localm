# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-authenticated loopback HTTP client: call THIS server's own API."""

from __future__ import annotations

from typing import Optional

import requests

from localm.bindhost import self_connect_host, url_host


def read_activity(scheme: str, port, instance_token: Optional[str] = None,
                  bind_host: Optional[str] = None) -> tuple:
    """Ask a running localm server what it is doing."""
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
    """This server's own ``/v1`` base URL, or None if it cannot be determined."""
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
    """Call this server's own API: ``method`` *path* against *base_url*, with the auth/TLS handling every self-call needs."""
    if not base_url:
        raise ValueError("self_request: base_url is required")
    from localm.auth import resolve_bearer_headers
    headers = resolve_bearer_headers(instance_token)
    from localm import tls as _tls
    url = f"{base_url}{path}"
    return requests.request(method, url, json=json, headers=headers,
                            timeout=timeout, verify=_tls.requests_verify(url))
