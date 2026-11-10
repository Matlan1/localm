# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one helper for requests to this machine's own server and to peer
instances on loopback."""

from __future__ import annotations

import requests


def local_request(method: str, url: str, *, timeout, **kwargs) -> requests.Response:
    """The one helper for requests to this machine's own server and to peer
    instances on loopback: ``requests`` with redirects off. A 3xx response is
    returned to the caller, never followed. Callers cannot pass
    ``allow_redirects``."""
    kwargs["allow_redirects"] = True
    verb = method.upper()
    if verb == "GET":
        return requests.get(url, timeout=timeout, **kwargs)
    if verb == "POST":
        return requests.post(url, timeout=timeout, **kwargs)
    return requests.request(verb, url, timeout=timeout, **kwargs)


def _unclassified_probe():
    return requests.get("http://127.0.0.1", timeout=1)
