# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-authenticated loopback HTTP client: call THIS server's own API.

Five call sites independently built the same request boilerplate - read
``LOCALM_API_KEY`` from the environment, set an ``Authorization: Bearer``
header, and POST to a loopback ``self_url`` with
``verify=localm.tls.requests_verify(url)`` - to have one part of the running
server call another: RAG's ``_make_self_embed``/``_make_self_classify``/
``_make_self_describe_image`` (``plugins/builtin/rag/plug.py``) and the
chat<->media VRAM swap's ``unload_chat_for_media``/``reload_chat_after_media``
(``vram.py``). Hoisted here, the same way ``bindhost.py``/``pathsafe.py``
already hoist other shared kernel-level plumbing, so every consumer shares one
implementation of the auth/TLS setup instead of five copies that can silently
drift.
"""

from __future__ import annotations

import os
from typing import Optional

import requests


def self_request(method: str, path: str, *, json: Optional[dict] = None,
                  timeout: float = 30, base_url: Optional[str] = None) -> requests.Response:
    """Call this server's own API: ``method`` *path* against *base_url*, with
    the auth/TLS handling every self-call needs.

    Builds an ``Authorization: Bearer`` header from ``LOCALM_API_KEY`` when one
    is set (open mode sends none - the endpoint allows it), and resolves the
    TLS verify argument via ``localm.tls.requests_verify`` so a loopback HTTPS
    self-call trusts this install's own local CA.

    *base_url* is the caller's already-resolved self-URL (e.g.
    ``http://127.0.0.1:PORT/v1``) - required, since every caller already has
    one (published on ``request.app.state.self_url`` or threaded through a
    job); there is no implicit default to guess. Returns the raw
    ``requests.Response`` - callers already have their own per-endpoint
    success/error handling (different payloads, different failure messages),
    so this never raises for a non-2xx status.
    """
    if not base_url:
        raise ValueError("self_request: base_url is required")
    headers = {}
    key = os.environ.get("LOCALM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    from localm import tls as _tls
    url = f"{base_url}{path}"
    return requests.request(method, url, json=json, headers=headers,
                            timeout=timeout, verify=_tls.requests_verify(url))
