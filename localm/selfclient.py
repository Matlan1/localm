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

from typing import Optional

import requests


def read_activity(scheme: str, port) -> tuple:
    """Ask a running localm server what it is doing. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body (may hold an empty list)
      ``"unauthorized"`` - the server wants a key this client does not have
      ``"unsupported"``  - the server has no activity route (an older localm)
      ``"http"``         - some other HTTP status; payload is the code
      ``"unreachable"``  - could not connect; payload is a short reason

    Every non-ok state exists so a caller can say WHICH of them happened.
    Folding them together, or folding any of them into an empty operation list,
    would report "nothing is running" on the evidence of never having found
    out - the failure ADR-0008 exists to remove. An empty list is a real answer
    and is only ever returned under ``"ok"``.

    Lives here rather than in the CLI because there are now two surfaces that
    must answer this question identically (``localm status`` and the MCP
    activity tool), and a second copy of a state machine whose entire value is
    telling five outcomes apart is exactly the kind of thing that drifts into
    telling four of them apart.
    """
    from localm import tls as _tls
    from localm.auth import get_api_key

    url = f"{scheme}://127.0.0.1:{port}/api/activity"
    headers = {}
    # env var, else the persisted <home>/auth.key. Reading the env ONLY is the
    # bug `localm unload` and `localm stop` still carry: a `localm key generate`
    # or launcher-keyed server keeps its key in auth.key, so an env-only client
    # 401s against exactly the servers a user is most likely to be running.
    key = get_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
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
        # A 200 whose body is not JSON is not an empty activity list; it means
        # something other than localm answered, or answered wrongly.
        return "http", r.status_code


def resolve_self_url(app) -> Optional[str]:
    """This server's own ``/v1`` base URL, or None if it cannot be determined.

    ``app.state.self_url`` is published by ``attach_gui`` only, so before
    ADR-0008 every self-call (the chat/media VRAM swap, RAG self-embedding) was
    reachable in GUI mode alone. Now that the background-job registry lives at
    kernel level, those same paths run under a headless ``localm serve`` too and
    need an address there.

    The fallback rebuilds it from what ``instances.advertise()`` publishes
    (``instance_scheme`` / ``instance_port``, both set before uvicorn accepts
    connections), which is the same shape the GUI launcher computes.

    Returns None rather than "" when it genuinely cannot tell, so a caller
    reports an honest "this server cannot determine its own address" instead of
    handing an empty string to self_request(), which raises a bare ValueError
    from deep inside a background job.
    """
    url = getattr(app.state, "self_url", "") or ""
    if url:
        return url
    scheme = getattr(app.state, "instance_scheme", None)
    port = getattr(app.state, "instance_port", None)
    if scheme and port:
        return f"{scheme}://127.0.0.1:{port}/v1"
    return None


def self_request(method: str, path: str, *, json: Optional[dict] = None,
                  timeout: float = 30, base_url: Optional[str] = None) -> requests.Response:
    """Call this server's own API: ``method`` *path* against *base_url*, with
    the auth/TLS handling every self-call needs.

    Builds an ``Authorization: Bearer`` header from the ACTIVE owner key
    (``localm.auth.get_api_key`` - the ``LOCALM_API_KEY`` env var, else the
    persisted ``<home>/auth.key``) when one is configured; open mode sends none
    (the endpoint allows it). Reading env-ONLY was the bug behind memory-audit
    cluster 19: on a ``localm key generate`` / launcher-keyed server the key
    lives in auth.key, not the env, so every self-call (RAG self-embed, the
    chat<->media VRAM swap) got a 401 and RAG silently degraded to lexical-only.
    Resolves the TLS verify argument via ``localm.tls.requests_verify`` so a
    loopback HTTPS self-call trusts this install's own local CA.

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
    from localm.auth import get_api_key
    key = get_api_key()          # env var, else the persisted <home>/auth.key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    from localm import tls as _tls
    url = f"{base_url}{path}"
    return requests.request(method, url, json=json, headers=headers,
                            timeout=timeout, verify=_tls.requests_verify(url))
