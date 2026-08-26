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

    *instance_token* (#953): a genuinely OPEN (keyless) server's open-mode
    middleware needs the caller to prove it is a local process, not a browser -
    an API key does not exist to send in that mode. The per-instance attach
    token from the 0600 registry file (``instances.attach_target``/``snapshot``)
    is that proof; a caller with filesystem access to it is exactly the "local
    process" principal the middleware means to admit. Used ONLY when no API key
    is configured, matching the server's own condition for requiring it at all -
    a protected-mode server keeps using the real key, unaffected.

    *bind_host* is the address that server BOUND (the instance registry
    records it). Omitted, this dials the IPv4 loopback exactly as before -
    correct for every bind that answers there. Passed, an IPv6-bound server
    is dialled on an address it is actually listening on, instead of being
    reported "unreachable" while it is running perfectly.
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
        # A 200 whose body is not JSON is not an empty activity list; it means
        # something other than localm answered, or answered wrongly.
        return "http", r.status_code


def read_model_file_hold(scheme: str, port, model: str,
                         instance_token: Optional[str] = None,
                         bind_host: Optional[str] = None) -> tuple:
    """Ask a running localm server whether one of ITS loaded engines is holding
    the file that removing *model* would delete. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body: ``{"held": bool, ...}``
      ``"absent"``       - that server does not have *model* registered at all
      ``"unauthorized"`` - the server wants a key this client does not have
      ``"unsupported"``  - the server has no hold route (an older localm)
      ``"http"``         - some other HTTP status; payload is the code
      ``"unreachable"``  - could not connect; payload is a short reason

    THE STATES ARE KEPT APART BECAUSE ONLY ONE OF THEM IS AN ANSWER. A caller
    about to delete a model file needs "that server says nothing holds it" and
    "I could not ask that server" to reach it as different facts, because they
    lead to opposite actions: proceed, or refuse. Folding any non-ok state into
    ``held: False`` would delete a live model's file on the evidence of never
    having found out - the same collapse :func:`read_activity` exists to
    prevent for the activity question, and the consequence here is a destroyed
    download rather than a wrong status line.

    ``"absent"`` (404) is deliberately NOT folded into ``"ok"/held: False``
    either. It means this instance serves a different data home, so it is
    genuinely not a holder of THIS file - but that is a conclusion about scope,
    not about residency, and a caller that wants to report accurately why it
    refused (or did not) has to be able to tell them apart.

    Deliberately mirrors :func:`read_activity`'s signature and state machine
    rather than inventing a second shape: both are "ask each discovered
    instance one question over the loopback", and the parameters that make that
    work (the per-instance attach token for a keyless server, the bound host so
    an IPv6-bound instance is dialled where it actually listens) are the same
    in both cases and wrong to re-derive.
    """
    from urllib.parse import quote

    from localm import tls as _tls
    from localm.auth import resolve_bearer_headers

    host = url_host(self_connect_host(bind_host))
    # quote with no safe characters: a registry name reaches here from a tool
    # argument, and a "/" in it would otherwise re-point the request at a
    # different route.
    url = (f"{scheme}://{host}:{port}"
           f"/v1/models/{quote(model, safe='')}/hold")
    headers = resolve_bearer_headers(instance_token)
    try:
        r = requests.get(url, headers=headers, timeout=5,
                         verify=_tls.requests_verify(url))
    except requests.exceptions.RequestException as e:
        return "unreachable", type(e).__name__
    if r.status_code in (401, 403):
        return "unauthorized", r.status_code
    if r.status_code == 404:
        # Ambiguous by status alone: an older server has no such ROUTE, a
        # current one answers 404 for a model IT does not carry. FastAPI's
        # unmatched-route body is {"detail": "Not Found"}; the route's own is
        # "Model not registered: <name>". Read the body rather than guessing,
        # and when it cannot be read, take the CAUTIOUS branch (unsupported,
        # which refuses) rather than the permissive one.
        try:
            detail = str((r.json() or {}).get("detail", ""))
        except ValueError:
            detail = ""
        if detail.startswith("Model not registered"):
            return "absent", r.status_code
        return "unsupported", r.status_code
    if not r.ok:
        return "http", r.status_code
    try:
        body = r.json()
    except ValueError:
        # A 200 whose body is not JSON is not a "nothing holds it"; it means
        # something other than localm answered, or answered wrongly.
        return "http", r.status_code
    if not isinstance(body, dict) or not isinstance(body.get("held"), bool):
        # Same rule one level in: a well-formed HTTP 200 carrying a shape this
        # client cannot read is not evidence that the file is free.
        return "http", r.status_code
    return "ok", body


def remote_hold_reason(model: str) -> Optional[str]:
    """Ask every OTHER running localm instance whether it holds the file that
    removing *model* would delete. Returns None only when every discovered
    instance POSITIVELY RULES ITSELF OUT; otherwise a ready-to-print reason
    naming which instance could not be ruled out (or could not be asked).

    Exists for a caller that shares no memory with any server that might be
    running: ``localm rm`` and the MCP ``remove_model`` tool both delete a
    registry entry's file from a fresh, one-shot process, so neither can
    consult an in-process engine map the way the HTTP server's own remove
    route does (``loaded_engine_holding_model_file``). Asking each discovered
    instance over :func:`read_model_file_hold` is the only way either caller
    can find out.

    EVERY OUTCOME THAT IS NOT AN ANSWER IS A REFUSAL, and the message says
    which one it was. "That server reports nothing holds it" and "I could not
    reach that server" are opposite conclusions, and collapsing them would
    delete a live model's file on the strength of never having found out. A
    refused delete costs one command and names the server to go and check; a
    deleted model file is gone.

    No server running at all is a certain all-clear (``instances.snapshot``
    yields nothing to loop over, so this returns None immediately), never a
    refusal - a tool that blocked every deletion whenever nothing happened to
    be running would be useless.
    """
    from localm import instances
    from localm.bindhost import self_connect_host, url_host
    from localm.config import home_dir

    # include_token=True: this ASKS each instance over HTTP (an internal,
    # non-display use), so it needs the attach token a genuinely open
    # (keyless) instance's middleware requires. Never for anything a human
    # reads.
    rows = instances.snapshot(home_dir(), include_token=True)
    for e in rows:
        scheme = e.get("scheme", "http")
        where = (scheme + "://"
                 + url_host(self_connect_host(e.get("host")))
                 + ":" + str(e.get("port")))
        if not e.get("alive"):
            # A failed /whoami is NOT proof the process is gone: snapshot()
            # reaps entries whose pid has died before this runs, and a listed
            # instance that did not answer is therefore a live process of
            # unknown state, and unknown refuses.
            return (f"a localm server at {where} is registered but did not "
                    f"answer an identity check, so whether it has this "
                    f"model loaded could not be established")
        state, payload = read_model_file_hold(
            scheme, e.get("port"), model, e.get("token"), e.get("host"))
        if state == "ok":
            if not payload.get("held"):
                continue          # this server positively ruled itself out
            key = payload.get("key") or "a loaded model"
            reason = payload.get("reason")
            if reason:
                return (f"the localm server at {where} has {key!r} loaded "
                        f"and {reason}, so it cannot be ruled out as "
                        f"holding this file")
            return (f"the localm server at {where} still has this model's "
                    f"file loaded as {key!r}")
        if state == "absent":
            continue              # that instance serves a different library
        if state == "unauthorized":
            return (f"the localm server at {where} requires an API key this "
                    f"process does not have, so whether it has this model "
                    f"loaded could not be established")
        if state == "unsupported":
            return (f"the localm server at {where} is an older localm that "
                    f"cannot report which models it holds, so whether it "
                    f"has this one loaded could not be established")
        if state == "unreachable":
            return (f"the localm server at {where} could not be reached "
                    f"({payload}), so whether it has this model loaded "
                    f"could not be established")
        return (f"the localm server at {where} answered HTTP {payload} "
                f"instead of reporting what it holds, so whether it has "
                f"this model loaded could not be established")
    return None


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
    persisted ``<home>/auth.key``) when one is configured. Reading env-ONLY
    was the bug behind memory-audit cluster 19: on a ``localm key generate`` /
    launcher-keyed server the key lives in auth.key, not the env, so every
    self-call (RAG self-embed, the chat<->media VRAM swap) got a 401 and RAG
    silently degraded to lexical-only.

    *instance_token*: in OPEN (keyless) mode there is no key to send, and the
    open-mode management gate (``_origin_guard``) requires proof of a local
    process for state-changing calls like ``/v1/models/unload``/``load`` - an
    empty ``Authorization`` header 403s there. Same fix as #953's
    ``read_activity`` (this module): the per-instance attach token from the
    0600 registry file is that proof, and the gate already accepts it. Used
    ONLY when no API key is configured, mirroring ``read_activity``'s own
    condition - a protected-mode server keeps using the real key, unaffected.

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
    from localm.auth import resolve_bearer_headers
    headers = resolve_bearer_headers(instance_token)
    from localm import tls as _tls
    url = f"{base_url}{path}"
    return requests.request(method, url, json=json, headers=headers,
                            timeout=timeout, verify=_tls.requests_verify(url))
