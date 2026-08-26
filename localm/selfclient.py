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

import re
from typing import Optional

import requests

from localm.bindhost import self_connect_host, url_host

# Matches every C0/C1 control byte, including ESC (the ANSI escape
# introducer). See test_remote_hold_reason_strips_control_characters.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _no_control_chars(value: str) -> str:
    return _CONTROL_CHARS.sub("", value)


def read_activity(scheme: str, port, instance_token: Optional[str] = None,
                  bind_host: Optional[str] = None) -> tuple:
    """Ask a running localm server what it is doing. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body (may hold an empty list)
      ``"unauthorized"`` - the server wants a key this client does not have
      ``"unsupported"``  - the server has no activity route (an older localm)
      ``"http"``         - some other HTTP status; payload is the code
      ``"unreachable"``  - could not connect; payload is a short reason

    Each non-ok state is distinct. An empty list is a real answer and is only
    ever returned under ``"ok"``.

    Shared by ``localm status`` and the MCP activity tool.

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

    Only ``"ok"`` is an answer; no other state is folded into
    ``held: False``. ``"absent"`` (404) means this instance serves a different
    data home, so it does not carry THIS file at all.

    Takes the same parameters as :func:`read_activity`: the per-instance attach
    token for a keyless server, and the bound host so an IPv6-bound instance is
    dialled where it actually listens.
    """
    from urllib.parse import quote

    from localm import tls as _tls
    from localm.auth import resolve_bearer_headers

    host = url_host(self_connect_host(bind_host))
    # Quote with no safe characters: a "/" in a registry name would otherwise
    # re-point the request at a different route.
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
        # "Model not registered: <name>". An unreadable body falls to
        # "unsupported", which refuses.
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
        # A 200 whose body is not JSON means something other than localm
        # answered, or answered wrongly.
        return "http", r.status_code
    if not isinstance(body, dict) or not isinstance(body.get("held"), bool):
        # A 200 whose JSON carries no boolean "held" is reported as "http".
        return "http", r.status_code
    return "ok", body


def remote_hold_reason(model: str) -> Optional[str]:
    """Ask every OTHER running localm instance whether it holds the file that
    removing *model* would delete. Returns None only when every discovered
    instance POSITIVELY RULES ITSELF OUT; otherwise a ready-to-print reason
    naming which instance could not be ruled out (or could not be asked).

    Used by ``localm rm`` and the MCP ``remove_model`` tool, which delete a
    registry entry's file from a fresh, one-shot process and so have no
    in-process engine map to consult. Each discovered instance is asked over
    :func:`read_model_file_hold`.

    Every outcome that is not an answer is a refusal, and the message says
    which one it was.

    With no server running at all, ``instances.snapshot`` yields nothing to
    loop over and this returns None immediately.
    """
    from localm import instances
    from localm.bindhost import self_connect_host, url_host
    from localm.config import home_dir

    # include_token=True: a genuinely open (keyless) instance's middleware
    # requires the attach token. The token is only sent over HTTP, never
    # displayed.
    rows = instances.snapshot(home_dir(), include_token=True)
    for e in rows:
        scheme = e.get("scheme", "http")
        where = (scheme + "://"
                 + url_host(self_connect_host(e.get("host")))
                 + ":" + str(e.get("port")))
        if not e.get("alive"):
            # snapshot() has already reaped entries whose pid has died, so a
            # listed instance that did not answer a /whoami is a live process
            # of unknown state, and unknown refuses.
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
                reason = _no_control_chars(str(reason))
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

    ``app.state.self_url`` is published by ``attach_gui`` only, so under a
    headless ``localm serve`` the fallback rebuilds it from what
    ``instances.advertise()`` publishes (``instance_scheme`` /
    ``instance_port``, both set before uvicorn accepts connections).

    Returns None rather than ``""`` when it cannot tell.
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
    configured.

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
