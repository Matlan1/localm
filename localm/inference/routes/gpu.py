# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-instance GPU/VRAM coordination routes (see ``localm.gpu_registry``).

The endpoints a sibling localm instance on this machine uses to talk to this one:

- ``GET /v1/instances/status``: this instance's live model / VRAM status. Without
  a credential it is redacted: no ``pid`` or ``root_dir``, each loaded model
  by name, file name and size only (no path, no sha256), and on a
  key-protected instance no model at all. This instance's attach token, or an
  API key granting ``models:read``, reads the full status; for a key that is
  not the owner's, without ``root_dir`` and with model paths scrubbed.
- ``POST /v1/instances/status``: the full status for a sibling that names itself
  as the requester, verified exactly like an unload request below.
- ``POST /v1/instances/cooperate-unload``: a sibling, itself out of local eviction
  candidates, asks THIS instance to release its own VRAM. A SEPARATE auth code path
  from ``require_scope``/``MODELS_WRITE``: the request is honoured only after the
  requester's port is found listening as the OS user running this instance and
  the requester confirms, called back on ``/v1/instances/vouch``, that it really
  sent it (``gpu_registry.verify_requester``). An API key alone does not grant it.
- ``POST /v1/instances/vouch``: confirms (once) that THIS instance sent a given
  unload or status request to the asking instance.

All of them exist only on an instance that coordinates (not ``--isolated``, not a
bare test app). Status and unload also require a loopback bind; vouch does not,
because it confirms only a request id this instance generated and sent. All refuse
any request that carries an ``Origin`` header: a sibling instance is not a browser,
so a browser page cannot read the status or trigger the others."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

import localm.inference.http_server as _hs

_REFUSED = "Cooperation request not verified."


def _from_a_sibling(request: Request) -> bool:
    """Whether the request can be from another localm instance rather than a
    browser page: it carries no ``Origin`` header."""
    return "origin" not in request.headers


def _coordinating(request: Request) -> bool:
    """Whether this instance coordinates, serves only loopback, and the request
    is not from a browser page."""
    return (bool(getattr(_hs, "_gpu_coord", None)) and _from_a_sibling(request)
            and _hs._is_loopback_host(
                getattr(request.app.state, "bind_host", "127.0.0.1")))


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _presents_attach_token(request: Request) -> bool:
    """Whether *request* presents this instance's attach token in its
    ``Authorization`` or ``x-api-key`` header (never a session cookie)."""
    from localm.auth import ct_equal
    presented, source = _hs._request_token(request)
    if source != "header" or not presented:
        return False
    inst_token = getattr(request.app.state, "instance_token", None)
    return bool(inst_token) and ct_equal(presented, inst_token)


def _holds_full_status_credential(request: Request) -> bool:
    """Whether *request* presents, in its ``Authorization`` or ``x-api-key``
    header (never a session cookie), this instance's attach token or an API key
    granting ``models:read``. Blocking (a key check can take the keystore
    lock)."""
    from localm import scopes
    from localm.auth import any_key_configured
    if _presents_attach_token(request):
        return True
    presented, source = _hs._request_token(request)
    if source != "header" or not presented:
        return False
    return any_key_configured() and _hs.caller_grants(request, scopes.MODELS_READ)


def _own_port(coord: dict) -> Optional[int]:
    """The port this instance serves on, from its coordination record, or None."""
    port = coord.get("port")
    return port if isinstance(port, int) and not isinstance(port, bool) else None


def _redacted_model(m: dict) -> dict:
    """A loaded model's identity without its path or sha256: ``name``, plus
    ``file`` (the file or folder name alone) and ``size`` when known."""
    import os
    out = {"name": m.get("name")}
    path = m.get("path")
    if isinstance(path, str) and path:
        out["file"] = os.path.basename(os.path.normpath(path))
    size = m.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        out["size"] = size
    return out


def _redacted(status: dict) -> dict:
    """*status* without ``pid`` and ``root_dir`` and with each loaded model
    reduced to its name, file name and size (:func:`_redacted_model`); on a
    key-protected instance, without any model at all."""
    from localm.auth import any_key_configured, require_auth_enabled
    out = {k: v for k, v in status.items() if k not in ("pid", "root_dir")}
    if any_key_configured() or require_auth_enabled():
        out["model"] = None
        out["models"] = []
    else:
        out["models"] = [_redacted_model(m) for m in status.get("models") or []
                         if isinstance(m, dict)]
    return out


def _for_a_scoped_key(status: dict, fmt) -> dict:
    """*status* with ``root_dir`` set to None and each loaded model's ``path``
    passed through *fmt*."""
    out = dict(status)
    out["root_dir"] = None
    models = []
    for m in status.get("models") or []:
        if isinstance(m, dict) and isinstance(m.get("path"), str) and m["path"]:
            m = {**m, "path": fmt(m["path"])}
        models.append(m)
    out["models"] = models
    return out


def _status_for(request: Request) -> Optional[dict]:
    """``_hs._gpu_status``, redacted unless *request* holds a full-status
    credential. With the attach token or as the operator (open mode or an
    owner key) the full status is returned as it is; for any other
    ``models:read`` key ``root_dir`` is None and each model path goes through
    ``_hs.path_formatter``. Blocking."""
    status = _hs._gpu_status()
    if status is None:
        return None
    if not _holds_full_status_credential(request):
        return _redacted(status)
    if _presents_attach_token(request) or _hs.caller_is_operator(request):
        return status
    return _for_a_scoped_key(status, _hs.path_formatter(request))


def register(app: FastAPI, ctx) -> None:
    @app.get("/v1/instances/status", include_in_schema=False)
    async def instance_status(request: Request):
        """This instance's live coordination status (see ``_hs._gpu_status``),
        redacted for a caller without a credential (see the module docstring)."""
        if not _coordinating(request):
            raise HTTPException(404, "Not Found")
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, _status_for, request)
        if status is None:
            raise HTTPException(404, "Not Found")
        return status

    @app.post("/v1/instances/status", include_in_schema=False)
    async def instance_status_for_sibling(request: Request):
        """This instance's full coordination status for a sibling that names
        itself in ``requester`` and confirms ``request_id`` when called back
        (``gpu_registry.verify_requester``). 403 otherwise, the same answer as
        a refused unload request."""
        from localm import gpu_registry
        body = await _json_body(request)
        coord = getattr(_hs, "_gpu_coord", None)
        loop = asyncio.get_running_loop()
        verified = False
        if coord and _coordinating(request):
            verified = await loop.run_in_executor(
                None, lambda: gpu_registry.verify_requester(
                    body.get("requester"), body.get("request_id"),
                    coord["instance_id"], purpose=gpu_registry.PURPOSE_STATUS,
                    self_port=_own_port(coord)))
        status = await loop.run_in_executor(None, _hs._gpu_status) if verified else None
        if status is None:
            raise HTTPException(403, _REFUSED)
        return status

    @app.post("/v1/instances/vouch", include_in_schema=False)
    async def vouch(request: Request):
        """Confirm that THIS instance sent request ``request_id`` to instance
        ``asker_instance_id`` (on ``asker_port``, for ``purpose``, when named).
        One confirmation per request."""
        from localm import gpu_registry
        body = await _json_body(request)
        if (not getattr(_hs, "_gpu_coord", None) or not _from_a_sibling(request)
                or not gpu_registry.vouch_for(
                    body.get("request_id"), body.get("asker_instance_id"),
                    body.get("asker_port"), body.get("purpose"))):
            raise HTTPException(403, _REFUSED)
        return {"vouched": True}

    @app.post("/v1/instances/cooperate-unload", include_in_schema=False)
    async def cooperate_unload(request: Request):
        """Unload THIS instance's currently-loaded model(s) - the one action a
        sibling instance's cooperation request can trigger. Runs only after the
        named requester confirms the request (``gpu_registry.verify_requester``);
        reuses ``unload_all_models()`` (the same VRAM-release-wait behavior as
        ``POST /v1/models/unload``) rather than duplicating it, never with
        force: no reply on this instance is cancelled, a model still in use
        after the wait stays loaded, and the answer is ``in_use`` when nothing
        was freed. See test_cooperate_unload_never_cuts_a_busy_reply."""
        from localm import gpu_registry
        body = await _json_body(request)
        coord = getattr(_hs, "_gpu_coord", None)
        verified = False
        if coord and _coordinating(request):
            loop = asyncio.get_running_loop()
            verified = await loop.run_in_executor(
                None, lambda: gpu_registry.verify_requester(
                    body.get("requester"), body.get("request_id"),
                    coord["instance_id"], purpose=gpu_registry.PURPOSE_UNLOAD,
                    self_port=_own_port(coord)))
        if not verified:
            # An identical 403 whether coordination is not enabled on this
            # instance or the requester did not confirm, so the two cases are
            # indistinguishable to a caller.
            raise HTTPException(403, _REFUSED)
        result = await _hs.unload_all_models()
        # Minimal response: a sibling only needs to know whether VRAM was
        # released, not this instance's model names/VRAM numbers.
        return {"status": result.get("status", "unloaded")}
