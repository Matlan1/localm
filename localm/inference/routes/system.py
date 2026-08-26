# SPDX-License-Identifier: AGPL-3.0-or-later
"""System routes: health, CA download, instance identity, and on-demand GUI mount.

The api_landing "/" route and all middleware stay in create_app. Reads the live
engine from the http_server module global, so a model swap is reflected here.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

import localm.inference.http_server as _hs
from localm import scopes
from localm.bindhost import is_loopback_host


def register(app: FastAPI, ctx) -> None:
    _bearer_token = _hs._bearer_token
    mount_gui_surface = _hs.mount_gui_surface

    # ---------------------------------------------------------------- #
    #  Health                                                            #
    # ---------------------------------------------------------------- #

    @app.get("/health")
    async def health():
        if _hs._engine is not None:
            return {
                "status": "ok",
                "model":  _hs._engine.display_name,
                "loaded": _hs._engine.loaded,
            }
        # No engine is warm. An eviction (unload_all_models) keeps the Engine in
        # _engines for a lazy reload, so resolve the unnamed model the way
        # get_engine does and 503 only when there is nothing to recover.
        name = _hs._resolve_unnamed_model_name()
        if name and name in _hs._engines:
            return {"status": "ok", "model": name, "loaded": False}
        raise HTTPException(503, "No engine initialised")

    # ---------------------------------------------------------------- #
    #  Instance identity                                                 #
    # ---------------------------------------------------------------- #

    @app.get("/whoami", include_in_schema=False)
    async def whoami(request: Request):
        """Identity handshake for instance discovery: confirms this really is a
        localm server and which instance/project it serves. Unauthenticated like
        /health; never returns the attach token or pid. Fields are set on
        app.state by the surface's advertise() wrapper (None before startup
        wiring, e.g. an app mounted standalone in tests).

        root_dir is an absolute host path that can carry the OS username, so it is
        disclosed only on a loopback bind and omitted over the network. Discovery
        matches root_dir from the 0600 registry file, not from /whoami."""
        from localm import instances
        st = request.app.state
        root = getattr(st, "root_dir", None)
        if root is not None:
            bind = getattr(st, "bind_host", "127.0.0.1")
            if not is_loopback_host(bind):
                root = None
        return instances.whoami_payload(
            instance_id=getattr(st, "instance_id", None),
            root_dir=root,
            mode=getattr(st, "instance_mode", None),
        )

    # ---------------------------------------------------------------- #
    #  Surface management: on-demand GUI mount                           #
    # ---------------------------------------------------------------- #

    @app.post("/v1/surfaces/gui", include_in_schema=False)
    async def mount_gui(request: Request):
        """Mount the GUI surface on this running instance.

        An ``api``-mode instance (``localm serve``) serves only /v1; a later
        ``localm gui`` in the same dir calls this to add the GUI + coder live, in
        the one process and with no second model load, then opens it.

        OWNER-level: mounting the GUI exposes the coder agent (shell + file
        edits). Authorized by EITHER this instance's attach token (the local
        same-user secret in the 0600 ``run/`` file, which the attaching process
        reads) OR an API key granting ADMIN. Exempt from the same-origin guard;
        this token/key check is the gate. Idempotent: a full instance returns
        already_mounted."""
        from localm.auth import ct_equal
        presented = _bearer_token(request)
        st = request.app.state
        inst_token = getattr(st, "instance_token", None)
        # ct_equal, not compare_digest: the presented token is a caller-supplied,
        # latin-1 decoded header and may be non-ASCII.
        ok = ct_equal(presented, inst_token)
        if not ok and presented:
            # Fall back to an owner/ADMIN API key (protected mode).
            from localm.auth import any_key_configured, verify
            if any_key_configured():
                held = verify(presented)
                ok = held is not None and scopes.grants(held, scopes.ADMIN)
        if not ok:
            raise HTTPException(
                403, "Surface management requires this instance's attach token "
                "or an owner API key.")
        mounted = mount_gui_surface(request.app)
        return {"status": "mounted" if mounted else "already_mounted",
                "mode": "full"}

    # ---------------------------------------------------------------- #
    #  Built-in TLS: CA download                                         #
    # ---------------------------------------------------------------- #

    @app.get("/localm-ca.crt", include_in_schema=False)
    async def localm_ca_cert():
        """Serve localm's local CA certificate, so a browser or phone can trust
        the built-in TLS. Public and unauthenticated; the CA private key never
        leaves ``<home>/tls``. 404 when this install has no CA (e.g. a loopback
        / plain-HTTP run that never generated one)."""
        from localm import tls
        from localm.config import home_dir
        ca = tls.ca_cert_path(home_dir())
        if not ca.is_file():
            raise HTTPException(404, "No localm CA on this server (TLS not enabled).")
        return Response(
            content=ca.read_bytes(),
            media_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": 'attachment; filename="localm-ca.crt"'},
        )
