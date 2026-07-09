# SPDX-License-Identifier: AGPL-3.0-or-later
"""System routes: health, CA download, instance identity, and on-demand GUI mount.

Extracted verbatim from create_app(); behavior unchanged. The api_landing "/"
route and all middleware stay in create_app (they are conditional / framework
plumbing). Reads the live engine from the http_server module global so a model
swap is reflected here.
"""

from __future__ import annotations

import hmac

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
        if _hs._engine is None:
            raise HTTPException(503, "No engine initialised")
        return {
            "status": "ok",
            "model":  _hs._engine.display_name,
            "loaded": _hs._engine.loaded,
        }

    # ---------------------------------------------------------------- #
    #  Instance identity (H6 server-rework, phase 3)                    #
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
        matches root_dir from the 0600 registry file (not from /whoami), so this
        omission breaks nothing (security review 2026-06-20)."""
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
    #  Surface management: on-demand GUI mount (H6 phase 5)              #
    # ---------------------------------------------------------------- #

    @app.post("/v1/surfaces/gui", include_in_schema=False)
    async def mount_gui(request: Request):
        """Mount the GUI surface on this running instance (the phase-5 on-demand
        mount). An ``api``-mode instance (``localm serve``) serves only /v1; a
        later ``localm gui`` in the same dir calls this to add the GUI + coder
        live - one process, no second model load - then opens it.

        Mounting the GUI exposes the coder agent (shell + file edits), so this is
        an OWNER-level action. Authorized by EITHER this instance's attach token
        (the local same-user secret in the 0600 ``run/`` file, which the
        attaching process reads) OR an API key granting ADMIN. It is exempt from
        the same-origin guard (a local non-browser caller has no Origin); this
        token/key check is the gate. Idempotent: a full instance returns
        already_mounted."""
        presented = _bearer_token(request)
        st = request.app.state
        inst_token = getattr(st, "instance_token", None)
        ok = bool(presented and inst_token
                  and hmac.compare_digest(presented, inst_token))
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
    #  Built-in TLS: CA download (NET-1)                                 #
    # ---------------------------------------------------------------- #

    @app.get("/localm-ca.crt", include_in_schema=False)
    async def localm_ca_cert():
        """Serve localm's local CA certificate so a browser or phone can trust
        the built-in TLS once - removing the warning and enabling PWA install.
        Deliberately public and unauthenticated: a CA *certificate* carries no
        secret (the CA private key never leaves ``<home>/tls``), and the client
        needs it before it can present a key. 404 when this install has no CA
        (e.g. a loopback / plain-HTTP run that never generated one)."""
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
