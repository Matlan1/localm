# SPDX-License-Identifier: AGPL-3.0-or-later
"""System routes: health, CA download, instance identity, and on-demand GUI mount."""

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
        # No engine is warm right now, but that is not the same question as
        # "is there nothing here": an eviction (unload_all_models, e.g. the
        # embedder freeing VRAM for a chat model) deliberately keeps the
        # Engine in _engines for a lazy reload, and get_engine's own
        # unnamed-request resolution (PR #1139) already recovers it on the
        # very next chat turn. Reporting a bare 503 here during exactly that
        # window told a GUI health check "no model" while chat itself would
        # have reloaded one on the spot - a plausible reason a user reaches
        # for a manual load instead of just sending a turn. Mirror the SAME
        # resolution chat uses, so /health cannot disagree with it, and only
        # 503 when there truly is nothing to recover.
        name = _hs._resolve_unnamed_model_name()
        if name and name in _hs._engines:
            return {"status": "ok", "model": name, "loaded": False}
        raise HTTPException(503, "No engine initialised")

    # ---------------------------------------------------------------- #
    #  Instance identity (H6 server-rework, phase 3)                    #
    # ---------------------------------------------------------------- #

    @app.get("/whoami", include_in_schema=False)
    async def whoami(request: Request):
        """Identity handshake for instance discovery: confirms this really is a localm server and which instance/project it serves."""
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
        """Mount the GUI surface on this running instance (the phase-5 on-demand mount)."""
        from localm.auth import ct_equal
        presented = _bearer_token(request)
        st = request.app.state
        inst_token = getattr(st, "instance_token", None)
        # ct_equal, not compare_digest: the presented token is a caller-supplied,
        # latin-1 decoded header, so a non-ASCII one would raise instead of 403.
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
    #  Built-in TLS: CA download (NET-1)                                 #
    # ---------------------------------------------------------------- #

    @app.get("/localm-ca.crt", include_in_schema=False)
    async def localm_ca_cert():
        """Serve localm's local CA certificate so a browser or phone can trust the built-in TLS once - removing the warning and enabling PWA install."""
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
