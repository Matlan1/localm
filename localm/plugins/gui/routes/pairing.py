# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI device-pairing routes: render a scan-to-save QR for the owner key or for a
freshly-minted scoped key.

The hand-built QR-SVG helper is a nested function inside register().
"""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response

from localm import scopes
from localm.executor import get_plugin_executor
from localm.inference.http_server import require_scope


def register(app: FastAPI, ctx) -> None:

    def _pairing_qr_svg(key: str) -> str:
        """DOMPurify-safe SVG QR encoding ``localm-key:<key>`` for device pairing.
        Hand-built from the module matrix (one black <path> over a white <rect>
        with a viewBox): the qrcode lib's SvgImage emits namespace-prefixed
        <svg:rect> in mm with no viewBox, which DOMPurify strips (blank box) and
        which never scales into the CSS box anyway. Plain <rect>/<path> with
        explicit fills survive sanitisation and render black-on-white on any theme."""
        import qrcode
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
        qr.add_data(f"localm-key:{key}")
        qr.make(fit=True)
        matrix = qr.get_matrix()
        n = len(matrix)
        segments = [
            f"M{x} {y}h1v1h-1z"
            for y, row in enumerate(matrix)
            for x, dark in enumerate(row) if dark
        ]
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" '
            f'shape-rendering="crispEdges" role="img" '
            f'aria-label="localm pairing code">'
            f'<rect width="{n}" height="{n}" fill="#ffffff"/>'
            f'<path d="{"".join(segments)}" fill="#000000"/></svg>'
        )

    @app.get("/api/pairing/qr",
             dependencies=[Depends(require_scope(scopes.ADMIN))])
    async def pairing_qr():
        """SVG QR encoding the OWNER API key (``localm-key:<key>``) so a phone can
        scan it on the onboarding screen and SAVE the key - no typing. Owner scope
        only: it carries the key. 404 in open mode (no key -> nothing to pair).
        Rendered server-side; never cached. For a scoped/limited key the Keys &
        devices manager POSTs the freshly-minted key to the sibling endpoint.

        Runs off-thread: qrcode/PIL are imported cold on first use in this
        process, which can itself take several seconds - keep that off the
        event loop the same way the other startup-time GUI probes do."""
        from localm import auth
        key = auth.get_api_key()
        if not key:
            raise HTTPException(404, "No API key configured - nothing to pair.")
        loop = asyncio.get_running_loop()
        svg = await loop.run_in_executor(get_plugin_executor(), _pairing_qr_svg, key)
        return Response(content=svg, media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/pairing/qr",
              dependencies=[Depends(require_scope(scopes.ADMIN))])
    async def pairing_qr_for_key(body: dict):
        """Render a pairing QR for an ARBITRARY scoped key the owner JUST minted -
        the plaintext is passed in the BODY so it never lands in a URL / access
        log. Owner-gated, never persisted or cached: the key is rendered into the
        SVG and discarded. The phone scans it exactly like the owner-key QR,
        pairing that device with the LIMITED key instead of full admin."""
        key = (body or {}).get("key")
        if not isinstance(key, str) or not key.strip():
            raise HTTPException(400, "Provide the minted key plaintext as 'key'.")
        # Only render a QR for a key that actually exists and is current: doing it
        # for arbitrary input is pointless, and this rejects garbage / an already
        # expired key (verify() returns None for both).
        from localm import auth
        if auth.verify(key.strip()) is None:
            raise HTTPException(400, "Not a current localm key (mint one first).")
        loop = asyncio.get_running_loop()
        svg = await loop.run_in_executor(get_plugin_executor(), _pairing_qr_svg, key.strip())
        return Response(content=svg,
                        media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})
