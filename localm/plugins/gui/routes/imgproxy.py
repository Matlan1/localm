# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remote-image proxy for the chat renderer.

A model reply can link an image (`![alt](https://host/pic.png)`). The shell's CSP
is `img-src 'self' data: blob:`, so the browser refuses to fetch it and the user
sees a broken image. Every comparable UI renders it by letting the BROWSER fetch
it, which hands the remote host the user's IP, User-Agent and referrer, and is the
standard model-driven exfiltration channel.

This route closes the capability gap without taking that trade: the client
rewrites the `<img src>` to point here, and localm fetches the bytes SERVER-side
through the same `netpolicy` path every other outbound request uses. The browser
never contacts the remote origin.

OFF BY DEFAULT (`gui_proxy_remote_images`), and the setting's help text says
plainly what it does and does not buy: proxying decides WHO makes the request,
not WHETHER it is made. A crafted URL still reaches the attacker's server the
moment the reply renders. Closing that channel is a separate decision (a per-image
affordance, or an allowlist) and is deliberately not attempted here.
"""

from __future__ import annotations

import urllib.parse

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response

from localm import scopes
from localm.inference.http_server import require_scope

# Display images, not model inputs. media.py allows 25 MB for a vision payload;
# this is the smaller cap that fits "something a person is looking at in a chat
# bubble", and it bounds what one rendered reply can make this server pull.
_MAX_BYTES = 10_000_000

# image/svg+xml is DELIBERATELY ABSENT and it is the sharpest edge in this file.
# An SVG is an image in an <img>, but served from OUR origin it renders as a
# DOCUMENT if the URL is opened directly, and SVG can carry script - so allowing
# it would turn a model-chosen URL into script execution on localm's own origin.
# Every other raster type is inert under any interpretation.
_ALLOWED_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
    "image/avif", "image/bmp", "image/x-icon", "image/vnd.microsoft.icon",
    "image/apng", "image/tiff",
})


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/image-proxy",
             dependencies=[Depends(require_scope(scopes.CHAT))])
    async def image_proxy(url: str):
        """Fetch a remote image server-side and return its bytes.

        Refuses unless `gui_proxy_remote_images` is on. The fetch itself goes
        through `netpolicy.safe_fetch_bytes`, so it inherits the per-hop
        `check_url`, the DNS pin against rebind, redirect re-validation and the
        byte cap rather than reimplementing any of them.
        """
        from localm.config import load_config

        if not load_config().get("gui_proxy_remote_images"):
            # 403 rather than 404: the endpoint exists and the caller is allowed
            # to ask, the OWNER has simply not turned it on. A 404 here would read
            # as "old server" to a client trying to tell those apart.
            raise HTTPException(
                403, "Showing remote images is off. Turn on 'Show remote images "
                     "in replies' under Settings > Network to enable it.")

        parsed = urllib.parse.urlparse(url or "")
        if parsed.scheme not in ("http", "https"):
            # netpolicy would refuse these too, but failing here keeps file: and
            # data: from ever reaching the fetch layer, and gives a clearer error.
            raise HTTPException(400, "Only http and https images can be proxied.")

        from localm import netpolicy
        try:
            _final, content_type, body = netpolicy.safe_fetch_bytes(
                url, max_bytes=_MAX_BYTES)
        except netpolicy.NetworkPolicyError as e:
            # The SSRF guard, the domain lists and the redirect re-check all land
            # here. Surface the reason rather than a bare failure: this is the one
            # a user hits when their own allow/deny list is the cause.
            raise HTTPException(403, f"Refused by the network policy: {e}")
        except Exception as e:
            raise HTTPException(502, f"Could not fetch the image: {e}")

        # Content-Type may carry parameters ("image/png; charset=binary").
        base_type = content_type.split(";", 1)[0].strip().lower()
        if base_type not in _ALLOWED_TYPES:
            raise HTTPException(
                415, f"Refused a non-image response ({base_type or 'no type'}). "
                     "Only raster image types are proxied.")

        return Response(
            content=body,
            media_type=base_type,
            headers={
                # These bytes are attacker-choosable, served from our own origin.
                # nosniff is already global; this pins the response to being inert
                # even if something downstream mis-reads it, and costs nothing for
                # an image.
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "Cache-Control": "private, max-age=300",
                # An image is never a download prompt and never a page.
                "X-Content-Type-Options": "nosniff",
            },
        )
