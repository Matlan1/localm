# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remote-image proxy for the chat renderer."""

from __future__ import annotations

import urllib.parse

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response

from localm import scopes
from localm.inference._threadpool_timeout import (ThreadCallTimeout,
                                                  run_in_threadpool_bounded)
from localm.inference.http_server import require_scope

# Display images, not model inputs. media.py allows 25 MB for a vision payload;
# this is the smaller cap that fits "something a person is looking at in a chat
# bubble", and it bounds what one rendered reply can make this server pull.
_MAX_BYTES = 10_000_000

def _fetch_budget_s() -> float:
    """Deadline for the offloaded fetch, DERIVED from netpolicy's own bounds rather than written as a literal."""
    from localm import netpolicy
    per_call = float(getattr(netpolicy, "_DEFAULT_TIMEOUT", 15))
    hops = int(getattr(netpolicy, "_MAX_REDIRECTS", 5)) + 1
    return hops * 2 * per_call + 20.0        # connect + read per hop, plus slack

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
        """Fetch a remote image server-side and return its bytes."""
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
        # OFF THE EVENT LOOP, like /api/gpus, /api/backend and /api/stats, and
        # for a sharper reason than any of them: safe_fetch_bytes is a blocking
        # urlopen whose worst case is a 15s connect plus a 15s read plus up to
        # _MAX_BYTES of transfer, AND the URL comes from a model-authored
        # `<img src>`. Called inline, a rendered reply chose how long the whole
        # server stalled. MEASURED before this offload: GET /api/activity went
        # from 0.018s to 13.76s with ONE in-flight proxy fetch to an unroutable
        # host, and the hang alarm fired independently with this frame on top.
        # The budget is derived from netpolicy's own bounds - see
        # _fetch_budget_s - so it cannot silently fall under the legitimate
        # worst case when either of those constants is retuned.
        #
        # A CLOSURE, not `run_in_threadpool_bounded(safe_fetch_bytes, url,
        # max_bytes=..., timeout=...)`, and that is not style: safe_fetch_bytes
        # has its OWN `timeout` keyword, while the wrapper's `timeout` is
        # keyword-only. Passing one `timeout=` would silently be eaten by the
        # wrapper and leave the fetch on its 15s default - two different waits
        # sharing one name, with no error either way.
        def _fetch():
            return netpolicy.safe_fetch_bytes(url, max_bytes=_MAX_BYTES)

        try:
            _final, content_type, body = await run_in_threadpool_bounded(
                _fetch, timeout=_fetch_budget_s())
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Fetching the image timed out: {e}")
        except netpolicy.NetworkPolicyError as e:
            # The SSRF guard, the domain lists and the redirect re-check all land
            # here. Surface the reason rather than a bare failure: this is the one
            # a user hits when their own allow/deny list is the cause.
            raise HTTPException(403, f"Refused by the network policy: {e}")
        except Exception as e:
            raise HTTPException(502, f"Could not fetch the image: {e}")

        # safe_fetch_bytes TRUNCATES at the cap rather than refusing: it breaks out
        # of the chunk loop and returns what it has (netpolicy.py, `if size >=
        # max_bytes: break`). For a text fetch a clipped body is degraded but
        # usable; for an IMAGE it is a corrupt file, and serving it with a 200 and
        # a valid image/* type would be a step that failed reporting success. The
        # browser would render a half-decoded strip or nothing, with no way for the
        # client to tell that apart from a small image. Refuse instead.
        if len(body) >= _MAX_BYTES:
            raise HTTPException(
                413, f"That image is larger than the {_MAX_BYTES // 1_000_000} MB "
                     "limit for proxied images, so it was not fetched completely.")

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
                # no-store, and this was MEASURED rather than chosen on taste.
                # An earlier `private, max-age=300` meant turning the feature OFF
                # did not take effect for five minutes: the browser kept serving
                # the cached image, so the switch the user had just used appeared
                # to do nothing. It also wrote model-influenced bytes into the
                # on-disk HTTP cache of an offline-first product. The client keeps
                # its own in-page blob cache, so a streaming re-render still does
                # not refetch - the HTTP cache was buying almost nothing.
                "Cache-Control": "no-store",
                # An image is never a download prompt and never a page.
                "X-Content-Type-Options": "nosniff",
            },
        )
