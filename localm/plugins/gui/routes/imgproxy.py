# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remote-image proxy for the chat renderer.

A model reply can link an image (`![alt](https://host/pic.png)`). The shell's CSP
is `img-src 'self' data: blob:`, so the browser refuses to fetch it and the user
sees a broken image.

The client rewrites the `<img src>` to point here, and localm fetches the bytes
SERVER-side through the same `netpolicy` path every other outbound request uses.
The browser never contacts the remote origin.

OFF BY DEFAULT (`gui_proxy_remote_images`). Proxying decides WHO makes the
request, not WHETHER it is made: a crafted URL still reaches the remote server
the moment the reply renders.

The URL here comes straight off a query parameter, so this is the one caller
that feeds `safe_fetch_bytes` a value the BROWSER chose. It is bounded by the
same `netpolicy` decisions as every other outbound request, including the
private-address guard - which is what `net_allow_private` turns OFF. So an owner
who has BOTH switched this on and separately disabled the SSRF guard has a
rendered reply able to make this machine fetch a private address. netpolicy is
the single authority on that; if the combination should be refused outright,
refuse it in `check_url` so every caller inherits it, not here.
"""

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
    """Deadline for the offloaded fetch, DERIVED from netpolicy's own bounds
    rather than written as a literal.

    run_in_threadpool_bounded must never fire for a slow-but-working call, only
    for a wedged one, so the budget sits above safe_fetch_bytes' worst case -
    which is not one timeout. netpolicy applies its timeout separately to the
    connect and to each read, and follows redirects MANUALLY (so every hop
    re-pays both), up to ``_MAX_REDIRECTS``.

    Computed from those two constants so the relation holds when either is
    retuned. Falls back to the same arithmetic on today's values if either
    private name disappears."""
    from localm import netpolicy
    per_call = float(getattr(netpolicy, "_DEFAULT_TIMEOUT", 15))
    hops = int(getattr(netpolicy, "_MAX_REDIRECTS", 5)) + 1
    return hops * 2 * per_call + 20.0        # connect + read per hop, plus slack

# image/svg+xml is DELIBERATELY ABSENT: served from OUR origin an SVG renders as
# a DOCUMENT when the URL is opened directly, and SVG can carry script, so
# allowing it would turn a model-chosen URL into script execution on localm's
# own origin. Every other type listed is inert under any interpretation.
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
        # OFF THE EVENT LOOP, like /api/gpus, /api/backend and /api/stats:
        # safe_fetch_bytes is a blocking urlopen whose worst case is a 15s
        # connect plus a 15s read plus up to _MAX_BYTES of transfer, AND the URL
        # comes from a model-authored `<img src>`. The budget is derived from
        # netpolicy's own bounds - see _fetch_budget_s - so it cannot fall under
        # the legitimate worst case when either of those constants is retuned.
        #
        # A CLOSURE, not `run_in_threadpool_bounded(safe_fetch_bytes, url,
        # max_bytes=..., timeout=...)`: safe_fetch_bytes has its OWN `timeout`
        # keyword, while the wrapper's `timeout` is keyword-only, so a single
        # `timeout=` would be eaten by the wrapper and leave the fetch on its 15s
        # default.
        def _fetch():
            return netpolicy.safe_fetch_bytes(url, max_bytes=_MAX_BYTES)

        try:
            _final, content_type, body = await run_in_threadpool_bounded(
                _fetch, timeout=_fetch_budget_s())
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Fetching the image timed out: {e}")
        except netpolicy.NetworkPolicyError as e:
            # The SSRF guard, the domain lists and the redirect re-check all
            # land here. The reason is surfaced rather than a bare failure: an
            # owner's own allow/deny list is a common cause.
            raise HTTPException(403, f"Refused by the network policy: {e}")
        except Exception as e:
            raise HTTPException(502, f"Could not fetch the image: {e}")

        # safe_fetch_bytes TRUNCATES at the cap rather than refusing: it breaks
        # out of the chunk loop and returns what it has (netpolicy.py, `if size
        # >= max_bytes: break`). For an IMAGE that is a corrupt file, and a 200
        # with a valid image/* type would report success for a step that failed,
        # so this refuses instead.
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
                # These bytes are attacker-choosable and are served from our own
                # origin. nosniff is already global; this pins the response to
                # being inert even if something downstream mis-reads it.
                "Content-Security-Policy": "default-src 'none'; sandbox",
                # no-store: a cached image would keep being served after the
                # feature is switched OFF, and would write model-influenced bytes
                # into the on-disk HTTP cache. The client keeps its own in-page
                # blob cache, so a streaming re-render still does not refetch.
                "Cache-Control": "no-store",
                # An image is never a download prompt and never a page.
                "X-Content-Type-Options": "nosniff",
            },
        )
