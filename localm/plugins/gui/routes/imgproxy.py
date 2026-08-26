# SPDX-License-Identifier: AGPL-3.0-or-later
"""Remote-image proxy for the chat renderer.

A model reply can link an image (`![alt](https://host/pic.png)`). The shell's CSP
is `img-src 'self' data: blob:`, so the browser refuses to fetch it and the user
sees a broken image.

The client rewrites the `<img src>` to point here, and localm fetches the bytes
SERVER-side through the same `netpolicy` path every other outbound request uses.
The browser never contacts the remote origin.

OFF BY DEFAULT (`gui_proxy_remote_images`). Proxying decides WHO makes the
request, not WHETHER it is made: a crafted URL still reaches the attacker's
server the moment the reply renders.

THE `ask` STATE refuses with 428 unless the request carries `consent=1`, and the
refusal happens BEFORE any outbound fetch, so a host the reader has not agreed to
is never contacted at all.

THE CONSENT ITSELF LIVES IN THE BROWSER, NOT HERE. It is remembered per ORIGIN
for the duration of one conversation in one page session, so it never reaches
disk and never crosses a conversation. `consent=1` is the client stating the
reader's answer, not a capability: it can only reach a state `on` would have
reached anyway, and every other boundary (scope, net_mode, net_allow/net_deny,
the SSRF guard, the size and type caps) is unchanged and still applies. A model
cannot smuggle it in through the URL it chose - the client builds the query with
encodeURIComponent, which escapes `&` and `=`. See
test_a_url_carrying_its_own_consent_parameter_is_still_refused.

The URL comes straight off a query parameter. It is bounded by exactly the same
`netpolicy` decisions as every other outbound request, including the
private-address guard - but that guard is what `net_allow_private` turns OFF. So
an owner who has BOTH switched this on and separately disabled the SSRF guard has
a rendered reply able to make this machine fetch a private address. netpolicy
stays the single authority on that; it is not re-checked here.
"""

from __future__ import annotations

import urllib.parse

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response

from localm import scopes
from localm.inference._threadpool_timeout import (ThreadCallTimeout,
                                                  run_in_threadpool_bounded)
from localm.inference.http_server import require_scope

# Byte cap for one proxied image.
_MAX_BYTES = 10_000_000

def _fetch_budget_s() -> float:
    """Deadline in seconds for the offloaded fetch, derived from netpolicy's own
    bounds rather than written as a literal.

    The budget must sit above safe_fetch_bytes' real worst case, so
    run_in_threadpool_bounded fires only for a wedged call and never for a
    slow-but-working one. That worst case is not one timeout: netpolicy applies
    its timeout separately to the connect and to each read, and follows
    redirects MANUALLY (so every hop re-pays both), up to ``_MAX_REDIRECTS``.
    See ``test_the_fetch_budget_stays_above_netpolicy_s_own_worst_case``.

    Falls back to the same arithmetic on default values if either private name
    is absent."""
    from localm import netpolicy
    per_call = float(getattr(netpolicy, "_DEFAULT_TIMEOUT", 15))
    hops = int(getattr(netpolicy, "_MAX_REDIRECTS", 5)) + 1
    return hops * 2 * per_call + 20.0        # connect + read per hop, plus slack

# image/svg+xml is absent: served from this origin an SVG renders as a DOCUMENT
# when its URL is opened directly, and SVG can carry script. Every type listed
# is inert under any interpretation.
_ALLOWED_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
    "image/avif", "image/bmp", "image/x-icon", "image/vnd.microsoft.icon",
    "image/apng", "image/tiff",
})


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/image-proxy",
             dependencies=[Depends(require_scope(scopes.CHAT))])
    async def image_proxy(url: str, consent: str = ""):
        """Fetch a remote image server-side and return its bytes.

        Refuses unless `gui_proxy_remote_images` allows it: never when it is
        `off`, and only with `consent=1` when it is `ask`. The fetch itself goes
        through `netpolicy.safe_fetch_bytes`, so it inherits the per-hop
        `check_url`, the DNS pin against rebind, redirect re-validation and the
        byte cap rather than reimplementing any of them.
        """
        from localm.config import (REMOTE_IMAGE_ASK, REMOTE_IMAGE_OFF,
                                   load_config, remote_image_mode)

        mode = remote_image_mode(load_config())
        if mode == REMOTE_IMAGE_OFF:
            raise HTTPException(
                403, "Showing remote images is off. Set 'Show remote images "
                     "in replies' to 'ask' or 'on' under Settings > Network to "
                     "enable it.")
        if mode == REMOTE_IMAGE_ASK and consent.strip() not in ("1", "true", "yes"):
            # Refused BEFORE any fetch, so a host the reader has not agreed to
            # is never contacted.
            raise HTTPException(
                428, "Showing remote images is set to 'ask', and this site has "
                     "not been allowed in this conversation. Choose to show "
                     "images from it when localm asks, or set 'Show remote "
                     "images in replies' to 'on' under Settings > Network.")

        parsed = urllib.parse.urlparse(url or "")
        if parsed.scheme not in ("http", "https"):
            # file: and data: never reach the fetch layer.
            raise HTTPException(400, "Only http and https images can be proxied.")

        from localm import netpolicy
        # Off the event loop: safe_fetch_bytes is a blocking urlopen, and the URL
        # comes from a model-authored `<img src>`.
        #
        # A CLOSURE, not `run_in_threadpool_bounded(safe_fetch_bytes, url,
        # max_bytes=..., timeout=...)`: safe_fetch_bytes has its OWN `timeout`
        # keyword, so a single `timeout=` binds to the wrapper and leaves the
        # fetch on its own default.
        def _fetch():
            return netpolicy.safe_fetch_bytes(url, max_bytes=_MAX_BYTES)

        try:
            _final, content_type, body = await run_in_threadpool_bounded(
                _fetch, timeout=_fetch_budget_s())
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Fetching the image timed out: {e}")
        except netpolicy.NetworkPolicyError as e:
            # The SSRF guard, the domain lists and the redirect re-check all land
            # here.
            raise HTTPException(403, f"Refused by the network policy: {e}")
        except Exception as e:
            raise HTTPException(502, f"Could not fetch the image: {e}")

        # safe_fetch_bytes TRUNCATES at the cap rather than refusing, so a body
        # that reaches the cap is a partial image. Refuse it rather than serving
        # a corrupt file under a 200 and a valid image/* type.
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
                # Remote-chosen bytes served from this origin, pinned inert.
                "Content-Security-Policy": "default-src 'none'; sandbox",
                # Turning the feature off takes effect at once, and no proxied
                # bytes reach the on-disk HTTP cache.
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
