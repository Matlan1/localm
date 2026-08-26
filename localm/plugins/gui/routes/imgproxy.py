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
plainly what "on" does and does not buy: proxying decides WHO makes the request,
not WHETHER it is made. A crafted URL still reaches the attacker's server the
moment the reply renders.

THE `ask` STATE IS WHAT CLOSES THAT CHANNEL, and the shape is worth reading
before changing anything here. It refuses with 428 unless the request carries
`consent=1`, and the refusal happens BEFORE any outbound fetch, so a host the
reader has not agreed to is never contacted at all. 428 rather than 403 because
the two say different things to a client: 403 is "refused, nothing you can do",
428 is "there is a precondition you can satisfy and then retry", which is
exactly the case - the client asks the reader and re-requests.

WHERE THE CONSENT ACTUALLY LIVES IS THE BROWSER, NOT HERE, and that is
deliberate. It is remembered per ORIGIN for the duration of one conversation in
one page session, so it never reaches disk and never crosses a conversation.
Keeping it server-side would mean inventing a lifetime for it and sharing one
reader's answer with every other tab and browser pointed at this instance. So
`consent=1` is the client stating the reader's answer, not a capability: it can
only reach a state `on` would have reached anyway, and every real boundary
(scope, net_mode, net_allow/net_deny, the SSRF guard, the size and type caps)
is unchanged and still applies. A model cannot smuggle it in through the URL it
chose - the client builds the query with encodeURIComponent, which escapes `&`
and `=`. See test_a_url_carrying_its_own_consent_parameter_is_still_refused.

ONE INTERACTION WORTH KNOWING, recorded rather than left silent. The URL here
comes straight off a query parameter, which makes this the first caller to feed
`safe_fetch_bytes` a value the BROWSER chose rather than one the model or localm
built. It is still bounded by exactly the same `netpolicy` decisions as every
other outbound request, including the private-address guard - but that guard is
what `net_allow_private` turns OFF. So an owner who has BOTH switched this on and
separately disabled the SSRF guard (a setting whose own label reads "disables the
SSRF guard") has a rendered reply able to make this machine fetch a private
address. netpolicy is deliberately left as the single authority on that rather
than second-guessing it here, because overriding a setting the owner explicitly
chose is its own failure mode. If that combination should be refused outright,
refuse it in ONE place - `check_url` - so every caller inherits it, not here.
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
    for a wedged one, so the budget has to sit above safe_fetch_bytes' real
    worst case - and that case is not one timeout. netpolicy applies its timeout
    separately to the connect and to each read, and follows redirects MANUALLY
    (so every hop re-pays both), up to ``_MAX_REDIRECTS``. A literal 40.0 - the
    first version of this - covered a single connect+read pair and would have
    expired part-way down a legitimate three-hop CDN chain, turning a working
    image into a 504.

    Computed from the two constants so the relation cannot break silently when
    either is retuned: the numbers live in netpolicy and the arithmetic here,
    which is exactly the shape that goes wrong quietly.
    ``test_the_fetch_budget_stays_above_netpolicy_s_own_worst_case`` asserts the
    RELATION rather than the number. Falls back to the same arithmetic on
    today's values if either private name ever disappears, rather than to an
    unrelated guess."""
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
            # 403 rather than 404: the endpoint exists and the caller is allowed
            # to ask, the OWNER has simply not turned it on. A 404 here would read
            # as "old server" to a client trying to tell those apart.
            raise HTTPException(
                403, "Showing remote images is off. Set 'Show remote images "
                     "in replies' to 'ask' or 'on' under Settings > Network to "
                     "enable it.")
        if mode == REMOTE_IMAGE_ASK and consent.strip() not in ("1", "true", "yes"):
            # BEFORE any fetch, so a host the reader has not agreed to is never
            # contacted. The client turns this into its own per-origin prompt;
            # the sentence matters anyway, because a client that does not (an
            # older cached shell, a direct API caller) shows it verbatim.
            raise HTTPException(
                428, "Showing remote images is set to 'ask', and this site has "
                     "not been allowed in this conversation. Choose to show "
                     "images from it when localm asks, or set 'Show remote "
                     "images in replies' to 'on' under Settings > Network.")

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
