# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pure-ASGI transport middleware, added after every BaseHTTPMiddleware
handler so it wraps them all: the request-body cap, the disconnect signal and,
outermost, the hang alarm's in-flight bookkeeping."""

from __future__ import annotations

from fastapi import FastAPI

import localm.inference.http_server as _hs


def add_transport_middleware(app: FastAPI) -> None:
    """Add the three, innermost first."""
    # Outside every BaseHTTPMiddleware handler added before it (none of which touch the
    # body), so it sees the raw ASGI receive() for its body-size accounting. The
    # _DisconnectSignalMiddleware added right after passes receive() through
    # untouched, so this still gets the raw stream.
    app.add_middleware(_hs._BodyStreamCapMiddleware)
    # Added LAST (== outermost) so its disconnect poll is bound to the raw receive,
    # OUTSIDE the BaseHTTPMiddleware handlers that otherwise mask http.disconnect
    # from the non-streaming inference path (see the class + _generate_full).
    app.add_middleware(_hs._DisconnectSignalMiddleware)
    # Outermost of all: in-flight/progress bookkeeping for the hang alarm's
    # starvation detector. Pure ASGI and content-free (counts and
    # clocks only); sits outside everything so a request wedged in ANY inner
    # layer still shows as in flight.
    from localm.inference._hang_alarm import RequestProgressMiddleware
    app.add_middleware(RequestProgressMiddleware)
