# SPDX-License-Identifier: AGPL-3.0-or-later
"""The monkeypatch seam for localm's shared outbound-HTTPS opener.

``http_ssl.verified_urlopen`` installs its redirect guard as a handler, which
needs an opener, so the seam is ``urllib.request.build_opener`` and the SSL
context arrives via ``urllib.request.HTTPSHandler`` rather than an ``urlopen``
keyword.

*responder* is called as ``responder(req, timeout=..., context=...)``, so an
existing ``fake_urlopen`` works unmodified. Swap

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

for

    patch_https_transport(monkeypatch, fake_urlopen)

Patching ``urlopen`` alone is WORSE THAN NOT PATCHING: the fake is never
consulted, the code under test dials the real network, and the test still looks
isolated.
"""

from __future__ import annotations

import urllib.request


def patch_https_transport(monkeypatch, responder):
    """Route ``http_ssl._open``'s opener through *responder*.

    Returns a dict that keeps recording as calls arrive:

    ``context``   the SSLContext of the most recent ``HTTPSHandler``;
    ``handlers``  the handler classes/instances passed to the most recent
                  ``build_opener`` - assert the redirect guard is among them.
    """
    seen = {"context": None, "handlers": ()}

    def capture_https_handler(*_a, context=None, **_k):
        seen["context"] = context
        # Stand-in handler; build_opener is faked below and never installs it.
        return object()

    class _Opener:
        def __init__(self, context):
            self._context = context

        def open(self, req, timeout=None):
            return responder(req, timeout=timeout, context=self._context)

    def fake_build_opener(*handlers):
        seen["handlers"] = handlers
        return _Opener(seen["context"])

    monkeypatch.setattr(urllib.request, "HTTPSHandler", capture_https_handler)
    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    return seen
