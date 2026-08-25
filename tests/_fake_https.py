# SPDX-License-Identifier: AGPL-3.0-or-later
"""The monkeypatch seam for localm's shared outbound-HTTPS opener.

``http_ssl.verified_urlopen`` used to call ``urllib.request.urlopen`` directly
whenever no caller supplied a handler, so a test isolated it by patching
``urlopen``. It cannot any more: a redirect guard has to be installed as a
handler, and the only way to install one is to build an opener. So the seam is
now ``urllib.request.build_opener`` and the SSL context arrives via
``urllib.request.HTTPSHandler`` instead of an ``urlopen`` keyword.

This keeps that a ONE-LINE change per test rather than a rewrite: *responder* is
called with exactly the old signature, ``responder(req, timeout=..., context=...)``,
so an existing ``fake_urlopen`` works unmodified. Swap

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

for

    patch_https_transport(monkeypatch, fake_urlopen)

Patching ``urlopen`` alone is now WORSE THAN NOT PATCHING, which is why this
exists rather than each test hand-rolling it: the fake is simply never
consulted, the code under test dials the real network, and the test still looks
isolated. Two tests in tests/test_cuda_setup.py did exactly that and reported
upstream's live release tag instead of their fixture's.
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
