# SPDX-License-Identifier: AGPL-3.0-or-later
"""The monkeypatch seam for localm's shared outbound-HTTPS opener."""

from __future__ import annotations

import urllib.request


def patch_https_transport(monkeypatch, responder):
    """Route ``http_ssl._open``'s opener through *responder*."""
    seen = {"context": None, "handlers": ()}

    def capture_https_handler(*_a, context=None, **_k):
        seen["context"] = context
        # build_opener is faked below, so this stand-in is never installed
        # anywhere - it only has to be a distinct object.
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
