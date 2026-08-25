# SPDX-License-Identifier: AGPL-3.0-or-later
"""One verified HTTPS opener for localm's OUTBOUND clients."""

from __future__ import annotations

import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Sequence

from localm.debuglog import logger


class RedirectDowngradeRefused(urllib.error.URLError):
    """An outbound request was redirected off HTTPS and localm refused to follow."""


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves HTTPS for a weaker scheme."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # newurl is already absolute: http_error_302 urljoins it before calling us.
        old = urllib.parse.urlsplit(req.get_full_url()).scheme.lower()
        new = urllib.parse.urlsplit(newurl).scheme.lower()
        if old == "https" and new != "https":
            raise RedirectDowngradeRefused(
                f"refused an HTTPS -> {new or 'scheme-less'} downgrade redirect "
                f"(HTTP {code}, to {newurl!r}) - the response after it would "
                "travel in cleartext")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(req, timeout, context, handlers):
    opener = urllib.request.build_opener(
        *handlers, urllib.request.HTTPSHandler(context=context))
    return opener.open(req, timeout=timeout)


def _with_redirect_guard(handlers: Sequence[type]) -> tuple:
    """*handlers* with :class:`HttpsOnlyRedirect` prepended, unless the caller already supplied a redirect policy of its own."""
    handlers = tuple(handlers)
    # Accepts a class or an instance, mirroring build_opener's own test.
    if any(issubclass(h, urllib.request.HTTPRedirectHandler) if isinstance(h, type)
           else isinstance(h, urllib.request.HTTPRedirectHandler)
           for h in handlers):
        return handlers
    return (HttpsOnlyRedirect,) + handlers


def verified_urlopen(req, *, timeout: Optional[float] = None,
                      handlers: Sequence[type] = ()):
    """Open *req* verifying TLS as described in the module docstring: the platform's native certificate store first, falling back to certifi's bundled root list only on a certificate-verification failure specifically."""
    handlers = _with_redirect_guard(handlers)
    try:
        return _open(req, timeout, ssl.create_default_context(), handlers)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        logger.debug("native certificate store did not verify %s (%s); "
                     "retrying against certifi's bundled root list",
                     getattr(req, "full_url", req), e)
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception as ce:
            logger.debug("certifi CA bundle unavailable (%s); the native-store "
                         "failure above is the real error", ce)
            raise e from ce
        return _open(req, timeout, ctx, handlers)
