# SPDX-License-Identifier: AGPL-3.0-or-later
"""One verified HTTPS opener for localm's OUTBOUND clients.

localm reaches out over HTTPS in a few places: `setup-llama` downloads the native
llama.cpp runtime from GitHub releases, `localm update` checks for and downloads a
new build (proxy + release CDN), the issues list reads the proxy, and a bug report
is uploaded to it. They all share :func:`verified_urlopen`, so every outbound
client verifies the SAME way.

Verification order:

1. The platform's NATIVE certificate store (``ssl.create_default_context()``
   with no override - on Windows this is the ROOT store, on Linux/macOS the
   system trust store). This is the same store an IT department provisions a
   corporate/security-product TLS-intercepting proxy's root into, and it matches
   uv's own ``--system-certs``/``UV_SYSTEM_CERTS`` (see setup.bat / setup.sh).

2. certifi's bundled Mozilla root list, ONLY if step 1 fails with a certificate
   verification error specifically. This covers a freshly-imaged Windows box
   whose ROOT store has not yet cached a legitimate CA chain: Windows' OpenSSL
   reads only the store's current snapshot and does not trigger Windows'
   on-demand root-certificate auto-update (only SChannel / the browser does), so
   ``CERTIFICATE_VERIFY_FAILED`` ("unable to get local issuer certificate") can
   happen even though a browser fetches the same URL fine. certifi carries the
   full bundle in-process, independent of the machine's cert-store state.

Any OTHER failure (HTTP error, timeout, DNS, or a certificate failure that survives
BOTH attempts) propagates unchanged, never swallowed.

:class:`HttpsOnlyRedirect` below is installed by default; see its docstring for the
redirect it refuses.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Sequence

from localm.debuglog import logger


class RedirectDowngradeRefused(urllib.error.URLError):
    """An outbound request was redirected off HTTPS and localm refused to follow.

    A ``URLError`` subclass: every outbound caller in this project already funnels
    ``URLError`` either into its own domain error (interpolating the reason, so the
    refusal is quoted to the user) or into a LOGGED best-effort fallback, so a
    refusal is reported wherever it happens and can never read as a success. Two
    sites catch it explicitly ahead of that funnel - setup_llama._download and
    updater.download - because their generic transport wording would misdescribe it
    as a network fault.
    """


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves HTTPS for a weaker scheme.

    urllib's own redirect handler follows up to 10 hops and, in
    ``http_error_302``, admits any target whose scheme is in
    ``('http', 'https', 'ftp', '')`` - so a plain ``http://`` Location IS
    followed, in cleartext, off a connection the caller verified. Verifying the
    first hop's certificate says nothing about the hops after it. This closes that
    for every :func:`verified_urlopen` caller at once.

    The rule is DOWNGRADE, not https-only: the target scheme is compared to the
    scheme of the request being redirected, so each hop is judged on its own. A
    caller that legitimately started on plain http (a user-configured http
    endpoint) has no confidentiality left to lose and keeps working; a caller on
    https can never be walked off it.
    """

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
    """*handlers* with :class:`HttpsOnlyRedirect` prepended, unless the caller
    already supplied a redirect policy of its own.

    That is the ONLY opt-out: a caller states its policy by passing a handler
    (comfy_client's ``_RefuseRedirect`` refuses every hop outright), and
    build_opener drops the stdlib default whenever a subclass of it is passed.
    No argument, and no empty ``handlers=()``, reaches urllib's permissive
    default.
    """
    handlers = tuple(handlers)
    # Accepts a class or an instance, mirroring build_opener's own test.
    if any(issubclass(h, urllib.request.HTTPRedirectHandler) if isinstance(h, type)
           else isinstance(h, urllib.request.HTTPRedirectHandler)
           for h in handlers):
        return handlers
    return (HttpsOnlyRedirect,) + handlers


def verified_urlopen(req, *, timeout: Optional[float] = None,
                      handlers: Sequence[type] = ()):
    """Open *req* verifying TLS as described in the module docstring: the
    platform's native certificate store first, falling back to certifi's
    bundled root list only on a certificate-verification failure specifically.

    A redirect off HTTPS is refused (:class:`HttpsOnlyRedirect`, raising
    :class:`RedirectDowngradeRefused`) unless *handlers* carries a redirect
    policy of its own. *handlers* are extra ``urllib.request`` handler classes
    installed ahead of the HTTPS handler. Returns whatever the underlying open
    call returns (a context-manager-compatible response).
    """
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
