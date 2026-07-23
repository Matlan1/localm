# SPDX-License-Identifier: AGPL-3.0-or-later
"""One verified HTTPS opener for localm's OUTBOUND clients.

localm reaches out over HTTPS in a few places: `setup-llama` downloads the native
llama.cpp runtime from GitHub releases, `localm update` checks for and downloads a
new build (proxy + release CDN), the issues list reads the proxy, and a bug report
is uploaded to it. They all share :func:`verified_urlopen` so every outbound client
verifies the SAME way and none can silently regress on its own.

Verification order, and why:

1. The platform's NATIVE certificate store first (``ssl.create_default_context()``
   with no override - on Windows this is the ROOT store, on Linux/macOS the
   system trust store). This is the SAME trust a browser already uses, and the
   same store an IT department provisions a corporate/security-product TLS-
   intercepting proxy's root into - so a managed machine behind one of those
   verifies correctly on the very first attempt, with no flag and no retry ever
   visible. This matches uv's own ``--system-certs``/``UV_SYSTEM_CERTS`` (see
   setup.bat / setup.sh), so every outbound client in this project makes the
   same trust choice.

2. certifi's bundled Mozilla root list, ONLY if step 1 fails with a certificate
   verification error specifically. This rescues the one case native-store-first
   does not cover: a freshly-imaged Windows box whose ROOT store has not yet
   cached a legitimate CA chain - Windows' OpenSSL reads only the store's current
   snapshot and does not trigger Windows' on-demand root-certificate auto-update
   (only SChannel / the browser does), so ``CERTIFICATE_VERIFY_FAILED`` ("unable
   to get local issuer certificate") can happen even though a browser fetches the
   same URL fine. certifi carries the full bundle in-process, independent of
   the machine's cert-store state.

Any OTHER failure (HTTP error, timeout, DNS, or a certificate failure that survives
BOTH attempts) propagates unchanged - never silently swallowed (AGENTS.md rule 5).
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from typing import Optional, Sequence

from localm.debuglog import logger


def _open(req, timeout, context, handlers):
    if handlers:
        opener = urllib.request.build_opener(
            *handlers, urllib.request.HTTPSHandler(context=context))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout, context=context)


def verified_urlopen(req, *, timeout: Optional[float] = None,
                      handlers: Sequence[type] = ()):
    """Open *req* verifying TLS as described in the module docstring: the
    platform's native certificate store first, falling back to certifi's
    bundled root list only on a certificate-verification failure specifically.

    *handlers* are extra ``urllib.request`` handler classes to install ahead of
    the HTTPS handler (e.g. a redirect-scheme guard) - when empty, this calls
    plain ``urlopen`` directly. Returns whatever the underlying open call
    returns (a context-manager-compatible response).
    """
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
