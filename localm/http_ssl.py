# SPDX-License-Identifier: AGPL-3.0-or-later
"""One verified SSL context for localm's OUTBOUND HTTPS clients.

localm reaches out over HTTPS in a few places: `setup-llama` downloads the native
llama.cpp runtime from GitHub releases, `localm update` checks for and downloads a
new build (proxy + release CDN), the issues list reads the proxy, and a bug report
is uploaded to it. They all share THIS context so every outbound client verifies
the SAME way and none can silently regress on its own.

Why not urllib's stdlib default context: it verifies against whatever CAs Python
can enumerate from the OS at that moment. On Windows that is the current snapshot
of the ROOT certificate store, and Python's OpenSSL does NOT trigger Windows'
on-demand root-certificate auto-update (only SChannel / the browser does). So on a
box whose ROOT store has not already cached the CA chain for GitHub's release CDN,
the download fails with CERTIFICATE_VERIFY_FAILED ("unable to get local issuer
certificate") even though Edge fetches the same URL fine. That is the reported
setup-llama install failure, and the updater would hit the identical wall on the
same box (it follows a 302 to the same release CDN).

certifi carries the full Mozilla CA bundle in-process, so verification no longer
depends on the machine's cert-store state; it is the same bundle the
huggingface_hub / httpx model downloads already verify against (which is why those
work while the raw-urllib paths did not). certifi is a declared core dependency of
localm (see pyproject.toml), so it is always installed by setup. If it is somehow
absent we fall back to the stdlib default context: still a real verified handshake,
never a downgrade to unverified (a security step that fails must not silently pass).
"""

from __future__ import annotations

import ssl

from localm.debuglog import logger


def client_ssl_context() -> ssl.SSLContext:
    """A verifying SSL context backed by certifi's CA bundle (see module docstring).

    Full verification always (``CERT_REQUIRED`` + hostname check); the only choice
    is WHICH CA set backs it - certifi (machine-independent) when available, else
    the stdlib default. Never returns an unverified context."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception as e:
        logger.debug("certifi CA bundle unavailable (%s); using the stdlib default "
                     "SSL verification", e)
        return ssl.create_default_context()
