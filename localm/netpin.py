# SPDX-License-Identifier: AGPL-3.0-or-later
"""DNS-pinning HTTP transport for the network policy.

Closes the SSRF DNS-rebinding TOCTOU. ``netpolicy.check_url`` resolves a hostname
and validates the resulting IP, but a plain ``requests.get`` re-resolves the
*same* hostname at connect time - so an attacker running authoritative DNS with
TTL 0 can answer a public address for the validation and an internal one
(127.0.0.1, 169.254.169.254, an RFC1918 service) for the actual connection.
Nothing in between is re-checked.

The host is resolved ONCE, every address it returns is validated, and the socket
is pinned to that address so there is no second lookup to poison. The original
hostname is still presented for TLS SNI, certificate matching and the ``Host``
header, so pinning is transparent to normal servers (virtual hosts and HTTPS keep
working).

Mechanism: a ``requests`` transport adapter whose connection pools override
``_new_conn`` to (1) preserve ``server_hostname`` (SNI + cert hostname) from the
real host before (2) repointing urllib3's ``_dns_host`` at the validated IP - the
attribute urllib3 hands to ``socket.create_connection``. The ``Host`` header is
set explicitly by the caller so virtual-host routing is unaffected. No new
dependency; no global ``socket`` monkeypatch, and the pin lives on the
per-request adapter rather than process-wide state, so it is thread-safe.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager


class _PinnedHTTPConnectionPool(HTTPConnectionPool):
    """HTTP pool whose new connections dial a fixed, pre-validated IP."""

    _pinned_ip: str | None = None

    def _new_conn(self):
        conn = super()._new_conn()
        if self._pinned_ip:
            conn._dns_host = self._pinned_ip   # the socket target; .host stays for the Host header via the caller
        return conn


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    """HTTPS pool whose new connections dial a fixed IP while presenting the
    real hostname for SNI + certificate validation."""

    _pinned_ip: str | None = None

    def _new_conn(self):
        conn = super()._new_conn()
        if self._pinned_ip:
            # Capture the real hostname for SNI and cert matching before
            # repointing the socket at the pinned IP.
            if conn.server_hostname is None:
                conn.server_hostname = conn.host
            conn._dns_host = self._pinned_ip
        return conn


class _PinnedPoolManager(PoolManager):
    """PoolManager that stamps a pinned IP onto every pool it creates."""

    def __init__(self, pinned_ip: str, **kwargs):
        self._pinned_ip = pinned_ip
        super().__init__(**kwargs)
        # Instance-local scheme->pool map so we do not mutate urllib3's global.
        self.pool_classes_by_scheme = {
            "http": _PinnedHTTPConnectionPool,
            "https": _PinnedHTTPSConnectionPool,
        }

    def _new_pool(self, scheme, host, port, request_context=None):
        pool = super()._new_pool(scheme, host, port, request_context=request_context)
        pool._pinned_ip = self._pinned_ip
        return pool


class PinnedIPAdapter(HTTPAdapter):
    """A ``requests`` transport adapter that forces every connection to a single
    pre-validated IP address (see the module docstring)."""

    def __init__(self, pinned_ip: str, **kwargs):
        self._pinned_ip = pinned_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        self.poolmanager = _PinnedPoolManager(
            self._pinned_ip,
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


def pinned_session(pinned_ip: str) -> requests.Session:
    """A ``requests.Session`` whose http(s) traffic is pinned to ``pinned_ip``.

    The caller owns the session lifetime (use it as a context manager, or close
    it after a streamed body is fully read) and must send the original hostname
    as the ``Host`` header so virtual-host routing survives the IP pin.

    ``trust_env`` is disabled: with it on (the ``requests`` default), an
    HTTP_PROXY/HTTPS_PROXY environment variable routes the connection through
    that proxy instead of dialling the pinned IP at all - a plain-HTTP request
    through a proxy is forwarded via a *different* connection pool
    (``requests.adapters.HTTPAdapter.proxy_manager_for``, unaware of
    ``_pinned_ip``), so the pin above is bypassed entirely rather than merely
    weakened. The same flag also stops ``requests`` from auto-attaching
    ``.netrc`` credentials to a request the caller never asked to authenticate.
    """
    session = requests.Session()
    session.trust_env = False
    adapter = PinnedIPAdapter(pinned_ip)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
