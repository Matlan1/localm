# SPDX-License-Identifier: AGPL-3.0-or-later
"""DNS-pinning HTTP transport for the network policy."""

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
    """HTTPS pool whose new connections dial a fixed IP while presenting the real hostname for SNI + certificate validation."""

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
    """A ``requests`` transport adapter that forces every connection to a single pre-validated IP address (see the module docstring)."""

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
    """A ``requests.Session`` whose http(s) traffic is pinned to ``pinned_ip``."""
    session = requests.Session()
    session.trust_env = False
    adapter = PinnedIPAdapter(pinned_ip)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
