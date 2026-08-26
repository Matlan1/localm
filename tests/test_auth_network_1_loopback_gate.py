# SPDX-License-Identifier: AGPL-3.0-or-later
"""cli/_core.py's _exposed_bind_warning() and _resolve_tls() gate two
security-relevant decisions - the unauthenticated-bind warning, and whether to
skip TLS - on bindhost.is_loopback_host(), which is ipaddress-based and covers
the whole 127.0.0.0/8 range plus ::1 rather than a literal
{"127.0.0.1", "localhost", "::1"} set.

So a bind host like "127.0.0.2" counts as loopback: no network-exposure
warning, and no TLS certificate minted. A real network bind still gets both.
"""

import os

from localm.bindhost import is_loopback_host
from localm.cli._core import _exposed_bind_warning, _resolve_tls


def test_bindhost_confirms_non_canonical_loopback():
    # bindhost.is_loopback_host() classifies 127.0.0.2 as loopback.
    assert is_loopback_host("127.0.0.2") is True
    assert is_loopback_host("127.1") is False  # not a form ipaddress.ip_address parses


def test_non_canonical_loopback_bind_is_not_warned_about(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    assert _exposed_bind_warning("127.0.0.2") is None
    assert _exposed_bind_warning("127.5.5.5") is None


def test_canonical_loopback_forms_still_silent(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    assert _exposed_bind_warning("127.0.0.1") is None
    assert _exposed_bind_warning("localhost") is None
    assert _exposed_bind_warning("::1") is None


def test_real_network_bind_still_warns(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    assert _exposed_bind_warning("0.0.0.0") is not None
    assert _exposed_bind_warning("192.168.1.4") is not None


def test_non_canonical_loopback_bind_stays_plain_http():
    # A loopback bind resolves to (None, None): plain HTTP, no TLS cert minted.
    assert _resolve_tls("127.0.0.2", no_tls=False, tls_cert=None, tls_key=None) == (None, None)


def test_real_network_bind_still_gets_tls():
    cert, key = _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None, tls_key=None)
    assert cert and key
    assert os.path.isfile(cert) and os.path.isfile(key)
