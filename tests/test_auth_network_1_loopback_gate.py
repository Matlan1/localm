# SPDX-License-Identifier: AGPL-3.0-or-later
"""AUTH-NETWORK-1 (security): cli/_core.py's _exposed_bind_warning() and
_resolve_tls() gated two security-relevant decisions (the unauthenticated-bind
warning, and whether to skip TLS) on a literal {"127.0.0.1", "localhost", "::1"}
set instead of the already-hoisted bindhost.is_loopback_host() - which exists
specifically because this exact check was independently copy-pasted five times
before. A bind host like "127.0.0.2" is loopback per ipaddress.is_loopback
(the whole 127.0.0.0/8 range) but is NOT in the literal set, so the two gates
misclassified it as network-exposed: an unauthenticated 127.0.0.2 bind wrongly
triggered the "anyone on the network can use this" warning, and a plain-HTTP
127.0.0.2 bind wrongly minted a TLS certificate instead of staying loopback
plain-HTTP.

Regression: both gates must use bindhost.is_loopback_host() (ipaddress-based,
covers the whole 127.0.0.0/8 range and ::1), not the narrower literal set.
"""

import os

from localm.bindhost import is_loopback_host
from localm.cli._core import _exposed_bind_warning, _resolve_tls


def test_bindhost_confirms_non_canonical_loopback():
    # Sanity: bindhost.is_loopback_host() (the canonical predicate) already
    # correctly classifies 127.0.0.2 as loopback - the bug is that cli/_core.py
    # did not use it.
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
    # A loopback bind must resolve to (None, None) - plain HTTP, no TLS cert
    # minted - the same as the canonical "127.0.0.1" form.
    assert _resolve_tls("127.0.0.2", no_tls=False, tls_cert=None, tls_key=None) == (None, None)


def test_real_network_bind_still_gets_tls():
    cert, key = _resolve_tls("0.0.0.0", no_tls=False, tls_cert=None, tls_key=None)
    assert cert and key
    assert os.path.isfile(cert) and os.path.isfile(key)
