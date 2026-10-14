# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numeric / short-form IPv4 SSRF bypass.

'2130706433', '0x7f000001', '0177.0.0.1' and '127.1' all resolve to the
loopback address 127.0.0.1 while evading the ipaddress-based public-address
check:

  * ipaddress.ip_address() refuses these dotless / hex / octal / short forms,
    so the getaddrinfo result loop never classifies them, and
  * socket.getaddrinfo may raise (or be patched to raise) for them, and the
    guard's documented behavior is "unresolvable hosts pass".

The guard normalizes such hosts with socket.inet_aton into canonical dotted
form and classifies THAT with the ipaddress module before the public-address
check. These tests pin that each adversarial form is refused when
net_allow_private is False, while a genuine public hostname still passes.
"""

import socket

import pytest

from localm.netpolicy import NetworkPolicyError, check_url


def _with_config(monkeypatch, cfg: dict):
    monkeypatch.setattr("localm.config.load_config", lambda: cfg)


def _no_dns(monkeypatch):
    """Make getaddrinfo unavailable so the test exercises the literal-IP
    normalization path, not host resolution. A raising getaddrinfo is treated
    as 'unresolvable -> pass', which is the adversarial condition."""
    def boom(host, port, *a, **k):
        raise socket.gaierror("forced: numeric host not resolved")
    monkeypatch.setattr("socket.getaddrinfo", boom)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)


# Each of these is a different obfuscation of 127.0.0.1.
_LOOPBACK_FORMS = [
    "2130706433",   # dotless decimal
    "0x7f000001",   # dotless hex
    "0177.0.0.1",   # octal first octet
    "127.1",        # short form (a.d)
]


@pytest.mark.parametrize("host", _LOOPBACK_FORMS)
def test_numeric_loopback_forms_refused(monkeypatch, host):
    """Each numeric/short IPv4 that maps to loopback is refused when private
    access is off, even when DNS would not resolve it."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    _no_dns(monkeypatch)
    with pytest.raises(NetworkPolicyError, match="non-public"):
        check_url(f"http://{host}/v1/models")


@pytest.mark.parametrize("host", _LOOPBACK_FORMS)
def test_numeric_loopback_message_names_canonical_ip(monkeypatch, host):
    """The refusal classifies the canonical 127.0.0.1 rather than echoing the
    obfuscated literal."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    _no_dns(monkeypatch)
    with pytest.raises(NetworkPolicyError, match="127.0.0.1"):
        check_url(f"http://{host}/")


def test_numeric_private_class_a_refused(monkeypatch):
    """Dotless decimal for 10.0.0.1 (a private RFC1918 address) is refused."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    _no_dns(monkeypatch)
    # 10.0.0.1 == 167772161
    with pytest.raises(NetworkPolicyError, match="non-public"):
        check_url("http://167772161/")


def test_numeric_loopback_allowed_when_private_enabled(monkeypatch):
    """With net_allow_private True the same obfuscated loopback must pass."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": True})
    _no_dns(monkeypatch)
    check_url("http://2130706433/v1/models")   # no raise


def test_normal_public_host_still_passes(monkeypatch):
    """A genuine public hostname is unaffected: inet_aton rejects it, so the
    code falls through to ordinary DNS-based classification."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    check_url("https://example.com/")   # no raise


def test_public_numeric_ip_passes(monkeypatch):
    """A numeric form that maps to a public address must NOT be blocked by the
    new normalization (no false positives)."""
    _with_config(monkeypatch, {"net_mode": "allow", "net_allow_private": False})
    _no_dns(monkeypatch)
    # 93.184.216.34 in dotless decimal.
    dotless = str(int.from_bytes(socket.inet_aton("93.184.216.34"), "big"))
    check_url(f"https://{dotless}/")   # no raise


# --------------------------------------------------------------------------- #
# is_link_local_host: the SAME numeric-literal-bypass discipline as above,
# for a caller (comfy_client's ComfyUI-address guard) that must allow
# loopback/private/public and refuse ONLY link-local/cloud-metadata - unlike
# check_url, which refuses every non-public class at once.
# --------------------------------------------------------------------------- #

from localm.netpolicy import is_link_local_host  # noqa: E402

# Each of these is a different obfuscation of 169.254.169.254 (cloud metadata).
_LINK_LOCAL_FORMS = [
    "0xa9fea9fe",              # dotless hex
    "2852039166",              # dotless decimal
    "0251.0376.0251.0376",     # octal, every octet
    "169.254.169.254",         # dotted (control: the un-obfuscated form)
]


@pytest.mark.parametrize("host", _LINK_LOCAL_FORMS)
def test_is_link_local_host_catches_every_numeric_spelling(monkeypatch, host):
    """ipaddress.ip_address() refuses these dotless/hex/octal forms outright,
    so without the _literal_ipv4 fast path is_link_local_host would classify
    them via getaddrinfo - which _no_dns makes unavailable, so a pass here
    proves the LITERAL normalization caught it, not a real DNS lookup."""
    _no_dns(monkeypatch)
    assert is_link_local_host(host) is True


def test_is_link_local_host_does_not_flag_the_numeric_loopback_literal(monkeypatch):
    """2130706433 is loopback (127.0.0.1), not link-local - is_link_local_host
    must stay narrow and not widen into flagging every non-public address."""
    _no_dns(monkeypatch)
    assert is_link_local_host("2130706433") is False


def test_is_link_local_host_unresolvable_host_passes(monkeypatch):
    _no_dns(monkeypatch)
    assert is_link_local_host("nonexistent.invalid") is False


def test_is_link_local_host_via_dns_resolution(monkeypatch):
    """A HOSTNAME (not a literal) that resolves to a link-local address must
    also be caught by the getaddrinfo loop, not only the literal fast path."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("169.254.1.1", 0))])
    assert is_link_local_host("metadata.internal") is True
