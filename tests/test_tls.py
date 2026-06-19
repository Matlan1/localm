# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for built-in TLS (NET-1): localm/tls.ensure_cert and SAN discovery.

Covers the happy path (valid CA + leaf, required SANs present), reuse stability
(no needless regeneration), and the regeneration triggers the spec calls out:
missing / expired / wrong-SAN leaf, and an IP-set change - with the CA reused
across regenerations so a device trusted once stays trusted.
"""

from datetime import timedelta

import pytest

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

from localm import tls


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _load(path):
    return x509.load_pem_x509_certificate(open(path, "rb").read())


def _sans(cert):
    return tls._cert_sans(cert)


# ------------------------------------------------------------------ #
#  san_targets                                                        #
# ------------------------------------------------------------------ #

def test_san_targets_always_includes_loopback():
    hosts, ips = tls.san_targets()
    assert "localhost" in hosts
    assert "127.0.0.1" in ips
    assert "::1" in ips


def test_san_targets_routes_extras_and_drops_wildcard():
    hosts, ips = tls.san_targets(
        extra_hostnames=["phone.local", "10.0.0.5", "0.0.0.0"],
        extra_ips=["192.168.1.50"],
    )
    assert "phone.local" in hosts          # a real hostname stays a DNS SAN
    assert "10.0.0.5" in ips               # a bare IP passed as hostname -> IP SAN
    assert "192.168.1.50" in ips
    assert "0.0.0.0" not in ips and "0.0.0.0" not in hosts   # all-interfaces, dropped


def test_is_tailscale_ip():
    assert tls.is_tailscale_ip("100.101.102.103")
    assert not tls.is_tailscale_ip("192.168.1.1")
    assert not tls.is_tailscale_ip("not-an-ip")


# ------------------------------------------------------------------ #
#  ensure_cert - happy path                                          #
# ------------------------------------------------------------------ #

def test_ensure_cert_creates_ca_and_leaf(tmp_path):
    cert_path, key_path = tls.ensure_cert(tmp_path, ips=["192.168.1.50"])

    assert cert_path == str(tls._leaf_cert_path(tmp_path))
    assert key_path == str(tls._leaf_key_path(tmp_path))
    assert tls.ca_cert_path(tmp_path).exists()
    assert tls._ca_key_path(tmp_path).exists()

    leaf = _load(cert_path)
    ca = _load(tls.ca_cert_path(tmp_path))

    # The leaf is a non-CA server cert; the CA is a CA.
    leaf_bc = leaf.extensions.get_extension_for_class(x509.BasicConstraints).value
    ca_bc = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert leaf_bc.ca is False
    assert ca_bc.ca is True

    # serverAuth EKU + the required SANs.
    eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku
    _, ips = _sans(leaf)
    assert {"127.0.0.1", "::1", "192.168.1.50"}.issubset(ips)


def test_leaf_is_signed_by_ca(tmp_path):
    cert_path, _ = tls.ensure_cert(tmp_path)
    leaf = _load(cert_path)
    ca = _load(tls.ca_cert_path(tmp_path))

    assert leaf.issuer == ca.subject
    # Verifying the signature proves the chain (raises on mismatch).
    ca.public_key().verify(
        leaf.signature,
        leaf.tbs_certificate_bytes,
        padding.PKCS1v15(),
        leaf.signature_hash_algorithm,
    )


def test_leaf_validity_under_apple_825_day_cap(tmp_path):
    cert_path, _ = tls.ensure_cert(tmp_path)
    leaf = _load(cert_path)
    span = leaf.not_valid_after_utc - leaf.not_valid_before_utc
    assert span <= timedelta(days=825), "iOS/Safari reject server certs > 825 days"


def test_leaf_private_key_matches_cert(tmp_path):
    cert_path, key_path = tls.ensure_cert(tmp_path)
    leaf = _load(cert_path)
    key = serialization.load_pem_private_key(open(key_path, "rb").read(), password=None)
    assert (key.public_key().public_numbers()
            == leaf.public_key().public_numbers())


# ------------------------------------------------------------------ #
#  Reuse vs regeneration                                             #
# ------------------------------------------------------------------ #

def test_reuse_is_stable(tmp_path):
    cert_path, _ = tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    first = open(cert_path, "rb").read()
    # Same inputs -> no regeneration (identical bytes, same serial).
    tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    second = open(cert_path, "rb").read()
    assert first == second


def test_ip_change_regenerates_leaf_but_reuses_ca(tmp_path):
    tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    ca_before = tls.ca_cert_path(tmp_path).read_bytes()
    leaf_before = _load(tls._leaf_cert_path(tmp_path))

    # A different bind IP set is not covered by the old leaf -> regenerate.
    tls.ensure_cert(tmp_path, ips=["10.9.9.9"])
    ca_after = tls.ca_cert_path(tmp_path).read_bytes()
    leaf_after = _load(tls._leaf_cert_path(tmp_path))

    assert ca_before == ca_after, "CA must be reused so trusted devices stay trusted"
    assert leaf_before.serial_number != leaf_after.serial_number, "leaf regenerated"
    _, ips = _sans(leaf_after)
    assert "10.9.9.9" in ips


def test_missing_leaf_regenerates(tmp_path):
    tls.ensure_cert(tmp_path)
    tls._leaf_cert_path(tmp_path).unlink()
    # Regeneration recreates the leaf, reusing the existing CA.
    ca_before = tls.ca_cert_path(tmp_path).read_bytes()
    cert_path, _ = tls.ensure_cert(tmp_path)
    assert tls._leaf_cert_path(tmp_path).exists()
    assert tls.ca_cert_path(tmp_path).read_bytes() == ca_before


def test_expired_leaf_regenerates(tmp_path, monkeypatch):
    tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    old = _load(tls._leaf_cert_path(tmp_path))

    # Jump the clock past the leaf's renewal margin -> not reusable -> regen.
    real_now = tls._now()
    far_future = real_now + timedelta(days=tls._LEAF_DAYS + 5)
    monkeypatch.setattr(tls, "_now", lambda: far_future)

    tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    new = _load(tls._leaf_cert_path(tmp_path))
    assert new.serial_number != old.serial_number


def test_wrong_san_leaf_regenerates(tmp_path):
    # First leaf does NOT cover 172.16.0.9.
    tls.ensure_cert(tmp_path, ips=["192.168.1.50"])
    before = _load(tls._leaf_cert_path(tmp_path)).serial_number
    # Asking for an IP the stored leaf lacks forces a regenerate.
    tls.ensure_cert(tmp_path, ips=["172.16.0.9"])
    after = _load(tls._leaf_cert_path(tmp_path))
    assert after.serial_number != before
    _, ips = _sans(after)
    assert "172.16.0.9" in ips


def test_corrupt_ca_is_replaced(tmp_path):
    tls.ensure_cert(tmp_path)
    tls.ca_cert_path(tmp_path).write_text("not a certificate")
    # A corrupt CA must not wedge startup: a fresh CA + leaf are minted.
    cert_path, _ = tls.ensure_cert(tmp_path)
    leaf = _load(cert_path)
    ca = _load(tls.ca_cert_path(tmp_path))
    assert leaf.issuer == ca.subject


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="POSIX file modes only",
)
def test_key_files_are_owner_only_on_posix(tmp_path):
    import stat
    _, key_path = tls.ensure_cert(tmp_path)
    mode = stat.S_IMODE(__import__("os").stat(key_path).st_mode)
    assert mode == 0o600
