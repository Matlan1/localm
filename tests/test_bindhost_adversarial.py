# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial completeness audit for localm.bindhost.is_loopback_host."""

from __future__ import annotations

import ipaddress
import socket

import pytest

from localm.bindhost import is_loopback_host


# --------------------------------------------------------------------------
# Fires-control: at least one input that MUST be classified UNSAFE, so a
# predicate stubbed to always-return-True (or otherwise broken toward
# "safe") cannot pass this file undetected.
# --------------------------------------------------------------------------
def test_fires_control_public_addresses_are_never_loopback():
    assert is_loopback_host("8.8.8.8") is False
    assert is_loopback_host("203.0.113.7") is False  # TEST-NET-3, RFC 5737


# --------------------------------------------------------------------------
# The adversarial table. Cases marked True are the ones the confinement
# argument depends on: if any of these flip to False, is_loopback_host()
# itself hasn't changed and something regressed. Cases marked False are the
# security-critical floor - none of these may ever become True.
# --------------------------------------------------------------------------
ADVERSARIAL_CASES = [
    # --- wildcard binds: MUST be unsafe, never loopback ---
    ("0.0.0.0", False),   # IPv4 wildcard - binds every interface (verified: real bind succeeds and IS non-loopback)
    ("::", False),        # IPv6 wildcard
    ("[::]", False),      # bracketed wildcard literal (ip_address() rejects brackets -> ValueError -> False anyway)

    # --- genuine loopback, canonical form ---
    ("127.0.0.1", True),
    ("::1", True),
    ("localhost", True),  # hardcoded exact-match branch, checked before ipaddress is even consulted

    # --- the whole 127.0.0.0/8 block is loopback under real OS routing
    # semantics (RFC 1122 3.2.1.3), not just 127.0.0.1. Verified: all three
    # actually bind on both Windows and Linux. ---
    ("127.0.0.2", True),
    ("127.255.255.254", True),
    ("127.42.42.42", True),

    # --- integer/alternate encodings of 127.0.0.1: ipaddress.ip_address()
    # requires strict canonical dotted-quad (4 decimal octets, no leading
    # zeros - CPython rejects those outright since the CVE-2021-29921 fix,
    # confirmed on this box's Python 3.13.14) and rejects every one of
    # these (ValueError -> False). VERIFIED empirically: on Linux, glibc's
    # resolver DOES accept "127.1" / a bare decimal / hex / octal and
    # binds them straight to real 127.0.0.1 - so this is a false-UNSAFE /
    # portability gap (the CLI would demand --insecure for a bind that is
    # actually loopback-only on Linux), never a false SAFE. Not fixed here:
    # replicating inet_aton-style shorthand parsing is its own can of
    # worms and the failure direction is already the safe one. ---
    ("127.1", False),
    ("127.000.000.001", False),
    ("2130706433", False),
    ("0x7f000001", False),
    ("0177.0.0.1", False),

    # --- localhost: exact case-sensitive, no-trailing-dot match only. Any
    # variant falls through to ipaddress (ValueError) -> False. Verified
    # both variants resolve to real loopback via the OS resolver on both
    # platforms, so this is a false UNSAFE (the CLI over-warns), never a
    # false SAFE. ---
    ("LOCALHOST", False),
    ("LocalHost", False),
    ("localhost.", False),

    # --- hostname resolution is deliberately NOT performed by this
    # predicate. A name that resolves to loopback today can resolve
    # elsewhere tomorrow (DNS rebinding, a hosts-file edit, a VPN
    # reconnect) - is_loopback_host only trusts literal IPs and the
    # literal string "localhost", by design. Any other hostname - whether
    # or not it happens to resolve to loopback on this box - is UNSAFE
    # from this predicate's point of view. This is the deliberate,
    # correct choice: a DNS-dependent security predicate is its own
    # problem, and this one avoids it entirely rather than reintroducing
    # it. ---
    ("router.local", False),
    ("some-vpn-hostname.example", False),

    # --- IPv4-mapped IPv6: CPython's IPv6Address.ipv4_mapped property is
    # narrow (only the exact ::ffff:0:0/96 prefix unwraps to the embedded
    # IPv4 address; see ipaddress.py source, read directly this session).
    # NAT64 (64:ff9b::/96), 6to4 (2002::/16), and the deprecated all-zero
    # "IPv4-compatible" form do NOT match that narrow check, so they fall
    # through to `self._ip == 1` (bare ::1) and correctly stay False even
    # though each embeds 127.0.0.1's low 32 bits. This is exactly the
    # embedded-address-confusion bug class a naive "extract the trailing
    # dotted-quad and check that" implementation would be vulnerable to -
    # confirmed this implementation is not. ---
    ("::ffff:127.0.0.1", True),
    ("::ffff:192.168.1.5", False),
    ("::127.0.0.1", False),          # deprecated IPv4-compatible form, NOT ipv4_mapped
    ("64:ff9b::127.0.0.1", False),   # NAT64 well-known prefix embedding 127.0.0.1
    ("64:ff9b::7f00:1", False),      # same NAT64 address, hex form
    ("2002:7f00:0001::", False),     # 6to4 embedding 127.0.0.1

    # --- brackets: ipaddress.ip_address() does not accept bracket-wrapped
    # literals at all (ValueError -> False). False UNSAFE, not exploitable
    # (on Windows the bracketed form does resolve/bind to real ::1 via
    # getaddrinfo; on Linux it does not resolve at all - either way this
    # predicate already refuses it, which is the safe direction). ---
    ("[::1]", False),

    # --- empty / blank / None: the actual danger case. An empty host
    # string was verified (Windows) to resolve via getaddrinfo to EVERY
    # real network interface address on the box - LAN and Tailscale
    # addresses included - the exact opposite of loopback. `if not host:
    # return False` at the top of is_loopback_host is what keeps this
    # safe; it must never regress to a truthy/loopback-shaped default. ---
    ("", False),
    (" ", False),
    ("\t", False),
    ("\n", False),
    (None, False),

    # --- plain LAN / public addresses ---
    ("192.168.1.5", False),
    ("10.0.0.7", False),
    ("8.8.8.8", False),

    # --- malformed / confusable strings: all correctly fail closed ---
    ("127.0.0.1:8080", False),   # host:port confusion
    (" 127.0.0.1", False),       # leading whitespace
    ("127.0.0.1 ", False),       # trailing whitespace
    ("0.0.0.0.0", False),        # 5 octets
]


@pytest.mark.parametrize("host,expected", ADVERSARIAL_CASES)
def test_is_loopback_host_adversarial(host, expected):
    assert is_loopback_host(host) is expected


# --------------------------------------------------------------------------
# Cross-check against reality: the predicate only matters if it agrees with
# what a real bind (getaddrinfo + socket.bind - the same mechanism uvicorn's
# asyncio server ultimately uses to interpret a --host string) actually does
# with the same string. The security-critical direction is one-way: whenever
# is_loopback_host(host) is True, EVERY address the OS resolves that host to
# must itself be a loopback address - never anything else. The reverse
# (predicate False, real resolution is loopback - e.g. "127.1" on Linux,
# "LOCALHOST") is the accepted false-UNSAFE/portability gap documented above,
# not a security bug, so it is not asserted here.
# --------------------------------------------------------------------------
LOOPBACK_LABELED_HOSTS = [h for h, expected in ADVERSARIAL_CASES if expected is True]


@pytest.mark.parametrize("host", LOOPBACK_LABELED_HOSTS)
def test_loopback_labeled_hosts_never_resolve_off_box(host):
    try:
        infos = socket.getaddrinfo(host, 0, proto=socket.IPPROTO_TCP)
    except OSError as e:
        pytest.skip(f"{host!r} did not resolve on this box ({e}) - nothing to cross-check")
    assert infos, f"{host!r} resolved to zero addresses - nothing to cross-check"
    for family, socktype, proto, canonname, sockaddr in infos:
        resolved_ip = sockaddr[0]
        assert ipaddress.ip_address(resolved_ip).is_loopback, (
            f"is_loopback_host({host!r}) is True, but the OS resolves it to "
            f"{resolved_ip!r} which is NOT a loopback address - this is the "
            f"false-SAFE / network-exposure bug this predicate exists to prevent"
        )


# --------------------------------------------------------------------------
# Regression guard for the empty-string danger case: is_loopback_host("")
# must stay False regardless of platform or box network config. (Verified on
# this Windows box: an empty host resolves via getaddrinfo to EVERY real
# network interface address, LAN and Tailscale included - the demonstration
# that motivates this guard. That resolution result is itself a fact about
# THIS box's network interfaces, not about the code under test, so it is not
# asserted here - a CI container with only a loopback interface would make
# such an assertion spuriously fail without the code having changed at all.
# The only code-behavior guarantee is the assertion below.) ---
# --------------------------------------------------------------------------
def test_empty_host_is_unsafe():
    assert is_loopback_host("") is False
