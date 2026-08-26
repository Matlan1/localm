# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial completeness audit for localm.bindhost.is_loopback_host.

is_loopback_host() is the ENTIRE boundary between localm's open-mode
loopback-owner shortcuts (effective_fs_access() == "host", the full
filesystem reach, no credential consulted) and the fail-closed refusal in
localm/cli/_core.py::_exposed_bind_warning, which is an allowlist:
``if is_loopback_host(host): return None`` - everything else falls through to
the unsafe branch (warn, then sys.exit(2) unless --insecure). A FALSE SAFE
result here (True for a routable address) would silently disable that
refusal and expose an unauthenticated, host-privileged API to the network.
A false UNSAFE is only an availability annoyance (the CLI demands
--insecure/an API key for a bind that was actually loopback-only).

Where the platforms disagree (e.g. bracket-wrapped IPv6 literals, or an empty
host string), the comments say so; the assertions below pin only
platform-INDEPENDENT behavior. is_loopback_host's own return value never
changes across platforms, since it makes no OS calls itself - only the live
cross-check test further below depends on what the OS resolves.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

from localm.bindhost import is_loopback_host


# --------------------------------------------------------------------------
# Fires-control: inputs that must classify UNSAFE.
# --------------------------------------------------------------------------
def test_fires_control_public_addresses_are_never_loopback():
    assert is_loopback_host("8.8.8.8") is False
    assert is_loopback_host("203.0.113.7") is False  # TEST-NET-3, RFC 5737


# --------------------------------------------------------------------------
# The adversarial table. True means the host must classify as loopback,
# False means it must never classify as loopback.
# --------------------------------------------------------------------------
ADVERSARIAL_CASES = [
    # --- wildcard binds: MUST be unsafe, never loopback ---
    ("0.0.0.0", False),   # IPv4 wildcard - binds every interface
    ("::", False),        # IPv6 wildcard
    ("[::]", False),      # bracketed wildcard literal (ip_address() rejects brackets -> ValueError)

    # --- genuine loopback, canonical form ---
    ("127.0.0.1", True),
    ("::1", True),
    ("localhost", True),  # hardcoded exact-match branch, checked before ipaddress is even consulted

    # --- the whole 127.0.0.0/8 block is loopback under real OS routing
    # semantics (RFC 1122 3.2.1.3), not just 127.0.0.1 ---
    ("127.0.0.2", True),
    ("127.255.255.254", True),
    ("127.42.42.42", True),

    # --- integer/alternate encodings of 127.0.0.1: ipaddress.ip_address()
    # requires a strict canonical dotted-quad and rejects every one of these
    # (ValueError -> False) ---
    ("127.1", False),
    ("127.000.000.001", False),
    ("2130706433", False),
    ("0x7f000001", False),
    ("0177.0.0.1", False),

    # --- localhost: exact case-sensitive, no-trailing-dot match only. Any
    # variant falls through to ipaddress (ValueError) -> False ---
    ("LOCALHOST", False),
    ("LocalHost", False),
    ("localhost.", False),

    # --- hostname resolution is not performed by this predicate: only literal
    # IPs and the literal string "localhost" classify as loopback ---
    ("router.local", False),
    ("some-vpn-hostname.example", False),

    # --- IPv4-mapped IPv6: only the exact ::ffff:0:0/96 prefix unwraps to the
    # embedded IPv4 address. NAT64 (64:ff9b::/96), 6to4 (2002::/16) and the
    # deprecated all-zero form fall through to the bare ::1 check ---
    ("::ffff:127.0.0.1", True),
    ("::ffff:192.168.1.5", False),
    ("::127.0.0.1", False),          # deprecated IPv4-compatible form, NOT ipv4_mapped
    ("64:ff9b::127.0.0.1", False),   # NAT64 well-known prefix embedding 127.0.0.1
    ("64:ff9b::7f00:1", False),      # same NAT64 address, hex form
    ("2002:7f00:0001::", False),     # 6to4 embedding 127.0.0.1

    # --- brackets: ipaddress.ip_address() does not accept bracket-wrapped
    # literals (ValueError -> False) ---
    ("[::1]", False),

    # --- empty / blank / None: the `if not host: return False` guard at the
    # top of is_loopback_host classifies these UNSAFE ---
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
# Cross-check against a real getaddrinfo resolution: whenever
# is_loopback_host(host) is True, every address the OS resolves that host to
# must itself be a loopback address. The reverse direction is not asserted.
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
# is_loopback_host("") stays False regardless of platform or box network
# config.
# --------------------------------------------------------------------------
def test_empty_host_is_unsafe():
    assert is_loopback_host("") is False
