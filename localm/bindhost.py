# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind-host classification: is a given host string a loopback address."""

from __future__ import annotations

import ipaddress
from typing import Optional


def is_loopback_host(host: str) -> bool:
    """True for a loopback bind/client host (127.0.0.0/8, ::1, localhost)."""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_valid_bind_host(host) -> bool:
    """True when *host* is a string this server can actually bind: ``localhost`` or a literal IPv4 address (including the ``0.0.0.0`` wildcard)."""
    if not host or not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not getattr(addr, "scope_id", None)


def url_host(host: str) -> str:
    """*host* as it must appear inside a URL authority: an IPv6 literal gets square brackets, everything else is returned unchanged."""
    if not host:
        return host
    if host.startswith("["):
        return host
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{host}]" if addr.version == 6 else host


def self_connect_host(bind_host: Optional[str]) -> str:
    """The address THIS machine should dial to reach a server bound on *bind_host*."""
    h = (bind_host or "").strip()
    if not h or h in ("0.0.0.0", "localhost"):
        return "127.0.0.1"
    if h == "::":
        return "::1"
    return h
