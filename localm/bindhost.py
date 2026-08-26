# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Bind-host classification: is a given host string a loopback address.

``is_loopback_host()`` lives here so every consumer shares one kernel-level
implementation. ``inference/http_server.py``, ``inference/routes/keys.py``,
``inference/routes/system.py``, ``plugins/deps_task.py`` and
``plugins/gui/web.py`` each re-export it under its original name.

Every consumer here needs the same security property: decide "is this a
local-only server" from the CONFIGURED bind host, never the request peer -
portmux relays every connection through an internal loopback socket, so the
peer always looks like 127.0.0.1 even for a genuinely remote client.
"""

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
    """True when *host* is a string this server can actually bind: ``localhost``
    or a literal IPv4 address (including the ``0.0.0.0`` wildcard). Used by
    BOTH the write-time validation of the ``bind_host`` config key
    (settings_schema) and its read site (cli._resolve_bind_host); a hand-edited
    config.json bypasses write-time validation entirely.

    Hostnames are NOT accepted: binding resolves them through the OS resolver,
    which can change answers (or fail) between restarts, and every consumer of
    the loopback/network split (is_loopback_host above) classifies literals, not
    names. ``localhost`` is the one accepted name. A value carrying a port
    (``0.0.0.0:8642``) is rejected too - the port has its own config key.

    IPv6 literals ARE accepted: the port probe resolves the family instead of
    assuming AF_INET, the listening socket is built family-aware (and dual-stack
    for ``::``), the printed URLs bracket a v6 literal, and mDNS advertises an
    address the bind answers on.

    A ZONE ID (``fe80::1%eth0``) is rejected. The zone index names an interface
    as numbered on ONE machine, and it does not survive the
    ``getaddrinfo(..., AI_PASSIVE)`` this server binds through - on Windows the
    scoped form fails with WinError 10049 while the same address unscoped binds.

    Bindability is NOT decided here: a well-formed address stops being bindable
    when DHCP moves the machine. That is ``cli._bind_preflight_error``'s job at
    the read site, and it runs for every config-driven bind including loopback
    ones (``::ffff:127.0.0.1`` is both genuinely loopback and unbindable on
    Windows)."""
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
    """*host* as it must appear inside a URL authority: an IPv6 literal gets
    square brackets, everything else is returned unchanged.

    ``f"https://{host}:{port}/"`` is correct for ``127.0.0.1`` and for
    ``localm.local`` and wrong for ``::1`` - it produces ``https://::1:8642/``,
    where the parser cannot tell the address's colons from the port separator.
    Every place that builds a host:port string routes through here (RFC 3986
    section 3.2.2).

    Already-bracketed input is returned as-is, so a value that has been through
    here once cannot be double-bracketed by a second caller downstream."""
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
    """The address THIS machine should dial to reach a server bound on
    *bind_host*. Never a wildcard, because a wildcard is not itself a
    connectable address.

    The server calls its own API constantly (the coder agent, RAG
    self-embedding, the chat/media VRAM handover, the activity route, the
    post-update health watchdog, the hang alarm's self-probe). A hardcoded
    ``127.0.0.1`` is right for the IPv4 wildcard and for loopback and WRONG for
    an IPv6 bind: a server bound only on ``::1`` has nothing listening on
    ``127.0.0.1``.

    ``::`` maps to ``::1`` rather than ``127.0.0.1``. localm binds the IPv6
    wildcard dual-stack (see ``netlisten.create_listen_socket``), so
    ``127.0.0.1`` works there while that upgrade succeeds; its fallback is a
    v6-only socket, and ``::1`` is reachable in BOTH cases.

    A specific literal is returned as itself: it is the only address that bind
    is guaranteed to answer on."""
    h = (bind_host or "").strip()
    if not h or h in ("0.0.0.0", "localhost"):
        return "127.0.0.1"
    if h == "::":
        return "::1"
    return h
