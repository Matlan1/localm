# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Bind-host classification: is a given host string a loopback address.

``is_loopback_host()`` was independently copy-pasted in five places
(``inference/http_server.py``, ``inference/routes/keys.py``,
``inference/routes/system.py`` (as a non-identical inline variant),
``plugins/deps_task.py``, ``plugins/gui/web.py``) - one of them explicitly
because "this core route does not import the gui package". Hoisted here, the
same way ``textguard.py`` was hoisted out of the coder plugin, so every
consumer shares one kernel-level implementation instead of five copies that
can silently drift (as the ``routes/system.py`` inline variant already had).
Each former definition site re-exports this function under its original name
for back-compat, so existing imports and tests are unchanged.

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
    (settings_schema) and its read site (cli._resolve_bind_host) - the
    read-time re-check matters because a hand-edited config.json bypasses
    write-time validation entirely, and an unbindable value reaching the
    server would kill it at startup, which for a config-driven bind may be a
    user with no terminal to recover from (the GUI is how they would fix it).

    Hostnames are deliberately NOT accepted: binding resolves them through the
    OS resolver, which can change answers (or fail) between restarts, and every
    consumer of the loopback/network split (is_loopback_host above) classifies
    literals, not names. ``localhost`` is the one name every consumer already
    understands. A value carrying a port (``0.0.0.0:8642``) is rejected too -
    the port has its own config key.

    IPv6 literals ARE accepted, as of the end-to-end IPv6 support this
    docstring's previous version made the precondition: the port probe resolves
    the family instead of assuming AF_INET, the listening socket is built
    family-aware (and dual-stack for ``::``), the printed URLs bracket a v6
    literal, and mDNS advertises an address the bind actually answers on. Before
    that work an IPv6 value raised ``socket.gaierror`` at startup and killed the
    process, which is why this validator refused them.

    A ZONE ID (``fe80::1%eth0``) is still rejected. The zone index names an
    interface as numbered on ONE machine, so it is meaningless in a config file
    that may be copied or restored elsewhere, and it does not survive the
    ``getaddrinfo(..., AI_PASSIVE)`` this server binds through - measured on
    this platform, where the scoped form fails with WinError 10049 while the
    same address unscoped binds fine. Rejecting it at write time turns that into
    a clear save-time error instead of a silent fallback at the next restart.

    Bindability is NOT decided here and cannot be: an address that is perfectly
    well-formed stops being bindable when DHCP moves the machine. That is
    ``cli._bind_preflight_error``'s job at the read site, and it now runs for
    every config-driven bind including loopback ones - ``::ffff:127.0.0.1`` is
    both genuinely loopback and unbindable on Windows, so a probe that skipped
    the loopback class would let this validator's own widening reopen the
    dead-server hole it exists to prevent."""
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
    ``localm.local`` and silently wrong for ``::1`` - it produces
    ``https://::1:8642/``, where the parser cannot tell the address's colons
    from the port separator. Every place that builds a host:port string routes
    through here so the bracketing rule lives once instead of at each caller
    (RFC 3986 section 3.2.2).

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
    post-update health watchdog, the hang alarm's self-probe). Every one of
    those used to hardcode ``127.0.0.1``, which is right for the IPv4
    wildcard and for loopback and WRONG for an IPv6 bind: a server bound only
    on ``::1`` has nothing listening on ``127.0.0.1``, so every self-call
    fails while the server itself is perfectly healthy.

    ``::`` maps to ``::1`` rather than ``127.0.0.1`` deliberately. localm binds
    the IPv6 wildcard dual-stack (see ``netlisten.create_listen_socket``), so
    ``127.0.0.1`` does in fact work there today - but only for as long as that
    upgrade succeeds, and its one documented fallback is a v6-only socket.
    ``::1`` is reachable in BOTH cases, so the self-call does not depend on an
    optimisation that is allowed to fail.

    A specific literal is returned as itself: it is the only address that bind
    is guaranteed to answer on. That is also the pre-existing behaviour of
    ``admin._watchdog_probe_host``, whose docstring already made this exact
    point about the hardcoded loopback URLs."""
    h = (bind_host or "").strip()
    if not h or h in ("0.0.0.0", "localhost"):
        return "127.0.0.1"
    if h == "::":
        return "::1"
    return h
