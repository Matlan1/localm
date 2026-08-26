# SPDX-License-Identifier: AGPL-3.0-or-later
"""Creating the server's public listening socket, family-aware.

``asyncio.start_server(host=...)`` cannot serve localm's IPv6 story on its own.
``BaseEventLoop.create_server`` sets ``IPV6_V6ONLY`` to True unconditionally for
every AF_INET6 socket it builds, so ``-H ::`` would listen for IPv6 only and an
IPv4 client on the same LAN could not reach it.

``::`` means "every interface"; splitting it into "every interface, but only
half the internet" would make the printed URLs lie and would break every
self-call the server makes to its own loopback API. So localm builds the socket
itself, clears ``IPV6_V6ONLY`` for the wildcard, and hands the ready socket to
asyncio (and to uvicorn on the fallback path, which accepts ``sockets=[...]``).

A SPECIFIC literal is never upgraded: ``::1`` or ``2001:db8::5`` names one
address in one family, and a dual-stack flag on it would change nothing.
"""

from __future__ import annotations

import socket
import sys
from typing import Optional

__all__ = ["create_listen_socket", "is_wildcard_host", "dual_stack_expected"]

# The two spellings of "every interface", one per family.
_WILDCARDS = ("0.0.0.0", "::", "")


def is_wildcard_host(host: Optional[str]) -> bool:
    """True when *host* means "every interface" rather than one address."""
    return (host or "").strip() in _WILDCARDS


def dual_stack_expected(host: Optional[str]) -> bool:
    """True when binding *host* is expected to accept IPv4 clients as well as
    IPv6 ones, i.e. it is the IPv6 wildcard AND this platform supports clearing
    ``IPV6_V6ONLY``.

    Callers use this to print the truth BEFORE the socket exists (the CLI prints
    its reachable URLs before the server binds). The capability comes from
    ``socket.has_dualstack_ipv6()``, the stdlib's own probe, rather than from a
    platform assumption."""
    if (host or "").strip() != "::":
        return False
    try:
        return bool(socket.has_dualstack_ipv6())
    except Exception:      # pragma: no cover - defensive; probe must never raise
        return False


def create_listen_socket(host: str, port: int) -> socket.socket:
    """A bound (NOT yet listening) server socket for ``(host, port)``.

    Bound only, because both consumers call ``listen()`` themselves:
    ``asyncio.create_server(sock=...)`` does it in ``_start_serving`` and
    uvicorn's own ``Config.bind_socket`` likewise binds without listening.

    Raises ``OSError`` when the address cannot be resolved or bound, which is the
    same failure the caller already handles for an IPv4 bind - a stale interface
    IP, or the port being taken between the availability probe and here.
    """
    family, socktype, proto, _canon, sockaddr = socket.getaddrinfo(
        host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[0]
    sock = socket.socket(family, socktype, proto)
    try:
        # SO_REUSEADDR on POSIX only, matching asyncio's create_server.
        if os_name_is_posix():
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                # Best-effort; a TIME_WAIT bind failure surfaces from the bind below.
                pass
        if family == socket.AF_INET6 and (host or "").strip() == "::":
            _try_dual_stack(sock)
        sock.bind(sockaddr)
    except BaseException:
        sock.close()
        raise
    sock.setblocking(False)
    return sock


def os_name_is_posix() -> bool:
    """Whether SO_REUSEADDR is the safe, asyncio-compatible choice here."""
    import os
    return os.name == "posix" and sys.platform != "cygwin"


def _try_dual_stack(sock: socket.socket) -> None:
    """Clear ``IPV6_V6ONLY`` on *sock* so the ``::`` wildcard also answers IPv4
    clients, then READ THE FLAG BACK and log the outcome.

    ``setsockopt`` succeeding is not the same fact as the option having taken (a
    platform may accept and ignore it). A degradation to IPv6-only is a real
    reduction in reach, so it is reported at WARNING rather than swallowed; the
    server still starts and still serves IPv6, so this is a note and not a hard
    failure.
    """
    from localm.debuglog import logger
    if not hasattr(socket, "IPPROTO_IPV6") or not hasattr(socket, "IPV6_V6ONLY"):
        logger.warning("netlisten: this platform has no IPV6_V6ONLY; the :: bind "
                       "serves IPv6 clients only")
        return
    try:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    except OSError as e:
        logger.warning("netlisten: could not clear IPV6_V6ONLY (%s); the :: bind "
                       "serves IPv6 clients only, not IPv4", e)
        return
    try:
        still_v6only = bool(sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY))
    except OSError as e:                     # pragma: no cover - readback failure
        logger.warning("netlisten: could not confirm IPV6_V6ONLY was cleared (%s); "
                       "IPv4 clients may not reach this :: bind", e)
        return
    if still_v6only:
        logger.warning("netlisten: IPV6_V6ONLY stayed set after clearing it; the "
                       ":: bind serves IPv6 clients only, not IPv4")
    else:
        logger.debug("netlisten: :: bound dual-stack (IPv4 and IPv6 clients)")
