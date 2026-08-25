# SPDX-License-Identifier: AGPL-3.0-or-later
"""Creating the server's public listening socket, family-aware."""

from __future__ import annotations

import socket
import sys
from typing import Optional

__all__ = ["create_listen_socket", "is_wildcard_host", "dual_stack_expected"]

# The two spellings of "every interface", one per family.
_WILDCARDS = ("0.0.0.0", "::", "")


def is_wildcard_host(host: Optional[str]) -> bool:
    """True when *host* means 'every interface' rather than one address."""
    return (host or "").strip() in _WILDCARDS


def dual_stack_expected(host: Optional[str]) -> bool:
    """True when binding *host* is expected to accept IPv4 clients as well as IPv6 ones, i.e. it is the IPv6 wildcard AND this platform supports clearing ``IPV6_V6ONLY``."""
    if (host or "").strip() != "::":
        return False
    try:
        return bool(socket.has_dualstack_ipv6())
    except Exception:      # pragma: no cover - defensive; probe must never raise
        return False


def create_listen_socket(host: str, port: int) -> socket.socket:
    """A bound (NOT yet listening) server socket for ``(host, port)``."""
    family, socktype, proto, _canon, sockaddr = socket.getaddrinfo(
        host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[0]
    sock = socket.socket(family, socktype, proto)
    try:
        # Match what asyncio's create_server would have done: SO_REUSEADDR on
        # POSIX only. On Windows it does not mean "reuse a TIME_WAIT port", it
        # means another process may steal a port we are already serving, so
        # asyncio deliberately omits it there and so do we.
        if os_name_is_posix():
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError:
                # Best-effort convenience only; a bind that then hits TIME_WAIT
                # surfaces as an ordinary OSError from the bind below, which the
                # caller already reports. Nothing is silenced.
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
    """Clear ``IPV6_V6ONLY`` on *sock* so the ``::`` wildcard also answers IPv4 clients, then READ THE FLAG BACK and log the outcome."""
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
