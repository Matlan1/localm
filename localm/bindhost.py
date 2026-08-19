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
    """True when *host* is a string a server could plausibly bind: ``localhost``
    or a literal IP address (IPv4 or IPv6, including the ``0.0.0.0``/``::``
    wildcards). Used by BOTH the write-time validation of the ``bind_host``
    config key (settings_schema) and its read site (cli._resolve_bind_host) -
    the read-time re-check matters because a hand-edited config.json bypasses
    write-time validation entirely, and an unbindable value reaching uvicorn
    would kill the server at startup, which for a config-driven bind may be a
    user with no terminal to recover from (the GUI is how they would fix it).

    Hostnames are deliberately NOT accepted: binding resolves them through the
    OS resolver, which can change answers (or fail) between restarts, and every
    consumer of the loopback/network split (is_loopback_host above) classifies
    literals, not names. ``localhost`` is the one name every consumer already
    understands. A value carrying a port (``0.0.0.0:8642``) is rejected too -
    the port has its own config key."""
    if not host or not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
