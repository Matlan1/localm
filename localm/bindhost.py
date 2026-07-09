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
