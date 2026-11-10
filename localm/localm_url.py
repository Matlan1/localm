# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``LOCALM_URL`` address of a localm server.

Every reader accepts both spellings: the server root (``http://host:port``)
and its OpenAI base (``http://host:port/v1``), with or without a trailing
slash."""

from __future__ import annotations

import os
from typing import Optional


def server_root(url: Optional[str]) -> Optional[str]:
    """*url* without a trailing slash or a trailing ``/v1``. None, empty and
    whitespace-only stay None."""
    if not url:
        return None
    url = url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url or None


def v1_base(url: Optional[str]) -> Optional[str]:
    """*url* as a ``/v1`` base. None, empty and whitespace-only stay None."""
    root = server_root(url)
    return root + "/v1" if root else None


def env_root() -> str:
    """``LOCALM_URL`` from the environment as a server root, or ``""``."""
    return server_root(os.environ.get("LOCALM_URL")) or ""


def _entry_root(entry: dict) -> Optional[str]:
    from .bindhost import self_connect_host, url_host
    port = entry.get("port")
    if not port:
        return None
    scheme = entry.get("scheme") or "http"
    return f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{port}"


def registry_token(url: Optional[str], home=None) -> Optional[str]:
    """The attach token of the running instance in this home's registry whose
    address is exactly *url* (root or ``/v1`` form), or None.

    The entry must pass the ``/whoami`` identity handshake, so a token is never
    handed to another process on a reused port. An address spelled differently
    from the registry entry (``localhost`` for ``127.0.0.1``) finds no entry.
    Never raises."""
    root = server_root(url)
    if not root:
        return None
    try:
        from . import instances
        from .config import home_dir
        for entry in instances.list_entries(home if home is not None else home_dir()):
            if not entry.get("token"):
                continue
            here = _entry_root(entry)
            if here is None or here.lower() != root.lower():
                continue
            if instances.default_probe(entry):
                return entry.get("token")
    except Exception:
        return None
    return None
