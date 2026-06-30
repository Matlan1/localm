# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI system routes: hardware-monitor stats, the companion-app address, and the
per-key navigation entitlements.

Extracted verbatim from attach_gui(); behavior unchanged. These read framework
state off ``app.state`` (the bind host, the plugin manager) rather than the shared
``ctx``.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import Depends, FastAPI

import localm.plugins.gui.web as _web
from localm import scopes
from localm.inference.http_server import (_require_auth, caller_scopes,
                                          require_scope)


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/stats", dependencies=[Depends(_require_auth)])
    async def gui_stats():
        """Live system load for the status-bar hardware monitor: CPU %, RAM,
        VRAM, and (NVIDIA only) GPU utilisation. Any section that cannot be
        measured on this box is simply absent - the frontend renders what it
        gets. Runs off-thread so a slow probe (e.g. nvidia-smi) never blocks
        the event loop."""
        from localm.sysstats import system_stats
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(None, system_stats)
        return stats

    @app.get("/api/companion", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def gui_companion():
        """LAN / Tailscale address a phone should open to reach THIS server, for
        the Companion-app card. The card builds full URLs from these plus the
        browser's own scheme + port (the server listens on one port across every
        interface), so it never shows the meaningless loopback address. On the
        default loopback bind (``localm gui``) no phone can connect yet, so
        ``network_bind`` is False and the card explains how to bind to the
        network instead."""
        from localm import tls
        bind_host = getattr(app.state, "bind_host", "127.0.0.1")
        addrs = tls.companion_addresses()
        return {
            "network_bind": not _web._is_loopback_host(bind_host),
            "lan": addrs.get("lan") or "",
            "tailscale": addrs.get("tailscale") or "",
        }

    @app.get("/api/capabilities", dependencies=[Depends(_require_auth)])
    async def gui_capabilities(caller: Optional[set] = Depends(caller_scopes)):
        """Effective navigation entitlements for the CURRENT key, so the GUI shows
        ONLY the tabs this key can actually use - a capability the key's scopes do
        not grant is never rendered (no show-then-'no access').

        Baseline-gated (any valid key) on purpose: a key must be able to learn its
        OWN entitlements without holding plugins:read, otherwise a narrow key could
        never see the tabs it IS allowed. In open mode (no key configured) caller is
        None and everything is granted. chat is always present - chatting needs no
        scope; the 'chat' scope gates the chat-HISTORY plugin, not the chat turn."""
        held = caller                      # None => open mode / full access
        def granted(required: str) -> bool:
            return held is None or scopes.grants(held, required)
        core = {
            "chat":     True,
            "models":   granted(scopes.MODELS_READ),
            "plugins":  granted(scopes.PLUGINS_READ),
            "settings": granted(scopes.CONFIG_READ),
        }
        mgr = getattr(app.state, "plugin_manager", None)
        plugins, suggest = [], True
        if mgr is not None:
            state = mgr.api_state()
            suggest = bool(state.get("suggest_plugins", True))
            # Only surface plugins this key's scopes grant; renderNav still filters
            # to active+tab, and the command-hint map keeps inactive-but-granted
            # entries so "/cmd needs the X plugin" still works for a scoped key.
            for p in state.get("plugins", []):
                if granted(p.get("scope") or p.get("name") or ""):
                    plugins.append(p)
        # Whether the in-app "Send to maintainer" upload channel is configured, so
        # the GUI shows that button only when it will actually work (otherwise the
        # report is saved-to-file and emailed, as before).
        try:
            from localm import bugreport
            bug_upload = bugreport.upload_available()
        except Exception:
            bug_upload = False
        # Whether the read-only issues view and the update banner should be shown
        # (both ride the same proxy; hidden when not configured).
        try:
            from localm import issue_tracker, updater
            issues_avail = issue_tracker.available()
            update_avail = updater.available()
        except Exception:
            issues_avail = update_avail = False
        return {"scopes": sorted(held) if held else [], "open": held is None,
                "core": core, "plugins": plugins, "suggest_plugins": suggest,
                "bugreport_upload": bug_upload,
                "issues_available": issues_avail, "update_available": update_avail}
