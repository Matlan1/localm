# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI system routes: hardware-monitor stats, the companion-app address, and the per-key navigation entitlements."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import Depends, FastAPI, Request

import localm.plugins.gui.web as _web
from localm import scopes
from localm.inference.http_server import (_require_auth, caller_scopes,
                                          effective_fs_access, require_scope)
from localm.executor import get_plugin_executor


def _probe_available(what: str, fn) -> bool:
    """Best-effort 'is this optional feature configured' probe, isolated per feature."""
    try:
        return bool(fn())
    except Exception:
        from localm.debuglog import logger as _dbg
        _dbg.debug("gui capabilities: the %s probe failed; reporting it "
                   "unavailable for this request", what, exc_info=True)
        return False


def _bugreport_upload_available() -> bool:
    from localm import bugreport
    return bugreport.upload_available()


def _issues_available() -> bool:
    from localm import issue_tracker
    return issue_tracker.available()


def _update_available() -> bool:
    from localm import updater
    return updater.available()


def register(app: FastAPI, ctx) -> None:

    @app.get("/api/stats", dependencies=[Depends(_require_auth)])
    async def gui_stats():
        """Live system load for the status-bar hardware monitor: CPU %, RAM, VRAM, and (NVIDIA only) GPU utilisation."""
        from localm.sysstats import system_stats
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(get_plugin_executor(), system_stats)
        return stats

    @app.get("/api/companion", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def gui_companion():
        """LAN / Tailscale address a phone should open to reach THIS server, for the Companion-app card."""
        from localm import tls
        bind_host = getattr(app.state, "bind_host", "127.0.0.1")
        addrs = tls.companion_addresses()
        return {
            "network_bind": not _web._is_loopback_host(bind_host),
            "lan": addrs.get("lan") or "",
            "tailscale": addrs.get("tailscale") or "",
            # Why a CONFIGURED network bind was not applied at startup (no
            # strong API key / TLS unavailable), or "". The card shows it so a
            # browser-only user learns what to fix - without it, setting
            # Bind address and restarting would look like it silently did
            # nothing (we do not hide problems).
            "bind_fallback": getattr(app.state, "bind_fallback", None) or "",
        }

    @app.get("/api/capabilities", dependencies=[Depends(_require_auth)])
    async def gui_capabilities(request: Request,
                               caller: Optional[set] = Depends(caller_scopes)):
        """Effective navigation entitlements for the CURRENT key, so the GUI shows ONLY the tabs this key can actually use - a capability the key's scopes do not grant is never rendered (no show-then-'no access')."""
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
        # Three INDEPENDENT optional features, probed independently - see
        # _probe_available for why they no longer share a try block. Each shows
        # its control only when it will actually work; a probe that raises
        # reports only ITS OWN feature unavailable, and says so in the debug log.
        bug_upload = _probe_available("bug-report upload",
                                      _bugreport_upload_available)
        issues_avail = _probe_available("issues view", _issues_available)
        update_avail = _probe_available("update channel", _update_available)
        return {"scopes": sorted(held) if held else [], "open": held is None,
                "core": core, "plugins": plugins, "suggest_plugins": suggest,
                "bugreport_upload": bug_upload,
                "issues_available": issues_avail, "update_available": update_avail,
                # Host-filesystem reach for this caller ("none"|"shared"|"host"),
                # so the GUI hides host-path config fields and the host file
                # browser from a key that lacks it (server still enforces).
                "fs_access": effective_fs_access(request)}

    @app.get("/api/backend", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def gui_backend():
        """The llama.cpp runtime backend actually installed on this box, plus enough hardware context for Settings to show it and, at most, offer a dismissable hint - NEVER to auto-switch anything."""
        from localm import hwdetect, setup_llama
        loop = asyncio.get_running_loop()

        def _read():
            installed = None
            try:
                installed = setup_llama.installed_backend()
            except Exception:
                pass
            vendor = recommended = None
            try:
                det = hwdetect.detect()
                vendor = det.vendors[0] if det.vendors else None
                recommended = hwdetect.recommended_install_backend(det)
            except Exception:
                pass
            return {"installed": installed, "vendor": vendor, "recommended": recommended}

        return await loop.run_in_executor(get_plugin_executor(), _read)
