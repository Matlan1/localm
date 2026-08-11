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

from fastapi import Depends, FastAPI, Request

import localm.plugins.gui.web as _web
from localm import scopes
from localm.inference.http_server import (_require_auth, caller_scopes,
                                          effective_fs_access, require_scope)
from localm.executor import get_plugin_executor


def _probe_available(what: str, fn) -> bool:
    """Best-effort "is this optional feature configured" probe, isolated per
    feature.

    ONE probe per try block, deliberately. These checks were previously bundled,
    so an exception in the FIRST left the others at their fail-closed default
    even though their own functions had never been called - a raise inside
    ``issue_tracker.available()`` silently withdrew the UPDATE banner, a feature
    it has nothing to do with (the two read independent endpoints). Collapsing
    independent outcomes into one is the bug; isolating them is the fix.

    Failing CLOSED stays correct - hiding a control that might not work beats
    offering one that errors - but failing closed SILENTLY is not, so the reason
    is logged. DEBUG rather than WARNING because this runs on every GUI capability
    fetch: a broken probe would flood at warning level, and log flood is how a
    real warning gets ignored. The user-visible consequence (a missing button) is
    the signal at the right altitude; this line is the detail behind it."""
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
        """Live system load for the status-bar hardware monitor: CPU %, RAM,
        VRAM, and (NVIDIA only) GPU utilisation. Any section that cannot be
        measured on this box is simply absent - the frontend renders what it
        gets. Runs off-thread so a slow probe (e.g. nvidia-smi) never blocks
        the event loop."""
        from localm.sysstats import system_stats
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(get_plugin_executor(), system_stats)
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
    async def gui_capabilities(request: Request,
                               caller: Optional[set] = Depends(caller_scopes)):
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
        """The llama.cpp runtime backend actually installed on this box, plus
        enough hardware context for Settings to show it and, at most, offer a
        dismissable hint - NEVER to auto-switch anything. An update must never
        silently re-provision a different backend than the one already
        running (see updater._installed_backend()); this route is purely
        informational.

        Runs off-thread: hwdetect.detect() shells out (nvidia-smi/wmic/lspci),
        same reason /api/gpus and /api/stats offload their own probes so a
        slow driver never blocks the event loop."""
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
