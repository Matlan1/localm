# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-to-speech plugin: Kokoro neural voices, rendered entirely in the browser.

The actual synthesis happens client-side (``static/tts.js`` loads Kokoro through
the vendored ``kokoro-js`` and caches the model in the browser), so this server
module is deliberately tiny: it serves the plugin's static assets and reports the
resolved engine/voice config. Nothing is written to disk, so privacy mode stays
trace-free with no extra gating (the in-browser path inherits chat's no-traces
property for free).

Routes (mounted by the engine, auto-scoped to the ``tts`` capability):
  GET /api/tts/status   - is the plugin usable / which engine
  GET /api/tts/config   - resolved {engine, model, device, dtype, voice, speed,
                          library, wasm_paths}

Config resolution mirrors the media plugins' template+override idea: the shipped
defaults live in the tracked ``tts.example.json`` template, and the user's
non-tracked overrides under ``config["plugins"]["tts"]`` win over them. The
default model id lives ONLY in the template, never hard-coded in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

_PLUGIN_DIR = Path(__file__).resolve().parent
_TEMPLATE = _PLUGIN_DIR / "tts.example.json"

_router = APIRouter()
_host = None


def _defaults() -> dict:
    """Shipped defaults from the tracked template (sans documentation keys)."""
    try:
        data = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _resolved() -> dict:
    """Template defaults overlaid with the user's non-tracked overrides in
    ``config["plugins"]["tts"]`` (set None to fall back to a default)."""
    cfg = _defaults()
    if _host is not None:
        try:
            override = _host.plugin_config("tts")
        except Exception:
            override = {}
        cfg.update({k: v for k, v in override.items() if v is not None})
    return cfg


@_router.get("/api/tts/status")
async def tts_status():
    return {"available": True, "engine": _resolved().get("engine", "kokoro")}


@_router.get("/api/tts/config")
async def tts_config():
    return _resolved()


def register(host) -> None:
    global _host
    _host = host
    host.mount_static("static")          # -> /plugins/tts/ (public; SPA import()s it)
    host.mount_router(_router)           # -> /api/tts/* (scope-gated to "tts")


def unregister() -> None:
    global _host
    _host = None
