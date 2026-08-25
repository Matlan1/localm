# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-to-speech plugin: Kokoro neural voices, rendered entirely in the browser."""

from __future__ import annotations

from fastapi import APIRouter

from localm.debuglog import logger
from localm.plugins.builtin.tts.settings import defaults as _defaults

_router = APIRouter()
_host = None


def _resolved() -> dict:
    """Template defaults overlaid with the user's non-tracked overrides in ``config['plugins']['tts']`` (set None to fall back to a default)."""
    cfg = _defaults()
    if _host is not None:
        try:
            override = _host.plugin_config("tts")
        except Exception as e:
            # Best-effort by design: a config-layer hiccup must not break TTS,
            # but the user's overrides silently reverting to the template
            # defaults needs a trace to be diagnosable.
            logger.debug("tts: plugin_config('tts') failed (%s); "
                         "using template defaults for this request", e)
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
