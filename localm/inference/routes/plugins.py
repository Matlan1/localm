# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin management routes: list, install, and remove.

Extracted verbatim from create_app(); behavior unchanged.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/plugins", dependencies=[Depends(require_scope(scopes.PLUGINS_READ))])
    async def list_plugins():
        from localm.plugins.loader import (discover_errors, discover_plugins,
                                           discover_warnings)
        return {
            "plugins": [
                {
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "entry": m.entry,
                    "path": str(m.path),
                    "tool_exports": m.tool_exports,
                }
                for m in discover_plugins()
            ],
            "errors": discover_errors(),
            # Non-fatal manifest warnings (unknown keys, LM-DA-007): the plugin
            # loads, but a typoed key silently does nothing, so say so.
            "warnings": discover_warnings(),
        }

    @app.post("/v1/plugins/install", dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def install_plugin_ep(body: dict):
        from pathlib import Path as _P
        from localm.plugins.loader import PluginError, install_plugin
        source = body.get("source", "")
        if not source:
            raise HTTPException(400, "Missing 'source' (local directory path)")
        try:
            manifest = install_plugin(_P(source), force=bool(body.get("force")))
        except PluginError as e:
            raise HTTPException(400, str(e))
        return {"status": "installed", "name": manifest.name,
                "version": manifest.version}

    @app.delete("/v1/plugins/{name}", dependencies=[Depends(require_scope(scopes.PLUGINS_ADMIN))])
    async def remove_plugin_ep(name: str):
        from localm.plugins.loader import PluginError, remove_plugin
        try:
            existed = remove_plugin(name)
        except PluginError as e:
            raise HTTPException(400, str(e))
        if not existed:
            raise HTTPException(404, f"Plugin not installed: {name}")
        return {"status": "removed", "name": name}
