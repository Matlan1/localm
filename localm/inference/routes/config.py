# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config routes: server config get/schema/patch, per-plugin media config, and
ComfyUI status.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
from the http_server module global and the session-scoped audit mode from ctx.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config():
        from localm.config import load_config
        from localm.audit import effective_mode
        cfg = load_config()
        # Read-only extras for the frontend (skipped by the settings form).
        # The server mode is fixed at startup (the audit log is opened then);
        # the coder default is resolved per new session.
        cfg["effective_mode"] = ctx.mode.value
        cfg["effective_coder_mode"] = effective_mode("coder").value
        # Resolved context ceiling (VRAM-derived when ctx_auto) - the GUI
        # bases its compaction threshold on this, not the static config.
        eff_ctx = getattr(_hs._engine, "effective_ctx_max", None) if _hs._engine else None
        cfg["effective_ctx_max"] = eff_ctx if isinstance(eff_ctx, int) else None
        return cfg

    @app.get("/v1/config/schema",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config_schema():
        """The typed settings schema (widget/label/help/group/owner/options/
        min/max) with each non-secret field's CURRENT value injected as its
        `default`, so the GUI can render the right control pre-filled. Secret
        fields never carry a value (schema_json omits secret defaults)."""
        from localm.config import load_config
        from localm.settings_schema import schema_json
        return {"fields": schema_json(values=load_config())}

    @app.patch("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def patch_config(body: dict):
        """Update known config keys and persist. Unknown keys are rejected.

        The read-only extras the GET handler injects (effective_mode etc.) are
        dropped first, so a client that round-trips the whole config object is
        not rejected for echoing back values it never edited."""
        from localm.config import load_config, save_config
        from localm.settings_schema import validate_update
        readonly = {"effective_mode", "effective_coder_mode", "effective_ctx_max"}
        body = {k: v for k, v in body.items() if k not in readonly}
        try:
            validated = validate_update(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # SEC-3: refuse to enable require_auth while no API key exists. Doing so
        # is a one-way self-lockout: the very next keyless request 401s and the
        # GUI sends no Bearer, so the toggle could never be undone from the GUI.
        # Only block ENABLING it (turning it off or unrelated edits are fine);
        # the auth-state check belongs here, not in the static schema validator.
        if validated.get("require_auth") is True:
            from localm.auth import any_key_configured
            if not any_key_configured():
                raise HTTPException(
                    400,
                    "Cannot enable require_auth while no API key is configured: "
                    "this would lock you out. Set an owner key (the launcher or "
                    "LOCALM_API_KEY) or create a named key first, then enable it.")
        cfg = load_config()
        cfg.update(validated)
        save_config(cfg)
        return cfg

    # ---------------------------------------------------------------- #
    #  Per-plugin media config (image / music / video)                   #
    # ---------------------------------------------------------------- #

    @app.get("/v1/media/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_media_config():
        """Per-plugin media (ComfyUI) config for image/music/video, each with its
        editable fields and RESOLVED values (the per-plugin block value, else the
        shared global comfy_* fallback). The GUI 'Media' section renders one
        subsection per plugin so the three are configured independently."""
        from localm.config import load_config
        from localm.settings_schema import MEDIA_PLUGINS, media_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        labels = {"image": "Image", "music": "Music", "video": "Video"}
        out = []
        for name in MEDIA_PLUGINS:
            block = plugins.get(name) if isinstance(plugins.get(name), dict) else {}
            out.append({"plugin": name, "label": labels[name],
                        "fields": media_schema_json(name, block, cfg)})
        return {"plugins": out}

    @app.post("/v1/media/config/{name}",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_media_config(name: str, body: dict):
        """Save ONE media plugin's own config block, deep-merged so the other
        fields and the other plugins are untouched. A blank field clears that
        plugin's override (it falls back to the shared global default)."""
        from localm.config import load_config, update_config
        from localm.settings_schema import (MEDIA_PLUGINS, media_schema_json,
                                             validate_media_block)
        if name not in MEDIA_PLUGINS:
            raise HTTPException(404, f"unknown media plugin: {name}")
        try:
            merge = validate_media_block(name, body or {})
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _deep_merge(dst: dict, src: dict) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_merge(dst[k], v)
                else:
                    dst[k] = v

        def _mutate(cfg: dict) -> None:
            plugins = cfg.get("plugins")
            if not isinstance(plugins, dict):
                plugins = cfg["plugins"] = {}
            block = plugins.get(name)
            if not isinstance(block, dict):
                block = plugins[name] = {}
            _deep_merge(block, merge)

        update_config(_mutate)
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        return {"plugin": name, "fields": media_schema_json(name, block, cfg)}

    @app.get("/v1/comfy/status", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_comfy_status():
        """Returns the alive status of the ComfyUI server."""
        # default_api_url is the current base-URL helper (the old _comfy_api_url
        # name no longer exists after the #292 shared-comfy-client refactor, so
        # importing it raised ImportError -> 500 on EVERY call) (NEW-COMFY-STATUS-IMPORT).
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        alive = _comfy_alive(default_api_url(), timeout=1.0)
        return {"alive": alive}
