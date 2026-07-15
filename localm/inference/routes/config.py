# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config routes: server config get/schema/patch, per-plugin media config, and
ComfyUI status.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
from the http_server module global and the session-scoped audit mode from ctx.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

import localm.inference.http_server as _hs
from localm import scopes


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config(request: Request):
        from localm.config import load_config
        from localm.audit import effective_mode
        from localm.settings_schema import admin_only_keys
        cfg = load_config()
        # REC-OWNER-SETTINGS: owner-only keys (e.g. the rag_* indexing settings)
        # widen a trust boundary, so their VALUES are not exposed to a non-owner
        # config:read caller (the schema hides the control too). The owner (open
        # mode -> caller_scopes None, or an ADMIN key) sees everything.
        held = _hs.caller_scopes(request)
        if held is not None and scopes.ADMIN not in held:
            for k in admin_only_keys():
                cfg.pop(k, None)
        # Read-only extras for the frontend (skipped by the settings form).
        # The server mode is fixed at startup (the audit log is opened then);
        # the coder default is resolved per new session.
        cfg["effective_mode"] = ctx.mode.value
        cfg["effective_coder_mode"] = effective_mode("coder").value
        # Resolved context ceiling (VRAM-derived when ctx_auto) - the GUI
        # bases its compaction threshold on this, not the static config.
        eff_ctx = getattr(_hs._engine, "effective_ctx_max", None) if _hs._engine else None
        cfg["effective_ctx_max"] = eff_ctx if isinstance(eff_ctx, int) else None
        # AUD-INSTANCEID: a stable per-data-directory id so the GUI can tell a
        # normal restart of THIS install apart from a different install that
        # happens to share the browser origin (localStorage is scoped by
        # origin only, not by data directory - see config.instance_id).
        from localm.config import instance_id
        cfg["instance_id"] = instance_id()
        return cfg

    @app.get("/v1/config/schema",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config_schema(request: Request):
        """The typed settings schema (widget/label/help/group/owner/options/
        min/max) with each non-secret field's CURRENT value injected as its
        `default`, so the GUI can render the right control pre-filled. Secret
        fields never carry a value (schema_json omits secret defaults).

        Owner-only fields (admin_only) are omitted for a non-owner caller so the
        GUI never renders a control they cannot use; the write is refused
        server-side regardless (see patch_config)."""
        from localm.config import load_config
        from localm.settings_schema import schema_json
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"fields": schema_json(values=load_config(), is_owner=is_owner)}

    @app.patch("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def patch_config(body: dict, request: Request):
        """Update known config keys and persist. Unknown keys are rejected.

        The read-only extras the GET handler injects (effective_mode etc.) are
        dropped first, so a client that round-trips the whole config object is
        not rejected for echoing back values it never edited."""
        from localm.config import update_config
        from localm.settings_schema import validate_update, admin_only_keys
        readonly = {"effective_mode", "effective_coder_mode", "effective_ctx_max",
                    "instance_id"}
        body = {k: v for k, v in body.items() if k not in readonly}
        # REC-OWNER-SETTINGS: an admin_only key widens a trust boundary, so a
        # non-owner config:write key must not set it. Today that is the rag_*
        # indexing settings (which host folders the indexer may read) and
        # net_allow_private (which DISABLES the SSRF guard, widening network reach)
        # - a filesystem boundary and a network one. Mirrors the media
        # launch_cmd/api_url guard: require an ADMIN principal; open mode (the
        # trusted local owner) has caller_scopes None and passes. Checked on the
        # RAW body before validation, so an unauthorized caller is refused up front
        # (a 403, not a 400 for a bad value it was never allowed to set anyway).
        locked = admin_only_keys() & set(body)
        if locked:
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key: it widens a trust boundary (which host "
                    "folders the server may read, or its network reach).")
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
        # APP-LIFECYCLE-1: update_config() is the atomic read-modify-write
        # helper - a bare load_config()/save_config() pair (as this call site
        # used to be) has an unlocked window where a concurrent config write
        # (e.g. set_media_config() below, which already uses update_config())
        # can be silently lost.
        #
        # OFF the event loop (REG-586): update_config() takes a cross-PROCESS
        # lock, so when another localm process holds it (the user running
        # `localm config ...` while the GUI is up) it waits in a blocking
        # time.sleep for up to _CROSS_LOCK_TIMEOUT. This handler is `async def`,
        # so doing that inline would freeze the whole server - health checks,
        # token streaming, every concurrent request - for the entire wait.
        return await run_in_threadpool(update_config,
                                       lambda cfg: cfg.update(validated))

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
    async def set_media_config(name: str, body: dict, request: Request):
        """Save ONE media plugin's own config block, deep-merged so the other
        fields and the other plugins are untouched. A blank field clears that
        plugin's override (it falls back to the shared global default)."""
        from localm.config import load_config, update_config
        from localm.settings_schema import (MEDIA_PLUGINS, media_schema_json,
                                             validate_media_block)
        if name not in MEDIA_PLUGINS:
            raise HTTPException(404, f"unknown media plugin: {name}")
        # REC-MEDIA-CMD: launch_cmd is run through the shell and api_url redirects
        # the render target, so setting either is privilege-escalation for a
        # non-owner config:write key. Require an ADMIN principal for those fields
        # in protected mode. Open mode is the trusted local owner (already gated by
        # the origin / shell-token guard), so caller_scopes is None there.
        if any(k in ("launch_cmd", "api_url") for k in (body or {})):
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Setting a media backend's launch_cmd or api_url requires "
                    "an admin key (it configures a shell command / network target).")
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

        # Off the event loop for the same reason as patch_config above (REG-586):
        # update_config() can block for up to _CROSS_LOCK_TIMEOUT waiting on the
        # cross-process lock, and this handler is `async def`.
        await run_in_threadpool(update_config, _mutate)
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        return {"plugin": name, "fields": media_schema_json(name, block, cfg)}

    @app.get("/v1/comfy/status", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_comfy_status():
        """Alive status of ComfyUI, and whether localm launched THIS one (so the
        GUI can show Stop/Restart only for a ComfyUI it can actually control).
        This is the "direct status request" trigger - always a real, uncached
        ping - and it primes the readiness cache with the fresh result, so a
        task submitted right after does not need to re-check (see
        comfy_client's module docstring on the readiness cache)."""
        # default_api_url is the current base-URL helper (the old _comfy_api_url
        # name no longer exists after the #292 shared-comfy-client refactor, so
        # importing it raised ImportError -> 500 on EVERY call) (NEW-COMFY-STATUS-IMPORT).
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        from localm.media.comfy_client import mark_comfy_alive, mark_comfy_dead, spawned_pid
        url = default_api_url()
        alive = _comfy_alive(url, timeout=1.0)
        (mark_comfy_alive if alive else mark_comfy_dead)(url)
        return {"alive": alive, "launched_by_localm": spawned_pid(url) is not None}

    @app.post("/v1/comfy/stop", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_stop():
        """Stop ComfyUI (NEW-STOPCOMFY): abort the in-flight render + clear the
        queue + free VRAM, and terminate the process localm launched (a ComfyUI the
        user started themselves is only aborted, never killed)."""
        from localm.media.comfy_client import stop_comfy
        ok, message = await run_in_threadpool(stop_comfy)
        return {"ok": ok, "message": message}

    @app.post("/v1/comfy/restart", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_restart():
        """Restart the ComfyUI localm launched (NEW-STOPCOMFY): stop it, then
        re-launch via the configured comfy_launch_cmd/comfy_workdir."""
        from localm.media.comfy_client import restart_comfy
        ok, message = await run_in_threadpool(restart_comfy)
        return {"ok": ok, "message": message}
