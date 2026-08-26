# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config routes: server config get/schema/patch, per-plugin media config,
generic plugin-contributed settings (host.add_settings()), and ComfyUI status.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
from the http_server module global and the session-scoped audit mode from ctx.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded

# Budgets for run_in_threadpool_bounded() below. Each bounds only the
# client-visible wait; the wrapped call keeps running.
#   _CONFIG_RMW_TIMEOUT_S: sits just above update_config()'s own cross-process
#     lock timeout (localm.config._CROSS_LOCK_TIMEOUT, 10s), so that more
#     specific TimeoutError surfaces first.
#   _COMFY_STATUS_TIMEOUT_S: _comfy_alive()'s own urlopen timeout is 1.0s.
#   _COMFY_STOP_TIMEOUT_S: stop_comfy()'s worst case is interrupt_comfy (two
#     sequential 10s urlopens) plus free_comfy_vram (one 30s urlopen) plus a
#     process-tree kill (a 15s taskkill on the Windows branch), so 65s.
_CONFIG_RMW_TIMEOUT_S = 20.0
_COMFY_STATUS_TIMEOUT_S = 15.0
_COMFY_STOP_TIMEOUT_S = 90.0


def _scrub_media_admin_only(cfg: dict) -> None:
    """Remove owner-only PER-PLUGIN media values from *cfg* in place.

    Driven off MEDIA_PLUGIN_FIELDS, not a hardcoded name list, so a field that
    gains admin_only later is scrubbed here automatically. Mutates the
    per-request dict from load_config(), never disk.
    """
    from localm.settings_schema import MEDIA_PLUGIN_FIELDS
    plugins = cfg.get("plugins")
    if not isinstance(plugins, dict):
        return
    for block in plugins.values():
        if not isinstance(block, dict):
            continue
        for field in MEDIA_PLUGIN_FIELDS:
            if not field.admin_only:
                continue
            node = block
            *parents, leaf = field.block_path
            for part in parents:
                node = node.get(part) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    break
            if isinstance(node, dict):
                node.pop(leaf, None)


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope

    @app.get("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config(request: Request):
        from localm.config import load_config
        from localm.audit import effective_mode
        from localm.settings_schema import admin_only_keys
        cfg = load_config()
        # Owner-only keys widen a trust boundary, so their values are not exposed
        # to a non-owner config:read caller; the schema hides the control too. The
        # owner (open mode, so caller_scopes None, or an ADMIN key) sees everything.
        held = _hs.caller_scopes(request)
        if held is not None and scopes.ADMIN not in held:
            for k in admin_only_keys():
                cfg.pop(k, None)
            # The media plugins keep their own copy of launch_cmd / api_url /
            # workdir at cfg["plugins"][<plugin>]["comfy"][...]. "plugins" is
            # engine_managed, which gates the WRITE only and is never popped above,
            # so those nested values are scrubbed here as well.
            _scrub_media_admin_only(cfg)
        # Read-only extras for the frontend (skipped by the settings form).
        # The server mode is fixed at startup (the audit log is opened then);
        # the coder default is resolved per new session.
        cfg["effective_mode"] = ctx.mode.value
        cfg["effective_coder_mode"] = effective_mode("coder").value
        # Resolved context ceiling (VRAM-derived when ctx_auto) - the GUI
        # bases its compaction threshold on this, not the static config.
        eff_ctx = getattr(_hs._engine, "effective_ctx_max", None) if _hs._engine else None
        cfg["effective_ctx_max"] = eff_ctx if isinstance(eff_ctx, int) else None
        # A stable per-data-directory id, so the GUI can tell a restart of this
        # install apart from a different install that shares the browser origin
        # (localStorage is scoped by origin only, not by data directory).
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
        from localm.settings_schema import (validate_update, admin_only_keys,
                                            engine_managed_keys)
        readonly = {"effective_mode", "effective_coder_mode", "effective_ctx_max",
                    "instance_id"}
        # `confirm` is this route's own protocol flag, not a config key: stripped
        # here alongside the read-only extras so validate_update never sees it.
        confirm = bool(body.get("confirm"))
        body = {k: v for k, v in body.items() if k not in readonly and k != "confirm"}
        # An admin_only key widens a trust boundary, so a non-owner config:write
        # key must not set it. Those keys are the rag_* indexing settings (which
        # host folders the indexer may read), net_allow_private (disables the SSRF
        # guard), the bugreport_upload_* / update_* endpoints (where diagnostics
        # are sent and updates fetched from), and cors_origins (which browser
        # origins may call the authenticated API; "*" also opts the sensitive
        # unauthenticated GETs in _CROSS_ORIGIN_GET_REFUSED out of their
        # cross-origin refusal). Requires an ADMIN principal; open mode has
        # caller_scopes None and passes. Checked on the RAW body before validation,
        # so an unauthorized caller gets a 403 rather than a 400.
        locked = admin_only_keys() & set(body)
        if locked:
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key: it widens a trust boundary (which host "
                    "folders the server may read, its network reach, or which "
                    "browser origins may call it).")
        # `plugins` / `plugins_enabled` are plugin STATE, not settings, and
        # validate_update has no schema for their contents - it stores them
        # verbatim. Their own write surfaces (/v1/tts/config,
        # /v1/media/config/<plugin>, /api/plugins/<name>/enable) enforce stronger
        # gates, so this route requires an owner for them too. Checked on the RAW
        # body before validation; open mode (caller_scopes None) passes.
        managed = engine_managed_keys() & set(body)
        if managed:
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(managed)) + " requires an "
                    "owner (admin) key: it is plugin state, not a setting. Use the "
                    "plugin's own endpoint (/v1/tts/config, /v1/media/config/"
                    "<plugin>, /api/plugins/<name>/enable), which validates the "
                    "value and enforces its own permission.")
        # The second writer of `embedding_model`, besides POST /api/rag/embedding.
        # A switch that would invalidate existing collections' semantic search
        # returns needs_confirm instead of taking effect. Placed after the
        # admin_only auth check above, so an unauthorized caller gets a 403 rather
        # than a report naming collections. Gated on an actual value change and on
        # there being collections at risk, so an ordinary settings save needs no
        # extra confirm round trip.
        if "embedding_model" in body and not confirm:
            from localm.config import load_config
            new_model = str(body["embedding_model"]).strip()
            current_model = str(load_config().get("embedding_model") or "")
            if new_model and new_model != current_model:
                from localm.rag import collection_provenance_note, collection_provenance_report
                affected = collection_provenance_report()
                if affected:
                    return {"needs_confirm": True, "model": new_model,
                            "collections": affected,
                            "note": collection_provenance_note(new_model, affected)}
        try:
            validated = validate_update(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        # Refuse to enable require_auth while no API key exists: the next keyless
        # request would 401 and the GUI sends no Bearer, so the toggle could never
        # be undone from the GUI. Only ENABLING is blocked; turning it off or an
        # unrelated edit is fine.
        if validated.get("require_auth") is True:
            from localm.auth import any_key_configured
            if not any_key_configured():
                raise HTTPException(
                    400,
                    "Cannot enable require_auth while no API key is configured: "
                    "this would lock you out. Set an owner key (the launcher or "
                    "LOCALM_API_KEY) or create a named key first, then enable it.")
        # update_config() is the atomic read-modify-write helper: a bare
        # load_config()/save_config() pair has an unlocked window in which a
        # concurrent config write can be lost.
        #
        # Off the event loop and bounded: update_config() takes a cross-PROCESS
        # lock and waits in a blocking time.sleep for up to _CROSS_LOCK_TIMEOUT
        # when another localm process holds it, and this handler is `async def`.
        # It is internally atomic (one in-memory mutation, one atomic file
        # replace under its own lock), so abandoning this await cannot leave a
        # half-written file.
        try:
            result = await run_in_threadpool_bounded(
                update_config, lambda cfg: cfg.update(validated),
                timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the config timed out: {e}")
        # update_config() returns the FULL merged config, not just the changed
        # keys, so an admin_only field's value is stripped from a non-owner's
        # response echo - the same boundary get_config applies above.
        if is_owner:
            return result
        for k in admin_only_keys():
            result.pop(k, None)
        return result

    # ---------------------------------------------------------------- #
    #  Per-plugin media config (image / music / video)                   #
    # ---------------------------------------------------------------- #

    @app.get("/v1/media/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_media_config(request: Request):
        """Per-plugin media (ComfyUI) config for image/music/video, each with its
        editable fields and RESOLVED values (the per-plugin block value, else the
        shared global comfy_* fallback). The GUI 'Media' section renders one
        subsection per plugin so the three are configured independently.

        launch_cmd/api_url are admin_only (a shell command / a render target), so
        their resolved value is OMITTED for a non-owner config:read caller -
        mirroring the write-side owner gate below and the admin_only_keys()
        treatment GET /v1/config gives the core schema."""
        from localm.config import load_config
        from localm.settings_schema import MEDIA_PLUGINS, media_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        labels = {"image": "Image", "music": "Music", "video": "Video"}
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        out = []
        for name in MEDIA_PLUGINS:
            block = plugins.get(name) if isinstance(plugins.get(name), dict) else {}
            out.append({"plugin": name, "label": labels[name],
                        "fields": media_schema_json(name, block, cfg,
                                                     is_owner=is_owner)})
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
        # launch_cmd is run through the shell, api_url redirects the render target,
        # and workdir is both where the launcher is AUTO-DISCOVERED when launch_cmd
        # is blank (discover_launch_cmd -> shlex.split -> Popen) and what the model
        # scanner walks into registry.json; the per-plugin workdir wins over the
        # global comfy_workdir. All three require an ADMIN principal in protected
        # mode. Open mode is the trusted local owner, so caller_scopes is None.
        if any(k in ("launch_cmd", "api_url", "workdir") for k in (body or {})):
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Setting a media backend's launch_cmd, api_url or workdir "
                    "requires an admin key (it configures a shell command, a network "
                    "target, or the folder a launcher is discovered in).")
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

        # Off the event loop and bounded, same as patch_config above:
        # update_config() can block for up to _CROSS_LOCK_TIMEOUT waiting on the
        # cross-process lock, and its own write is atomic, so an abandoned caller
        # cannot leave it half-finished.
        try:
            await run_in_threadpool_bounded(update_config, _mutate,
                                            timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the {name} config timed out: {e}")
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        # Same admin_only filter as the GET route above: a non-owner config:write
        # key must not have an admin_only field's resolved value echoed back here
        # either.
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"plugin": name, "fields": media_schema_json(name, block, cfg,
                                                              is_owner=is_owner)}

    # ---------------------------------------------------------------- #
    #  The tts plugin's config block (the browser-rendered Kokoro voice) #
    # ---------------------------------------------------------------- #

    def _tts_payload(request: Request) -> dict:
        """Shared by GET and POST /v1/tts/config, so the same admin_only filter
        (library/wasm_paths) applies to both the plain read and whatever a write
        response echoes back: a non-owner config:read/write caller must not learn
        the script/wasm path it is not allowed to set either."""
        from localm.config import load_config
        from localm.settings_schema import TTS_PLUGIN, tts_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        block = plugins.get(TTS_PLUGIN)
        # `active` is informational (the GUI hides a section whose plugin is not
        # running); the write itself is not gated on it, so settings can be prepared
        # before the plugin is enabled. An app with no plugin engine attached has no
        # manager. is_active() touches the disk (it lists the installed-plugins dir
        # and reads the config), like the load_config() above.
        mgr = getattr(request.app.state, "plugin_manager", None)
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"plugin": TTS_PLUGIN,
                "active": bool(mgr and mgr.is_active(TTS_PLUGIN)),
                "fields": tts_schema_json(block, is_owner=is_owner)}

    @app.get("/v1/tts/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_tts_config(request: Request):
        """The tts plugin's editable settings with their RESOLVED values (the
        user's override, else the shipped template default).

        This is the SETTINGS surface. The plugin's own /api/tts/config is the
        resolved runtime config the browser loads; this one carries the field
        metadata (widget/label/help/options) the GUI renders and edits."""
        return _tts_payload(request)

    @app.post("/v1/tts/config",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_tts_config(body: dict, request: Request):
        """Save the tts plugin's own config block, merged key by key so
        unlisted keys are untouched. A blank field clears that override (back to
        the shipped template default)."""
        from localm.config import update_config
        from localm.settings_schema import (TTS_PLUGIN, tts_admin_only_fields,
                                            validate_tts_block)
        # library / wasm_paths become a script URL and a WASM base URL that every
        # browser client loads, so setting them requires an ADMIN principal.
        # Checked on the RAW body before validation. Open mode is the trusted local
        # owner, so caller_scopes is None and this passes.
        locked = tts_admin_only_fields() & set(body or {})
        if locked:
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key: it sets the script the text-to-speech "
                    "plugin loads in every browser.")
        try:
            merge = validate_tts_block(body or {})
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _mutate(cfg: dict) -> None:
            plugins = cfg.get("plugins")
            if not isinstance(plugins, dict):
                plugins = cfg["plugins"] = {}
            block = plugins.get(TTS_PLUGIN)
            if not isinstance(block, dict):
                block = plugins[TTS_PLUGIN] = {}
            block.update(merge)

        # Off the event loop and bounded, same as patch_config above:
        # update_config() can block on the cross-process lock, and its own write is
        # atomic.
        try:
            await run_in_threadpool_bounded(update_config, _mutate,
                                            timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the tts config timed out: {e}")
        return _tts_payload(request)

    # ---------------------------------------------------------------- #
    #  Generic plugin-contributed settings (host.add_settings())         #
    # ---------------------------------------------------------------- #

    @app.get("/v1/plugins/settings",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_plugin_settings(request: Request):
        """Settings sections any ACTIVE plugin contributed via host.add_settings(),
        each with its RESOLVED values (the plugin's own config["plugins"][name]
        block, else the field's own declared default) - the generic counterpart
        to the tts/media sections above, for a plugin the core has no bespoke
        schema for (docs/plugin-interop.md's Open WebUI Valves interop seam).

        Unlike GET /v1/tts/config, there is no per-plugin "not active" flag to
        check here: an inactive plugin simply has no entry (its fields are not
        known anywhere while it is unloaded), so the list is naturally just the
        currently active sections."""
        from localm.config import load_config
        from localm.settings_schema import plugin_settings_schema_json
        manager = getattr(request.app.state, "plugin_manager", None)
        if manager is None:
            return {"plugins": []}
        cfg = load_config()
        plugins_cfg = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        out = []
        for sec in manager.get_all_plugin_settings():
            block = plugins_cfg.get(sec["plugin"])
            fields = plugin_settings_schema_json(sec["fields"], block, is_owner=is_owner)
            if not fields:
                continue        # every field was admin_only and hidden from this caller
            out.append({"plugin": sec["plugin"], "label": sec["label"], "fields": fields})
        return {"plugins": out}

    @app.post("/v1/plugins/{name}/settings",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_plugin_settings(name: str, body: dict, request: Request):
        """Save one plugin's add_settings() block, merged key by key so other
        fields (and other plugins) are untouched. A blank field clears that
        override back to the field's own declared default, same convention as
        POST /v1/tts/config.

        404s for a plugin with no active add_settings() fields: unlike the tts
        block (a fixed schema known ahead of time, so it can be pre-configured
        before the plugin is enabled), a dynamically-registered field list only
        exists while the plugin is actually loaded - there is nothing to
        validate against otherwise."""
        from localm.config import load_config, update_config
        from localm.settings_schema import (plugin_settings_admin_only_fields,
                                            plugin_settings_schema_json,
                                            validate_plugin_settings_update)
        manager = getattr(request.app.state, "plugin_manager", None)
        sections = {s["plugin"]: s["fields"]
                   for s in (manager.get_all_plugin_settings() if manager else [])}
        fields = sections.get(name)
        if fields is None:
            raise HTTPException(
                404, f"No such plugin, or it has no active settings: {name!r}")
        # An admin_only field widens a trust boundary, so a non-owner config:write
        # key must not set it. Checked on the RAW body before validation.
        locked = plugin_settings_admin_only_fields(fields) & set(body or {})
        if locked:
            held = _hs.caller_scopes(request)
            if held is not None and scopes.ADMIN not in held:
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key.")
        try:
            merge = validate_plugin_settings_update(fields, body or {})
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _mutate(cfg: dict) -> None:
            plugins = cfg.get("plugins")
            if not isinstance(plugins, dict):
                plugins = cfg["plugins"] = {}
            block = plugins.get(name)
            if not isinstance(block, dict):
                block = plugins[name] = {}
            block.update(merge)

        # Off the event loop / bounded, same reason as set_tts_config above.
        try:
            await run_in_threadpool_bounded(update_config, _mutate,
                                            timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving {name}'s settings timed out: {e}")
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"plugin": name,
                "fields": plugin_settings_schema_json(fields, block, is_owner=is_owner)}

    @app.get("/v1/comfy/status", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_comfy_status():
        """Alive status of ComfyUI, and whether localm launched THIS one (so the
        GUI can show Stop/Restart only for a ComfyUI it can actually control).
        This is the "direct status request" trigger - always a real, uncached
        ping - and it primes the readiness cache with the fresh result, so a
        task submitted right after does not need to re-check (see
        comfy_client's module docstring on the readiness cache)."""
        # default_api_url is the current base-URL helper.
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        from localm.media.comfy_client import mark_comfy_alive, mark_comfy_dead, spawned_pid

        # Off the event loop: _comfy_alive is a blocking urlopen with a 1.0s
        # timeout, hit on the GUI's own poll timer, so inline it would hold the
        # loop for up to 1s per poll whenever ComfyUI is unreachable.
        def _check() -> dict:
            url = default_api_url()
            alive = _comfy_alive(url, timeout=1.0)
            (mark_comfy_alive if alive else mark_comfy_dead)(url)
            return {"alive": alive, "launched_by_localm": spawned_pid(url) is not None}

        try:
            return await run_in_threadpool_bounded(_check, timeout=_COMFY_STATUS_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Checking ComfyUI status timed out: {e}")

    @app.post("/v1/comfy/stop", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_stop():
        """Stop ComfyUI: abort the in-flight render + clear the
        queue + free VRAM, and terminate the process localm launched (a ComfyUI the
        user started themselves is only aborted, never killed)."""
        from localm.media.comfy_client import stop_comfy
        try:
            ok, message = await run_in_threadpool_bounded(
                stop_comfy, timeout=_COMFY_STOP_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Stopping ComfyUI timed out: {e}")
        return {"ok": ok, "message": message}

    @app.post("/v1/comfy/restart", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_restart():
        """Restart the ComfyUI localm launched: stop it, then
        re-launch via the configured comfy_launch_cmd/comfy_workdir."""
        from localm.config import load_config
        from localm.media.comfy_client import comfy_launch_wait_seconds, restart_comfy
        # restart_comfy() is stop_comfy() (bounded by _COMFY_STOP_TIMEOUT_S) then
        # ensure_comfy()'s own launch wait, so the budget reads the same
        # comfy_launch_timeout ensure_comfy honours (comfy_launch_wait_seconds).
        budget = _COMFY_STOP_TIMEOUT_S + comfy_launch_wait_seconds(load_config()) + 30.0
        try:
            ok, message = await run_in_threadpool_bounded(restart_comfy, timeout=budget)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Restarting ComfyUI timed out: {e}")
        return {"ok": ok, "message": message}
