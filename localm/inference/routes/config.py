# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config routes: server config get/schema/patch, per-plugin media config,
generic plugin-contributed settings (host.add_settings()), and ComfyUI status.

Extracted verbatim from create_app(); behavior unchanged. Reads the live engine
from the http_server module global and the session-scoped audit mode from ctx.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

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


def _credentials_not_saved(reason: str, config_saved: bool) -> str:
    """Error detail for a credential write that failed after update_config had
    already saved the rest of the request."""
    reason = reason.rstrip()
    if not reason.endswith((".", "!", "?")):
        reason += "."
    detail = f"The credential change was not saved: {reason}"
    if config_saved:
        detail += " The other settings in this request were saved."
    return detail


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
        # operator (open mode, or an ADMIN key) sees everything.
        owner = _hs.caller_is_operator(request)
        if not owner:
            for k in admin_only_keys():
                cfg.pop(k, None)
            # The media plugins keep their own copy of launch_cmd / api_url /
            # workdir at cfg["plugins"][<plugin>]["comfy"][...]. "plugins" is
            # engine_managed, which gates the WRITE only and is never popped above,
            # so those nested values are scrubbed here as well.
            _scrub_media_admin_only(cfg)
        # Read-only extras for the frontend (skipped by the settings form): the
        # server's session mode now, the mode the GUI chat runs under, and the
        # coder default a new session starts with. A non-owner caller's chat is
        # recorded under the server's mode, so that is its chat mode.
        cfg["effective_mode"] = ctx.mode.value
        cfg["effective_chat_mode"] = (effective_mode("chat").value if owner
                                      else cfg["effective_mode"])
        cfg["effective_coder_mode"] = effective_mode("coder").value
        # Resolved context ceiling (VRAM-derived when ctx_auto) - the GUI
        # bases its compaction threshold on this, not the static config.
        eff_ctx = getattr(_hs._engine, "effective_ctx_max", None) if _hs._engine else None
        cfg["effective_ctx_max"] = eff_ctx if isinstance(eff_ctx, int) else None
        # The model that ceiling belongs to.
        ctx_model = getattr(_hs._engine, "display_name", None) if _hs._engine else None
        cfg["effective_ctx_model"] = ctx_model if isinstance(ctx_model, str) else None
        ctx_key = _hs._active_model_name if _hs._engine else None
        cfg["effective_ctx_model_key"] = ctx_key if isinstance(ctx_key, str) else None
        # A stable per-data-directory id, so the GUI can tell a restart of this
        # install apart from a different install that shares the browser origin
        # (localStorage is scoped by origin only, not by data directory).
        from localm.config import instance_id
        cfg["instance_id"] = instance_id()
        # The live bound port, which may differ from the persisted "port"
        # default after an explicit -p override or an auto-bump onto a
        # different free port.
        cfg["instance_port"] = getattr(request.app.state, "instance_port", None)
        return cfg

    @app.get("/v1/config/schema",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_config_schema(request: Request):
        """The typed settings schema (widget, label, help, group, owner, options,
        min, max) with each non-secret field's current value as its `default`.
        Secret fields never carry a value. Owner-only fields are omitted for a
        non-owner caller; writing them through ``PATCH /v1/config`` is refused
        with 403 regardless."""
        from localm.config import load_config
        from localm.settings_schema import schema_json
        is_owner = _hs.caller_is_operator(request)
        return {"fields": schema_json(values=load_config(), is_owner=is_owner)}

    @app.patch("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def patch_config(body: dict, request: Request):
        """Update known config keys and persist them. Unknown keys are rejected
        with 400, and owner-only keys from a non-owner caller with 403.

        The read-only fields that ``GET /v1/config`` adds (``effective_mode``,
        ``instance_id`` and the like) are ignored when echoed back, so a client
        can send back the whole config object it read.

        Model-source credential keys are stored separately, only after every
        other check and the config write have succeeded, so an error response
        leaves stored credentials unchanged. If that final credential write
        fails, the error detail says so and says whether the other settings in
        the request were saved. A config or credential file that exists but
        cannot be read answers 409 naming that file."""
        from localm.config import ConfigUnreadable, update_config
        from localm.settings_schema import (validate_update, admin_only_keys,
                                            engine_managed_keys)
        readonly = {"effective_mode", "effective_chat_mode", "effective_coder_mode",
                    "effective_ctx_max", "effective_ctx_model",
                    "effective_ctx_model_key", "instance_id",
                    "instance_port"}
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
        # cross-origin refusal). Requires an ADMIN principal; open mode passes.
        # Checked on the RAW body before validation,
        # so an unauthorized caller gets a 403 rather than a 400.
        locked = admin_only_keys() & set(body)
        if locked:
            if not _hs.caller_is_operator(request):
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
        # body before validation; open mode passes.
        managed = engine_managed_keys() & set(body)
        if managed:
            if not _hs.caller_is_operator(request):
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(managed)) + " requires an "
                    "owner (admin) key: it is plugin state, not a setting. Use the "
                    "plugin's own endpoint (/v1/tts/config, /v1/media/config/"
                    "<plugin>, /api/plugins/<name>/enable), which validates the "
                    "value and enforces its own permission.")
        # Model-source credentials (hf_token, civitai_api_key) are not in
        # DEFAULT_CONFIG and must never reach config.json: popped out of body and
        # validated here, then written to their own owner-only file after
        # update_config below returns. The admin_only check above already covers
        # authorization for these two keys.
        from localm.model_source_credentials import (CREDENTIAL_KEYS,
                                                     check_credentials_readable,
                                                     credentials_path,
                                                     set_credentials,
                                                     validate_credential_updates)
        cred_updates = {k: body.pop(k) for k in list(body) if k in CREDENTIAL_KEYS}
        if cred_updates:
            try:
                cred_updates = validate_credential_updates(cred_updates)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
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
                affected = await run_in_threadpool(
                    collection_provenance_report, candidate_model=new_model)
                if affected:
                    return {"needs_confirm": True, "model": new_model,
                            "collections": affected,
                            "note": collection_provenance_note(new_model, affected)}
        try:
            validated = validate_update(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        is_owner = _hs.caller_is_operator(request)
        # Refuse to enable require_auth while no API key exists: the next keyless
        # request would 401 and the GUI sends no Bearer, so the toggle could never
        # be undone from the GUI. Only ENABLING is blocked; turning it off or an
        # unrelated edit is fine.
        if validated.get("require_auth") is True:
            from localm.auth import require_auth_refusal
            refusal = require_auth_refusal()
            if refusal:
                raise HTTPException(400, refusal)
        # INVARIANT: the credential store is written last, synchronously, after
        # update_config has returned, and only checked (never written) before
        # it. A request that returns an error therefore leaves stored
        # credentials unchanged. See
        # test_config_save_timeout_leaves_the_stored_credential_alone.
        if cred_updates:
            try:
                check_credentials_readable()
            except ConfigUnreadable as e:
                raise HTTPException(409, str(e)) from e
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
            raise HTTPException(504, f"Saving the config timed out: {e}") from e
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        if cred_updates:
            try:
                await run_in_threadpool(set_credentials, cred_updates)
            except ConfigUnreadable as e:
                raise HTTPException(409, _credentials_not_saved(str(e), bool(validated))) from e
            except OSError as e:
                from localm.debuglog import logger as _dbg
                if isinstance(e, TimeoutError):
                    status, reason = 504, str(e)
                else:
                    status = 500
                    reason = (f"{credentials_path().name} could not be written "
                              f"({e.strerror or type(e).__name__})")
                _dbg.warning("credential write after a config save failed: %s", reason)
                raise HTTPException(status, _credentials_not_saved(reason, bool(validated))) from e
        # update_config() returns the FULL merged config, not just the changed
        # keys, so an admin_only field's value is stripped from a non-owner's
        # response echo - the same boundary get_config applies above.
        if is_owner:
            return result
        for k in admin_only_keys():
            result.pop(k, None)
        _scrub_media_admin_only(result)
        return result

    # ---------------------------------------------------------------- #
    #  Per-plugin media config (image / music / video)                   #
    # ---------------------------------------------------------------- #

    @app.get("/v1/media/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_media_config(request: Request):
        """Per-plugin media (ComfyUI) config for image, music and video, each
        with its editable fields and resolved values (the plugin's own value, else
        the shared global ComfyUI setting). The three plugins are configured
        independently.

        The launch command and API URL are owner-only, so their values are
        omitted for a non-owner caller, as for ``GET /v1/config/schema``."""
        from localm.config import load_config
        from localm.settings_schema import MEDIA_PLUGINS, media_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        labels = {"image": "Image", "music": "Music", "video": "Video"}
        is_owner = _hs.caller_is_operator(request)
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
        """Save one media plugin's own config block, merged into the stored
        block so other fields and other plugins are untouched. A blank field
        clears that plugin's override, which falls back to the shared global
        default. Answers 404 for an unknown media plugin, 400 for an invalid
        value, and 403 when a non-owner caller sets an owner-only field (launch
        command, API URL, working folder or a native model file)."""
        from localm.config import ConfigUnreadable, load_config, update_config
        from localm.settings_schema import (MEDIA_PLUGINS, media_admin_only_fields,
                                             media_schema_json, validate_media_block)
        if name not in MEDIA_PLUGINS:
            raise HTTPException(404, f"unknown media plugin: {name}")
        # Every field media_admin_only_fields() names (launch_cmd, api_url, workdir
        # and the native model file fields) requires an ADMIN principal in protected
        # mode. Open mode is the trusted local owner and passes.
        owner_only = sorted(k for k in (body or {}) if k in media_admin_only_fields())
        if owner_only:
            if not _hs.caller_is_operator(request):
                raise HTTPException(
                    403, f"Setting {', '.join(owner_only)} requires an admin key (it "
                    "configures a shell command, a network target, a folder a launcher "
                    "is discovered in, or a file on this machine that is loaded as a "
                    "model).")
        try:
            merge = validate_media_block(name, body or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

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
            raise HTTPException(504, f"Saving the {name} config timed out: {e}") from e
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        # Same admin_only filter as the GET route above: a non-owner config:write
        # key must not have an admin_only field's resolved value echoed back here
        # either.
        is_owner = _hs.caller_is_operator(request)
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
        is_owner = _hs.caller_is_operator(request)
        return {"plugin": TTS_PLUGIN,
                "active": bool(mgr and mgr.is_active(TTS_PLUGIN)),
                "fields": tts_schema_json(block, is_owner=is_owner)}

    @app.get("/v1/tts/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_tts_config(request: Request):
        """The tts plugin's editable settings with their resolved values (the
        user's override, else the shipped default), plus the field metadata
        (widget, label, help, options) needed to edit them and whether the plugin
        is active."""
        return _tts_payload(request)

    @app.post("/v1/tts/config",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_tts_config(body: dict, request: Request):
        """Save the tts plugin's own config block, merged key by key so
        unlisted keys are untouched. A blank field clears that override back to
        the shipped default. Setting the library or WASM path fields needs an
        owner key (403 otherwise); an invalid value answers 400."""
        from localm.config import ConfigUnreadable, update_config
        from localm.settings_schema import (TTS_PLUGIN, tts_admin_only_fields,
                                            validate_tts_block)
        # library / wasm_paths become a script URL and a WASM base URL that every
        # browser client loads, so setting them requires an ADMIN principal.
        # Checked on the RAW body before validation. Open mode is the trusted local
        # owner and passes.
        locked = tts_admin_only_fields() & set(body or {})
        if locked:
            if not _hs.caller_is_operator(request):
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key: it sets the script the text-to-speech "
                    "plugin loads in every browser.")
        try:
            merge = validate_tts_block(body or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

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
            raise HTTPException(504, f"Saving the tts config timed out: {e}") from e
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        return _tts_payload(request)

    # ---------------------------------------------------------------- #
    #  Generic plugin-contributed settings (host.add_settings())         #
    # ---------------------------------------------------------------- #

    @app.get("/v1/plugins/settings",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_plugin_settings(request: Request):
        """Settings sections contributed by active plugins, each with its
        resolved values (the plugin's own stored value, else the field's declared
        default). Inactive plugins have no entry, and fields that are owner-only
        are omitted for a non-owner caller."""
        from localm.config import load_config
        from localm.settings_schema import plugin_settings_schema_json
        manager = getattr(request.app.state, "plugin_manager", None)
        if manager is None:
            return {"plugins": []}
        cfg = load_config()
        plugins_cfg = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        is_owner = _hs.caller_is_operator(request)
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
        """Save one plugin's settings block, merged key by key so other fields
        and other plugins are untouched. A blank field clears that override back
        to the field's declared default. Answers 404 for a plugin that is not
        active or contributes no settings, since its fields are known only while
        it is loaded."""
        from localm.config import ConfigUnreadable, load_config, update_config
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
            if not _hs.caller_is_operator(request):
                raise HTTPException(
                    403, "Changing " + ", ".join(sorted(locked)) + " requires an "
                    "owner (admin) key.")
        try:
            merge = validate_plugin_settings_update(fields, body or {})
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

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
            raise HTTPException(504, f"Saving {name}'s settings timed out: {e}") from e
        except ConfigUnreadable as e:
            raise HTTPException(409, str(e)) from e
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        is_owner = _hs.caller_is_operator(request)
        return {"plugin": name,
                "fields": plugin_settings_schema_json(fields, block, is_owner=is_owner)}

    @app.get("/v1/comfy/status", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_comfy_status():
        """Whether ComfyUI is reachable (``alive``) and whether localm launched
        it (``launched_by_localm``), so a client can offer Stop and Restart only
        for a ComfyUI localm controls. Each call pings ComfyUI afresh. When it is
        not reachable and not launched by localm, ``port_held`` names the process
        occupying its port. Answers 504 if the check times out."""
        # default_api_url is the current base-URL helper.
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        from localm.media.comfy_client import (
            comfy_port_held_message, comfy_port_holder, mark_comfy_alive, mark_comfy_dead,
            spawned_pid)

        # Off the event loop: _comfy_alive is a blocking urlopen with a 1.0s
        # timeout, hit on the GUI's own poll timer, so inline it would hold the
        # loop for up to 1s per poll whenever ComfyUI is unreachable.
        def _check() -> dict:
            url = default_api_url()
            alive = _comfy_alive(url, timeout=1.0)
            (mark_comfy_alive if alive else mark_comfy_dead)(url)
            status: dict = {"alive": alive,
                            "launched_by_localm": spawned_pid(url) is not None}
            if not alive and not status["launched_by_localm"]:
                holder = comfy_port_holder(url, timeout=1.0)
                if holder is not None:
                    status["port_held"] = {"pid": holder["pid"], "name": holder["name"],
                                           "message": comfy_port_held_message(url, holder)}
            return status

        try:
            return await run_in_threadpool_bounded(_check, timeout=_COMFY_STATUS_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Checking ComfyUI status timed out: {e}") from e

    @app.post("/v1/comfy/stop", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_stop():
        """Stop ComfyUI: abort the in-flight render, clear the queue, free VRAM,
        and terminate the process localm launched (a ComfyUI the user started
        themselves is aborted but not terminated). Returns ``{ok, message}``;
        504 if stopping times out."""
        from localm.media.comfy_client import stop_comfy
        try:
            ok, message = await run_in_threadpool_bounded(
                stop_comfy, timeout=_COMFY_STOP_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Stopping ComfyUI timed out: {e}") from e
        return {"ok": ok, "message": message}

    @app.post("/v1/comfy/restart", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_restart():
        """Restart the ComfyUI localm launched: stop it, then launch it again
        with the configured launch command and working folder. Returns
        ``{ok, message}``; 504 if the restart times out."""
        from localm.config import load_config
        from localm.media.comfy_client import comfy_launch_wait_seconds, restart_comfy
        # restart_comfy() is stop_comfy() (bounded by _COMFY_STOP_TIMEOUT_S) then
        # ensure_comfy()'s own launch wait, so the budget reads the same
        # comfy_launch_timeout ensure_comfy honours (comfy_launch_wait_seconds).
        budget = _COMFY_STOP_TIMEOUT_S + comfy_launch_wait_seconds(load_config()) + 30.0
        try:
            ok, message = await run_in_threadpool_bounded(restart_comfy, timeout=budget)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Restarting ComfyUI timed out: {e}") from e
        return {"ok": ok, "message": message}
