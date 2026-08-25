# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config routes: server config get/schema/patch, per-plugin media config, generic plugin-contributed settings (host.add_settings()), and ComfyUI status."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded

# Budgets for run_in_threadpool_bounded() below - see that module's docstring
# for what "the caller gives up" does and does not buy (the real call keeps
# running; only the client-visible wait is bounded). Each is sized generously
# over the wrapped call's own worst-case legitimate duration, so it only ever
# fires for a call that has gone genuinely beyond that:
#   _CONFIG_RMW_TIMEOUT_S: update_config()'s own cross-process lock already
#     times out at localm.config._CROSS_LOCK_TIMEOUT (10s) - this only needs
#     to be a bit larger so THAT more specific TimeoutError surfaces first.
#   _COMFY_STATUS_TIMEOUT_S: _comfy_alive()'s own urlopen timeout is 1.0s.
#   _COMFY_STOP_TIMEOUT_S: stop_comfy()'s own worst case is bounded by
#     interrupt_comfy (two sequential 10s urlopens = 20s) + free_comfy_vram
#     (one 30s urlopen, already wrapped in try/except there) + a
#     process-tree kill (comfy_client._kill_process_tree's Windows branch is
#     an explicit `subprocess.run(["taskkill", ...], timeout=15)`, NOT
#     "normally instant" as an earlier version of this comment claimed -
#     verified against the actual code) = 65s worst case, so budget 90s for
#     genuine margin (a prior 70s budget left only ~5s, not the ~20s that
#     earlier, uncorrected arithmetic implied).
_CONFIG_RMW_TIMEOUT_S = 20.0
_COMFY_STATUS_TIMEOUT_S = 15.0
_COMFY_STOP_TIMEOUT_S = 90.0


def _scrub_media_admin_only(cfg: dict) -> None:
    """Remove owner-only PER-PLUGIN media values from *cfg* in place."""
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
        # REC-OWNER-SETTINGS: owner-only keys (e.g. the rag_* indexing settings)
        # widen a trust boundary, so their VALUES are not exposed to a non-owner
        # config:read caller (the schema hides the control too). The owner (open
        # mode -> caller_scopes None, or an ADMIN key) sees everything.
        held = _hs.caller_scopes(request)
        if held is not None and scopes.ADMIN not in held:
            for k in admin_only_keys():
                cfg.pop(k, None)
            # The top-level pop above is NOT sufficient on its own. The media
            # plugins keep their OWN copy of launch_cmd / api_url / workdir at
            # cfg["plugins"][<plugin>]["comfy"][...] (written by set_media_config),
            # and "plugins" is engine_managed, which gates the WRITE only - it is
            # never popped here. So without this, a config:read key reads from the
            # GENERIC route exactly the values GET /v1/media/config deliberately
            # hides from it (media_schema_json's admin_only handling), and the
            # owner gate on those fields becomes cosmetic. Same
            # generic-route-outranks-the-specific-one shape as X8, on the read side.
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
        """The typed settings schema (widget/label/help/group/owner/options/ min/max) with each non-secret field's CURRENT value injected as its `default`, so the GUI can render the right control pre-filled."""
        from localm.config import load_config
        from localm.settings_schema import schema_json
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"fields": schema_json(values=load_config(), is_owner=is_owner)}

    @app.patch("/v1/config", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def patch_config(body: dict, request: Request):
        """Update known config keys and persist."""
        from localm.config import update_config
        from localm.settings_schema import (validate_update, admin_only_keys,
                                            engine_managed_keys)
        readonly = {"effective_mode", "effective_coder_mode", "effective_ctx_max",
                    "instance_id"}
        # NEW-RAG-DIM-NO-REEMBED: `confirm` is this route's own protocol flag
        # (mirrors POST /api/rag/embedding's), not a config key - strip it here
        # alongside the read-only extras so validate_update never sees it.
        confirm = bool(body.get("confirm"))
        body = {k: v for k, v in body.items() if k not in readonly and k != "confirm"}
        # REC-OWNER-SETTINGS: an admin_only key widens a trust boundary, so a
        # non-owner config:write key must not set it. Today that is the rag_*
        # indexing settings (which host folders the indexer may read),
        # net_allow_private (which DISABLES the SSRF guard, widening network reach),
        # the bugreport_upload_* / update_* endpoints (WHERE collected
        # diagnostics are sent and where updates are fetched from), and
        # cors_origins (WHICH BROWSER ORIGINS may call the authenticated API,
        # and "*" also opts the sensitive unauthenticated GETs in
        # _CROSS_ORIGIN_GET_REFUSED out of their cross-origin refusal) - a
        # filesystem boundary, three network ones, and a browser-origin one.
        # Mirrors the media
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
                    "folders the server may read, its network reach, or which "
                    "browser origins may call it).")
        # X8: `plugins` / `plugins_enabled` are plugin STATE, not settings, and
        # validate_update has no schema for their contents - it stores them
        # verbatim. Their real write surfaces enforce STRONGER gates
        # (/v1/tts/config and /v1/media/config/<name> require an owner for the
        # script-URL and launch_cmd/api_url fields; /api/plugins/<name>/enable
        # requires plugins:admin), so without this check the generic route is a
        # back door around the specific one: a non-owner config:write key could
        # set config["plugins"]["tts"]["library"] and have every browser import
        # that script. Same shape and same placement as the admin_only gate above
        # - on the RAW body, before validation. Open mode is the trusted local
        # owner (caller_scopes None) and passes.
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
        # NEW-RAG-DIM-NO-REEMBED: this is the second writer of `embedding_model`
        # besides the RAG picker's POST /api/rag/embedding (rag/plug.py), which
        # already dry-runs a switch that would invalidate existing collections'
        # semantic search before it takes effect. Without this, a Settings-page
        # edit silently invalidated them with no warning at all. Placed AFTER
        # the admin_only auth check above (embedding_model is admin_only), so an
        # unauthorized caller still gets a 403 rather than a report naming
        # collections it has no business seeing. Gated on an ACTUAL value change
        # (not just the key's presence) and on there being something to lose -
        # unlike the RAG-picker route this is a generic multi-key route used for
        # ordinary settings saves too, so a no-op or nothing-at-risk write must
        # not force every caller through an extra confirm round trip.
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
        # Bounded (follow-up to #1057): update_config() is internally atomic
        # (one in-memory mutation, one atomic file replace under its own
        # lock), so abandoning this await never risks a half-written file -
        # see run_in_threadpool_bounded's module docstring.
        try:
            result = await run_in_threadpool_bounded(
                update_config, lambda cfg: cfg.update(validated),
                timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the config timed out: {e}")
        # REC-OWNER-SETTINGS: update_config() returns the FULL merged config
        # (every key, not just the ones this call changed), so without this
        # filter a config:write-only, non-owner key's PATCH response would echo
        # back an admin_only field's value (e.g. update_token) even though the
        # write itself already refuses to let it SET that field - the same
        # owner/non-owner boundary get_config applies above, applied to the
        # echo here too.
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
        """Per-plugin media (ComfyUI) config for image/music/video, each with its editable fields and RESOLVED values (the per-plugin block value, else the shared global comfy_* fallback)."""
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
        """Save ONE media plugin's own config block, deep-merged so the other fields and the other plugins are untouched."""
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
        #
        # workdir joined this list in the CORE_FIELDS gating sweep. It is not
        # merely a folder: when launch_cmd is blank the launcher is AUTO-DISCOVERED
        # inside workdir (comfy_client.py:1125 discover_launch_cmd -> shlex.split ->
        # Popen), so an attacker-chosen workdir reaches execution with no
        # launch_cmd at all. It is also what the model scanner walks into
        # registry.json (scan.py:51-58), and the PER-PLUGIN value WINS over the
        # global comfy_workdir - so gating the core field alone left this open.
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

        # Off the event loop for the same reason as patch_config above (REG-586):
        # update_config() can block for up to _CROSS_LOCK_TIMEOUT waiting on the
        # cross-process lock, and this handler is `async def`.
        #
        # Bounded (follow-up to #1057): safe for the same reason as patch_config
        # above - update_config()'s own atomic write can never be left
        # half-finished by an abandoned caller.
        try:
            await run_in_threadpool_bounded(update_config, _mutate,
                                            timeout=_CONFIG_RMW_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the {name} config timed out: {e}")
        cfg = load_config()
        block = (cfg.get("plugins") or {}).get(name) or {}
        # Same admin_only filter as the GET route above: a non-owner config:write
        # key that just saved an ORDINARY field (the launch_cmd/api_url gate
        # above only blocks setting those two) must not have their resolved
        # value echoed back in this response either.
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"plugin": name, "fields": media_schema_json(name, block, cfg,
                                                              is_owner=is_owner)}

    # ---------------------------------------------------------------- #
    #  The tts plugin's config block (the browser-rendered Kokoro voice) #
    # ---------------------------------------------------------------- #

    def _tts_payload(request: Request) -> dict:
        """Shared by GET and POST /v1/tts/config, so the same admin_only filter (library/wasm_paths - REC-MEDIA-CMD's tts counterpart) applies to both the plain read and whatever a write response echoes back: a non-owner config:read/write caller must not learn the script/wasm path it is not allowed to set eith..."""
        from localm.config import load_config
        from localm.settings_schema import TTS_PLUGIN, tts_schema_json
        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        block = plugins.get(TTS_PLUGIN)
        # `active` is INFORMATIONAL (the GUI hides a section whose plugin is not
        # running); the write itself is deliberately NOT gated on it, so the
        # settings can be prepared before the plugin is enabled. An app with no
        # plugin engine attached (a bare test app) simply has no manager.
        # is_active() does touch the disk (it lists the installed-plugins dir and
        # reads the config), the same blocking reads this handler already makes
        # via load_config(), exactly like the media GET beside it.
        mgr = getattr(request.app.state, "plugin_manager", None)
        held = _hs.caller_scopes(request)
        is_owner = held is None or scopes.ADMIN in held
        return {"plugin": TTS_PLUGIN,
                "active": bool(mgr and mgr.is_active(TTS_PLUGIN)),
                "fields": tts_schema_json(block, is_owner=is_owner)}

    @app.get("/v1/tts/config",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def get_tts_config(request: Request):
        """The tts plugin's editable settings with their RESOLVED values (the user's override, else the shipped template default)."""
        return _tts_payload(request)

    @app.post("/v1/tts/config",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def set_tts_config(body: dict, request: Request):
        """Save the tts plugin's own config block, merged key by key so unlisted keys are untouched."""
        from localm.config import update_config
        from localm.settings_schema import (TTS_PLUGIN, tts_admin_only_fields,
                                            validate_tts_block)
        # SEC: library / wasm_paths become a script URL and a WASM base URL that
        # EVERY browser client loads, so setting them is privilege escalation for
        # a non-owner config:write key - same shape as the media launch_cmd /
        # api_url guard (REC-MEDIA-CMD). Checked on the RAW body before
        # validation, so an unauthorized caller is refused up front. Open mode is
        # the trusted local owner, so caller_scopes is None and this passes.
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

        # Off the event loop for the same reason as patch_config above (REG-586):
        # update_config() can block on the cross-process lock.
        #
        # Bounded (follow-up to #1057): see patch_config above - safe against
        # a half-written config file for the same reason.
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
        """Settings sections any ACTIVE plugin contributed via host.add_settings(), each with its RESOLVED values (the plugin's own config['plugins'][name] block, else the field's own declared default) - the generic counterpart to the tts/media sections above, for a plugin the core has no bespoke schema for (d..."""
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
        """Save one plugin's add_settings() block, merged key by key so other fields (and other plugins) are untouched."""
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
        # Same shape as REC-MEDIA-CMD / the tts library/wasm_paths gate: an
        # admin_only field widens a trust boundary, so a non-owner config:write
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
        """Alive status of ComfyUI, and whether localm launched THIS one (so the GUI can show Stop/Restart only for a ComfyUI it can actually control)."""
        # default_api_url is the current base-URL helper (the old _comfy_api_url
        # name no longer exists after the #292 shared-comfy-client refactor, so
        # importing it raised ImportError -> 500 on EVERY call) (NEW-COMFY-STATUS-IMPORT).
        from localm.image_gen.comfy import _comfy_alive, default_api_url
        from localm.media.comfy_client import mark_comfy_alive, mark_comfy_dead, spawned_pid

        # Off the event loop (same reason as update_config above, REG-638's
        # shape): _comfy_alive is a blocking urlopen with a 1.0s timeout, hit on
        # the GUI's own poll timer. Called inline this froze the WHOLE server -
        # every concurrent chat stream, SSE, and request - for up to 1s on every
        # poll whenever ComfyUI is not reachable, the common case.
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
        """Stop ComfyUI (NEW-STOPCOMFY): abort the in-flight render + clear the queue + free VRAM, and terminate the process localm launched (a ComfyUI the user started themselves is only aborted, never killed)."""
        from localm.media.comfy_client import stop_comfy
        try:
            ok, message = await run_in_threadpool_bounded(
                stop_comfy, timeout=_COMFY_STOP_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Stopping ComfyUI timed out: {e}")
        return {"ok": ok, "message": message}

    @app.post("/v1/comfy/restart", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def post_comfy_restart():
        """Restart the ComfyUI localm launched (NEW-STOPCOMFY): stop it, then re-launch via the configured comfy_launch_cmd/comfy_workdir."""
        from localm.config import load_config
        from localm.media.comfy_client import comfy_launch_wait_seconds, restart_comfy
        # restart_comfy() = stop_comfy() (bounded by _COMFY_STOP_TIMEOUT_S,
        # see post_comfy_stop above) THEN ensure_comfy()'s own launch wait -
        # read the SAME comfy_launch_timeout ensure_comfy will actually
        # honour (comfy_launch_wait_seconds), not an independent guess, or
        # this budget could silently drift smaller than the user's own
        # config and abort a launch that was still legitimately progressing.
        budget = _COMFY_STOP_TIMEOUT_S + comfy_launch_wait_seconds(load_config()) + 30.0
        try:
            ok, message = await run_in_threadpool_bounded(restart_comfy, timeout=budget)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Restarting ComfyUI timed out: {e}")
        return {"ok": ok, "message": message}
