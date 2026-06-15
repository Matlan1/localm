# localm: plugin architecture, permissions & settings - plan

Living spec for turning localm into a plugin-first app with a real permission
system and a schema-driven settings page. Phases are shipped in dependency order
on `feat/plugin-architecture`, each reviewable and tested.

## Vision

Everything except **chat** and the **model manager** is a plugin - coder, image,
music, video, rag (knowledge), web, voice, mcp - loaded through one uniform
engine. Chat itself is a *built-in, protected plugin* (plugin #0): the reference
implementation that proves the contract is rich enough for the most important
surface. First-party (in-tree) and third-party (`~/.localm/plugins/`) plugins use
the same contract.

## Locked decisions

- **Kernel (not pluggable):** server, plugin engine/host, auth/permissions,
  settings, inference engine + model manager. **Chat = built-in protected plugin.**
- **Lifecycle:** one in-process engine. Enable/disable/install/uninstall happen
  **without restarting the server or reloading the model** (model lifecycle is
  decoupled from plugin lifecycle).
- **Install security:** admin-scope-gated + capability manifest + explicit
  consent (Jenkins / Home-Assistant / browser-extension model). No sandbox or
  signing in v1.
- **Data on disable/uninstall:** disable keeps everything; uninstall keeps user
  content (rag collections, generated media) with an opt-in "also delete data",
  plus an optional plugin `on_uninstall` cleanup hook (WordPress deactivate-vs-delete).
- **Rollout:** phased on a branch.

## Two shared foundations (consumed by everything)

1. **Capability/scope taxonomy** (`localm/scopes.py`) - one string per
   capability; each plugin owns the scope equal to its name; kernel scopes are
   explicit (`models:write`, `config:write`, `plugins:admin`, `keys:admin`,
   `admin`). Used by permissions, plugin gating, and the chat control surface.
2. **Settings schema** (`localm/settings_schema.py`) - typed metadata per field
   (widget, label, help, group, options, secret, applies-on, owner). Drives the
   settings redesign, dropdown-vs-freeform, secret masking, and plugin-contributed
   settings. `owner` doubles as the Phase-3 migration map (comfy_*/net_*/voice_*/
   coder_* are plugin-owned).

## The plugin contract (`localm/plugins/contract.py`)

A **superset** of today's CLI manifest (`loader.py`). Manifest (`PluginSpec`):
`name, version, api_version, description, scope, requires_extras, capabilities,
data_subdir, builtin, protected, surface (tab + settings group), cli_entry
(legacy Click command), register_entry`. Code entrypoint: `register(host)` /
`unregister()` + optional `on_install` / `on_first_use` / `on_uninstall` hooks.

The **Host** API (versioned, `API_VERSION`) is what a plugin uses to attach
itself - `mount_router`, `add_settings`, `register_tab`, `plugin_config` /
`save_plugin_config`, `has_scope` / `require_scope`, `engine`, `audit`,
`browse_dirs`. A plugin never imports the app or global config directly, which is
what makes runtime load/unload possible.

## Phases

- **Phase 0 - Foundations (DONE):** scope taxonomy, settings-schema format + all
  core fields, the plugin manifest + host-API interfaces, this doc. Additive,
  tested (`test_scopes.py`, `test_settings_schema.py`, `test_plugin_contract.py`),
  no runtime wiring.
- **Phase 1 - Permissions (DONE):** scoped keystore (`auth.json`: named keys with
  scopes, hashed) added ADDITIVELY - the owner key (env `LOCALM_API_KEY` /
  `auth.key`) is implicitly `admin`, so no migration. `require_scope(...)` +
  `_enforce_scope` replace the binary check (default-deny once any key exists,
  open on loopback when none; owner implies every scope). Privileged routes
  gated (plugins install/delete -> plugins:admin, GET -> plugins:read; config
  GET/PATCH -> config:read/write; models load/unload -> models:write) plus a
  key-management API (`/v1/keys`, keys:admin). `/v1/models/{id}` path leak fixed
  (basename only).
- **Phase 2 - Plugin engine (`PluginManager`):** discovery (in-tree + external),
  manifest validation, load/unload (dynamic import + register/unregister, runtime
  route mount/unmount), enable/disable (config-persisted), install (admin +
  consent + capability display) / uninstall (keep + opt-in delete + hook),
  failure isolation, lazy dep-loading, `GET /api/plugins`. Model untouched.
- **Phase 3 - Convert features to first-party plugins:** coder, image, music,
  video, rag, web, voice, mcp each become a plugin (manifest + `register()`
  mounting existing routes + settings + tab), one at a time, behavior-preserving.
  Reconcile the external-plugin loader into the unified model. **Chat becomes the
  built-in protected plugin.**
- **Phase 4 - Dynamic GUI:** SPA renders nav/tabs from `GET /api/plugins`
  (not hardcoded); the Plugins page becomes enable/disable/install/uninstall.
- **Phase 5 - Settings redesign:** schema-driven, grouped page; dropdowns for
  enums, free text for urls, number, toggle, **folder picker** (via `/api/fs/dirs`),
  masked secrets (not dumped by `GET /v1/config`), help, applies-on badges,
  validation, reset-to-default, search; per-plugin sections. Plugin settings live
  under `config["plugins"][name]`; `PATCH /v1/config` accepts that namespace.
- **Phase 6 - First-run + launcher:** plugin selection in the launcher (it can
  `uv pip install` extras) + setup.bat; per-plugin setup-on-install or
  prompt-on-first-use; launcher Auth card extends to owner key/scopes; one config
  source of truth shared by launcher and GUI.
- **Phase 7 - Docs + example plugin + hardening:** plugin authoring guide, an
  example plugin, manifest api_version checks, audit of plugin actions, full tests.
- **Phase 8 - Chat control surface:** chat can manage the app via explicit,
  scope-gated **tools** (one action layer shared with the GUI buttons); read =
  instant, write/destructive = human-in-the-loop confirm + undo (reuse the coder's
  approval flow); all ingested content untrusted (proposal != execution); every
  action audited; default chat scope = "chat + read-management". Built on Phases
  1-3.

## Cross-cutting

- **Migrations:** single key -> keystore; flat config -> `config["plugins"][name]`.
- **Risks:** runtime route unmount in Starlette (track added routes + tests);
  heavy deps lazy-loaded; one bad plugin must never crash the server (extend the
  existing `discover_errors()` isolation).
- **Host<->plugin API versioning** so plugins survive localm updates.
- **Single source of truth** for enabled plugins (config), shared launcher + GUI.

## Status

Phase 0 + Phase 1 complete and merged. Phase 1 added the scoped keystore,
`require_scope` enforcement, the `/v1/keys` management API, route gating, and the
`/v1/models/{id}` path-leak fix (full suite 1118 pass). Next: Phase 2 (plugin engine).
