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
- **Platforms:** Windows, Linux, and macOS are all first-class (DONE - PR #32; see
  docs/linux-setup.md). Native llama.cpp via copy-from-build (llama.dll /
  libllama.so / libllama.dylib); GPU = ROCm + CUDA + CPU on Linux, ROCm on Windows;
  install via setup.bat / setup.sh / one-click install.sh. Every plugin (Phase 3+)
  must stay cross-platform (declare platform-specific extras; no OS-only paths).
- **Defaults (nothing installed but chat):** ONLY chat is active by default (the
  protected builtin plugin #0); the model manager stays kernel/core (not a plugin).
  EVERY other plugin is AVAILABLE but NOT INSTALLED out of the box - it lives in
  the catalog until the user selects it. A not-installed plugin is fully inert:
  no routes, no tab, no command, nothing "present ready to use". The user opts in
  via first-run selection or the Plugins page. (During development a plugin may be
  installed only to test it.)
- **Two axes (install/uninstall, enable/disable):** `builtin/` + the external dir
  are the AVAILABLE catalog (discovered, inert). `config["plugins_installed"]` is
  the user's selected set; `config["plugins_enabled"]` toggles an installed plugin
  active vs inactive (you can keep a plugin installed but disabled). A plugin is
  active (loaded) iff installed AND enabled; installing also enables by default,
  uninstalling clears both. Uninstall keeps a builtin in the bundled catalog (so it
  can be reinstalled) but deletes a third-party plugin's copied directory; user
  data is kept unless an explicit delete-data is requested. Engine API:
  `install`/`uninstall`/`enable`/`disable` (+ config-only `set_installed_state`/
  `set_enabled_state` for CLI) and `/api/plugins/{name}/{install,uninstall,enable,
  disable}`; CLI `localm plugin install/uninstall/enable/disable/status`.
- **Dependencies follow install (pip extras):** a plugin's heavy deps live behind
  its pip extra (e.g. `localm[coder]`), so a chat-only user never installs torch/
  ComfyUI stacks. The small plugin code ships in the wheel but stays inert until
  installed; the heavy deps are what is genuinely absent. First-run / the Plugins
  page orchestrate the matching `pip install <extra>` on select (Phase 6), reusing
  the robust setup-script path (vendor index, etc.) - not a naive runtime pip.

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

## Distribution, commands & tools

How plugins reach a user, and how they contribute commands and tools.

- **Distribution model (available, not installed).** There is no "default set"
  that ships installed. The kernel knows a **catalog** of first-party plugins
  (the builtins in `localm/plugins/builtin/`); each is installable but ships
  NOT INSTALLED. Only chat is active out of the box. The user opts in two ways:
  first-run selection (pick from the catalog during setup) or the Plugins page
  later. A not-installed plugin is catalog-only and fully inert; its heavy deps
  (behind a pip extra) are not pulled until install. See "Two axes" and
  "Dependencies follow install" under Locked decisions for the mechanics.
- **Third-party install location (in-app, admin-gated).** Third-party plugins
  are installed from inside the running app via the Plugins page (admin scope +
  capability-consent prompt), NOT bundled into the installer. The installer/
  first-run only offers first-party catalog plugins. This keeps the install
  surface trusted and the consent decision close to where the user sees what a
  plugin asks for. (Rationale: matches the Jenkins / Home-Assistant / browser-
  extension model already locked for install security.)
- **Commands (plugin-contributed, discoverable when disabled).** Slash commands
  are declared in the manifest and registered only while the plugin is enabled.
  The kernel keeps a catalog of known commands across ALL first-party plugins,
  so a command from a known-but-disabled plugin is RECOGNISED: it replies
  "`/generate-image` needs the image plugin - enable it?" rather than "unknown
  command". A truly unknown command still errors normally. **Rename:** the legacy
  `/imagine` becomes `/generate-image` (and `/generate-music`, `/generate-video`)
  - a command name must say plainly what it does.
- **Third-party commands & tools.** Plugins declare `commands` and `tools` in the
  manifest; both are registered when the plugin is enabled and unregistered on
  disable. Each is scope-gated to the plugin's capability, so a command/tool can
  only do what the plugin is already permitted to do. Write/destructive tools go
  through the Phase-8 human-in-the-loop confirm path. Name collisions (two plugins
  claiming the same command/tool) are namespaced by plugin name and surfaced as a
  warning. `PluginSpec` gains `commands` and `tools` fields (added in Phase 3 when
  the first command-bearing feature lands).

## Media generation plugins (image, music, video)

Three SEPARATE plugins, each fully standalone (owner decision). No assumption
that they share a backend or are installed together; any subset works (image
only, music+video, etc.). ComfyUI is just the owner's current backend, not a
given.

- **Backend (shared plumbing, per-plugin choice).** The generic ComfyUI HTTP
  plumbing (launch + reachability + queue/poll/download + free-VRAM) stays one
  shared module (`localm/image_gen/comfy.py`, parameterised so each caller passes
  its own api_url / launch_cmd / workdir / output_dir). Each media plugin selects
  a backend by name (default `"comfy"`) and supplies its OWN workflow template +
  OWN config; a plugin can later use a completely different program (native
  ACE-Step server, etc.) without touching the others. The per-media workflow
  graphs already diverge (Flux vs ACE vs Wan), so only the transport is shared.
- **Per-plugin config.** `config["plugins"][<name>]` holds the backend type, its
  typed sub-block (`comfy.api_url/launch_cmd/workdir/output_dir`), and
  `reload_llm_after_generate`. A one-time shim migrates the legacy global keys
  (`comfy_launch_cmd`/`comfy_workdir`/`comfy_output_dir`/`reload_llm_after_imagine`)
  into the image plugin's block.
- **Share config ("use config from").** Each media plugin's config has an opt-in
  `use_config_from` selector naming ANOTHER media plugin. While active, this
  plugin resolves its backend config from the source LIVE (edit the source once,
  the sharer follows); the sharer's own fields are greyed out in the UI but NEVER
  cleared (toggle off and they take effect again). Cycle-prevented (no image<-video
  while video<-image; validated on save, defensively broken on read). Applies only
  to the three media plugins. If the source is disabled/uninstalled, the sharer
  falls back to its own preserved block (with a warning), never a silent default.
- **Tab design: hybrid, grouped under "Studio".** Surface gains an optional
  `group` field (`group = "studio"`). The SPA (Phase 4) renders: nothing when 0
  media plugins are enabled, a single flat tab when exactly 1, and one "Studio"
  parent that expands to the installed children when 2+. Parent rail position is
  stable so the nav does not reshuffle as plugins toggle.
- **Increment:** one plugin per PR (image, then music, then video). A formal
  `MediaBackend` protocol is deferred until a second (non-ComfyUI) backend
  actually lands; until then the seam is "a backend module selected by config
  name". `_localm_unload` (LLM-unload handoff, cross-cutting) can move to a host
  utility when convenient.
- **Phase-5 hardening (when the backend config becomes GUI-editable):** the
  backend `launch_cmd` is run through the shell (today it is the user's own local
  config, so this is trusted) and `api_url` is used unvalidated (it is meant to be
  loopback, e.g. 127.0.0.1:8188, so the netpolicy SSRF guard must NOT be applied
  naively or it would block legitimate local backends). Once a settings editor can
  change these from a request, gate edits behind admin scope and validate/escape
  there. Same pattern lives in music_gen/video_gen.

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
- **Phase 2 - Plugin engine (DONE):** `localm/plugins/engine.py` - `PluginManager`
  + concrete `PluginHost`. Discovery (in-tree `localm/plugins/builtin/` + external
  `~/.localm/plugins`), `parse_spec` (richer manifest), runtime load/unload via
  dynamic import + `register(host)`/`unregister()` with route mount/unmount on the
  live app (each plugin's routes auto-scoped to its capability; model untouched),
  enable/disable persisted to `config["plugins_enabled"]`, install/uninstall
  (keep-data default + `on_uninstall` hook), failure isolation, and the
  `/api/plugins` state + enable/disable API (scope-gated). `attach_engine` wired
  into `create_app`. 7 tests.
- **Phase 3 - Convert features to first-party plugins (incremental, one per PR):**
  coder, image, music, video, rag, web, voice, mcp each become a builtin plugin
  (manifest + `register()` mounting its routes + settings + tab), moved out of
  gui/web.py. They ship **DISABLED by default** - only chat is default-enabled.
  Reconcile the external-plugin loader into the unified model. **Chat becomes the
  built-in protected, default-enabled plugin #0.**
- **Phase 4 - Dynamic GUI:** SPA renders nav/tabs from `GET /api/plugins`
  (not hardcoded); the Plugins page becomes enable/disable/install/uninstall.
- **Phase 5 - Settings redesign:** schema-driven, grouped page; dropdowns for
  enums, free text for urls, number, toggle, **folder picker** (via `/api/fs/dirs`),
  masked secrets (not dumped by `GET /v1/config`), help, applies-on badges,
  validation, reset-to-default, search; per-plugin sections. Plugin settings live
  under `config["plugins"][name]`; `PATCH /v1/config` accepts that namespace.
- **Phase 6 - First-run + launcher:** plugin selection in the launcher (it can
  `uv pip install` extras) across setup.bat (Windows) AND setup.sh / install.sh
  (Linux/macOS, already shipped); per-plugin setup-on-install or prompt-on-first-
  use; launcher Auth card extends to owner key/scopes; one config source of truth
  shared by launcher and GUI.
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

Phases 0, 1, 2 complete and merged, plus native Linux/macOS support. Phase 2
added the plugin engine (PluginManager + PluginHost: runtime load/unload with
route mount/unmount, enable/disable, install/uninstall, `/api/plugins`), wired
into create_app. Full suite 1141 pass. Next: Phase 3 (convert the bundled
features into first-party plugins; chat becomes the protected plugin #0).
