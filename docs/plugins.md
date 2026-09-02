# Plugins

localm is plugin-first. The core is a model loader plus the plugin engine; the
only always-present feature is **chat**, which itself ships as the protected,
preinstalled plugin #0. Everything else - the coder agent, image/music/video
generation, RAG (Knowledge), web access, durable memory, voice (Whisper STT),
text-to-speech (Kokoro), scheduled jobs, and the MCP server - is a plugin you
install.

This guide covers the plugin lifecycle, the `plugin.toml` manifest, the
`register(host)` contract, the Host API, and how to ship server routes, a GUI
tab, settings, and client-side assets. For wrapping extensions from other
ecosystems (Open WebUI, oobabooga, Anthropic Skills), see
[plugin-interop.md](plugin-interop.md).

## Concept: store, installed, enabled, active

A plugin moves along two independent axes:

| Term | Meaning |
|------|---------|
| **Available** (store) | Bundled first-party plugins live read-only in `localm/plugins/builtin/` (the "store"). They are NOT loaded from there. |
| **Installed** | Physically present on disk in the installed folder (`<data dir>/plugins/<name>/` with a `plugin.toml`). Installing copies a plugin from the store into the installed folder. "Installed" is disk presence, not a config flag. |
| **Enabled** | A config toggle in `config["plugins_enabled"]` (WordPress-style). |
| **Active** | `installed AND enabled`. Only active plugins are discovered and loaded; their routes, tabs, and assets are mounted on the live server. |

Out of the box **only chat is active**. Nothing else exists until you install it,
so the base install is small and the attack surface is opt-in. `chat` is
`protected` (cannot be disabled or uninstalled) and `default_enabled` (active on
first run).

Manage plugins from the CLI:

```
localm plugin status            # list available / installed / enabled / active
localm plugin install <name>    # copy store -> installed, and enable
localm plugin enable <name>     # enable an already-installed plugin
localm plugin disable <name>    # keep installed, make inactive
localm plugin uninstall <name>  # remove from the installed folder (keeps data unless --delete-data)
localm plugin refresh [<name>]  # re-copy installed builtins whose store code changed
```

A running GUI server picks up newly enabled HTTP plugins on its next start;
toggling a plugin while the server runs (via the GUI Plugins page or the
`/api/plugins/{name}/...` routes) mounts/unmounts it without a restart.

### Upgrades: refreshing a stale installed copy

An installed first-party plugin is a *copy* of the store source taken at install
time. A localm upgrade ships newer plugin code, but the older installed copy in
your data dir keeps shadowing it - so without a refresh you would silently run
stale plugin code (including missing fixes). Staleness is detected by a content
hash of the store source, not the plugin version (a bugfix often does not bump
it).

Builtins are refreshed automatically on server launch (and on `enable`); you can
also force it with `localm plugin refresh` (all installed builtins) or
`localm plugin refresh <name>` (one), or the **Refresh** button on the GUI
Plugins page. The refresh re-copies only the plugin directory, so your per-plugin
config (kept in `config.json`, see [Per-plugin config](#per-plugin-config-and-privacy))
and plugin data (under the data dir) are preserved. A plugin you installed from
your own directory (a third-party plugin, marked `source = "external"`) is never
a refresh target and is never overwritten.

Some plugins need heavy Python dependencies shipped as a pip extra. By default
localm installs the extra for you on the host when you install or enable such a
plugin (the `auto_install_plugin_deps` setting, which `localm plugin setup` asks
about and remembers); see [Dependencies](#dependencies) for the full behaviour and
the `localm plugin install-deps` command. `localm doctor` reports any enabled
plugin whose extras are missing.

## Anatomy of a plugin

A plugin is a directory with a `plugin.toml` manifest and a Python module that
exposes `register(host)`:

```
my-plugin/
  plugin.toml
  plug.py            # exposes register(host) [and optional unregister()]
  static/            # optional client-side assets (served at /plugins/<name>/)
```

The bundled `web` plugin (routes only) and `tts` plugin (client-side assets)
are the two smallest worked examples in `localm/plugins/builtin/`.

## `plugin.toml` reference

```toml
[plugin]
name = "myplugin"            # required; CLI name, must be a valid identifier
version = "1.0.0"
api_version = 1              # contract version the engine must support (default 1)
description = "What it does"
scope = "myplugin"           # capability scope gating its routes (default: name)
register = "plug"            # module exposing register(host) (default "plugin");
                             # "module:attr" is also accepted
requires = ["other-plugin"]  # other plugins that must be installed first
requires_extras = ["myextra"]# pip extras carrying heavy deps (pip install "localm[myextra]")
capabilities = ["feature"]   # declared capabilities (parsed and stored; not surfaced anywhere yet)
data_subdir = "my_data"      # storage under the data dir (<data dir>/my_data); "" = none
protected = false            # cannot be disabled/uninstalled (chat only)
default_enabled = false      # active on first run (chat only)
cli = "module:attr"          # optional legacy Click command entry point

[surface]                    # optional GUI/SPA contribution
tab_id = "mytab"             # "" = no tab (settings-only or headless plugin)
label = "My Plugin"
icon = "book"                # one of a fixed set: chat, code, image, music,
                             # video, book, clock; anything else falls back to
                             # a generic icon
assets_dir = "static"        # client assets dir; served at /plugins/<name>/ when set
client_entry = "myplugin.js" # ES module under assets_dir the SPA import()s on boot
settings_group = "My Plugin" # label for this plugin's settings section
group = "studio"             # nav group; tabs sharing a group collapse together
```

The dataclasses backing this are `Surface` and `PluginSpec` in
[contract.py](../localm/plugins/contract.py); the parser is `parse_spec` in
[engine.py](../localm/plugins/engine.py).

## The `register(host)` contract

```python
def register(host) -> None:
    """Called once when the plugin is loaded. Attach yourself via host."""

def unregister() -> None:
    """Optional: clean up when the plugin is disabled/unloaded."""

# Optional lifecycle hooks, called if present:
def on_install() -> None: ...                # called on an HTTP/GUI-driven install
def on_uninstall(delete_data: bool = False) -> None: ...  # called at uninstall
def on_first_use() -> None: ...              # called once the first time the plugin is loaded/activated; persisted so it never re-fires on a later server start
```

The engine hands `register` a `PluginHost`; everything the plugin attaches
through it is tracked and removed again on `unregister`, so enabling/disabling is
instant and clean.

All four hooks are best-effort: a hook that raises never blocks the
install/enable/uninstall it was called from. `on_install` and `on_first_use`
failures are logged at WARNING (so they show up in a bug report's activity log,
not just a debug file); an `on_uninstall` failure stays silent, since uninstall
must proceed regardless.

`on_install` fires only from the HTTP/GUI install routes (`PluginManager.install`
/ `install_external`). The CLI's `localm plugin install <name>` (and `install
/path/to/plugin`) install headless, without loading the plugin, and never call
it - nor does it fire retroactively the first time the server loads the plugin.
A plugin whose `on_install` does real work (the bundled `voice` plugin prefetches
its Whisper model there) must not assume it ran; use `on_first_use` for anything
that has to happen regardless of install path.

### The Host API

| Method | Purpose |
|--------|---------|
| `mount_router(router)` | Mount a FastAPI `APIRouter`; every route is auto-gated by the plugin's capability scope. |
| `mount_static(directory, *, url_prefix="")` | Serve a static dir at `/plugins/<name>/` (the SPA import()s `client_entry` from here). Returns the URL prefix. |
| `add_settings(fields)` | Add fields to the plugin's settings section in the GUI (see [Settings fields](#settings-fields)). |
| `register_tab(surface)` | Register a GUI tab in the SPA. |
| `plugin_config(name=None)` | Read this (or another) plugin's config block (`config["plugins"][name]`). |
| `save_plugin_config(name, cfg)` | Write a plugin's config block, atomically (safe against a concurrent config write from another plugin, the CLI, or the HTTP API). |
| `engine()` | Handle to the inference engine. |
| `driving_engine(engine=None)` | Context manager: wrap around a real generation call to pin the engine busy and reset its idle-unload clock for the duration. Never wrap a bare `.loaded`/name check with it. |
| `on_startup(callback)` | Queue work to run once the server's event loop is up (register() runs before uvicorn creates it on a normal start). |
| `audit(event, data)` | Log a plugin event. |
| `browse_dirs(path)` | Server-side folder picker helper. |
| `register_chat_hook(phase, fn, *, priority=0)` | Register an inlet/stream/outlet transform that runs on every chat turn (see [Chat pipeline hooks](#chat-pipeline-hooks)). |
| `register_model_role(descriptor)` | Declare a model slot the plugin needs, by registry `model_type` (see [Model roles](#model-roles)). |

A routes-only plugin (`voice`) is:

```python
from fastapi import APIRouter
_router = APIRouter()

@_router.get("/api/voice/status")
async def status():
    ...

def register(host):
    host.mount_router(_router)   # routes gated by the "voice" scope

def unregister():
    pass
```

### Capability scopes

Every plugin declares a `scope` (default: its name). `mount_router` gates all of
the plugin's routes behind that scope, enforced per-request when an API key is
configured (the server is fail-open with no key). Scopes keep a plugin's reach
explicit and let a key be issued for a subset of capabilities.

## Settings fields

A plugin can contribute its own settings section without writing any GUI code:

```python
from localm.plugins.contract import PluginSettingField
from localm.settings_schema import Widget

def register(host):
    host.add_settings([
        PluginSettingField("api_key", Widget.SECRET, "API key",
                           "Used to call the upstream service.", admin_only=True),
        PluginSettingField("max_results", Widget.NUMBER, "Max results",
                           "Results returned per query.", default=5, min=1, max=50),
    ])
```

Each field lives at `config["plugins"][<name>][key]` - the same block
`plugin_config()` / `save_plugin_config()` already read and write - and is
rendered with the same per-widget control the core settings form and the
tts/media sections use (see [settings_schema.py](../localm/settings_schema.py)'s
`Widget` for the valid `widget` values). `add_settings()` validates the field
shape immediately: a non-`PluginSettingField` entry or an unknown widget raises
at `register()` time rather than silently never rendering.

The GUI reads every active plugin's fields from `GET /v1/plugins/settings` and
saves one plugin's block through `POST /v1/plugins/<name>/settings`, gated on
`config:read` / `config:write` like the core settings form - not the plugin's
own capability scope, so a key that may merely USE a plugin cannot reconfigure
it. `admin_only` fields (a secret, a script URL, anything that widens a trust
boundary) additionally require an owner (admin) key to see or set, mirroring
the media backends' `launch_cmd`/`api_url` gate. A blank/`null` save clears an
override back to the field's own `default`.

`localm plugin config <name>` reaches the same fields from the terminal, over
those same two routes. It needs a running localm to do it: your field list is
declared when the plugin loads, so a process that has not loaded it (as the CLI
deliberately never does) has nothing to list. With no server up the command says
that, rather than reporting your plugin as having no settings.

Because the field list is supplied at `register()` time rather than declared
statically, it only exists while the plugin is loaded: there is no way to
pre-configure a plugin's settings before installing/enabling it (unlike the
built-in tts block, which ships a fixed schema). This is the seam
[plugin-interop.md](plugin-interop.md) maps Open WebUI's `Valves` onto.

## Shipping client-side assets

Set `assets_dir` and `client_entry` in `[surface]`. The engine serves the dir at
`/plugins/<name>/`, and the SPA fetches `/api/plugins`, then `import()`s each
active plugin's `client_entry` module and calls its exported `register(ctx)`.
This lets a plugin add browser-side behaviour with no tab and no Python backend.

The `tts` plugin is the worked example: it ships `static/tts.js`, which loads
Kokoro neural TTS entirely in the browser and registers a speech provider via
`ctx.registerTTS(...)`. It writes nothing to the server, so it stays trace-free
in privacy mode with no gating. The client context (`ctx`) exposes
`registerTTS`, `toast`, `authHeaders`, and `voicesChanged`.

## Exporting a coder tool

A plugin can add a tool the coder agent may call. This uses a DIFFERENT
manifest form from the rest of this document, so read this section before
adding one.

Everything above describes the ENGINE contract: `[plugin] register = "..."`,
loaded by the plugin engine. Coder tool discovery does not read those. It walks
the LEGACY plugin form, which requires an `entry` instead:

```toml
[plugin]
name  = "myplugin"
entry = "myplugin.tools:setup"     # <module>:<attr>, and it is required here

[tools]
exports = ["do_the_thing"]
```

A plugin that declares `register` and no `entry` is treated as engine-owned and
skipped by that scan entirely, so **a `[tools] exports` block on an
engine-contract plugin is parsed, shown in the GUI's External plugins card, and
never registered as a coder tool.** That split is a real limitation of the
current loader, not a convention: a plugin that wants both surfaces has to ship
the legacy form.

Each name in `exports` is looked up on the imported module. The callable takes
the working directory as its first positional argument and everything else by
keyword. Three optional attributes describe it to the model:

| Attribute | Meaning |
|---|---|
| `tool_description` | One-line description; falls back to the docstring's first line |
| `tool_params` | `{param: {"type", "description", "required"}}` |
| `tool_destructive` | Whether it asks before running. **Defaults to `True`** |

`tool_destructive` defaulting to true is deliberate: a tool that forgets to
declare itself asks for confirmation rather than silently acting.

Registered names are namespaced `plugin_<plugin>_<export>`, with hyphens in the
plugin name converted to underscores, so an export cannot shadow a built-in.

One current limit worth knowing: an exported tool cannot mark its output as
untrusted. The built-in network tools are recognised by name rather than by a
flag, so a plugin tool that returns fetched content is not put through the
untrusted-content handling that `fetch_url` gets.

For the built-in tools the coder already has, see the table in
[cli.md](cli.md#built-in-tools).

## Chat pipeline hooks

A plugin can intercept and transform a chat turn server-side by registering
hooks on the kernel chat pipeline. This is the seam that wraps Open WebUI
*Filter* functions and oobabooga input/output text modifiers (see
[plugin-interop.md](plugin-interop.md)); it is also a clean way to inject
context, redact, translate, or annotate every turn.

`host.register_chat_hook(phase, fn, *, priority=0)` takes one of three phases:

| Phase | Signature | When |
|-------|-----------|------|
| `inlet` | `fn(messages, ctx) -> messages` (sync or async) | before token counting and inference; transform the message list |
| `stream` | `fn(token, ctx) -> token` (**sync only**) | per streamed text piece on the hot path; an async fn is skipped |
| `outlet` | `fn(text, messages, ctx) -> text` (sync or async) | on the final reply text |

Returning `None` keeps the prior value (so a hook may mutate in place and return
nothing). Lower `priority` runs first; ties keep registration order. A hook that
raises is logged and skipped, never breaking the turn. The engine drops a
plugin's hooks automatically when it is disabled or uninstalled.

`ctx` (`ChatHookContext`) carries `model_id`, `stream`, `request_id`, and a
mutable `state` dict shared across the inlet/stream/outlet of one request (so an
inlet can stash data its outlet reads). `principal` / `scopes` are reserved for
future per-user gating and are unset in open mode.

```python
def _inlet(messages, ctx):
    messages.insert(0, {"role": "system", "content": "Answer concisely."})
    return messages

def _outlet(text, messages, ctx):
    return text.replace("secret", "[redacted]")

def register(host):
    host.register_chat_hook("inlet", _inlet)
    host.register_chat_hook("outlet", _outlet)

def unregister():
    pass
```

Scope and limits:

- The chain runs for **every** `/v1/chat/completions` client (the GUI, raw API
  callers, and the coder agent pointed at localm). It does **not** run for the
  in-process `localm run` REPL, which calls the engine directly.
- Hooks see scrubbed content text, not model-internal control markers (the
  pipeline sits downstream of the engine's marker scrubbing).
- This is a server-side seam. It is independent of localm's existing
  client-side RAG / web injection (assembled in the SPA before the request is
  sent), which it does not replace. The memory plugin, by contrast, injects
  recalled facts through an `inlet` hook on this very chain (server-side).
- In a streaming turn, `outlet` runs after all chunks have been sent, so it only
  shapes the recorded reply (audit / transcript). Use a `stream` hook to rewrite
  streamed output live.

## Model roles

A plugin that runs a model pipeline of its own (the image, music and video
plugins each drive a ComfyUI graph) declares WHICH kinds of model it needs, so
the rest of localm can talk about them by name instead of by raw field.

```python
from localm.plugins.contract import ModelRoleDescriptor

def register(host):
    host.register_model_role(ModelRoleDescriptor(
        "image-unet", "Diffusion model (UNet)", "diffusion-unet"))
    host.register_model_role(ModelRoleDescriptor(
        "image-vae", "VAE", "vae", required=False))
```

| Field | Meaning |
|-------|---------|
| `role_id` | Stable id, unique within the plugin (e.g. `image-unet`). |
| `label` | What a user sees (e.g. `Diffusion model (UNet)`). |
| `model_type` | One of the registry's `MODEL_TYPES`. An unknown value raises. |
| `required` | Default `True`. `False` marks an optional component. |
| `description` | Optional longer text. |
| `plugin_name` | Filled in by the host; do not set it. |

`model_type` is the join: it is the same value the registry records per model
(`localm set-type`, and the Import-from-ComfyUI scan), so a role says "this slot
takes a VAE" in the same vocabulary the model library uses. Registration is
validated against `MODEL_TYPES` (an unknown one raises at `register()` time
rather than surfacing later as an empty picker). A disabled or uninstalled
plugin's roles stop being reported: the roles live on the plugin's own host, and
the engine only reads hosts that are currently loaded.

What consumes them today:

- `GET /api/models/roles` lists every declared role across active plugins.
- The media plugins' `GET /api/{imagine,music,video}/comfy-models` joins the
  declared roles to the active workflow's model slots and to the registry's own
  models of each type, so the Workflow panel's picker can label each dropdown,
  flag a required component the workflow has no slot for, and point out a model
  you have registered that ComfyUI is not offering. That join lives in
  `localm/plugins/media_roles.py`; a plugin only has to declare the roles.

Two things a role does **not** do. It does not reserve or load anything - it is
a declaration, not an allocation. And it does not constrain what a user may
pick: selection stays fitness-for-purpose, so a model whose recorded type is
imperfect is still choosable.

## Dependencies

- **`requires_extras`** - pip extras carrying heavy Python deps. Declaring
  `requires_extras = ["voice"]` means the plugin needs `pip install "localm[voice]"`.
  By default localm installs the extra for you on the host when the plugin is
  installed or enabled (gated by the `auto_install_plugin_deps` setting, and only
  ever on the local host - never triggered by a remote client). With the setting
  off, or on a remote client, the two stay separate steps: install the plugin,
  then `localm plugin install-deps [<name>|--all]` (or the GUI Plugins page's
  **Install dependencies** button). Plugins with no Python dependency
  (like `tts`, which runs in the browser) declare no extra.
- **`requires`** - other plugins that must be installed first. `missing_requires`
  surfaces these at install time.

## Per-plugin config and privacy

A plugin's settings live under `config["plugins"][<name>]`, read/written via
`host.plugin_config()` / `host.save_plugin_config()`. A plugin that writes
session-derived data to disk (sidecars, caches, generated media) must gate that
write on the privacy contract - check `effective_mode()` in
[audit.py](../localm/audit.py) before writing, exactly as the media plugins do.
Personal model/voice choices, such as the tts plugin's, belong under
`config["plugins"]["tts"]` in the gitignored `config.json`, never committed;
the tracked `tts.example.json` only supplies the shipped defaults.

A plugin block that users are meant to EDIT needs a write surface, or those
settings are hand-edit-only in practice. The two worked examples are the media
blocks (`GET /v1/media/config`, `POST /v1/media/config/{name}`) and the tts
block (`GET/POST /v1/tts/config`): both validate the update in
[settings_schema.py](../localm/settings_schema.py) and merge it into the plugin's
own block, and both are gated on `config:read` / `config:write` rather than on
the plugin's own capability, so a key that may merely USE a plugin cannot
reconfigure it. Fields that widen a trust boundary (a shell command, a network
target, a script URL loaded by every browser) additionally require an owner
(admin) key.

## Third-party plugins

Any directory with a valid `plugin.toml` can be installed as a plugin:

```
localm plugin install /path/to/my-plugin     # --force overwrites an existing install
```

`plugin install` takes either a first-party plugin NAME (from the bundled store)
or a path to a DIRECTORY containing a `plugin.toml`; a directory is validated,
copied into `<data dir>/plugins/`, and enabled, then loaded with the same contract
as a first-party plugin. Third-party plugins run unsandboxed in-process, so
install only code you trust. See [plugin-interop.md](plugin-interop.md) for the
adapter approach to wrapping extensions from other ecosystems.

## API version compatibility

`api_version` in the manifest must match the engine's `API_VERSION` (currently
`1`). The engine refuses to load a plugin built against an incompatible contract,
so bumping the contract is a deliberate, visible break.

## Before you ship a plugin (checklist)

**Enforced by the guard suite**
([tests/test_builtin_plugins_contract.py](../tests/test_builtin_plugins_contract.py),
which enumerates every builtin so a new one is covered the moment it ships):

- **Manifest conforms**: `plugin.toml` parses, targets `api_version = 1`, and
  declares a `scope` and a `register` entry.
- **`client_entry` is served**: if `[surface]` declares `client_entry`, the file
  exists under `assets_dir` and is served at `/plugins/<name>/<entry>`. The
  engine auto-mounts a surface's `assets_dir` after `register()`, so a
  client_entry plugin never has to call `mount_static` itself and can never
  silently 404.

**Reviewer judgment** (not machine-checkable):

- **Routes are scope-gated**: mount every route via `host.mount_router` (which
  applies the plugin's scope), never on the bare app. Secrets live behind `/api`
  routes, never under the public static assets. See
  [Capability scopes](#capability-scopes).
- **Runtime enable/disable**: the plugin loads and unloads with no server restart
  - `register` mounts, `unregister` tears down, and a disable removes the routes.
- **Privacy mode**: gate any session-derived disk write on `effective_mode()`.
  See [Per-plugin config and privacy](#per-plugin-config-and-privacy). A
  browser-only or read-only plugin needs no gating.
- **No published personal choices**: model ids, encoders, and workflows live in a
  tracked `*.example.json` template or user config, never hardcoded or committed
  (see `AGENTS.md`).
- **Tested**: behaviour changes are covered by a test that fails before the fix;
  `python scripts/check_hygiene.py` and `pytest -m "not integration"` pass.
  This bullet is about contributing a plugin into the localm repo itself,
  where those scripts live; an independent third-party plugin shipped from
  outside the repo (e.g. by someone who installed localm via `pip install
  localm`) has no access to them and should verify equivalently by its own
  means instead.
