# Plugins

localm is plugin-first. The core is a model loader plus the plugin engine; the
only always-present feature is **chat**, which itself ships as the protected,
preinstalled plugin #0. Everything else - the coder agent, image/music/video
generation, RAG (Knowledge), web access, voice (Whisper STT), text-to-speech
(Kokoro), and the MCP server - is a plugin you install.

This guide covers the plugin lifecycle, the `plugin.toml` manifest, the
`register(host)` contract, the Host API, and how to ship server routes, a GUI
tab, settings, and client-side assets. For wrapping extensions from other
ecosystems (Open WebUI, oobabooga, Anthropic Skills), see
[plugin-interop.md](plugin-interop.md).

## Concept: store, installed, enabled, active

A plugin moves along two independent axes:

| Term | Meaning |
|------|---------|
| **Available** (store) | Bundled first-party plugins live in `localm/plugins/builtin/` (the read-only "store"). They are NOT loaded from there. The static catalog (`localm/plugins/catalog.py`) is the core's only knowledge of plugins it has not installed. |
| **Installed** | Physically present on disk in the installed folder (`~/.localm/plugins/<name>/` with a `plugin.toml`). Installing copies a plugin from the store into the installed folder. "Installed" is disk presence, not a config flag. |
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
```

A running GUI server picks up newly enabled HTTP plugins on its next start;
toggling a plugin while the server runs (via the GUI Plugins page or the
`/api/plugins/{name}/...` routes) mounts/unmounts it without a restart.

Some plugins need heavy Python dependencies shipped as a pip extra (see
[Dependencies](#dependencies)). Installing the plugin selects it; installing the
extra provides its libraries - they are separate steps.

## Anatomy of a plugin

A plugin is a directory with a `plugin.toml` manifest and a Python module that
exposes `register(host)`:

```
my-plugin/
  plugin.toml
  plug.py            # exposes register(host) [and optional unregister()]
  static/            # optional client-side assets (served at /plugins/<name>/)
```

The bundled `voice` plugin (routes only) and `tts` plugin (client-side assets)
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
capabilities = ["feature"]   # declared capabilities (shown at install)
data_subdir = "my_data"      # storage under the data dir (~/.localm/my_data); "" = none
protected = false            # cannot be disabled/uninstalled (chat only)
default_enabled = false      # active on first run (chat only)
cli = "module:attr"          # optional legacy Click command entry point

[surface]                    # optional GUI/SPA contribution
tab_id = "mytab"             # "" = no tab (settings-only or headless plugin)
label = "My Plugin"
icon = "gear"                # emoji or static-asset name
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

# Optional lifecycle hooks, called only if present:
def on_install() -> None: ...
def on_first_use() -> None: ...
def on_uninstall(delete_data: bool = False) -> None: ...
```

The engine hands `register` a `PluginHost`; everything the plugin attaches
through it is tracked and removed again on `unregister`, so enabling/disabling is
instant and clean.

### The Host API

| Method | Purpose |
|--------|---------|
| `mount_router(router)` | Mount a FastAPI `APIRouter`; every route is auto-gated by the plugin's capability scope. |
| `mount_static(directory, *, url_prefix="")` | Serve a static dir at `/plugins/<name>/` (the SPA import()s `client_entry` from here). Returns the URL prefix. |
| `add_settings(fields)` | Add fields to the plugin's settings section in the GUI. |
| `register_tab(surface)` | Register a GUI tab in the SPA. |
| `plugin_config(name=None)` | Read this (or another) plugin's config block (`config["plugins"][name]`). |
| `save_plugin_config(name, cfg)` | Write a plugin's config atomically. |
| `engine()` | Handle to the inference engine. |
| `audit(event, data)` | Log a plugin event. |
| `browse_dirs(path)` | Server-side folder picker helper. |
| `register_chat_hook(phase, fn, *, priority=0)` | Register an inlet/stream/outlet transform that runs on every chat turn (see [Chat pipeline hooks](#chat-pipeline-hooks)). |

A routes-only plugin (`voice`) is just:

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
  client-side RAG / memory / web injection (assembled in the SPA before the
  request is sent), which it does not replace.
- In a streaming turn, `inlet` and `stream` affect what the client receives, but
  `outlet` runs after all chunks have been sent, so it only shapes the recorded
  reply (audit / transcript / side-effects). Use a `stream` hook to rewrite
  streamed output live.

## Dependencies

- **`requires_extras`** - pip extras carrying heavy Python deps. Declaring
  `requires_extras = ["voice"]` means the plugin needs `pip install "localm[voice]"`.
  Installing the plugin does NOT auto-install the extra; the two are separate,
  consent-gated steps. Plugins with no Python dependency (like `tts`, which runs
  in the browser) declare no extra.
- **`requires`** - other plugins that must be installed first. `missing_requires`
  surfaces these at install time.

## Per-plugin config and privacy

A plugin's settings live under `config["plugins"][<name>]`, read/written via
`host.plugin_config()` / `host.save_plugin_config()`. A plugin that writes
session-derived data to disk (sidecars, caches, generated media) must gate that
write on the privacy contract - check `effective_mode()` in
[audit.py](../localm/audit.py) before writing, exactly as the media plugins do.
Personal model/voice choices belong in a gitignored override (e.g. `tts.json`
overriding the tracked `tts.example.json`), never committed.

## Third-party plugins

Any directory with a valid `plugin.toml` can be installed as a plugin:

```
localm plugin install /path/to/my-plugin     # --force overwrites an existing install
```

`plugin install` takes either a first-party plugin NAME (from the bundled store)
or a path to a DIRECTORY containing a `plugin.toml`; a directory is validated,
copied into `~/.localm/plugins/`, and enabled, then loaded with the same contract
as a first-party plugin. Third-party plugins run unsandboxed in-process, so
install only code you trust. See [plugin-interop.md](plugin-interop.md) for the
adapter approach to wrapping extensions from other ecosystems.

## API version compatibility

`api_version` in the manifest must match the engine's `API_VERSION` (currently
`1`). The engine refuses to load a plugin built against an incompatible contract,
so bumping the contract is a deliberate, visible break.
