# Architecture

## Overview

localm core is a model loader plus a plugin engine. The CLI is a thin shell
over a pluggable inference backend, and everything above bare chat is a
plugin. The core design rule: the CLI knows nothing about inference;
inference knows nothing about CLI. They communicate through `Engine`. CHAT
is the protected, preinstalled plugin (#0); coder, image, music, video, rag,
web, voice (Whisper STT), tts (Kokoro in-browser TTS), jobs (scheduled tasks),
and mcp are all
plugins layered on top.

```
CLI (localm/cli/)                  Plugin engine (localm/plugins/)
  └── Engine (inference/engine.py)   ├── engine.py    PluginManager
        ├── GgufBackend              ├── contract.py  Host / Surface / PluginSpec
        │     └── LlamaCpp (ctypes)  ├── catalog.py   first-party catalog
        └── HFBackend                ├── builtin/     store (read-only)
              └── HF Transformers     └── ~/.localm/plugins/  installed
```

## Engine

`Engine` auto-detects the backend from the model path (`.gguf` file or
Ollama `sha256-*` blob to `GgufBackend`, directory with `config.json` to
`HFBackend`), exposes `chat_stream(messages, ...)`, `embed(texts)`, and
`count_tokens(text)`, and reloads the backend transparently when something
unloaded it (e.g. image generation borrowing the VRAM).

## GgufBackend and the ctypes binding

`GgufBackend` wraps `LlamaCpp`, a pure-ctypes binding to the native
`llama.dll` (no llama-cpp-python). Key behaviour:

- **Dynamic context window**: contexts start at `n_ctx` and are rebuilt
  larger in `n_ctx_grow` steps, capped at `n_ctx_max`; `ctx_auto` sizes the
  cap from free VRAM at load. Prefill is always chunked to `n_batch`.
- **KV prefix reuse**: between calls the common token prefix with the
  previous request stays in the KV cache (llama_memory_* API, probed at
  runtime); only the new suffix is prefilled.
- **Sampler chain**: grammar (GBNF), repetition penalty, top-k, top-p,
  min-p, temperature, dist; greedy when temperature is 0.
- **Output filtering**: a stop-string filter handles end-of-turn sequences
  split across tokens; an internal-marker scrubber strips thinking-channel
  tags some finetunes emit as text (bypassed in debug mode).
- **VRAM pre-flight**: free VRAM is checked against model size before load,
  with a warning rather than a block.
- No fallback by design: if the DLL cannot be loaded, `load()` raises a clear
  error pointing at `localm setup-llama` rather than degrading to a slower,
  lower-fidelity `llama-cli.exe` subprocess.

## HTTP server

`inference/http_server.py` builds the FastAPI app (`create_app(engine)`) and
holds the shared inference state; the route handlers themselves live in
`inference/routes/` modules (chat, models, config, keys, session, admin,
system, plugins). The synchronous `engine.chat_stream()` runs in a thread;
tokens cross into the event loop via `call_soon_threadsafe` and stream out as
SSE. One asyncio semaphore serialises all inference. Endpoints are documented
in [server-api.md](server-api.md).

## Conversation compaction

`inference/compact.py` summarises older chat turns through the model when a
conversation reaches 70% of the context ceiling, keeping the system prompt
and the last two exchanges verbatim, with a hard-trim fallback that never
raises. Used by `localm run` interactive chat; the GUI (itself a plugin
surface now) implements the same protocol client-side. The coder agent has
its own compaction in the `localm/plugins/coder/agent/` package (GBNF-structured
summaries); coder is the builtin `coder` plugin (store dir
`localm/plugins/builtin/coder/` wrapping `localm/plugins/coder/`, which is split
into `agent/`, `tools/`, `cli/`, and `backends/` packages).

## Plugin engine

Everything above bare chat is a plugin, managed by `PluginManager` in
`localm/plugins/engine.py`. A plugin ships a `plugin.toml` manifest
(`[plugin]` + `[surface]` tables) plus a module exporting `register(host)`
and `unregister()`.

**Two locations.** The *store* is `localm/plugins/builtin/` (the bundled,
read-only first-party plugins: coder, image, music, video, rag, web, voice,
tts, jobs, mcp). *Installed* plugins live under `~/.localm/plugins/`. Installing
copies a plugin from the store into the installed location.

**Four states.** A plugin is *installed* (present under `~/.localm/plugins/`),
*enabled* (listed in `config["plugins_enabled"]`), and *active* only when it
is both installed AND enabled. The store also tracks what is *available* (in
the catalog but not yet installed). By default only chat is active.

**Chat is plugin #0.** CHAT is protected and preinstalled; it cannot be
disabled or uninstalled.

**Contract.** `localm/plugins/contract.py` defines the protocols a plugin
sees: `Surface` and `PluginSpec` (manifest shape) and `Host` (the API the
engine hands to `register`). The host exposes `mount_router`, `mount_static`,
`add_settings`, `register_tab`, `plugin_config`, `save_plugin_config`,
`engine`, `audit`, and `browse_dirs`. `catalog.py` holds the static
first-party catalog; `loader.py` is the legacy external/CLI loader;
`media_config.py` resolves shared media-plugin config.

**Lifecycle.** `PluginManager` discovers installed plugins, then
`load_enabled` mounts each active plugin at runtime through `PluginHost`
(`mount_router` for FastAPI routes, `mount_static` for assets). Enable and
disable toggle the config entry; install copies store -> installed (and
enables), uninstall removes the installed directory. Mounting and unmounting
happen at runtime without restarting the server.

**api_version gating.** Each manifest declares the contract version it
targets; the engine refuses to load a plugin whose `api_version` it does not
support, surfacing an error rather than crashing.

**Capability scopes.** Every plugin declares a capability scope, and its
HTTP routes are gated to that scope, so a plugin cannot reach beyond the
permissions it asked for.

See [plugins.md](plugins.md) for the full authoring guide. A broken plugin
warns and is skipped, never crashing the host.

## Debug mode

`debuglog.py` implements `--debug`: a shared log file under
`~/.localm/logs/` (path carried in `LOCALM_DEBUG` so child processes append
to the same file), HTTP request timing, and redirection of the native
llama.cpp stderr into the log so crash abort reasons are captured instead
of suppressed.

## Protocol

`inference/protocol.py` defines Pydantic v2 models for the OpenAI wire
format: `ChatRequest`, `CompletionRequest`, `EmbeddingRequest`, `Message`
(string or multipart content), `ChatChunk`/`ChatResponse`, and `UsageInfo`
including `ttft_ms` and `tokens_per_sec`.
