# Architecture

## Overview

localm core is a model loader plus a plugin engine. The CLI is a thin shell
over a pluggable inference backend, and everything above bare chat is a
plugin. The core design rule: the CLI knows nothing about inference;
inference knows nothing about CLI. They communicate through `Engine`, with
a couple of narrow, deliberate exceptions (`localm/cli/doctor.py` imports
the llama.cpp loader directly for GPU-probe diagnostics; `localm/cli/chat.py`
catches specific backend exception types). CHAT
is the protected, preinstalled plugin (#0); coder, browser, image, music,
video, rag, web, memory, voice (Whisper STT), tts (Kokoro in-browser TTS),
jobs (scheduled tasks), and mcp are all plugins layered on top.

```
CLI (localm/cli/)                  Plugin engine (localm/plugins/)
  └── Engine (inference/engine.py)   ├── engine.py    PluginManager
        ├── GgufBackend              ├── contract.py  Host / Surface / PluginSpec
        │     └── LlamaCpp (ctypes)  ├── catalog.py   first-party catalog
        └── HFBackend                ├── builtin/     store (read-only)
              └── HF Transformers     └── <data dir>/plugins/  installed
                    (+ native AWQ)
```

## Engine

`Engine` auto-detects the backend from the model path (`.gguf` file or
Ollama `sha256-*` blob to `GgufBackend`, directory with `config.json` to
`HFBackend`), exposes `chat_stream(messages, ...)`, `embed(texts)`, and
`count_tokens(text)`, and reloads the backend transparently when something
unloaded it (e.g. image generation borrowing the VRAM).

## Model loading isolation

Both backends run the real model - and every native call it makes - inside a
disposable child process, never in the server process itself:

- **GGUF** (`llamacpp/_worker.py`'s `GgufWorker`, spawned by `_runner.py`):
  `llama_load_model_from_file` and every later native call (context growth,
  token-by-token decode) can hard-abort the whole process on a native
  CUDA/HIP driver failure, which no Python `try`/`except` can catch. The
  model's entire lifecycle - load, generate, grammar-check, unload - runs in
  the child, so a native abort kills only that child; the parent reports it
  as a clean, catchable error and reloads fresh on the next request.
- **HF Transformers** (`_hf_worker.py`'s `HFWorker`, spawned by
  `_hf_runner.py`): the tokenizer, `model.generate()` and a torch forward
  pass are equally uninterruptible from Python, so a hang there would
  otherwise burn a slot in the server's shared thread pool permanently.
  Isolating it is what makes such a hang killable without restarting the
  server.

`GgufBackend`/`HFBackend` are thin parent-side proxies. `GgufBackend` runs a
preflight VRAM check in the parent before a child is even spawned, so a load
that can never fit fails fast without paying a process-spawn cost.
`HFBackend` has no such preflight; it does two other parent-side checks
(custom-code trust, tokenizer regex safety) and leaves VRAM budgeting to the
spawned child, during `device_map` construction.

## GgufBackend and the ctypes binding

`GgufBackend` wraps `LlamaCpp`, a pure-ctypes binding to the native llama.cpp
library (`llama.dll` on Windows, `libllama.so` on Linux, `libllama.dylib` on
macOS; no llama-cpp-python). Key behaviour:

- **Dynamic context window**: contexts start at `n_ctx` and are rebuilt
  larger in `n_ctx_grow` steps, capped at `n_ctx_max`; `ctx_auto` sizes the
  cap from free VRAM at load. Prefill is always chunked to `n_batch`.
- **KV prefix reuse**: between calls the common token prefix with the
  previous request stays in the KV cache (llama_memory_* API, probed at
  runtime); only the new suffix is prefilled.
- **Sampler chain**: grammar (GBNF), repetition penalty, top-k, top-p,
  min-p, temperature, dist; greedy when temperature is 0.
- **Multi-Token Prediction (MTP) speculative decoding**: a model trained
  with its own next-n draft head can draft and verify two tokens per step
  through a dedicated draft context, instead of a separate draft model. Off
  by default (`mtp_enabled`) and engages only where the runtime can build
  and feed a real draft head for that model; see
  [llamacpp-binding.md](llamacpp-binding.md) for the mechanism.
- **Output filtering**: a stop-string filter handles end-of-turn sequences
  split across tokens; an internal-marker scrubber (`localm/textnorm.py`)
  strips thinking-channel tags and leaked chat-template control tokens some
  finetunes emit as text. Chat output is always scrubbed; debug mode
  additionally logs the raw, unscrubbed text.
- **VRAM pre-flight**: free VRAM is checked against model size before load.
  If the model could never fit even on an empty card, load is refused
  outright; if something else is merely using the VRAM, it is a warning and
  the load continues.
- No fallback by design: if the DLL cannot be loaded, `load()` raises a clear
  error pointing at `localm setup-llama`.

## HFBackend and native AWQ

`HFBackend` drives HuggingFace `transformers` (`AutoModelForCausalLM` /
`AutoProcessor`) for model directories `GgufBackend` cannot load - anything
without a `.gguf` file, including multimodal checkpoints such as
Gemma4UnifiedForConditionalGeneration (text + image + audio). GPU use goes
through `torch.cuda`, which maps to ROCm on AMD systems running PyTorch+ROCm.

`localm/inference/backends/awq.py` registers a native AWQ (Activation-aware
Weight Quantization) quantizer into `transformers`' own
`AUTO_QUANTIZER_MAPPING` the first time the HF worker imports
`transformers`. A HuggingFace AWQ 4-bit checkpoint (`quantization_config`
naming `"awq"` in its `config.json`) then loads and runs through the normal
`from_pretrained` path with no extra flag - `NativeAWQLinear` dequantizes
each packed 4-bit layer on the fly during the forward pass. This works
across Windows ROCm, Linux ROCm, NVIDIA CUDA, Intel XPU and CPU without
external compiled dependencies (gptqmodel, autoawq, torchao), which either
do not build on Windows ROCm or are not installed by default. Multimodal and
hybrid-attention layers that AWQ checkpoints leave unquantized (vision
towers, projectors, some hybrid-attention sublayers) are skipped and kept in
their original precision.

## HTTP server

`inference/http_server.py` builds the FastAPI app (`create_app(engine)`) and
holds the shared inference state; the route handlers themselves live in
`inference/routes/` modules (chat, models, config, keys, session, admin,
system, gpu). The plugin management API is mounted separately, by
`localm/plugins/engine.py`. The synchronous `engine.chat_stream()` runs in a
thread; tokens cross into the event loop via `call_soon_threadsafe` and
stream out as SSE. Inference is serialised per loaded model (an asyncio
semaphore per display name), not globally - two concurrently loaded models
can generate at once. Endpoints are documented in
[server-api.md](server-api.md).

`inference/capability_routing.py` decides, for a chat request that did not
pin a model by name, whether the loaded model can actually serve it (vision,
tool use, reasoning, context length) and picks an installed one that can when
it cannot; `http_server.plan_capability_route()` builds the request's
`CapabilityNeeds` and `routes/chat.py` applies the decision. `peer_routing.py`
is a separate, unrelated mechanism: it forwards a chat request straight to
another localm instance on this machine that already has the model loaded,
after verifying the forward target resolves to loopback.

## Conversation compaction

`inference/compact.py` summarises older chat turns through the model when a
conversation reaches 70% of the context ceiling, keeping the system prompt
and the last two exchanges verbatim, with a hard-trim fallback that never
raises. Used by `localm run` interactive chat; the GUI (itself a plugin
surface now) implements the same protocol client-side. The coder agent has
its own GBNF-structured compaction in the `localm/plugins/coder/agent/` package.

## Plugin engine

Everything above bare chat is a plugin, managed by `PluginManager` in
`localm/plugins/engine.py`. A plugin ships a `plugin.toml` manifest
(`[plugin]` + `[surface]` tables) plus a module exporting `register(host)`
and `unregister()`.

**Two locations.** The *store* is `localm/plugins/builtin/` (the bundled,
read-only first-party plugins listed above). *Installed* plugins live under
`<data dir>/plugins/`. Installing copies a plugin from the store into the
installed location.

**States.** A plugin is *active* only when it is both installed (present under
`<data dir>/plugins/`) and enabled (listed in `config["plugins_enabled"]`); the
store also tracks what is *available* (in the catalog but not yet installed).
By default only chat is active. See [plugins.md](plugins.md) for the full model.

**Chat is plugin #0.** CHAT is protected and preinstalled; it cannot be
disabled or uninstalled.

**Contract.** `localm/plugins/contract.py` defines the protocols a plugin
sees: `Surface` and `PluginSpec` (manifest shape) and `Host` (the API the
engine hands to `register`). The host exposes `mount_router`, `mount_static`,
`add_settings`, `register_tab`, `plugin_config`, `save_plugin_config`,
`engine`, `audit`, and `browse_dirs`. `catalog.py` holds the static
first-party catalog.

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
`<data dir>/logs/` (path carried in `LOCALM_DEBUG` so child processes append
to the same file), HTTP request timing, and redirection of the native
llama.cpp stderr into the log so crash abort reasons are captured instead
of suppressed.

## Protocol

`inference/protocol.py` defines Pydantic v2 models for the OpenAI wire
format: `ChatRequest`, `CompletionRequest`, `EmbeddingRequest`, `Message`
(string or multipart content), `ChatChunk`/`ChatResponse`, and `UsageInfo`
including `ttft_ms` and `tokens_per_sec`.
