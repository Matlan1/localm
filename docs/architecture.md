# Architecture

## Overview

localm is structured as a thin CLI shell over a pluggable inference backend
system, with plugins layered on top. The core design rule: the CLI knows
nothing about inference; inference knows nothing about CLI. They communicate
through `Engine`.

```
CLI (cli.py)                       Plugins (localm/plugins/)
  └── Engine (inference/engine.py)   ├── coder/      AI coding agent
        ├── GgufBackend              ├── gui/        web GUI (FastAPI routes + static)
        │     └── LlamaCpp (ctypes)  ├── mcpserver/  `localm mcp` stdio server
        └── HFBackend                └── loader.py   external plugin discovery
              └── HF Transformers
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

`inference/http_server.py` is a FastAPI app (`create_app(engine)`). The
synchronous `engine.chat_stream()` runs in a thread; tokens cross into the
event loop via `call_soon_threadsafe` and stream out as SSE. One asyncio
semaphore serialises all inference. Endpoints are documented in
[server-api.md](server-api.md).

## Conversation compaction

`inference/compact.py` summarises older chat turns through the model when a
conversation reaches 70% of the context ceiling, keeping the system prompt
and the last two exchanges verbatim, with a hard-trim fallback that never
raises. Used by `localm run` interactive chat; the GUI implements the same
protocol client-side. The coder agent has its own compaction in
`plugins/coder/agent.py` (GBNF-structured summaries).

## Plugins

Built-in plugins live in `localm/plugins/` (coder, gui, mcpserver).
External plugins are folders under `~/.localm/plugins/` with a
`plugin.toml` manifest; `loader.py` discovers them at CLI start, mounts
their Click command, and registers exported agent tools. A broken plugin
warns and is skipped, never crashing the CLI.

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
