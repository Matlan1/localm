# Architecture

## Overview

localm is structured as a thin CLI shell over a pluggable inference backend system. The core design rule is: the CLI knows nothing about inference; inference knows nothing about CLI. They communicate through `Engine`.

```
CLI (cli.py)
  └── Engine (inference/engine.py)
        ├── GgufBackend (inference/backends/gguf.py)
        │     └── LlamaCpp (inference/backends/llamacpp/)
        └── HFBackend   (inference/backends/hf.py)
              └── HuggingFace Transformers
```

## Engine

`Engine` is a context manager that:

1. Auto-detects the backend from the model path:
   - `.gguf` file → `GgufBackend`
   - Directory with `config.json` → `HFBackend`
   - Ollama blob (`sha256-*` filename) → `GgufBackend`
2. Exposes a single method: `chat_stream(messages, ...)` → `Iterator[str]`
3. Holds `display_name` for UI (registered alias or resolved Ollama name)

## Backend ABC

```python
class BaseBackend:
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def chat_stream(self, messages, *, max_tokens, temperature, ...) -> Iterator[str]: ...
    @property
    def loaded(self) -> bool: ...
```

Backends are responsible for their own resource management. `GgufBackend` holds the `LlamaCpp` instance (which owns `llama_model` + `llama_context` handles); `HFBackend` holds the `transformers` pipeline.

## GgufBackend

Load sequence:
1. `_load_native()` — call `load_lib()` to load the DLL chain, then instantiate `LlamaCpp`
2. On failure → fall back to spawning `llama-cli.exe` as a subprocess (slow but always works)

`chat_stream()` delegates to `LlamaCpp.create_chat_completion(stream=True)`, consuming each streaming chunk.

## HTTP Server

`inference/http_server.py` is a FastAPI app created by `create_app(engine)`. The engine is loaded before the server starts (in `cli.py:serve`) so the first request doesn't pay the model-load cost.

Non-blocking generation: the synchronous `engine.chat_stream()` runs in a `threading.Thread`; tokens are put into an `asyncio.Queue` via `loop.call_soon_threadsafe`, then consumed by the async SSE generator. This keeps FastAPI's event loop free.

## Protocol

`inference/protocol.py` defines Pydantic v2 models for the OpenAI wire format:

- `ChatRequest` — request body (`model`, `messages`, `stream`, sampling params)
- `Message` — role + content (str or multipart list)
- `ChatChunk` / `ChatResponse` — streaming and non-streaming response bodies
- `StreamChoice`, `FullChoice`, `ChoiceDelta` — choice wrappers
