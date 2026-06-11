# localm

**Run local LLMs offline.** GGUF models via a pure-Python ctypes binding to `llama.dll`, HuggingFace Transformers models, an OpenAI-compatible HTTP server, a browser GUI, an AI coding agent, and MCP support in both directions. One CLI, no cloud required.

```
localm run mymodel --prompt "Explain RDNA2 in one sentence."
localm gui
localm coder "add type hints to utils.py"
localm serve mymodel
```

Everything that does not strictly need the internet works fully offline. Online providers (OpenAI, Anthropic) exist as explicit opt-ins for the coder agent and are never a default.

---

## Features

| Feature | Details |
|---|---|
| **GGUF inference** | Pure-Python ctypes wrapper around `llama.dll`, no llama-cpp-python required |
| **GPU support** | AMD (ROCm / HIP), NVIDIA (CUDA), CPU. Auto-detected from DLL loading order |
| **HF Transformers** | Full HuggingFace model directories |
| **OpenAI-compatible server** | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, streaming SSE, TTFT and tok/s in usage |
| **Web GUI** | `localm gui`: chat, coder agent, model manager, image generation, plugins, and settings in the browser; zero build step, fully offline ([guide](docs/gui.md)) |
| **Coding agent** | `localm coder` / `localcoder`: agentic loop with file, shell, search, test, and image tools |
| **MCP client** | The coder consumes external MCP tool servers from `.localcoder/config.toml` |
| **MCP server** | `localm mcp` exposes your local models to Claude Desktop and other MCP clients ([guide](docs/mcp.md)) |
| **Interactive chat** | Multi-turn shell with `/imagine`, `/compact`, `/clear`, `/image`, `/system`, `/save` |
| **Model registry** | Pull from HuggingFace (split GGUF supported), aliases, SHA256 dedup, tab completion |
| **Image generation** | `generate_image` tool drives a local ComfyUI FLUX pipeline with VRAM handover |
| **Plugins** | Drop a folder with `plugin.toml` into `~/.localm/plugins/` to add CLI commands and agent tools |
| **Multimodal** | Image attachment via `--image` or `/image` (requires mmproj GGUF) |
| **Ollama interop** | Register Ollama blobs directly via `localm add <manifest-dir>` |

---

## Requirements

- Python 3.10+
- For GGUF GPU inference: a compiled `llama.dll` + GPU runtime DLLs
  - AMD: ROCm `ggml-hip.dll`
  - NVIDIA: CUDA `ggml-cuda.dll`
  - CPU: only `llama.dll` + `ggml*.dll`

---

## Install

```bash
# Recommended: install with uv as a tool (isolated, always-on PATH)
uv tool install -e .

# Or with pip in a virtualenv
pip install -e .
```

For HuggingFace Transformers models (full-precision, multimodal):
```bash
uv tool install -e ".[gpu]"    # AMD ROCm
```

On Windows, double-click `localm-launcher.bat` in the repo root for a graphical
launcher: pick Web GUI, terminal chat, API server, or the coder agent, choose a
model, toggle debug mode, set port/context/GPU layers, and hit Launch. Your
choices are remembered. (`localm.bat` still starts a plain chat directly.)

---

## Quick Start

### Pull a model

```bash
# Specific GGUF from any HF repo (split files are handled automatically)
localm pull owner/repo:model-Q4_K_M.gguf

# Full HuggingFace model directory (transformers format)
localm pull owner/model-name

# Direct URL, with optional integrity check
localm pull https://example.com/m.gguf --sha256 <hash>
```

Duplicate downloads are detected by path and SHA256. When you add or pull something already registered, localm offers alias / copy / move / skip instead of silently duplicating gigabytes.

### Register an existing model

```bash
localm add C:\models\mymodel.gguf
localm add D:\ollama\manifests\registry.ollama.ai\library\<model>\<tag>
localm add D:\models\my-hf-model --name mymodel
localm alias mymodel short                # second name for the same file
```

### Run inference

```bash
localm run mymodel --prompt "What is 42?"
echo "Translate 'hello' to Japanese." | localm run mymodel
localm run mymodel                        # interactive chat
localm run mymodel --system "You are terse." --prompt "How does TCP work?"
localm run mymodel --prompt "Describe this." --image photo.jpg --mmproj mmproj.gguf
```

### Open the GUI

```bash
localm gui                # picks the first registered model, opens your browser
localm gui mymodel        # or name one
```

Chat, the coder agent, model management, and image generation in one page.
The model preloads in the background so the first reply is fast, and typing
`/` in any composer opens a command menu (`/imagine` generates images
inline). See [docs/gui.md](docs/gui.md).

### Start the inference server

```bash
localm serve mymodel
```

localm owns the port range 8642-8741: the default is 8642 and the server bumps to the next free port automatically when it is taken. This range deliberately avoids ComfyUI (8188), A1111 (7860), and the usual dev-server ports.

```bash
curl http://localhost:8642/health
curl http://localhost:8642/v1/models
curl -X POST http://localhost:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mymodel", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

Use it with any OpenAI client:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8642/v1", api_key="localm")
resp = client.chat.completions.create(
    model="mymodel",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The final usage block includes `ttft_ms` and `tokens_per_sec`. `/v1/embeddings` and `/v1/completions` are also available, and per-request `seed` makes sampling reproducible.

### Run the coding agent

```bash
localm coder --model mymodel              # interactive session in the current repo
localm coder "fix the failing test"       # single task
localcoder --model mymodel                # same thing, standalone command
```

The agent auto-starts `localm serve` when needed, plans with tool calls (read, write, edit, patch, shell, search, tests, image generation), asks before destructive actions, tracks a turn budget so it asks for help instead of guessing forever, and verifies its own code changes before answering. Privacy mode is the default: nothing is persisted unless you opt into `--mode log` or `--mode full`.

### Serve your models over MCP

```bash
localm mcp --print-config     # JSON block for Claude Desktop and friends
```

See [docs/mcp.md](docs/mcp.md) for both directions: localm as an MCP server, and the coder consuming external MCP tool servers.

---

## CLI Reference

### Core commands

```bash
localm run MODEL [opts]          # chat or single prompt
localm gui [MODEL] [opts]        # browser GUI (chat + coder)
localm serve MODEL [opts]        # OpenAI-compatible server
localm coder [TASK] [opts]       # AI coding agent
localm coder --estimate "task"   # planning turn only: approach + effort, no execution
localm mcp [opts]                # MCP stdio server
localm benchmark MODEL           # TTFT and tok/s at increasing prompt sizes
```

`--debug` on `gui`, `serve`, and `run` writes a log to `~/.localm/logs/` with
request timing and the native llama.cpp stderr stream (including crash abort
reasons), and shows raw model output without marker scrubbing.

### Model management

```bash
localm pull owner/repo:file.gguf     # specific GGUF (multi-part handled)
localm pull owner/repo               # full HF model directory
localm pull https://...gguf          # direct URL (--sha256 optional)
localm add /path/to/model.gguf       # register local file
localm alias MODEL NEWNAME           # add a second name
localm list                          # registered models
localm models                        # available shortcuts
localm rm MODEL [--yes]              # alias-aware removal
localm info                          # paths + current config
```

`localm rm` only deletes the file when the last alias pointing at it is removed, and the confirmation prompt states exactly what will happen.

### Configuration

```bash
localm config temperature 0.7
localm config n_gpu_layers 99
localm config n_ctx 8192
localm config port 8650
localm config confirm_remove false
localm config comfy_launch_cmd "D:\path\to\comfyui.bat"   # auto-start ComfyUI for image generation
```

### Dynamic context window

The context window starts at `n_ctx` (default 4096) and grows automatically
when a conversation outgrows it, in `n_ctx_grow` steps (default 4096), up to
`n_ctx_max` (default 16384). Small windows load fast; long chats get room
when they need it; the ceiling keeps VRAM use predictable.

```bash
localm config n_ctx_max 32768    # raise the ceiling
localm config n_ctx_grow 8192    # grow in bigger steps (fewer rebuilds)
localm config ctx_auto true      # derive the ceiling from free VRAM at load
```

With `ctx_auto`, localm measures free VRAM at load time, subtracts the model
weights and a fixed overhead, and sizes the ceiling from what remains. When a
conversation reaches the ceiling, replies shorten to fit; when even that is
impossible you get a clear error instead of an out-of-memory crash. An
explicit `-c/--ctx` larger than the ceiling always wins.

Long chats compact automatically before they collide with the ceiling: at 70%
fill, older turns are summarised by the model and replaced with a short
summary, keeping the last two exchanges verbatim. If summarisation is
unavailable the history is trimmed with a visible note instead: chat keeps
working either way. This applies to both `localm run` and the GUI; both also
have a manual trigger (`/compact` in the terminal, the compact button in the
browser).

Config lives at `~/.localm/config.json`. Set `LOCALM_API_KEY` to require bearer auth on the HTTP API (recommended before binding to anything other than 127.0.0.1; the CLI warns you about exposed unauthenticated binds). CORS is locked to localhost by default and can be widened with the `cors_origins` config key.

### Shell completion

```bash
localm completion powershell   # also: bash, zsh, fish
```

Model names complete everywhere a model argument is expected.

### Plugins

```bash
localm plugin list
localm plugin install /path/to/plugin-folder
localm plugin remove NAME
```

A plugin is a folder with a `plugin.toml` manifest and Python files. It can add a CLI command (`localm <name>`) and export tools into the coder agent. Installation is a local directory copy, fully offline.

---

## GPU Setup (AMD)

localm auto-detects the GPU DLL directory by scanning:

1. `LLAMA_CPP_LIB` environment variable (explicit path to `llama.dll`)
2. `D:\projects\llama-gfx1030-prebuilt\` (default gfx1030 build location)
3. `D:\projects\llama.cpp\build\bin\`

The DLL loading order is: `ggml.dll` → `ggml-base.dll` → `ggml-cpu.dll` → `ggml-hip.dll` → `llama.dll`.

Before loading a model, localm checks free VRAM against the model size and warns when it will not fit, instead of crashing mid-load. KV cache prefix reuse keeps multi-turn chat fast by only prefilling the new suffix of the conversation.

To use a custom build:
```bash
set LLAMA_CPP_LIB=C:\path\to\llama.dll
localm run mymodel --prompt "..."
```

---

## Architecture

```
localm/
├── cli.py                    # Click commands
├── config.py                 # ~/.localm/ paths, config, port range
├── model_manager.py          # registry, pull, dedup, aliases, Ollama manifests
├── image_gen/
│   └── comfy.py              # ComfyUI FLUX pipeline driver
├── inference/
│   ├── engine.py             # unified Engine: GGUF vs HF detection
│   ├── http_server.py        # FastAPI app: /v1/* endpoints
│   ├── protocol.py           # Pydantic models (OpenAI wire format)
│   └── backends/
│       ├── gguf.py           # GgufBackend + VRAM pre-flight
│       ├── hf.py             # HFBackend (Transformers)
│       └── llamacpp/         # pure-Python ctypes llama.cpp binding
└── plugins/
    ├── loader.py             # external plugin discovery (~/.localm/plugins/)
    ├── coder/                # the coding agent (agent loop, tools, MCP client)
    ├── gui/                  # web GUI (FastAPI routes + static frontend)
    └── mcpserver/            # `localm mcp` stdio server
```

### The ctypes llama.cpp binding

`localm.inference.backends.llamacpp` is a zero-dependency Python wrapper around the native `llama.dll`. It replaces `llama-cpp-python` entirely, meaning:

- **No C compiler** needed at install time
- **No Python wheel** tied to a specific Python/CUDA version
- Any prebuilt `llama.dll` works: Ollama's DLL, a custom build, any binary
- The struct layouts in `_structs.py` were derived by probing `llama_model_default_params()` / `llama_context_default_params()` against known default values and cross-referenced with `llama.h`

The generation loop (`LlamaCpp._generate`) implements the full sampler chain:
`top_k → top_p → min_p → temperature → dist (random draw)`, or `greedy` when `temperature=0`.

Stop strings (`<|im_end|>`, `<end_of_turn>`, etc.) are filtered via a streaming buffer that watches for multi-token sequences across piece boundaries.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/gui.md](docs/gui.md) | The web GUI: chat, coder sessions, approvals, security notes |
| [docs/mcp.md](docs/mcp.md) | MCP in both directions: serving models, consuming tool servers |
| [docs/server-api.md](docs/server-api.md) | HTTP API details |
| [docs/architecture.md](docs/architecture.md) | Design notes |
| [docs/llamacpp-binding.md](docs/llamacpp-binding.md) | The ctypes binding internals |
| [docs/gpu-setup.md](docs/gpu-setup.md) | GPU/DLL setup |
| [docs/flux-setup.md](docs/flux-setup.md) | ComfyUI FLUX image pipeline |
| [docs/tls.md](docs/tls.md) | API keys, TLS, and reverse proxies for LAN serving |

---

## License

MIT
