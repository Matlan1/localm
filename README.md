# localm

**Run local LLMs offline.** GGUF models via a pure-Python ctypes binding to `llama.dll`, HuggingFace Transformers models, an OpenAI-compatible HTTP server, and an interactive chat shell — all from one CLI.

```
localm run mymodel --prompt "Explain RDNA2 in one sentence."
echo "Write me a haiku." | localm run mymodel
localm serve mymodel --port 8080
```

---

## Features

| Feature | Details |
|---|---|
| **GGUF inference** | Pure-Python ctypes wrapper around `llama.dll` — no llama-cpp-python required |
| **GPU support** | AMD (ROCm / HIP), NVIDIA (CUDA), CPU — auto-detected from DLL loading order |
| **HF Transformers** | Full HuggingFace model directories |
| **OpenAI-compatible server** | `/v1/chat/completions`, `/v1/models`, `/health` — streaming SSE + JSON |
| **Interactive chat** | Multi-turn shell with `/clear`, `/image`, `/system`, `/temp`, `/save` |
| **Model registry** | Pull from HuggingFace, register local paths, manage aliases |
| **Stdin pipe** | `echo "prompt" \| localm run model` |
| **Multimodal** | Image attachment via `--image` or `/image` command (requires mmproj GGUF) |
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

---

## Quick Start

### Pull a model

```bash
# Named shortcut (if registered)
localm pull mymodel

# Specific GGUF from any HF repo
localm pull owner/repo:model-Q4_K_M.gguf

# Full HuggingFace model directory (transformers format)
localm pull owner/model-name
```

### Register an existing model

```bash
# Local GGUF file
localm add C:\models\mymodel.gguf

# Ollama model (resolves manifest → GGUF blob automatically)
localm add D:\ollama\manifests\registry.ollama.ai\library\<model>\<tag>

# HuggingFace directory
localm add D:\models\my-hf-model --name mymodel
```

### Run inference

```bash
# Single prompt
localm run mymodel --prompt "What is 42?"

# Stdin pipe
echo "Translate 'hello' to Japanese." | localm run mymodel

# Interactive chat (TTY)
localm run mymodel

# With a system prompt
localm run mymodel --system "You are a terse assistant." --prompt "How does TCP work?"

# With an image (multimodal)
localm run mymodel --prompt "Describe this image." --image photo.jpg --mmproj mmproj.gguf
```

### Start the inference server

```bash
localm serve mymodel --port 8080

# OpenAI-compatible endpoints:
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mymodel",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Use it with any OpenAI client:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="localm")
resp = client.chat.completions.create(
    model="mymodel",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## CLI Reference

### `localm run`

```
localm run MODEL [OPTIONS]

Options:
  -p, --prompt TEXT       Single prompt (non-interactive)
  -s, --system TEXT       System prompt
  -m, --max-tokens INT    Maximum tokens to generate  [default: 1024]
  -t, --temperature FLOAT Sampling temperature        [default: 0.8]
  -c, --ctx INT           Context window size (GGUF)  [default: 4096]
  -g, --gpu-layers INT    GPU layers, 99=all (GGUF)   [default: 99]
  --mmproj PATH           Multimodal projection GGUF
  --image PATH            Image to attach (repeatable)
  --output-dir PATH       Save model-generated images here
```

### Interactive chat commands

| Command | Effect |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/image <path>` | Queue a local image for the next message |
| `/images` | List queued images |
| `/system <text>` | Set or replace the system prompt |
| `/save [file]` | Save conversation to JSON |
| `/temp <float>` | Adjust sampling temperature live |
| `/tokens <int>` | Adjust max tokens live |
| `/exit` or `exit` | Quit |

### `localm serve`

```
localm serve MODEL [OPTIONS]

Options:
  -H, --host TEXT      Bind address  [default: 127.0.0.1]
  -p, --port INT       Port          [default: 8080]
  -c, --ctx INT        Context window size
  -g, --gpu-layers INT GPU layers
```

### Model management

```bash
localm pull owner/repo:file.gguf     # specific GGUF from HuggingFace
localm pull owner/repo               # full HF model directory
localm pull https://example.com/m.gguf  # direct URL

localm add /path/to/model.gguf       # register local file
localm add /path/to/hf-dir           # register HF directory
localm add /ollama/manifests/...     # register Ollama blob

localm list                          # show registered models
localm models                        # show available shortcuts
localm rm MODEL [--yes]              # remove (deletes file if in ~/.localm)
```

### Configuration

```bash
localm info                          # paths + current config
localm config temperature 0.7        # set a config value
localm config n_gpu_layers 99
localm config n_ctx 8192
```

Config is stored at `~/.localm/config.json`.

---

## GPU Setup (AMD)

localm auto-detects the GPU DLL directory by scanning:

1. `LLAMA_CPP_LIB` environment variable (explicit path to `llama.dll`)
2. `D:\projects\llama-gfx1030-prebuilt\` (default gfx1030 build location)
3. `D:\projects\llama.cpp\build\bin\`

The DLL loading order is: `ggml.dll` → `ggml-base.dll` → `ggml-cpu.dll` → `ggml-hip.dll` → `llama.dll`.

To use a custom build:
```bash
set LLAMA_CPP_LIB=C:\path\to\llama.dll
localm run mymodel --prompt "..."
```

---

## Architecture

```
localm/
├── cli.py                    # Click commands: run, serve, pull, add, list, rm, info, config
├── config.py                 # ~/.localm/ paths, load/save config
├── model_manager.py          # registry, pull (HF + URL), Ollama manifest resolution
└── inference/
    ├── engine.py             # unified Engine — detects GGUF vs HF, context manager
    ├── http_server.py        # FastAPI app: /health, /v1/models, /v1/chat/completions
    ├── protocol.py           # Pydantic request/response models (OpenAI wire format)
    ├── media.py              # image → data-URI, PIL helper
    └── backends/
        ├── base.py           # BaseBackend ABC
        ├── gguf.py           # GgufBackend — wraps LlamaCpp, subprocess fallback
        ├── hf.py             # HFBackend — HuggingFace Transformers
        └── llamacpp/         # Pure-Python ctypes llama.cpp binding
            ├── __init__.py   # exports LlamaCpp
            ├── _loader.py    # DLL loading, dependency order, PATH extension
            ├── _structs.py   # ctypes Structures: LlamaModelParams, LlamaContextParams, …
            ├── _api.py       # low-level C API bindings (llama_load_model, llama_decode, …)
            └── llama.py      # LlamaCpp class — tokenizer, sampler chain, generate loop
```

### The ctypes llama.cpp binding

`localm.inference.backends.llamacpp` is a zero-dependency Python wrapper around the native `llama.dll`. It replaces `llama-cpp-python` entirely, meaning:

- **No C compiler** needed at install time
- **No Python wheel** tied to a specific Python/CUDA version
- Any prebuilt `llama.dll` works — Ollama's DLL, a custom build, any binary
- The struct layouts in `_structs.py` were derived by probing `llama_model_default_params()` / `llama_context_default_params()` against known default values and cross-referenced with `llama.h`

The generation loop (`LlamaCpp._generate`) implements the full sampler chain:
`top_k → top_p → min_p → temperature → dist (random draw)`, or `greedy` when `temperature=0`.

Stop strings (`<|im_end|>`, `<end_of_turn>`, etc.) are filtered via a streaming buffer that watches for multi-token sequences across piece boundaries.

---

## License

MIT
