# LocaLM

**Offline local LLM inference and extension platform - GGUF and HuggingFace models, OpenAI-compatible server, agentic coding, media generation, RAG, and more through a plugin engine.**

![License: AGPL v3](https://img.shields.io/badge/license-AGPL%20v3-blue)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Platform: Windows | Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

LocaLM runs GGUF models through a pure-Python ctypes binding to `llama.dll` (no `llama-cpp-python`), runs HuggingFace Transformers models, and exposes both through an OpenAI-compatible HTTP server. At its core it is a **model loader plus a plugin engine**: the only always-present feature is **chat**, shipped as a protected, preinstalled plugin. Everything else - the coder agent, image/music/video generation, Knowledge (RAG), web access, voice, text-to-speech, scheduled jobs, and MCP - is a plugin you install when you want it. One CLI, no cloud required.

Everything that does not strictly need the internet works fully offline. Online providers (OpenAI, Anthropic) exist only as explicit opt-ins for the coder agent and are never a default. When a task does need the web (current docs, the weather), the coder and chat reach it through a single policy choke point: `off` / `ask` / `allow` modes, domain allow/deny lists, and a private-address SSRF guard ([guide](docs/network.md)).

## Quick start

```bash
localm pull owner/repo:model-Q4_K_M.gguf   # Download a GGUF from HuggingFace
localm gui                                  # Chat + coder + plugins in your browser
localm run mymodel --prompt "Explain RDNA2" # Offline terminal chat or single prompt
localm serve mymodel                        # OpenAI-compatible API server
```

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
- [First success](#first-success)
- [CLI reference](#cli-reference)
- [Plugins](#plugins)
- [GPU setup](#gpu-setup)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **Local model inference.** GGUF files load through a small ctypes binding to `llama.dll`, so there is no `llama-cpp-python` to build. HuggingFace models work too. The installer detects your GPU and provisions the matching llama.cpp backend - **Vulkan** runs on any AMD/NVIDIA/Intel GPU with no vendor toolkit, with CUDA/ROCm offered for peak performance and CPU when there is no GPU - and localm auto-detects the GPU at load.

- **Pick how you talk to it.** A browser GUI, a plain terminal chat, and an OpenAI-compatible server for when you want other apps to connect.

- **A coding agent that does the work (coder plugin).** `localm coder` works through a task with tools for files, the shell, search, and tests; you can redirect it mid-run or review what it touched with session diffs. It speaks MCP both ways, so localm can expose your models to clients like Claude Desktop, and the coder can pull in external MCP tool servers.

- **Media generation (image/music/video plugins).** localm drives a local media-generation server - **ComfyUI is the backend supported today** - it is what the maintainer happens to run, not a recommendation. You bring the server and models; localm orchestrates generation and VRAM handover from the LLM, and surfaces it as the Images/Music/Video pages and chat commands. The plugins keep a backend seam so other media servers can be added later.

- **Bring your own data (rag, voice, and tts plugins).** Attach files or index whole folders and chat against them with citations (Knowledge), dictate with local Whisper speech-to-text, or have replies read back to you with in-browser Kokoro text-to-speech.

- **Schedule it (jobs plugin).** Run a chat or coder prompt on an interval or a cron schedule from an in-app scheduler, the terminal, or a REST API ([guide](docs/jobs.md)).

- **Model management that stays out of the way.** Pull from HuggingFace with aliases and SHA256 dedup, browse quants with a note on whether they fit your VRAM, and let localm register whatever you drop into the models folder.

- **Offline first.** Nothing leaves your machine unless you allow it. The optional online parts (cloud providers for the coder, web access for fetching pages) are opt-in and run through one network policy you set.

---

## Requirements

- **Python 3.12** (the project pins `requires-python = >=3.12,<3.13`). The installers build a 3.12 venv, and the `[gpu]` ROCm torch wheels are cp312-only.

- **For GGUF GPU inference:** a compiled `llama.dll` + GPU runtime DLLs. `localm setup-llama` provisions these for you (see [GPU setup](#gpu-setup)). The installer detects your hardware and chooses the backend automatically.

- **NVIDIA CUDA (optional, for peak performance):** a recent NVIDIA driver (new enough for CUDA >= 12.4). **No CUDA Toolkit needed** - localm fetches a self-contained CUDA runtime. An older driver still works: setup falls back to Vulkan and tells you how to enable CUDA later. The default install uses Vulkan (any GPU, driver-only); choose CUDA in the setup menu (or `localm setup-llama --backend cuda`) for peak NVIDIA performance.

Run `localm doctor` after installing to check Python, the native library, GPU driver, VRAM, and optional packages in one shot.

---

## Install

> **Stable release:** to install a specific tagged version instead of the latest
> `master`, clone that tag, for example:
> `git clone --depth 1 --branch v0.1.0 https://github.com/Matlan1/localm.git`
> then follow the setup below inside the clone.

### Recommended (Windows): self-contained setup

Clone anywhere and double-click `setup.bat`. It creates a private `.venv` inside the clone (always Python 3.12), installs localm into it, detects your GPU and provisions the matching llama.cpp backend (Vulkan for any GPU, CUDA/ROCm for peak performance, or CPU), and asks where data should live:

- **inside the clone** (`.\home`) - fully portable; multiple clones on one machine are completely independent,
- **shared** (`~/.localm`) - clones share models and settings, or
- **a custom path** (recorded in `localm-home.cfg`).

It then offers an optional desktop shortcut and walks you through which plugins to enable (`localm plugin setup`). Nothing is installed globally and PATH is untouched. The `LOCALM_HOME` environment variable overrides the data location at any time.

### Manual (any OS)

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv -e ".[coder,voice]"       # base: NVIDIA / Intel / CPU (GGUF chat needs no torch)
uv pip install -p .venv -e ".[gpu,coder,voice]"   # AMD RDNA2 ROCm ONLY - do NOT use on NVIDIA/Intel
localm setup-llama                                 # provision the native llama.cpp backend (Vulkan/CUDA/CPU)
```

(`voice` adds the GUI mic's Whisper speech-to-text; drop it if you do not want it.)

The base line above works on NVIDIA, Intel, and CPU; only AMD RDNA2 users add `[gpu]` (the ROCm torch stack). `localm setup-llama` is required after a manual install to fetch the native backend (`setup.bat` does this for you). NVIDIA users who also want HuggingFace/torch models install CUDA torch separately: `uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

A pip extra and a plugin install are two separate steps. The extra installs a plugin's heavy Python dependencies into the venv; `localm plugin install <name>` activates the plugin itself. The defined extras are:

| Extra | What it adds |
|---|---|
| `coder` | The AI coding agent (opt-in marker; deps are already core) |
| `gpu` | AMD RDNA2 ROCm 7.13 stack: torch, transformers, rocm-sdk (Windows, Python 3.12) |
| `audio` | Audio multimodal input (`soundfile`) |
| `rag` | PDF parsing for Knowledge (`pypdf`); other formats are stdlib |
| `voice` | Whisper speech-to-text for the GUI mic (`faster-whisper`, CPU, no torch) |
| `abliterate` | Opt-in marker for `localm abliterate` (no new deps; shells out to Heretic) |
| `monitor` | Live hardware monitor in the GUI status bar (`psutil`) |
| `qr` | QR code for phone pairing (the Settings key QR / companion onboarding) |
| `grammar` | Grammar-constrained decoding for HuggingFace models (`xgrammar`, layers on `[gpu]`) |
| `gguf` | Optional `llama-cpp-python` path (rarely needed; core ships its own ctypes binding) |
| `cpu` | Explicit CPU-only marker (empty; core already runs GGUF on CPU) |
| `dev` | Contributor / CI tooling: `ruff` + `pytest` |

Not every plugin needs an extra: the image/music/video plugins talk to an external media-generation server you run (ComfyUI today), jobs and web have no extra, and tts runs in the browser.

> **Avoid `uv tool install` for this project.** Tool installs are *global per package name*: a second clone installing the `localm` tool silently replaces the first one's `localm.exe`/`localcoder.exe`. The `[gpu]` extra also needs `--python 3.12` there (the ROCm torch wheels are cp312-only).

### No models yet?

On a fresh install the launcher's **Import** row gets you a first model three ways: *from file...* / *from folder...* register a GGUF or a HuggingFace directory already on disk, and *from URL...* opens the Web GUI on its Models page and downloads the model there with a live progress bar. You can also just launch the Web GUI with nothing registered (`localm gui --no-model`); it opens straight to the Models page.

---

## First success

### 1. Download a model

```bash
# Specific GGUF from any HF repo (split files are handled automatically)
localm pull owner/repo:model-Q4_K_M.gguf

# Full HuggingFace model directory (transformers format)
localm pull owner/model-name

# Direct URL, with optional integrity check
localm pull https://example.com/m.gguf --sha256 <hash>
```

Duplicate downloads are detected by path and SHA256. When you add or pull something already registered, localm offers alias / copy / move / skip instead of silently duplicating gigabytes.

### 2. Open the GUI

```bash
localm gui                # picks the first registered model, opens your browser
localm gui mymodel        # or name one
localm gui --no-model     # open model-less, straight to the Models page
```

Chat, the coder agent, model management, and any enabled plugin tabs in one page. The model preloads in the background so the first reply is fast. See [docs/gui.md](docs/gui.md) for details.

### 3. Or use the terminal

```bash
localm run mymodel --prompt "What is 42?"     # single prompt
echo "Translate 'hello' to Japanese." | localm run mymodel  # pipe input
localm run mymodel                             # interactive chat
```

### 4. Or start a server

```bash
localm serve mymodel
```

localm owns the port range 8642-8741: the default is 8642 and the server bumps to the next free port automatically when it is taken.

```bash
curl http://localhost:8642/health
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
)
print(resp.choices[0].message.content)
```

The streaming usage block adds `ttft_ms` and `tokens_per_sec`. See [docs/server-api.md](docs/server-api.md) for the full API surface.

---

## CLI reference

The full CLI reference lives in [docs/cli.md](docs/cli.md). Here are the core commands:

```bash
localm run MODEL                  # interactive chat or single prompt
localm gui [MODEL]                # browser GUI
localm serve MODEL                # OpenAI-compatible server
localm coder [TASK]               # AI coding agent
localm benchmark MODEL            # TTFT and tok/s measurements
localm doctor                      # check installation
localm info                        # paths and config
```

Model management:

```bash
localm pull SPEC                  # download from HuggingFace or URL
localm search QUERY               # search HuggingFace for GGUF models
localm add PATH                   # register a local model
localm alias MODEL NEWNAME        # add a second name
localm list                        # registered models
localm rm MODEL                   # remove a model
```

Plugins:

```bash
localm plugin status              # what is installed
localm plugin install NAME        # enable a plugin
localm plugin disable NAME        # turn off a plugin
```

Full reference: [docs/cli.md](docs/cli.md).

---

## Plugins

localm core is a model loader plus a plugin engine. Chat is the protected, preinstalled plugin and is the only feature active out of the box; every other feature (coder, image, music, video, rag, web, voice, tts, jobs, mcp) is a plugin you install when you want it.

**Plugin states.** Bundled plugins live read-only in `localm/plugins/builtin/` (the "store"). *Installing* copies one into `~/.localm/plugins/` (installed = on disk); *enabling* adds it to `config["plugins_enabled"]`; a plugin is *active* only when it is both installed and enabled. Chat is protected (cannot be disabled or uninstalled) and `default_enabled`, so it is active on first run; nothing else is.

**First-party store plugins** are managed by name:

```bash
localm plugin status            # what is installed and which installs are active
localm plugin install NAME      # copy NAME from the store and enable it
localm plugin enable NAME       # enable an already-installed plugin
localm plugin disable NAME      # disable but keep it installed
localm plugin uninstall NAME    # remove it (add --delete-data to drop its data)
localm plugin setup             # pick a starter set interactively
```

The store names are `coder`, `image`, `music`, `video`, `rag`, `web`, `voice`, `tts`, `jobs`, and `mcp` (plus the protected `chat`). For plugins with heavy Python dependencies, also install the matching pip extra (for example `pip install "localm[rag]"` alongside `localm plugin install rag`); see [Install](#install). A running GUI server picks up new HTTP routes and tabs at runtime, while stdio plugins like mcp take effect on the next `localm mcp`.

**Third-party plugins** are folders containing a `plugin.toml` manifest and Python files. Install a third-party plugin from a local path with `localm plugin install <path>` (the same command takes a store name or a directory); installation is a local directory copy, fully offline. See [docs/plugins.md](docs/plugins.md) for the full authoring contract.

---

## GPU setup

The native llama.cpp binaries live **inside this install**, packaged as the `localm-llama-runtime` wheel in the venv, so the project never depends on a folder elsewhere on disk. `localm setup-llama` provisions the backend matching your hardware (full list in [docs/gpu-setup.md](docs/gpu-setup.md)):

```bash
localm setup-llama                       # auto-detect the GPU, fetch the right backend
localm setup-llama --backend vulkan      # any GPU (AMD/NVIDIA/Intel), no vendor toolkit
localm setup-llama --backend cuda        # NVIDIA  /  --backend amd-rocm (AMD)  /  --backend cpu
localm setup-llama --from <build-dir>    # or copy your own llama.cpp build
```

Vulkan runs on any GPU with just the vendor's normal driver; CUDA/ROCm give peak performance when their runtime is present; CPU always works. The AMD ROCm build is self-contained (it bundles its ROCm runtime via the `[gpu]` extra's `rocm-sdk` wheels). macOS/Metal is experimental. See [docs/phone.md](docs/phone.md) to reach the GUI from a phone.

**NVIDIA users:** the installer recommends **Vulkan** by default (works with just your GPU driver, no CUDA Toolkit) - press Enter to accept it and your GPU is used. For **peak performance choose `cuda`** in the setup menu (or run `localm setup-llama --backend cuda`): it fetches a self-contained CUDA runtime (no Toolkit needed), verifies the build actually loads, and falls back to Vulkan with a clear message if your driver is too old for CUDA 12.4+.

localm resolves the binary directory in order: `LLAMA_CPP_LIB` env > `binary_dir` config > the bundled runtime wheel. No absolute path is ever assumed as a default; an unprovisioned install resolves to nothing and points you at `localm setup-llama`. Before loading a model, localm checks free VRAM against the model size and warns when it will not fit, instead of crashing mid-load.

---

## Architecture

```
runtime/                      # localm-llama-runtime wheel: native llama.cpp
|                             #   binaries bundled in the venv (self-contained)
localm/
├── cli/                      # Click command package, split by area (core + plugin command groups)
├── config.py                 # data-dir paths, config, port range, defaults
├── auth.py, scopes.py        # API-key auth and capability scopes
├── netpolicy.py              # the off/ask/allow web-access policy + SSRF guard
├── setup_llama.py            # `localm setup-llama`: provision native binaries
├── model_manager/            # registry, pull, dedup, aliases, Ollama manifests (gguf/registry/pull)
├── image_gen/                # shared ComfyUI FLUX transport (used by image plugin)
├── music_gen/                # shared ComfyUI ACE-Step transport (used by music plugin)
├── video_gen/                # shared ComfyUI Wan 2.2 transport (used by video plugin)
├── media/                    # shared ComfyUI client (comfy_client.py), used by image/music/video
├── inference/
|   ├── engine.py             # unified Engine: GGUF vs HF detection
|   ├── http_server.py        # FastAPI app factory (create_app) + shared inference state
|   ├── routes/               # HTTP route handlers (admin, chat, config, keys, models, plugins, session, system)
|   ├── protocol.py           # Pydantic models (OpenAI wire format)
|   ├── chat_pipeline.py      # kernel chat-pipeline hook chain (inlet/stream/outlet)
|   ├── compact.py            # automatic history compaction
|   └── backends/
|       ├── base.py           # shared backend base class
|       ├── gguf.py           # GgufBackend + VRAM pre-flight + dynamic context
|       ├── hf.py             # HFBackend (Transformers)
|       └── llamacpp/         # pure-Python ctypes llama.cpp binding
└── plugins/                  # the plugin engine and every feature plugin
    ├── engine.py             # PluginManager: install/enable/active state, route mounting
    ├── contract.py           # the plugin contract (manifest schema, surfaces, API version)
    ├── catalog.py            # the first-party store listing
    ├── loader.py             # external plugin discovery (~/.localm/plugins/)
    ├── media_config.py       # shared ComfyUI config for the media plugins
    ├── coder/                # the coding agent: agent loop, tools, MCP client, skills
    ├── gui/                  # the web GUI server (FastAPI routes + static SPA)
    ├── abliterate/           # `localm abliterate`: Heretic hand-off + register
    ├── mcpserver/            # `localm mcp` stdio server
    └── builtin/              # the read-only store (only chat is active by default)
        ├── chat/             # protected plugin #0: history, memory, personas
        ├── coder/            # manifest + shim -> plugins/coder
        ├── image/            # FLUX image generation (consumes image_gen/)
        ├── music/            # ACE-Step music generation (consumes music_gen/)
        ├── video/            # Wan 2.2 video generation (consumes video_gen/)
        ├── rag/              # Knowledge: collections + cited retrieval
        ├── web/              # web_search + fetch_url under the network policy
        ├── voice/            # local Whisper speech-to-text
        ├── tts/              # in-browser Kokoro text-to-speech (client assets)
        ├── jobs/             # scheduled recurring chat/coder prompts (scheduler + Jobs tab)
        └── mcp/              # manifest + shim -> plugins/mcpserver
```

Each `builtin/<name>/` entry is a manifest (`plugin.toml`) plus its plugin code; the coder and mcp entries are thin shims that delegate to the top-level implementation packages (`plugins/coder/`, `plugins/mcpserver/`), and the image/music/video plugins consume the shared `image_gen/` / `music_gen/` / `video_gen/` transports.

### The ctypes llama.cpp binding

`localm.inference.backends.llamacpp` is a zero-dependency Python wrapper around the native `llama.dll`. It replaces `llama-cpp-python` entirely, meaning:

- **No C compiler** needed at install time
- **No Python wheel** tied to a specific Python/CUDA version
- Any prebuilt `llama.dll` works: Ollama's DLL, a custom build, any binary
- The struct layouts in `_structs.py` were derived by probing `llama_model_default_params()` / `llama_context_default_params()` against known default values and cross-referenced with `llama.h`

The generation loop (`LlamaCpp._generate`) implements the full sampler chain: `top_k -> top_p -> min_p -> temperature -> dist (random draw)`, or `greedy` when `temperature=0`. Stop strings (`<|im_end|>`, `<end_of_turn>`, etc.) are filtered via a streaming buffer that watches for multi-token sequences across piece boundaries.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/cli.md](docs/cli.md) | Full CLI reference (model management, media, RAG, jobs, config, shell completion) |
| [docs/plugins.md](docs/plugins.md) | Authoring a plugin: manifest, tool exports, surfaces, chat-pipeline hooks, privacy rules |
| [docs/plugin-interop.md](docs/plugin-interop.md) | Foreign-ecosystem interop: importing plugins from other tools over the MCP spine |
| [docs/skills.md](docs/skills.md) | The Agent Skills (`SKILL.md`) importer for the coder agent |
| [docs/gui.md](docs/gui.md) | The web GUI: chat, coder sessions, studios, Jobs, approvals, security notes |
| [docs/jobs.md](docs/jobs.md) | Scheduled jobs: the scheduler, `localm job` CLI, `/api/jobs`, the Jobs tab |
| [docs/mcp.md](docs/mcp.md) | MCP in both directions: serving models, consuming tool servers |
| [docs/server-api.md](docs/server-api.md) | HTTP API: OpenAI endpoints, management routes, auth, CORS |
| [docs/architecture.md](docs/architecture.md) | Design notes |
| [docs/llamacpp-binding.md](docs/llamacpp-binding.md) | The ctypes binding internals |
| [docs/gpu-setup.md](docs/gpu-setup.md) | GPU/DLL setup |
| [docs/linux-setup.md](docs/linux-setup.md) | Running localm on Linux: venv, runtime, GPU notes |
| [docs/flux-setup.md](docs/flux-setup.md) | ComfyUI FLUX image pipeline (shared by music/video) |
| [docs/video.md](docs/video.md) | Wan 2.2 video generation: model setup, timing, workflow override |
| [docs/rag.md](docs/rag.md) | Knowledge: chat with your documents, collections, retrieval design |
| [docs/network.md](docs/network.md) | Internet access for coder and chat: modes, domain rules, SSRF guard |
| [docs/tls.md](docs/tls.md) | API keys, TLS, and reverse proxies for LAN serving |
| [docs/phone.md](docs/phone.md) | Using localm from your phone: the installable PWA companion |

---

## License

Copyright (C) 2025-2026 Matlan1

localm is free software, licensed under the **GNU Affero General Public License, version 3 or later** (AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.

In short: you may use, study, modify, and redistribute localm freely, including commercially. But if you redistribute it, or run a modified version as a network service, you must make the complete corresponding source available to your users under the same license. This keeps localm, and anything built on top of it, open.
