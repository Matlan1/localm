# localm

**Run local LLMs offline.** GGUF models via a pure-Python ctypes binding to `llama.dll`, HuggingFace Transformers models, and an OpenAI-compatible HTTP server. localm core is a model loader plus a plugin engine: the only always-present feature is **chat**, which ships as the protected, preinstalled plugin. Everything else - the coder agent, image/music/video generation, Knowledge (RAG), web access, voice (speech-to-text), text-to-speech, and MCP - is a plugin you install when you want it. One CLI, no cloud required.

**Plugin states.** Bundled plugins live read-only in `localm/plugins/builtin/` (the "store"). *Installing* a plugin copies it into `~/.localm/plugins/` (installed = on disk); *enabling* adds it to `config["plugins_enabled"]`; a plugin is *active* only when it is both installed and enabled. **Out of the box only chat is active** (it is protected, so it cannot be disabled or uninstalled, and `default_enabled` makes it active on first run). You turn the rest on with `localm plugin install <name>` (see [Plugins](#plugins) below).

```
localm run mymodel --prompt "Explain RDNA2 in one sentence."
localm gui
localm coder "add type hints to utils.py"
localm serve mymodel
```

Everything that does not strictly need the internet works fully offline. Online providers (OpenAI, Anthropic) exist as explicit opt-ins for the coder agent and are never a default. When a task does need the web (current docs, the weather), the coder and chat can search and fetch pages through a single policy choke point - `off` / `ask` / `allow` modes, domain allow/deny lists, SSRF guard ([guide](docs/network.md)).

---

## Features

- **Local model inference.** GGUF files load through a small ctypes binding to `llama.dll`, so there's no llama-cpp-python to install. HuggingFace models work too, and it works out whether to run on your AMD or NVIDIA GPU (or fall back to CPU) from what actually loads at startup.
- **Pick how you talk to it.** A browser GUI, a plain terminal chat, and an OpenAI-compatible server for when you want other apps to connect.
- **A coding agent that does the work (coder plugin).** `localm coder` works through a task with tools for files, the shell, search, and tests, and you can redirect it mid-run or review what it touched with session diffs. It speaks MCP both ways, so localm can expose your models to clients like Claude Desktop, and the coder can pull in external MCP tool servers.
- **Media generation (image/music/video plugins, drives your ComfyUI).** Point localm at a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and it makes images with FLUX, music of any length with ACE-Step, and short video clips from a prompt or a still with Wan 2.2. You install ComfyUI and download the models once (roughly 20 to 25 GB for FLUX; see [docs/flux-setup.md](docs/flux-setup.md)); localm orchestrates the rest, including VRAM handover from the LLM.
- **Bring your own data (rag, voice, and tts plugins).** Attach files or index whole folders and chat against them with citations (Knowledge), dictate with local Whisper speech-to-text, have replies read back to you with in-browser Kokoro text-to-speech, or hand the model an image to look at.
- **Model management that stays out of the way.** Pull from HuggingFace with aliases and SHA256 dedup, browse quants with a note on whether they fit your VRAM, and let it register whatever you drop into the models folder. Abliteration is a single command that passes a model to Heretic and registers what comes back.
- **Offline first.** Nothing leaves your machine unless you allow it. The optional online parts, meaning cloud providers for the coder and web access for fetching pages, are opt-in and run through one network policy you set.

<details>
<summary>Full feature list</summary>

localm core (model loader plus plugin engine) provides GGUF/HF inference, the OpenAI-compatible server, the GUI shell, model management, and chat (the protected plugin that is active by default). Everything tagged *(plugin)* below is shipped in the `builtin/` store but is inactive until you run `localm plugin install <name>`; see [Plugins](#plugins).

| Feature | Details |
|---|---|
| **GGUF inference** (core) | Pure-Python ctypes wrapper around `llama.dll`, no llama-cpp-python required |
| **GPU support** (core) | AMD (ROCm / HIP), NVIDIA (CUDA), CPU. Auto-detected from DLL loading order |
| **HF Transformers** (core) | Full HuggingFace model directories |
| **OpenAI-compatible server** (core) | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, streaming SSE, TTFT and tok/s in usage |
| **Web GUI** (core) | `localm gui`: chat, coder agent, model manager, image/music/video generation, plugins, and settings in the browser; zero build step, fully offline ([guide](docs/gui.md)) |
| **Chat** (plugin #0, protected) | Conversation history, assistant memory, and personas. The only plugin active out of the box; cannot be disabled or uninstalled |
| **Coding agent** (coder plugin) | `localm coder` / `localcoder`: agentic loop with file, shell, search, test, and image tools; mid-task steering, cumulative session diffs (`/changes`, `/diff`), circuit breaker on repeated failures, tab-completed REPL |
| **Web access (opt-in)** (web plugin) | `web_search` + `fetch_url` for coder and chat via one network policy: `off`/`ask`/`allow`, domain allow/deny, private-address SSRF guard ([guide](docs/network.md)) |
| **Knowledge (RAG)** (rag plugin) | Chat with your documents: attach files in chat (in-memory, privacy-clean) or index folders into collections with cited retrieval - BM25 always, embeddings blended in when the backend supports them ([guide](docs/rag.md)) |
| **Voice (speech-to-text)** (voice plugin) | Local Whisper speech-to-text into the composer (`localm[voice]` extra, CPU, no torch) |
| **Text-to-speech** (tts plugin) | Read replies aloud with in-browser Kokoro neural voices. Synthesis runs entirely client-side (vendored kokoro-js), so it ships no Python dependency and writes nothing to disk, keeping privacy mode trace-free; shipped as a client-asset plugin |
| **MCP client** (mcp plugin) | The coder consumes external MCP tool servers from `.localcoder/config.toml` |
| **MCP server** (mcp plugin) | `localm mcp` exposes your local models to Claude Desktop and other MCP clients ([guide](docs/mcp.md)) |
| **Interactive chat** | Multi-turn shell with `/imagine` (image plugin), `/compact`, `/clear`, `/image`, `/system`, `/save` |
| **Model registry** (core) | Pull from HuggingFace (split GGUF supported), aliases, SHA256 dedup, tab completion |
| **Model discovery** (core) | Search HF from the Models page or `localm search`; per-quant sizes with "fits your VRAM" badges (torch-free VRAM detection) |
| **Abliteration** | `localm abliterate`: decensor a model with [Heretic](https://github.com/Matlan1/heretic-win-AMD) (a separate AGPL program, run as a subprocess), then auto-register the result (`localm[abliterate]` extra) |
| **Folder auto-sync** (core) | `localm list`/`gui`/launcher reconcile the registry with the models folder on start; missing files are flagged, not deleted (opt-in `autoprune_missing_models`) |
| **Image generation** (image plugin) | `generate_image` tool drives a local ComfyUI FLUX pipeline with VRAM handover (requires ComfyUI + models, see [docs/flux-setup.md](docs/flux-setup.md)) |
| **Music generation** (music plugin) | ACE-Step via the same ComfyUI server: arbitrary track length, lyrics or instrumental (`localm music`, Music page, `/music`) |
| **Video generation** (video plugin) | Wan 2.2 short clips (~5 s native, text- or image-to-video) via ComfyUI (`localm video`, Video page, `/video`; [guide](docs/video.md)) |
| **Plugins** | First-party store plugins (above) plus third-party folders: install/enable/disable/uninstall from the CLI or GUI, export agent tools, add CLI commands and GUI tabs ([authoring guide](docs/plugins.md)) |
| **Multimodal** (core) | Image attachment via `--image` or `/image` with a HuggingFace-format vision model. The built-in GGUF backend is text-only and rejects an attached image with a clear error rather than silently ignoring it. |
| **Ollama interop** (core) | Register Ollama blobs directly via `localm add <manifest-dir>` |

</details>

---

## Requirements

- Python 3.10+
- For GGUF GPU inference: a compiled `llama.dll` + GPU runtime DLLs
  - AMD: ROCm `ggml-hip.dll`
  - NVIDIA: CUDA `ggml-cuda.dll`
  - CPU: only `llama.dll` + `ggml*.dll`

---

## Install

**Recommended (Windows): self-contained setup.** Clone anywhere and double-click
`setup.bat`. It creates a private `.venv` inside the clone, installs localm into
it (you pick the full AMD-GPU flavour or base), and asks where data should live:

- **inside the clone** (`.\home`) - fully portable; multiple clones on one
  machine are completely independent,
- **shared** (`~/.localm`) - clones share models and settings, or
- **a custom path** (recorded in `localm-home.cfg`).

Nothing is installed globally and PATH is untouched. The `LOCALM_HOME` env var
overrides the data location at any time.

Manual equivalent (any OS):
```bash
uv venv --python 3.12 .venv
uv pip install -p .venv -e ".[gpu,coder,audio]"   # AMD ROCm flavour
uv pip install -p .venv -e ".[coder]"             # base / CPU flavour
```

A pip extra and a plugin install are two separate steps. The extra installs a
plugin's heavy Python dependencies into the venv; `localm plugin install <name>`
activates the plugin itself. The defined extras are `coder`, `gpu`, `gguf`,
`audio`, `rag`, `voice`, `cpu`, and `abliterate`. Not every plugin needs one:
the image/music/video plugins talk to an external ComfyUI (no extra), and tts
runs in the browser (no extra, no Python dependency). So enabling RAG is, for
example, `pip install "localm[rag]"` followed by `localm plugin install rag`.

> **Avoid `uv tool install` for this project.** Tool installs are *global per
> package name*: a second clone installing the `localm` tool silently replaces
> the first one's `localm.exe`/`localcoder.exe`. The `[gpu]` extra also needs
> `--python 3.12` there (the ROCm torch wheels are cp312-only).

On Windows, double-click `localm-launcher.bat` in the repo root for a graphical
launcher: pick Web GUI, terminal chat, API server, or the coder agent, choose a
model, toggle debug mode, set port/context/GPU layers, and hit Launch. Your
choices are remembered. (`localm.bat` still starts a plain chat directly.) Both
use the clone's own `.venv` automatically.

**No models yet?** On a fresh install the launcher's **Import** row gets you a
first model three ways: *from file…* / *from folder…* register a GGUF or a
HuggingFace directory already on disk, and *from URL…* opens the Web GUI on its
Models page and downloads the model there with a live progress bar. You can
also just launch the Web GUI with nothing registered - it opens straight to the
Models page so you can pull or import from the browser.

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
`/` in any composer opens a command menu. `/imagine` is provided by the image
plugin and generates images inline; it is unavailable until that plugin is
installed (`localm plugin install image`). See [docs/gui.md](docs/gui.md).

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

The agent auto-starts `localm serve` when needed, plans with tool calls (read, write, edit, patch, shell, search, tests, image generation, plus tools exported by other installed plugins, which are registered as `plugin_<plugin>_<tool>`), asks before destructive actions, tracks a turn budget so it asks for help instead of guessing forever, and verifies its own code changes before answering. Privacy mode is the default: nothing is persisted unless you opt into `--mode log` or `--mode full`.

### Session privacy modes (all surfaces)

Every surface - terminal chat (`localm run`), the API server (`localm serve`), the web GUI (`localm gui`), and the coder agent - honours the same three persistence modes:

| mode | what is written |
|---|---|
| `privacy` (default) | **Nothing, anywhere.** No audit trail, no transcripts, no coder checkpoints, no image/music/video prompt sidecars; GUI conversations stay in memory only (gone on reload); readline + child-shell history suppressed; shell history scrubbed on exit. Explicit actions (`/save`, `/export`, generated files themselves) still work. |
| `log` | JSONL audit trail per session in `~/.localm/sessions/` (user messages, replies, tool calls). GUI chat conversations additionally persist server-side in `~/.localm/chats/` and reload on any browser; past coder audit logs are browsable from the GUI's history button. |
| `full` | Everything in `log`, plus a human-readable Markdown transcript (coder: `.localcoder/sessions/` in the project; chat/server: `~/.localm/sessions/`). |

Resolution order: `--mode` flag > project `.localcoder/config.toml` (coder only) > per-surface config (`chat_mode` / `coder_mode`) > global config `mode` > `privacy`. Set them in `~/.localm/config.json`, the GUI Settings page, or the launcher's Privacy card (global + chat/coder overrides).

What privacy mode cannot suppress: OS-level process logs, DNS/network traces, files you explicitly ask the agent to write - and `--debug`, which is an explicit toggle that records requests and raw model output into its log (a warning is printed when both are active).

### Serve your models over MCP

```bash
localm mcp --print-config     # JSON block for Claude Desktop and friends
```

See [docs/mcp.md](docs/mcp.md) for both directions: localm as an MCP server, and the coder consuming external MCP tool servers.

### Abliterate (decensor) a model

`localm abliterate` hands a model off to [Heretic](https://github.com/Matlan1/heretic-win-AMD)
to remove refusals ("safety alignment"), then registers the result so you can run
it like any other model:

```bash
localm abliterate --model Qwen/Qwen3-4B-Instruct-2507        # HF repo or local path
localm abliterate --model ./model.gguf --export-gguf q5_k_m  # also emit a GGUF
localm abliterate --model <id> --print-command               # preview, don't launch
```

Heretic is a **separate program** (AGPL-3.0) - localm never bundles or imports it,
only runs it. If it isn't found, localm offers to clone the fork into a gitignored
`.heretic/` under your data dir; point at an existing checkout with the
`heretic_path` config key (or the `LOCALM_HERETIC_PATH` env var). Heretic runs in
your terminal: when it finishes, choose "Save the model to a local folder", paste
the path localm prints, and localm registers the saved model on exit. Enable with
`pip install "localm[abliterate]"`.

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
localm abliterate --model M      # decensor M with Heretic, then register it
```

`--debug` on `gui`, `serve`, and `run` writes a log to `~/.localm/logs/` with
request timing and the native llama.cpp stderr stream (including crash abort
reasons), and shows raw model output without marker scrubbing.

### Model management

```bash
localm search qwen 7b                # find GGUF repos on HuggingFace
localm search owner/repo --files     # quants + sizes + "fits your VRAM"
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

`localm list`, `localm gui`, and the desktop launcher auto-scan the models folder
on start: new GGUF files and HuggingFace directories (any with a `config.json`)
are registered automatically, and entries whose file has gone missing are
**flagged** (shown in `localm list`), not deleted - so a temporarily-unavailable
model (unmounted drive, moved file) isn't forgotten. Set
`autoprune_missing_models true` to delete missing entries instead; even then only
files under the models folder are removed, and a registry backup is written first.

### Knowledge (RAG)

```bash
localm rag add NAME PATH...      # index files/folders into a collection
localm rag list                  # collections with doc/chunk counts
localm rag query NAME "text"     # show the top matching excerpts
localm rag rm NAME [--yes]       # delete a collection (index only, files kept)
```

### Configuration

```bash
localm config temperature 0.7
localm config n_gpu_layers 99
localm config n_ctx 8192
localm config port 8650
localm config confirm_remove false
localm config comfy_launch_cmd "D:\path\to\comfyui.bat"   # auto-start ComfyUI for image generation
localm config heretic_path "D:\path\to\heretic"          # Heretic checkout for `localm abliterate` (else auto-detect/clone)
localm config autoprune_missing_models true              # delete missing-file entries (default: flag and keep)
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

> Reading free VRAM needs `torch` (the `[gpu]` extra). On a CPU-only install
> without it, `ctx_auto` cannot measure VRAM and falls back to the fixed
> `n_ctx_max` ceiling (16384 by default); the pre-flight "Low VRAM" warning and
> the post-load VRAM usage line are skipped for the same reason. Plain GGUF
> inference still runs - only the VRAM-aware sizing and warnings need torch.

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

localm core is a model loader plus a plugin engine. Chat is the protected,
preinstalled plugin and is the only feature active out of the box; every other
feature (coder, image, music, video, rag, web, voice, tts, mcp) is a plugin you
install when you want it.

**Plugin states.** Bundled plugins live read-only in `localm/plugins/builtin/`
(the "store"). *Installing* copies one into `~/.localm/plugins/` (installed = on
disk); *enabling* adds it to `config["plugins_enabled"]`; a plugin is *active*
only when it is both installed and enabled. Chat is protected (cannot be disabled
or uninstalled) and `default_enabled`, so it is active on first run; nothing else
is.

**First-party store plugins** are managed by name:

```bash
localm plugin status            # what is installed and which installs are active
localm plugin install NAME      # copy NAME from the store and enable it
localm plugin enable NAME       # enable an already-installed plugin
localm plugin disable NAME      # disable but keep it installed
localm plugin uninstall NAME    # remove it (add --delete-data to drop its data)
```

The store names are `coder`, `image`, `music`, `video`, `rag`, `web`, `voice`,
`tts`, and `mcp` (plus the protected `chat`). For plugins with heavy Python
dependencies, also install the matching pip extra (for example
`pip install "localm[rag]"` alongside `localm plugin install rag`); see
[Install](#install). A running GUI server picks up new HTTP routes on its next
start, while stdio plugins like mcp take effect immediately.

**Third-party plugins** are folders containing a `plugin.toml` manifest and
Python files. A plugin can add a CLI command (`localm <name>`), export tools into
the coder agent, and contribute a GUI tab or client assets. The full authoring
contract (manifest fields, tool-export signature, surfaces, privacy rules) lives
in **[docs/plugins.md](docs/plugins.md)**; foreign-ecosystem interop (importing
plugins from other tools over the MCP spine) is in
[docs/plugin-interop.md](docs/plugin-interop.md). Folder install of a third-party
plugin from a local path is supported by the engine; installation is a local
directory copy, fully offline.

Tools exported by an installed plugin are registered with the coder as
`plugin_<plugin>_<tool>` and described to the model exactly like an MCP tool, and
external plugin code defaults to needing confirmation before it runs.

---

## GPU Setup (AMD)

The native llama.cpp binaries live **inside this install** - packaged as the
`localm-llama-runtime` wheel in the venv - so the project never depends on a
folder elsewhere on disk. Provision them once:

```bash
localm setup-llama                      # download the default gfx1030 prebuilt
localm setup-llama --from <build-dir>   # or copy your own llama.cpp build
```

This places `llama.dll` + `ggml-*.dll` (and, for the prebuilt, the matched ROCm
runtime + `llama-cli`/`llama-server`) into `runtime/localm_llama_runtime/lib/`
and installs the wheel. The ROCm runtime they need at load time
(`amdhip64`, `rocm_kpack`, `rocblas`, …) comes from the `rocm-sdk` wheels the
`[gpu]` extra already installed into the same venv.

localm resolves the binary directory in order: `LLAMA_CPP_LIB` env →
`binary_dir` config → the bundled runtime wheel. No absolute path is ever
assumed as a default; an unprovisioned install resolves to nothing and points
you at `localm setup-llama`. ggml deps load before `llama.dll`, and the venv's
`_rocm_sdk_*/bin` dirs are added to the DLL search path automatically.

Before loading a model, localm checks free VRAM against the model size and warns
when it will not fit, instead of crashing mid-load (this check needs `torch` to
read VRAM; on a CPU-only install it is skipped). KV cache prefix reuse keeps
multi-turn chat fast by only prefilling the new suffix of the conversation.

To use a one-off custom build without provisioning the wheel:
```bash
set LLAMA_CPP_LIB=C:\path\to\llama.dll
localm run mymodel --prompt "..."
```

> Source of the default prebuilt and from-source build instructions for gfx1030
> (RDNA2): see the `windows-native` directory in
> https://github.com/Matlan1/rocm-canary-forge.

---

## Architecture

```
runtime/                      # localm-llama-runtime wheel: native llama.cpp
│                             #   binaries bundled in the venv (self-contained)
localm/
├── cli.py                    # Click commands
├── config.py                 # ~/.localm/ paths, config, port range
├── setup_llama.py            # `localm setup-llama` - provision native binaries
├── model_manager.py          # registry, pull, dedup, aliases, Ollama manifests
├── image_gen/                # shared ComfyUI FLUX transport (used by image plugin)
├── music_gen/                # shared ComfyUI ACE-Step transport (used by music plugin)
├── video_gen/                # shared ComfyUI Wan 2.2 transport (used by video plugin)
├── inference/
│   ├── engine.py             # unified Engine: GGUF vs HF detection
│   ├── http_server.py        # FastAPI app: /v1/* endpoints
│   ├── protocol.py           # Pydantic models (OpenAI wire format)
│   └── backends/
│       ├── gguf.py           # GgufBackend + VRAM pre-flight
│       ├── hf.py             # HFBackend (Transformers)
│       └── llamacpp/         # pure-Python ctypes llama.cpp binding
└── plugins/                  # the plugin engine and every feature plugin
    ├── engine.py             # PluginManager: install/enable/active state, route mounting
    ├── contract.py           # the plugin contract (manifest schema, surfaces)
    ├── catalog.py            # discovers the builtin/ store + external installs
    ├── loader.py             # external plugin discovery (~/.localm/plugins/)
    ├── media_config.py       # shared ComfyUI config for the media plugins
    └── builtin/              # the read-only store (only chat is active by default)
        ├── chat/             # protected plugin #0: history, memory, personas
        ├── coder/            # the coding agent (agent loop, tools, MCP client)
        ├── image/            # FLUX image generation (consumes image_gen/)
        ├── music/            # ACE-Step music generation (consumes music_gen/)
        ├── video/            # Wan 2.2 video generation (consumes video_gen/)
        ├── rag/              # Knowledge: collections + cited retrieval
        ├── web/              # web_search + fetch_url under the network policy
        ├── voice/            # local Whisper speech-to-text
        ├── tts/              # in-browser Kokoro text-to-speech (client assets)
        └── mcp/              # `localm mcp` stdio server + MCP client
```

The `image_gen/`, `music_gen/`, and `video_gen/` packages are shared transport
helpers (the ComfyUI HTTP/websocket drivers); the user-facing image, music, and
video plugins under `builtin/` consume them.

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
| [docs/plugins.md](docs/plugins.md) | Authoring a plugin: manifest, tool exports, surfaces, privacy rules |
| [docs/plugin-interop.md](docs/plugin-interop.md) | Foreign-ecosystem interop: importing plugins from other tools over the MCP spine |
| [docs/plugin-architecture-plan.md](docs/plugin-architecture-plan.md) | The plugin-first re-architecture: vision, phases, and contract |
| [docs/gui.md](docs/gui.md) | The web GUI: chat, coder sessions, approvals, security notes |
| [docs/mcp.md](docs/mcp.md) | MCP in both directions: serving models, consuming tool servers |
| [docs/server-api.md](docs/server-api.md) | HTTP API details |
| [docs/architecture.md](docs/architecture.md) | Design notes |
| [docs/llamacpp-binding.md](docs/llamacpp-binding.md) | The ctypes binding internals |
| [docs/gpu-setup.md](docs/gpu-setup.md) | GPU/DLL setup |
| [docs/linux-setup.md](docs/linux-setup.md) | Running localm on Linux: venv, runtime, GPU notes |
| [docs/flux-setup.md](docs/flux-setup.md) | ComfyUI FLUX image pipeline |
| [docs/video.md](docs/video.md) | Wan 2.2 video generation: model setup, timing expectations, workflow override |
| [docs/rag.md](docs/rag.md) | Knowledge: chat with your documents, collections, retrieval design |
| [docs/network.md](docs/network.md) | Internet access for coder and chat: modes, domain rules, SSRF guard |
| [docs/tls.md](docs/tls.md) | API keys, TLS, and reverse proxies for LAN serving |

---

## License

MIT
