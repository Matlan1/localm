# CLI Reference

This page documents the full localm command-line interface. For a quick introduction, see the [README](../README.md#cli-reference).

## Core commands

```bash
localm run MODEL [opts]          # chat or single prompt
localm gui [MODEL] [opts]        # browser GUI (chat + coder + plugin tabs)
localm serve MODEL [opts]        # OpenAI-compatible server
localm benchmark MODEL [opts]    # TTFT and tok/s at increasing prompt sizes
localm coder [TASK] [opts]       # AI coding agent (coder plugin)
localm job ... [opts]            # scheduled recurring jobs (jobs plugin)
localm mcp [opts]                # MCP stdio server (mcp plugin)
localm abliterate --model M      # decensor M with Heretic, then register it
localm doctor                    # check Python, llama.dll, GPU driver, VRAM, packages
localm info                      # paths + current config
localm setup-llama [opts]        # provision native llama.cpp binaries
```

`--debug` on `gui`, `serve`, and `run` writes a log to `~/.localm/logs/` with request timing and the native llama.cpp stderr stream (including crash abort reasons), and shows raw model output without marker scrubbing.

### Running models

```bash
localm run mymodel --prompt "What is 42?"
localm run mymodel --system "You are terse." --prompt "How does TCP work?"
echo "Translate 'hello' to Japanese." | localm run mymodel
localm run mymodel                        # interactive chat
localm run mymodel --max-tokens 256       # max tokens to generate
localm run mymodel --temperature 0.5      # sampling temperature
localm run mymodel --ctx 8192             # context window (GGUF only)
localm run mymodel --gpu-layers 99        # GPU layers (GGUF only; 99=all)
localm run mymodel --image photo.jpg --prompt "Describe this image."
localm run mymodel --debug                # write debug log
localm run mymodel --mode privacy         # privacy/log/full persistence mode
```

MODEL can be a registered name or a direct path:

```bash
localm run gemma4-12b
localm run D:\models\llama3.gguf
localm run D:\hf-models\gemma-3-4b-it
```

### GUI

```bash
localm gui                       # picks the first registered model, opens browser
localm gui mymodel               # or name one
localm gui --no-model            # open model-less, straight to the Models page
localm gui --no-browser          # just start the server, open the URL yourself
localm gui --pull <spec>         # start a download immediately
localm gui -H 0.0.0.0            # bind to all interfaces (requires auth key)
localm gui --qr                  # [PoC] scannable QR of LAN URL for phones
```

Chat, the coder agent, model management, and any enabled plugin tabs in one page. The model preloads in the background so the first reply is fast. See [docs/gui.md](../docs/gui.md) for details.

### Server (OpenAI-compatible API)

```bash
localm serve mymodel
localm serve mymodel --port 8650            # explicit port
localm serve mymodel --ctx 8192             # context window
localm serve mymodel --gpu-layers 99        # GPU layers
localm serve mymodel -H 0.0.0.0              # bind to all interfaces
localm serve mymodel --debug                # debug logging
```

localm owns the port range 8642-8741: the default is 8642 and the server bumps to the next free port automatically when it is taken. Use it with any OpenAI client - set `base_url="http://localhost:8642/v1"` and `api_key="localm"`.

The streaming usage block adds `ttft_ms` and `tokens_per_sec`. See [docs/server-api.md](../docs/server-api.md) for the full API surface, including scope-gated management endpoints.

### Benchmark

```bash
localm benchmark mymodel
localm benchmark mymodel --gen-tokens 256         # tokens per run (default 128)
localm benchmark mymodel --prompts "64,512,2048"  # prompt sizes to test
localm benchmark mymodel --ctx 8192               # context window
localm benchmark mymodel --gpu-layers 99          # GPU layers
```

Runs a fixed prompt padded to each requested size, streams tokens, and reports time to first token, tokens per second, and total time.

---

## Model management

### Download models

```bash
# Specific GGUF from any HF repo (split files are handled automatically)
localm pull owner/repo:model-Q4_K_M.gguf

# Full HuggingFace model directory (transformers format)
localm pull owner/repo

# Direct URL, with optional integrity check
localm pull https://example.com/m.gguf --sha256 <hash>

# Alias for the downloaded model
localm pull owner/repo --name myalias
```

Duplicate downloads are detected by path and SHA256. When you add or pull something already registered, localm offers alias / copy / move / skip instead of silently duplicating gigabytes.

### Search HuggingFace

```bash
localm search qwen2.5 7b instruct            # find GGUF repos
localm search bartowski/Qwen2.5-7B-GGUF --files  # quants + sizes + VRAM fit
```

### Register existing models

```bash
localm add C:\models\mymodel.gguf
localm add D:\models\my-hf-model --name mymodel
localm add D:\ollama\manifests\registry.ollama.ai\library\<model>\<tag>
localm alias mymodel short                   # second name for the same file
```

### List and remove

```bash
localm list                     # registered models
localm models                   # available shortcuts
localm rm MODEL [--yes]         # alias-aware removal
```

`localm rm` only deletes the file when the last alias pointing at it is removed, and the confirmation prompt states exactly what will happen.

---

## Media generation

These are core CLI commands (they need only a running ComfyUI, not a plugin install). The GUI Media pages and the `/generate-*` chat commands belong to the media plugins. See [docs/flux-setup.md](../docs/flux-setup.md) and [docs/video.md](../docs/video.md) for model setup.

```bash
localm image "A cat on a sunny beach" -s 1024 1024
localm music "lofi, jazzy, mellow" --lyrics song.txt -d 180
localm video "a fox runs through snow" --duration 5
```

---

## Knowledge (RAG)

```bash
localm rag add NAME PATH...      # index files/folders into a collection
localm rag list                  # collections with doc/chunk counts
localm rag query NAME "text"     # show the top matching excerpts
localm rag rm NAME [--yes]       # delete a collection (index only, files kept)
```

Enable the rag plugin and install `pip install "localm[rag]"` for PDF parsing. See [docs/rag.md](../docs/rag.md) for retrieval design.

---

## Scheduled jobs

```bash
localm job add NAME --prompt "..." [--cron "0 9 * * 1-5" | --every SECONDS]
localm job add NAME --prompt "..." --coder --cwd DIR --scope "tests/**"
localm job list                  # id, name, schedule, state, last status
localm job run JOB_ID            # run once now, record the result
localm job enable JOB_ID
localm job disable JOB_ID
localm job remove JOB_ID         # delete the job and its results
```

The `localm job` CLI, the Jobs GUI tab, and the `/api/jobs` routes share one on-disk store. The scheduler only ticks while a `localm gui`/`localm serve` (with the jobs plugin active) is up. See [docs/jobs.md](../docs/jobs.md).

---

## Configuration

```bash
localm config temperature 0.7
localm config n_gpu_layers 99
localm config n_ctx 8192
localm config port 8650
localm config confirm_remove false
localm config comfy_launch_cmd "D:\path\to\comfyui.bat"
localm config heretic_path "D:\path\to\heretic"
localm config autoprune_missing_models true
```

Config lives at `~/.localm/config.json` and only known keys are settable (both the CLI and the GUI validate against the schema). Only the settings you actually changed are stored in the file; everything else follows the current built-in defaults, so a new localm version's improved defaults apply automatically unless you overrode them. Auth: configure an API key (`LOCALM_API_KEY` env var, the launcher's Auth card / `auth.key` file, or a named key) to require bearer auth on the HTTP API; auth is recommended before binding to anything other than 127.0.0.1, and the CLI warns about exposed unauthenticated binds. Set `require_auth true` (or `LOCALM_REQUIRE_AUTH=1`) to fail closed and refuse requests until a key exists. CORS is locked to localhost by default and can be widened with `cors_origins`. See [docs/tls.md](../docs/tls.md) for LAN serving.

### Dynamic context window

The context window starts at `n_ctx` (default 4096) and grows automatically when a conversation outgrows it, in `n_ctx_grow` steps (default 4096), up to `n_ctx_max` (default 16384). Small windows load fast; long chats get room when they need it; the ceiling keeps VRAM use predictable.

```bash
localm config n_ctx_max 32768    # raise the ceiling
localm config n_ctx_grow 8192    # grow in bigger steps (fewer rebuilds)
localm config ctx_auto false     # use the fixed n_ctx_max ceiling instead of VRAM sizing
```

`ctx_auto` is **on by default**: localm measures free VRAM at load time, subtracts the model weights and a fixed overhead, and sizes the ceiling from what remains (clamped to 4k-64k). When a conversation reaches the ceiling, replies shorten to fit; when even that is impossible you get a clear error instead of an out-of-memory crash. An explicit `-c/--ctx` larger than the ceiling always wins.

Reading free VRAM needs `torch` (the `[gpu]` extra). On a CPU-only install without it, `ctx_auto` cannot measure VRAM and falls back to a fixed 16384-token ceiling (to change that ceiling on a CPU-only install, set `ctx_auto false` and raise `n_ctx_max`); the pre-flight "Low VRAM" warning and the post-load VRAM usage line are skipped for the same reason.

Long chats compact automatically before they collide with the ceiling: at 70% fill, older turns are summarised by the model and replaced with a short summary, keeping the last two exchanges verbatim. If summarisation is unavailable the history is trimmed with a visible note instead.

---

## Shell completion

```bash
localm completion powershell   # also: bash, zsh, fish
```

Model names complete everywhere a model argument is expected.

---

## Plugins

```bash
localm plugin status            # what is installed and which installs are active
localm plugin install NAME      # copy NAME from the store and enable it
localm plugin enable NAME       # enable an already-installed plugin
localm plugin disable NAME      # disable but keep it installed
localm plugin uninstall NAME    # remove it (add --delete-data to drop its data)
localm plugin setup             # pick a starter set interactively
localm plugin install-deps NAME # install a plugin's pip extras on this host
localm plugin install-deps --all# fill in missing extras for every enabled plugin
```

The store names are `coder`, `image`, `music`, `video`, `rag`, `web`, `voice`, `tts`, `jobs`, and `mcp` (plus the protected `chat`). Plugins with heavy Python dependencies carry them in a pip extra: by default `install`/`enable`/`setup` install it for you on the host (the `auto_install_plugin_deps` setting; pass `--no-deps` to skip, or `--with-deps` to force). A remote client never triggers a server-side pip - it is told to install on the host, e.g. with `localm plugin install-deps`. A running GUI server picks up new HTTP routes and tabs at runtime, while stdio plugins like mcp take effect on the next `localm mcp`.

Third-party plugins are folders containing a `plugin.toml` manifest and Python files. Install from a local path with `localm plugin install <path>` (the same command takes a store name or a directory); installation is a local directory copy, fully offline. See [docs/plugins.md](../docs/plugins.md) for the full authoring contract.

---

## Coding agent

```bash
localm coder --model mymodel              # interactive session in the current repo
localm coder "fix the failing test"       # single task
localcoder --model mymodel                # same thing, standalone entry point (installed with the coder plugin)
localm coder --system "always run pytest before finishing"   # custom instructions for this run
```

The agent auto-starts `localm serve` when needed, plans with tool calls (read, write, edit, patch, shell, search, tests, image generation, plus tools exported by other installed plugins), asks before destructive actions, tracks a turn budget so it asks for help instead of guessing forever, and verifies its own code changes before answering. Privacy mode is the default: nothing is persisted unless you opt into `--mode log` or `--mode full`.

Give the agent standing guidance (conventions, style, constraints) with a `.localcoder/system.md` file in the repo - it is injected into the system prompt under "## User Instructions" for every session in that project. The `--system TEXT` flag overrides the file for a single run. This is separate from `LOCALCODER.md`, which is auto-managed project memory (facts the agent appends via `/remember` and its own reflection).

---

## MCP server

```bash
localm mcp --print-config     # JSON block for Claude Desktop and friends
localm mcp [--stdio]          # stdio transport (default)
```

Exposes your local models (chat, list_models, embed, generate_image) to Claude Desktop and other MCP clients. See [docs/mcp.md](../docs/mcp.md) for both directions: localm as an MCP server, and the coder consuming external MCP tool servers.

---

## Abliteration (decensor)

```bash
localm abliterate --model Qwen/Qwen3-4B-Instruct-2507        # HF repo or local path
localm abliterate --model ./model.gguf --export-gguf q5_k_m  # also emit a GGUF
localm abliterate --model <id> --name decensored             # custom registry name
localm abliterate --model <id> --print-command               # preview, don't launch
```

Hands a model off to [Heretic](https://github.com/Matlan1/heretic-win-AMD) to remove refusals ("safety alignment"), then registers the result so you can run it like any other model. Heretic is a **separate program** (AGPL-3.0): localm never bundles or imports it, only runs it. If it is not found, localm offers to clone the fork into a gitignored `.heretic/` under your data dir; point at an existing checkout with the `heretic_path` config key (or the `LOCALM_HERETIC_PATH` env var). Enable with `pip install "localm[abliterate]"`.

---

## Setup and diagnostics

```bash
localm setup-llama                       # auto-detect the GPU, fetch the right backend
localm setup-llama --backend vulkan      # any GPU (AMD/NVIDIA/Intel), no vendor toolkit
localm setup-llama --backend cuda        # NVIDIA  /  --backend amd-rocm (AMD)  /  --backend cpu
localm setup-llama --from <build-dir>    # or copy your own llama.cpp build

localm doctor                            # check Python, llama.dll, GPU driver, VRAM, packages
localm info                              # data directory, config file, registry, registry file
```

See [docs/gpu-setup.md](../docs/gpu-setup.md) for the full GPU setup guide.
