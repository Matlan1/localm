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
localm doctor                    # check Python, llama.dll, GPU driver, VRAM, packages
localm info                      # paths + current config
localm setup-llama [opts]        # provision native llama.cpp binaries
```

`--debug` on `gui`, `serve`, and `run` writes a log to `<data dir>/logs/` with request timing and the native llama.cpp stderr stream (including crash abort reasons), and shows raw model output without marker scrubbing.

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
localm gui --no-browser          # start the server, open the URL yourself
localm gui --pull <spec>         # start a download immediately
localm gui -H 0.0.0.0            # bind to all interfaces (requires auth key)
localm gui --qr                  # [PoC] scannable QR of LAN URL for phones
```

Chat, the coder agent, model management, and any enabled plugin tabs in one page. The model preloads in the background so the first reply is fast. See [docs/gui.md](../docs/gui.md) for details.

### Server (OpenAI-compatible API)

```bash
localm serve mymodel
localm serve mymodel --port 8650            # explicit port (must be free, else errors)
localm serve mymodel --ctx 8192             # context window
localm serve mymodel --gpu-layers 99        # GPU layers
localm serve mymodel -H 0.0.0.0              # bind to all interfaces
localm serve mymodel --debug                # debug logging
```

localm owns the port range 8642-8741: the default is 8642, and when no `--port` is given the server bumps to the next free port in that range automatically if it is taken. An explicit `--port` is different - it must be free, or startup errors rather than moving you onto another port. Use it with any OpenAI client - set `base_url="http://localhost:8642/v1"` and `api_key="localm"`.

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

# Force a fresh download even if an identical model is already registered
localm pull owner/repo:model-Q4_K_M.gguf --redownload

# Download an mmproj vision projector alongside the main model
localm pull owner/repo:model-Q4_K_M.gguf --mmproj mmproj-model-f16.gguf

# Set the model type explicitly instead of relying on auto-detection
localm pull owner/repo:model-Q4_K_M.gguf --type llm
```

Duplicate downloads are detected by path and SHA256. When you add or pull something already registered, localm offers alias / copy / move / register / skip instead of silently duplicating gigabytes. Two downloads to the same destination (from the CLI, the GUI, or both) are serialised rather than allowed to interleave their bytes into one file; starting a download that is already running is refused up front, and one interrupted by a crash resumes instead of restarting.

### Search HuggingFace

```bash
localm search qwen2.5 7b instruct            # find GGUF repos
localm search bartowski/Qwen2.5-7B-GGUF --files  # quants + sizes + VRAM fit
```

### Register existing models

```bash
localm add D:\models\mymodel.gguf
localm add D:\models\my-hf-model --name mymodel
localm add D:\ollama\manifests\registry.ollama.ai\library\<model>\<tag>
localm add D:\models\mymodel.gguf --on-duplicate alias   # ask|alias|copy|move|register|skip
localm add D:\models\bulk-dir --fast                     # dedup on file size, skip the SHA256 hash
localm alias mymodel short                   # second name for the same file
localm rename mymodel newname                # rename it outright (unlike alias, the old name stops working)
```

Duplicate detection is two-tier: the resolved path first, then the file's SHA256 against digests already in the registry (large files hash off the main thread with a progress bar; `--fast` skips the hash and dedups on file size alone; `--no-hash` disables content-level dedup entirely). `--on-duplicate` picks the answer non-interactively instead of prompting (default `ask`).

By default `add` (and `pull` with a local path) registers the file where it already is - nothing is copied or moved. Pass `--store copy` or `--store move` to bring it into `<data dir>/models` first and register it from there instead, so it's managed exactly like a pulled model:

```bash
localm add D:\models\mymodel.gguf --store copy   # duplicate into <data dir>/models, keep the original
localm add D:\models\mymodel.gguf --store move   # relocate into <data dir>/models
```

`--store` moves/copies a split GGUF's every part and a sibling mmproj vision-projector file together with the model, so multi-part loading and vision capability survive the move. It refuses (no changes made) if a different file already occupies that name in `<data dir>/models`, if there isn't enough free disk space, or if a copy's SHA256 doesn't match the original afterward.

### Model type

Every registered model has a type, detected deterministically from the file
itself, never from fuzzy tag matching: a GGUF or Ollama blob is an `llm`; a
HuggingFace directory is read from its `config.json` architectures (or
`adapter_config.json` for a `lora`). Anything without a hard signal is left as
`unknown` rather than guessed. The types are `llm`, `mmproj`, `diffusion-unet`,
`text-encoder`, `vae`, `lora`, `embedding`, and `unknown`.

An `unknown` model still runs when you name it explicitly (`localm run NAME`, or an
API request naming it), but is never auto-picked as the default chat model, so a
diffusion checkpoint or text encoder cannot get loaded as if it were a chat model.
Correct a misdetected or `unknown` type at any time:

```bash
localm set-type MODEL llm        # types: llm mmproj diffusion-unet text-encoder vae lora embedding unknown
```

Registering a lone `.safetensors` file scans its parent directory: if that folder
is a real HuggingFace model (config plus weights and tokenizer), the folder is
registered; otherwise the file is rejected with an "incomplete model" message
rather than added as a half-model.

### List and remove

```bash
localm list [--type TYPE]       # registered models, optionally filtered by type
localm models                   # available shortcuts
localm rm MODEL [--yes]         # alias-aware removal
localm relocate MODEL NEW_PATH  # re-point a registered model after you moved its file
localm unload [MODEL]           # free VRAM on the running server: all models, or just one
```

`localm rm` only deletes the file when the last alias pointing at it is removed, and the confirmation prompt states exactly what will happen. It also refuses when a running localm server (this directory's or another one's) still has the model's file loaded - checked regardless of `--yes`, before the confirmation prompt, since this one-shot process shares no memory with a server and has to ask over HTTP; a server it cannot reach counts as "cannot rule it out", not as a green light. `localm unload` talks to the server serving the current directory (or `LOCALM_URL`); with no argument it unloads every model, and a named model that is not loaded is a no-op.

---

## Media generation

These are core CLI commands (they need only a running ComfyUI, not a plugin install). The GUI Media pages and the `/generate-*` chat commands belong to the media plugins. See [docs/flux-setup.md](../docs/flux-setup.md) and [docs/video.md](../docs/video.md) for model setup.

```bash
localm image "A cat on a sunny beach"
localm music "lofi, jazzy, mellow" --lyrics song.txt -d 180
localm video "a fox runs through snow" --duration 5
```

### localm's own ComfyUI (optional)

localm can run its own managed ComfyUI instead of depending on your install, so it
can pin a known-good version and carry fixes. Off by default; your own ComfyUI is
never modified. Full guide: [docs/managed-comfyui.md](../docs/managed-comfyui.md).

```bash
localm comfy setup                 # provision it (copies your ComfyUI, or a fresh hardware-matched install)
                                    # media routes to it right away (comfy_target defaults to "own")
localm comfy status                # is one installed, which ComfyUI is targeted, and is it running
localm comfy update                # advance to the shipped pinned version, re-apply localm's patches
localm comfy remove [--models]     # delete it (keeps the managed models unless --models)
```

`localm comfy setup` takes `--copy-custom-nodes` / `--no-custom-nodes` (copy path
only; you are asked when custom nodes are present). `localm comfy update` takes
`--reinstall-requirements` and, for testing, `--commit <sha>`.

### Starting and stopping ComfyUI

These drive whichever ComfyUI localm targets - the managed one or your own.

```bash
localm comfy status                # ...including whether it is running, and whether localm launched it
localm comfy status --no-ping      # skip the liveness check (offline and instant)
localm comfy start                 # start it without running a generation
localm comfy start --media music   # start it with the music plugin's own ComfyUI settings
localm comfy stop                  # abort the render, clear the queue, free its VRAM
localm comfy restart               # stop it, then launch a fresh one
```

These need a running localm server (`localm gui` or `localm serve`), because the
handle on a ComfyUI localm launched belongs to the process that launched it -
only that process can end it. `localm comfy stop` never kills a ComfyUI you
started yourself: it aborts the render and frees the VRAM, and says so rather
than implying the process is gone. A cold start compiles GPU kernels and can take
minutes, so `start` and `restart` wait for the configured `comfy_launch_timeout`.

Pressing Ctrl-C during `localm image` / `localm music` / `localm video` now tells
ComfyUI to abort that render and free its VRAM, instead of leaving it running.

Whether localm targets the managed instance is decided by one setting,
`comfy_target` (`own` by default, or `user` to force your own ComfyUI) - localm
uses the managed instance only when it is `own` AND an instance is installed;
otherwise it uses your own ComfyUI.

### ComfyUI workflows

Each media plugin can generate with an uploaded ComfyUI workflow (in ComfyUI:
Save > API format) instead of the shipped default - the same picker the GUI's
Image/Music/Video pages expose. This governs `localm image`/`music`/`video`
too, not just the GUI, and works fully offline (no running server needed).

```bash
localm comfy workflow list image                    # uploaded workflows, and which is active
localm comfy workflow add image my_flux.json --use   # upload, and select it
localm comfy workflow use image my_flux.json         # select an already-uploaded one
localm comfy workflow use image --clear              # back to the shipped default
localm comfy workflow rm image my_flux.json          # delete (refuses the active one)
```

`localm plugin config <name> workflow <file>` reaches the same selection - it
is what `add --use`/`use` write to - but only `workflow list` shows what is
actually uploaded.

---

## Knowledge (RAG)

```bash
localm rag add NAME PATH... [--embed] [--force]   # index files/folders into a collection
localm rag list                  # collections with doc/chunk counts
localm rag docs NAME             # list a collection's documents: path, chunks, status
localm rag query NAME "text" [--embed] [-k N]   # show the top matching excerpts
localm rag resync NAME [--embed] [--prune-missing]  # re-walk the indexed folders: pick up new and
                                 # changed files, flag ones that have vanished
localm rag repair NAME [--embed]  # re-index every known document from scratch
localm rag reembed NAME          # recompute vectors with the current embedding model, without re-reading source files
localm rag rm-doc NAME PATH      # remove one document from the index (file kept)
localm rag rm NAME [--yes]       # delete a collection (index only, files kept)
```

CLI indexing is lexical-only (BM25) by default, with no engine involved; pass
`--embed` to `add`/`query`/`resync`/`repair` to also compute embeddings via a
running localm server (`--url` to point at a non-default one), matching the
GUI's hybrid retrieval.

`resync` is the incremental one to run regularly (`--prune-missing` to also drop
entries whose file is gone; off by default so an unplugged drive cannot delete
your index). Put it on a schedule with
`localm job add sync-docs --rag --collection NAME --cron "0 3 * * *"` - see
[docs/jobs.md](../docs/jobs.md#keeping-an-indexed-folder-current).

A named API key's RAG *indexing* reach can be confined to specific folders
with `localm key create NAME --scope rag --rag-root DIR` (repeatable) - see
[API keys](#api-keys) and [docs/rag.md](../docs/rag.md#per-key-folder-scoping).

Enable the rag plugin and install `pip install "localm[rag]"` for PDF parsing. See [docs/rag.md](../docs/rag.md) for retrieval design.

---

## Scheduled jobs

```bash
localm job add NAME --prompt "..." [--cron "0 9 * * 1-5" | --every SECONDS]
localm job add NAME --prompt "..." --coder --cwd DIR --scope "tests/**"
localm job add NAME --memory --every 21600         # synthesise memory facts, no prompt needed
localm job add NAME --rag --collection COLL --cron "0 3 * * *"   # keep a Knowledge collection current
localm job list                  # id, name, schedule, state, last status
localm job show JOB_ID           # full definition: schedule, prompt, cwd/scope, allow_shell
localm job run JOB_ID            # run once now, record the result
localm job results JOB_ID [--limit N] [--offset N]  # past run results, newest first
localm job enable JOB_ID
localm job disable JOB_ID
localm job remove JOB_ID         # delete the job and its results
```

A job's `--model NAME` picks a specific registered model to run it with (chat
and coder jobs; otherwise it uses the server's active model), and a coder job
needs `--allow-shell` to run unrestricted (off by default: read plus confined
edits, no shell, no network).

The `localm job` CLI, the Jobs GUI tab, and the plugin's `/api/jobs` routes share one on-disk store. The scheduler only ticks while a `localm gui`/`localm serve` (with the jobs plugin active) is up. See [docs/jobs.md](../docs/jobs.md).

---

## Memory

```bash
localm memory list [--all] [--json]        # what localm has remembered about you
localm memory show ID                      # one fact in full, with its provenance
localm memory add "fact" [--kind semantic|episodic] [--importance F]
localm memory forget ID [-y]               # delete one fact (not recoverable)
localm memory forgotten                    # facts localm archived on its own
localm memory restore ID                   # bring an archived fact back
localm memory corrections [--json]         # proposals consolidation left for review
localm memory accept ID | reject ID        # resolve one proposal
localm memory clear [-y]                   # erase everything (not recoverable)
```

Reads and writes the same store the app and the `memory` plugin's chat recall
use, and works whether or not that plugin is installed - the plugin only adds
the automatic recall-on-chat and background consolidation hooks. See
[docs/memory.md](../docs/memory.md).

---

## Configuration

```bash
localm config temperature 0.7
localm config n_gpu_layers 99
localm config n_ctx 8192
localm config port 8650
localm config confirm_remove false
localm config comfy_launch_cmd "D:\path\to\comfyui.bat"
localm config autoprune_missing_models true
```

Config lives at `<data dir>/config.json` and only known keys are settable (both the CLI and the GUI validate against the schema). Only the settings you actually changed are stored in the file. Set `require_auth true` (or `LOCALM_REQUIRE_AUTH=1`) to fail closed and refuse requests until a key exists; localm warns when binding to a non-loopback address without a key; `cors_origins` widens CORS (locked to localhost by default). See the [API keys](#api-keys) section and [SECURITY.md](../SECURITY.md) for the auth and scope model, and [docs/tls.md](../docs/tls.md) for LAN serving.

### Dynamic context window

The context window starts at `n_ctx` (default 4096) and grows automatically when a conversation outgrows it, in `n_ctx_grow` steps (default 4096), up to `n_ctx_max` (default 16384). Small windows load fast; long chats get room when they need it; the ceiling keeps VRAM use predictable.

```bash
localm config n_ctx_max 32768    # raise the ceiling
localm config n_ctx_grow 8192    # grow in bigger steps (fewer rebuilds)
localm config ctx_auto false     # use the fixed n_ctx_max ceiling instead of VRAM sizing
```

`ctx_auto` is **on by default**: it sizes the ceiling from free VRAM at load time (clamped to 4k-64k), overriding `n_ctx_max`. When a conversation reaches the ceiling, replies shorten to fit; when even that is impossible you get a clear error instead of an out-of-memory crash. An explicit `-c/--ctx` larger than the ceiling always wins.

Reading free VRAM needs `torch` (the `[gpu]` extra). On a CPU-only install without it, `ctx_auto` cannot measure VRAM and falls back to a fixed 16384-token ceiling (set `ctx_auto false` and raise `n_ctx_max` to change it).

Long chats compact automatically before they collide with the ceiling. See [docs/architecture.md](../docs/architecture.md) for the compaction and VRAM-sizing details.

### Multi-GPU: picking the main device

On a multi-GPU system, localm loads models onto device 0 by default. `localm gpus` lists every detected device (index, name, VRAM) and marks the configured one:

```bash
localm gpus                        # list detected GPUs
localm config main_gpu_index 1     # load models onto device 1 instead
```

The GUI has the same control: Settings > Live tuning shows a "Main GPU" dropdown once more than one GPU is detected. An index that no longer matches a currently-detected device falls back to device 0 with a logged warning rather than silently loading onto the wrong card.

`localm gpus` omits the free-VRAM figure (printing a note instead) rather than show a number it cannot stand behind, when the reading is process-scoped or otherwise untrusted for the platform - notably AMD ROCm/HIP on Windows.

### Mixture-of-Experts: reducing VRAM footprint

`n_cpu_moe` keeps the expert weights of the first N layers of a MoE model in
system RAM while the rest still runs on the GPU, off by default:

```bash
localm config n_cpu_moe 16        # keep 16 layers' worth of experts on CPU
localm config n_cpu_moe 0         # off - the default
```

It is a footprint dial, not a speedup - at the same VRAM it runs at about the
same speed - so it is worth reaching for when a model would not otherwise fit,
or when something else needs the card. No effect on a normal (dense) model.
The Settings page has the same field ("MoE expert layers on CPU"). See
[docs/gpu-setup.md](../docs/gpu-setup.md#mixture-of-experts-reducing-vram-footprint)
for measured VRAM numbers.

### Multi-GPU: splitting one model across several cards

A model too large for any single card's VRAM can load using the combined VRAM of 2 or more GPUs, instead of picking just one:

```bash
localm gpus                              # (split) marks any device in the split
localm config gpu_split_indices 0,1      # split the model across devices 0 and 1
localm config gpu_split_ratios 3,1       # optional: PIN device 0 to three times device 1's share
localm config gpu_split_indices ""       # clear the split - back to a single GPU (main_gpu_index)
```

How much of the model lands on each card is figured out automatically: at load time localm reads every split device's free VRAM and sizes each card's share proportionally, so a card that is half-occupied gets a half-sized share instead of an equal one that would not fit. Set `gpu_split_ratios` only to pin exact weights (that disables the automatic distribution); when free VRAM cannot be measured per device, the split falls back to even shares, and the decision is logged either way.

GGUF models use llama.cpp's native layer-split; HF (transformers) models use accelerate's `device_map="auto"` restricted to just the listed devices. Fewer than 2 currently-detected devices in `gpu_split_indices` (a stale index, or only one still present) falls back to the single-GPU behavior above, with a logged warning - it never crashes a load. On the `vulkan` runtime build the indices are passed to the native loader as-is (torch and nvidia-smi cannot see or number Vulkan-only devices, so there is nothing to cross-check them against); there the numbers mean the Vulkan backend's own device order, which is exactly what the GUI's selectors list on that build. The GUI has the same control: Settings > Live tuning shows "Split across GPUs" checkboxes next to the Main GPU dropdown, and the model search results hint when a model would fit split across your GPUs but not on the largest one alone.

### Keeping more than one model loaded

localm already keeps several models resident at once when they fit: loading a
second model does **not** evict the first if a live free-VRAM reading shows it
fits alongside (the model's estimated need plus a 1 GB headroom, and, with a
split configured, enough room on every split device). Only when it does not fit
is a resident model evicted, least-recently-used first, and never one that is
currently generating. When free VRAM cannot be measured, or the reading is
inconclusive, localm deliberately stays single-resident rather than stacking
models until the driver runs out. This applies to the HTTP server (`localm
serve` / `localm gui`) and the MCP server (`localm mcp`) alike.

Two optional knobs override that arithmetic when you would rather decide
yourself:

```bash
localm config max_resident_models 2         # keep at most 2 models loaded at once
localm config max_resident_models 1         # strict single-resident (evict on every switch)
localm config pinned_models tiny-a,tiny-b   # never evict these to make room
localm config max_resident_models ""        # clear the cap - back to free-VRAM arithmetic
localm config pinned_models ""              # clear every pin
```

Both are off by default, so behavior is unchanged unless you set them. The cap
bounds how many models stay loaded regardless of how much headroom there is;
pinning protects named models from being chosen as the eviction victim (it never
loads a model on its own). If pins leave nothing that can be evicted, the load
still goes ahead and the missed policy is logged rather than silently dropped.

**Note:** the native 2-device split load path is verified end to end (real device enumeration, a lopsided configured ratio honored in the per-layer placement, coherent inference across the split) against two genuinely distinct Vulkan devices, and the ratio/index validation and parameter wiring are unit-tested. Not yet verified on real multi-GPU hardware: VRAM pressure and out-of-memory behavior across cards (the second test device is a software one backed by system RAM), and the amd-rocm (HIP) multi-device path. Report an issue if a split load misbehaves.

---

## API keys

localm's HTTP surface is protected by a bearer key. Manage the owner key and mint named, scope-limited keys from the CLI:

```bash
localm key show [--reveal]      # show the active owner key (masked, or full with --reveal) or "open mode"
localm key generate             # generate a random owner key, persist it, print it once
localm key set KEY              # persist a specific owner key you provide
localm key clear                # remove the owner key, return to open mode
localm key create NAME --scope chat --scope rag   # mint a named key limited to those scopes
localm key create NAME -s rag --fs-access host --rag-root D:\docs --expires-in 86400
localm key list                 # list named keys (metadata only, never the secret)
localm key rm KEY_ID            # revoke a named key by ID
localm key recover              # recover owner access after a lockout (run locally on the server machine)
```

The owner key can also be set outside these commands: the `LOCALM_API_KEY` env var, or a `<data dir>/auth.key` file.

`key create` takes three optional confinements beyond `--scope`, each narrowing what the key can do below what the owner can:

- `--fs-access [none|host]` - host filesystem reach for the coder/file tools. Defaults to `none` (device-upload-only); `host` grants the whole server disk, same as the owner.
- `--rag-root DIR` (repeatable) - confines the key's RAG **indexing** reach to exactly these folders instead of the global folder policy; querying and listing collections are not scoped by this. Omit for no per-key restriction. See [docs/rag.md](../docs/rag.md#per-key-folder-scoping).
- `--expires-in SECONDS` - the key stops working this many seconds after creation. Omitted means it never expires.

Privileged scopes (`config:write`, `plugins:admin`, `keys:admin`, `admin`, `coder:full`) are refused by `key create` unless you pass `--allow-privileged`; an owner-authenticated `POST /v1/keys` API call can mint them too. See [SECURITY.md](../SECURITY.md) for the auth and scope model, and [docs/tls.md](../docs/tls.md) for serving over a LAN.

---

## Shell completion

```bash
localm completion powershell   # also: bash, zsh, fish
```

In bash/zsh/fish, model names complete everywhere a model argument is expected. In PowerShell, model-name completion only covers `run`, `serve`, `rm`, and `alias`; other commands that take a model argument (`relocate`, `set-type`, `unload`, `rename`, `benchmark`, `gui`) complete only the command name, not the model.

---

## Plugins

```bash
localm plugin status            # what is installed and which installs are active
localm plugin install NAME      # copy NAME from the store and enable it
localm plugin enable NAME       # enable an already-installed plugin
localm plugin disable NAME      # disable but keep it installed
localm plugin uninstall NAME    # remove it (add --delete-data to drop its data)
localm plugin refresh [NAME]    # re-sync installed first-party plugin(s) with the bundled store
localm plugin setup             # pick a starter set interactively
localm plugin install-deps NAME # install a plugin's pip extras on this host
localm plugin install-deps --all# fill in missing extras for every enabled plugin
localm plugin config NAME       # show one plugin's own settings and their values
localm plugin config NAME KEY   # show one setting
localm plugin config NAME KEY VALUE   # set it (a blank VALUE clears the override)
```

`plugin config` edits the per-plugin settings blocks the GUI edits under
Settings. `localm config` cannot reach them: it writes top-level keys, while
these live under a plugin of their own.

```bash
localm plugin config image workdir /srv/comfy   # for image only, not music/video
localm plugin config video float_type fp16
localm plugin config music use_config_from image  # reuse image's backend settings
localm plugin config image workflow mine.json     # pick an uploaded workflow
localm plugin config tts voice af_heart
```

`image`, `music`, `video` and `tts` work offline, and each can hold its own
value for a field instead of sharing the global `comfy_*` default. Any other
plugin declares its settings as it loads, so listing or setting those needs a
running localm (found the way `localm status` finds it, or set `LOCALM_URL`).

The store names are `coder`, `image`, `music`, `video`, `rag`, `web`, `memory`, `voice`, `tts`, `jobs`, and `mcp` (plus the protected `chat`). Plugins with heavy Python dependencies carry them in a pip extra, installed on the host by default (the `auto_install_plugin_deps` setting; `--no-deps` to skip, `--with-deps` to force, or `localm plugin install-deps` later). A running GUI server picks up new HTTP routes and tabs at runtime; stdio plugins like mcp take effect on the next `localm mcp`. See [docs/plugins.md](../docs/plugins.md).

Third-party plugins are folders containing a `plugin.toml` manifest and Python files. Install from a local path with `localm plugin install <path>` (the same command takes a store name or a directory); installation is a local directory copy, fully offline. See [docs/plugins.md](../docs/plugins.md) for the full authoring contract.

---

## Coding agent

```bash
localm coder --model mymodel              # interactive session in the current repo
localm coder "fix the failing test"       # single task
localcoder --model mymodel                # same thing, standalone entry point (installed with the coder plugin)
localm coder --system "always run pytest before finishing"   # custom instructions for this run
localm coder "make the suite pass" --until "pytest -x"       # one-shot, verified by exit code
localm coder --model mymodel --verify "pytest -x"            # interactive, same check per turn
localm coder --model mymodel --seed 1234                     # reproducible sampling
```

The agent auto-starts `localm serve` when needed and plans with tool calls: reading and searching the project, writing and patching files, running the shell and the test suite, git, web fetch and search, image generation, Knowledge-collection search, background jobs, and delegating to sub-agents. Plus any MCP server tools, plugin-exported tools, and Agent Skills available in that project. The full list is in [Built-in tools](#built-in-tools) below. It asks before destructive actions, tracks a turn budget so it asks for help instead of guessing forever, and verifies its own code changes before answering. Privacy mode is the default: nothing is persisted unless you opt into `--mode log` or `--mode full`.

### Built-in tools

The agent ships 34 built-in tools. "Confirms" means the agent stops and asks
before running it by default; "Restricted" means a shared, non-owner session may
use it at all. The GUI coder has the same set (see [gui.md](gui.md)).

| Tool | What it does | Confirms | Restricted |
|---|---|---|---|
| `read_file` | Read a file, truncating large ones (re-read a region with offset) | no | yes |
| `list_dir` | List a directory | no | yes |
| `tree` | Recursive tree with sizes, skipping noise dirs | no | yes |
| `search_files` | Find files matching a glob | no | yes |
| `grep` | Regex search over file contents, skipping noise dirs | no | yes |
| `git_status` | Working-tree status | no | yes |
| `git_diff` | Unstaged diff, or staged with `staged=true` | no | yes |
| `git_log` | Recent commits, oneline | no | yes |
| `read_todos` | Read back the task list written with `set_todos` | no | yes |
| `set_todos` | Record the session task list (replaces the previous one) | no | yes |
| `write_file` | Write or overwrite a file | yes | yes |
| `edit_file` | Replace the first occurrence of exact text | yes | yes |
| `edit_files` | Apply the same exact-string edit across several files, all-or-nothing | yes | yes |
| `edit_notebook_cell` | Replace one cell's source in a `.ipynb` | yes | yes |
| `patch_file` | Apply a unified diff, more reliable than `edit_file` for multi-hunk | yes | yes |
| `search_replace` | Regex search and replace across many files | yes | yes |
| `read_env` | Read `.env` and environment, with secret-looking values masked | no | no |
| `run_tests` | Run the detected suite (pytest, cargo, go, npm/yarn) | no | no |
| `run_shell` | Execute a shell command | yes | no |
| `run_shell_background` | Start a shell command in the background, returning a job id | yes | no |
| `check_shell_job` | Check a background shell job's state and output | no | no |
| `kill_shell_job` | Stop a background job and its process tree | yes | no |
| `git_commit` | Stage and commit | yes | no |
| `git_create_branch` | Create a branch, optionally checking it out | yes | no |
| `git_push` | Push the current branch | yes | no |
| `spawn_agent` | Spawn a sub-agent for a sub-task and return its result | yes | no |
| `spawn_agent_background` | Start a sub-agent in the background, returning a job id | yes | no |
| `check_agent_job` | Check a background sub-agent's state and result | no | no |
| `dispatch_parallel` | Run up to 2 sub-tasks at once, each in its own git worktree | yes | no |
| `fetch_url` | Fetch a URL and return its text, HTML stripped | see below | no |
| `web_search` | Search the web, returning titles, URLs and snippets | see below | no |
| `generate_image` | Generate or refine an image with the local pipeline | no | no |
| `rag_list_collections` | List indexed Knowledge collections and their stats | no | no |
| `rag_search` | Search a named Knowledge collection for excerpts | yes | no |

Four things the table cannot express, and each of them matters:

- **`fetch_url` and `web_search` follow the network policy, not this column.**
  They confirm when `net_mode` is `ask` (the default) and run without asking when
  it is `allow`. See [network.md](network.md).
- **"Confirms: no" is not "harmless".** `run_tests` and `generate_image` both
  have real side effects and neither asks by default. `run_tests` runs your
  project's own test command, which can do anything that command does.
- **Restricted sessions are allowlisted, not denylisted.** A newly added tool is
  unavailable to a shared session until it is explicitly added, so this column
  fails closed rather than open.
- **The `--scope` glob confines file tools only.** `run_shell` and `run_tests`
  start a process, which a path check cannot bound, so a command can still reach
  outside the scope. Disable those tools for a hard boundary.

**Verification by exit code.** The agent's own "I am done" is not evidence, so localm
judges a change by running a command and reading its exit code - the harness runs it,
not the model, which is what makes it un-gameable. For a one-shot task that is
`--until "pytest -x"`: the task is retried (up to `--goal-max-iters`, default 5) until
the command exits 0, and a run that never gets there exits non-zero rather than
claiming success. Interactive sessions (the REPL and the GUI coder) run the same check
at the moment the agent would otherwise finish a turn that changed files. The command
defaults to the project's obvious one - `cargo test`, `go test ./...`, `npm test` when
package.json defines a test script, or pytest when the project has a pytest setup -
and only when that runner is actually installed: a Rust project on a machine without
cargo gets no check rather than one that can only fail. A project with no detectable
check simply runs without one. If the command cannot be started at all, that is
reported as "could not run - nothing was actually verified", which is neither a pass
nor a failure charged to the agent; a command that runs and fails is retried as
normal, including when it fails because something it calls is missing. Override it with
`--verify COMMAND`, a `verify = "..."` key in `.localcoder/config.toml`, or `/verify`
mid-session; turn it off with `--no-verify` or `/verify off`. When the check keeps
failing, the agent is told (and told not to edit the check to force a pass); when the
attempts run out, the turn is reported as NOT verified rather than as done. Sessions
opened with a shared, scoped key never run a verify command - those sessions have no
process execution at all. In the GUI the same three controls sit in the coder setup
form: a verification command, a fix-attempt cap, and a skip-verification toggle.

**Reproducible runs.** `--seed N` pins the sampler's RNG, so the same seed with the same
model, prompt and settings reproduces the same output. Measured bit-for-bit on one AMD
gfx1030 box with the bundled llama.cpp runtime and Qwen2.5-Coder-7B Q6_K: 5/5 identical
responses at `temperature 0.8` with a seed, 5/5 *different* without one, and identical
again after a full model reload. That is one hardware and software combination, not a
guarantee: different GPUs, backends, llama.cpp builds, or concurrent load were not
measured, and `--anthropic` ignores the flag because the Anthropic API has no seed
parameter.

**Episodic memory.** The agent keeps a per-project log of lessons learned across sessions. Manage it without starting a session:

```bash
localm coder --episodes                    # list stored lessons and exit
localm coder --episodes-archive            # list dropped (merged/evicted/forgotten) lessons and exit
localm coder --forget-episode ID           # drop one lesson (archived, recoverable) and exit
localm coder --forget-episodes             # erase ALL episodic memory for this project and exit
localm coder --restore-episode ID          # bring an archived lesson back into recall and exit
localm coder --consolidate-episodes        # ask the model to merge related lessons (opt-in, manual only)
```

The GUI offers all of this under **lessons** in the coder setup panel: a stored and a
dropped tab, per-lesson forget and restore, an erase-everything button behind a
confirmation, and consolidate - ids included, so what you read there stays actionable
from a terminal. The GUI also carries web forms of `--estimate` (the **estimate** button under the
composer), `--patch-mode` (a setup toggle plus a **patch** download), `--native-tools` (a
setup toggle, which reports back when the server cannot honour it) and
`--output-format json` (the **export** dialog's result-JSON option).

Give the agent standing guidance (conventions, style, constraints) with a `.localcoder/system.md` file in the repo - it is injected into the system prompt under "## User Instructions" for every session in that project. The `--system TEXT` flag overrides the file for a single run. This is separate from `LOCALCODER.md`, the project-memory file, which holds facts **you** add with `/remember` and drop with `/forget`; the agent does not write it itself (its own close-time reflection is stored in the localm data dir, not in your repo).

Both files are injected into every system prompt, so both are capped at 3000 characters each to leave room for the repo map and the conversation. Going over is not silent: the agent prints which file was over budget and by how much, and the prompt itself carries a note saying the file was cut. Normal-sized files are injected verbatim.

---

## MCP server

```bash
localm mcp --print-config     # JSON block for Claude Desktop and friends
localm mcp                    # run as an MCP server over stdio (launched by the client)
localm mcp --no-images        # do not expose generate_image
localm mcp --no-coder         # do not expose run_coder_task
```

Exposes your local models and localm management (chat, model and plugin management, diagnostics, and more) to Claude Desktop and other MCP clients, with tool annotations so a client can confirm destructive calls. See [docs/mcp.md](../docs/mcp.md) for the full tool list and both directions: localm as an MCP server, and the coder consuming external MCP tool servers.

---

## Setup and diagnostics

```bash
localm setup-llama                       # auto-detect the GPU, fetch the right backend
localm setup-llama --backend vulkan      # any GPU (AMD/NVIDIA/Intel), no vendor toolkit
localm setup-llama --backend cuda        # NVIDIA  /  --backend amd-rocm (AMD)  /  --backend cpu
localm setup-llama --from <build-dir>    # or copy your own llama.cpp build
localm setup-embeddings                  # install the on-device embedding model (semantic memory + RAG)

localm make-launcher                     # build the native LocaLM app launcher (see docs/native-app.md)
localm make-launcher --force             # rebuild it (use after a Python upgrade)

localm doctor                            # check Python, llama.dll, GPU driver, VRAM, packages
localm info                              # data directory, config file, registry, registry file
localm memory list                       # what localm has remembered about you (see docs/memory.md)
localm status                            # show the localm server serving this directory, and what it is doing
localm ps                                # list running localm servers (per-directory instances)
localm cancel <id>                       # cancel one operation that server is running (id from `status`)
localm stop                              # stop the server serving this directory
localm stop <id>                         # stop one instance by id (or an id prefix, as shown by `ps`)
localm stop --all                        # stop every running localm instance
```

`localm run`/`localm gui`/`localm serve` start a background server that keeps running after the command exits; `localm stop` is how you end it - it asks the server to shut down cleanly (model unloaded, same as the GUI's Settings page), and force-ends the process if it does not confirm within `--timeout` seconds (default 10).

`localm status` lists what that server is working on right now - a model pull, a RAG re-embed, a media generation - with each operation's id. `localm cancel <id>` stops one of them (a unique id prefix is enough); a pull's download is ended, and an in-process job stops at its next checkpoint. Scheduled jobs are a separate thing under their own ids - use `localm job` for those.

`localm setup-embeddings` fetches a small on-device embedding model (default `bge-small-en-v1.5`) so semantic memory and RAG retrieval work without a lexical-only fallback; pass `--model` to choose a known key, a registered model, or a GGUF path. Switching to a different model than the one currently configured reports which existing Knowledge collections have embeddings and asks you to confirm before it happens (`-y`/`--yes` skips the confirmation) - see [docs/rag.md](../docs/rag.md#how-retrieval-works-and-why-its-lexical-first).

See [docs/gpu-setup.md](../docs/gpu-setup.md) for the full GPU setup guide.

---

## Updates and reporting

```bash
localm update                   # check for and apply a newer localm build (you always initiate it)
localm update --check           # only report whether an update is available; do not apply
localm update -y                # apply without the confirmation prompt
localm update --rollback        # restore the previous build from the last update backup
localm bug-report -m "..."      # generate an editable bug report and offer to send it to the maintainer
localm issues [NUMBER]          # list the project's issues, or show one by number
```

Updates are signed: localm verifies an Ed25519 signature against a key pinned in
its own source before applying anything, and refuses a build that is not newer.
See [SECURITY.md](../SECURITY.md) for the update trust model. The GUI Models page
has an "Update now" button for the same flow.

`localm bug-report` needs a working localm. When localm will not start at all (a
failed install, a broken venv, setup itself failing), use the standalone reporter
instead: double-click `report-issue.bat` (Windows) or run `bash report-issue.sh`
(Linux/macOS) from the clone. It shows you exactly what will be sent, files an
account-less GitHub issue via the same proxy (no GitHub login), and works with no
Python or a broken install. The `setup.bat` / `setup.sh` failure paths and the
launcher's "Report a problem" button both open it for you.

Updates are always user-initiated: localm never updates itself in the background.
