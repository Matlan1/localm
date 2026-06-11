# localm Feature Report (June 2026, historical)

Point-in-time gap audit that drove the foundation-hardening work. Everything
listed as missing here has since been implemented or explicitly deferred;
see TODO.md for current status. Kept for the reasoning and comparisons.

---

# localm — Feature Report

Comprehensive audit of the project against professional-grade expectations for
a local LLM runner and AI coding agent. Each section describes what is
implemented, what comparable tools provide, and what the requirements are to
close the gap.

Reference tools used for comparison:
- **LLM runner**: LM Studio, Ollama, Jan
- **Coding agent**: Claude Code, Aider, Cursor, Continue.dev
- **Image generation**: ComfyUI (standalone), Automatic1111

---

## 1. Inference Engine

### Current state

- Custom ctypes bindings to `llama.dll` — no dependency on llama-cpp-python
- Full sampler chain: top-k, top-p, min-p, temperature, dist
- GBNF grammar sampling via `llama_sampler_init_grammar` (offline, at the sampler layer)
- HuggingFace Transformers backend for full-precision HF model directories
- Multimodal support in the HF backend (image, audio via Gemma4 architecture)
- Chat template auto-detection from the model's embedded Jinja template; falls back to ChatML
- Context re-creation per call (KV cache not persisted between requests)
- Streaming stop-string filter handles split-token boundary cases
- `llama-cli.exe` subprocess fallback if the DLL cannot be loaded

### Professional expectation

LM Studio and Ollama both maintain a persistent KV cache across requests from
the same session. Users expect:

- In-session KV cache reuse — the prompt prefix does not need re-evaluation on
  every follow-up message
- Speculative decoding with a draft model for faster throughput on large models
- Flash attention and grouped query attention exposed as options
- Quantization options beyond what ships in the downloaded file (GGUF
  requantization, or at minimum clear labelling of the quant level in use)
- Context window extension methods (RoPE scaling, YaRN) configurable without
  reloading the model
- Performance metrics available to the caller: time to first token, tokens per
  second, prompt eval time

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Persistent KV cache across calls within a session — add `llama_kv_self_clear` or session-keyed context reuse so prefills are not repeated |
| High | Expose inference timing: TTFT, token/s, prompt eval time in the `usage` response field or a separate header |
| Medium | GPU memory reporting — surface VRAM in use and free via `llama_get_state_size` or ROCm SMI so users can tune `n_gpu_layers` |
| Medium | RoPE / YaRN context extension — `rope_scaling_type` and `rope_freq_scale` in context params |
| Medium | Per-request seed control — expose `seed` on `ChatRequest` so generation is reproducible |
| Low | Speculative decoding — draft model loaded alongside main model; `n_draft` tokens sampled per step |
| Low | GGUF metadata inspection — surface architecture, parameter count, quant level, context limit from the file header before loading |

---

## 2. Model Management

### Current state

- `localm pull <name>` downloads GGUF files from HuggingFace Hub with a
  progress bar
- Shortcut table maps human names to `bartowski` quantized builds
- `localm add <path>` registers local files or directories
- Ollama manifest resolution — recognises the Ollama blob layout and points at
  the underlying GGUF file
- `localm models` lists registered models with name, type, size, source, path
- `localm rm <name>` removes from registry; deletes the file only if it lives
  inside the managed models directory

### Professional expectation

LM Studio has a full model browser with search, filter by size and
architecture, one-click download, and automatic stale-model detection. Ollama
has `ollama update` and a curated model library with tags. Professionals expect:

- Ability to pull split GGUF files (multi-part, e.g. `Q8_0-00001-of-00003.gguf`)
- Metadata shown before committing to a download: parameter count, architecture,
  quant level, license, context limit
- Disk space check before starting a download
- Resumable downloads — network interruptions should not force a restart
- Model update detection — flag when a newer quant or version is available

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Resumable downloads — use `Range` headers and a `.part` file; verify SHA256 on completion |
| High | Disk space pre-flight — check `shutil.disk_usage` before downloading; warn or abort if insufficient |
| High | Split GGUF support — detect `*-of-N.gguf` naming, download all parts, pass the first part path to llama.cpp |
| Medium | GGUF metadata display before load — read the file header to show arch, quant, context, parameter count |
| Medium | `localm update <name>` — compare local file SHA or timestamp against the HF repo and prompt to re-download |
| Low | Model tagging — allow users to add free-form tags to registry entries for personal organisation |

---

## 3. HTTP Inference API

### Current state

- FastAPI server at `/v1/chat/completions` (streaming and non-streaming)
- `/v1/models` list endpoint
- `/v1/models/load` and `/v1/models/unload` lifecycle endpoints
- `/health` status endpoint
- Full CORS headers (open origin)
- OpenAI-compatible request schema (`ChatRequest`) with `grammar` field
- Token usage estimation (character-count heuristic, not exact)
- SSE streaming with role announcement chunk and final usage chunk

### Professional expectation

The OpenAI API spec is the de facto standard. Tools like Continue.dev, the
OpenAI Python SDK, and LangChain all consume it directly. Professionals building
on top of a local server expect:

- Exact token counts (not a character-length estimate)
- `/v1/completions` non-chat endpoint for raw prompt completion
- `/v1/embeddings` endpoint for semantic search and RAG pipelines
- Bearer token authentication — even a simple static key stops accidental
  exposure when the port is bound to 0.0.0.0
- Concurrent request handling — multiple clients should queue rather than
  receive errors
- Standard HTTP 429 rate limiting so well-behaved clients back off cleanly

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Exact prompt token counts — tokenize the input messages and return the real count in `usage.prompt_tokens` |
| High | Bearer token auth — `LOCALM_API_KEY` environment variable; requests without a matching `Authorization` header get 401 |
| High | Request queueing — serialise inference calls with `asyncio.Queue` so a second request waits rather than colliding |
| Medium | `/v1/completions` — raw completion endpoint (no chat template applied) for tools that use it |
| Medium | `/v1/embeddings` — embedding endpoint backed by a dedicated embedding model or the main model's last hidden state |
| Medium | `/v1/models/{id}` GET endpoint — model detail card with metadata |
| Low | Rate limiting via `slowapi` or a manual token bucket — configurable requests/minute |
| Low | `X-Request-ID` response header for tracing requests through logs |

---

## 4. Coding Agent (localcoder)

### 4a. Context and Memory

#### Current state

- `LOCALCODER.md` per project — persistent memory injected at session start
- Codebase index built at startup: file tree, per-file symbol extraction for 15
  languages, kept under ~3 000 characters
- Incremental index refresh when the agent writes or edits a file
- Conversation compaction at 70% / 90% context fill — produces a ~300-word
  prose summary
- `.localcoder/config.toml` per project — model, cwd, max turns, always_confirm
- Checkpoint save on Ctrl+C, `/resume` to restore

#### Professional expectation

Claude Code and Cursor build a semantic index, not just a symbol list. They can
answer "where is X used" across the whole codebase without the agent having to
grep for it. Professionals expect:

- Semantic / vector search over the codebase so the agent retrieves relevant
  context without exhausting the context window on grep-and-read cycles
- Compaction that produces a structured summary (decisions, open tasks, file
  changes), not free prose, so resumption is more reliable
- Cross-session memory of decisions made, not just of the file state

#### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Structured compaction format — produce JSON `{summary, changed_files, open_tasks}` using grammar sampling; inject it back as a compact system block rather than a conversation turn |
| Medium | Embedding-based code search — embed function docstrings and file summaries at index time; retrieve top-K by cosine similarity at query time; requires `/v1/embeddings` |
| Medium | Cross-session decision log — append significant decisions to `LOCALCODER.md` automatically rather than relying on the user to update it |
| Low | Incremental re-indexing on git checkout / branch switch — detect `HEAD` change and refresh the index |

---

### 4b. Tool Set

#### Current state

16 tools: `read_file`, `write_file`, `edit_file`, `patch_file`, `list_dir`,
`tree`, `grep`, `run_shell`, `fetch_url`, `git_status`, `git_diff`, `git_log`,
`edit_notebook_cell`, `undo_last_write`, `generate_image`

#### Professional expectation

Claude Code can run tests and interpret results, commit and push changes, and
interact with the terminal. Aider has direct git commit integration.
Professionals expect the agent to close the loop on tasks without manual steps:

- Test runner integration — run tests and feed the output back as context so
  the agent can fix failures without the user copying output
- Git commit and push tools — not just status/diff/log; the agent should be
  able to commit staged changes with a generated message
- LSP integration — go-to-definition, find-references, type errors as a tool
  so the agent gets compiler feedback rather than guessing
- Multi-file search-and-replace atomically

#### Requirements

| Priority | Requirement |
|----------|-------------|
| High | `run_tests` tool — wraps `pytest`, `cargo test`, `npm test`, etc.; returns pass/fail counts and failure output; non-destructive |
| High | `git_commit` tool — stages specified files and commits with a model-generated message; respects pre-commit hooks |
| Medium | `search_replace` tool — multi-file regex search-and-replace with preview; atomic across all matched files |
| Medium | `git_push` / `git_create_branch` tools — close the loop on feature branch workflows |
| Medium | `read_env` tool — reads `.env` and active environment variables; strips secrets before injecting into context |
| Low | `lsp_symbols` / `lsp_references` tool — queries a running LSP server for definitions and references |
| Low | `screenshot` tool — captures a region or window; useful for UI debugging and the image generation pipeline |

---

### 4c. Agent Loop Quality

#### Current state

- Consecutive failure streak tracker with escalating hints at 2× and 3×
- Token budget tracking with warnings at 70% and auto-compact at 90%
- Tool call XML suppressed from stream output
- Grammar-constrained sampling infrastructure in place for structured output
- Native tools API for OpenAI/Anthropic; text parsing for local backends
- Per-model-family system prompt variants (gemma, thinking, small, default)
- Thinking budget injected for r1-family and qwq models

#### Professional expectation

Claude Code's agent loop is reliable on complex tasks partly because it
verifies its own work before reporting success. Aider runs tests after applying
changes and feeds failures back into the loop. Professionals notice:

- The agent declaring success without verifying the change compiles or tests pass
- No mechanism to express uncertainty and interrupt rather than guess
- No parallel tool execution even when two reads are completely independent

#### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Self-verification step — after writing a file, the agent should automatically check syntax (`py_compile`, `tsc --noEmit`, `cargo check`) without the user asking |
| High | Uncertainty escalation — if a subtask exceeds a turn budget, interrupt and ask the user rather than making a guess |
| Medium | Parallel tool calls — when the response contains multiple independent tool calls, execute them concurrently with `asyncio.gather` |
| Medium | Tool result compression — large tool outputs should be summarised before injection if they would push the fill ratio above 50% |
| Low | Confidence scoring — if the model self-corrects more than a threshold per session, warn that a larger model may be needed |

---

### 4d. Developer Experience

#### Current state

- Coloured diff preview before any destructive file operation
- `/undo` reverts last write/edit/patch from a snapshot stack
- `--dry-run` runs read-only tools, skips destructive ones with a report
- `--interactive-confirm` gates specific tools even under `--yes`
- `--patch-mode FILE` captures all writes as unified diffs without touching disk
- `--ci` flag: auto-approve, plain output, exit 0/1/2, JSON output format
- `--native-tools` for OpenAI-compatible servers that support the tools API
- Shell autocomplete for `--model`
- Multiline input via backslash continuation
- Privacy mode: readline history suppressed, subprocess env sanitized, shell
  history scrubbed on exit

#### Professional expectation

Cursor and Copilot live inside the editor — the agent sees the file you are
looking at and can propose inline changes without you describing which file to
touch. Professionals also expect:

- A way to constrain the agent to a subset of the repo without writing a custom
  system prompt
- Conversation export in a portable format for sharing or archiving
- A "what would this cost" dry run before committing a long agent run
- Automatic back-off when online providers return rate limit errors

#### Requirements

| Priority | Requirement |
|----------|-------------|
| High | `--scope GLOB` flag — agent may only read/write paths matching the glob; any tool call outside it is blocked and reported |
| High | VS Code extension — opens a side panel backed by a localcoder session; sends the active file and selection as initial context |
| Medium | Conversation export — `/export` REPL command writes the session as a clean markdown transcript |
| Medium | Cost estimation — `--estimate` flag runs one planning turn (no tool execution) and prints expected token usage before the real run |
| Medium | Retry budget per online provider — catch HTTP 429 from OpenAI/Anthropic and back off with exponential jitter |
| Low | Neovim plugin — `localcoder.nvim` sends buffer path and visual selection to the agent; shows diffs in the quickfix list |

---

### 4e. Integrations

#### Current state

- GitHub Actions CI mode via `--ci` and `--output-format json`
- `--patch-mode` produces diffs consumable by CI pipelines
- `fetch_url` for lightweight documentation retrieval

#### Professional expectation

Real development workflows involve code review tools, issue trackers, and CI
systems. Professionals expect the agent to open a PR when it finishes a task
and to pull failing CI output without copy-paste.

#### Requirements

| Priority | Requirement |
|----------|-------------|
| Medium | MCP server support — dynamic tool registry so third-party tools (GitHub, Linear, Jira) register themselves at startup |
| Medium | GitHub tool set — `create_pr`, `list_issues`, `comment_on_pr` via the GitHub CLI or API |
| Low | CI log fetcher — given a GitHub Actions run URL, fetch and summarise the failure output |

---

## 5. Image Generation

### Current state

- Full ComfyUI FLUX pipeline via the ComfyUI API at port 8188
- `generate_image()` with: prompt, guidance, negative prompt
  (ConditioningConcat — no CFG mode change required), seed, dual CLIP encoder
  override, LoRA loading, img2img with denoise, output path
- GGUF T5 auto-detection routes to `DualCLIPLoaderGGUF` automatically
- `localm_url` parameter triggers model unload before generation and reload
  after, so the LLM and image pipeline share VRAM without manual steps
- Encoder comparison test script with fixed seed per run, named outputs

### Professional expectation

A professional image generation setup expects multiple backends, a generation
history, batch sweeps, and post-processing steps. ComfyUI (standalone) provides
all of these through its node editor but without a programmatic Python API for
all workflows.

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Generation history — save metadata (prompt, seed, guidance, encoder, model) alongside every output image in a sidecar JSON file |
| Medium | Abstract `BaseImageBackend` — common interface with ComfyUI and A1111 implementations behind the same `generate_image()` call |
| Medium | Batch generation — accept a list of seeds or prompt variants; run sequentially; return a list of output paths |
| Medium | Inpainting support — pass a mask image to a ComfyUI inpaint workflow variant |
| Low | ControlNet conditioning — add ControlNet node injection to the workflow builder for pose, depth, and canny |
| Low | Post-process pipeline — optional ESRGAN upscale step after generation |

---

## 6. Privacy and Security

### Current state

- `--mode privacy`: readline history suppressed, subprocess env vars sanitized
  (HISTFILE, HISTSIZE, LESSHISTFILE, MYSQL_HISTFILE, SQLITE_HISTORY), shell
  history files scrubbed of localcoder references on exit
- External provider warning when privacy mode is combined with an online flag
- Online providers are always explicit opt-in (never the default path)
- No network call is made without an explicit flag or tool invocation
- Audit log is not written in privacy mode

### Professional expectation

Security-conscious teams deploying locally need sandboxed shell execution, path
confinement, and clear documentation of what leaves the machine and when.

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Path confinement — all file tools validate that the resolved absolute path is under `cwd`; return a clear error if not |
| High | Shell tool scope documentation — clarify in the README and help text that `--yes` without `always_confirm = ["run_shell"]` bypasses shell confirmation |
| Medium | Argument list subprocess mode — when the shell command contains no pipes or redirects, pass it as an argument list rather than a shell string to avoid injection |
| Medium | Network audit log — in privacy mode, print to stderr when `fetch_url` is called so the user is aware of outbound requests |
| Low | `--no-network` flag — blocks `fetch_url` entirely; useful for air-gapped environments |

---

## 7. Observability and Performance

### Current state

- Per-turn and running total token counts displayed in the turn divider
- Audit log in JSONL + optional markdown transcript
- `--dry-run` shows what would run without executing
- Token fill ratio drives compaction warnings

### Professional expectation

A serious local setup needs to know how much GPU memory a model is using, how
fast inference is running, and whether the current quantization is worth the
quality tradeoff.

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Live tokens/sec display — compute from wall-clock time between stream chunks; show next to the turn divider |
| High | VRAM display at load — query ROCm SMI or nvidia-smi after model load; report MB allocated |
| Medium | `localm benchmark <model>` command — runs a standard prompt, measures TTFT and generation speed at multiple context lengths, writes a report |
| Medium | Audit log performance fields — add `ttft_ms`, `tokens_per_sec`, `prompt_tokens_exact` to the JSONL entries |
| Low | Context recall profiling — run verbatim function recall tests against the current workspace to determine the maximum reliable context depth for a given model on the current hardware |

---

## 8. Plugin and Extension Architecture

### Current state

- Plugins live in `localm/plugins/` — `coder` is the only plugin
- `TOOL_REGISTRY` is a static dict built at import time
- No discovery mechanism for external plugins

### Professional expectation

Continue.dev has a plugin SDK with a documented API, versioning, and a
community registry. Professionals building on top of the platform expect a
stable API, one-command install, and sandboxing.

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | Plugin discovery — scan `~/.localm/plugins/` at startup; any directory with a `plugin.toml` manifest is loaded |
| High | Plugin manifest format — `plugin.toml` declares name, version, entry point, tool exports, required localm version |
| Medium | `localm plugin install <path-or-url>` — clones or copies a plugin, validates the manifest, makes tools available |
| Medium | `localm plugin list` / `localm plugin remove` — lifecycle management |
| Medium | Plugin API versioning — `requires_localm` field so incompatible plugins are rejected cleanly |
| Low | Plugin sandboxing — run plugin tool functions in a subprocess with a restricted import allowlist |

---

## 9. Distribution and Installation

### Current state

- Python package installable via `uv pip install -e .`
- DLL discovery via `_loader.py` — searches StableMatrix, Ollama, and common
  install paths for `llama.dll` / `libllama.so`
- No bundled binary — users need a working Python environment and the DLL
  separately

### Professional expectation

LM Studio and Jan ship as double-clickable installers. For developer tools, one-line
install via pip or winget is the minimum bar. GPU setup should not require the
user to find the right DLL manually.

### Requirements

| Priority | Requirement |
|----------|-------------|
| High | PyPI publish — package to PyPI so `pip install localm` works; include `localm[gpu]` and `localm[cpu]` extras |
| High | GPU auto-detection — at install time or first run, detect ROCm/CUDA availability and download the matching DLL if absent |
| Medium | Windows installer — produces a signed `.exe` installer including Python, the DLL, and the GUI |
| Medium | `localm doctor` command — checks DLL presence, Python version, GPU driver, available VRAM, prints a health report with fix suggestions |
| Low | Winget / Homebrew manifest — register `localm` in winget-pkgs and Homebrew |

---

## 10. GUI (Planned)

### Current state

Branch `feat/gui` created. Full implementation plan at `gui/PLAN.md`. Stack:
Tauri 2 + Svelte 5 + Tailwind CSS + shadcn-svelte. Five phases: scaffold, chat,
coder interface, plugin/skills manager, polish.

### Requirements not yet captured in `gui/PLAN.md`

| Priority | Requirement |
|----------|-------------|
| High | First-run setup wizard — detects GPU, offers to download a starter model, configures the server port |
| High | Live tokens/sec indicator during generation |
| Medium | Model performance card — shows benchmark results alongside each model in the model list |
| Medium | Diff viewer for coder sessions — side-by-side display of every file the agent touched |
| Medium | Prompt library — save and reuse system prompts and common task templates |
| Low | Image generation gallery — grid view of past generations with metadata on hover |
| Low | Log viewer tab — tail the audit log and server log in real time |

---

## Summary Table

| Section | Maturity | Biggest Gap |
|---------|----------|-------------|
| Inference engine | Good | No KV cache persistence; no inference timing metrics |
| Model management | Good | No resumable downloads; no split GGUF; no disk space check |
| HTTP API | Solid | Token counts are estimates; no auth; no embeddings endpoint |
| Coding agent — context | Good | No semantic search; compaction produces free prose |
| Coding agent — tools | Good | No test runner tool; no git commit tool |
| Coding agent — loop quality | Good | No self-verification; no parallel tool execution |
| Coding agent — UX | Strong | No `--scope` constraint; no cost estimation |
| Coding agent — integrations | Partial | MCP not implemented; no GitHub tools |
| Image generation | Good | No generation history; single backend (ComfyUI only) |
| Privacy / security | Strong | Path confinement not enforced; shell uses string mode |
| Observability | Partial | No live tok/s; no VRAM display; no benchmark command |
| Plugin architecture | Early | No discovery; no install tooling; static registry only |
| Distribution | Early | No PyPI publish; no installer; no GPU auto-detection |
| GUI | Planned | Not started |
