# localm: TODO / Feature Roadmap

Coder gaps identified by comparing against Claude Code, Aider, Cursor, Copilot.
GUI gaps identified by comparing against LM Studio, Jan, Open WebUI.

---

## Context & Memory

- [x] Project map / codebase index: build a semantic map of the repo upfront; don't re-read files on demand every turn
- [x] Persistent memory across sessions (`CLAUDE.md` equivalent): `LOCALCODER.md` per project via `memory.py`
- [x] Conversation compaction / summarisation: `_maybe_compact()` in agent.py; warns at 70%, auto-compacts at 90%; `/compact` REPL command
  <!-- REVIEW NOTE: Consider using codeneedle context-profiling methodology to trigger compaction dynamically when recall drops below a measured threshold. -->
- [x] `.localcoder` project config file: per-repo defaults (model, cwd, max-turns, auto-approve rules)


---

## Tool Depth

- [x] `patch_file`: unified diff application via `_patch.py`
- [x] `undo_last_write`: `/undo` REPL command + `Agent.undo()` with file snapshot stack
- [x] `fetch_url`: fetches URL, strips HTML tags, truncates to context budget
- [x] `tree` tool: recursive directory tree with file sizes (richer than `list_dir`)
- [x] Multi-file grep with context lines: `tool_grep` supports `context=N` and `glob=` filter
- [x] Notebook support: read/edit `*.ipynb` files; `read_file` renders cells as text, `edit_notebook_cell` patches individual cells
- [x] Git-aware first-class tools: `git_status`, `git_diff`, `git_log` implemented in tools.py

---

## Agent Quality

- [x] Interruption / resume: Ctrl+C saves .localcoder/checkpoint.json; /resume restores state and continues
- [x] Token budget tracking: `_fill_ratio()` + `_total_tokens` in agent.py; warns at 70%, compacts at 90%
- [x] Retry / error recovery strategy: consecutive failure streak tracker; escalating hints injected at 2× and 3× failures
- [x] Structured output enforcement: online providers use native tools API; local backends retain text parsing
- [x] Grammar-constrained sampling for local models: GBNF grammar threaded through `llama_sampler_init_grammar` → `LlamaCpp` → `GgufBackend` → `Engine` → HTTP server; pre-built grammars in `localm/inference/gbnf.py`; HF backend accepts and ignores the param
- [x] Parallel tool execution: non-destructive tool calls in a turn run concurrently via `ThreadPoolExecutor`; destructive calls always serialised
- [x] Structured JSON compaction: `_compact_history()` uses GBNF JSON grammar when backend supports it; produces `{summary, changed_files, open_tasks}`
- [x] `scope` parameter: `--scope GLOB` CLI flag; file-access tools reject paths outside the active glob; `/scope` REPL command to inspect/change at runtime
- [x] Tool call streaming: tool_call XML blocks suppressed from stream display; full parse-on-arrival refactor deferred

---

## Observability

- [x] Cost / token tracking display: per-turn and running total shown in turn divider
- [x] Turn replay / audit log: `audit.py`; LOG mode = JSONL, FULL mode = JSONL + markdown transcript
- [x] `--dry-run` flag: destructive tools report skipped; read-only tools still run
- [x] Live tok/s display: printed after each response in `localm run` and interactive mode

---

## UX

- [x] `--interactive-confirm` granularity: `always_confirm` set gates specific tools (e.g. run_shell) even under --yes; configurable in .localcoder/config.toml
- [x] Diff preview before write: write_file/edit_file/patch_file all show coloured diff before confirming
- [x] `/undo` REPL command: revert the last `write_file` / `edit_file`
- [x] Multiline input in REPL: backslash continuation (end line with \\ to keep typing)
- [x] `/compact` REPL command: implemented in cli.py
- [x] `/export [path]` REPL command: write session markdown on demand
- [x] Shell autocomplete for `--model`: Click shell_complete callback reads localm registry

---

## Ecosystem

- [x] Exact token counts in HTTP API: `count_tokens()` on each backend (GGUF: native tokenizer, HF: transformers tokenizer); replaces chars÷4 heuristic
- [x] Bearer token auth in HTTP server: `LOCALM_API_KEY` env var; open mode when unset; protected endpoints return 401 on mismatch
- [x] Request queueing in HTTP server: `asyncio.Semaphore(1)` serialises concurrent inference requests
- [x] `localm doctor`: checks Python version, llama.dll, GPU driver, VRAM, and required packages
- [x] VRAM display at model load: shown for both GGUF and HF backends when torch+CUDA available
- [x] Disk space preflight in model downloader: HEAD request to get Content-Length, `shutil.disk_usage` check before any download starts
- [x] Resumable downloads: `.part` file + `Range: bytes=N-` header; atomically renamed on completion
- [x] New coder tools: `run_tests`, `git_commit`, `git_push`, `git_create_branch`, `search_replace`: all registered in TOOL_REGISTRY
- [x] Path confinement for file tools: `_confine()` helper raises PermissionError on path traversal
- [x] Syntax verification after writes: `_verify_syntax()` auto-runs after write/edit/patch; warns agent on failure
- [x] MCP client support: external MCP servers from `.localcoder/config.toml` register tools dynamically as `mcp_<server>_<tool>`; untrusted servers gated as destructive
- [x] MCP server: `localm mcp` exposes chat/list_models/embed/generate_image to any MCP client (Claude Desktop etc.); `--print-config` emits the client JSON
- [ ] VS Code / Neovim extension: terminal integration so the agent sees the file you have open
- [x] GitHub Actions / CI mode: `--ci` flag: auto-approve, plain-text output, exit 0/1/2; `--output-format json` for machine-readable results
- [x] `--patch-mode FILE`: captures write/edit/patch calls as unified diffs; writes to FILE or stdout ('-')

---

## Model Quality Workarounds

- [x] Per-model-family system prompt variants: `detect_model_family()` in prompts.py: gemma / thinking / small / default
- [x] Native function-calling API mode: enabled for OpenAI and Anthropic automatically; --native-tools flag for --url servers (Ollama etc.)
- [x] Thinking / scratchpad budget: thinking hints injected for deepseek-r1, qwq, qwen3 in prompts.py

---

## Web GUI (`localm gui`)

### Done

- [x] Chat: streaming, markdown + highlighted code, copy buttons (message + code block)
- [x] Chat parameters: temperature, top-p, max tokens, seed, system prompt
- [x] Conversation history in localStorage: list, switch, delete, auto-title
- [x] Stop generation mid-stream
- [x] Usage stats per reply: total tokens, TTFT, tok/s
- [x] Model selector with live engine switching (waits for in-flight inference)
- [x] Coder sessions: cwd input, persistence mode, auto-approve toggle
- [x] Coder feed: streaming reasoning, expandable tool-call cards with args/output
- [x] Browser approval flow for destructive tools with unified diff preview (10 min timeout)
- [x] Coder stop / end session; busy-state handling; turn + token counter
- [x] Bearer auth (`LOCALM_API_KEY`) honoured on all /api routes; key prompted once and kept in localStorage
- [x] Port auto-pick in the localm range; `--no-browser` flag

### Chat (round 2, shipped)

- [x] Image attachment (multimodal): CLI has `--image`, the GUI composer has no attach button
- [x] top-k and repeat-penalty in the parameters drawer (server already accepts them)
- [x] GBNF grammar field (server already accepts `grammar`)
- [x] Regenerate last reply / edit a sent message
- [x] Rename and export conversations (markdown/JSON)

### Coder (round 2, shipped)

- [x] Session reattach after page reload: needs `GET /api/coder/sessions` (list) so the UI can reconnect to a live session instead of orphaning it
- [x] Rendered diff in tool cards when auto-approve is on (currently raw args JSON; diffs only show in approval cards)
- [x] Expose max-turns and temperature in the session setup form (API already accepts both)
- [x] Per-session model choice (sessions always use the active chat model)
- [x] Undo / compact / scope: REPL commands with no GUI equivalent
- [x] Session audit-log viewer (JSONL from log/full modes)
- [x] Multiple concurrent sessions in the UI (backend already supports it)

### Pages (round 2, shipped)

- [x] Model management page: pull with progress, remove, aliases (registry list exists; mutations are CLI-only)
- [x] Plugin manager page (needs `/v1/plugins` endpoints)
- [x] Settings page: server config editing (needs `/v1/config` GET/PATCH)
- [x] Image generation panel (ComfyUI at 8188)
- [x] Image management: delete / move to folder / use as img2img input / send to chat
- [x] Light theme toggle (dark only today)
- [ ] Tauri 2 native shell wrapping this frontend (window, sidecar lifecycle, tray)

### Music generation (to be implemented)

Backend scaffold is in place; the user-facing parts are still to do.

- [x] `localm/music_gen/` — ACE-Step workflow (`ace_workflow.json`) + `generate_music()` via ComfyUI (arbitrary track length in seconds, lyrics or instrumental, FLAC output, sidecar metadata, VRAM handoff)
- [x] API endpoints: `POST /api/music`, `GET /api/music/history`, `GET/DELETE /api/music/file/{name}`, `POST /api/music/file/{name}/move`
- [ ] GUI "Music" page: tags/lyrics/duration form, job log, inline audio player, history with manage actions (mirror the Images page)
- [ ] `/music` slash command in chat (mirror `/imagine`)
- [ ] CLI command (`localm music "tags" --lyrics file --duration 180`)
- [ ] Verify end-to-end against a ComfyUI install with `ace_step_v1_3.5b.safetensors`; document model download in README

---

## Server / platform

- [x] `GET /v1/models/{id}`: model detail endpoint with registry metadata
- [x] `localm benchmark <model>`: standard prompt, TTFT, tok/s at multiple context lengths
- [x] 429 retry/backoff for cloud coder backends (OpenAI/Anthropic opt-ins)
- [x] Tool result compression: summarise large tool outputs when context fill > 50%
- [x] `read_env` coder tool: reads `.env` and active env vars with secrets stripped
- [x] `--estimate` flag: one planning turn without execution, prints expected token usage
- [x] PyPI packaging polish: classifiers, `localm[gpu]`/`localm[cpu]` extras, publish workflow
- [x] TLS / reverse-proxy guide for LAN serving

---

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

