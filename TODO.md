# localllm-coder — TODO / Feature Roadmap

Gaps identified by comparing against Claude Code, Aider, Cursor, Copilot.

---

## Context & Memory

- [x] Project map / codebase index — build a semantic map of the repo upfront; don't re-read files on demand every turn
- [x] Persistent memory across sessions (`CLAUDE.md` equivalent) — `LOCALCODER.md` per project via `memory.py`
- [x] Conversation compaction / summarisation — `_maybe_compact()` in agent.py; warns at 70%, auto-compacts at 90%; `/compact` REPL command
  <!-- REVIEW NOTE: Consider using codeneedle context-profiling methodology to trigger compaction dynamically when recall drops below a measured threshold. -->
- [x] `.localcoder` project config file — per-repo defaults (model, cwd, max-turns, auto-approve rules)


---

## Tool Depth

- [x] `patch_file` — unified diff application via `_patch.py`
- [x] `undo_last_write` — `/undo` REPL command + `Agent.undo()` with file snapshot stack
- [x] `fetch_url` — fetches URL, strips HTML tags, truncates to context budget
- [x] `tree` tool — recursive directory tree with file sizes (richer than `list_dir`)
- [x] Multi-file grep with context lines — `tool_grep` supports `context=N` and `glob=` filter
- [x] Notebook support — read/edit `*.ipynb` files; `read_file` renders cells as text, `edit_notebook_cell` patches individual cells
- [x] Git-aware first-class tools: `git_status`, `git_diff`, `git_log` implemented in tools.py

---

## Agent Quality

- [x] Interruption / resume — Ctrl+C saves .localcoder/checkpoint.json; /resume restores state and continues
- [x] Token budget tracking — `_fill_ratio()` + `_total_tokens` in agent.py; warns at 70%, compacts at 90%
- [x] Retry / error recovery strategy — consecutive failure streak tracker; escalating hints injected at 2× and 3× failures
- [x] Structured output enforcement — online providers use native tools API; local backends retain text parsing
- [x] Grammar-constrained sampling for local models — GBNF grammar threaded through `llama_sampler_init_grammar` → `LlamaCpp` → `GgufBackend` → `Engine` → HTTP server; pre-built grammars in `localm/inference/gbnf.py`; HF backend accepts and ignores the param
- [x] Parallel tool execution — non-destructive tool calls in a turn run concurrently via `ThreadPoolExecutor`; destructive calls always serialised
- [x] Structured JSON compaction — `_compact_history()` uses GBNF JSON grammar when backend supports it; produces `{summary, changed_files, open_tasks}`
- [x] `scope` parameter — `--scope GLOB` CLI flag; file-access tools reject paths outside the active glob; `/scope` REPL command to inspect/change at runtime
- [x] Tool call streaming — tool_call XML blocks suppressed from stream display; full parse-on-arrival refactor deferred

---

## Observability

- [x] Cost / token tracking display — per-turn and running total shown in turn divider
- [x] Turn replay / audit log — `audit.py`; LOG mode = JSONL, FULL mode = JSONL + markdown transcript
- [x] `--dry-run` flag — destructive tools report skipped; read-only tools still run
- [x] Live tok/s display — printed after each response in `localm run` and interactive mode

---

## UX

- [x] `--interactive-confirm` granularity — `always_confirm` set gates specific tools (e.g. run_shell) even under --yes; configurable in .localcoder/config.toml
- [x] Diff preview before write — write_file/edit_file/patch_file all show coloured diff before confirming
- [x] `/undo` REPL command — revert the last `write_file` / `edit_file`
- [x] Multiline input in REPL — backslash continuation (end line with \\ to keep typing)
- [x] `/compact` REPL command — implemented in cli.py
- [x] `/export [path]` REPL command — write session markdown on demand
- [x] Shell autocomplete for `--model` — Click shell_complete callback reads localm registry

---

## Ecosystem

- [x] Exact token counts in HTTP API — `count_tokens()` on each backend (GGUF: native tokenizer, HF: transformers tokenizer); replaces chars÷4 heuristic
- [x] Bearer token auth in HTTP server — `LOCALM_API_KEY` env var; open mode when unset; protected endpoints return 401 on mismatch
- [x] Request queueing in HTTP server — `asyncio.Semaphore(1)` serialises concurrent inference requests
- [x] `localm doctor` — checks Python version, llama.dll, GPU driver, VRAM, and required packages
- [x] VRAM display at model load — shown for both GGUF and HF backends when torch+CUDA available
- [x] Disk space preflight in model downloader — HEAD request to get Content-Length, `shutil.disk_usage` check before any download starts
- [x] Resumable downloads — `.part` file + `Range: bytes=N-` header; atomically renamed on completion
- [x] New coder tools: `run_tests`, `git_commit`, `git_push`, `git_create_branch`, `search_replace` — all registered in TOOL_REGISTRY
- [x] Path confinement for file tools — `_confine()` helper raises PermissionError on path traversal
- [x] Syntax verification after writes — `_verify_syntax()` auto-runs after write/edit/patch; warns agent on failure
- [ ] MCP server support — let third-party tools register themselves (static `TOOL_REGISTRY` → dynamic)
- [ ] VS Code / Neovim extension — terminal integration so the agent sees the file you have open
- [x] GitHub Actions / CI mode — `--ci` flag: auto-approve, plain-text output, exit 0/1/2; `--output-format json` for machine-readable results
- [x] `--patch-mode FILE` — captures write/edit/patch calls as unified diffs; writes to FILE or stdout ('-')

---

## Model Quality Workarounds

- [x] Per-model-family system prompt variants — `detect_model_family()` in prompts.py: gemma / thinking / small / default
- [x] Native function-calling API mode — enabled for OpenAI and Anthropic automatically; --native-tools flag for --url servers (Ollama etc.)
- [x] Thinking / scratchpad budget — thinking hints injected for deepseek-r1, qwq, qwen3 in prompts.py

---

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

