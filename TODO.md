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
- [x] Structured output enforcement — OpenAI backend uses native tools API (schema-validated calls); local backends retain text parsing
- [x] Tool call streaming — tool_call XML blocks suppressed from stream display; full parse-on-arrival refactor deferred

---

## Observability

- [x] Cost / token tracking display — per-turn and running total shown in turn divider
- [x] Turn replay / audit log — `audit.py`; LOG mode = JSONL, FULL mode = JSONL + markdown transcript
- [x] `--dry-run` flag — destructive tools report skipped; read-only tools still run

---

## UX

- [x] `--interactive-confirm` granularity — `always_confirm` set gates specific tools (e.g. run_shell) even under --yes; configurable in .localcoder/config.toml
- [x] Diff preview before write — write_file/edit_file/patch_file all show coloured diff before confirming
- [x] `/undo` REPL command — revert the last `write_file` / `edit_file`
- [x] Multiline input in REPL — backslash continuation (end line with \\ to keep typing)
- [x] `/compact` REPL command — implemented in cli.py
- [x] Shell autocomplete for `--model` — Click shell_complete callback reads localm registry

---

## Ecosystem

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

