# localllm-coder — TODO / Feature Roadmap

Gaps identified by comparing against Claude Code, Aider, Cursor, Copilot.

---

## Context & Memory

- [x] Project map / codebase index — build a semantic map of the repo upfront; don't re-read files on demand every turn
- [x] Persistent memory across sessions (`CLAUDE.md` equivalent) — `LOCALCODER.md` per project via `memory.py`
- [x] Conversation compaction / summarisation — `_maybe_compact()` in agent.py; warns at 70%, auto-compacts at 90%; `/compact` REPL command
  <!-- REVIEW NOTE: Consider using codeneedle context-profiling methodology to trigger compaction dynamically when recall drops below a measured threshold. -->
- [ ] `.localcoder` project config file — per-repo defaults (model, cwd, max-turns, auto-approve rules)


---

## Tool Depth

- [x] `patch_file` — unified diff application via `_patch.py`
- [ ] `undo_last_write` — single-step rollback; destructive writes currently have no undo
- [x] `fetch_url` — fetches URL, strips HTML tags, truncates to context budget
- [ ] `tree` tool — recursive directory tree with file sizes (richer than `list_dir`)
- [x] Multi-file grep with context lines — `tool_grep` supports `context=N` and `glob=` filter
- [ ] Notebook support — read/edit `*.ipynb` files
- [x] Git-aware first-class tools: `git_status`, `git_diff`, `git_log` implemented in tools.py

---

## Agent Quality

- [ ] Interruption / resume — Ctrl+C mid-task loses all progress; checkpoint and offer to resume
- [x] Token budget tracking — `_fill_ratio()` + `_total_tokens` in agent.py; warns at 70%, compacts at 90%
- [ ] Retry / error recovery strategy — currently just feeds errors back; add smarter "try a different approach" logic
- [ ] Structured output enforcement — use grammar-constrained sampling or native function-calling APIs (when available) to guarantee valid tool calls instead of relying on instruction-following
- [ ] Tool call streaming — parse tool calls as tokens arrive instead of waiting for the full response

---

## Observability

- [ ] Cost / token tracking display — show `~2,400 tokens` in the footer each turn
- [x] Turn replay / audit log — `audit.py`; LOG mode = JSONL, FULL mode = JSONL + markdown transcript
- [ ] `--dry-run` flag — show what the agent *would* do without executing anything

---

## UX

- [ ] `--interactive-confirm` granularity — approve writes but auto-approve reads, etc.
- [ ] Diff preview before write — show a coloured diff and prompt "apply?" instead of writing silently
- [ ] `/undo` REPL command — revert the last `write_file` / `edit_file`
- [ ] Multiline input in REPL — paste blocks of code as context
- [x] `/compact` REPL command — implemented in cli.py
- [ ] Shell autocomplete for `--model` sourced from `localm list`

---

## Ecosystem

- [ ] MCP server support — let third-party tools register themselves (static `TOOL_REGISTRY` → dynamic)
- [ ] VS Code / Neovim extension — terminal integration so the agent sees the file you have open
- [ ] GitHub Actions / CI mode — `localcoder --ci "run tests and fix failures"` with structured exit codes
- [ ] `--patch-mode` — output a `.patch` file instead of modifying files directly

---

## Model Quality Workarounds

- [x] Per-model-family system prompt variants — `detect_model_family()` in prompts.py: gemma / thinking / small / default
- [ ] Native function-calling API mode — when the server supports it (OpenAI, Anthropic), use it instead of text-format tool calls
- [x] Thinking / scratchpad budget — thinking hints injected for deepseek-r1, qwq, qwen3 in prompts.py

---

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

