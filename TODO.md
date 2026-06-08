# localllm-coder — TODO / Feature Roadmap

Gaps identified by comparing against Claude Code, Aider, Cursor, Copilot.

---

## Context & Memory

- [x] Project map / codebase index — build a semantic map of the repo upfront; don't re-read files on demand every turn
- [ ] Persistent memory across sessions (`CLAUDE.md` equivalent) — user preferences, repo conventions, known gotchas
- [ ] Conversation compaction / summarisation — long sessions eat context; summarise old turns automatically
  <!-- REVIEW NOTE: Consider using codeneedle context-profiling methodology to trigger compaction dynamically when recall drops below a measured threshold. -->
- [ ] `.localcoder` project config file — per-repo defaults (model, cwd, max-turns, auto-approve rules)


---

## Tool Depth

- [ ] `patch_file` — unified diff application (safer than `edit_file` for multi-hunk changes)
- [ ] `undo_last_write` — single-step rollback; destructive writes currently have no undo
- [ ] `fetch_url` — let agent pull docs, Stack Overflow answers, READMEs
- [ ] `tree` tool — recursive directory tree with file sizes (richer than `list_dir`)
- [ ] Multi-file grep with context lines (currently grep is basic)
- [ ] Notebook support — read/edit `*.ipynb` files
- [ ] Git-aware first-class tools: `git_diff`, `git_log`, `git_status` (not just `run_shell` wrappers)

---

## Agent Quality

- [ ] Interruption / resume — Ctrl+C mid-task loses all progress; checkpoint and offer to resume
- [ ] Token budget tracking — agent has no visibility into context usage; warn + auto-compact near limit
- [ ] Retry / error recovery strategy — currently just feeds errors back; add smarter "try a different approach" logic
- [ ] Structured output enforcement — use grammar-constrained sampling or native function-calling APIs (when available) to guarantee valid tool calls instead of relying on instruction-following
- [ ] Tool call streaming — parse tool calls as tokens arrive instead of waiting for the full response

---

## Observability

- [ ] Cost / token tracking display — show `~2,400 tokens` in the footer each turn
- [ ] Turn replay / audit log — structured JSON log of every tool call + result per session
- [ ] `--dry-run` flag — show what the agent *would* do without executing anything

---

## UX

- [ ] `--interactive-confirm` granularity — approve writes but auto-approve reads, etc.
- [ ] Diff preview before write — show a coloured diff and prompt "apply?" instead of writing silently
- [ ] `/undo` REPL command — revert the last `write_file` / `edit_file`
- [ ] Multiline input in REPL — paste blocks of code as context
- [ ] `/compact` REPL command — manually trigger context summarisation mid-session
- [ ] Shell autocomplete for `--model` sourced from `localm list`

---

## Ecosystem

- [ ] MCP server support — let third-party tools register themselves (static `TOOL_REGISTRY` → dynamic)
- [ ] VS Code / Neovim extension — terminal integration so the agent sees the file you have open
- [ ] GitHub Actions / CI mode — `localcoder --ci "run tests and fix failures"` with structured exit codes
- [ ] `--patch-mode` — output a `.patch` file instead of modifying files directly

---

## Model Quality Workarounds

- [ ] Per-model-family system prompt variants — Gemma4, Llama3, Qwen, Mistral each respond better to different instruction styles
- [ ] Native function-calling API mode — when the server supports it (OpenAI, Anthropic), use it instead of text-format tool calls
- [ ] Thinking / scratchpad budget — pass `<think>` token budget for models that support chain-of-thought (Qwen3, DeepSeek-R1)

---

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

