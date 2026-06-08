# localm — Implementation Plan
*Updated post-merge: localm + localllm-coder unified project*

---

## Architecture context

The project is now a single repository:

```
localm/
├── localm/                   ← inference engine (core)
│   ├── inference/            ← GGUF, HF, llama.cpp ctypes backends
│   ├── model_manager.py      ← pull / list / add / rm
│   ├── cli.py                ← localm run / serve / pull / list / coder
│   └── plugins/
│       └── coder/            ← AI coding agent  (pip install "localm[coder]")
│           ├── agent.py      ← agentic loop
│           ├── tools.py      ← read/write/edit/shell/grep/spawn/generate_image
│           ├── indexer.py    ← ProjectMap codebase index
│           ├── parser.py     ← tool-call parser (XML + Gemma4 native)
│           ├── prompts.py    ← system prompt builder
│           ├── server.py     ← ManagedServer (auto-starts localm serve)
│           ├── display.py    ← Rich terminal UI
│           └── backends/     ← HTTPBackend (OpenAI-compat)
└── tests/plugins/coder/
```

Install:
- `pip install "localm"` — inference engine only (`localm run`, `localm serve`, ...)
- `pip install "localm[coder]"` — full platform (`localm coder`, `localcoder`)

---

## File review and assessment

### TODO.md — evaluated

Gaps vs. production tools (Claude Code, Aider, Cursor, Copilot).  Status after merge:

| Item | Status | Priority |
|---|---|---|
| Project map / codebase index | **Done** | — |
| Persistent memory (CLAUDE.md equiv.) | Open | **P1** |
| Conversation compaction | Open | **P1** |
| `.localcoder` project config | Open | P2 |
| `patch_file` tool | Open | **P1** |
| `undo_last_write` | Open | P2 |
| `fetch_url` | Open | P2 |
| `tree` tool | Open | P3 |
| Multi-file grep with context | Open | P2 |
| Git-aware tools | Open | P2 |
| Interruption/resume | Open | P2 |
| **Token budget tracking** | Open | **P1** |
| Retry/error recovery | Open | P2 |
| Structured output enforcement | Open | P3 |
| **Cost/token display** | Open | **P1** |
| Turn replay/audit log | Open | P2 |
| `--dry-run` | Open | P2 |
| Diff preview before write | Open | P2 |
| `/undo` REPL command | Open | P2 |
| Per-model-family system prompts | Open | **P1** |
| Native function-calling API mode | Open | P2 |
| Thinking/scratchpad budget | Open | P2 |
| MCP server support | Open | P3 |
| `codeneedle` context profiling | Under review | P3 |

### flux_local_setup_guide.md + implementation_plan.md + task.md + walkthrough.md — evaluated

**Status of FLUX / generate_image integration:**

| Step | Status | Notes |
|---|---|---|
| `flux_workflow.json` template | **Done** | ComfyUI API-format, GGUF-compatible node graph |
| `tool_generate_image` in tools.py | **Done** | Queues prompt, polls history, downloads image |
| `FLUX_API_URL` env var support | **Done** | Defaults to `http://127.0.0.1:8188` |
| Offline failure redirect | **Done** | Points to `flux_local_setup_guide.md` |
| Unit tests (mock HTTP) | **Done** | 2/2 passing |
| End-to-end test (live ComfyUI) | **Pending** | ComfyUI not yet installed |
| **VRAM hot-swap** | **Missing** | See below — critical gap |

**Critical gap: VRAM hot-swap not implemented**

`flux_local_setup_guide.md` correctly identifies that a 16 GB VRAM card cannot hold
both the LLM (Gemma4-12B quantised ~8 GB) and FLUX (flux1-dev Q8_0 ~12 GB) simultaneously.
The described solution is:

1. Agent calls `generate_image` tool
2. `generate_image` sends `POST /v1/models/unload` to localm — frees GPU memory
3. ComfyUI loads FLUX and generates the image
4. On the next LLM turn, localm detects the model is unloaded and reloads it automatically

Neither the HTTP server nor the tool implement this today. The HTTP server has no
`/v1/models/unload` endpoint, and `Engine.unload()` / `Engine.load()` are never called
via HTTP. This MUST be implemented before FLUX is usable on single-GPU systems.

---

## Prioritised roadmap

### Phase 1 — Foundational gaps (do first; enables everything else)

#### 1.1  Model hot-swap API in localm HTTP server  (~2h)

**Why now:** Blocks real FLUX use on any single-GPU machine.
**Files:** `localm/inference/http_server.py`, `localm/plugins/coder/tools.py`

New endpoints to add to `http_server.py`:
```
POST /v1/models/unload     → engine.unload(); returns {"status": "unloaded"}
POST /v1/models/load       → engine.load();   returns {"status": "loaded"}
GET  /v1/models            → add loaded: bool to each model entry
```

In `Engine.chat_stream` / `Engine.chat`, add auto-reload if unloaded:
```python
def chat_stream(self, messages, **kwargs):
    if not self._backend.loaded:
        self._backend.load()   # transparent hot-reload
    yield from self._backend.chat_stream(messages, **kwargs)
```

In `tool_generate_image`, before queuing to ComfyUI:
```python
# 1. Ask localm to unload LLM, freeing VRAM for FLUX
_try_localm_unload()   # reads LOCALM_URL env, fails silently if not localm
# 2. Run ComfyUI generation (existing logic)
# 3. Do NOT reload — localm auto-reloads on next /chat/completions call
```

The `LOCALM_URL` env var should be set by `ManagedServer` when it spawns
`localm serve`.

#### 1.2  Token / context budget tracking  (~3h)

**Why now:** Agent has zero visibility into context consumption. Near-limit behaviour
(truncation, hallucination, tool-call malformation) is silent and confusing.

**Files:** `localm/inference/http_server.py`, `localm/plugins/coder/agent.py`,
`localm/plugins/coder/display.py`

Plan:
- HTTP server response includes `usage: {prompt_tokens, completion_tokens, total_tokens}`.
  For the GGUF backend, derive from `llama_get_n_tokens` / count tokens from the
  tokenizer directly (the ctypes wrapper already has `llama_tokenize`).
- `Agent` accumulates `_total_tokens: int` from each LLM call.
- REPL footer after each turn: `[dim]~4,200 tokens  ·  turn 3[/dim]`.
- Warn at 80% of configured context window; offer `/compact` at 90%.

#### 1.3  Persistent memory (`CLAUDE.md` equivalent)  (~4h)

**Why now:** Without cross-session memory the agent forgets repo conventions, user
preferences, and known gotchas. This is one of the most impactful gaps vs Claude Code.

**Files:** `localm/plugins/coder/agent.py`, `localm/plugins/coder/cli.py`,
new `localm/plugins/coder/memory.py`

Plan:
- Look for `LOCALCODER.md` (or `.localcoder/memory.md`) in `cwd` at startup.
- If found, prepend to system prompt under a `## Project Memory` heading.
- `/remember <text>` REPL command appends a bullet to `LOCALCODER.md`.
- `/forget <pattern>` removes matching bullets.
- Entries are free-form markdown — the LLM reads them as context, no special parsing.

#### 1.4  `patch_file` tool  (~2h)

**Why now:** `edit_file` requires exact string matching — brittle on large files or
after whitespace normalisation. Unified diff application is more reliable for
multi-hunk changes.

**Files:** `localm/plugins/coder/tools.py`, `localm/plugins/coder/prompts.py`

```python
def tool_patch_file(cwd, path: str, diff: str) -> ToolResult:
    """Apply a unified diff to a file. diff must be in standard patch format."""
    import subprocess, tempfile, os
    # Try system `patch` first; fall back to Python difflib-based apply on Windows
    ...
```

Teach the agent the format: `diff -u old new` output, with `--- a/file` / `+++ b/file` headers.

#### 1.5  Per-model-family system prompt variants  (~3h)

**Why now:** Gemma4, Llama3, Qwen3, Mistral respond better to different instruction styles.
A single prompt creates unnecessary failures.

**Files:** `localm/plugins/coder/prompts.py`

Derive `model_family` from the model name string:
- `gemma*` — use `<start_of_turn>model` format in examples; note that Gemma uses
  its native `<|tool_call>` format so the system prompt should instruct it to use
  that format rather than the XML format
- `llama3*`, `qwen*`, `mistral*` — standard XML tool-call format
- `deepseek*`, `qwen3*` — add scratchpad hint: "Use `<think>...</think>` before
  deciding which tool to call"
- `phi*` — shorter, more directive prompts (smaller context budget)

---

### Phase 2 — Quality and UX improvements

#### 2.1  Conversation compaction  (~4h)

*Depends on 1.2 for threshold detection.*

When context exceeds ~70% of the context window, summarise old turns using a
second LLM call (same backend) and replace them with the summary. Expose as
`/compact` REPL command and auto-trigger at threshold.

```python
def _compact_history(self) -> None:
    summary_prompt = "Summarise the following coding session in <=300 words, "
                     "focusing on decisions made, files changed, and open problems."
    summary = self.backend.chat([{"role": "user", "content": summary_prompt + ...}])
    self._messages = [
        {"role": "user",      "content": f"[Session summary]\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing."},
        *self._messages[-4:],   # keep last 2 turns verbatim for continuity
    ]
```

<!-- REVIEW NOTE: codeneedle-style recall profiling (see TODO.md, Future Benchmarking)
could provide a model-specific threshold rather than a fixed 70%. Worth exploring
after the codeneedle API is understood. -->

#### 2.2  `fetch_url` tool  (~1h)

```python
def tool_fetch_url(cwd, url: str) -> ToolResult:
    """Fetch a URL and return plain-text content (HTML stripped)."""
    import urllib.request
    ...
```

Add `html2text>=2020.1` to `[project.optional-dependencies.coder]`.
Useful for: docs pages, Stack Overflow, GitHub raw files, package changelogs.

#### 2.3  Diff preview before write  (~2h)

When `auto_approve=False`, show a coloured unified diff before `write_file`
and prompt "Apply? [y/N]". Uses `difflib.unified_diff` in `display.py` with
Rich `Syntax` highlighting.

#### 2.4  Git-aware tools  (~2h)

Add `git_status`, `git_diff`, `git_log` as first-class tools (not `run_shell` wrappers).
Return structured summaries that are easier for the LLM to parse and act on.

#### 2.5  Turn replay / audit log  (~1h)

Write `~/.localm/sessions/<timestamp>.jsonl` — one line per event:
`{"turn": 1, "type": "user"|"tool_call"|"tool_result"|"llm", "content": "...", "ms": 230}`.
Read back with `/replay` REPL command. Low effort, high debugging and reproducibility value.

#### 2.6  `.localcoder` project config  (~2h)

Project-level config file at `.localcoder/config.toml`:
```toml
model = "gemma4-4b"
max_turns = 20
auto_approve = false
memory_file = ".localcoder/memory.md"
```

Loaded in `localm/plugins/coder/cli.py` before Click option processing
(CLI flags always override config file values).

---

### Phase 3 — FLUX end-to-end and VRAM integration

#### 3.1  Manual end-to-end test  (depends on ComfyUI installation)

Procedure from `flux_local_setup_guide.md`:
1. Install Stability Matrix; add ComfyUI with `ComfyUI-GGUF` custom node
2. Download model files to ComfyUI model directories:
   - `models/unet/flux1-dev-Q8_0.gguf` (or Q6_K for lower VRAM)
   - `models/clip/t5-v1_1-xxl-encoder-Q8_0.gguf`
   - `models/vae/ae.safetensors`
3. Load `localm/plugins/coder/flux_workflow.json` in ComfyUI to verify node compatibility
4. Run: `localm coder --model gemma4-4b "generate a photorealistic image of a retro computer terminal and save it to test.png"`
5. Verify `test.png` is created in working directory

Blockers:
- [ ] 1.1 (VRAM hot-swap) must land before testing on a single-GPU machine
- [ ] ComfyUI installation by user

#### 3.2  VRAM hot-swap wiring in generate_image  (depends on 1.1)

After 1.1 lands, update `tool_generate_image` to unload localm before generation.
The `LOCALM_URL` env var is set by `ManagedServer` at startup — the tool reads it.

---

### Phase 4 — Advanced / ecosystem

#### 4.1  MCP server support  (~8h)

Make `TOOL_REGISTRY` dynamic. Third-party tools register via JSON manifests in
`.localcoder/mcp/`. Enables VS Code extensions, company-internal tools, etc.

#### 4.2  Structured output enforcement via grammar sampling  (~4h)

Use llama.cpp's grammar-constrained sampling for tool calls when the backend is
local GGUF. The localm HTTP server can accept a `grammar` field (GBNF format)
and pass it to `llama_sampler_init_grammar`. This eliminates malformed tool calls
from smaller/weaker models and is the local equivalent of OpenAI function calling.

#### 4.3  `codeneedle` context profiling  (research phase)

Two ideas from the TODO:
1. `localm profile <model>` — run a standardised verbatim-recall benchmark
   against the user's own codebase to measure effective context depth
2. Dynamic compaction threshold (see 2.1 note) — use measured recall decay to
   determine when to compact rather than relying on a fixed percentage

Status: **research/evaluate** — no implementation until the codeneedle API is
understood and the per-inference overhead is measured.

---

## Execution order

```
1.1  HTTP model hot-swap         unblocks FLUX on single-GPU
1.2  Token budget tracking       foundational for 2.1 compaction thresholds
1.3  Persistent memory           highest UX impact, low complexity
1.4  patch_file tool             reliability improvement, quick win
1.5  Per-model prompts           reliability improvement

2.1  Compaction                  depends on 1.2
2.2  fetch_url                   quick win (~1h)
2.3  Diff preview                UX polish
2.4  Git tools                   frequently needed
2.5  Audit log                   low effort, high debugging value
2.6  .localcoder config          quality-of-life

3.1  FLUX E2E test               depends on user environment + 1.1
3.2  VRAM hot-swap wiring        depends on 1.1

4.1  MCP support                 architecture work, do after 2.x stable
4.2  Grammar sampling            needs llama.cpp struct work
4.3  codeneedle                  research, no timeline
```

---

## GitHub / housekeeping

- **Repo renamed:** `Matlan1/localm` → **`Matlan1/hearth`**
  GitHub auto-redirects old clone URLs so existing installs keep working.
  Local remote updated: `git remote set-url origin https://github.com/Matlan1/hearth.git`

- **`localllm-coder`:** README replaced with migration guide pointing to localm.
  Recommend archiving on GitHub: Settings → "Archive this repository".

- **Package name:** `localm` unchanged — fully backward-compatible.

---

*Last updated: 2026-06-08*
