# localm: TODO / Feature Roadmap

Coder gaps identified by comparing against Claude Code, Aider, Cursor, Copilot.
GUI gaps identified by comparing against LM Studio, Jan, Open WebUI.

---

## Status note: the `[x]` marks below are a dev log, NOT verified status

The checkboxes in this file are a running development log of what was *built*. They
are NOT a verified statement that each item works end to end - a 2026-06-16
ground-truth audit found that a number of `[x]` items were facades (a description over
a stub, a silent `except: pass`, or a test that asserted nothing). Treat this file as
historical/aspirational.

The verifiable source of truth for what actually works is, in order:

1. `issues/audit-ground-truth-2026-06-16.md` + `issues/test-plans/` - the per-subsystem
   audit and adversarial test plans (claimed-vs-actual, with file:line evidence).
2. The test suite + CI: `pytest` (~1556 tests) and `npm test` (the GUI jsdom harness),
   run by GitHub Actions on every PR. A feature is "done" here only when a test fails
   without the fix and passes with it.

Several `[x]` items the audit proved were facades have since been genuinely fixed and
test-guarded (settings page / `--scope` confinement / MCP `--print-config` / B4 media
containment / privilege checks). The **genuinely open** roadmap items remain the `[ ]`
boxes below (the notable ones: real-ComfyUI verification of music gen, the suite-parity
"Medium" + "Polish/later" lists, the Tauri shell, and the VS Code/Neovim extension).
"Done end to end" still requires exercising the real-model / real-ComfyUI / real-browser
paths, which the mock-based suite does not cover.

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

### Round 3 (shipped 2026-06-11)

- [x] Fix: answered approval cards replayed as still-pending after a page reload - `confirm_resolved` event now recorded in the stream and replay buffer; cards resolve idempotently (covers approve, reject, timeout, and stop)
- [x] Remember the active page across reloads (`localm.activeView`; never written in privacy mode)
- [x] Server-side chat conversation persistence in non-privacy modes: `PUT/GET/DELETE /api/conversations`, stored in `<data dir>/chats/`, merged with the localStorage cache at load; privacy mode unchanged (memory only, 403 on the store)
- [x] Coder session history browser: `GET /api/coder/history[/{name}]` lists past audit logs (`~/.localm/sessions/*.jsonl`) incl. pre-restart sessions; history button in the coder bar + "past sessions" on the setup form
- [x] Settings "clear conversations" also clears the server store when persistence is on

### Round 4 (shipped 2026-06-11) - internet access

- [x] `localm/netpolicy.py`: single policy choke point for model-initiated requests - net_mode off/ask/allow (+ `LOCALM_NET_MODE` env), net_allow/net_deny domain suffix rules, private/loopback/link-local SSRF guard with `net_allow_private` escape hatch, per-hop redirect re-validation, size caps
- [x] Coder: `web_search` tool (DuckDuckGo no-key default, SearXNG via `net_search_url`); `fetch_url` rerouted through the policy; both gated in the agent (off = fail fast, ask = approval flow); privacy-mode stderr audit
- [x] `/api/web/search` + `/api/web/fetch` endpoints (403 on policy refusal)
- [x] Chat: `/web <query>` command and a per-conversation "Web access" toggle - the model emits `<tool_call>` web requests, the GUI executes them through the policy and injects results as visible dimmed "Web" messages (max 3 rounds per send)
- [x] Behaviour change: `fetch_url` to localhost/private addresses is now blocked by default (`net_allow_private true` restores it)

### Round 15 (shipped 2026-06-12) - coder overhaul (QoL round)

Agent core (both surfaces):
- [x] Changed-files tracker: every successful write/edit/patch/notebook-edit recorded with its first-seen original; `changed_files()` + `session_diff()` produce cumulative per-file and whole-session diffs (original → current, not edit-by-edit)
- [x] Mid-task steering: `queue_message()` (thread-safe) injects user messages at the next turn boundary as a steering note - no more "agent is busy" wall; leftovers run as a follow-up task
- [x] Circuit breaker: 4 identical consecutive tool failures abort the task with the conversation intact (hints still escalate at 2 and 3)
- [x] Context meter: turn events carry `ctx_ratio`; CLI turn divider and GUI usage line show `ctx N%` (colored in the CLI)
- [x] Parallel batch timeout (120 s): one hung non-destructive tool no longer blocks the whole batch
- [x] Patch-mode guard: write tools the interceptor can't express as a diff are blocked instead of silently writing to disk
- [x] `read_file(offset, limit)`: re-read the middle of truncated files; truncation note says how
- [x] Explicit truncation markers: grep per-file caps + "N files NOT searched", tree file-limit message with the fix
- [x] `edit_file` failures show the closest-matching file region (line-numbered) instead of a repr blob

CLI REPL:
- [x] `/changes` (files touched) + `/diff [path]` (cumulative session diff, syntax-colored)
- [x] Tab completion for slash commands and project paths; persistent REPL history in `.localcoder/repl_history` (log/full modes only - never privacy)
- [x] `/undo` reports remaining stack depth; `/help` covers everything (and the `\exit` markup typo is gone)

GUI:
- [x] Queue-while-busy through `POST /message` (returns `queued`), with *Queued* feed labels
- [x] Files panel: `GET /files` + `GET /files/diff?path=` endpoints, bar button + `/files` command, per-file and full-session diff views
- [x] Approval cards: "always allow <tool> this session" checkbox (server-side allowlist, shown in session info); confirm timeout configurable via `coder_confirm_timeout`
- [x] Tool cards show args *and* diff, plus elapsed time on the result line
- [x] Dry-run toggle at session setup; final feed line counts changed files; `export` downloads the feed as markdown; audit-log viewer gains a filter box
- [x] Tests: 52 new (agent QoL, tools QoL, GUI session/endpoints) + audit holes closed: checkpoint/resume, undo stack, parallel ordering/timeout, retry streaks, stop midstream, read_env redaction, edit_notebook_cell, patch-mode guard

### Round 14 (shipped 2026-06-12) - video generation

- [x] `localm/video_gen/` - Wan 2.2 TI2V 5B workflow (`wan_workflow.json`, public Comfy-Org stack; gitignored `wan_workflow_local.json` override) + `generate_video()` via ComfyUI: duration snapped to Wan's 4k+1 frame rule (~5 s native at 24 fps, up to 20 s accepted), text-to-video or image-to-video (`start_image` via the shared upload helper), MP4 output, privacy-gated sidecar, VRAM handoff
- [x] API endpoints: `POST /api/video` (progress-streamed job), `GET /api/video/history`, `GET/DELETE /api/video/file/{name}` (confined), `POST /api/video/file/{name}/move`
- [x] GUI "Video" page: prompt/negative/duration/fps/size/seed/steps/CFG/start-image form → job log → inline `<video>` player; history with play/move/delete; `videoMoveDest` scrubbed under the privacy contract
- [x] `/video <prompt>` in chat: inline clip with a video-player message (messages gain a `video` field, rendered like `audio` via blob URLs)
- [x] CLI: `localm video "prompt" [-d s] [--fps n] [--width/--height] [--image start.png] [-o out.mp4] [--seed/--steps/--cfg]`
- [x] Tests: frame snapping, fail-fast before LLM unload, mocked end-to-end with sidecar privacy, save-node output-key variants (`images`/`gifs`/`videos`), endpoint validation + path confinement
- [x] Verified end-to-end 2026-06-12 against a real ComfyUI with the Wan 2.2 5B files: 1 s clip at the native 1280x704, 20 steps → crisp, on-prompt h264 MP4 in ~7.5 min on a 16 GB RDNA2 card (~13.5 s/step + model load); privacy mode left no sidecar; measured timings in docs/video.md
- [x] Quality lesson from the e2e run, folded into template + docs + GUI: the 5B is **720p-native** - sub-native resolutions (e.g. 640x368) produce washed-out mush, so the template default is now 1280x704 and "iterate by shortening the clip, never by shrinking the frame"
- [x] Post-download cleanup: the duplicate clip in ComfyUI's own output dir is deleted when `COMFY_OUTPUT_DIR`/`comfy_output_dir` is set (same behaviour as image generation; found during the e2e run, then verified live on the second render)

### Round 13 (shipped 2026-06-12) - music surfaces

- [x] Music nav page: tags/lyrics/duration (arbitrary seconds)/seed/steps/CFG form → progress-streamed job → inline player; history with play / move-to-folder / delete
- [x] `/music <tags>` in chat: inline generation with an audio-player message (messages gain an `audio` field; bearer-protected files load as blob URLs)
- [x] `localm music "tags" [--lyrics file] [-d seconds] [-o out.flac] [--seed/--steps/--cfg]` with a clear ComfyUI-not-running hint
- [x] `imgMoveDest`/`musicMoveDest` localStorage keys gated + scrubbed under the privacy contract

### Round 12 (shipped 2026-06-12) - voice

- [x] `localm/voice.py` + `[voice]` extra: Whisper STT via faster-whisper (CPU int8 - runs on the GGUF-only base install, no torch); model from config `voice_stt_model` (default "base"), downloaded once on first use, then fully offline
- [x] `POST /api/voice/transcribe` (in-memory decode, never touches disk → privacy-clean; 501 with install hint when the extra is missing)
- [x] 🎤 mic button in the composer (MediaRecorder, click to start/stop, transcript lands in the input)
- [x] TTS with zero backend: 🔊 read-aloud per reply (toggle to stop) + "Speak replies aloud" drawer checkbox - browser speechSynthesis, offline by construction

### Round 11 (shipped 2026-06-12) - assistant memory

- [x] `<data dir>/chat-memory.md`: plain markdown the user can read/edit; `GET/PUT /api/memory` + `POST /api/memory/append`; size-capped; clearing deletes the file
- [x] 🧠 drawer toggle injects memory into the system prompt across all chats; `/remember <fact>` appends a bullet; `/memory` opens a view/edit modal
- [x] Privacy semantics: writes 403 in privacy mode (memory persists conversation-derived facts), reads stay allowed - privacy means no new traces, not amnesia

### Round 10 (shipped 2026-06-12) - prompt library / personas

- [x] `/api/prompts` CRUD on `<data dir>/prompts.json` (atomic writes; explicit user assets, available in every session mode)
- [x] Params drawer: persona select applies system prompt + sampling defaults; save… captures the current drawer values under a name; delete removes the saved persona without touching the drawer
- [x] `/persona <name>` slash command (case-insensitive; bare `/persona` lists what's saved)

### Round 9 (shipped 2026-06-12) - message branching

- [x] Fork-point model: `conv.messages` stays the live linear branch (compaction, retrieval injection, export, and the API mapping untouched); alternative timelines park in `conv.branches` records keyed by the preceding message's id
- [x] Edit-and-fork: editing a sent message parks the old tail as a sibling instead of destroying it; regenerate parks the old reply as a variant
- [x] ‹ k/N › navigation in the message meta row at any fork point; switching writes the live tail back into its slot and splices in the chosen sibling
- [x] Branches persist through the conversation store; cleared by `/clear`; fork records anchored in compacted-away history are pruned (conservatively - anchors inside parked tails are kept)

### Round 8 (shipped 2026-06-12) - conversation organization

- [x] Sidebar search across ALL chats: matches titles and message content; content hits show a one-line snippet; searching auto-expands collapsed groups so matches are never hidden
- [x] 📌 pin-to-top and 📁 folders (hover buttons + `/pin`, `/folder <name>` slash commands); folders render as collapsible groups, collapse state remembered (scrubbed in privacy mode - folder names are conversation-derived)
- [x] `pinned`/`folder` persist through the conversation store (`ConversationUpsert` extended; old stored chats get safe defaults)

### Round 7 (shipped 2026-06-12) - model discovery

- [x] `localm/discover.py`: HF model search (empty query = most downloaded GGUF - dynamic "starter picks", no hardcoded model names), repo tree parsing with quant-label extraction and split-GGUF grouping (sizes summed, first part = pull spec)
- [x] "Fits your VRAM" badges vs **total** VRAM (capacity, not currently-free - the active model occupies the GPU while browsing) using the GGUF preflight overhead estimate; VRAM detection works without torch: torch → nvidia-smi → Windows display-adapter registry (`qwMemorySize`)
- [x] `/api/discover/search` + `/api/discover/files` (403 on net_mode=off - the kill switch covers discovery even though model downloads are otherwise outside the network policy)
- [x] Models page "Find models" card: lazy search (no network call until asked), downloads/likes, expandable per-quant list with colored fit badges and one-click pull into the existing progress flow
- [x] CLI: `localm search [query…]` and `localm search owner/repo --files`

### Round 6 (shipped 2026-06-12) - knowledge / RAG

- [x] `localm/rag/` package: extraction (txt/md/code/html/docx/ipynb stdlib; pdf via the `[rag]` extra), paragraph-aware chunking, pure-stdlib BM25, JSON collection store under `<data dir>/rag/` with mtime-based re-indexing and atomic rewrites
- [x] Lexical-first retrieval by design: the ctypes GGUF binding has no embedding support, so BM25 is the always-on baseline; embeddings (via the server's own `/v1/embeddings`) are stored when available and blended 50/50 - failures degrade, never break
- [x] `/api/rag/*`: collection CRUD, indexing as a progress-streamed job, query, remove-doc, and `/api/rag/extract` (uploaded attachment → text entirely in memory - privacy-mode chats can use documents trace-free)
- [x] GUI: Knowledge page (create/index/search/inspect/delete), chat params-drawer collection selector with cited excerpt injection, paperclip accepts documents alongside images ("Doc"/"Sources" dimmed messages)
- [x] CLI: `localm rag add/list/query/rm`; `[rag]` extra (pypdf only)

### Round 5 (shipped 2026-06-11) - onboarding with no models

- [x] `localm gui` opens model-less on an empty registry (engine starts when the user loads a model) instead of `exit(1)`; `/v1/models` + `/health` null-safe
- [x] `localm gui --pull SPEC` deep-links the browser to the Models page (`?view=models&pull=…`) and auto-starts the download with the existing progress UI; query string stripped after handling
- [x] Launcher **Import** row: *from file…* / *from folder…* register a local GGUF / HF dir via `localm add` (off-thread, selects the new model); *from URL…* launches a model-less GUI with `--pull`; the Web GUI can also be launched with no model selected

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

- [x] `localm/music_gen/` - ACE-Step workflow (`ace_workflow.json`) + `generate_music()` via ComfyUI (arbitrary track length in seconds, lyrics or instrumental, FLAC output, sidecar metadata, VRAM handoff)
- [x] API endpoints: `POST /api/music`, `GET /api/music/history`, `GET/DELETE /api/music/file/{name}`, `POST /api/music/file/{name}/move`
- [x] GUI "Music" page: tags/lyrics/duration form, job log, inline audio player, history with play/move/delete (shipped 2026-06-12, Round 13)
- [x] `/music` slash command in chat - default-length instrumental with an inline player message (Round 13)
- [x] CLI command `localm music "tags" --lyrics file --duration 180` (Round 13)
- [ ] Verify end-to-end against a ComfyUI install with `ace_step_v1_3.5b.safetensors`; document model download in README - **needs a manual run on a machine with the model**; all surfaces are mock-tested

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

## Suite parity roadmap

Gap analysis vs the polished consumer suites (LM Studio, Msty, Jan, Open
WebUI, GPT4All, AnythingLLM), 2026-06-11. Goal: everything below, eventually.
Persistence-touching items are always gated on `effective_mode()` - privacy
mode stays trace-free.

### High impact

- [x] RAG / chat-with-documents (shipped 2026-06-12, see Round 6): in-chat document attachments (in-memory, privacy-clean) + persistent knowledge collections with cited retrieval; lexical-first BM25 with embeddings blended in when the backend supports them ([docs/rag.md](docs/rag.md))
- [x] In-app model discovery (shipped 2026-06-12, see Round 7): HF search on the Models page + `localm search` CLI; per-quant sizes with "fits your VRAM" badges; "starter picks" = most-downloaded GGUF repos (dynamic, nothing hardcoded)
- [x] Conversation search + folders/pinning (shipped 2026-06-12, see Round 8): sidebar full-text search with snippets, 📌 pin-to-top, collapsible 📁 folders, `/pin` + `/folder` commands; persisted through the server store in non-privacy modes
- [x] Message branching (shipped 2026-06-12, see Round 9): edit forks instead of deleting, regenerate keeps the old reply as a variant, ‹ k/N › navigation at fork points
- [x] Persistent assistant memory for chat (shipped 2026-06-12, see Round 11): `chat-memory.md` injected via the 🧠 drawer toggle; `/remember` + `/memory`; writes blocked in privacy mode, reads allowed
- [x] Prompt library / personas (shipped 2026-06-12, see Round 10): named system prompts with sampling defaults, saved/applied from the params drawer or `/persona <name>`
- [x] Voice (shipped 2026-06-12, see Round 12): 🎤 Whisper STT via the `[voice]` extra (faster-whisper, CPU int8, no torch); 🔊 read-aloud + auto-speak via the browser's offline speechSynthesis
- [x] Web search grounding for chat and coder (shipped 2026-06-11, see Round 4): `localm/netpolicy.py` policy choke point (off/ask/allow, domain allow/deny, SSRF guard), coder `web_search` tool + gated `fetch_url`, `/api/web/*`, chat `/web` command + per-conversation web-access toggle with bounded tool loop ([docs/network.md](docs/network.md))

### Medium

- [ ] Hardware monitor in the GUI status bar (live RAM/VRAM/CPU/GPU)
- [ ] Sampler presets; per-model saved defaults; chat-template editor
- [ ] GPU offload / context sliders with live VRAM estimate in Settings (CLI config covers the function, not the feel)
- [ ] Multi-model side-by-side compare; JIT model load + idle TTL auto-unload (currently `Semaphore(1)` + one engine)
- [ ] Download manager panel: background queue, pause, parallel (CLI already resumes)
- [ ] Mermaid diagram rendering; artifacts/canvas live HTML/SVG preview; sandboxed code interpreter
- [ ] Command palette (Ctrl+K), keyboard shortcuts, drag-and-drop files into chat
- [ ] Flash attention / KV-cache quantization / speculative decoding toggles

### Polish / later

- [ ] First-run wizard: detect hardware, recommend a starter model
- [ ] Empty-state funnels ("no models yet → pull one" guided flow)
- [ ] One-file backup / export-import of all user data (chats, prompts, settings)
- [ ] i18n, accessibility pass, mobile/PWA layout
- [ ] Profiles / multi-user accounts (likely out of scope for home use)
- [ ] Native shell, tray, auto-update, installer - tracked above as the Tauri 2 item

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

