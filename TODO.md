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

1. `issues/audit-ground-truth-2026-06-16.md` + `qa/test-plans/` - the per-subsystem
   audit and adversarial test plans (claimed-vs-actual, with file:line evidence).
   (The by-hand test campaign - matrix, results, plans, recorder - lives in `qa/`.)
2. The test suite: `pytest` (~2264 tests) and `npm test` (the GUI jsdom harness). A CI
   workflow exists (`.github/workflows/ci.yml`), but GitHub Actions is currently disabled
   (quota), so PRs are verified LOCALLY (`ruff check .` + the full suite; see the note at
   the end of this file). A feature is "done" here only when a test fails without the fix
   and passes with it.

Several `[x]` items the audit proved were facades have since been genuinely fixed and
test-guarded (settings page / `--scope` confinement / MCP `--print-config` / B4 media
containment / privilege checks). The **genuinely open** roadmap items remain the `[ ]`
boxes below (the notable ones: real-ComfyUI verification of music gen, the suite-parity
"Medium" + "Polish/later" lists, the Tauri shell, and the VS Code/Neovim extension).
"Done end to end" still requires exercising the real-model / real-ComfyUI / real-browser
paths, which the mock-based suite does not cover.

---

## Open items consolidated from dev-notes (2026-06-30)

This is the SINGLE canonical todo / roadmap list. On 2026-06-30 the scattered dev-notes
worklogs and open-point files were re-verified against master (b878a66) by a 6-agent
fan-out; almost all of their `[ ]` items had already shipped. The genuinely-open,
not-already-tracked items are folded in here. Division of labour: **bugs live in
`issues/issues.txt`**, the recovered backlog in `issues/RECOVERED-BACKLOG-localm.md`,
the by-hand QA campaign in `qa/`; everything else (todos/tasks/requests) lives in THIS file.

### Audit follow-ups (full-audit /localm-checkup 2026-06-30 @ d53e8ec)
The BUGS this audit found live in `issues/issues.txt` (the `AUD-*` block, 2 HIGH + 5 MEDIUM
+ 8 LOW); the full write-up + 4 engagement reports are in `dev-notes/checkup/REPORT-2026-06-30.md`.
The non-bug (design / docs / process) follow-ups are tracked here:
- [ ] Config-default migration beyond read-time: a CHANGED default value or a nested-dict subkey
  stays frozen on an existing install (only new TOP-LEVEL keys reach existing users). Add a
  versioned migration for changed/nested defaults (audit LM-DA-004; see also the config-defaults
  discussion). [arch]
- [ ] Plugin-boundary docstring overstates isolation: the `Host`-only plugin boundary is convention,
  not enforcement (a plugin can `import localm.config` directly). Soften the contract docstring, or
  add an import guard only if third-party plugins ever become untrusted (audit LM-DA-006). [chore/docs]
- [ ] Follow-up fuzzing pass for GGUF-native load + GBNF/xgrammar samplers: NOT fuzzable in the
  checkup venv (no provisioned llama runtime / model; `xgrammar`/`pypdf` absent). Run a dedicated
  pass against a provisioned runtime + a tiny model (audit FUZZING coverage gap). [chore/test]
- [ ] (conditional) Named-key hash is unsalted SHA-256 - fine today because keys are 256-bit random
  `token_urlsafe`, but if keys ever become user-chosen, switch to a salted KDF (audit SA-5). [security/arch]

### Setup / installer
- [ ] Explicit HuggingFace opt-in at install: setup.bat/setup.sh auto-install the torch stack
  from TORCHSPEC today; ask "Run HuggingFace (non-GGUF) models too? [y/N]" first instead of
  pulling the heavy stack unprompted (U4/U5; setup.bat:148-164 / setup.sh:199-217)
- [ ] `localm setup-torch [variant]`: add the HF/torch stack AFTER install without re-running
  the whole installer (U6)
- [ ] Reject unknown plugin tokens loudly: `localm plugin ...` junk-accepts a bogus name (e.g.
  "ewew") instead of erroring (GO-PUBLIC-READINESS note)

### Server / coder
- [ ] Completions SSE error parity: `_stream_sse_completion` swallows a mid-stream inference
  error (bare try/finally) while chat's `_stream_sse` emits `[inference error: ...]`; surface it
  on the completions path too (~5-line fix; B17b, HANDOFF-2026-06-20)

### Media
- [ ] Autodetect `comfy_launch_cmd`: optional follow-up to the U10 Settings dir-picker - find the
  ComfyUI launcher automatically instead of requiring the path (HANDOFF-2026-06-22)

### Release / licensing (pairs with META-1 + the Security-pentest standing item below)
- [ ] Ship MIT notices for the bundled llama.cpp / ggml binaries with the release artifacts (S4)
- [ ] Add `CONTRIBUTING.md` + a CLA/DCO template BEFORE accepting any external PR (S5; neither
  exists in the tree today)

### Tester distribution via the bug-report proxy (private-repo testers, no GitHub account)
The Cloudflare Worker bug-report proxy shipped (PR #266: in-app "Send to maintainer" files a
GitHub issue via a server-side Issues:write token; `tools/bugreport-proxy/`). Two more surfaces
were designed to ride the SAME Worker (full design + build checklist in
`dev-notes/self-updater-design-2026-06-30.md`, local/gitignored). NOT built yet.
- [ ] **Self-updater (check-only auto, apply always user-initiated).** Most updates = swap files
  + reboot because the install is editable (`uv pip install -e`) and the R18 `os.execv` restart
  exists; only `pyproject`-deps changes need `uv pip install`, only a llama.cpp build bump needs
  `setup-llama`. Source = published GitHub Releases (a build zip), pulled through the Worker via a
  SEPARATE `UPDATE_GITHUB_TOKEN` (Contents:read), shared secret then mandatory. Needs: a live-read
  `VERSION` file + `localm/_version.py` (editable-install dist-info does not update on a code swap);
  `localm update [--check]` + a Settings "Check for updates" button + a quiet throttled startup
  check (notify only, never auto-apply); a detached apply helper with backup + health-checked
  rollback that never touches `.venv`/data/`.git`/config/models; a `make-release` packaging helper.
- [ ] **Issues tracker (read-only).** `localm issues [--open|--closed|--all]` + a GUI "Issues" view
  showing open/resolved issues (highlight the tester's own filed reports) via a Worker `GET /issues`
  route reusing the Issues token. Closes the loop: report -> track -> "fixed in vX" -> update.
- Branch `claude/self-updater` exists (design doc only); no code yet.
- Future hardening (not v1): cryptographically sign the release zip + verify in-app before apply.

### Real-hardware / real-resource verification (the suite is mock-based)
- [ ] Real-HW inference verification on NVIDIA + Intel + macOS - dev box is AMD-only; the code
  paths are unit-tested and docs flag them "experimental/unverified" (V3/V4)
- [ ] VRAM unload -> media-gen smoke on a real 16 GB AMD GPU (swap code merged #114, never
  re-confirmed live) (V1)
- [ ] Real-API integration test for the OpenAI / Anthropic coder backends (currently mock-only) (B18)
- (Music real-ComfyUI E2E with `ace_step_v1_3.5b` is already tracked at the Music item below.)

### Bug candidates surfaced during consolidation (triage into issues/issues.txt - your call)
Not added to issues.txt yet; each needs a quick live verify (may be cosmetic or already fixed):
- R39: on Windows, closing the console window can still print a native forrtl/rocBLAS abort even
  though the CTRL_CLOSE cleanup handler runs (winconsole.py) - distinct from SRV-CTRLC
- U-STOP: confirm the chat Stop button aborts inference SERVER-SIDE (not just the client stream);
  the note said it persisted a partial reply / kept reading aloud / triggered web
- VIS-1 image-wedge: confirm an image sent to a text-only model rejects cleanly without wedging
  the engine (the VIS-1 messaging work #221 likely covers this - verify)

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

## Release readiness, security & QA (standing items)

PERMANENT, recurring gates - never "done"; re-run and refreshed each release cycle.

- [ ] **Security pentesting** - run a full pen-test before any public/external handoff and
  refresh it each release. Latest: the user-content XSS render path was reviewed 2026-06-23
  (dev-notes/SECURITY-xss-render-review-2026-06-23.md) - SAFE (DOMPurify-gated), 0 exploitable;
  the one gap (no CSP backstop) now has a report-only CSP shipped server-side, to be flipped to
  enforcing after the GUI inline-script nonce work. Top probes next time: (1) DOMPurify
  default-config mXSS bypass (the keystone), (2) artifact-iframe sandbox + CSP-injection regex,
  (3) any new innerHTML / HTML-injection sink reachable by model/server strings.
- [ ] **Live QA** - keep the by-hand QA campaign continuing and refreshed (the gitignored `qa/`
  feature matrix): deep, adversarial real-use of every feature in GUI and CLI each cycle, not a
  one-time pass.
- [ ] **ZIP-handoff readiness (META-1)** - define and maintain a concrete "external tester is
  handed the repo as a ZIP" checklist (cold install succeeds on a clean box; every plugin works;
  no machine-specific assumptions; docs honest about the AMD-only-tested caveat) and diff the
  project against it before handoff. See dev-notes/GO-PUBLIC-READINESS-2026-06-22.md.

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

(Statuses reconciled against the code on 2026-06-22 - several were shipped by
later rounds but never re-marked here. Verified by grep / reading the GUI source.)

- [x] Hardware monitor in the GUI status bar (live RAM/VRAM/CPU/GPU) - shipped (`#hw-stats` + renderHwStats, #109)
- [x] GPU offload / context sliders with live VRAM estimate in Settings - shipped (Settings -> Performance: `#sec-performance` sliders + `perf-estimate`, #112)
- [x] Command palette (Ctrl+K), keyboard shortcuts, drag-and-drop files into chat - shipped (cmdk #111; drag-drop `.drag-over` #110)
- [x] Idle TTL model auto-unload - shipped (opt-in `idle_unload_seconds`, A5, #203)
- [x] Artifacts / canvas live HTML/SVG preview - shipped (hard-sandboxed iframe + CSP, A3, #203)
- [x] Sampler presets - shipped as personas (a saved system prompt + sampling defaults, Round 10)
- [ ] Per-model saved defaults; chat-template editor (personas cover global presets, not per-model defaults or template editing)
- [ ] Multi-model side-by-side compare; JIT model load; keep several models resident (A6 - still `Semaphore(1)` + one engine)
- [ ] Download manager panel: background queue, pause, parallel (CLI already resumes)
- [ ] Mermaid diagram rendering; sandboxed code interpreter (artifacts/canvas split out above as shipped)
- [ ] Flash attention / KV-cache quantization / speculative decoding toggles (the llama struct fields exist in the ctypes binding; no user-facing toggle yet)

### Polish / later

- [ ] First-run wizard: detect hardware, recommend a starter model (C1; the pieces exist - doctor, fit-badged pull, plugin setup - but not the guided path)
- [ ] Empty-state funnels - partly shipped (`localm gui --no-model` + `--pull` deep-link, Round 5); a fuller guided "pull one" flow still open
- [ ] One-file backup / export-import of all user data (chats, prompts, settings)
- [x] Mobile / PWA layout - shipped (installable PWA + native-feeling mobile chat, #202)
  - [ ] Mobile keyboard handling: the on-screen keyboard can hide the composer on iOS (`100dvh` does not shrink for it); add a `visualViewport`-based inset so the input always floats above the keyboard
  - [ ] Real-device verification of #202 (it was verified by DOM measurement on Android-sized viewports only, never on a phone): notch / home-bar `env(safe-area-inset-*)`, and iOS Safari (16px no-zoom font, `viewport-fit=cover`, Add-to-Home-Screen)
  - [ ] Real-device verification of #201 auth-cookie persistence: confirm the API key survives a full browser / installed-PWA close+reopen (it is now a ~400-day persistent cookie, not a session cookie). Mobile OSes can still evict cookies under storage pressure / ITP even with `max-age`; QR-pair is the fallback.
  - [ ] Optional companion-app polish: show the conversation title / active model name in the mobile top bar (currently just the "localm" wordmark)
- [ ] i18n + accessibility pass
- [ ] Profiles / multi-user accounts (out of scope for home use; scoped API keys already cover per-device sharing)
- [ ] Native shell, tray, auto-update, installer - tracked above as the Tauri 2 item

---

## Competitive analysis 2026-06-22 (vs LM Studio / Jan / AnythingLLM / llama.cpp + Claude)

Source: two XDA articles (local-LLM tools tested; Jan won) + a Claude
consumer-feature article + the MCP docs, compared against the code. The full
criteria matrix and per-tool analysis live in the maintainer's local dev-notes
(not in git). Statuses below were code-verified on 2026-06-22.

### Shipped + merged this session (#203)

- [x] A5 - idle-unload VRAM TTL (opt-in `idle_unload_seconds`, 0 = off) [LM Studio / Ollama]
- [x] B4 - `localm ps` / `localm status` (running per-directory instances) [`lms ps`]
- [x] A3 - Artifacts canvas: html/svg reply blocks in a hard-sandboxed iframe (no same-origin; a CSP blocks all network) [Claude Artifacts]
- [x] A2 - memory auto-synthesis: `localm job add --memory` distils durable facts from session logs into chat-memory.md, privacy-gated [Claude memory synthesis]

### Open - decision needed

- [ ] A1 - curated, toggleable MCP-server catalog surfaced in chat [Jan]. The catalog/toggle/persist half is straightforward; "execute MCP tools in the GENERAL chat path" is a security design fork (chat does no tool-calling today). DECIDE: (a) reuse the coder tool-loop + per-call approval, (b) a new chat-only mechanism, or (c) ship catalog/toggle only and defer execution.

### Open - to implement

- [ ] B1 - workspaces: bundle a system prompt + a RAG collection + scoped memory, inherited by chats inside it [AnythingLLM workspaces / Claude Projects]
- [ ] B2 - RAG ingestion connectors: GitHub repo / website / YouTube transcript, netpolicy-gated [AnythingLLM] (next conflict-light pick)
- [ ] A4 - audio-understanding input (Voxtral / Qwen-ASR via mmproj). NOTE: the GGUF VISION half already shipped (#200); this is the audio half (native + needs an audio model to verify) [Jan / llama.cpp]
- [ ] A6 - keep several models resident (parallel slots / LRU); unblocks instant routing + side-by-side (also in the Medium list above) [LM Studio / llama.cpp]
- [ ] A7 - auto model-routing / tiers with a speed/VRAM hint; needs A6 [Claude tiers]
- [ ] B3 - reasoning-effort selector. CAVEAT: local GGUF has no native effort knob, so this risks being a facade; needs a real design or a drop [Claude effort levels]
- [ ] C2 - review navigation / information architecture before adding more tabs
- [ ] C3 - GUI visual-polish pass (keep the zero-build approach) [LM Studio polish]
- [ ] C4 - one-tap remote attach (extend the QR device-pairing) [LM Studio LM Link]
- [ ] C5 - a general quantize / convert CLI (abliterate can only `--export-gguf`)
- [ ] (C1 first-run wizard is tracked under Polish / later above)

### Feature / design ideas (food-for-thought, not committed)

- [ ] Agent-runner "manager view": queue N coder tasks, watch several sessions, review each diff (near coder + jobs + C2) [Antigravity-as-agent-runner]
- [ ] Transparent hybrid local<->cloud router: pre-classify a query, show "which model will answer", escalate fact-sensitive prompts to an opt-in cloud model (enhances A7). localm already grounds on web access as its local answer to hallucination (#199).
- [ ] "Research mode": a jobs/agent task that fans out web + RAG into a cited briefing report (enhances B1/B2; the pieces exist)

### Known gaps / follow-ups

- [ ] Apple Metal/MLX and Intel SYCL backends - localm is weakest of the field on Mac / Intel
- [ ] Docs: document the 3 shipped surfaces (`idle_unload_seconds`, `localm ps`/`status`, `localm job add --memory`) in README + docs/jobs.md
- Note: CI is disabled (Actions quota), so PRs are verified LOCALLY; a periodic `ruff check .` + full suite sweep is the only regression backstop (3 pre-existing ruff errors had slipped onto master and were cleaned in #203).

## Auth, TLS & media-workflow follow-ups (2026-06-22)

From the seamless auth/cert session (#201) + the reopened I3 Wan-video item.
`issues/issues.txt` carries the bug-level detail (K1, I3);
`dev-notes/SESSION-seamless-auth-cert-2026-06-22.md` is the session record.

### Shipped + merged this session (#201)

- [x] Seamless auth - the API key PERSISTS across a browser/PWA restart: both
  `localm_session` + `localm_csrf` now carry a shared ~400-day `max_age` (replacing
  the session cookies that were dropped on close); and the key gate stops nagging to
  reinstall the cert - it only shows "Install certificate" when the CA is genuinely
  untrusted, signalled by `window.__swFailed`. Closes the deeper root cause behind
  J1/J2. Logout stays reachable (Settings: leave the key blank and Save).

### Open - to implement

- [ ] I3 - Wan video "model not in list" is a workflow-management gap, not just the
  user's missing files. Approved scope ("validate models pre-submit"): (a) role-based
  param injection in `localm/video_gen/comfy.py` - resolve KSampler / positive+negative
  prompt / latent (width/height/length/start_image) / CreateVideo (fps) nodes by
  `class_type` + graph connections instead of hardcoded node IDs, so a user's OWN Wan
  workflow works; (b) pre-submit validation against ComfyUI `/object_info` that names
  the exact missing model file BEFORE the chat model is unloaded (auto-substitute a
  close precision-variant where unambiguous); (c) an actionable error pointing to the
  Workflow panel; then give image (`image_gen/comfy.py`) + music (`music_gen/comfy.py`)
  the same treatment. Diagnosis done; no code written yet.

### Open - verify / decide

- [ ] `tls.py` SAN-coverage edge: a bind to a specific IP/hostname not in
  `san_targets()` auto-discovery could leave the browser "not secure" even with the CA
  trusted. Common case is covered (`cli.py` passes the bind host). Act only on a real
  repro.
- [ ] Decision to confirm: the auth-cookie lifetime shipped at ~400 days (the browser
  cap) to honour "store the key permanently". A shorter TTL (30d, or 7d on a network
  bind) trades seamlessness for a smaller local-device-theft window - one-line change
  if the maintainer prefers it.
- (Real-phone persistence check is tracked under the Mobile / PWA item in "Polish /
  later".)

## Future Benchmarking (Under Review)

<!-- REVIEW MARKER: Integration of alexziskind1/codeneedle for local GGUF context profiling.
     * Idea 1: Add a `localm profile` or `localm benchmark` CLI command to let users evaluate actual context window recall decay on their specific hardware/quantization levels.
     * Idea 2: Add a `localcoder profile-project` command to run verbatim function recall tests against the current workspace codebase, determining which local model is most reliable for the project's size. -->

## Recovered backlog: untracked items (verified vs #267, 2026-06-30)

Reconstructed from lost session logs, cross-verified against the code (origin/master
#267) by an independent verifier + adversarial counter-check, confirmed genuinely
open/partial, and confirmed ABSENT from the rest of this TODO at the time of writing.
97 distinct items (deduplicated from 183; the 25 bug items and 11 security findings were moved to issues/issues.txt per the bugs/findings-live-in-issues convention). "(P)" = partially done. "rec#N" = the
recovered-backlog id(s); full per-item evidence lives in the gitignored local files
issues/RECOVERED-BACKLOG-localm-VERIFIED-2026-06-30.md and
issues/TODO-RECONCILIATION-2026-06-30.md.

Related: the bug + security-finding items moved out of this list live in
issues/issues.txt (the RECOVERED-BACKLOG BUGS block + the SECURITY section).

### Security (3)
- [ ] Reconsider a scoped network policy for the coder run_shell capability (P) - run_shell is owner/coder:full-gated, but a scoped NETWORK policy for shell-spawned access is explicitly not implemented (gated only by human approval).  (rec#245)
- [ ] Gate /v1/chat/completions (+completions/embeddings) on a grantable CHAT scope (KEY-SCOPE-1) - Routes are gated on any-valid-key not require_scope(CHAT), so an mcp-only key can still chat; needs key-mgmt design (chat reframed as baseline by the maintainer).  (rec#249,611)
- [ ] Add plugin sandboxing for untrusted third-party code - Third-party plugin Python still runs unsandboxed in-process (no sandbox or signing in v1).  (rec#351)

### Testing / verification (11)
- [ ] Rewrite the entire test suite from scratch to kill mock-theater (VRAM-escape exemplar) (P) - Some real-behavior/contract tests were added (incl the unblocking unload-auth test), but the mandated from-scratch ~142-file rewrite with per-cluster mutation/negative testing is not done.  (rec#87,178,498,588,617)
- [ ] Add test_music_gen_robustness.py for the Music page - No test_music_gen_robustness.py mirroring the image one, despite the music page existing.  (rec#225)
- [ ] Verify img2img dimension fix renders at native size end-to-end (P) - The fix code + a unit test are present, but the live-ComfyUI E2E native-size render cannot be verified from the worktree.  (rec#230)
- [ ] Verify Linux AMD ROCm GPU acceleration end-to-end on real hardware - Hardware-verification item not closable by code inspection; docs still mark native-Linux AMD ROCm best-effort/unverified and WSL2 ROCm experimental.  (rec#339)
- [ ] Test HF vision and audio multimodal input paths (P) - The HF multimodal image/audio processing path exists but testing it needs a vision/audio HF checkpoint not on the AMD box.  (rec#377)
- [ ] Test split-GGUF model load preflight (P) - The missing-part preflight is implemented but testing it needs a multi-part GGUF not in the test set.  (rec#378)
- [ ] Verify model pull --name/--sha256/--redownload by a real download (P) - All three flags pass through, but the actual real-download verification cannot be performed in-repo.  (rec#388)
- [ ] Verify temperature parameter actually changes output distribution (U7) - Pass-through is tested but no behavioral test confirms different temperatures change the distribution (model/HW-dependent).  (rec#477)
- [ ] Add a CI grep gate for dangerous DOM HTML-injection patterns (R41-D6) - No CI grep gate fails on dangerous DOM HTML-injection / dynamic-code patterns in non-vendor static JS.  (rec#567)
- [ ] Add light extras ([voice]/[audio]) to CI coverage - CI installs only [dev,rag]; light [voice]/[audio] extras exist but their tests still importorskip out.  (rec#628)
- [ ] Broader plugin-first test-suite review (CI blind spots + stale tests) - No tracked plugin-first coverage / CI-blind-spot audit artifact; de-rot/isolation work is maintenance, not the parked audit.  (rec#630)

### Features (33)
- [ ] Multi-modality model browser (search/pull diffusion image + transcription/ASR models) - Model browser searches/pulls GGUF text LLMs only; no diffusion or Whisper/STT model discovery/pull.  (rec#1,614)
- [ ] Add agent-enqueued goals / when-file-changes triggers to the jobs scheduler - Scheduler supports cron/interval only; no agent-enqueued-goal or file-change trigger.  (rec#5)
- [ ] Implement server-side native tool parsing for local models - /v1/chat/completions neither accepts tools= nor emits structured tool_calls; only cloud providers use native tools.  (rec#41)
- [ ] Add agents-emit-reviewable-Artifacts pattern (plans, recordings, self-validation) (P) - Verification primitives exist but saved plans / browser recordings / validation artifacts are not emitted; the plan streams inline.  (rec#49)
- [ ] Build Phase 6b: server accepts the per-instance registry token as an owner-equivalent bearer - Attach token is owner-equivalent only for the GUI surface-mount; a general server-accepted owner bearer for keyless coder/thin-client attach is deferred.  (rec#100,144)
- [ ] Build an AI-assisted dialogue-tree creator - Dialogue-tree creator is only a floated idea; nothing built on the branch-anywhere edit system.  (rec#151,173,504)
- [ ] Build the A1111/sdapi-compatible image backend (support media backends beyond ComfyUI) - Only the seam exists; no A1111/sdapi adapter, dispatch branch, or GUI picker, so non-ComfyUI image gen is unsupported.  (rec#154,620)
- [ ] Implement plugin architecture Phases 5-8 (settings migration, pip-install extras, example plugin, chat control) (P) - Phases 0-3 wired but the four Phase 5-8 gaps (flat->nested migration shim, pip-install requires_extras, example plugin, commands/tools on PluginSpec) remain.  (rec#188)
- [ ] Implement remote plugin install via GitHub (fetch a plugin missing from the store) - GitHub remote-fetch-on-install is an unreachable stub: GITHUB_BASE empty and provisioning raises NotImplementedError.  (rec#189,441,631,685)
- [ ] Build the interop adapters roadmap (OWUI Tools, Filters, oobabooga, Pipes) - Only the chat-pipeline hook and skills importer ship; the OWUI Tools/Filters/Pipes and oobabooga adapters are all unbuilt.  (rec#205,483)
- [ ] Make 'localm' usable as a resolvable address on the local machine / LAN (R33) (P) - An mDNS service broadcast was added under <hostname>.local, but there is no browser-resolvable 'http://localm' name, localm.local A-record, or hosts-file mechanism.  (rec#264,545)
- [ ] Build the OpenWebUI Pipes adapter (virtual-model backend) - No virtual-model backend whose inference calls pipe(body); scheduled 'bigger, later' and absent.  (rec#322,634)
- [ ] Build OpenWebUI Tools adapter (openwebui-compat plugin: loader + Tools/Valves) - No openwebui-compat loader/Tools adapter plugin exists; only the coder MCP/plugin tool-registry machinery it would reuse.  (rec#323,632)
- [ ] Build plugin Phase 8 chat control surface (manage the app via scope-gated tools) - No scope-gated chat-management tool layer; Phase 8 only referenced in plan/scopes comments.  (rec#328,436,637)
- [ ] Implement per-user gating for chat hooks (principal/scopes reserved) - ChatHookContext.principal/scopes are reserved for future per-user gating and unset in open mode; no per-user gating implemented.  (rec#332)
- [ ] GGUF embedding support in the ctypes binding - GgufBackend.embed raises NotImplementedError and /v1/embeddings returns 422 for GGUF, so RAG hybrid retrieval is BM25-only by default.  (rec#333)
- [ ] Embedding-based semantic code search for the coder - Coder index is a symbol list and recall is BM25; no cosine/vector code search though /v1/embeddings exists.  (rec#341)
- [ ] Image generation: batch sweeps, inpainting, ControlNet, upscale + BaseImageBackend (P) - Pluggable image-backend seam exists but no inpainting/ControlNet/ESRGAN-upscale/batch-sweep and no shipped A1111 impl.  (rec#343)
- [ ] GitHub tool set for the coder (create_pr, list_issues, CI log fetch) - No create_pr/list_issues/comment_on_pr or CI-log-fetcher coder tools.  (rec#345)
- [ ] Build Flow-A vision-router auto-switch / captioner-route to a VLM (P) - A2 GGUF mmproj is wired, but A4 VLM auto-switch and A3 captioner-route are unbuilt; the server only suggests a manual `localm run <vlm>`.  (rec#416,451)
- [ ] Add an opt-in chat-assisted 'refine my prompt' step for media generation (Flow B) - No opt-in step where the chat model expands a short prompt into a detailed generation prompt with confirm-before-generate.  (rec#417,587)
- [ ] Build a real updater + verify/repair commands (localm update/repair/verify) - No update/repair/verify command; doctor only reports and nothing verifies venv/package/registry/store/RAG integrity.  (rec#427)
- [ ] Server-side RAG injection for non-GUI clients (migrate client-side app.js injection to server hooks) - RAG/memory/web injection is still assembled client-side in app.js; not migrated onto the server-side chat-pipeline inlet for non-GUI clients.  (rec#431,578,635,690)
- [ ] Ship a first-party plugin that consumes register_chat_hook + extend pipeline to MCP chat tool (P) - A hook consumer exists and /v1/completions runs the pipeline, but the MCP chat tool still bypasses it via engine.chat_stream directly.  (rec#433)
- [ ] Add CLI REPL /generate-music and /generate-video commands - REPL exposes only /generate-image (+/imagine); no /generate-music or /generate-video slash commands.  (rec#437)
- [ ] Curate and badge coder-capable local models in the picker + docs - No coder-capability tagging, one-click pull, or 'may narrate / weak at tools' badge exists.  (rec#461,582)
- [ ] Add plan-then-execute mode for multi-file coder tasks - Only a read-only --estimate dry-run exists; no plan-then-write-one-by-one mode with per-file confirmations.  (rec#462,583)
- [ ] Vet and build a Skills manager (discovery + install front-end from trusted repos) - No skills-manager front-end pulling/installing skills from curated trusted repos; trust/secret-hygiene crux still only researched.  (rec#489,622)
- [ ] Publish an opt-in ai-catalog.json + LiteLLM provider interop + background/async coder runs - No opt-in ai-catalog.json, no LiteLLM-provider confirmation, and no background submit/poll/retrieve coder API.  (rec#579)
- [ ] Add a custom coder-instructions field (GUI/CLI --system + .localcoder/system.md) - Only LOCALCODER.md memory exists; no custom-instructions GUI/CLI field, --system flag, or .localcoder/system.md.  (rec#584)
- [ ] Vet and build custom model sources (e.g. Civitai with per-site login) - Model sources are HF/URL/Ollama only; no Civitai or per-site login/token fetch adapter.  (rec#623)
- [ ] Build OWUI-Filter / oobabooga text-hook adapters on register_chat_hook - Only the chat-pipeline hook primitive ships; no OWUI-Filter or oobabooga text-modifier adapter built on it.  (rec#633,688)
- [ ] Add full OWUI dunder shimming + GUI filter toggles + a shipped example plugin (Phase 7 parity) - Full OWUI dunder shimming (__user__/__event_emitter__/__request__), GUI filter toggles, and an example plugin are all absent.  (rec#691)

### Enhancements (24)
- [ ] Extend the generation budget mid-stream / dynamic mid-generation token budget (CHAT-3) - _fit_generation_budget clamps at prefill and finishes with reason 'length' even with VRAM free; no mid-stream budget extension.  (rec#13,524)
- [ ] Implement lazy/triggered grammars behind the coder_tool_grammar flag - The flag only wires TOOL_CALLS_ONLY (not lazy/triggered grammar); deferred/dormant pending a grammar-capable runtime.  (rec#40)
- [ ] Implement self-calibrating media VRAM peak measurement per workflow (Phase 1.5) - Media VRAM is still a conservative static per-backend estimate; no mem_get_info / system_stats runtime peak cached per workflow.  (rec#134,170,422)
- [ ] Make media backend config fail loud for unimplemented backends - Still silently falls back to comfy for any unknown backend name; the requested fail-loud 'backend X not implemented' behavior was not adopted.  (rec#155)
- [ ] Switch music/video history move-destination from prompt() to a folder picker - Music/video history move still uses prompt(); the existing folder picker was not wired to these surfaces.  (rec#210)
- [ ] Professional overhaul of the model/asset download progress bar - Pull progress is still the same modest dl-bar component; no professional overhaul of the download-progress experience.  (rec#262)
- [ ] Broader setup flow/wording redesign pass (SETUP-3 umbrella) - Discrete SETUP-3 sub-bugs shipped, but the guided first-run wizard / guided-pull / wording overhaul remains an unchecked TODO.  (rec#275)
- [ ] Multi-sequence (batch) GGUF inference - The ctypes binding is single-sequence (get_one path); inference is serialised by a semaphore with no parallel sequences.  (rec#335)
- [ ] Expose RoPE / YaRN context extension configurable without reload - rope_scaling_type/rope_freq_scale exist only as C-struct ABI fields; not exposed as configurable params and not without reload.  (rec#347)
- [ ] GGUF metadata inspection before load (architecture/params/quant/ctx) - No pre-load GGUF header/KV inspection of architecture, parameter count, quant level, or context limit.  (rec#348)
- [ ] localm update <name> stale-model / stale-quant detection - No localm update or SHA/timestamp-vs-HF stale-quant detection exists.  (rec#349)
- [ ] Fix LAUNCHER-UI cluster: stable width, info-text truncation/marquee, responsive resize (P) - Width-stability shipped, but the window is still non-resizable and the info text is statically ellipsized with no model-name marquee.  (rec#380)
- [ ] Add a plugin-load progress indicator at startup (server + client) - Plugins load synchronously with no 'loading plugins' nav indicator; the only such text is the Plugins-page catalog render.  (rec#381,410)
- [ ] Add a GUI embed on/off toggle for RAG indexing - Indexing UI always defaults embed=true server-side; no GUI on/off embed toggle though the backend supports the param.  (rec#438)
- [ ] Record embedding-model identity with stored RAG vectors + add verify/rebuild/repair (P) - dim is written and rag repair/rebuild exists, but no embedding-model identity is stored (same-dim swap still corrupts scores) and no dedicated verify command.  (rec#444,593)
- [ ] Improve coder GUI look/feel (render-on-no-op, no-progress banner, presets, inline tests, per-hunk confirm) (P) - Live feed and a Stop button exist, but safe-review/trusted-build presets, no-progress banner, inline test results, and per-hunk confirm/edit-before-apply are absent.  (rec#464,585)
- [ ] Add multi-step undo stack / revert-all and file-tree-of-changes to the coder - Changed files are a flat list, GUI undo is single-step with no visible stack, and there is no revert-all action.  (rec#502)
- [ ] Auto-run tests after coder writes and show pass/fail inline with a fix button (P) - run_tests exists and a nudge message is shown, but tests are not auto-executed and there is no inline pass/fail or 'fix the failing test' button.  (rec#503)
- [ ] Harden the coder against test-rewriting: protect CI/test config as confined-edit (R19) (P) - Test/CI-config edits are flagged advisory-only; no confined-edit gate and it counts files, not changed assertions.  (rec#575)
- [ ] Add a reversible tool-result compression hook + byte-stable prompt prefix for prompt caching (R19) (P) - Tool-result compression is lossy/irreversible with no retrieve tool, and no byte-stable system-prompt-prefix work for llama.cpp prompt caching.  (rec#576)
- [ ] Add per-model coder harness profiles + run_shell tool-call cap + duplicate-command detector (R19) (P) - Per-family harness profiles are wired (keyed on family not model-id); no run_shell tool-call cap or duplicate-command detector.  (rec#577)
- [ ] Add a runtime 'launch directly' shortcut toggle - Only the install-time launch choice exists; no runtime toggle for the launch-directly shortcut behavior.  (rec#662)
- [ ] Subprocess-isolate native crashes (segfault / OOM-kill / os._exit) - The graceful-failure net catches only Python exceptions; native llama.dll crashes need subprocess isolation, not done (inference runs in-process).  (rec#664)
- [ ] Wire the in-process 'localm run' REPL into the chat-pipeline hooks - The hook chain runs only for /v1/chat/completions; the in-process REPL still does not invoke the pipeline.  (rec#689)

### Infra / CI (5)
- [ ] Fetch an ai-dock prebuilt for a self-contained Linux native-CUDA path - No ai-dock fetch; the Linux CUDA path uses the non-self-contained upstream prebuilt and requires the CUDA runtime present.  (rec#81)
- [ ] Give Linux the same treatment as Windows (parity umbrella) + assess launcher progress (P) - Linux+all-vendor scaffolding exists, but 'works seamlessly on every HW+OS combo' is an unfinished umbrella; Intel/NVIDIA/Mac paths self-documented as unverified.  (rec#98,115)
- [ ] Give Linux AMD a first-class [gpu] install route (provide Linux ROCm torch wheels) (P) - The [gpu] extra hard-pins torch/rocm to win32 only, so Linux+AMD users get no GPU torch from the extra; only docs advanced.  (rec#401,658)
- [ ] Route Linux+NVIDIA CUDA pick to Vulkan with an honest message (U3) - The dead bin-ubuntu-cuda matcher is still live (not routed to Vulkan), and the SYCL 'needs oneAPI runtime' note is not made Linux-only.  (rec#548)
- [ ] Unify hardware detection across setup.sh and hwdetect.py (delete the bash detector) (P) - The Intel-Arc-defaults-to-cpu drift is effectively closed via python hwdetect, but the prescribed cleanup (delete the bash detector) was not done.  (rec#625)

### Ops (5)
- [ ] Investigate automating stale-branch deletion vs manual git push origin --delete - No stale-branch deletion automation; an unresolved decision with manual delete as status quo.  (rec#60)
- [ ] Set up cli-music (ACE-Step) and cli-video (umt5) models on the dev machine (P) - Dev-machine ops act not determinable from code; video was verified, music remains a manual local run.  (rec#96)
- [ ] Add a GitHub Sponsor button / FUNDING.yml - No .github/FUNDING.yml; the Sponsor button is still parked.  (rec#156)
- [ ] Answer PyPI name-locking downsides and publish a placeholder to lock 'localm' - No publish workflow exists and no evidence the 0.1.0 placeholder was published to lock the name.  (rec#214)
- [ ] Winget / Homebrew packaging manifests + signed installer - No winget or Homebrew manifests or signed installer present.  (rec#350)

### Refactor (4)
- [ ] Define a formal MediaBackend protocol when a second backend lands - No formal MediaBackend protocol; backend is still 'a module selected by config name', correctly deferred until a second backend lands.  (rec#329)
- [ ] Make capability awareness a kernel concern (one vision/audio/gen detector) - No single kernel-level vision/audio/generation capability detector; detection stays per-command.  (rec#419)
- [ ] Reconcile/remove the legacy plugin loader CLI vs the plugin engine - The legacy loader (entry=/[tools], plugin list/remove) still coexists with the engine (register=/[surface], install/uninstall); no reconcile/removal happened.  (rec#493,629)
- [ ] Address remaining low tech-debt items L1-L7 (P) - 3 of L1-L7 fixed; 4 still open (L2 REPL KeyError, L5 ROCm index drift, L6 silent corrupt-vectors fallback, L7 non-atomic model move+registry).  (rec#499)

### Research (7)
- [ ] Run the research pass over the 11 saved articles/links (R19) (P) - Two named threads produced wired code (AutoJack defense, '7 types of memory'), but the full 11-link research deliverable/notes are not present.  (rec#19,271)
- [ ] Decide whether the HF-vs-GGUF / Vulkan-vs-CUDA backend-selection path is coherent - Path-based dispatch and a vulkan-first policy exist, but no recorded decision answers the maintainer's 'why pick Vulkan if CUDA runs both' question.  (rec#77)
- [ ] Verify the Open WebUI load mechanism before building the adapter - Still a to-verify item: the exact OWUI load mechanism (DB exec vs file import) and which open_webui internals are imported is not recorded.  (rec#324)
- [ ] Audit which ecosystem extensions stay in the reachable interop set - The doc gives only a qualitative reachable/out-of-reach table; the sample audit of top community OWUI/oobabooga extensions is undone.  (rec#325)
- [ ] Decide canonical /v1 owner pinning via a default_instance_dir config key - No default_instance_dir key exists; the floated pin-the-8642-owner determinism idea was never implemented.  (rec#421)
- [ ] Investigate resilience to third-party (ComfyUI/StabilityMatrix) breaking changes - No drift-resilience deliverable; only targeted ComfyUI node-crash error surfacing exists.  (rec#490)
- [ ] Evaluate Piper-WASM in-browser TTS as an upgraded read-aloud voice (P) - The privacy-preserving neural read-aloud goal ships via Kokoro, but the specific Piper-WASM candidate was not integrated (Kokoro chosen instead).  (rec#704)

### Docs (5)
- [ ] Run a full transitive pip-licenses scan after the AGPL relicense (P) - A pip-licenses scan is documented at closure level but no committed full-transitive scan output exists.  (rec#147)
- [ ] Document the regulatory roadmap and budget alongside the product roadmap - No regulatory roadmap or budget document exists alongside the product roadmap.  (rec#149)
- [ ] Embed a screenshot in the README and finish go-public polish - No screenshot embedded in the README and chat_and_queue.png is not present in the repo.  (rec#224)
- [ ] Reconcile the coder-route-session-files documented shape vs implementation (N2) - The endpoint returns {path,writes,created,exists,last_tool}; the stale matrix doc lives only in gitignored qa/, so doc-fix status is unknowable here.  (rec#362)
- [ ] User-facing README/docs rewrite (shorten/restyle in the maintainer's voice) (H-4) - The README is still long and there is no evidence the shorten/restyle rewrite deliverable landed.  (rec#494)
