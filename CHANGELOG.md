# Changelog

All notable changes to localm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and localm aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Being pre-1.0,
minor versions may include breaking changes.

Each release adds its section on top. Published (versioned) sections are the
permanent public record of what shipped and are never rewritten; the in-progress
`[Unreleased]` section is maintained until it is cut into a release.

## [Unreleased]

### Added
- **Automatic server-hang diagnostics:** an event-loop stall watchdog runs by
  default and, if the server ever freezes, dumps every thread's stack to
  `<home>/logs/hang_*.log` (the file is created only when a real stall happens, so
  a healthy run leaves nothing behind). A captured trace is bundled into a bug
  report automatically, so an intermittent "it just hung" becomes diagnosable with
  no setup on the reporter's part. It respects privacy: in privacy mode (the
  default) it writes nothing automatically, unless you turn on "Keep diagnostics
  for bug reports" (below). `LOCALM_HANG_WATCHDOG=0` turns it off entirely (and `=1`
  forces it on with verbose logging even in privacy mode); a loopback-only
  `GET /debug/stacks` returns thread and task state on demand.
- **"Keep diagnostics for bug reports" privacy setting:** privacy mode saves
  nothing automatically, which also means a hang or crash leaves nothing to
  report. This new toggle (off by default) keeps the diagnostic bits a report
  needs - the hang stack trace, the restart breadcrumb log, and a debug log -
  even in privacy mode. It is available in Settings > Privacy (in-app) and as a
  checkbox in the desktop launcher (and `--keep-diagnostics` on the command line).
  It keeps operational diagnostics only - see Security below for the chat-content
  guarantee this is held to.
- **Clearer bug-report send failures:** when a bug report cannot be filed, the app
  now tells you WHERE it failed - you appear offline, the server is unreachable, a
  secure-connection problem, or the server rejected it - instead of a raw error. It
  always keeps your report and offers to retry, or to download the report file so
  you can send it by email/Discord yourself (works from a phone or another device
  where a server-side path is useless). Both the WebUI and `localm bug-report`.

### Fixed
- **`localm doctor` no longer cries "CPU mode only" when your GPU is in use:**
  doctor decided GPU capability from `nvidia-smi` / `rocm-smi` / torch alone, none
  of which see localm's default GPU paths - Vulkan (Intel/NVIDIA/AMD via the
  display driver), Metal on Apple Silicon, or the bundled AMD-on-Windows ROCm
  build - so it wrongly reported "CPU mode only" on the majority of non-CUDA-toolkit
  GPU setups while inference was actually running on the GPU. It now reports the
  GPU from what localm will actually load: the real device the provisioned
  llama.cpp runtime registers (e.g. "GPU: ROCm0 - used for inference"), falling
  back to the provisioned backend name; the smi/torch lines remain as extra detail.
- **Server freezing while the system is idle:** reading GPU/VRAM state (opening
  Settings > Performance, the Models page, or the periodic cross-instance GPU
  heartbeat) ran a synchronous GPU driver query directly on the server's single
  event loop; if the driver was momentarily busy or wedged, the whole web UI froze
  even though the machine was idle. GPU probes are now time-bounded and run off the
  event loop, so a slow or stuck driver can no longer stall the server.
- **A cancelled reply no longer blocks the next request:** if a client
  disconnected in the middle of a response (closed the tab, pressed stop, or
  dropped the connection), the model kept generating the abandoned reply all the
  way to the end while still holding that model's single-inference lock, so the
  very next request to the same model stalled until the discarded generation ran
  itself out. A disconnect now stops the abandoned generation promptly and
  releases the model, so the next request starts right away - for both streaming
  and plain (non-streaming) requests.
- **Multi-GPU split fit checks:** a model too large for the single main GPU
  alone, but that fits combined across a configured multi-GPU split, was
  wrongly refused ("Not enough VRAM") when loading, and mis-badged "too big"
  in model search - the pre-load check, the search/CLI/MCP fit badges, the
  Settings performance-slider estimate, the scheduled-job/media-generation
  VRAM swap decisions, model-unload VRAM-release detection, and the GUI's
  hardware-monitor VRAM readout only ever weighed a load against one GPU's
  capacity, never the split's combined capacity. They now correctly sum
  capacity across the configured split.
- **Multi-GPU split vs. multiple loaded models:** a GGUF model load (chat
  model or the embedding model) could pass the pre-load VRAM check (enough
  COMBINED free VRAM across a configured split) and still fail or crash on
  a single GPU, if another already-loaded model left that specific device
  with less free room than its configured share of the new model needed -
  the aggregate check alone cannot catch an uneven split. Loading now also
  checks each configured GPU's own share before handing off to the native
  loader, evicting further (chat models) or refusing clearly instead of
  risking a native crash. That VRAM check also now runs off the server's
  event loop (like the idle-freeze fix above), so a slow GPU driver during
  a model load can no longer stall other requests.
- **GGUF context/KV-cache VRAM checks:** loading a GGUF model with a large
  context window ignored the KV cache entirely when judging whether it would
  fit, only weighing model weights - so a model whose weights fit could still
  ask the driver to reserve a KV cache far bigger than VRAM, which on some
  drivers either silently spilled into slow system memory or crashed the GPU
  driver with nothing shown to the user (it looked like the model "loaded"
  then went silent on the first prompt). The preflight now accounts for the
  KV cache at the requested context size and refuses clearly, before the
  native load, when it cannot fit; conversation growth (which can double the
  context on literally the first prompt, since the default reply budget
  already exceeds the default starting context) is now checked the same way,
  so a request that would overflow VRAM fails with a clear message instead of
  silently returning nothing.
- **Phone pairing with a VPN active:** a VPN client's virtual tunnel adapter can
  become the machine's default route and hand out an ordinary private address,
  which used to get picked as the "LAN" address for the Companion pairing card
  and the `<name>.local` mDNS advertisement - pointing a phone at an address
  only reachable through the VPN. VPN-like adapters are now excluded, so the
  real LAN address is shown (and advertised) instead.
- **Setup:** a silent fallback during AMD ROCm asset lookup (when the upstream
  release-asset check fails) now prints a warning instead of happening
  invisibly.
- **Log export reports real failures instead of "no logs found":** exporting logs
  to a folder that could not be written (full disk, read-only, or a permissions
  problem) used to report success with "No log files were found to export", hiding
  the write failure. It now fails with the actual reason when nothing could be
  copied, and a partial export lists the files it could not copy.
- **`localm setup-embeddings` no longer overstates what changed:** the completion
  message claimed memory and RAG would now use semantic search, but existing RAG
  collections stay lexical until re-indexed with embeddings. It now says memory
  retrieves semantically right away and names the re-index step
  (`localm rag add <name> <path> --embed`, queried with `--embed`) that RAG
  collections need.
- **`localm run` stops contradicting its own auto-start:** after it tried to start
  a background server that did not come up in time, it used to say "no server is
  serving this directory; start one with `localm serve`". It now acknowledges the
  timed-out start and loads the model in this process instead of advising exactly
  what it just attempted.

### Security
- **Privacy mode never logs chat content, even with diagnostics on.** Two
  code paths (the GGUF backend's raw model output, and the coder's web-tool
  call logging) could write your actual chat content - prompts and replies -
  to the debug log while in privacy mode, if `--debug` or the new "keep
  diagnostics" setting above was active. Both now stay off in privacy mode
  regardless of diagnostic settings; operational details (timings, request
  shapes) are unaffected.

## [0.1.1] - 2026-07-09

### Added
- **Unified model browser (phase 1):** registry model types, a ComfyUI model
  scan, and a type-filtered model list.
- **Search HuggingFace for HF (transformers) models, not just GGUF.** The Models
  page search has GGUF / HF format toggles; HF results show a total size and a
  VRAM fit badge estimated from the model's parameter count (or "size unknown"
  when the metadata is absent), and pull the whole repo, with a non-blocking hint
  when no transformers runtime is installed (the files still download). Both
  formats are interleaved so one never crowds the other out of the results.
- **Model management:** deterministic model type-detection with an explicit
  `unknown` sentinel and a `.safetensors` directory scan; a `--store` option when
  adding a model; a main-GPU selector with multi-instance GPU coordination and
  explicit model unload.
- **Split a model across multiple GPUs.** A model too large for any single
  card's VRAM can now load using the combined VRAM of 2 or more GPUs
  (`localm config gpu_split_indices 0,1`, or the "Split across GPUs"
  checkboxes next to the Main GPU selector in Settings), alongside the
  existing single-GPU selector. The model search results also hint when a
  model would fit split across your GPUs but not on the largest one alone.
- **Media: localm can manage its own ComfyUI (opt-in).** For image, music, and
  video generation, localm can install and run an isolated, hardware-matched
  ComfyUI kept separate from any existing one, via `localm comfy setup`
  (copy-path or a fresh install) and GUI setup/status/remove on the media pages;
  `localm doctor` hints at it. It coexists with a user's own ComfyUI rather than
  replacing it, and an in-memory shim works around an upstream ComfyUI crash.
- **RAG indexes more:** arbitrary text files by content sniffing, zip/tar archive
  extraction, and image description via a vision model. Each chunk is tagged with
  its document format (json, yaml, python, ...), heuristic-first, with the AI
  classifier only a tie-break for an unclear extension when a chat model is loaded.
- **MCP server exposes more tools:** setup, model removal, diagnostics, and plugin
  management, with annotations so clients can confirm destructive calls.
- **View the changelog in the app.** A "Show changelog" button in Settings >
  System renders the full release history in-app (backed by a read-only
  `/api/changelog`), so what changed is visible without leaving localm.
- **Signed, out-of-the-box self-update.** `localm update` works with no
  per-install setup: builds are assembled from a declared file manifest and
  signed, and the client verifies the signature before applying (see Security).
- **Recover from a bad update.** `localm update --rollback` restores the previous
  build from the last update's backup; and a standalone `rollback.bat` /
  `rollback.sh` in the install folder does the same WITHOUT needing localm to
  start, so an update that breaks the app can always be undone to a working one.
- **Report a problem even when localm will not start.** A standalone,
  account-less reporter (the `report-issue` entry) files a bug report through the
  localm proxy with no working install and no GitHub login, previewing exactly
  what will be sent first.
- **Setup dead-ends removed:** bootstrap `uv` automatically when it is missing,
  and prompt to replace already-provisioned native binaries instead of exiting
  (non-interactive runs keep the existing binaries).

### Changed
- **Media:** managed-ComfyUI status is checked only when it matters, not every 5s.
- **Media:** legacy in-package personal workflow overrides are migrated to the
  data directory on startup, so a self-update cannot wipe them.
- **RAG:** an embedding-only index no longer stalls or burns the request timeout,
  because format tagging is heuristic-first and the AI classifier runs only as a
  tie-break with a chat model loaded.
- **Inference:** repeated native-stderr lines are de-duplicated during generation,
  and native stderr redirection is tightened.
- **Dependencies:** huggingface-hub 1.22, transformers 5.x, fastapi 0.139,
  pillow 12.3, plus dev/tooling bumps; Dependabot now tracks the native `uv`
  ecosystem.

### Fixed
- **Models and API:** a local model file registers fully offline with no
  HuggingFace path leak; `/api/models` no longer 500s on a forward reference; a
  hidden native-load failure cause is surfaced.
- **HF snapshot pulls:** a full-repo HuggingFace pull now preflight-checks free
  disk space (like the GGUF/URL pull paths already did) instead of running
  until the OS hits ENOSPC mid-transfer; and "already downloaded" now compares
  every file the repo lists against what's actually on disk, not just
  `config.json`'s presence, so a disk-full mid-download no longer gets
  silently registered as a complete, ready model on retry.
- **RAG safety:** archive extraction is bounded (zip/tar bombs), compressed tars
  are handled, and a folder index skips model weights and secrets and does not
  index member-read errors.
- **RAG API under `localm serve` (api-mode):** indexing, upload, and embedding
  setup used to crash with an opaque HTTP 500 when hit directly (not through the
  GUI) because they read GUI-only server state; they now degrade to a clean 503
  ("run `localm gui`"), and querying falls back to lexical-only search instead of
  crashing. Attachment extraction (`/api/rag/extract`) no longer runs
  synchronously on the event loop, so a large or crafted archive attachment can
  no longer freeze every other request on the server while it extracts.
- **RAG on Windows:** a transient `PermissionError` during a collection write's
  atomic rename (antivirus or the search indexer briefly holding the file handle)
  is now retried instead of failing the write.
- **VRAM and multi-model handling:** safe multi-model VRAM eviction; idle-unload
  keeps the engine for a lazy reload; concurrent loads of different models no
  longer preempt each other; the active model is marked in `/v1/models` and
  attached to instead of the first entry; `/v1/embeddings` no longer force-loads
  the chat model; a stale VRAM estimate and a cross-instance conversation leak are
  fixed.
- **Do not hide problems:** removed production code paths that detected
  pytest/mocks and fabricated behavior; a swallowed VRAM-gate failure is now logged.
- **Inference:** a zero-n_tokens decode failure, batch memory-safety and
  token-position bugs, and a tool-call grammar that let small models loop; the
  lenient tool-call JSON parser no longer mangles backslashes (Windows paths).
- **Chat:** `localm run`'s default attach mode (the common case) no longer
  silently drops a thinking model's reasoning - it is now dimmed in the terminal
  like the in-process path already did.
- **Chat:** trimmed the injected web-access prompts so weak models stop fixating on them.
- **Coder:** `coder_confirm_timeout=0` now means wait forever, as documented.
- **Coder episodic memory:** a concurrent read racing the atomic write (a GUI poll
  landing mid session-close reflection) could hit a transient Windows
  `PermissionError`; both sides now retry briefly instead of raising.
- **Coder confirmations:** a coder session tracked only one pending confirmation
  at a time, so two tool calls needing approval in the same turn (e.g. two
  `fetch_url`/`web_search` calls under `net_mode=ask`, which the agent runs
  concurrently) would have the second clobber the first - the first could never
  be answered and just sat until the 10-minute timeout auto-rejected it. Pending
  confirmations are now tracked per call, so concurrent approvals no longer
  collide.
- **Coder reasoning:** the coder's HTTP backend (the common case - `localm serve`,
  OpenAI, Anthropic, or any OpenAI-compatible endpoint) never read a thinking
  model's `reasoning_content`, so its reasoning was silently dropped - no
  "thinking" display in the terminal or GUI, unlike the CLI attach-mode fix
  above. It now streams separately (a dimmed terminal aside, its own GUI event
  and collapsible block, and its own audit-log field) rather than being mixed
  into the visible answer, which the coder loop resends to the model and stores
  in history verbatim with no splitter of its own.
- **Memory:** the owner's chat memory stays in the shared `owner` namespace; a
  missing per-namespace write lock let concurrent writers (two requests, or a
  background consolidation pass racing a live edit) silently drop each other's
  facts, now fixed with a lock mirroring RAG's existing one.
- **GUI:** an empty ComfyUI scan shows the reason instead of a bare "Added 0"; the
  GUI no longer sends a chat request when no model is loaded; a background job's
  progress stream (model pull, ComfyUI setup, image/music/video generation) now
  fans out to every viewer independently, so reloading the page or opening the
  same job in two tabs no longer splits its events between them (one tab could
  end up hanging forever with no completion event).
- **CLI:** `localm.bat` argument forwarding is fixed.
- **CLI:** a `localm serve` start path that skipped the bind-security gates now
  enforces them like every other start path.
- **CLI:** an `UnboundLocalError` in the chat runner is fixed.
- **Updater:** the "runtime" update class's post-swap command used a bare `localm`
  argv that resolved back to the native launcher exe itself on the default install
  and rolled back the whole update; it now re-invokes through the current
  interpreter, like every other self-invocation site.
- **Bug reporter:** removed the only path that could ask for a GitHub login; the
  non-interactive path now names the account-less send channel.
- **Setup:** warn when a llama.cpp download has no checksum to verify (and verify
  against a published checksum by default); discard stray keyboard input before the
  CUDA prompt; skip console-window hiding in debug mode; skip draft releases with
  no uploaded asset.

### Security
- **Open-mode metadata reads are now origin-bound too.** The default CORS policy
  trusts every `localhost`/`127.0.0.1` origin to read a matching response, so
  another local program could steal the loopback GUI shell's open-mode
  management token from a plain cross-origin `GET /` and replay it against
  metadata routes (named keys, server config, host stats, the filesystem
  browser) to read real local data with no credentials of its own. That token
  is now gated on the same same-origin/allowlist check state-changing routes
  already enforced; state changes themselves were never affected.
- **Request-body size cap now covers chunked uploads.** The 160MB body cap
  only checked the client-supplied `Content-Length` header, so a
  `Transfer-Encoding: chunked` request (which sends no `Content-Length` at
  all) bypassed it entirely - reachable pre-auth on the CORS-exempt inference
  routes, and capable of buffering a multi-gigabyte body into memory from one
  connection. The cap is now enforced on the actual byte stream.
- **Plugin/tool calls (rag, web, voice, coder sessions, GUI model routes) can no
  longer starve chat completions.** Every blocking offload in the server -
  inference (model load/unload, chat/completion generation) and plugin tool
  calls alike - drew from the SAME process-wide thread pool
  (`min(32, cpu_count+4)` workers). A caller holding only a narrow plugin
  scope (or any loopback caller under open/no-key mode) could pipeline enough
  slow tool calls (e.g. archive extraction, which can legitimately run
  8-30s+ per file) to occupy every worker thread in that pool, which starved
  the SAME pool's inference slot and stalled chat replies for every user of
  the server, including the admin. Plugin/tool work now runs on its own
  dedicated, equally-sized pool (`localm/plugins/executor.py`), completely
  isolated from the pool inference relies on.
- **Privacy mode now deletes ComfyUI's own on-disk output copy everywhere media
  is generated, not just from the GUI/API.** ComfyUI keeps its own copy of every
  generated image/track/clip with the full prompt and workflow embedded as
  metadata; privacy mode is supposed to remove it after use. That fold-in was
  only wired up on the GUI/API path - the `localm image`/`music`/`video` CLI
  commands, the `/generate-image`/`/generate-music`/`/generate-video` REPL
  commands, the MCP server's `generate_image` tool, and the coder agent's
  `generate_image` tool all suppressed the prompt sidecar in privacy mode but
  left ComfyUI's own copy (and any img2img source image) on disk indefinitely.
  All six call sites now fold privacy mode into `delete_outputs`, matching the
  GUI/API behaviour.
- **`?pull=` deep link no longer downloads a model with zero confirmation.**
  `localm gui --pull SPEC` opens the browser at `?pull=SPEC`, but any page (or a
  hidden iframe on any site, while localm runs locally) could forge the same
  link and silently start a real download. The CLI now mints a single-use,
  spec-bound token passed alongside the link, so ITS OWN deep link still
  auto-starts with zero clicks; a link without a valid token falls back to an
  explicit confirmation dialog instead of firing automatically.
- **Media gallery and share-inbox ownership fixed:** the image/music/video
  generated-media routes (serve/delete/move/rename/history) and the PWA
  share-inbox routes had no per-key ownership check, so any key holding the
  plugin's own (non-privileged) scope could enumerate, read, delete, move, or
  rename another principal's generated media or shared files. Both now stamp
  and check ownership the same way the jobs API already did.
- **Job API privilege escalation fixed:** correct principal-ID hashing for
  admin/owner keys, so owner-created jobs are no longer reachable by
  loopback-anonymous roles.
- **Coder `spawn_agent` no longer bypasses confirmation.** A child agent spawned
  via `spawn_agent` now inherits the parent session's `auto_approve`, `dry_run`,
  `always_confirm`, and `confirm_handler` instead of always auto-approving, so a
  parent that requires confirmation (or is running `--dry-run`) can no longer be
  routed around by having the model delegate destructive work to a sub-agent.
- **A scheduled coder job's shell access no longer outlives its creating key.**
  The autonomous job scheduler now re-validates the owning key is still live
  (not revoked, not expired) before running an `allow_shell` job, instead of
  trusting the stored opt-in forever; a dead key downgrades the run to the
  safe restricted coder rather than keeping shell access.
- **Coder privacy-mode history scrub** now warns when a shell history file
  cannot be *read* for scrubbing, matching the existing warning when it cannot
  be *written* - an unreadable file is no longer silently treated as clean.
- **Authenticated self-update.** Each release build is signed with an offline
  Ed25519 key and verified against a pinned public key before it is extracted or
  executed, so a compromised release channel cannot push a forged build.
  Anti-rollback refuses an older signed build, the download stays HTTPS-pinned, and
  a self-update no longer wipes the provisioned native runtime. Publishing is gated
  on a clean tree, a full CI pass over the repo, and a build that imports and runs;
  the build itself is now assembled from that exact CI-validated commit (not a
  live disk snapshot taken after CI finishes), so a change landing on disk during
  the wait can no longer ship unreviewed.
- **ComfyUI launch** no longer has a shell-injection vector on Windows.
- **RAG folder index** skips model weights and secrets rather than indexing them.
- **MCP** destructive tools are annotated so clients can confirm them.

## [0.1.0] - 2026-07-04

First tagged release. A self-contained, offline local-LLM platform.

### Added
- **Inference**: run GGUF models via a bundled llama.cpp runtime, and HuggingFace
  models via a torch backend. GPU acceleration on AMD (ROCm), NVIDIA (CUDA or
  Vulkan), and Intel/Vulkan, with automatic hardware detection and a CPU fallback.
  Auto context-window sizing from free VRAM.
- **CLI**: `pull`/`search`/`add`/`list`/`rm` for models, `run` (interactive or
  single-prompt, with image input), `serve` (OpenAI-compatible API), `gui`,
  `doctor`, `benchmark`, `config`, `info`, and shell completion.
- **GUI**: an installable PWA served by the local server - chat with conversation
  history, assistant memory, and personas; the coder agent; model management; and
  per-plugin pages. Loopback by default; an optional network bind for phone access.
- **Coder agent**: an offline AI coding agent with file, shell, search, and test
  tools, a project map, an agentic loop, and MCP interop, gated behind an explicit
  scope.
- **Plugins** (everything but chat is a plugin): image / music / video generation
  (ComfyUI backends), a knowledge base for retrieval-augmented chat (RAG), a job
  scheduler, an MCP server, text-to-speech, speech-to-text, and web search/fetch.
- **Security**: loopback-only by default; a network bind forces a strong API key
  and enables built-in TLS. Scoped API keys, an owner/admin model, HttpOnly session
  cookies with CSRF protection, path confinement, and an SSRF-guarded outbound
  fetch policy.
- **Privacy**: an opt-in privacy mode that keeps sessions in memory and writes no
  traces to disk; a configurable network policy (off / ask / allow, allow/deny
  lists).
- **Networking**: reach a network-bound instance by name over mDNS (`localm.local`)
  and Tailscale MagicDNS, folded into the TLS certificate.
- **Model tools**: HuggingFace search and local model registration.

### Known limitations
- Media generation (image / music / video) requires a local ComfyUI backend; RAG's
  semantic mode requires the on-device embedding model (`localm setup-embeddings`).
- The in-app bug-report upload and the self-updater require the maintainer's
  Cloudflare Worker to be deployed; until then those two features are inert.
- The NVIDIA GPU path is validated by design and CI-adjacent testing; the primary
  development hardware is AMD.

[Unreleased]: https://github.com/Matlan1/localm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Matlan1/localm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
