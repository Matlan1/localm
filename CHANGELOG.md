# Changelog

All notable changes to localm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and localm aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Being pre-1.0,
minor versions may include breaking changes.

Each release adds its section on top. Published (versioned) sections are the
permanent public record of what shipped and are never rewritten; the in-progress
`[Unreleased]` section is maintained until it is cut into a release.

## [Unreleased]

Everything since 0.1.0. (0.1.0 and a same-day 0.1.1 micro-tag were both cut on
2026-07-04; that tag was never distributed, so its fixes are folded in here.)

### Added
- **Media: localm can manage its own ComfyUI (opt-in).** For image, music, and
  video generation, localm can install and run an isolated, hardware-matched
  ComfyUI kept separate from any existing one, via `localm comfy setup`
  (copy-path or a fresh install) and GUI setup/status/remove on the media pages;
  `localm doctor` hints at it. It coexists with a user's own ComfyUI rather than
  replacing it, and an in-memory shim works around an upstream ComfyUI crash.
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
- **Unified model browser (phase 1):** registry model types, a ComfyUI model
  scan, and a type-filtered model list.
- **Search HuggingFace for HF (transformers) models, not just GGUF.** The Models
  page search has GGUF / HF format toggles; HF results show a total size and a
  VRAM fit badge estimated from the model's parameter count (or "size unknown"
  when the metadata is absent), and pull the whole repo, with a non-blocking hint
  when no transformers runtime is installed (the files still download). Both
  formats are interleaved so one never crowds the other out of the results.
- **View the changelog in the app.** A "Show changelog" button in Settings >
  System renders the full release history in-app (backed by a read-only
  `/api/changelog`), so what changed is visible without leaving localm.
- **RAG indexes more:** arbitrary text files by content sniffing, zip/tar archive
  extraction, and image description via a vision model. Each chunk is tagged with
  its document format (json, yaml, python, ...), heuristic-first, with the AI
  classifier only a tie-break for an unclear extension when a chat model is loaded.
- **MCP server exposes more tools:** setup, model removal, diagnostics, and plugin
  management, with annotations so clients can confirm destructive calls.
- **Model management:** deterministic model type-detection with an explicit
  `unknown` sentinel and a `.safetensors` directory scan; a `--store` option when
  adding a model; a main-GPU selector with multi-instance GPU coordination and
  explicit model unload.
- **Setup dead-ends removed:** bootstrap `uv` automatically when it is missing,
  and prompt to replace already-provisioned native binaries instead of exiting
  (non-interactive runs keep the existing binaries).

### Changed
- **Media:** managed-ComfyUI status is checked only when it matters, not every 5s.
- **Inference:** repeated native-stderr lines are de-duplicated during generation,
  and native stderr redirection is tightened.
- **RAG:** an embedding-only index no longer stalls or burns the request timeout,
  because format tagging is heuristic-first and the AI classifier runs only as a
  tie-break with a chat model loaded.
- **Media:** legacy in-package personal workflow overrides are migrated to the
  data directory on startup, so a self-update cannot wipe them.
- **Dependencies:** huggingface-hub 1.22, transformers 5.x, fastapi 0.139,
  pillow 12.3, plus dev/tooling bumps; Dependabot now tracks the native `uv`
  ecosystem.

### Fixed
- **VRAM and multi-model handling:** safe multi-model VRAM eviction; idle-unload
  keeps the engine for a lazy reload; concurrent loads of different models no
  longer preempt each other; the active model is marked in `/v1/models` and
  attached to instead of the first entry; `/v1/embeddings` no longer force-loads
  the chat model; a stale VRAM estimate and a cross-instance conversation leak are
  fixed.
- **Inference:** a zero-n_tokens decode failure, batch memory-safety and
  token-position bugs, and a tool-call grammar that let small models loop; the
  lenient tool-call JSON parser no longer mangles backslashes (Windows paths).
- **RAG safety:** archive extraction is bounded (zip/tar bombs), compressed tars
  are handled, and a folder index skips model weights and secrets and does not
  index member-read errors.
- **Models and API:** a local model file registers fully offline with no
  HuggingFace path leak; `/api/models` no longer 500s on a forward reference; a
  hidden native-load failure cause is surfaced.
- **Do not hide problems:** removed production code paths that detected
  pytest/mocks and fabricated behavior; a swallowed VRAM-gate failure is now logged.
- **Bug reporter:** removed the only path that could ask for a GitHub login; the
  non-interactive path now names the account-less send channel.
- **Memory:** the owner's chat memory stays in the shared `owner` namespace.
- **GUI:** an empty ComfyUI scan shows the reason instead of a bare "Added 0"; the
  GUI no longer sends a chat request when no model is loaded.
- **Setup:** warn when a llama.cpp download has no checksum to verify (and verify
  against a published checksum by default); discard stray keyboard input before the
  CUDA prompt; skip console-window hiding in debug mode; skip draft releases with
  no uploaded asset.
- **CLI:** `localm.bat` argument forwarding; a `localm serve` start path that
  skipped the bind-security gates; an `UnboundLocalError` in the chat runner; and
  `coder_confirm_timeout=0` now means wait forever.
- **Chat:** trimmed the injected web-access prompts so weak models stop fixating on them.

### Security
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
- **Authenticated self-update.** Each release build is signed with an offline
  Ed25519 key and verified against a pinned public key before it is extracted or
  executed, so a compromised release channel cannot push a forged build.
  Anti-rollback refuses an older signed build, the download stays HTTPS-pinned, and
  a self-update no longer wipes the provisioned native runtime. Publishing is gated
  on a clean tree, a full CI pass over the repo, and a build that imports and runs.
- **Job API privilege escalation fixed:** correct principal-ID hashing for
  admin/owner keys, so owner-created jobs are no longer reachable by
  loopback-anonymous roles.
- **ComfyUI launch** no longer has a shell-injection vector on Windows.
- **RAG folder index** skips model weights and secrets rather than indexing them.
- **MCP** destructive tools are annotated so clients can confirm them.

### Internal
- **Release pipeline.** A release-file manifest and verification gate, a build.zip
  assembler and signer, a runtime-completeness smoke gate, a pre-publish CI gate,
  and a live functional-verification gate (cold-install and exercise every
  changelog item across both inference backends before publish). The publish path
  also refuses to reuse an existing version tag or to build from a commit that is
  not the CI-tested `origin/master`, so the signed artifact always matches what CI
  validated.
- **Changelog is a guarded record.** `CHANGELOG.md` is enforced append-only for
  published (versioned) sections: `check_hygiene.py` fails the build if a shipped
  entry is deleted or rewritten. The `[Unreleased]` draft stays freely editable.
- Contributor-guide and test-cadence clarifications, test isolation via a temp
  `LOCALM_HOME` in `conftest.py`, and a documentation pass across the user manual.

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

[Unreleased]: https://github.com/Matlan1/localm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
