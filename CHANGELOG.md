# Changelog

All notable changes to localm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and localm aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Being pre-1.0,
minor versions may include breaking changes.

## [Unreleased]

### Added
- **RAG**: Support dynamic file type sniffing, zip/tar archives, and image OCR fallback (NEEDS A FULL VERIFICATION).
- **Model Browser**: Implement Phase 1 of Unified Model Browser.
- **MCP**: Expose setup, model removal, diagnostics, and plugin tools to the MCP server.
- **Setup**: Prompt the user to confirm replacement of already provisioned native llama.cpp binaries instead of bailing out with instructions to use `--force`. Guarded with a `sys.stdin.isatty()` check to ensure non-interactive script setups and test suites (e.g. `pytest`) bypass the prompt and default to safe bails.

### Fixed
- **Inference**: Manually set n_tokens on initialized batch to avoid zero n_tokens decode failure.
- **Setup**: Add default checksum verification for setup-llama downloads.
- **Setup**: Suppress unconfigured data directory warning during setup phase.
- **Inference**: Fix lenient JSON parser unescaped backslashes.
- **Tests**: Resolved a test suite signature mismatch in `test_model_manager_phase3.py` by adding `**kw` keyword argument support to GGUF and snapshot download mocks, preventing crashes when pulling models.
- **Lint**: Cleared 19 Ruff lint errors (unused imports and undefined `Optional` names) across the repository to restore CI check pipeline sanity.

## [0.1.1] - 2026-07-04

### Fixed
- **Inference**: Fixed a threading race/concurrency crash in lazy grammar sampling by ensuring mock Llama structures consistently initialize the underlying `_inference_lock`.
- **Security**: Hardened background job API routes by restoring correct principal ID hashing for admin/owner keys, preventing a privilege escalation regression where owner-created jobs became accessible/owned by loopback anonymous roles.
- **CLI**: Resolved a server start delegation issue in `localm serve` where fail-fast binding security gates were skipped due to late port-bind warning evaluations.
- **CLI**: Fixed a potential `UnboundLocalError` for `sys` in the chat runner module.
- **CLI**: Avoid ComfyUI launch shell injection vectors on Windows platform.
- **Setup**: Discarded stray buffered keyboard inputs prior to interactive CUDA configuration prompts to prevent accidental selections.
- **Setup**: Skip console window hiding behavior in debug mode on Windows.
- **Setup**: Fixed release tag asset resolution logic to skip draft releases with un-uploaded binary archives.
- **Plugins**: Ensure builtin `chat` dependency is pre-installed/self-healed prior to running CLI plugin dependency checks.
- **Tests**: Isolated all suite environments using a temp directory `LOCALM_HOME` in `conftest.py` to prevent local configuration state leakages from polluting test executions.

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

[0.1.1]: https://github.com/Matlan1/localm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matlan1/localm/releases/tag/v0.1.0
