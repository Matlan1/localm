# localm feature delivery audit (2026-06)

Method: six parallel code-grounded audits (inference core + server, coding agent,
GUI, generators, model-management + RAG, peripheral systems). Every grade is from
reading the implementation, not the README or docstrings. The three highest-stakes
findings were re-verified by hand. "Tested" means a test exercises it; it does not
mean "works against real hardware" unless stated.

Grades: Solid | Works-with-caveats | Partial | Fragile | Stub/Broken | Missing.

## Top-line verdict

localm is a real, unusually well-engineered local-LLM platform, not a demo. The
server protocol, the coder loop and its file/patch/git tooling, the GUI, model
dedup, the network policy, and the privacy contract are genuinely solid and, in
several cases, better than comparable hobby tools. The gap between the README and
the metal is concentrated in a handful of specific over-claims and in stateful
paths that have no tests. Nothing is fraudulent; a few things are sold as done
that are not.

## The credibility gaps (README says X, code does Y) - ranked

1. GGUF multimodal is a silent no-op. `--mmproj` / `--image` and the
   "Multimodal | Image attachment (requires mmproj GGUF)" row are documented, the
   flags are accepted and stored (`gguf.py:42`), then never used. `llama.py` has
   zero image handling (verified: 0 occurrences). The image part is dropped and
   the model answers about an image it never saw, with no error. Only HF-format
   vision models actually see images, and they do not use `--mmproj`. This is the
   worst kind of gap: a headline feature that fails silently and plausibly.
2. Plugin "agent tools" do not exist. README: "export tools into the coder
   agent." `tool_exports` is parsed and shown in `plugin list` / the GUI, but
   nothing in `localm/plugins/coder/` ever imports or registers it (verified). The
   CLI-command half of the plugin system works and is tested; the agent-tool half
   is vaporware.
3. "Built-in generation" hides a large external dependency. Image/music/video
   all require a separate ComfyUI install on :8188 plus roughly 25 GB of FLUX /
   ACE-Step / Wan models hand-downloaded into ComfyUI's folders. localm owns a full
   HF/URL download stack and a `setup-llama` provisioner but wires none of it to
   ComfyUI: the only "help" is a sentence in an error string. "Through a local
   ComfyUI" is a disclosure, but reads as a detail rather than "install and feed a
   second heavyweight app first."
4. RAG embeddings are real in code but unreachable for the default user. "Embeddings
   blended in when the backend supports them" is true, but the GGUF binding has no
   `create_embedding` (verified), so `/v1/embeddings` 422s and indexing/query
   always fall back to BM25 for the GGUF-first user the project targets. Hybrid
   only lights up with an HF model loaded, and the vector path has never been
   tested against a real backend (only a fake embedder).
5. TLS is doc-only and the GUI can bind to the LAN unguarded. There is no TLS code
   anywhere (verified: no ssl_keyfile/certfile); `docs/tls.md` correctly says "use a
   reverse proxy," but the README feature row says "TLS" as if it were a capability.
   Worse, `_exposed_bind_warning` is wired into `serve` but not `gui` (`gui/cli.py`),
   so the shell-executing GUI can be served on 0.0.0.0 with no warning and no TLS.
6. The "lean GGUF, no torch" pitch is undercut by torch-gated features. ctx_auto
   VRAM sizing, the VRAM pre-flight warning, and GGUF embeddings all silently no-op
   or degrade (ctx_auto becomes a hardcoded 16384) without torch, which is exactly
   the install the README promotes.
7. Slash-command and docs drift. `/music` and `/video` exist only in the GUI
   composer; the terminal REPL implements only `/imagine`, and there is no
   `localm imagine` despite docs implying CLI parity. The FLUX negative-prompt tool
   description and `docs/flux-setup.md` say "via ConditioningConcat," which is the
   opposite of what the code does (it correctly uses a CFGGuider and warns against
   concat). `docs/flux-setup.md` lists a T5 encoder filename the shipped workflow
   does not reference (guaranteed first-run 400). `docs/llamacpp-binding.md` still
   lists the old hardcoded `D:\projects\...` DLL paths.

## Corrected false alarm

Auth is fine. `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` all
carry `Depends(_require_auth)` (verified: lines 332/371/400). Only `/health` and the
read-only `/v1/models` list are open, which is normal for an OpenAI-compatible
server. There is no unauthenticated-inference hole.

## Untested stateful / destructive paths (the risk cluster)

These have working-looking code and guardrails but no test coverage, and several
run automatically or mutate disk:

- Folder autoprune (`sync_models_dir`) runs on every launch and deletes registry
  entries when enabled. The all-missing guardrail and registry backup are correct
  by inspection but entirely unverified.
- URL-pull resume (Range/206 handling, .part accounting): only the fresh-200 path
  is tested.
- Music generator: zero tests (the only generator with none).
- Image img2img / negative-prompt / LoRA: the differentiated FLUX features, untested.
- Abliterate: zero tests; `clone_heretic` calls bare `git`/`uv` with no `shutil.which`
  guard or try/except, so a machine without git on PATH gets a raw traceback.
- ensure_comfy auto-launch subprocess path: untested, fails opaquely (stdout/stderr
  to DEVNULL).

## Feature ledger

### Inference core + server
- OpenAI server protocol (chat stream/non-stream, completions, models, load/unload,
  TTFT/tok-s, CORS, bearer auth): Solid.
- GGUF in-process text inference (ctypes binding, templates, stop strings, marker
  scrub, rep-penalty, native-fault recovery): Works-with-caveats (ABI-fragile,
  pinned to one prebuilt DLL, never tested against a live DLL).
- Subprocess llama-cli fallback: Fragile (forces ChatML, drops sampling params /
  grammar / seed, leaks control tokens, chars/4 token counts).
- HF text: Works-with-caveats. HF multimodal: Partial (real code, narrow processor
  assumptions, untested).
- GGUF multimodal: Stub/Broken (silent no-op, see gap 1).
- Dynamic ctx growth + ctx_auto: Works-with-caveats (math solid+tested; VRAM-sizing
  becomes a constant without torch).
- KV-cache prefix reuse: Works-with-caveats (good concurrency hygiene; unverified vs
  real memory API).
- VRAM pre-flight + report: Works-with-caveats (silent no-op without torch).
- GPU auto-detection for GGUF: Partial/overstated (no vendor detection; you get the
  DLL you installed; the "proof" banner is suppressed by verbose=False).
- GBNF grammar: Works-with-caveats (wired for GGUF; HF silently ignores it).
- CLI conversation compaction: Works-with-caveats (keys off static n_ctx_max, not
  the effective auto ceiling).

### Coding agent
- Loop (max_turns, circuit breaker, turn-budget escalation, self-verify nudge): Solid.
- File/patch tools + cwd confinement + offline syntax check + fuzzy edit hints: Solid.
- Destructive-tool approval + diff preview + dry-run + auto-approve + always-allow: Solid.
- Session diffs (/changes, /diff) + file undo: Solid (shell side-effects not undoable).
- Project memory (LOCALCODER.md): Solid.
- MCP client (stdio JSON-RPC, tested vs a real subprocess server): Solid; Partial vs
  full spec (no server->client callbacks, stdio only, snapshot tool list).
- git tools / list_dir / tree / search_files / grep: Solid.
- run_tests: Works-with-caveats (no Makefile/tox/nox/unittest detection).
- Tool-call parser: Works-with-caveats. Handles XML and the documented Gemma/finetune
  mangles well, but bare JSON and ```json / ```tool_code fences (what the weakest
  local models emit) parse to nothing with no repair turn. "Works with local models"
  is really "works with strong-enough models."
- --scope: Fragile (fnmatch semantics reject same-dir files, no-path nav tools bypass
  it, run_shell ignores it entirely; it is a focus hint, not a boundary).
- Single-shot `localm coder "..."`: auto-approves all destructive tools unless
  always_confirm is set. Sharp edge.
- spawn_agent: Works-with-caveats (the lean sub-agent prompt is dead code; sub-agents
  carry the full prompt + re-index + re-spawn MCP servers; recursion depth unbounded;
  always auto_approve).
- Model-family prompt tuning: Works-with-caveats (sensible+tested; name-prefix
  detection is brittle; the gemma prompt invites a fenced format the parser rejects).
- Online providers (OpenAI/Anthropic): Works-with-caveats (mock-tested only;
  endpoint/model-id drift unverified, esp. Anthropic).

### GUI
- "Fully offline / zero build step": Solid (verified: all libs + fonts vendored, no CDN).
- Model-output XSS safety: Solid (DOMPurify on the only two content innerHTML writes).
- Privacy mode in the browser: Solid (no localStorage writes, active wipe, server 403s).
- Chat (stream, branching/regenerate/edit, personas, memory, knowledge grounding,
  web toggle, attachments, voice, compaction, export, slash menu, truncation note): Solid.
- Models / Knowledge / coder-session pages: Solid (well-tested).
- Object-URL lifecycle: Works-with-caveats (real leak: fetchImageURL blobs never
  revoked for inline media/thumbnails).
- GBNF field on HF models: silently ignored, no UI disable.
- Plugins page: Works-with-caveats (install not live until restart, disclosed).
- Settings page: Works-with-caveats (schema-less form, no validation/help).
- Chat/job SSE: Works-with-caveats (no reconnect; coder stream does reconnect).

### Generators (all gated on external ComfyUI + models)
- Image FLUX: Works-with-caveats (solid driver incl. real CFG negatives; img2img/
  negative/LoRA untested and node-id coupled).
- Music ACE-Step: Works-with-caveats (functional, wired everywhere claimed, zero tests).
- Video Wan 2.2: Solid (best-built and best-tested generator).
- ensure_comfy launch + 400-error surfacing: Works-with-caveats.
- History + move/delete/rename (GUI): Solid.
- LLM<->ComfyUI VRAM choreography: Works-with-caveats (best-effort unload; bare
  `localm music`/`video` never unload a running server).

### Model management + RAG
- pull (repo:file / repo / URL / split-GGUF / progress sentinel / disk preflight):
  Works-with-caveats (no HF revision pinning; URL resume untested).
- SHA256 dedup + two-tier identity: Solid (well-tested).
- aliases: Solid. discovery + VRAM fit badges: Solid (torch-free via registry).
- folder auto-sync / autoprune: Works-with-caveats (zero tests; runs every launch).
- Ollama interop: Works-with-caveats (untested).
- registry integrity: Solid in-process; cross-process is last-writer-wins (documented).
- text extraction: Solid (careful encoding sniffing; docx zip-bomb edge unguarded).
- BM25 + chunking + citations: Solid. embeddings/hybrid: Works-with-caveats (GGUF
  cannot embed; vector path never tested against a real backend).
- collections: Solid. chat-with-docs (attachments + collections): Works-with-caveats.

### Peripheral
- Voice STT/TTS: Works-with-caveats (HTTP plumbing tested; real Whisper path mocked;
  PyAV/ffmpeg dependency unsurfaced).
- MCP server: Solid (real JSON-RPC, round-trips vs localm's own client; "works with
  Claude Desktop" untested; protocol version hardcoded).
- Plugins (CLI command half): Solid + tested. Plugins (agent-tool half): Missing
  (parsed and displayed, never wired).
- Abliterate: Works-with-caveats (untested; fragile without git/uv; fork-flag coupled).
- Network policy: Solid (verified against loopback / link-local / metadata / file://).
- Privacy / audit modes: Solid (verified surface-by-surface, no leak found).
- Config + home detection + atomic persistence: Solid core; setup/launcher
  Works-with-caveats (.pyw double-click gotcha; ensure_dirs lacks parents=True;
  setup.bat pins 3.12 for both flavours).
- TLS: Stub/doc-only (no code; GUI LAN bind unwarned).

## What a home-to-pro user can rely on today
Offline chat with a GGUF text model through the GUI or CLI; the OpenAI-compatible
server; the coder agent for file edits / patches / git / tests with a real approval
flow and session diffs (with a strong-enough model); model pull / dedup / discovery;
BM25 document chat; the MCP server for exposing models; and a privacy mode that
genuinely leaves no traces. These are real and mostly tested.

## What will bite them
Attaching an image to a GGUF model (silently ignored); expecting plugin-exported
agent tools to work; expecting "built-in" generation without first installing
ComfyUI and 25 GB of models; expecting semantic RAG (it is BM25 unless you run an HF
model); a small local model whose tool calls silently fail to parse; `--scope` not
actually confining; a one-shot `coder` command auto-running shell tools; serving the
GUI on the LAN expecting TLS/auth warnings.

## Recommended priority fixes
1. Make GGUF `--mmproj`/`--image` raise a clear "not supported on GGUF" instead of
   silently dropping the image (or implement it).
2. Parser: accept bare JSON and ```json/```tool_code fences and add a one-shot
   "your tool call did not parse, reformat" repair turn. Biggest lever for the
   product's whole premise (working with local models).
3. Either wire plugin `tool_exports` into the coder or remove the claim from the README.
4. Add a LAN-bind warning to the `gui` command and relabel the TLS row as
   "reverse-proxy TLS guidance"; consider refusing non-loopback bind without a key.
5. Fix `--scope` to use path-aware matching and apply to nav tools (or document it as
   a non-boundary).
6. Honestly frame generation as "drives your ComfyUI" and document the model
   download burden; fix the ConditioningConcat and FLUX-encoder-filename doc errors.
7. Document that ctx_auto / VRAM warning / GGUF-embeddings require torch.
8. Add tests for autoprune, URL-pull resume, and the music generator.
