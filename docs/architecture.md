# Architecture

## Overview

localm core is a model loader plus a plugin engine. The CLI is a thin shell
over a pluggable inference backend, and everything above bare chat is a
plugin. The core design rule: the CLI knows nothing about inference;
inference knows nothing about CLI. They communicate through `Engine`, with
a couple of narrow, deliberate exceptions (`localm/cli/doctor.py` imports
the llama.cpp loader directly for GPU-probe diagnostics; `localm/cli/chat.py`
catches specific backend exception types). CHAT
is the protected, preinstalled plugin (#0); coder, browser, image, music,
video, rag, web, memory, voice (Whisper STT), tts (Kokoro in-browser TTS),
jobs (scheduled tasks), and mcp are all plugins layered on top.

```
CLI (localm/cli/)                  Plugin engine (localm/plugins/)
  └── Engine (inference/engine.py)   ├── engine.py    PluginManager
        ├── GgufBackend              ├── contract.py  Host / Surface / PluginSpec
        │     └── LlamaCpp (ctypes)  ├── catalog.py   first-party catalog
        └── HFBackend                ├── builtin/     store (read-only)
              └── HF Transformers     └── <data dir>/plugins/  installed
                    (+ native AWQ)
```

## Layering

[layering.toml](layering.toml) declares the package's tiers, top to bottom,
and places every top-level unit under `localm/` in exactly one of them. The
rule is small: a module-level import may target only a unit in a lower tier;
units that share a tier are peers and never import each other at module
level; the package root (`localm/__init__.py`, which holds only the version)
sits below every tier. Only import statements that run at module import
time are covered: a function-local import (how a genuine import cycle is
broken), an `importlib` call, and a plugin loaded under its own module name
are outside it, and so are tests and scripts.

`scripts/check_hygiene.py` enforces the map on every commit and in CI. It
also fails on a unit the map does not place, on a placed unit that no longer
exists, and on a unit placed twice, so the map cannot rot silently. There is
no allow-list: a tier carries exactly `name`, `role` and `units`, and any
other key is rejected. A new module goes in the lowest tier whose role fits
and that sits above everything the module imports at module level; moving an
existing unit between tiers is a design change and the pull request says why.

## Removing or renaming a config key or a response field

A test can depend on a name that never contains the one being removed.
Dropping the config key `managed_comfy_enabled` also dropped the `enabled`
field from `/api/comfy/managed-status`, and a test asserting
`body["enabled"]` kept failing while a search for the key found nothing.
Two tools cover that gap.

`scripts/check_hygiene.py` (check 10) reads every route whose handler
builds its JSON response from dict literals, and every test that reads a
top-level key from that route's response, and fails when the key is one the
handler cannot produce. It runs on every commit through the pre-commit hook
and in CI's hygiene gate, before any test runs. It does not see a handler
whose response comes from another module, a model class, a file or stream,
or a non-literal merge (that shape is open and not judged); a nested key; a
key read through a helper or a fixture rather than a `.json()` call in the
test function itself; or a GUI script reading the response.

`scripts/affected_tests.py` prints the test files a change affects: the
changed tests, every test importing a changed module, every test naming a
route path a changed module registers, and every test naming a changed
module or file. It reads the change from git (the branch's diff from
`origin/master` plus uncommitted work) and prints one path per line, so the
selection substitutes straight into a pytest command:

    pytest $(python scripts/affected_tests.py) -m "not integration"

`--why` prints the reason next to each file, `--files` takes an explicit
list instead of git, and `--depth 1` also follows the modules that import a
changed one. The substitution above can never hand pytest an empty or a
whole-suite argument list: when nothing is affected the script prints
`tests/NO_TEST_FILE_IS_AFFECTED`; when the selection exceeds a quarter of
the suite it prints `tests/SELECTION_TOO_WIDE_FOR_A_TARGETED_RUN_SEE_STDERR`
and exits 3 (the change touches a module most tests import, and no targeted
run stands in for the suite there; `--list-wide` prints the selection
anyway); when the script itself fails it prints
`tests/AFFECTED_TESTS_FAILED_SEE_STDERR` and exits 1. None of those paths
exists, so pytest stops with "file or directory not found".

CI runs that selection on every pull request without the `full-ci` label.
The `python-pr-gate` job in `.github/workflows/ci.yml`
(`scripts/run_affected_tests.py`) runs the `--depth 1` selection on ubuntu
after the lockfile check, the hygiene gate and ruff. It never runs the
whole suite: when the depth-1 selection is wider than a quarter of the
suite it runs the depth-0 selection instead, and it fails when that is wide
too, when the selector fails, prints nothing, names a path that is not an
existing test file, or cannot resolve the base ref it diffs from. A change
that fails the gate that way needs the two-platform matrix with coverage,
which runs on a PR carrying the `full-ci` label in place of the gate.

`merge-policy` (`scripts/merge_policy.py`) is the one check that sums the
others up. It runs on every pull request once `python-pr-gate`, `lint`,
`gui-tests`, `test`, `mutation-scope` and `mutation-test` have finished,
whatever their results, and it is never skipped on a pull request, so a
needed job that was skipped or failed cannot read as a pass. It passes when
`lint`, `gui-tests` and `mutation-scope` succeeded, `mutation-test` did not
fail when it ran (skipped is neutral) and, on a PR without the `full-ci`
label, `python-pr-gate` succeeded and the change is not a release
(`VERSION` unchanged); on a PR with the label, when the `test` matrix
succeeded. The two-platform matrix runs at release, not on
an ordinary pull request: a release PR without the label fails
`merge-policy` with the label named, and every other PR merges on the four
cheap jobs. Adding the label to an open PR leaves the earlier unlabelled
run's `merge-policy` in place next to the new one; the newest check run of
that name is the verdict. The summary also lists the matrix categories the
change touches, for the release run to know what it covers: the trust
boundary (`auth`, `scopes`, `tls`, `bindhost`, `netlisten`, `portmux`,
`netpolicy`, `netpin`, `pathsafe`, `config`), the plugin engine and contract
(`localm/plugins/*.py`), inference, the workers and the native binding
(`localm/inference/` except `routes/`, `_mp_spawn`, `_torch_gpu_probe`,
`setup_llama`, `runtime/`), packaging and the installers, and the CI
workflows and gates. None of those blocks a merge; the list lives in
`scripts/merge_policy.py`.

`mutation-test` is the mutation-testing gate for the trust boundary: the
eight modules in `[tool.mutmut] only_mutate` (`pyproject.toml`). On the
weekly schedule, on a dispatch, or on a pull request carrying the
`mutation-test` label, one `mutation-run` shard per module runs mutmut
(`scripts/mutmut_run.py run "localm.<module>.*"`) and uploads its result
file; `mutation-test` merges the eight and runs
`scripts/check_mutation_floors.py` against the committed baseline
`scripts/mutation_baseline.json`. The baseline records every mutant's
disposition - `killed`, `survived` (a known gap, counted against the score)
or `{"equivalent": "<reason>"}` (excluded, never silently) - plus a per-module
score floor and the `controls`: concrete mutants, at least one per
security-decision class (a weakened scope check, an authorization fallback
flipped to allow, a skipped SSRF redirect re-validation, a widened
net_mode=off exemption, a path-confinement bypass, a loopback classifier
that accepts an unparseable host), that must stay killed. The gate fails on
a score below its floor, a mutant recorded as killed that now survives, a
mutant with no disposition
(new, or in a function whose source hash changed), a control not killed, or
an incomplete run. The job uploads a proposed baseline
(`mutation-baseline-proposed`, floors ratcheted up, equivalents kept) so a
changed function's new mutants can be classified and committed without a
local run; mutmut itself runs on Linux only. The shards never run
automatically on a pull request (the `auth` shard alone takes about an hour);
the weekly schedule is what gates master. `mutation-scope`
(`scripts/mutation_scope.py`) instead annotates a pull request that touches
a mutated module or the gate with a notice: that the gate runs on it when it
carries the `mutation-test` label, and that the gate did not run on it
otherwise. `mutation-scope` fails when it cannot compute the diff, and
`merge-policy` fails with it. The two decision classes that
live in `localm/inference/http_server.py` rather than a mutated module - an
unsafe route exempted from the origin gate, and `bind_host` replaced by the
peer address - are pinned by `tests/test_trust_boundary_controls.py`.

So before removing or renaming a config key, a route, or a response field:
search for the old name and for every field name the route derives from
it, run the hygiene check, and run the affected selection.

## Engine

`Engine` auto-detects the backend from the model path (`.gguf` file or
Ollama `sha256-*` blob to `GgufBackend`, directory with `config.json` to
`HFBackend`), exposes `chat_stream(messages, ...)`, `embed(texts)`, and
`count_tokens(text)`, and reloads the backend transparently when something
unloaded it (e.g. image generation borrowing the VRAM).

## Model loading isolation

Both backends run the real model - and every native call it makes - inside a
disposable child process, never in the server process itself:

- **GGUF** (`llamacpp/_worker.py`'s `GgufWorker`, spawned by `_runner.py`):
  `llama_load_model_from_file` and every later native call (context growth,
  token-by-token decode) can hard-abort the whole process on a native
  CUDA/HIP driver failure, which no Python `try`/`except` can catch. The
  model's entire lifecycle - load, generate, grammar-check, unload - runs in
  the child, so a native abort kills only that child; the parent reports it
  as a clean, catchable error and reloads fresh on the next request.
- **HF Transformers** (`_hf_worker.py`'s `HFWorker`, spawned by
  `_hf_runner.py`): the tokenizer, `model.generate()` and a torch forward
  pass are equally uninterruptible from Python, so a hang there would
  otherwise burn a slot in the server's shared thread pool permanently.
  Isolating it is what makes such a hang killable without restarting the
  server.

`GgufBackend`/`HFBackend` are thin parent-side proxies. `GgufBackend` runs a
preflight VRAM check in the parent before a child is even spawned, so a load
that can never fit fails fast without paying a process-spawn cost.
`HFBackend` has no such preflight; it does two other parent-side checks
(custom-code trust, tokenizer regex safety) and leaves VRAM budgeting to the
spawned child, during `device_map` construction.

## GgufBackend and the ctypes binding

`GgufBackend` wraps `LlamaCpp`, a pure-ctypes binding to the native llama.cpp
library (`llama.dll` on Windows, `libllama.so` on Linux, `libllama.dylib` on
macOS; no llama-cpp-python). Key behaviour:

- **Dynamic context window**: contexts start at `n_ctx` and are rebuilt
  larger in `n_ctx_grow` steps, capped at `n_ctx_max`; `ctx_auto` sizes the
  cap from free VRAM at load. Prefill is always chunked to `n_batch`.
- **KV prefix reuse**: between calls the common token prefix with the
  previous request stays in the KV cache (llama_memory_* API, probed at
  runtime); only the new suffix is prefilled.
- **Sampler chain**: grammar (GBNF), repetition penalty, top-k, top-p,
  min-p, temperature, dist; greedy when temperature is 0.
- **Multi-Token Prediction (MTP) speculative decoding**: a model trained
  with its own next-n draft head can draft and verify two tokens per step
  through a dedicated draft context, instead of a separate draft model. Off
  by default (`mtp_enabled`) and engages only where the runtime can build
  and feed a real draft head for that model; see
  [llamacpp-binding.md](llamacpp-binding.md) for the mechanism.
- **Output filtering**: a stop-string filter handles end-of-turn sequences
  split across tokens; an internal-marker scrubber (`localm/textnorm.py`)
  strips thinking-channel tags and leaked chat-template control tokens some
  finetunes emit as text. Chat output is always scrubbed; debug mode
  additionally logs the raw, unscrubbed text.
- **VRAM pre-flight**: free VRAM is checked against model size before load.
  If the model could never fit even on an empty card, load is refused
  outright; if something else is merely using the VRAM, it is a warning and
  the load continues.
- No fallback by design: if the DLL cannot be loaded, `load()` raises a clear
  error pointing at `localm setup-llama`.

## HFBackend and native AWQ

`HFBackend` drives HuggingFace `transformers` (`AutoModelForCausalLM` /
`AutoProcessor`) for model directories `GgufBackend` cannot load - anything
without a `.gguf` file, including multimodal checkpoints such as
Gemma4UnifiedForConditionalGeneration (text + image + audio). GPU use goes
through `torch.cuda`, which maps to ROCm on AMD systems running PyTorch+ROCm.

`localm/inference/backends/awq.py` registers a native AWQ (Activation-aware
Weight Quantization) quantizer into `transformers`' own
`AUTO_QUANTIZER_MAPPING` the first time the HF worker imports
`transformers`. A HuggingFace AWQ 4-bit checkpoint (`quantization_config`
naming `"awq"` in its `config.json`) then loads and runs through the normal
`from_pretrained` path with no extra flag - `NativeAWQLinear` dequantizes
each packed 4-bit layer on the fly during the forward pass. This works
across Windows ROCm, Linux ROCm, NVIDIA CUDA, Intel XPU and CPU without
external compiled dependencies (gptqmodel, autoawq, torchao), which either
do not build on Windows ROCm or are not installed by default. Multimodal and
hybrid-attention layers that AWQ checkpoints leave unquantized (vision
towers, projectors, some hybrid-attention sublayers) are skipped and kept in
their original precision.

## HTTP server

`inference/http_server.py` builds the FastAPI app (`create_app(engine)`) and
holds the shared inference state; the route handlers themselves live in
`inference/routes/` modules (chat, models, config, keys, session, admin,
system, gpu). The plugin management API is mounted separately, by
`localm/plugins/engine.py`. The synchronous `engine.chat_stream()` runs in a
thread; tokens cross into the event loop via `call_soon_threadsafe` and
stream out as SSE. Inference is serialised per loaded model (an asyncio
semaphore per display name), not globally - two concurrently loaded models
can generate at once. Endpoints are documented in
[server-api.md](server-api.md).

`inference/capability_routing.py` decides, for a chat request that did not
pin a model by name, whether the loaded model can actually serve it (vision,
tool use, reasoning, context length) and picks an installed one that can when
it cannot; `http_server.plan_capability_route()` builds the request's
`CapabilityNeeds` and `routes/chat.py` applies the decision. `peer_routing.py`
is a separate, unrelated mechanism: it forwards a chat request straight to
another localm instance on this machine that already has the model loaded,
after verifying the forward target resolves to loopback.

## Conversation compaction

`inference/compact.py` summarises older chat turns through the model when a
conversation reaches 70% of the context ceiling, keeping the system prompt
and the last two exchanges verbatim, with a hard-trim fallback that never
raises. Used by `localm run` interactive chat; the GUI (itself a plugin
surface now) implements the same protocol client-side. The coder agent has
its own GBNF-structured compaction in the `localm/plugins/coder/agent/` package.

## Plugin engine

Everything above bare chat is a plugin, managed by `PluginManager` in
`localm/plugins/engine.py`. A plugin ships a `plugin.toml` manifest
(`[plugin]` + `[surface]` tables) plus a module exporting `register(host)`
and `unregister()`.

**Two locations.** The *store* is `localm/plugins/builtin/` (the bundled,
read-only first-party plugins listed above). *Installed* plugins live under
`<data dir>/plugins/`. Installing copies a plugin from the store into the
installed location.

**States.** A plugin is *active* only when it is both installed (present under
`<data dir>/plugins/`) and enabled (listed in `config["plugins_enabled"]`); the
store also tracks what is *available* (in the catalog but not yet installed).
By default only chat is active. See [plugins.md](plugins.md) for the full model.

**Chat is plugin #0.** CHAT is protected and preinstalled; it cannot be
disabled or uninstalled.

**Contract.** `localm/plugins/contract.py` defines the protocols a plugin
sees: `Surface` and `PluginSpec` (manifest shape) and `Host` (the API the
engine hands to `register`). The host exposes `mount_router`, `mount_static`,
`add_settings`, `register_tab`, `plugin_config`, `save_plugin_config`,
`engine`, `audit`, and `browse_dirs`. `catalog.py` holds the static
first-party catalog.

**Lifecycle.** `PluginManager` discovers installed plugins, then
`load_enabled` mounts each active plugin at runtime through `PluginHost`
(`mount_router` for FastAPI routes, `mount_static` for assets). Enable and
disable toggle the config entry; install copies store -> installed (and
enables), uninstall removes the installed directory. Mounting and unmounting
happen at runtime without restarting the server.

**api_version gating.** Each manifest declares the contract version it
targets; the engine refuses to load a plugin whose `api_version` it does not
support, surfacing an error rather than crashing.

**Capability scopes.** Every plugin declares a capability scope, and its
HTTP routes are gated to that scope, so a plugin cannot reach beyond the
permissions it asked for.

See [plugins.md](plugins.md) for the full authoring guide. A broken plugin
warns and is skipped, never crashing the host.

## Debug mode

`debuglog.py` implements `--debug`: a shared log file under
`<data dir>/logs/` (path carried in `LOCALM_DEBUG` so child processes append
to the same file), HTTP request timing, and redirection of the native
llama.cpp stderr into the log so crash abort reasons are captured instead
of suppressed.

## Protocol

`inference/protocol.py` defines Pydantic v2 models for the OpenAI wire
format: `ChatRequest`, `CompletionRequest`, `EmbeddingRequest`, `Message`
(string or multipart content), `ChatChunk`/`ChatResponse`, and `UsageInfo`
including `ttft_ms` and `tokens_per_sec`.
