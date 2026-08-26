# Plugin interop: compatibility with third-party plugin ecosystems

Status: Open WebUI and oobabooga adapters are roadmap; Skills importer is implemented (see `docs/skills.md`). The kernel primitives (chat-pipeline hooks, tool registry) are in place and stable.

## Why this exists

localm's architecture is a fixed kernel (server, plugin engine, auth/permissions, settings, inference engine, model manager) with *everything else* as optional plugins: the coder agent, image/music/video generation, RAG, web access, voice input/output, and the MCP server. A natural question follows: can localm run the plugins that already exist for *other* projects (LM Studio, Open WebUI, oobabooga, agent skills)?

The honest answer: **there is no universal plugin ABI across projects, so you cannot run arbitrary foreign plugin code unchanged.** But that is not the real bar. Interop in practice is always *targeted*: you pick a concrete ecosystem, study what its plugins expect from the host, and write an adapter, one ecosystem at a time.

This document assesses three ecosystems (Open WebUI, oobabooga, Agent Skills), maps each ecosystem's demands onto localm's plugin contract, and proposes the small set of reusable primitives that unlock most of the value.

## How to judge "compatible"

Compatibility is a spectrum, not a yes/no. For any target project, its plugins make a bounded set of *demands on the host*. Enumerate them and grade each demand:

- **clean** - maps onto an existing localm capability with no new kernel surface.
- **shim** - maps once a small adapter (or one new kernel hook) exists.
- **infeasible** - assumes the other project's runtime or frontend in a way that does not port (deep in-process model access, a different UI framework, etc.).

The recurring demands an LLM-app plugin makes: call the model; register a tool the model can call; intercept/transform the prompt or response; read/write a setting; add a UI panel; access files/network/subprocess; be discovered and packaged.

localm's host contract (see `localm/plugins/contract.py`) already covers `mount_router`, `mount_static`, `add_settings`, `register_tab`, `plugin_config` / `save_plugin_config`, `has_scope` / `require_scope`, `engine`, `audit`, `browse_dirs`, and `register_chat_hook`. Plus, outside the plugin host: the coder agent's `TOOL_REGISTRY` (model-callable tools), the kernel `/v1/chat/completions` path, the capability scope taxonomy (`localm/scopes.py`), the settings schema, and the network policy.

A decisive reuse point: **localm already adapts foreign tools into one registry.** The coder's `register_mcp_tools` (external MCP servers) and `register_plugin_tools` (plugin tool exports) both feed one `TOOL_REGISTRY`, gating untrusted sources as destructive. Every "tools" mapping below is just a new *source* into that same machinery.

## The synthesis: two primitives + one importer

The three ecosystems collapse onto a very small amount of kernel surface:

- **(A) Foreign-tool adapter into the tool registry.** MCP + plugin-export adapters exist (used by the coder agent today). Unlocks Open WebUI Tools.
- **(B) Chat-pipeline hook.** IMPLEMENTED (see `localm/inference/chat_pipeline.py` and the "Chat pipeline hooks" section of `docs/plugins.md`). The three phases - `inlet(messages, ctx)` pre-inference, `stream(token, ctx)` per-streamed text (sync only), and `outlet(text, messages, ctx)` post-inference - serve Open WebUI Filters *and* oobabooga's text-pipeline hooks (input/output modifiers). A plugin registers with `host.register_chat_hook(phase, fn)`. It is a server-side seam that runs for every `/v1/chat/completions` call, downstream of the engine's marker scrubbing. localm's existing RAG / assistant-memory / web injection is client-side in the SPA, so this hook chain is the new server-side home for those and for foreign filters.
- **(C) Skills importer.** IMPLEMENTED (see `docs/skills.md`). An Agent Skill is markdown instructions plus bundled resources, and localm's coder agent is exactly the consumer it expects. Needs neither A nor B.

Everything else is either a larger one-off (a virtual-model backend) or out of reach (runtime/frontend mismatch); see "Reachable vs out of reach" below.

## Open WebUI

### Anatomy

Open WebUI plugins are **single `.py` files** with a frontmatter docstring (`title`, `author`, `version`, `required_open_webui_version`, `requirements`), imported via the admin Workspace UI or community site, stored in its database, and they **execute arbitrary Python on the server** with `requirements` pip-installed at import. Two families:

- **Tools** - a `class Tools:` whose methods become model-callable functions. Parameter type hints plus the `:param:` docstring are compiled into the JSON function schema the model sees (native function-calling). Optional `Valves` / `UserValves` (pydantic `BaseModel` inner classes) auto-generate admin / per-user settings. Methods may declare injected context params: `__user__`, `__event_emitter__`, `__event_call__`, `__request__`, `__metadata__`, `__model__`, `__id__`.
- **Functions**, three kinds:
  - **Pipe** - registers as a *selectable model*; `async def pipe(self, body)` handles the whole turn (returns a string, dict, or async generator for streaming). `pipes()` returns several `{id, name}` models (a "manifold").
  - **Filter** - `inlet(body, __user__)` pre-processes the request, `stream(event)` mutates streamed chunks, `outlet(body, __user__)` post-processes. `Valves.priority` orders filters; `self.toggle` / `self.icon` make it a user-toggleable chip.
  - **Action** - adds a button to the message toolbar; `action()` runs on click via the event system.

### Mapping

| Open WebUI demand | localm mapping | Verdict |
|---|---|---|
| Tool methods (schema from hints + docstring) | introspect `Tools` -> emit defs into `TOOL_REGISTRY`; dispatch to the method | shim, high value |
| injected dunders (`__user__`, `__event_emitter__`, `__request__`, ...) | supply from localm context: auth principal, the job/SSE stream as emitter, a Request or stub | shim (most pure-logic tools ignore these) |
| Valves / UserValves | `add_settings` + `plugin_config` / `save_plugin_config` | clean |
| Filter inlet/stream/outlet | the chat-pipeline hook (B) | clean |
| Pipe (code-as-model) | a new virtual-model backend whose "inference" calls `pipe(body)`; `pipes()` -> several virtual models | shim, larger lift |
| Action (toolbar button + event UI) | server `action()` can run; the rich button/confirm/input UI is OWUI-frontend-coupled | partial / defer |
| single `.py` + pip `requirements` + arbitrary exec | ingest one file, parse frontmatter, consented pip into the venv, load the module | shim + a security decision |

### Adapter: an `openwebui-compat` plugin

Itself a normal localm plugin. It discovers Open WebUI `.py` files, parses the frontmatter, consent-gates `requirements`, loads the module, and routes by class (per the Mapping table above). Everything scope-gated and audited, with side-effecting tool calls marked destructive (the coder-style confirm flow).

**Verdict:** v1 = **Tools + Valves** (reuses the existing tool-adapter machinery, unlocks the large community tool catalog, mostly introspection + dunder/valve shims). Filters next (they force the chat hook, which is independently useful). Pipes later (the virtual-model backend). Actions are server-runnable but UI-partial.

## oobabooga (text-generation-webui)

### Anatomy

A folder `extensions/<name>/script.py` with a `params` dict (`display_name`, `is_tab`) and any of ~13 hook functions, activated with `--extensions`, chained in order, with deep access to live state via `modules.shared`. The UI is **Gradio**. The hooks fall into three tiers:

- **Text-pipeline** (strings / prompt / history in and out): `input_modifier`, `output_modifier`, `chat_input_modifier`, `bot_prefix_modifier`, `history_modifier`, `state_modifier`, `custom_generate_chat_prompt`.
- **Deep-inference** (model internals): `logits_processor_modifier`, `tokenizer_modifier`, `custom_generate_reply`, `custom_tokenized_length`.
- **UI** (Gradio): `ui`, `custom_css`, `custom_js`, `setup`, `is_tab`.

### Mapping

| Tier | localm | Verdict |
|---|---|---|
| text-pipeline hooks | the chat-pipeline hook (B) - the same one OWUI Filters need | shim, clean |
| `logits_processor_modifier` | HF backend supports `LogitsProcessor`; the GGUF ctypes binding cannot inject a Python logits processor mid-decode | HF = shim, GGUF = infeasible |
| `tokenizer_modifier` / `custom_tokenized_length` | per-backend `count_tokens` exists; live token/embed access is HF-only | HF = partial, GGUF = no |
| `custom_generate_reply` | replaces the decode loop - only meaningful with an in-process HF model | HF-only, deep |
| `ui` / `is_tab` (Gradio) | localm SPA is vanilla JS, not Gradio | infeasible |
| `custom_css` / `custom_js` | could inject into the SPA shell | partial |
| `modules.shared` global | no equivalent; provide a shim exposing active model + settings | shim (partial) |

### The convergence (key finding)

Open WebUI Filters and oobabooga's text-pipeline hooks want the *same thing*: intercept and mutate the prompt and the response in the chat pipeline. Building **one** clean chat-pipeline hook (B) unlocks the most common extension type in *both* ecosystems at once.

**Verdict:** partial, concentrated on the text-pipeline hooks (riding B). The deep-inference tier is HF-backend-only and largely infeasible on localm's default GGUF runtime; the Gradio UI tier does not port. Honest scope: support the string-modifier extensions (translators, prompt templaters, memory injectors).

## Agent Skills

### Anatomy

A folder with a `SKILL.md` (YAML frontmatter: `name`, `description`, optional `allowed-tools`) plus a markdown instruction body and bundled scripts/resources. Loaded by **progressive disclosure**: the agent sees `name` + `description` first, loads the body when relevant, then reads bundled files on demand.

This is a *format*, not a runtime, and it maps to localm almost for free because the coder agent is exactly the consumer a Skill expects: a system prompt + memory + file/shell tools.

### Mapping

| Skill element | localm mapping | Verdict |
|---|---|---|
| `name` + `description` (disclosure L1) | `list_skills()` tool, read-only, never gated | clean |
| body instructions (L2) | `use_skill(name)` injects the `SKILL.md` body into agent context | clean |
| bundled scripts/resources (L3) | `use_skill(name, file=...)` plus the agent's existing `read_file` / `run_shell` | clean, reuses existing tools |
| `allowed-tools` | a dispatch-time tool-name restriction, hard-enforced for the rest of the turn (not merely advisory) - see `docs/skills.md` | clean, implemented |

**Verdict:** the cleanest and lowest-risk of the three, and the only one fully implemented. The coder gains `list_skills` and `use_skill(name)` tools (see `localm/plugins/coder/skills.py`) that let it discover and load skills from `<data dir>/skills/` and `.localcoder/skills/` (project-local). Lower risk than Open WebUI / oobabooga: the agent *chooses* to run a bundled script through its existing confirm/scope/audit, and a skill's declared `allowed-tools` is enforced rather than the host `exec`-ing foreign Python unconstrained. (For plain non-agentic chat its value drops to a structured persona + context.)

## Reachable vs out of reach

| Ecosystem | Reachable | Out of reach |
|---|---|---|
| Open WebUI | Tools, Valves, Filters (via B) | Actions' rich event-UI (frontend-coupled); Pipes (roadmap) |
| oobabooga | text-pipeline hooks (via B) | logits/tokenizer/custom_generate (HF-only; infeasible on GGUF), Gradio UI |
| Agent Skills | essentially all of it (implemented) | (nothing significant) |

## Recommended build order

Leverage-weighted:

1. **Skills importer** - IMPLEMENTED. Built directly into the coder agent (not a registered plugin under `localm/plugins/contract.py`), exposing two tools (`list_skills`, `use_skill`), zero kernel change. Skills from `<data dir>/skills/` and `.localcoder/skills/` are discovered and loaded on agent start (see `docs/skills.md`).
2. **Chat-pipeline hook (B)** - IMPLEMENTED. The kernel hook runs on every `/v1/chat/completions` call; the Tools and text-pipeline adapters build on top of it (see the per-ecosystem sections above).
3. **Open WebUI Tools adapter** - ROADMAP. Reuses the existing tool-adapter machinery (see Open WebUI above).
4. **Open WebUI Pipes** - ROADMAP. A virtual-model backend (bigger, deferred).
5. **oobabooga text-pipeline adapter** - ROADMAP. Bridges oobabooga text-pipeline hooks to the chat hook (see oobabooga above).

## Security posture

Foreign code is untrusted. Every adapter runs it behind localm's existing guarantees: capability scopes, consent on install, audit of actions, and destructive-gating of side-effecting calls (the coder's confirm flow) - opt-in, never default-on. This mirrors how the MCP client already treats untrusted servers. Skills are the gentlest (the agent invokes bundled scripts via gated tools); Open WebUI and oobabooga `exec` foreign Python directly and must be treated accordingly, including consented `pip` of declared requirements into the environment.

## Architectural placement

Each compatibility layer is **itself a localm plugin** (e.g. `openwebui-compat`, `oobabooga-compat`). The kernel only ever gains the two small primitives (B exists; A is already present). "Support ecosystem X" stays an installable - the distro model applied to interop itself.

## Verification details (for adapter development)

Before building, confirm these facts against real source code, not docs:

- The exact Open WebUI load mechanism (DB source-string + `exec` vs file import) and which `open_webui.*` internals its plugins import - shapes the adapter's loader and how much of the dunder/event surface must be shimmed. Confirm against Open WebUI source.
- Which real-world extensions in each ecosystem actually stay within the "reachable" set, to size the payoff (a sample audit of the top community tools/extensions).
- MCP server configuration: the coder reads external MCP servers from `.localcoder/config.toml` (section `[mcp.servers.<name>]`). See `docs/mcp.md` for full details.
- Claude Desktop MCP config paths (for the localm MCP server):
  - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
  - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - Linux: Claude Desktop has no official Linux build; `localm mcp --print-config` prints `~/.config/Claude/claude_desktop_config.json` for third-party clients that follow the same config layout
- ComfyUI: runs at default `http://127.0.0.1:8188`; launched with `python main.py`, prints its URL on startup.

## Sources

- Open WebUI: Tools and Functions plugin docs, Filter/Pipe function docs, Valves (docs.openwebui.com/features/extensibility/).
- oobabooga: text-generation-webui Extensions wiki and `extensions/example/script.py`.
- Agent Skills: https://github.com/anthropics/skills.
