# Plugin interop: compatibility with third-party plugin ecosystems

Status: design / assessment. Author-assisted survey, 2026-06. Implemented so
far: the **Skills importer** (see `docs/skills.md`) and the kernel
**chat-pipeline hook** (primitive B below, see the "Chat pipeline hooks" section
of `docs/plugins.md`). The ecosystem adapters that ride those primitives (Open
WebUI / oobabooga) are still to come.

## Why this exists

localm's north star is a Linux-distro shape: a small fixed kernel (server, plugin
engine, auth/permissions, settings, inference engine + model manager) with
*everything else* a plugin the user can add, remove, or swap, ideally imported
from many different repos. A natural question follows: can localm run the addons
that already exist for *other* projects (LM Studio, Open WebUI, oobabooga, agent
"skills", etc.)?

The honest answer up front: **there is no universal plugin ABI across projects,
so you cannot run arbitrary foreign plugin code unchanged.** But that is not the
real bar. Interop in practice is always *targeted*: you pick a concrete ecosystem,
study what its addons expect from the host, and write an adapter. VS Code reads
TextMate grammars; dozens of tools ship "OpenAI-compatible" modes. "No universal
ABI" only means you do it one ecosystem at a time, which is normal and tractable.

This document assesses three ecosystems (Open WebUI, oobabooga, Agent Skills),
maps each ecosystem's demands onto localm's plugin contract, and proposes the
small set of reusable primitives that unlock most of the value.

## How to judge "compatible"

Compatibility is a spectrum, not a yes/no. For any target project, its addons
make a bounded set of *demands on the host*. Enumerate them and grade each
demand:

- **clean**  - maps onto an existing localm capability with little glue.
- **shim**   - maps once a small adapter (or one new kernel hook) exists.
- **infeasible** - assumes the other project's runtime or frontend in a way that
  does not port (deep in-process model access, a different UI framework, etc.).

The recurring demands an LLM-app addon makes: call the model; register a tool the
model can call; intercept/transform the prompt or response; read/write a setting;
add a UI panel; access files/network/subprocess; be discovered and packaged.

localm's host contract (see `localm/plugins/contract.py`) already offers:
`mount_router`, `mount_static`, `add_settings`, `register_tab`, `plugin_config` /
`save_plugin_config`, `has_scope` / `require_scope`, `engine`, `audit`,
`browse_dirs`. Plus, outside the plugin host: the coder agent's `TOOL_REGISTRY`
(model-callable tools), the kernel `/v1/chat/completions` path, the capability
scope taxonomy (`localm/scopes.py`), the settings schema, and the network policy.

A decisive reuse point: **localm already adapts foreign tools into a registry.**
The coder's `register_mcp_tools` (external MCP servers) and `register_plugin_tools`
(plugin tool exports) both feed one `TOOL_REGISTRY`, gating untrusted sources as
destructive. Every "tools" mapping below is just a new *source* into that same
machinery.

## The synthesis: two primitives + one importer

The three ecosystems collapse onto a very small amount of new kernel surface:

- **(A) Foreign-tool adapter into the tool registry.** About 80% exists already
  (MCP + plugin-export adapters). Unlocks Open WebUI **Tools**.
- **(B) A chat-pipeline hook (`inlet` / `stream` / `outlet`).** IMPLEMENTED (the
  kernel piece; see `localm/inference/chat_pipeline.py` and the "Chat pipeline
  hooks" section of `docs/plugins.md`). The convergence point: it serves Open
  WebUI **Filters** *and* oobabooga's input/output **modifiers** - the single
  most common extension type in both ecosystems - and a plugin registers
  transforms via `host.register_chat_hook(phase, fn)`. It is a server-side seam
  that runs for every `/v1/chat/completions` client. Note: localm's existing
  RAG / assistant-memory / web injection is done client-side in the SPA (before
  the request is sent), so the hook chain does not replace it; it is the
  universal server-side place those, and foreign filters, can live.
- **(C) A Skills importer.** Needs neither A nor B: an Agent Skill is instructions
  plus resources, and localm's coder agent is exactly the consumer it expects.

Everything else (Open WebUI Pipes, deep-inference oobabooga hooks, Gradio UI,
Open WebUI Actions) is either a larger one-off (a virtual-model backend) or out
of reach (runtime/frontend mismatch).

## Open WebUI

### Anatomy

Open WebUI plugins are **single `.py` files** with a frontmatter docstring
(`title`, `author`, `version`, `required_open_webui_version`, `requirements`),
imported via the admin Workspace UI or the community site, stored in its database,
and they **execute arbitrary Python on the server** with `requirements`
pip-installed at import. Two families:

- **Tools** - a `class Tools:` whose methods become model-callable functions.
  Parameter type hints plus the `:param:` docstring are compiled into the JSON
  function schema the model sees (native function-calling). Optional `Valves` /
  `UserValves` (pydantic `BaseModel` inner classes) auto-generate admin / per-user
  settings. Methods may declare injected context params: `__user__`,
  `__event_emitter__`, `__event_call__`, `__request__`, `__metadata__`,
  `__model__`, `__id__`.
- **Functions**, three kinds:
  - **Pipe** - registers as a *selectable model*; `async def pipe(self, body)`
    handles the whole turn (returns a string, dict, or async generator for
    streaming). `pipes()` returns several `{id, name}` models (a "manifold").
  - **Filter** - `inlet(body, __user__)` pre-processes the request, `stream(event)`
    mutates streamed chunks, `outlet(body, __user__)` post-processes. `Valves.priority`
    orders filters; `self.toggle` / `self.icon` make it a user-toggleable chip.
  - **Action** - adds a button to the message toolbar; `action()` runs on click
    via the event system.

### Mapping

| Open WebUI demand | localm mapping | Verdict |
|---|---|---|
| Tool methods (schema from hints + docstring) | introspect `Tools` -> emit defs into `TOOL_REGISTRY`; dispatch to the method | shim, high value |
| injected dunders (`__user__`, `__event_emitter__`, `__request__`, ...) | supply from localm context: auth principal, the job/SSE stream as emitter, a Request or stub | shim (most pure-logic tools ignore these) |
| Valves / UserValves | `add_settings` + `plugin_config` / `save_plugin_config` | clean |
| Filter inlet/stream/outlet | the new chat-pipeline hook (B) | shim + one worthwhile kernel hook |
| Pipe (code-as-model) | a new virtual-model backend whose "inference" calls `pipe(body)`; `pipes()` -> several virtual models | shim, larger lift |
| Action (toolbar button + event UI) | server `action()` can run; the rich button/confirm/input UI is OWUI-frontend-coupled | partial / defer |
| single `.py` + pip `requirements` + arbitrary exec | ingest one file, parse frontmatter, consented pip into the venv, load the module | shim + a security decision |

### Adapter: an `openwebui-compat` plugin

Itself a normal localm plugin. It discovers Open WebUI `.py` files, parses the
frontmatter, consent-gates `requirements`, loads the module, and routes by class:
`Tools` -> register tools; `Filter` -> the chat hook; `Pipe` -> a virtual model;
`Valves` -> settings. Everything scope-gated and audited, with side-effecting tool
calls marked destructive (the coder-style confirm flow) - exactly how the MCP
client already treats untrusted servers.

**Verdict:** v1 = **Tools + Valves** (reuses the existing tool-adapter machinery,
unlocks the large community tool catalog, mostly introspection + dunder/valve
shims). Filters next (they force the chat hook, which is independently useful).
Pipes later (the virtual-model backend). Actions are server-runnable but UI-partial.

## oobabooga (text-generation-webui)

### Anatomy

A folder `extensions/<name>/script.py` with a `params` dict (`display_name`,
`is_tab`) and any of ~13 hook functions, activated with `--extensions`, chained
in order, with deep access to live state via `modules.shared`. The UI is **Gradio**.
The hooks fall into three tiers:

- **Text-pipeline** (strings / prompt / history in and out): `input_modifier`,
  `output_modifier`, `chat_input_modifier`, `bot_prefix_modifier`,
  `history_modifier`, `state_modifier`, `custom_generate_chat_prompt`.
- **Deep-inference** (model internals): `logits_processor_modifier`,
  `tokenizer_modifier`, `custom_generate_reply`, `custom_tokenized_length`.
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

Open WebUI Filters and oobabooga's input/output modifiers want the *same thing*:
intercept and mutate the prompt and the response in the chat pipeline. Building
**one** clean chat-pipeline hook (B) unlocks the most common extension type in
*both* ecosystems at once.

**Verdict:** partial, concentrated on the text-pipeline hooks (riding B). The
deep-inference tier is HF-backend-only and largely infeasible on localm's default
GGUF runtime; the Gradio UI tier does not port. Honest scope: support the
string-modifier extensions (translators, prompt templaters, memory injectors).

## Agent Skills

### Anatomy

A folder with a `SKILL.md` (YAML frontmatter: `name`, `description`, optional
`license`, `allowed-tools`, `metadata`) plus a markdown instruction body and
bundled scripts/resources. Loaded by **progressive disclosure**: the agent sees
`name` + `description` first, loads the body when relevant, then reads bundled
files on demand.

This is a *format*, not a runtime, and it maps to localm almost for free because
the coder agent is exactly the consumer a Skill expects: a system prompt + memory
+ file/shell tools.

### Mapping

| Skill element | localm mapping | Verdict |
|---|---|---|
| `name` + `description` (disclosure L1) | inject into the coder system prompt as available skills, or a `use_skill(name)` tool | clean |
| body instructions (L2) | inject `SKILL.md` body into agent context on activation | clean |
| bundled scripts/resources (L3) | the agent's existing `read_file` / `run_shell` read and run them on demand | clean, reuses existing tools |
| `allowed-tools` | map to localm capability scopes | clean |

**Verdict:** the cleanest and lowest-risk of the three. A `skills` plugin on the
**coder** surface that does progressive disclosure and lets the agent's
already-gated tools touch the bundled files. Lower risk than Open WebUI /
oobabooga: the agent *chooses* to run a bundled script through its existing
confirm/scope/audit, rather than the host `exec`-ing foreign Python. (For plain
non-agentic chat its value drops to a structured persona + context.)

## Reachable vs out of reach

| Ecosystem | Reachable | Out of reach |
|---|---|---|
| Open WebUI | Tools, Valves, Filters (via B), Pipes (via a virtual-model backend, later) | Actions' rich event-UI (frontend-coupled) |
| oobabooga | text-pipeline hooks (via B) | logits/tokenizer/custom_generate (HF-only; infeasible on GGUF), Gradio UI |
| Agent Skills | essentially all of it | (nothing significant) |

## Recommended build order

Leverage-weighted:

1. **Skills importer** - DONE (a coder-surface plugin, zero kernel change; see
   `docs/skills.md`).
2. **Chat-pipeline hook (B)** - the kernel hook is DONE. Next on top of it: the
   Open WebUI **Tools** adapter (reusing the tool-registry machinery) and the
   oobabooga / OWUI **text-hook** adapters (each translates a foreign filter's
   signature to `register_chat_hook`). One kernel piece, two ecosystems.
3. **Open WebUI Pipes** - a virtual-model backend; bigger, later.

## Security posture

Foreign code is untrusted. Every adapter runs it behind localm's existing
guarantees: capability scopes, consent on install, audit of actions, and
destructive-gating of side-effecting calls (the coder's confirm flow) - opt-in,
never default-on. This mirrors how the MCP client already treats untrusted
servers. Skills are the gentlest (the agent invokes bundled scripts via gated
tools); Open WebUI and oobabooga `exec` foreign Python directly and must be
treated accordingly, including consented `pip` of declared requirements into the
environment.

## Architectural placement

Each compatibility layer is **itself a localm plugin** (e.g. `openwebui-compat`,
`oobabooga-compat`, `skills`). The kernel only ever gains the two small primitives
(A is mostly present; B is worth building regardless). "Support ecosystem X" stays
an installable - the distro model applied to interop itself.

## To verify before building

- The exact Open WebUI load mechanism (DB source-string + `exec` vs file import)
  and which `open_webui.*` internals its plugins import - shapes the adapter's
  loader and how much of the dunder/event surface must be shimmed. Confirm against
  Open WebUI source, not docs.
- Which real-world extensions in each ecosystem actually stay within the
  "reachable" set, to size the payoff (a sample audit of the top community
  tools/extensions).
- ~~The chat-pipeline hook's placement in `/v1/chat/completions` and how it
  composes with the existing RAG/memory/web injection.~~ RESOLVED: the hook
  runs server-side in the kernel `/v1/chat/completions` path (inlet before
  inference, stream per piece, outlet on the reply), downstream of the engine's
  marker scrubbing. The existing RAG/memory/web injection is client-side in the
  SPA and is left in place; the hook is the new server-side seam alongside it.

## Sources

- Open WebUI: Tools and Functions plugin docs, Filter/Pipe function docs, Valves
  (docs.openwebui.com/features/extensibility/).
- oobabooga: text-generation-webui Extensions wiki and `extensions/example/script.py`.
- Agent Skills: anthropics/skills.
