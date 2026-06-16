# The localm web GUI

`localm gui` starts the inference server with a browser frontend on top. There is no build step, no Node, and no network dependency: the frontend is plain HTML/JS served by the same FastAPI process, with vendored libraries for markdown rendering and syntax highlighting.

Only two surfaces are always present: **Chat** (the protected plugin #0) and the **model manager** (the Models page). Every other tab is contributed by a plugin and is absent until that plugin is installed **and** enabled - the image, music, and video studios (grouped under a "studio" nav category), coder, Knowledge (the rag plugin), and so on. The **Plugins** page is where you install and enable them; see [plugins.md](plugins.md) for authoring. Throughout this page, sections other than Chat and Models describe surfaces that only appear once their plugin is active.

```bash
localm gui              # first registered model, opens your default browser
localm gui mymodel      # pick a model
localm gui --no-browser # just start the server, open the URL yourself
localm gui -p 8650      # explicit port (auto-bumps when busy)
localm gui --pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

The selected model preloads in a background thread at startup, so the first
reply does not pay the load cost.

**Starting with no models.** On a fresh install `localm gui` (no model
argument, empty registry) still opens - it lands on the Models page so you can
pull or import a first model from the browser; the engine starts once you load
one. `localm gui --pull <spec>` goes further and begins downloading `<spec>` (a
HuggingFace repo, `repo:file.gguf`, or an https URL) immediately, with progress
shown on the Models page. The graphical launcher's **Import** row drives the
same flows (file / folder / URL).

## Chat

- Typing `/` opens a command menu: `/imagine <prompt>` (generate an image
  inline - provided by the image plugin, so it is unavailable until that
  plugin is installed and enabled), `/web <query>` (search the web, answer with
  sources), `/clear`, `/compact`, `/export`, `/rename <title>`, `/system`,
  `/new`. Slash input is always handled by the UI, never sent to the model.
- Web access: `/web` grounds one answer in search results, and the "Web
  access" checkbox in the parameters drawer lets the model search and read
  pages on its own mid-conversation (bounded rounds; every request and result
  is shown as a dimmed "Web" message). Both run through the server's network
  policy - see [network.md](network.md). Off by default; without them chat is
  fully offline.
- Documents: the paperclip attaches PDFs, docx, text, and code files alongside
  images. They are converted to text in memory (nothing written to disk, so
  privacy mode stays clean) and shown as a dimmed "Doc" message the model
  reads before your question.
- Knowledge: pick an indexed collection in the parameters drawer and every
  question is answered against the most relevant excerpts, cited as `[1]`
  (file + line). Collections are managed on the Knowledge page - see
  [rag.md](rag.md).
- Model selector in the sidebar lists every registered model. Switching loads the new model and unloads the old one (the switch waits for any in-flight request to finish).
- Streaming responses with markdown and highlighted code blocks, copy buttons on messages and code.
- The parameters drawer sets temperature, top-p, max tokens, seed, and a system prompt per conversation.
- Personas: save the current system prompt + sampling values under a name
  (drawer → save…), then apply them from the drawer select or with
  `/persona <name>`. Stored in `prompts.json` in the localm data directory -
  explicit user assets, available in every session mode.
- Memory: `/remember <fact>` adds a line to `chat-memory.md` in the data
  directory; `/memory` views or edits the whole file; the 🧠 drawer toggle
  injects it into the system prompt so the model knows it across every chat.
  Privacy semantics: privacy mode blocks memory **writes** (no new traces)
  but still injects what earlier non-privacy sessions saved - privacy means
  no traces, not amnesia.
- Voice input: the 🎤 button records from the microphone and transcribes
  locally with Whisper into the composer (needs `pip install "localm[voice]"`;
  the model downloads once on first use, then everything is offline; recordings
  are processed in memory, never written to disk).
- Read-aloud: every reply has a 🔊 button, and the "Speak replies aloud" toggle
  reads each finished reply automatically. By default this uses the browser's
  built-in `speechSynthesis` voices (no setup at all). Installing and enabling
  the **tts plugin** upgrades it to neural **Kokoro** voices synthesised
  entirely in the browser: the ~86 MB model is fetched once and cached
  client-side, so no text ever leaves the machine and nothing is written to the
  server (privacy mode stays trace-free). The plugin adds a voice picker in the
  "Voice" settings group and has no tab of its own; `speechSynthesis` remains
  the fallback whenever the plugin is not installed or its model has not
  loaded.
- Conversation persistence follows the session mode. In `privacy` (the default) conversations live in memory only and vanish on reload. In `log`/`full` they are saved to `chats/` in the localm data directory (with localStorage as a cache), so they survive reloads, browser profile wipes, and server restarts. Deleting one removes it everywhere.
- The sidebar search filters chats by title **and** message content (content
  matches show a snippet). Hover a chat for 📌 pin-to-top and 📁 move-to-folder;
  folders are collapsible groups, and `/pin` and `/folder <name>` do the same
  from the composer. Pins and folders persist with the conversation.
- Branching: regenerating a reply keeps the old one as a variant, and editing
  a sent message forks the conversation instead of deleting what followed.
  A ‹ 2/3 › control in the message meta row switches between siblings at any
  fork point. Branches persist with the conversation; export and the model's
  context always use the currently selected branch, and forks anchored in
  history that gets compacted away are pruned.
- The page you were on (chat, coder, models, …) is restored after a reload - except in privacy mode, which leaves no trace of it.
- The usage line under the composer shows total tokens, time to first token, and tokens per second for the last reply.

## Coder

Coder is itself a plugin: this tab and its routes only appear once the coder
plugin is installed and enabled.

Start a session by pointing the agent at a project directory. The agent gets the same tools as the terminal version: read, write, edit, patch, shell, search, tests, image generation, plus any MCP tools configured for that project. It can also call tools exported by other installed plugins, registered under namespaced names like `plugin_<plugin>_<tool>` so they never collide with the built-ins.

What you see in the feed:

- The agent's reasoning streams live.
- Every tool call becomes a card showing its arguments (and the diff, for
  file writes - both at once). Click to expand; the result line shows the
  outcome and how long the tool took.
- Destructive actions (file writes, shell commands) pause the agent and show
  an approval card with a unified diff of exactly what would change. Approve
  or reject from the browser - or tick **always allow <tool> this session**
  on approval and that tool skips the flow for the rest of the session.
  Unanswered approvals time out (default 10 minutes, configurable with
  `localm config coder_confirm_timeout <seconds>`) and are rejected. Answered
  approvals keep showing their outcome - including after a page reload, and
  in other tabs attached to the same session.
- Auto-approve can be enabled at session start if you trust the task, and
  **dry run** makes destructive tools report what they would do without
  touching anything - a safe way to preview an agent's plan.
- **You can keep typing while the agent works**: a message sent mid-task is
  queued and injected at the next turn boundary as a steering note ("also add
  logging", "skip the tests"), shown in the feed with a *Queued* label.
- The usage line shows tokens, the turn number, and how full the model's
  context window is (`ctx 42%`).

Session persistence follows the coder's modes: `privacy` (default, nothing saved), `log` (JSONL audit trail), `full` (audit trail plus markdown transcript).

Stop asks the agent to halt at the next safe point. End session terminates it.

## Coder extras

- Multiple sessions can run side by side; the dropdown in the bar switches
  between them, and reloading the page reattaches to running sessions and
  replays their feeds.
- File-writing tool calls show a rendered diff in their card even under
  auto-approve.
- The bar's **files** button lists every file the session has changed, with
  per-file cumulative diffs (original → current, across all edits) and a
  full-session diff; the task-finished line in the feed says how many files
  changed.
- The bar's undo button reverts the last file write, compact summarises old
  turns to free context, **export** downloads the session feed as markdown,
  and log opens the JSONL audit trail (log/full modes) - with a filter box
  to narrow entries by tool name, type, or text.
- The history button (also "past sessions" on the setup form) lists the audit
  logs earlier log/full-mode sessions left behind - including sessions from
  before a server restart - and opens them in the same log viewer.
- Session setup accepts a model (switches the engine), max turns, temperature,
  a scope glob that confines file tools, and the dry-run toggle.
- Typing `/` opens the coder command menu (`/undo`, `/files`, `/compact`,
  `/export`, `/log`, `/stop`, `/end`, `/help`); commands run in the UI
  instead of being sent to the agent as a task.
- If a tool fails 4 times in a row the agent stops with a circuit-breaker
  message instead of burning its remaining turns; the conversation stays
  intact so you can adjust and continue.

## Other pages

Of the pages below, only **Models** and **Plugins** are part of the core shell.
The **Images**, **Music**, **Video**, and **Knowledge** tabs are each
contributed by a plugin and are absent until that plugin is installed and
enabled. Image, music, and video declare the same `"studio"` nav group, so the
SPA collapses them into a single **studio** category in the sidebar rather than
three top-level tabs; Knowledge is the rag plugin.

- **Models**: search HuggingFace for GGUF models right on the page (empty
  query shows the most downloaded), expand a repo to see every quantization
  with its size and a "fits your VRAM" badge (compared against total VRAM,
  measured via torch, nvidia-smi, or the Windows display-adapter registry -
  no torch required), and pull any file with one click. Plus: pull by spec
  with live progress, switch the active engine, add aliases, inspect
  path/hash/size, and remove models (alias-aware, never the active one).
  Search is lazy - no network request until you ask.
- **Images**: drive the local ComfyUI FLUX pipeline; prompt, negative prompt,
  seed, guidance, img2img with denoise; history grid with per-image metadata
  from the sidecar files. If ComfyUI is not running, the job says how to
  start it, or starts it automatically when `comfy_launch_cmd` is set in the
  config. After a successful generation, ComfyUI is asked to release its
  models and the chat model reloads so the next reply is instant.
- **Music**: generate tracks with the local ComfyUI ACE-Step workflow - style
  tags, optional lyrics ([verse]/[chorus] markers), and an **arbitrary track
  length in seconds**; seed/steps/CFG for control. Same VRAM handover as image
  generation. History with inline playback, move-to-folder, and delete.
  `/music <tags>` in chat generates a default-length instrumental inline, and
  `localm music "tags" --lyrics song.txt -d 180` does it from the terminal.
- **Video**: generate short clips with the local ComfyUI Wan 2.2 workflow -
  prompt + negative, duration (snapped to the model's frame rule; ~5 s is the
  native length), fps, resolution, seed/steps/CFG, and an optional start image
  (image-to-video). Same VRAM handover; history with inline playback,
  move-to-folder, and delete. `/video <prompt>` in chat generates a default
  ~5 s clip inline, and `localm video "prompt"` does it from the terminal.
  Video is the slowest generator - see [docs/video.md](video.md) for model
  setup and honest timing expectations.
- **Knowledge**: create document collections, index files or folders with live
  progress, inspect/remove indexed documents, test-search a collection, and
  delete collections (index only - original files untouched). Collections show
  `hybrid` when embeddings are available, `BM25` otherwise.
- **Plugins**: browse the bundled store, install a plugin, then enable or
  disable it - all at runtime, with no server restart. Installing copies the
  plugin into the installed folder; enabling mounts its routes, static assets,
  and any tab it contributes onto the live app (disabling removes them again).
  This is the page that makes every plugin-contributed surface on this list
  appear or disappear. See [plugins.md](plugins.md) for authoring.
- **Settings**: edit the server config (`~/.localm/config.json`) and the GUI's
  API key; light/dark theme toggle lives in the sidebar.

## Math rendering

Chat output renders LaTeX math offline via vendored KaTeX: `$inline$`,
`$$display$$`, `\(...\)`, and `\[...\]`. Code blocks are excluded so source
code with dollar signs is never mangled.

## Debug mode

```bash
localm gui --debug      # also available on serve and run
```

Debug mode writes a timestamped log to `~/.localm/logs/` containing every
HTTP request with timing, and captures the native llama.cpp stderr stream
(model loading details, KV cache messages, and crash abort reasons) that is
normally suppressed to keep chat output clean. If the server ever dies
mid-generation, the native abort message at the end of that file says why.

Normal operation also strips internal model markers from chat output -
thinking-channel tags like `<|channel|>analysis` and reserved placeholder
tokens that some finetunes emit as text. Debug mode shows them raw so model
behaviour can be analysed.

## Security notes

- The server binds to 127.0.0.1 by default and CORS is locked to localhost, so other websites you visit cannot call your API from browser JS.
- If `LOCALM_API_KEY` is set, the GUI prompts for the key once and stores it in localStorage.
- Binding to 0.0.0.0 without an API key triggers a CLI warning. Do not expose the GUI to a network you do not trust: the coder agent can write files and run shell commands on this machine.

## How it fits together

```
browser (static JS)
   │  /v1/chat/completions      streaming chat
   │  /api/models               registry + engine switching
   │  /api/plugins              install / enable / disable; client-asset list
   │  /api/coder/sessions/*     session lifecycle, SSE events, approvals (coder plugin)
   ▼
localm FastAPI server ── Engine (GGUF/HF) ── your GPU
            ├── PluginManager (mounts each active plugin's routes + static assets)
            └── coder Agent (per session, in-process thread; coder plugin)
```

The coder agent talks to the model through the server's own OpenAI-compatible endpoint, so chat and agent share one engine and inference is serialised cleanly between them.

The plugin engine ties the rest together. At startup, and again whenever you
install, enable, or disable a plugin from the Plugins page, the **PluginManager**
mounts each active plugin's API routes (scope-gated) and static assets onto the
live FastAPI app, and unmounts them on disable - no server restart. On the
browser side the SPA fetches `/api/plugins` and, for every active plugin that
declares a `client_entry`, `import()`s that JS module from `/plugins/<name>/`
and calls its `register(ctx)`. That is how a client-asset plugin like **tts**
adds behaviour (its Kokoro voice provider and voice picker) without contributing
a tab or any server-side code.
