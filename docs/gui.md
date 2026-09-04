# The localm web GUI

`localm gui` starts an inference server and opens a browser interface. No build step, no Node, no network dependency - the frontend is plain HTML/JS served by the same FastAPI process, with vendored libraries for markdown rendering and syntax highlighting.

Only three tabs are always present, outside the plugin system: **Chat** (the protected plugin #0), **Models**, and **Settings** - plus **Plugins** itself, the page that makes everything else appear. Every other tab - the image, music, and video studios, coder, Knowledge (RAG), Jobs, and so on - is contributed by a plugin and appears only when that plugin is installed and enabled. See [plugins.md](plugins.md) to learn more.

## Getting started

```bash
localm gui              # launch with the first registered model
localm gui mymodel      # launch with a named model
localm gui --no-browser # start the server only, no browser tab or app window
localm gui -p 8650      # explicit port (must be free, else startup errors)
localm gui --pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

1. **Launch.** Running `localm gui` starts the server and opens the GUI at http://127.0.0.1:8642 (or your configured port) - in its own app window if you installed that option, otherwise your default browser. See [native-app.md](native-app.md).
2. **Pick a model.** If a model is already registered, it loads in the background so the first reply is instant. Otherwise, land on the Models page to download or import one. You can also use `--pull <spec>` to begin downloading a HuggingFace model immediately with progress shown on the Models page.
3. **Type and reply.** Once a model is loaded, click in the composer and start typing. The model replies with streaming text, markdown rendering, and syntax highlighting on code blocks. Use `/` to see available commands and access slash features like `/web` (search and cite sources) and `/generate-image` (requires the image plugin).

**On your phone.** The GUI is a PWA: open your server's address on a phone and tap "Add to Home screen" to use localm as an installable app. See [phone.md](phone.md) for same-Wi-Fi and remote (Tailscale) setup.

**Starting with no models.** On a fresh install, `localm gui` still opens and lands on the Models page so you can pull or import a first model from the browser. The engine starts once you load one.

## Chat

Ask questions and have conversations with the model. The model can access documents, search the web, and remember facts across sessions (with the memory plugin - see [memory.md](memory.md)).

**Basic features:**
- Model selector in the sidebar switches between registered models. Switching unloads the old one and loads the new one.
- Streaming responses with markdown and syntax highlighting. Copy buttons on all messages and code blocks.
- Parameters drawer (click the settings icon) controls temperature and system prompt per conversation, plus web access, memory, and knowledge toggles. Top-p, top-k, repeat penalty, max tokens, seed, and a GBNF grammar box sit behind an **Advanced** fold so the drawer leads with what you actually touch; it opens itself if a persona or a restored setting fills one of those fields in.
- Search box (in the sidebar) filters chats by title and message content; hover to pin-to-top or move-to-folder. `/pin` and `/folder <name>` do the same from the composer.
- Branching: regenerating a reply keeps the old one as a variant. Editing a message forks the conversation instead of deleting what followed. A control in the message meta shows which branch you are viewing.
- A reply containing a self-contained HTML or SVG block gets a canvas button that renders it in a sandboxed frame with no network access and no access to the app. On by default; Settings > Model can turn it off entirely or restrict it to owner sessions (see Settings below).
- Session persistence follows your session mode: `privacy` (default, memory only, vanishes on reload), `log` (saved to `sessions/` in the data directory, survives reloads and server restarts), `full` (log plus markdown transcript). The page you were on is restored after reload (except in privacy mode).

**Commands:** Type `/` to open the command menu.
- `/web <query>` - search the web and answer with cited sources (requires the web plugin; see network.md).
- `/generate-image <prompt>` - generate an image inline (image plugin required).
- `/remember <fact>` - save a fact to your long-term memory so the model recalls it in later chats (requires the memory plugin).
- `/memory` - view or edit remembered facts in the memory manager.
- `/persona <name>` - apply a saved persona (system prompt + sampling values).
- `/clear` - clear this chat.
- `/compact` - compact old turns to free context.
- `/export` - download the conversation as markdown.
- `/rename <title>` - rename this chat.
- `/new` - start a new chat.
- `/system` - edit the system prompt.

**Documents:** Click the paperclip to attach PDFs, docx, text, and code files alongside images. They are converted to text in memory (nothing written to disk) and shown as a dimmed "Doc" message the model reads before your question.

**Knowledge:** Pick an indexed collection in the parameters drawer and every question is answered against the most relevant excerpts, cited as `[1]` (file + line). Collections are managed on the Knowledge page - see [rag.md](rag.md).

**Web access:** The "Web access" checkbox in the parameters drawer lets the model search and read pages on its own, mid-conversation (bounded rounds; every request and result is shown as a dimmed "Web" message). Off by default; without it, chat is fully offline. Both `/web` and the toggle run through the server's network policy - see [network.md](network.md).

**Remote images in replies:** A reply that links an image (`![alt](https://...)`) cannot load it by default - showing one is off. Turn "Show remote images in replies" on or to "ask" under Settings &rsaquo; Server & network &rsaquo; Outbound access; either way this machine fetches the image server-side, so the remote site never learns your IP or browser. On "ask" (the middle setting), the first image from a given site in a conversation prompts you to allow or refuse it; your answer is remembered for that site in that conversation only, never written to disk.

**Voice:**
- **Input:** The microphone button records from your device and transcribes locally with Whisper (needs `pip install "localm[voice]"`). The Whisper model is fetched exactly once (installing the voice plugin prefetches it), then everything is offline. That one fetch follows the network policy: under `net_mode=allow` it happens automatically on first use; under `ask` the mic is greyed out with the reason and offers a one-time download (needs the same permission that governs the network policy; nothing is written to settings); under `off` it is blocked entirely. Recordings are processed in memory, never written to disk.
- **Output:** The speaker button reads the reply aloud. By default this uses the browser's built-in `speechSynthesis` voices (no setup). Install and enable the tts plugin to upgrade to neural Kokoro voices synthesised entirely in the browser (the ~86 MB model is cached client-side, so no text leaves the machine). Voices are grouped by language and each carries a letter grade (e.g. "Heart (en-us, Female, A)") rating that voice's training data, not a version - explained inline in Settings.
- **Which voice:** Settings &rsaquo; Text-to-speech holds the server-side defaults every browser starts from (default voice, speaking speed, voice model, and an Advanced box for compute device and model precision). The **voice model** picker offers the shipped default plus a verified alternate with word-level timestamps, with a "Custom" option that still accepts any other Kokoro-compatible HuggingFace repo id. The Voice picker in the chat parameters drawer is a **this-browser** override: it changes what you hear here without changing the default for anyone else. When your browser has its own pick, the Settings section says so and offers to clear it.

**Memory and personas:**
- `/remember <fact>` saves a durable fact to your memory store (via the memory plugin); the 🧠 drawer toggle recalls the relevant facts into the system prompt each turn, and memory also grows on its own as you chat. Memory needs `log` or `full` mode: in privacy mode (the default) nothing is written and, unless you opt into read-only privacy recall, nothing is recalled. A "used N memories" chip on the reply shows what was recalled. See [memory.md](memory.md).
- Personas save the current system prompt and sampling values under a name (drawer → save), then apply them with `/persona <name>` or pick from the drawer select. Stored in `prompts.json` in the data directory.

**Usage and context:** The line under the composer shows total tokens, time to first token, and tokens per second for the last reply. The sidebar also shows model name, current session mode, and whether memory is enabled.

## Coder

Coder is a plugin: this tab and its routes appear only when the coder plugin is installed and enabled.

Pair with an AI agent on code tasks. Point it at a project directory and give it a goal - the agent reads files, writes code, runs tests, searches, and can call any MCP tools configured for that project.

**Start a session:** Click the "New session" button, pick a directory, and give the agent a task. The agent gets the same tools as the terminal version: reading and searching the project, writing and patching files, shell and tests, git, web fetch and search, image generation, Knowledge-collection search, background jobs, and sub-agent delegation, plus any MCP server tools, plugin-exported tools, and Agent Skills available in that project. The full list, including which tools ask before running, is in [cli.md](cli.md#built-in-tools). A session opened with a shared, non-owner key is restricted to a read-and-confined-edit subset: no shell, no git writes, no sub-agents.

**Common workflows:**
- **Build a feature:** "Add a login form to pages/auth.tsx that validates email and password" - the agent writes components, runs tests, and shows you diffs before applying changes.
- **Debug a crash:** "The API returns 500 on POST /users. Debug and fix it" - the agent reads logs, traces code, and proposes a fix.
- **Refactor:** "Replace all lodash imports with es6 equivalents" - the agent reads files, edits them, and runs tests to verify.
- **Add tests:** "Write test coverage for the billing module" - the agent reads code and writes comprehensive tests.

**What you see in the feed:**
- The agent's reasoning streams live.
- Every tool call shows its arguments and result (diffs for file writes). Click to expand; the result line shows the outcome and execution time.
- Destructive actions (file writes, shell commands) pause and show an approval card with a unified diff. Approve, reject, or tick **always allow <tool> this session**. Unanswered approvals time out (default 10 minutes, `localm config coder_confirm_timeout <seconds>`). Enable **auto-approve** at session start if you trust the task, or use **dry run** to preview without touching anything.
- You can type while the agent works: a message sent mid-task is queued and shown as *Queued*, injected at the next turn boundary as a steering note ("skip the tests", "add logging").
- Usage line shows tokens, turn number, and how full the model's context window is.

**Session management:**
- Multiple sessions run side by side; the dropdown switches between them. Reload reattaches to running sessions.
- **Files** button lists every file changed, with per-file cumulative diffs (original to current).
- **Undo** reverts the last file write. **Compact** frees context. **Export** offers the feed as markdown, or the last finished task's result as JSON (the same payload `localcoder --output-format json` prints). **Log** shows the JSONL audit trail (with a filter box).
- **Patch** appears for a patch-mode session and downloads the diff it captured. Reading it never consumes it, so you can take it as often as you like.
- **Estimate** (under the composer) plans the task you have typed without running it: one turn, no tools, nothing written, and the plan does not enter the conversation. Same as `localcoder --estimate`.
- **History** button lists audit logs from earlier log/full-mode sessions (surviving server restarts) and opens them in the log viewer.
- **Stop** halts the agent at the next safe point. **End session** terminates it. Sessions survive a page reload and keep showing their outcome.

**Setup:** The setup form accepts a model (switches the engine), max turns, temperature, a scope glob that confines file tools, custom instructions, and the toggles below. Session persistence follows the mode: `privacy` (default, nothing saved), `log` (JSONL audit trail), `full` (audit trail plus markdown transcript).

- **Verification command** is the exit-code oracle: the harness runs it before a turn that changed files may finish and reads its exit code, so the agent cannot declare a success the check disagrees with. Leave it blank to use the project's detected check, set **verification fix attempts** to bound the retries, or tick **skip verification** to run none. This is the same oracle `localcoder --until` uses for a one-shot task.
- **Patch mode** captures every file write as a unified diff instead of applying it; nothing is written to disk and the **patch** button downloads the result.
- **Seed** pins the sampler's RNG, so the same seed with the same model, prompt and settings reproduces the same output. Leave it blank for a fresh random seed each run.
- **Still confirm shell commands** only bites alongside auto-approve, which is what it carves an exception out of: file writes go through unattended while shell commands still stop for you. Same as `localcoder --interactive-confirm`.
- **Native tools API** asks the model server for the OpenAI-compatible `tools` protocol. localm's own server does not implement it and says so when you tick this; it uses grammar-constrained tool calls instead, which give the same guarantee.
- **Lessons** manages the episodic-memory lessons stored for the directory in the form. It has two tabs: **stored** (what the coder currently recalls) and **dropped** (what it has let go of, which can be brought back). Each lesson can be forgotten, each dropped one restored, and the whole set can be **erased** - that last one takes the recoverable copies too and cannot be undone, so it asks first. **Consolidate** asks the model to merge related lessons into one; it is opt-in and manual, every original is archived so a bad merge is reversible, and it tells you how many groups it left alone. Ids are shown because they are what `localcoder --forget-episode` and `--restore-episode` take.

**In-session commands:** Type `/` to open the coder command menu: `/undo`, `/files`, `/compact`, `/export`, `/log`, `/stop`, `/end`, `/help`.

**Circuit breaker:** If a tool fails 4 times in a row the agent stops with a circuit-breaker message instead of burning turns; the conversation stays intact so you can adjust and continue.

## Other pages

Of the pages below, **Models**, **Plugins**, and **Settings** are part of the core shell. The **Images**, **Music**, **Video**, **Knowledge**, **Jobs**, and **Browser** tabs are each contributed by a plugin and appear only when installed and enabled. Coder and Browser share one collapsible "Agent" group in the sidebar, the way the media tabs already do.

**Models:** Search HuggingFace or CivitAI for models - a source picker switches the whole search card between the two (empty query shows most downloaded for HuggingFace; format and type checkboxes narrow it). HuggingFace results show architecture family, parameter count, and a MoE badge, plus a "fits your VRAM" badge (compared against total VRAM, no torch required) - per quantization for a GGUF repo (expand it to see every file), for the whole repo for an HF one. CivitAI results show its own commercial-use, derivatives, and credit permissions instead of an SPDX license string (CivitAI does not use one) and each file's CivitAI safety-scan status; NSFW and legacy-format results are opt-in toggles, off by default, matching the CLI. A **curated shortcuts** dropdown fills in a known-good small model that pulls even with network access off; a **sha256** field verifies a single-file pull; a vision-projector (mmproj) picker offers a matching file alongside a GGUF pull; **register in place / copy into library / move into library** controls how a local folder or file is added.

**Browser:** Shows the page live as the coding agent drives it, and lists any address the network policy refused, along with the reason. A "Watch the agent" button appears on a coder session that has a browser page open. Off by default - see [network.md](network.md#the-browser-plugin) for what it can reach and [cli.md](cli.md#built-in-tools) for the tools themselves.

Registered models are tabbed by type - **All, LLMs, Embedding, Diffusion, Encoders, VAEs, LoRAs, Other** (Other catches vision projectors and anything localm could not classify) - each tab showing its count. "show other types here" merges the Other-tab models into All; "group by type" breaks the list into one section per type instead of one flat table. Switch the active engine (the "use" button shows "loading…" for the duration), add aliases or rename outright, inspect path/hash/size, change a model's recorded type inline, remove a model, or **unload all** loaded models at once. A model whose file went missing shows a "missing" badge and a **relocate** button that points the registry entry at the file's new location, without losing its aliases or hash. Sort by column (Name, Role, Source, Size, Modified - remembered across reloads). Search is lazy - no network request until you ask.

**Images:** Drive the local ComfyUI FLUX pipeline. Prompt and negative prompt up front; seed, guidance, img2img denoise, and an optional LoRA (picked from what is installed in your ComfyUI, with separate strength fields for the model and CLIP) sit behind an **Advanced** fold. History grid with metadata from sidecar files. If ComfyUI is not running, the job tells you how to start it, or starts it automatically if `comfy_launch_cmd` is set in the config. After generation, ComfyUI releases its models and the chat model reloads for instant replies.

**Music:** Generate tracks with the local ComfyUI ACE-Step workflow - style tags and optional lyrics ([verse]/[chorus] markers) up front, track length in seconds, and an **Advanced** fold for seed/steps/CFG. History with inline playback, move-to-folder, and delete. `/generate-music <tags>` in chat generates a default-length instrumental inline. Use `localm music "tags" --lyrics song.txt -d 180` from the terminal.

**Video:** Generate short clips with the local ComfyUI Wan 2.2 workflow - prompt, negative, duration (snapped to the model's frame rule; ~5 s is native) and optional start image (image-to-video) up front, with fps, resolution, and seed/steps/CFG behind an **Advanced** fold. Same VRAM handover as images. History with inline playback. `/generate-video <prompt>` in chat generates a ~5 s clip inline. Use `localm video "prompt"` from the terminal. Video is the slowest generator - see [docs/video.md](video.md) for model setup and timing expectations.

**Knowledge:** Create document collections, index files or folders with live progress, inspect/remove indexed documents, test-search a collection, and delete collections (index only - originals untouched). Collections show `hybrid` when embeddings are available, `BM25` otherwise. Manage collections from chat too - see [rag.md](rag.md).

**Jobs:** Schedule a chat prompt, a coder task, or a knowledge-collection re-sync to run on a schedule - every N hours, a daily or weekly time, or a custom interval in seconds or 5-field cron expression. Jobs list as a table with their schedule and last-run status; **Run now**, **Enable/Disable**, **Results** (browse past runs), and **Delete** per row. A scheduled job cannot run the coder agent with full shell access unless it was created by the owner - that opt-in is not exposed in the form itself. An in-app scheduler runs due jobs while the GUI or server is up. Manage jobs from the terminal with `localm job` - see [jobs.md](jobs.md).

**Plugins:** Browse the bundled store, install a plugin, then enable or disable it - all at runtime, no server restart. Installing copies the plugin into the installed folder; enabling mounts its routes, static assets, and tab onto the live app (disabling removes them). This page makes every plugin-contributed surface appear or disappear. See [plugins.md](plugins.md) for authoring.

**Settings:** Edit the server config and manage the running server - see [Settings](#settings) below. Light/dark theme toggle lives in the sidebar.

## Settings

Everything here is stored in `config.json` in the data directory; each section saves on its own, and a search box at the top matches a term against every section at once, whichever tab it lives under. Sections are grouped into seven tabs:

- **Model:** engine tuning (context size, GPU layers, timeouts), generation defaults (system prompt, sampling, and the preview-canvas toggles - on, and open to every session, by default; see Chat above), a **Library** card for optional HuggingFace and CivitAI API tokens (raise rate limits, reach gated models - neither required; see [cli.md](cli.md#search-and-pull-from-civitai)), model-library import depth, and embeddings (with a "warm up now" button). A **Live tuning** card applies without a restart: GPU layers and context window for the model that is currently loaded, which GPU a model loads onto, splitting a model across two or more GPUs (with an optional relative-weight input per checked GPU, so one card can take a larger share than another), a cap on how many models may stay resident at once, and model names pinned against eviction. An **Avatars** card sets your display name (shown next to your own messages instead of "You") and a small icon next to each chat turn: your own icon, a default model icon, and per-model overrides - each icon either a short emoji/glyph or a small uploaded image (downscaled and stored as a data URI client-side - it never becomes a URL, so nothing here makes a network request). With nothing set, a reply gets a deterministic monogram derived from the model's name.
- **Server & network:** bind address, port, TLS (a built-in certificate by default, or your own PEM pair; off serves plain HTTP), CORS origins, and the mDNS name your LAN uses to find this server. Binding past loopback without a strong API key is refused and the server stays on 127.0.0.1 - only the CLI's `--insecure` flag overrides that, with no Settings equivalent. Separate cards restart or shut down the server, list every other `localm gui`/`localm serve` running on this machine and stop one, and show the addresses (plus a pairing QR) to open on a phone - see [phone.md](phone.md). The Outbound access card's "Show remote images in replies" controls whether a model-linked image in a chat reply loads (see Chat above).
- **Security:** require an API key, mint a scope-limited key for another device or person (shown as a QR to scan, no typing - a `coder` key is read-and-confined-edit only, `coder (full)` and `admin` are owner-only), and roll, set, or remove the **owner key**, the one credential with full access (rolling it does not sign this browser out; every other device needs the new key to keep working; removing it asks first and returns the server to open mode).
- **Plugins:** per-plugin settings - coder, Knowledge, voice, text-to-speech, and anything a third-party plugin adds via `host.add_settings()`.
- **Media:** shared and per-generator settings for Images/Music/Video - localm's own managed ComfyUI install, or one you already run.
- **Privacy & data:** session persistence mode, memory settings, and a button to delete every stored conversation.
- **System:** appearance (sidebar logo style, an **interface language** picker - English or German so far, saved on the server so it follows the instance to every browser that connects, same as `localm config language de` - and a chat background image), the desktop app window mode (see [native-app.md](native-app.md)), reporting a bug, the changelog, exporting logs, and uploading files into the server's `uploads` folder, plus Updates and Diagnostics below.

**Updates** covers everything that changes what is installed: a new localm build and rolling back to the previous one, rebuilding the native launcher (after a Python upgrade, for instance), and the **inference runtime** - install a llama.cpp backend on a machine that has none yet, switch backends, pin an exact release tag, or roll back to the previous build. You always start it; a build that fails to load here is never kept.

**Diagnostics** runs the same five active self-checks `localm doctor` performs in a terminal - the llama.cpp library, the native ABI, spawning the worker process every model load needs, creating a nested venv, and the HF/transformers backend - in an isolated process so a check cannot crash the running server. About half a minute; nothing is installed or changed. Each check reports its own status pill; the aggregate verdict is scoped to these five checks, not a claim about the whole machine.

## Math rendering

Chat output renders LaTeX math offline via vendored KaTeX: `$inline$`, `$$display$$`, `\(...\)`, and `\[...\]`. Code blocks are excluded so source code with dollar signs is never mangled.

## Debug mode

```bash
localm gui --debug      # also available on serve and run
```

Debug mode writes a timestamped log to `<data dir>/logs/` containing every HTTP request with timing, and captures the native llama.cpp stderr stream (model loading details, KV cache messages, and crash abort reasons). Normal operation strips internal model markers from chat output - thinking-channel tags like `<|channel|>analysis` and reserved placeholder tokens that some finetunings emit. Debug mode shows them raw so model behaviour can be analysed.

If the server dies mid-generation, the native abort message at the end of that file says why.

## Security notes

- The server binds to 127.0.0.1 by default. CORS is locked to `localhost`/`127.0.0.1` origins, so a genuinely remote website you visit gets no CORS access at all and cannot call your API from browser JS. Another program ALSO running on localhost (a dev server, an npm postinstall page) is treated as trusted for CORS purposes, since it is on the same machine - it can use the OpenAI-compatible inference API (`/v1/chat/completions` and friends stay deliberately cross-origin callable for local apps) and unauthenticated reads (`/v1/models`, `/health`) by design, but it cannot drive any state-changing route or read management/metadata endpoints (keys, config, host stats, the filesystem browser) without a real API key - in open mode, the per-process shell token those need is gated on the SAME same-origin/allowlist check as writes, so a token seen in the page's own HTML cannot be replayed from another origin either.
- If `LOCALM_API_KEY` is set, the GUI prompts for the key once and exchanges it for an HttpOnly session cookie, so the key itself is never kept in browser-readable storage (localStorage or JS).
- Binding past loopback (e.g. `-H 0.0.0.0`) without an API key is refused: localm exits rather than expose the unauthenticated coder agent to the network. Set `LOCALM_API_KEY` first, or pass `--insecure` to override on a trusted, isolated network. On a network bind, traffic is TLS-encrypted by default (a built-in local-CA certificate). Even so, do not expose the GUI to a network you do not trust: the coder agent can write files and run shell commands on this machine.
- The same network bind, TLS on/off, and owner-key changes are also reachable from the GUI itself, in Settings, once you already hold the owner key or key-generation privilege - no terminal needed after the first key exists. See [Settings](#settings).

## How it fits together

```
browser (static JS)
   |  /v1/chat/completions      streaming chat
   |  /api/models               registry + engine switching
   |  /api/plugins              install / enable / disable; client-asset list
   |  /api/coder/sessions/*     session lifecycle, SSE events, approvals (coder plugin)
   v
localm FastAPI server -- Engine (GGUF/HF) -- your GPU
            |-- PluginManager (mounts each active plugin's routes + static assets)
            L-- coder Agent (per session, in-process thread; coder plugin)
```

The coder agent talks to the model through the server's own OpenAI-compatible endpoint, so chat and agent share one engine and inference is serialised cleanly between them.

The plugin engine ties the rest together. At startup, and again whenever you install, enable, or disable a plugin from the Plugins page, the **PluginManager** mounts each active plugin's API routes (scope-gated) and static assets onto the live FastAPI app, and unmounts them on disable - no server restart. This is how a client-asset plugin like **tts** can add behaviour (its Kokoro voice provider and voice picker) without contributing a tab. See [architecture.md](architecture.md) and [plugins.md](plugins.md) for how plugins load their client-side JS.
