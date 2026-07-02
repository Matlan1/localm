# The localm web GUI

`localm gui` starts an inference server and opens a browser interface. No build step, no Node, no network dependency - the frontend is plain HTML/JS served by the same FastAPI process, with vendored libraries for markdown rendering and syntax highlighting.

Only two surfaces are always present: **Chat** (the protected plugin #0) and the **Models** page. Every other tab - the image, music, and video studios, coder, Knowledge (RAG), and so on - is contributed by a plugin and appears only when that plugin is installed and enabled. See [plugins.md](plugins.md) to learn more.

## Getting started

```bash
localm gui              # launch with the first registered model
localm gui mymodel      # launch with a named model
localm gui --no-browser # start the server; open the URL yourself
localm gui -p 8650      # explicit port (auto-bumps when busy)
localm gui --pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

1. **Launch.** Running `localm gui` starts the server and opens your default browser to http://127.0.0.1:8642 (or your configured port).
2. **Pick a model.** If a model is already registered, it loads in the background so the first reply is instant. Otherwise, land on the Models page to download or import one. You can also use `--pull <spec>` to begin downloading a HuggingFace model immediately with progress shown on the Models page.
3. **Type and reply.** Once a model is loaded, click in the composer and start typing. The model replies with streaming text, markdown rendering, and syntax highlighting on code blocks. Use `/` to see available commands and access slash features like `/web` (search and cite sources) and `/generate-image` (if the image plugin is enabled).

**On your phone.** The GUI is a PWA: open your server's address on a phone and tap "Add to Home screen" to use localm as an installable app. See [phone.md](phone.md) for same-Wi-Fi and remote (Tailscale) setup.

**Starting with no models.** On a fresh install, `localm gui` still opens and lands on the Models page so you can pull or import a first model from the browser. The engine starts once you load one.

## Chat

Ask questions and have conversations with the model. The model can access documents, search the web, and remember facts across sessions (if enabled).

**Basic features:**
- Model selector in the sidebar switches between registered models. Switching unloads the old one and loads the new one.
- Streaming responses with markdown and syntax highlighting. Copy buttons on all messages and code blocks.
- Parameters drawer (click the settings icon) controls temperature, top-p, max tokens, seed, and system prompt per conversation.
- Search box (in the sidebar) filters chats by title and message content; hover to pin-to-top or move-to-folder. `/pin` and `/folder <name>` do the same from the composer.
- Branching: regenerating a reply keeps the old one as a variant. Editing a message forks the conversation instead of deleting what followed. A control in the message meta shows which branch you are viewing.
- Session persistence follows your session mode: `privacy` (default, memory only, vanishes on reload), `log` (saved to `sessions/` in the data directory, survives reloads and server restarts), `full` (log plus markdown transcript). The page you were on is restored after reload (except in privacy mode).

**Commands:** Type `/` to open the command menu.
- `/web <query>` - search the web and answer with cited sources (requires the web plugin; see network.md).
- `/generate-image <prompt>` - generate an image inline (image plugin required).
- `/remember <fact>` - save a fact to `chat-memory.md` in the data directory so the model knows it across every chat.
- `/memory` - view or edit the full memory file.
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

**Voice:**
- **Input:** The microphone button records from your device and transcribes locally with Whisper (needs `pip install "localm[voice]"`). The Whisper model downloads once on first use, then everything is offline. Recordings are processed in memory, never written to disk.
- **Output:** The speaker button reads the reply aloud. By default this uses the browser's built-in `speechSynthesis` voices (no setup). Install and enable the tts plugin to upgrade to neural Kokoro voices synthesised entirely in the browser (the ~86 MB model is cached client-side, so no text leaves the machine).

**Memory and personas:**
- `/remember <fact>` adds a line to `chat-memory.md` in the data directory; the 🧠 drawer toggle injects it into the system prompt so the model knows it across every session. In privacy mode, writes are blocked (no traces) but the model still injects previously saved facts.
- Personas save the current system prompt and sampling values under a name (drawer → save), then apply them with `/persona <name>` or pick from the drawer select. Stored in `prompts.json` in the data directory.

**Usage and context:** The line under the composer shows total tokens, time to first token, and tokens per second for the last reply. The sidebar also shows model name, current session mode, and whether memory is enabled.

## Coder

Coder is a plugin: this tab and its routes appear only when the coder plugin is installed and enabled.

Pair with an AI agent on code tasks. Point it at a project directory and give it a goal - the agent reads files, writes code, runs tests, searches, and can call any MCP tools configured for that project.

**Start a session:** Click the "New session" button, pick a directory, and give the agent a task. The agent gets the same tools as the terminal version: read, write, edit, patch, shell, search, tests, image generation, plus any MCP tools configured in `.localcoder/config.toml` for that project.

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
- **Undo** reverts the last file write. **Compact** frees context. **Export** downloads the feed as markdown. **Log** shows the JSONL audit trail (with a filter box).
- **History** button lists audit logs from earlier log/full-mode sessions (surviving server restarts) and opens them in the log viewer.
- **Stop** halts the agent at the next safe point. **End session** terminates it. Sessions survive a page reload and keep showing their outcome.

**Setup:** The setup form accepts a model (switches the engine), max turns, temperature, a scope glob that confines file tools, and the dry-run toggle. Session persistence follows the mode: `privacy` (default, nothing saved), `log` (JSONL audit trail), `full` (audit trail plus markdown transcript).

**In-session commands:** Type `/` to open the coder command menu: `/undo`, `/files`, `/compact`, `/export`, `/log`, `/stop`, `/end`, `/help`.

**Circuit breaker:** If a tool fails 4 times in a row the agent stops with a circuit-breaker message instead of burning turns; the conversation stays intact so you can adjust and continue.

## Other pages

Of the pages below, only **Models** and **Plugins** are part of the core shell. The **Images**, **Music**, **Video**, **Knowledge**, and **Jobs** tabs are each contributed by a plugin and appear only when installed and enabled.

**Models:** Search HuggingFace for GGUF models (empty query shows most downloaded). Expand a repo to see every quantization with its size and a "fits your VRAM" badge (compared against total VRAM, no torch required). Pull any file with one click; pull by spec with live progress, switch the active engine, add aliases, inspect path/hash/size, and remove models. Search is lazy - no network request until you ask.

**Images:** Drive the local ComfyUI FLUX pipeline. Prompt, negative prompt, seed, guidance, img2img with denoise. History grid with metadata from sidecar files. If ComfyUI is not running, the job tells you how to start it, or starts it automatically if `comfy_launch_cmd` is set in the config. After generation, ComfyUI releases its models and the chat model reloads for instant replies.

**Music:** Generate tracks with the local ComfyUI ACE-Step workflow - style tags, optional lyrics ([verse]/[chorus] markers), and arbitrary track length in seconds. Seed/steps/CFG for control. History with inline playback, move-to-folder, and delete. `/generate-music <tags>` in chat generates a default-length instrumental inline. Use `localm music "tags" --lyrics song.txt -d 180` from the terminal.

**Video:** Generate short clips with the local ComfyUI Wan 2.2 workflow - prompt, negative, duration (snapped to the model's frame rule; ~5 s is native), fps, resolution, seed/steps/CFG, and optional start image (image-to-video). Same VRAM handover as images. History with inline playback. `/generate-video <prompt>` in chat generates a ~5 s clip inline. Use `localm video "prompt"` from the terminal. Video is the slowest generator - see [docs/video.md](video.md) for model setup and timing expectations.

**Knowledge:** Create document collections, index files or folders with live progress, inspect/remove indexed documents, test-search a collection, and delete collections (index only - originals untouched). Collections show `hybrid` when embeddings are available, `BM25` otherwise. Manage collections from chat too - see [rag.md](rag.md).

**Jobs:** Schedule a chat or coder prompt to run on a repeating schedule (every N seconds or a 5-field cron expression). Create, enable/disable, edit, run-now, and delete jobs; each run records a result you can browse. An in-app scheduler runs due jobs while the GUI or server is up. Manage jobs from the terminal with `localm job` - see [jobs.md](jobs.md).

**Plugins:** Browse the bundled store, install a plugin, then enable or disable it - all at runtime, no server restart. Installing copies the plugin into the installed folder; enabling mounts its routes, static assets, and tab onto the live app (disabling removes them). This page makes every plugin-contributed surface appear or disappear. See [plugins.md](plugins.md) for authoring.

**Settings:** Edit the server config (`~/.localm/config.json`) and the GUI's API key. Light/dark theme toggle lives in the sidebar.

## Math rendering

Chat output renders LaTeX math offline via vendored KaTeX: `$inline$`, `$$display$$`, `\(...\)`, and `\[...\]`. Code blocks are excluded so source code with dollar signs is never mangled.

## Debug mode

```bash
localm gui --debug      # also available on serve and run
```

Debug mode writes a timestamped log to `~/.localm/logs/` containing every HTTP request with timing, and captures the native llama.cpp stderr stream (model loading details, KV cache messages, and crash abort reasons). Normal operation strips internal model markers from chat output - thinking-channel tags like `<|channel|>analysis` and reserved placeholder tokens that some finetunings emit. Debug mode shows them raw so model behaviour can be analysed.

If the server dies mid-generation, the native abort message at the end of that file says why.

## Security notes

- The server binds to 127.0.0.1 by default and CORS is locked to localhost, so other websites you visit cannot call your API from browser JS.
- If `LOCALM_API_KEY` is set, the GUI prompts for the key once and exchanges it for an HttpOnly session cookie, so the key itself is never kept in browser-readable storage (localStorage or JS).
- Binding past loopback (e.g. `-H 0.0.0.0`) without an API key is refused: localm exits rather than expose the unauthenticated coder agent to the network. Set `LOCALM_API_KEY` first, or pass `--insecure` to override on a trusted, isolated network. On a network bind, traffic is TLS-encrypted by default (a built-in local-CA certificate). Even so, do not expose the GUI to a network you do not trust: the coder agent can write files and run shell commands on this machine.

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

The plugin engine ties the rest together. At startup, and again whenever you install, enable, or disable a plugin from the Plugins page, the **PluginManager** mounts each active plugin's API routes (scope-gated) and static assets onto the live FastAPI app, and unmounts them on disable - no server restart. On the browser side the SPA fetches `/api/plugins` and, for every active plugin that declares a `client_entry`, `import()`s that JS module from `/plugins/<name>/` and calls its `register(ctx)`. That is how a client-asset plugin like **tts** adds behaviour (its Kokoro voice provider and voice picker) without contributing a tab or any server-side code.
