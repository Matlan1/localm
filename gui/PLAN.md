# localm GUI — Implementation Plan

**Stack:** Tauri 2 + Svelte 5 + TypeScript + Tailwind CSS + shadcn-svelte  
**Location:** `gui/` directory inside the localm repo  
**Branch:** `feat/gui`

---

## Why this stack

- Tauri 2 wraps a native OS window (WebView2 on Windows) — the result is a standalone `.exe`, not a browser tab
- The existing FastAPI server (`localm serve`) stays as-is; the GUI connects over HTTP and WebSocket
- Svelte 5 compiles to minimal vanilla JS; no virtual DOM overhead
- shadcn-svelte gives us unstyled-but-accessible components we can theme exactly how we want
- Tailwind handles layout and spacing without a custom CSS mess

---

## Architecture

```
┌─────────────────────────────────┐
│   Tauri window (WebView2)       │
│   Svelte 5 SPA                  │
│                                 │
│  Chat │ Coder │ Plugins │ Settings │
└──────┬──────────────────────────┘
       │  HTTP + WebSocket
┌──────▼──────────────────────────┐
│   localm FastAPI server         │
│   (started as Tauri sidecar)    │
│                                 │
│  /v1/chat/completions (stream)  │
│  /v1/models                     │
│  plugin & skill management API  │
└─────────────────────────────────┘
```

Tauri manages the Python process via the `sidecar` plugin. On first launch it starts `localm serve`; on window close it shuts it down. If a server is already running on the configured port, Tauri skips the launch and connects to the existing instance.

---

## Phases

### Phase 1 — Scaffold

- [ ] `gui/` directory: `npm create tauri-app` with Svelte + TypeScript template
- [ ] Tailwind CSS + shadcn-svelte set up
- [ ] Basic layout shell: sidebar navigation + main content area
- [ ] Tauri sidecar config: start/stop `localm serve` with the window
- [ ] Health check on startup — wait for server ready before showing UI
- [ ] Settings stored in Tauri's app data dir (port, model path, GPU layers)

### Phase 2 — Chat interface

- [ ] Model selector (calls `/v1/models`)
- [ ] Conversation view: message bubbles, streaming tokens via `EventSource` or WebSocket
- [ ] Markdown rendering with syntax-highlighted code blocks (`marked` + `highlight.js` or `shiki`)
- [ ] System prompt editor (collapsible)
- [ ] Parameter panel: temperature, top-p, max tokens, grammar (GBNF) field
- [ ] Conversation history: save/load named sessions to local JSON files
- [ ] Stop generation button
- [ ] Copy message / copy code block buttons

### Phase 3 — Coder interface

- [ ] Launch a `localcoder` agent session (spawns the agent process, streams output)
- [ ] Task input: multiline editor with submit
- [ ] Output panel: tool calls rendered as expandable cards (tool name, args, result)
- [ ] File diffs displayed inline when `write_file` / `edit_file` / `patch_file` runs
- [ ] Working directory picker
- [ ] Model selector (separate from chat — coder typically uses a different model)
- [ ] `--yes` / `--interactive-confirm` toggles
- [ ] Session log viewer (reads the JSONL audit log)

### Phase 4 — Plugin and skills manager

New FastAPI endpoints needed in `localm`:
- `GET /v1/plugins` — list installed plugins with metadata
- `POST /v1/plugins/install` — install from local path or git URL
- `DELETE /v1/plugins/{name}` — uninstall
- `GET /v1/skills` — list skills across all plugins
- `GET /v1/skills/{name}` — skill detail (SKILL.md content)

UI:
- [ ] Plugin list: name, description, version, enabled toggle
- [ ] Install dialog: path input or git URL; progress indicator
- [ ] Skill browser: searchable list, click to view SKILL.md rendered as markdown
- [ ] LOCALCODER.md editor per project (per-project memory)

### Phase 5 — Settings and polish

- [ ] Settings page: server port, model path, n_ctx, n_gpu_layers, default model
- [ ] Theme: dark by default, light mode toggle
- [ ] System tray icon: show/hide window, server status indicator
- [ ] Auto-start server with OS login (optional, off by default)
- [ ] Window state persistence (size, position)
- [ ] Keyboard shortcuts: Cmd/Ctrl+K for command palette, Cmd/Ctrl+Enter to send
- [ ] Image generation panel (optional Phase 5+): connects to ComfyUI at port 8188

---

## File layout

```
gui/
  src-tauri/           Tauri Rust project
    src/
      main.rs          window setup, sidecar management
    tauri.conf.json
    Cargo.toml
  src/                 Svelte frontend
    lib/
      api.ts           typed wrappers for localm HTTP API
      stream.ts        SSE streaming helper
      store.ts         Svelte stores (model, settings, sessions)
    routes/
      +layout.svelte   sidebar + nav shell
      chat/            Phase 2
      coder/           Phase 3
      plugins/         Phase 4
      settings/        Phase 5
    components/
      MessageBubble.svelte
      ToolCallCard.svelte
      FileDiff.svelte
      ModelSelector.svelte
      MarkdownBlock.svelte
  package.json
  tailwind.config.ts
  vite.config.ts
```

---

## New localm server endpoints needed

Beyond what exists today:

| Method | Path | Notes |
|--------|------|-------|
| GET | `/v1/plugins` | list with metadata |
| POST | `/v1/plugins/install` | body: `{source: "path or url"}` |
| DELETE | `/v1/plugins/{name}` | uninstall |
| GET | `/v1/skills` | flat list across plugins |
| GET | `/v1/skills/{name}` | SKILL.md content |
| GET | `/v1/config` | current server config |
| PATCH | `/v1/config` | update and persist |
| POST | `/v1/coder/sessions` | start a coder agent session |
| GET | `/v1/coder/sessions/{id}/stream` | SSE stream of agent output |
| POST | `/v1/coder/sessions/{id}/input` | send REPL input mid-session |
| DELETE | `/v1/coder/sessions/{id}` | terminate session |

---

## Prerequisites before starting

- Node.js 20+ and Rust toolchain installed
- `npm install -g @tauri-apps/cli`
- WebView2 runtime (ships with Windows 11; installer available for Win 10)
