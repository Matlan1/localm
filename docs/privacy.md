# Privacy modes and diagnostics

localm is offline-first and privacy-first: by default it saves nothing about your
conversations. This page lists every mode and switch that controls what localm
writes to disk, and exactly what each one does.

There are two independent dials:

1. **Session persistence mode** - what localm saves about your CHAT/CODER
   sessions (messages, replies, transcripts, memory).
2. **Diagnostics** - whether localm keeps OPERATIONAL logs (crash/hang traces,
   a debug log) to help diagnose a problem. Diagnostics never contain chat
   content in privacy mode (see the guarantee at the bottom).

---

## 1. Session persistence modes

Set globally (`mode`), per surface (`chat_mode`, `coder_mode`, which inherit the
global when unset), by the `--mode` flag on `localm gui` / `serve` / `run`, by the
`LOCALM_MODE` environment variable, in the desktop launcher's Privacy card, or in
the app under Settings > Privacy. Precedence: `LOCALM_MODE` env > `--mode` flag >
per-surface config > global config > **privacy** (the default). See
`localm/audit.py`.

| Mode | What it saves automatically |
| --- | --- |
| **privacy** (default) | **Nothing.** No session audit log, no transcript, no chat history on disk. Memory is not grown (no new facts written); existing memories are recalled only if you turn on "Allow memory recall in privacy mode". No automatic diagnostic traces either (see below). This is the "no traces" mode. |
| **log** | A JSONL audit trail of chat traffic under `<data dir>/sessions/` (one record per exchange). No markdown transcript. |
| **full** | Everything `log` does, PLUS a human-readable markdown transcript of each session. |

Notes:
- The mode is resolved per surface, so you can, for example, keep the coder in
  `log` while chat stays `privacy` (`coder_mode = log`).
- "Nothing written automatically" is the promise privacy mode makes; anything you
  do explicitly (saving a conversation, filing a bug report, `--debug`) is a
  deliberate action, not an automatic trace.

---

## 2. Diagnostics (crash/hang capture and the debug log)

These are orthogonal to the session mode: they control OPERATIONAL logs, not chat
content.

### The hang watchdog (`LOCALM_HANG_WATCHDOG`)

If the server's event loop ever freezes, an off-loop watchdog dumps every thread's
stack to `<home>/logs/hang_*.log` (created only when a real stall happens). Thread
stacks are code locations, not variable values - they never contain your prompts
or replies.

| `LOCALM_HANG_WATCHDOG` | Behaviour |
| --- | --- |
| unset (default) | On in `log`/`full` mode, and in `privacy` mode only when "Keep diagnostics" is on. Off in privacy mode otherwise (no automatic trace). |
| `0` / `false` / `off` | Off entirely. |
| `1` / `true` / `on` | Forced on regardless of mode, plus verbose asyncio slow-callback logging. |

`GET /debug/stacks` (loopback-only, owner-gated) returns the current thread stacks
and asyncio task list on demand, independent of the watchdog.

### The debug log (`--debug` / `LOCALM_DEBUG`)

`localm gui --debug` (or `LOCALM_DEBUG=1`) writes a debug log under
`<data dir>/logs/`, captures native llama.cpp stderr (model metadata/timings), and
logs each request (method, path, status, timing - never request bodies). It does
NOT record your chat content in privacy mode (see the guarantee below).

### "Keep diagnostics for bug reports" (`keep_diagnostics`)

Privacy mode saves nothing - which also means a hang or crash leaves nothing to
put in a bug report. This opt-in (off by default) keeps the diagnostic bits a
report needs, even in privacy mode:

- the hang watchdog trace,
- the crash/restart breadcrumb log (`pre_restart.log` + the in-memory activity
  ring), and
- a debug log (operational lines only).

Set it in **Settings > Privacy** (in-app, persistent), with the **desktop
launcher** checkbox, or with `--keep-diagnostics` on `localm gui` / `serve` (a
per-run override via `LOCALM_KEEP_DIAGNOSTICS`). In `log`/`full` mode diagnostics
are already kept, so this toggle only changes behaviour in privacy mode.

---

## The guarantee: no chat content in the debug log in privacy mode

Even with the debug log on (via `--debug` or "Keep diagnostics"), **your chat
messages and the model's replies are never written to the debug log in privacy
mode.** Content-bearing debug lines (the GGUF backend's raw model output, a
job's web-search query) are gated on `debug_content_enabled()` - debug on AND no
relevant session surface in privacy - which fails safe to "no content" if the
mode cannot be resolved. Operational lines (requests, timings, native metadata,
stack traces) are unaffected. In `log`/`full` mode, `--debug` still records raw
model output as before, since those modes already persist your sessions.
