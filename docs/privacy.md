# Privacy modes and diagnostics

localm is offline-first and privacy-first: by default it saves nothing about your
conversations. This page lists every mode and switch that controls what localm
writes to disk, and exactly what each one does.

There are two independent dials for what gets WRITTEN, plus one network
behaviour covered separately below:

1. **Session persistence mode** - what localm saves about your CHAT/CODER
   sessions (messages, replies, transcripts, memory).
2. **Diagnostics** - whether localm keeps OPERATIONAL logs (crash/hang traces,
   a debug log) to help diagnose a problem. Diagnostics never contain chat
   content in privacy mode (see the guarantee at the bottom).
3. **Update checks** - whether localm phones its update server; not a
   persistence mode, but still something that leaves your machine.

Bug reports (section 3 below) are not a fourth dial - they are the one
deliberate action that pulls diagnostics and a config snapshot together into
text meant to leave your machine, so what gets redacted from one is its own
section.

---

## 1. Session persistence modes

Set globally (`mode`), per surface (`chat_mode`, `coder_mode`, which inherit the
global when unset), by the `--mode` flag on `localm gui` / `serve` / `run`, by the
`LOCALM_MODE` environment variable, in the desktop launcher's Privacy card, or in
the app under Settings > Privacy. The `--mode` flag works by setting the
`LOCALM_MODE` environment variable for the process (and its children), so the two
are one precedence level, not two. Precedence: `LOCALM_MODE` env (direct, or via
`--mode`) > a project's `.localcoder/config.toml` `mode` (coder surface only) >
per-surface config (`chat_mode` / `coder_mode`) > global config (`mode`) >
**privacy** (the default). See `localm/audit.py`.

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
- In privacy mode, the interactive `localm run` chat and the coder REPL also
  suppress Python's own `readline` history, so what you type is not left
  behind in `~/.python_history` either. This is separate from and in addition
  to not writing a session transcript.

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

`GET /debug/stacks` (loopback-only) returns the current thread stacks and
asyncio task list on demand, independent of the watchdog. It needs full host
filesystem access - the owner key, or a key explicitly granted it - and in open
mode (no key configured) it additionally requires the per-process shell token
the loopback GUI shell holds, because open mode grants every caller host access
and so would otherwise leave this unauthenticated. Directory prefixes in the
returned frames (your data directory, the install location, the Python
environment) are redacted; the file, line and function are kept.

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
launcher** checkbox, with `--keep-diagnostics` on `localm gui` (`serve` and
`run` have no such flag - use the `LOCALM_KEEP_DIAGNOSTICS` environment
variable for those, a per-run override either way). The hang watchdog trace
and the crash/restart breadcrumb log are already kept automatically in
`log`/`full` mode; the debug log is not - it always needs `--debug` or this
toggle, in any session mode. So in privacy mode the toggle turns on all
three; in `log`/`full` mode it only adds the debug log.

---

## 3. Bug reports: what's redacted, and what needs your OK

Filing a bug report - Settings > Report a problem in the GUI, `localm
bug-report` at a terminal, or the standalone `report-issue.bat` /
`report-issue.sh` (repo root) that works even when localm cannot start at
all - is one of the deliberate actions above, but it is worth its own
section because it is the one path that pulls together diagnostics, recent
logs and your config into text meant to leave your machine. The standalone
reporter runs `scripts/report_issue.py`, or its no-Python PowerShell
equivalent (`report_issue.ps1`) when even that cannot run. It needs a git
clone of the repo, so it is not available after `pip install localm`; if
localm will not start at all on a pip install, file an
[Issue](https://github.com/Matlan1/localm/issues) directly instead, since
`localm bug-report` itself needs a working localm to run.

**You always see the full report before anything is sent, and sending needs
your explicit OK.** The GUI shows the assembled text and requires you to
press Send. The standalone reporter and `localm bug-report` preview the
same text at a terminal prompt - and when there is no terminal to prompt
(piped/redirected stdin, or run non-interactively), both the Python and
PowerShell standalone reporters now treat that exactly like you said no: the
report is saved locally and nothing is sent. This was a real gap: the
PowerShell fallback used to fall through to sending on its own when it had
no way to show the confirmation prompt at all.

**What is redacted from a report:**
- Your account name and home directory, in every textual form the report
  might quote it in (a plain path, a `repr()`'d or JSON-encoded one with
  doubled backslashes, or a path embedded in a native crash traceback's own
  message) - the backstop regex runs unconditionally, so one unresolvable
  path lookup can't leave the raw text unscrubbed.
- URL credentials (`user:pass@...`), credential-named query parameters and
  HTTP header lines (`api_key=`, `token=`, `X-Api-Key:`, `Auth-Token:`, ...),
  and bearer / API-key-shaped tokens - so `Authorization: Bearer ...` is
  caught, but a raw or Basic-auth `Authorization:` value is not, wherever
  they appear in diagnostic text, recent log tails, or the activity ring.
- Config values are sent only from a small allowlist of operational keys
  (port, context size, GPU layer count, mode, ...); the API key itself is
  never stored in config and never included.

**What a report deliberately keeps**, because it is what makes the report
useful to debug: your install location, data directory path, native runtime
library names, dependency versions, the loaded model's name and backend, and
a tail of the crashed run's own log. Only the account name is stripped from
these, not the whole path - unlike an HTTP response handed to a
lower-privileged API caller, which strips the install/data-dir paths too.

None of this is affected by session persistence mode: filing a report is the
same deliberate action, and gets the same scrubbing, whether you are in
`privacy`, `log`, or `full` mode.

---

## 4. Update checks (network policy, not a persistence mode)

Separate from both dials above: localm periodically asks its update server whether
a newer release exists. This is a network behaviour, not something that writes to
disk, so it isn't one of the two dials - it's covered by the [network
policy](network.md) instead.

- **When it fires.** A quiet check at most once every 6 hours while the GUI is
  open (a client-side throttle - there is no background daemon; a headless
  `localm serve` that nobody opens the GUI on never checks), plus whenever you
  press "Check for updates" in the GUI or run `localm update` / `localm update
  --check`.
- **What it sends, and what it does not.** A bare GET request identifying itself
  as `localm`, plus a shared-secret header if one is configured. It does **not**
  send your installed version, chat content, or anything else - the comparison
  to the latest release happens locally, on your machine. What it does expose to
  the update server is your IP address, the time you checked, and the fact that
  a localm install checked in.
- **It obeys `net_mode`.** Like every other outbound request, the update check
  goes through the network policy (`localm/netpolicy.py`) - setting network
  access to `off` blocks it too, and it fails honestly (never a false "you are
  up to date") when blocked. Turn on "Check for updates even when network
  access is off" in **Settings > Updates** to exempt just this one channel;
  it is off by default, so `net_mode=off` is a real kill switch unless you
  opt back in.
- **Turning the channel off entirely.** Clear the update endpoint
  (`update_url` and `bugreport_upload_url` both blank) to disable update
  checks regardless of network policy.

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
