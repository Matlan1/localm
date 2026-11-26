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
the app under Settings > Privacy & data. The `--mode` flag works by setting the
`LOCALM_MODE` environment variable for the process (and its children), so the two
are one precedence level, not two. Precedence: `LOCALM_MODE` env (direct, or via
`--mode`) > a project's `.localcoder/config.toml` `mode` (coder surface only) >
per-surface config (`chat_mode` / `coder_mode`) > global config (`mode`) >
**privacy** (the default). See `localm/audit.py`.

| Mode | What it saves automatically |
| --- | --- |
| **privacy** (default) | **Nothing automatic about your conversations** (see "What privacy mode does not cover" below). No session audit log, no transcript, no chat history on disk. Memory is not grown (no new facts written); existing memories are recalled only if you turn on "Allow memory recall in privacy mode". No automatic diagnostic traces either (see below). This is the "no traces" mode. |
| **log** | A JSONL audit trail of chat traffic under `<data dir>/sessions/` (one record per exchange). No markdown transcript. |
| **full** | Everything `log` does, PLUS a human-readable markdown transcript of each session. |

Notes:
- The mode is resolved per surface, so you can, for example, keep the coder in
  `log` while chat stays `privacy` (`coder_mode = log`).
- The chat page in the app and `localm run` save conversations by the chat mode,
  and a coder session saves its own log and transcript by the coder mode.
- The server's own audit log and transcript (`*_server.jsonl` / `*_server.md`)
  record a chat or coder request under the more private of the global mode and
  that surface's mode (for the coder, the session's own mode, including a
  project's `.localcoder/config.toml`). So `chat_mode = privacy` keeps chat out
  of the server's records even when `mode = full`, while `chat_mode = full`
  under `mode = privacy` saves conversations but adds nothing to the server's
  records. Any other client of the server's API is recorded under the global
  mode. The surface's mode applies only to requests made with the owner key (or
  in open mode); a scoped key's requests are recorded under the global mode,
  and the chat page opened with one follows the global mode too.
- Images, music and video made from the chat page (`/imagine`, `/generate-music`,
  `/generate-video`) follow the same rule: under `chat_mode = privacy` they are
  not kept (see "Generated images, music and video" below), even when the global
  mode keeps them. Under `chat_mode = log` or `full` with `mode = privacy` they
  stay in the gallery with the saved chat, with no prompt sidecar. The automatic diagnostics the server decides on for such a request
  and its job are off unless you turned on `keep_diagnostics`: the finished job
  is not kept in the activity record, and a chat model the request or job loads,
  including the one reloaded after the job, gets no crash trace. A model or
  media backend already running when the request arrives keeps the crash trace
  it started with, and the server's hang watchdog follows the global mode.
  A failed `localm run` follows the chat mode for its bug report.
- "Nothing written automatically" is the promise privacy mode makes; anything you
  do explicitly (saving a conversation, filing a bug report, `--debug`) is a
  deliberate action, not an automatic trace.
- Changing the mode does not move anything across it. Switching the chat to
  `privacy` does not delete conversations or session records saved earlier (the
  app's chat sidebar says how many saved chats are still on disk). Switching from
  `privacy` to `log` or `full` while the app is open does not save the
  conversations held under privacy mode: they stay in that browser tab only, and
  a message sent from one of them is still handled in privacy mode (with a scoped
  key, whose requests follow the global mode, it is recorded under that mode).
- The chat page reads the mode from the server before it writes anything to the
  browser's storage, so a mode changed from a terminal applies to the next thing
  the page would save. While the server cannot be reached, or does not answer
  within 4 seconds, the last mode it reported stands.
- In `log` and `full` mode the browser's storage holds a copy of recent
  conversations and what you have typed, attached or queued but not sent. Each
  entry belongs to one install and one key: another install at the same address
  or another key does not see, save or remove it. Signing out, or signing in with
  another key, removes what was kept for the old one; the other open tabs drop
  what they hold and write none of it again. When the server no longer accepts
  the key, an open tab also drops what it holds and writes nothing more; what was
  kept before waits for the next sign-in. Privacy mode keeps none of it there and
  empties what an earlier mode left; if the browser refuses that or does not
  answer, the app says so and tries again until it succeeds.
- When you delete a conversation in `log` or `full` mode, its id (a short code
  made from the time it was started, no title or text) is added to
  `chats/deleted-ids.txt`, so that another browser that still holds the
  conversation does not save it again. If the file then cannot be removed, the
  entry is taken back and the app says that the chat was not deleted. A
  conversation whose file is missing
  without such an entry is saved again from a browser that holds it. The newest
  5000 to 10000 ids are kept.
- In privacy mode, the interactive `localm run` chat and the coder REPL also
  suppress Python's own `readline` history, so what you type is not left
  behind in `~/.python_history` either. This is separate from and in addition
  to not writing a session transcript.

### Generated images, music and video

In privacy mode a generated image, track or clip is not kept unless you choose to:

- **Keep in privacy mode** (`privacy_keep`, in Settings > Privacy & data beside the
  session modes, or `localm config privacy_keep media`; `localm config privacy_keep ""`
  turns it off again) is empty by default. With `media` in it, generated media stays in
  its History in privacy mode as well, without the prompt sidecar. Only the owner can
  change it. More kinds of data may join it later.
- **Off (the default).** The result is generated into a folder of the running localm
  process, `<data dir>/run/private-media/`, readable by your account only. While that
  process runs it can be seen, played, downloaded, renamed, moved to a folder, used as
  an input and upscaled; the Images, Music and Video pages list such results under the
  page's result, not in History, and **Remove all now** deletes them at once. No
  sidecar, owner index or other record of them is written, and the browser is told not
  to store them. With several keys, a result can be seen, saved and used as an input only
  by the key that made it and the owner; one made before any key existed is the owner's
  once there is one. The folder is deleted when localm stops or
  restarts; a generation still running then whose result is not to be kept is cancelled
  first. Any other generation is not: a job in a ComfyUI you run goes on there, and the
  built-in backends end with localm. After a crash, a killed
  process or a closed console window it is deleted at the next start (`localm gui`,
  `serve`, `run`, `mcp`, or `image`, `music` and `video` in privacy mode), and so is what a generation that does not end within 12
  seconds of a stop writes after it. A file another program holds open at
  that moment is reported in the log and tried again at the next start. Files are
  deleted, not overwritten: recovery tools may find them on the disk.
- On the pages, in the chat page and for `POST /v1/images/generations`, whether a result
  is kept is decided when it is written: after a switch to privacy mode the result being
  made and the rest of that request are not kept, nor are the job's files in ComfyUI.
  Once a request is not kept it stays so. The result is then moved out of its History
  folder and its sidecar deleted; when another program has the file open, it stays there
  for its owner, the job's output names it, and it is deleted when localm stops or at its
  next start. The commands (`localm image`, `music`, `video`
  and `run`), the MCP tool and the coder decide when the generation starts.
- **Saving always works**: the download and move actions, `-o` of `localm image`,
  `localm music` and `localm video`, `output_path` of the MCP `generate_image` tool,
  `/save-media` in `localm run`, and the coder's `generate_image` tool, which writes the
  project file it was asked for. A result moved into a History folder belongs to the key
  that made it, and its entry in the owner index is written then. Without `-o` the three
  commands ask whether to save the result, and for another path while it cannot be saved;
  with nobody at the terminal to ask they refuse before generating.
  `POST /v1/images/generations` returns the image in the response and keeps none.
- Media kept earlier, or made in `log` or `full` mode, is never touched.

When ComfyUI makes the media, localm removes the job's files from ComfyUI's own output
and temp folders and its history entry, also when the job fails or is cancelled, when
localm stops or restarts, when the session turned private during it, and when ComfyUI's
answer to the job did not arrive (a job sent in privacy mode is then looked for in its
queue and history; for one sent before the switch the job's output says so). It
deletes only files ComfyUI lists for that job whose content is what ComfyUI serves, in
the output folder it can resolve (localm's own ComfyUI's folder, the ComfyUI output dir
setting, or the `output` folder inside the ComfyUI folder) and the `temp` folder beside
it. An output file that stays makes the job fail with "Output withheld" and the reason,
and localm then keeps no file of it; a preview that stays in the temp folder is named in
the job's output and is removed by ComfyUI when it next starts. A cancelled job has ten
seconds to stop, and localm is at the end of a stop after 12: what ComfyUI writes after
that stays there, and the job's output names the job. After a crash or a kill nothing is
left to remove what ComfyUI holds: a job queued in a ComfyUI that keeps running stays
there, with its files, the image it was given and its history entry, until you clear it.
An image it was given stays while another running generation uses the same one.
While a private job, or one that turned private, is in ComfyUI,
`<data dir>/run/private-media/` notes its ComfyUI address and job id and nothing else.
After a crash, or a stop that ComfyUI did not follow, the next start of the server says on
its console that the job was cut off, and the app tells the owner once; until it has, the
note stays and each start of the server tells it again. `localm image`, `music`, `video`,
`run` and `mcp` do not tell it. What a ComfyUI you run
yourself or on another machine writes elsewhere (its own log, files a custom node saves
without listing them) is outside localm's control. With `media` in `privacy_keep` the
output file and history entry of a finished job are still removed from ComfyUI; its other
files stay, and so does what a failed or cancelled job left there.

### What privacy mode does not cover

The two dials above are about automatic SESSION traces (a transcript, an
audit log). They are not the whole of what localm writes to disk. A few
kinds of data are explicit, deliberate assets - like a file you save
yourself - and are written in every session mode, privacy included:

- **Scheduled job results.** Every run of a chat or coder job writes its
  prompt, the full reply, status, and timing to
  `<data dir>/jobs/results/<job_id>/<timestamp>.json`, on its own schedule,
  whether or not anyone is watching. You asked the job to run and keep its
  output, so this is not gated on session mode.
- **Knowledge (RAG) collections.** Indexing a document into a collection
  writes to `<data dir>/rag/`, the same way saving a generated image does.
  One-off attachment extraction (`/api/chat/extract`, and `/api/rag/extract`
  with the Knowledge plugin) stays in memory only and never touches disk.
- **The prompt library.** A persona you save (a system prompt plus sampling
  defaults) is written to `prompts.json`.
- **GUI uploads.** A file you upload for a model or tool to read is written
  to `<home>/uploads/`.
- **Files the coder writes.** What the coder is asked to create or change in a
  project, a generated image included, stays in the project.
- **Download records.** A model download in progress is recorded in
  `<data dir>/pending-pulls/` so an interrupted one can be resumed. A record
  holds the model spec and options, its source URLs, sizes and digests, the
  absolute paths of the partial files on this machine, and the identity of the
  localm process that ran the download (its process id and start time). It is
  removed when that download finishes, when you delete it, when none of its
  partial data is left on disk (noticed the next time interrupted downloads are
  listed), and when a later download with the same options finishes (a
  full-repository download of the same folder also clears it when the later
  one used other options). Resuming does not remove it at once: the resumed
  download writes its own record, which replaces it. A record you dismiss stays until one of those happens. Like the
  model registry, it holds model source details only, never conversation
  content.
- **ComfyUI launch records.** When localm starts ComfyUI (for a generation or
  `localm comfy start`), it writes `<data dir>/comfy-launched-<id>.json`, one
  per ComfyUI address, so `localm comfy stop` can end that ComfyUI from another
  localm command. A record holds the ComfyUI address, the process id, start
  time, program path and full command line of the ComfyUI process (the command
  line can contain folder paths on this machine), and the process id and start
  time of the localm process that started it. It is removed when localm stops
  that ComfyUI, when the launch fails at once, and when a later `localm comfy
  stop` finds the process has exited or its process id now belongs to another
  program. It holds no prompt or conversation content.

None of this is silent: each is something you explicitly asked localm to
keep, not a trace of a conversation. But if you are relying on privacy mode
alone to leave nothing behind, know that a job you scheduled, a document you
indexed, a persona you saved, a file you uploaded, a model download that
stopped before it finished, or a ComfyUI localm started still lands on disk.

localm also keeps its own operational files in every mode. They hold no prompt,
reply or generated content, but they can show that localm ran, when, and with
which models:

- **Model facts.** `<data dir>/model_meta.json` holds the file name, size, change
  time and layer count of every model file loaded (the newest 256), so a later load
  needs no new measurement. `registry.json` lists the models you added.
- **Running operations.** While a background operation runs (a generation, a model
  download), `<data dir>/activity/<pid>-<run>.json` holds its kind, label, start time,
  process id and the id of the key that started it. In privacy mode its row is removed
  when it ends, unless "Keep diagnostics" is on; a server that dies leaves it, and the
  next start reports the operation as interrupted.
- **Coordination files.** `<data dir>/run/` holds the running instance's entry, locks
  that keep two localm processes from loading models at once (`run/gpu-lease/`), the
  address and id of a private ComfyUI job in progress and the name of a History file that
  could not be deleted yet (`run/private-media/`), and the
  crash marker of the running server, removed at a normal stop.
- **Caches.** `<data dir>/cache/` holds what the GPU driver compiles for a model
  (`cache/comgr` on AMD ROCm) and package and download caches. On AMD ROCm the driver
  also makes empty `comgr-<pid>-...` folders in the temp folder; localm removes those of
  its own processes when they end, or at its next start (`<data dir>/run/comgr-temp/`
  lists the running ones by process id and start time).

### Another localm instance answering for this one

A chat can be answered by another localm instance on this machine that
already has the model loaded: a route you accept in the GUI, a scheduled job
on a routed model, or `localm mcp --share-loaded-models`. That instance would
otherwise record the chat in its own session mode. So unless this instance
records the chat in `full` mode, the other instance must first confirm that it
records it no more than this instance's mode allows: it is in that mode or a
more private one, or it is reached with its owner key and agrees to record
less. If it does not confirm, nothing is sent to it: accepting the route and
each routed request are refused with a message naming that instance (and its
mode, when it reports one), a routed job fails saying so, and the MCP server
loads the model itself. Reached with any other key, that instance records at
its own mode as checked before each request, so a change it makes to its own
mode while a request is running is not covered.

### The list of running processes

In every session mode each running localm process, and each long-lived program it
starts (ComfyUI, a coder shell, an MCP server), has one small file in `.localm-run/`
in the install folder: its process id, when it started, what kind of process it is,
the process that started it and the process ids of programs it opened for you (a web
browser), never a command line, an environment or any content; the file also holds one
number made from the ids of this machine and this folder, which is the same for no
other folder; the file stays when the process ends and is deleted by a localm process
that starts later, or that asks which processes run (one written by another machine or
container, copied here with the folder, or, on a Linux system without a machine id,
left from before a restart, stays). A process that waits while an update of localm is
not finished has an empty file in `.localm-update/` named after its process id and start
time until it goes on; one that was ended while it waited leaves that file until the
update is put back. After an older localm (0.3.2 to 0.3.4b4) installed this version with
its own updater, the starts that check that update write, in every session mode:
`.localm-update/checked.json` in the install folder (this version; the version text,
size, times and file id of the `VERSION` file in that updater's backup; whether the
checks passed, whether that backup was this install's, removed or is being put back;
when the checks first ran; and the reason each failed check gives: the paths of release
files that are missing or differ, the package requirements the venv lacks, the last
lines this version printed when it did not start, and error messages, which can name
folders on this computer; never a file's content); `.localm-update/checking.lock` while
a start runs them (its process id and start time, a random token and a hashed id of this
computer, as every localm lock file holds); and in the data folder
`updates/update-result.json` (the two versions, the outcome, that reason and the install
folder). After the previous version was put back, that reason is also written into
`updates/runtime-update.json` and `updates/components-update.json`, where the previous
version shows it.

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
the loopback GUI shell holds (or this instance's attach token, which only a
local process running as you can read), because open mode grants every caller host access
and so would otherwise leave this unauthenticated. Directory prefixes in the
returned frames (your data directory, the install location, the Python
environment) are redacted; the file, line and function are kept.

`GET /metrics` (off by default; see [the server API](server-api.md#get-metrics))
exposes request counts and latencies, token counts, tokens per second, time to
first token, queue depth, loaded-model count and GPU memory. Every label is a
route template or a status code, and no metric carries a prompt, a reply, a
model name or a model path, in any session mode. It writes nothing to disk.

### The debug log (`--debug` / `LOCALM_DEBUG`)

`localm gui --debug` (or `LOCALM_DEBUG=1`) writes a debug log under
`<data dir>/logs/`, captures native llama.cpp stderr (model metadata/timings), and
logs each request (method, path, status, timing - never request bodies; for a
refused or failed request the server's reason is logged too, which can name a
file you attached or a site that was contacted). It does
NOT record your chat content in privacy mode (see the guarantee below).

### GUI error messages

Every error message the GUI shows (an error toast, the sidebar status error, an
error shown inside a page) and every script error in the page is also sent to the
server, which writes it to its log as one `gui:` line at WARNING, with home paths,
credentials, email addresses and machine names removed. A burst of the same error
is sent as one line with a count. Like the server's own warnings, that line goes to the
in-memory activity list behind the status window in every mode, and to the debug
log file only when the debug log is on. The Settings "Recent errors" list (the
last five) is kept in the browser's memory in privacy mode and gone on reload; in
`log` and `full` mode it is also kept in the browser's local storage.

### "Keep diagnostics for bug reports" (`keep_diagnostics`)

Privacy mode saves nothing - which also means a hang or crash leaves nothing to
put in a bug report. This opt-in (off by default) keeps the diagnostic bits a
report needs, even in privacy mode:

- the hang watchdog trace,
- the crash/restart breadcrumb log (`pre_restart.log` + the in-memory activity
  ring),
- the bug report localm saves on its own when it hits an unexpected error, and
- a debug log (operational lines only).

Without it, privacy mode saves no such bug report: a command that fails at a
terminal asks whether to save one (the default is no), and an error with no
terminal to ask (a background task, a server) saves nothing.

Set it in **Settings > Privacy & data** (in-app, persistent), with the **desktop
launcher** checkbox, with `--keep-diagnostics` on `localm gui` (`serve` and
`run` have no such flag - use the `LOCALM_KEEP_DIAGNOSTICS` environment
variable for those, a per-run override either way). The hang watchdog trace
and the crash/restart breadcrumb log are already kept automatically in
`log`/`full` mode; the debug log is not - it always needs `--debug` or this
toggle, in any session mode. So in privacy mode the toggle turns on all
three; in `log`/`full` mode it only adds the debug log.

The debug log, the crash guard's trace and the stall watchdog read the setting
when the server starts, so changing the Settings toggle reaches them from the next
server start. The hang dump, the restart breadcrumbs, the next model load's crash
trace and the bug report follow the toggle at once.

---

## 3. Bug reports: what's redacted, and what needs your OK

Filing a bug report - Settings > System > Report a bug in the GUI, `localm
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
`localm bug-report` itself needs a working localm to run. The Python standalone reporter attaches the same
log digest as the in-app reporter, with the records that carry chat content
left out (no log at all when it cannot load that filter), and the
PowerShell one attaches no log tail when the log holds such a record.

**In the GUI and the standalone reporter you see the full report before
anything is sent, and sending needs your explicit OK.** `localm bug-report`
saves the report and prints its path so you can open or edit it; it is sent
only when you choose to send it at the prompt, or at once when you pass
`--send` (no prompt, also without a terminal). In the GUI, "Send to maintainer" saves the report and
shows its whole text; nothing is uploaded until you press "Send this
report", and what is uploaded is that saved text, apart from its closing
note about editing the report, which is left out. The standalone reporter
previews the same text at a terminal prompt - and when
there is no terminal to prompt (piped/redirected stdin, or run
non-interactively), every standalone reporter (Python, PowerShell, and the
`report-issue.sh` curl fallback used when there is no Python) treats that
exactly like you said no: nothing is sent. The curl fallback also sends
nothing when `perl`, which it needs to remove personal details, is missing.
Every reporter follows Network access: with it off nothing is sent unless you
choose to send that one report anyway. Where the send is refused, the GUI
shows a "Send anyway" button (to the owner only), `localm bug-report` and the
standalone reporter ask "Send this one report anyway?" at their terminal prompt
(no is the default), and both take `--send-anyway` for a run without a prompt.
Sending anyway makes one connection to the bug-report server for that report
only; Network access stays off and no setting is changed. When
the standalone reporter cannot read the setting (no `config.json` in the data
directory yet, or one it cannot parse) it asks before sending, with no as the
default, and sends nothing without a terminal; `--send-anyway` answers that
question with yes.

**Feedback and suggestions carry no diagnostics.** Settings > System > Feedback and
Suggestion, and `localm feedback`, save and send only the text you write, the localm
version and which of the two it is: no log, model, configuration, environment or
browser details. The text goes through the same redaction, the same review or prompt
and the same Network access rule, and becomes a public GitHub issue when sent.

**What is redacted from a report:**
- Your home directory, replaced by `~`, and your account name wherever it is
  the folder under a home root (`C:\Users\<name>`, `/home/<name>`,
  `/Users/<name>`), in every textual form the report might quote it in (a
  plain path, a `repr()`'d or JSON-encoded one with doubled backslashes, or a
  path embedded in a native crash traceback's own message) - the backstop
  regex runs unconditionally, so one unresolvable path lookup can't leave the
  raw text unscrubbed. The account name is not searched for elsewhere (a
  folder like `D:\<name>\localm`, or the name in prose).
- URL credentials (`user:pass@...`), credential-named query parameters and
  HTTP header lines (`api_key=`, `token=`, `X-Api-Key:`, `Auth-Token:`, ...),
  and bearer / API-key-shaped tokens - an `Authorization:` value is caught
  whole whatever its scheme (Bearer, Basic or a raw token) - wherever they
  appear in diagnostic text, recent log tails, or the activity ring.
- Email addresses in the usual `name@domain` form (also URL-encoded as
  `name%40domain`), anywhere in the report including the configuration
  section and the subject of an emailed report, replaced with `<redacted-email>`, except the maintainer's contact
  address the report is addressed to.
- Machine names, replaced with `<host>`: the server in a network path
  (`\\FILESERVER01\share`, also in its doubled-backslash and `\\?\UNC\` forms),
  a host name ending in a local-network suffix (`nas.local`, `pc.lan`,
  `printer.home.arpa`, `.internal`, `.localdomain`; the suffix is kept), and this
  computer's own name. The name is always replaced where it stands as a machine
  name: in a URL (`http://server:8080`), after `@` (`me@server`), before a port
  (`server:5000`), or as the value of a `host`, `computer`, `machine`, `node`,
  `device` or `pc` setting, also with `name` after it (`Host Name: SERVER`,
  `hostname=server`). A name with a digit, hyphen or other non-letter (such as
  `DESKTOP-4F2K9` or `node01`) is also replaced wherever else it appears as a
  whole word (not inside a dotted name or a path). A name made only of letters
  can be an ordinary word (`server`, `ubuntu`), so elsewhere in the text it is
  kept: if your computer has such a name and you mention it in your own words,
  review the report before sending it. `\\wsl$`, `\\wsl.localhost`, `localhost`, localm's own
  `localm.local`, Docker's `host.docker.internal` and public domain names are
  kept.
- Config values are sent only from a small allowlist of operational keys
  (port, context size, GPU layer count, mode, ...); the API key itself is
  never stored in config and never included. A few of those keys hold text
  you typed (the ComfyUI launch command, folder and URL, the search URL, the
  allowed CORS origins, the coder reviewer); they pass through the same
  redaction as the rest of the report.

**What a report deliberately keeps**, because it is what makes the report
useful to debug: your install location, data directory path, native runtime
library names, dependency versions, the loaded model's name and backend, and
a tail of the crashed run's own log. Only the home directory and the account
name under a home root are stripped from these, not the whole path - unlike an HTTP response handed to a
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

- **When it fires.** A quiet check at most once every 6 hours per browser while
  the GUI is open (a client-side throttle kept in that browser's storage, so a
  browser that blocks storage checks on every load - there is no background daemon; a headless
  `localm serve` that nobody opens the GUI on never checks), plus whenever you
  press "Check for updates" in the GUI or run `localm update` / `localm update
  --check`.
- **What it sends, and what it does not.** A bare GET request identifying itself
  as `localm`, plus a shared-secret header (the one localm ships with, or the
  `update_token` you configure). The query string
  carries your installed localm version (`client=<version>`, which the update
  server uses to decide which release to offer) and, if you opted into
  prereleases, `channel=prerelease`. It does **not** send chat content, settings
  or an identifier of your install - the comparison to the latest release
  happens locally, on your machine. What it does expose to the update server is
  your IP address, the time you checked, your localm version, and the fact that
  a localm install checked in.
- **It obeys `net_mode`.** The update check reads the network mode you set
  (`localm/netpolicy.py`) and is blocked when it is `off`, failing honestly
  (never a false "you are up to date") rather than silently skipping. It does
  not go through the same per-request domain-list/SSRF checks a chat or coder
  web fetch does, since the update endpoint is fixed by localm itself rather
  than a URL you or a fetched page supplied - see
  [network.md](network.md#what-the-policy-does-not-govern). Turn on "Check
  for updates even when network access is off" in **Settings > System > Updates** to
  exempt just this one channel; it is off by default. `net_mode=off` blocks
  the update check and everything a model triggers unless you opt back in; the
  separate admin-only `net_allow_model_downloads` setting (also off by default)
  exempts only the downloads you trigger yourself.
- **Turning the channel off entirely.** Clear the update endpoint
  (`update_url` and `bugreport_upload_url` both blank) to disable update
  checks regardless of network policy.

---

## The Microsoft Visual C++ Redistributable download (Windows)

On a Windows PC without the Microsoft Visual C++ Redistributable (or with an
older one), setup and `localm setup-llama` ask before they download
Microsoft's installer from `download.visualstudio.microsoft.com`. It is a plain
download of a fixed file: besides a User-Agent naming localm it sends nothing
about you or your install, and Microsoft's server sees your IP address and the
time, as with any download. The installer runs only after its size, sha256 and Microsoft
signature check out, and Windows shows one administrator prompt for it. If you
decline, nothing is downloaded. It obeys `net_mode`: under `off` it is refused
unless `net_allow_model_downloads` is on.

---

## The management log (log and full mode only)

localm appends one line to `logs/management.jsonl` in the data folder when an API key is created or revoked, a login with a wrong key is refused, an update is applied, a setting is saved or a plugin is installed or removed. A line holds the time, the kind of event, a key id or a fixed label (never a key), the names of the settings a save changed (never their values), the plugin name and versions, and for a refused login only whether the client was on this machine. It holds no conversation content and no secret. It is written in log and full mode only; privacy mode writes nothing. The file is capped at 1 MB and one earlier file (`management.jsonl.1`) is kept. If it cannot be written, localm logs a warning and carries on with the operation.

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

The guarantee follows each chat, not only the server. A privacy-mode session
that attaches to a server running in `log` or `full` mode with the debug log on
(`localm run MODEL --mode privacy`, or a coder session in privacy mode) has
none of its messages, replies or raw model output written to that server's
debug log, and memory treats it as privacy mode (nothing stored, and nothing
recalled unless you opted into read-only recall in privacy mode).
While such a request is being answered, the debug log holds back the content
of every other chat on that server as well. That includes the native llama.cpp
lines that repeat generated text (the tool-call grammar's trigger trace): they
are written without that text, and replies the model generates alongside a
privacy one are recorded the same way until the model next has no reply in
progress.

The browser plugin's log lines (a request it refused, a request that failed,
a window it replaced) name the host only; the full address (path and query
string) is logged only when the debug log is on and no session is in privacy
mode.
