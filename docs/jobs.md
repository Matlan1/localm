# Scheduled jobs

> Scheduled jobs are provided by the `jobs` plugin. The GUI **Jobs** tab, the
> in-app scheduler, and the seven `/api/jobs` routes listed below appear only
> when it is installed and enabled (`localm plugin install jobs`). The plugin
> has no pip extra. Two other paths under the same prefix,
> `/api/jobs/{id}/events` and `/api/jobs/{id}/cancel`, belong to a different
> thing and are always present; see
> [Two things live under `/api/jobs`](#two-things-live-under-apijobs).

A job runs a chat prompt, a coder task, a memory-synthesis pass, or a knowledge
collection re-sync on a
repeating schedule: a fixed interval or a 5-field cron expression. An in-app
scheduler wakes periodically while a `localm gui` or `localm serve` process
is up, runs every enabled job that is due, and records each run's result so
you can read it later. Typical uses: a nightly "summarise today's changes"
coder task, an hourly digest, a weekday morning briefing.

Three things to know up front:

- **The scheduler lives inside a running server.** It only ticks while a
  `localm gui` or `localm serve` (with the jobs plugin active) is running. There
  is no background daemon; close the server and nothing fires until it is back
  up. A job that came due while the server was down runs ONCE on the next tick
  after restart (a single catch-up, not one run per missed slot): an interval
  job runs as soon as its interval has elapsed, and a cron job back-fires only a
  slot missed within the last 24 hours; a slot missed longer ago is skipped.
- **The CLI, GUI, and API share one on-disk store.** `localm job ...`,
  the Jobs page, and the seven `/api/jobs` routes below all read and write the
  same files under `<data dir>/jobs/`. A change made from the terminal is picked up by a
  running server on its next tick.
- **Run results are explicit user data.** Like generated images, a job's
  recorded results are saved in every session mode, including privacy; the
  underlying chat or coder turn still honours the effective session mode, so it
  leaves no extra transcript in privacy mode.

## From the terminal

```bash
# A chat job every hour (the default when neither --cron nor --every is given)
localm job add digest --prompt "Summarise the top AI news in five bullets."

# An interval job, in seconds
localm job add ping --prompt "Say hello." --every 1800        # every 30 min

# A cron job: 09:00 Monday-Friday  (min hour day-of-month month day-of-week)
localm job add briefing --prompt "Draft my morning briefing." --cron "0 9 * * 1-5"

# A coder job: run an agent task in a repo, scoped to a path glob
localm job add nightly-tests \
  --coder --cwd D:\projects\myapp --scope "tests/**" \
  --prompt "Run the test suite and summarise any failures." \
  --cron "0 2 * * *"

# A knowledge re-sync: keep an indexed folder current, nightly at 03:00
localm job add sync-manuals --rag --collection manuals --cron "0 3 * * *"

localm job list                 # id, name, schedule, enabled state, last status
localm job show <job_id>        # full definition: schedule, prompt, cwd/scope, allow_shell
localm job run <job_id>         # run once now and record the result
localm job results <job_id> [--limit N] [--offset N]  # past run results, newest first
localm job disable <job_id>     # keep it but stop it firing
localm job enable <job_id>
localm job remove <job_id>      # delete the job and its stored results
```

`localm job add` options:

| Option | Meaning |
|---|---|
| `--prompt` | The prompt to run on schedule (required for chat/coder jobs; omit for `--memory`). |
| `--memory` | Synthesise durable facts from recent sessions into the assistant memory; needs no `--prompt`. |
| `--rag --collection NAME` | Re-sync a knowledge collection against the folders it was indexed from; needs no `--prompt`. See [Keeping an indexed folder current](#keeping-an-indexed-folder-current). |
| `--cron "M H Dom Month Dow"` | A 5-field cron schedule: minute, hour, day-of-month, month, day-of-week (0 = Sunday). Mutually exclusive with `--every`. |
| `--every SECONDS` | Interval schedule in seconds. Defaults to hourly if neither schedule flag is given. |
| `--coder` | Run a coder agent task instead of a chat prompt. |
| `--cwd DIR` | Working directory for a coder job. |
| `--scope GLOB` | File-access glob for a coder job. Confines the file tools only: `run_shell`/`run_tests` execute a process and cannot be bounded by a path check, so leave `--allow-shell` off (the default) if you need the scope to be a hard boundary. |
| `--allow-shell` | Coder jobs only: allow full shell execution. Off by default; a scheduled coder job runs restricted (read plus confined edits, no shell, no network, no sub-agents) unless you pass this. |
| `--model M` | Model to run the job with (otherwise the server's active model). |
| `--disabled` | Create the job disabled; it will not run until you enable it. |

## From the GUI

The **Jobs** tab (visible once the plugin is enabled) lists every job with its
schedule and last result. From there you can create a job (chat, coder, or a
knowledge re-sync, interval or cron), enable or disable it, run it now, and
browse each job's past run results. There is no edit action in the GUI; to
change a job's schedule or prompt, delete it and add it again, or use
`PUT /api/jobs/{id}` directly. See [gui.md](gui.md).

## HTTP API

When the plugin is active the server mounts a small REST surface, scoped to the
`jobs` capability (it requires a valid API key when auth is configured, the
same as the rest of the management API; see [server-api.md](server-api.md)).
In open mode (no key configured) these routes still need this instance's
shell token or attach token, same as `/api/activity` below - a bare `curl`
with no credentials is refused here too.
Each job is bound to the key that created it: a `jobs`-scoped key sees and can
touch only its own jobs (an owner/`admin` key sees every job); a job created
with no key configured (open mode) is unrestricted. A foreign job id 404s the
same as a nonexistent one, so a key can never confirm another principal's job
even exists.

| Method + path | Purpose |
|---|---|
| `GET /api/jobs` | List the caller's own jobs. |
| `POST /api/jobs` | Create a job. |
| `GET /api/jobs/{id}` | Job detail. |
| `PUT /api/jobs/{id}` | Update a job. |
| `DELETE /api/jobs/{id}` | Delete a job and its results. |
| `POST /api/jobs/{id}/run` | Run the job now and record the result. `409` if another run (this job or another) is already in progress - jobs never stack model loads. |
| `GET /api/jobs/{id}/results` | Past run results, newest first. Paginated with `?limit=` (default 100, max 1000) and `?offset=`. |

A create/update body carries: `name`, `task_kind` (`chat`, `coder`, `memory`, or
`rag` - the memory and rag kinds mirror the CLI's `--memory` / `--rag` flags and
need no `prompt`), `prompt`, `schedule_kind` (`interval` or `cron`), `schedule`
(seconds as an integer, or a 5-field cron string), and the optional `model`,
`cwd`, `scope`, `collection`, `allow_shell`, and `enabled` fields.
`allow_shell` (coder jobs only) is privileged: setting it requires the owner key
or a `coder:full` key, so a plain `jobs`-scoped client cannot schedule a
shell-capable job. `collection` is required for a `rag` job and refused at
creation if it is not a valid collection name.

## Two things live under `/api/jobs`

Two unrelated things answer under this prefix, and they are meant to. The seven
routes above are this plugin's: they manage recurring task DEFINITIONS in a
store on disk, and once an API key is configured every one of them is gated on
the `jobs` capability.

`GET /api/jobs/{id}/events` and `POST /api/jobs/{id}/cancel` belong to the
server's registry of operations running RIGHT NOW: a model pull, a knowledge
index, an image or video generation, a ComfyUI setup. They are part of the
server itself, so they answer on a headless `localm serve` with no plugins
installed, and they are gated on a valid key plus ownership of that particular
operation, never on the `jobs` scope. Their ids come from the response of
whichever POST started the operation. Nothing is shared between the two
families: an id from one returns 404 on the other, exactly as an id that does
not exist does, and the two kinds of id look identical, so the path you call is
what decides which registry answers.

`GET /api/activity` is how you find those in-flight operations without already
holding an id. It answers `{"now": <server clock>, "operations": [...]}`, where
each entry carries `id`, `kind`, `status`, `created_at`, `finished_at` (null
while it runs), `cancellable`, a `label` that is set for a model pull, a
ComfyUI setup or update, a llama.cpp runtime setup or update, and system
diagnostics (other kinds fall back to `kind`), and `pct`/`phase` once the
operation has something real to report. `pct` is absent, never `0`, before then. Compute an
age as `now - created_at`, using the server's `now` rather than your own clock.
It is deliberately not under `/api/jobs`, so that one prefix does not mean two
things. `localm status` and the MCP activity tool read this same route; on a
server with no API key configured they authenticate with that instance's local
attach token, which is why a bare `curl` with no credentials is refused there.

A model pull, a runtime install, and a ComfyUI setup are also mirrored to disk
under `<data dir>/activity/`, so a restart or a crash does not lose track of
what was happening: on the next start, any row still marked `running` is
reported as **`interrupted`** rather than `failed`, since whether the work
actually finished is genuinely unknown, and stopping or restarting the server
also terminates the background child processes it started instead of
abandoning them. Progress (`pct`/`phase`) is not mirrored and does not survive
a restart.

What it does not promise beyond that: a finished operation is dropped about an
hour after it finishes, and only when some new operation starts, so an idle
server may keep showing it longer; an operation still marked `running` is
never dropped, at any age.

Cancelling always answers `{"status": "cancelling"}`, whatever state the
operation was in. It terminates the subprocess behind a model pull, a model
removal or a ComfyUI setup; for the in-process operations it is only a request
the work has to notice, and image, video and music generation are the ones that
notice it, while a knowledge index, upload, re-embed or embedding setup keeps
running to completion after being marked `cancelled`. On a server with API keys
configured, you see and can touch only the operations your own key started (an
owner or `admin` key sees all); with no key configured there are no owners, so
any authenticated local caller sees everything, which is the intended behaviour
for a single-owner machine.

## Keeping an indexed folder current

A knowledge collection is indexed once, from files and folders you name. Files
added to (or deleted from) an indexed folder afterwards are not noticed on their
own: there is no filesystem watcher, deliberately, because a watcher daemon
would break localm's self-contained design. A **`rag` job** is the supported way
to keep the index current.

```bash
localm job add sync-manuals --rag --collection manuals --cron "0 3 * * *"
localm rag resync manuals        # or run it by hand, any time
```

Each run re-walks the folders the collection was indexed from and re-indexes
incrementally: a new file is added, a changed file is re-indexed, and an
unchanged file is skipped by content hash, so a nightly run over a large folder
costs a read per file and nothing more. It loads no chat model.

**A deleted file is flagged, not removed.** If a document's source file has
disappeared, the re-sync marks that entry `missing` and reports it; the document
stays in the index and stays searchable, and the flag clears by itself if the
file comes back. This is deliberate: a moved file, an unplugged drive, or a
half-finished cloud sync must not be able to silently delete part of your index
(the model registry treats a missing model file the same way). Remove such
entries when you are sure, with `localm rag resync NAME --prune-missing` or by
removing the document from the Knowledge page.

**An unreachable folder is skipped whole.** If an indexed folder is not
available at run time (deleted, unmounted, or replaced by a file), the run
reports it and touches nothing underneath it - no indexing, no flagging, no
pruning. That includes the case the filesystem hides: unmounting a drive on
Linux or macOS leaves the mount point behind as an ordinary empty folder, so a
mount point that is empty *while documents are still indexed under it* is
treated as unavailable rather than as a folder whose files were all deleted.

**A degraded vector index is reported, never quietly repaired away.** If the
stored embeddings no longer match the chunks (an interrupted embed, or a
truncated or hand-edited file), the run says so in its output and leaves the
vector file on disk, moved aside as `vectors.json.rejected` if the chunks had to
be rewritten. Search keeps working lexically meanwhile. Rebuild the index with
`localm rag repair NAME --embed`.

**A manual `rag` write and a scheduled run cannot corrupt each other.** They are
separate processes, and writes to one collection are serialised across processes:
whichever starts second waits for the collection and then stands down rather than
interleaving. A scheduled run that stands down says so in its output ("is being
written by ...") and re-walks the same folders on its next run, so nothing is
lost - and a hand-run `localm rag add|resync` that hits a running job refuses
without changing anything, naming the job's process. Holding a collection has no
time limit, so a long index is never cut short; a holder that dies stops
reporting and its lock is reclaimed automatically.

**Confinement still applies.** A scheduled re-sync runs under the same allowed
or denied folder policy as an interactive add (Settings > Knowledge), including
the always-refused credential folders. A folder that was legal when it was
indexed but is outside the policy now is skipped and reported, never indexed.
Secret-looking files dropped into an indexed folder are filtered exactly as they
are on a normal add.

Embeddings: a re-sync indexes new documents with the configured embedding model
when one is available. If it is not, the new documents are indexed lexical-only,
and the run says so explicitly when that would dilute a collection that has
semantic search.

## How scheduling works

A schedule is either an **interval** (seconds between runs) or a **cron**
expression (see the `--every` / `--cron` rows above). The 5-field cron matcher
supports ranges (`1-5`), lists (`1,3,5`), `*`, and step values (`*/15`,
`1-30/5`, `5/15`).

The scheduler polls about every 30 seconds, so a job fires at the first tick at
or after its due time, not to the exact second.

## Troubleshooting

- **A run failed.** The error is surfaced, not swallowed: the CLI prints
  `Job failed: <reason>`, the Jobs tab shows a toast and the run in the history.
  Check the error there (and `--debug`/the server log for the full trace).
- **No automatic retry.** A failed run does not retry immediately; the job fires
  again at its next scheduled tick. Re-run it now from the Jobs tab or the CLI.
- **A job never fires.** Schedules only advance while the server process is alive
  (see above) - keep `localm serve`/`localm gui` running, or run it on a machine
  that stays up. Confirm the job is enabled and its next-run time is in the future.
- **Stop a job.** Disable or delete it from the Jobs tab or with `localm job`.
- **The job store was corrupt.** If `<data dir>/jobs/`'s definitions file fails
  to parse (a crash mid-write, a damaged filesystem), localm does not lose your
  jobs silently: the corrupt file is copied aside as
  `<file>.corrupt-<timestamp>` (owner-key digests redacted, the last 3 copies
  kept) and a warning is logged, then localm starts with an empty job list. A
  single unreadable job entry inside an otherwise-valid file is skipped the
  same way, logged, and does not block the rest of the store from loading.
