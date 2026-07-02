# Scheduled jobs

> Jobs are provided by the `jobs` plugin. The GUI **Jobs** tab, the `/api/jobs`
> routes, and the in-app scheduler appear only when it is installed and enabled
> (`localm plugin install jobs`). The plugin has no pip extra.

A job runs a chat or coder prompt on a repeating schedule: a fixed interval or
a 5-field cron expression. An in-app scheduler wakes periodically while a
`localm gui` or `localm serve` process is up, runs every enabled job that is
due, and records each run's result so you can read it later. Typical uses: a
nightly "summarise today's changes" coder task, an hourly digest, a weekday
morning briefing.

Three things to know up front:

- **The scheduler lives inside a running server.** It only ticks while a
  `localm gui` or `localm serve` (with the jobs plugin active) is running. There
  is no background daemon; close the server and nothing fires until it is back
  up. A job that came due while the server was down runs ONCE on the next tick
  after restart (a single catch-up, not one run per missed slot): an interval
  job runs as soon as its interval has elapsed, and a cron job back-fires a slot
  missed within the last 24 hours. A slot missed longer ago than that is skipped
  (a machine off for a week does not suddenly run last Monday's briefing).
- **The CLI, GUI, and API share one on-disk store.** `localm job ...`,
  the Jobs page, and the `/api/jobs` routes all read and write the same files
  under `<data dir>/jobs/`. A change made from the terminal is picked up by a
  running server on its next tick (about every 30 seconds).
- **Run results are explicit user data.** Like generated images, a job's
  recorded results are saved in every session mode, including privacy. The
  *prompt run itself* (the underlying chat or coder turn) still honours the
  effective session mode, so it leaves no extra transcript in privacy mode.

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

localm job list                 # id, name, schedule, enabled state, last status
localm job run <job_id>         # run once now and record the result
localm job disable <job_id>     # keep it but stop it firing
localm job enable <job_id>
localm job remove <job_id>      # delete the job and its stored results
```

`localm job add` options:

| Option | Meaning |
|---|---|
| `--prompt` | The prompt to run on schedule (required for chat/coder jobs; omit for `--memory`). |
| `--memory` | Synthesise durable facts from recent sessions into the assistant memory; needs no `--prompt`. |
| `--cron "M H Dom Month Dow"` | A 5-field cron schedule: minute, hour, day-of-month, month, day-of-week (0 = Sunday). Mutually exclusive with `--every`. |
| `--every SECONDS` | Interval schedule in seconds. Defaults to hourly if neither schedule flag is given. |
| `--coder` | Run a coder agent task instead of a chat prompt. |
| `--cwd DIR` | Working directory for a coder job. |
| `--scope GLOB` | File-access glob for a coder job. |
| `--allow-shell` | Coder jobs only: allow full shell execution. Off by default; a scheduled coder job runs restricted (read plus confined edits, no shell) unless you pass this. |
| `--model M` | Model to run the job with (otherwise the server's active model). |
| `--disabled` | Create the job disabled; it will not run until you enable it. |

## From the GUI

The **Jobs** tab (visible once the plugin is enabled) lists every job with its
schedule and last result. From there you can create a job (chat or coder,
interval or cron), enable or disable it, edit it, run it now, and browse each
job's past run results. See [gui.md](gui.md).

## HTTP API

When the plugin is active the server mounts a small REST surface, scoped to the
`jobs` capability (it requires a valid API key only when auth is configured, the
same as the rest of the management API; see [server-api.md](server-api.md)).

| Method + path | Purpose |
|---|---|
| `GET /api/jobs` | List all jobs. |
| `POST /api/jobs` | Create a job. |
| `GET /api/jobs/{id}` | Job detail. |
| `PUT /api/jobs/{id}` | Update a job. |
| `DELETE /api/jobs/{id}` | Delete a job and its results. |
| `POST /api/jobs/{id}/run` | Run the job now and record the result. |
| `GET /api/jobs/{id}/results` | Past run results, newest first. |

A create/update body carries: `name`, `task_kind` (`chat` or `coder`),
`prompt`, `schedule_kind` (`interval` or `cron`), `schedule` (seconds as an
integer, or a 5-field cron string), and the optional `model`, `cwd`, `scope`,
`allow_shell`, and `enabled` fields. `allow_shell` (coder jobs only) is
privileged: setting it requires the owner key or a `coder:full` key, so a
plain `jobs`-scoped client cannot schedule a shell-capable job.

## How scheduling works

Schedules are one of two kinds:

- **interval**: a number of seconds between runs.
- **cron**: a self-contained 5-field matcher, `minute hour day-of-month month
  day-of-week`, with `0` meaning Sunday in the day-of-week field. Ranges
  (`1-5`), lists (`1,3,5`), and `*` are supported.

The scheduler polls about every 30 seconds, so a job fires at the first tick at
or after its due time, not to the exact second. Because it runs inside the
server process, schedules are evaluated only while that process is alive.

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
