# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm job`` CLI: manage scheduled recurring jobs from the terminal.

Subcommands:
  localm job add NAME --prompt "..." [--cron "..." | --every SECONDS]
                                     [--coder --cwd DIR --scope GLOB --allow-shell]
                                     [--rag --collection NAME]
                                     [--model M] [--disabled]
  localm job list
  localm job show JOB_ID             full definition: schedule, prompt, cwd/
                                     scope, allow_shell (everything `list`
                                     leaves out)
  localm job run JOB_ID              run a job once now (records a result)
  localm job results JOB_ID          past run results, newest first
  localm job remove JOB_ID
  localm job enable JOB_ID
  localm job disable JOB_ID

The CLI mutates the same on-disk store the GUI/server use, so a running server
picks changes up on its next scheduler tick.
"""

from __future__ import annotations

import sys
from datetime import datetime

import click


def _safe_echo(message: object = "", **kwargs) -> None:
    """``click.echo`` of text localm did not author, without terminal control characters."""
    from localm.console import strip_controls
    click.echo(strip_controls(str(message)), **kwargs)


@click.group(name="job")
def main() -> None:
    """Manage scheduled recurring jobs (run a chat/coder prompt on a schedule)."""


def _store():
    from localm.plugins.builtin.jobs.store import JobStore
    return JobStore()


def _fmt_schedule(schedule_kind: str, schedule) -> str:
    """One-line schedule rendering, shared so `list` and `show` never disagree
    on how the same job reads."""
    return (f"every {schedule}s" if schedule_kind == "interval"
            else f"cron '{schedule}'")


def _fmt_ts(ts) -> str:
    """Readable local timestamp for a detail view, or '-' when unset (never
    run / a result missing its own timing). Seconds-precision, unlike
    cli.keys._fmt_ts's minute-precision: a results listing can hold several
    runs a minute apart and needs to tell them apart."""
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "-"


@main.command("add")
@click.argument("name")
@click.option("--prompt", default=None, help="The prompt to run on schedule "
              "(required for chat/coder jobs; omit for --memory).")
@click.option("--cron", "cron", default=None,
              help='5-field cron schedule, e.g. "0 9 * * 1-5".')
@click.option("--every", "every", type=int, default=None,
              help="Interval schedule in seconds (mutually exclusive with --cron).")
@click.option("--coder", "coder", is_flag=True, default=False,
              help="Run a coder agent instead of a chat prompt.")
@click.option("--memory", "memory", is_flag=True, default=False,
              help="Synthesise durable facts from recent sessions into the "
                   "assistant memory (no prompt needed).")
@click.option("--rag", "rag", is_flag=True, default=False,
              help="Re-sync a knowledge collection against the folders it was "
                   "indexed from, picking up files added or changed since "
                   "(needs --collection, no prompt).")
@click.option("--collection", default=None,
              help="The knowledge collection a --rag job re-syncs.")
@click.option("--cwd", default=None, help="Working directory for a coder job.")
@click.option("--scope", default=None, help="File-access glob for a coder job.")
@click.option("--allow-shell", "allow_shell", is_flag=True, default=False,
              help="Coder jobs only: allow full shell execution. Off by default, a "
                   "scheduled coder job runs restricted (read + confined edit, no "
                   "shell/network) so an unattended run cannot be steered into "
                   "run_shell by hostile content.")
@click.option("--model", default=None,
              help="Registered model to run the job with (a name from "
                   "'localm list', not a path).")
@click.option("--disabled", is_flag=True, default=False,
              help="Create the job disabled (it will not run until enabled).")
def job_add(name, prompt, cron, every, coder, memory, rag, collection, cwd,
            scope, allow_shell, model, disabled):
    """Add a new scheduled job."""
    from localm.plugins.builtin.jobs.store import Job, cwd_unc_error

    # The THIRD write path into `model`, after POST and PUT /api/jobs. Checking
    # at the write keeps _check_model_name's promise ("a poisoned row never
    # reaches disk") true for every writer, and turns an unregistered name from a
    # silent repeating failure on every scheduled tick into one immediate error.
    from localm.model_manager import unregistered_model_error
    _bad = unregistered_model_error(model)
    if _bad:
        click.echo(f"Invalid job: {_bad}", err=True)
        sys.exit(1)

    # THIRD write path into `cwd`, after POST and PUT /api/jobs (plug.py's
    # _check_cwd): a poisoned row must never reach disk regardless of which
    # writer created it. Shares cwd_unc_error's wording with plug.py and the
    # runner's run-time re-check, so all three never drift apart.
    _bad_cwd = cwd_unc_error(cwd)
    if _bad_cwd:
        click.echo(f"Invalid job: {_bad_cwd}", err=True)
        sys.exit(1)

    picked = [f for f, on in (("--coder", coder), ("--memory", memory),
                              ("--rag", rag)) if on]
    if len(picked) > 1:
        click.echo(f"Use only one of {', '.join(picked)}.", err=True)
        sys.exit(1)
    if cron and every is not None:
        click.echo("Use either --cron or --every, not both.", err=True)
        sys.exit(1)
    if not cron and every is None:
        every = 3600        # default: hourly
    if cron:
        schedule_kind, schedule = "cron", cron
    else:
        schedule_kind, schedule = "interval", int(every)
    try:
        kind = "chat"
        for flag, kind_name in ((memory, "memory"), (coder, "coder"),
                                (rag, "rag")):
            if flag:
                kind = kind_name
                break
        job = Job(
            name=name,
            task_kind=kind,
            prompt=prompt or "",
            schedule_kind=schedule_kind,
            schedule=schedule,
            model=model,
            cwd=cwd,
            scope=scope,
            collection=collection,
            allow_shell=allow_shell,
            enabled=not disabled,
        )
        _store().add(job)
    except ValueError as e:
        click.echo(f"Invalid job: {e}", err=True)
        sys.exit(1)
    click.echo(f"Added job {job.id} ({job.name}).")


@main.command("list")
def job_list():
    """List all scheduled jobs."""
    jobs = _store().list()
    if not jobs:
        click.echo("No jobs.")
        return
    for j in jobs:
        sched = _fmt_schedule(j.schedule_kind, j.schedule)
        state = "enabled" if j.enabled else "disabled"
        last = j.last_status or "never run"
        click.echo(f"  {j.id}  {j.name}  [{j.task_kind}, {sched}, {state}]  "
                   f"last: {last}")


@main.command("show")
@click.argument("job_id")
def job_show(job_id):
    """Show a job's full definition: schedule, prompt, cwd/scope, allow_shell -
    everything `list`'s one-line summary leaves out."""
    job = _store().get(job_id)
    if job is None:
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)

    # Reuse the API's own redaction: `_job_dict` strips
    # `owner`/`owner_is_owner_key` (internal principal bindings never meant to
    # reach a client), so this shows exactly what `GET /api/jobs/{id}` would.
    from localm.plugins.builtin.jobs.plug import _job_dict
    d = _job_dict(job)

    click.echo(f"Job {d['id']}  ({d['name']})")
    click.echo(f"  kind:        {d['task_kind']}")
    click.echo(f"  state:       {'enabled' if d['enabled'] else 'disabled'}")
    click.echo(f"  schedule:    {_fmt_schedule(d['schedule_kind'], d['schedule'])}")
    click.echo(f"  model:       {d['model'] or '(default)'}")
    # Printed unconditionally rather than only for task_kind == 'coder'/'rag':
    # this command exists so an operator can AUDIT a job, and a stale or
    # oddly-edited job carrying a value that no longer matches its kind is the
    # case worth seeing.
    click.echo(f"  cwd:         {d['cwd'] or '-'}")
    click.echo(f"  scope:       {d['scope'] or '-'}")
    click.echo(f"  collection:  {d['collection'] or '-'}")
    click.echo(f"  allow_shell: {'yes' if d['allow_shell'] else 'no'}")
    click.echo(f"  prompt:      {d['prompt'] or '-'}")
    click.echo(f"  created:     {_fmt_ts(d['created'])}")
    last_run = _fmt_ts(d['last_run'])
    if d['last_status']:
        last_run += f"  ({d['last_status']})"
    click.echo(f"  last_run:    {last_run}")
    click.echo(f"  last_result: {d['last_result_id'] or '-'}")


def _running_server():
    """``(base_url, registry_entry)`` of the live localm server serving this
    data folder and directory, or None when none answers."""
    from localm import config, instances
    from localm.bindhost import self_connect_host, url_host

    entry = instances.find_attachable(config.HOME_DIR, instances.resolve_root_dir())
    if entry is None:
        return None
    scheme = entry.get("scheme", "http")
    host = url_host(self_connect_host(entry.get("host")))
    return f"{scheme}://{host}:{entry.get('port')}", entry


def _run_on_server(job_id: str, url: str, entry: dict):
    """Run *job_id* through ``POST /api/jobs/{id}/run`` on the server at *url*.

    Returns the run result dict (already recorded by the server), or None when
    the server has no jobs route (the caller then runs the job in this process).
    Exits 1 with a message on a refusal or an unreachable server.
    """
    from urllib.parse import quote

    import requests

    from localm import tls
    from localm.auth import resolve_bearer_headers

    headers = resolve_bearer_headers(entry.get("token"))
    try:
        resp = requests.post(f"{url}/api/jobs/{quote(job_id, safe='')}/run",
                             headers=headers, timeout=(10, None),
                             verify=tls.requests_verify(url))
    except requests.ConnectionError as e:
        _safe_echo(f"Could not reach the running localm server at {url}: {e}", err=True)
        sys.exit(1)
    except requests.RequestException as e:
        _safe_echo(f"No reply from the running localm server at {url} ({e}); the run "
                   "may still finish there. See 'localm job results "
                   f"{job_id}'.", err=True)
        sys.exit(1)

    detail = ""
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        detail = str(body.get("detail") or "")
    if resp.status_code == 404 and detail == "Not Found":
        return None
    if resp.status_code == 409:
        click.echo(f"Another job is running in the localm server at {url}; "
                   "try again once it finishes.", err=True)
        sys.exit(1)
    if resp.status_code == 401:
        click.echo("The running localm server refused the request (401). Set "
                   "LOCALM_API_KEY to the server's key.", err=True)
        sys.exit(1)
    if not resp.ok:
        _safe_echo(f"The running localm server refused the run "
                   f"({resp.status_code}): {detail or resp.text}", err=True)
        sys.exit(1)
    if not isinstance(body, dict):
        click.echo("The running localm server answered, but its reply was not "
                   f"valid JSON. See 'localm job results {job_id}'.", err=True)
        sys.exit(1)
    return body


@main.command("run")
@click.argument("job_id")
def job_run(job_id):
    """Run a job once now and record its result.

    When a localm server is running for this data folder and directory, the
    job runs inside that server (``POST /api/jobs/{id}/run``), on the model it
    already holds; otherwise it runs in this process.
    """
    from localm.plugins.builtin.jobs.runner import run_job

    store = _store()
    job = store.get(job_id)
    if job is None:
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)
    server = _running_server()
    if server is not None:
        result = _run_on_server(job_id, *server)
        if result is not None:
            if result.get("status") == "ok":
                _safe_echo(result.get("output", ""))
                return
            _safe_echo(f"Job failed: {result.get('error')}", err=True)
            sys.exit(1)
        click.echo("The running localm server has no jobs plugin; running the "
                   "job in this process.", err=True)
    claimed = store.claim(job_id)
    if claimed is None:
        click.echo(f"Job {job_id} is already running; not starting a second run.",
                   err=True)
        sys.exit(1)
    try:
        result = run_job(claimed, engine=None)
    except BaseException:
        store.release_claim(job_id, claimed.claim)
        raise
    store.record_result(job_id, result)
    if result.get("status") == "ok":
        _safe_echo(result.get("output", ""))
    else:
        _safe_echo(f"Job failed: {result.get('error')}", err=True)
        sys.exit(1)


@main.command("results")
@click.argument("job_id")
@click.option("--limit", type=int, default=None,
              help="Show at most N results (newest first). Default: all.")
@click.option("--offset", type=int, default=0, show_default=True,
              help="Skip the newest N results before applying --limit.")
def job_results(job_id, limit, offset):
    """Show a job's past run results, newest first."""
    store = _store()
    if store.get(job_id) is None:
        # Checked explicitly rather than trusting an empty result list: without
        # this, "no such job" and "this job has never run" print identically, and
        # only one of those means an unknown id.
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)
    results = store.list_results(job_id, limit=limit, offset=offset)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        when = _fmt_ts(r.get("finished") or r.get("started"))
        status = r.get("status") or "?"
        click.echo(f"[{when}] {status}  (result {r.get('result_id', '-')})")
        is_error = status == "error"
        text = (r.get("error") or "(error, no detail)") if is_error \
            else (r.get("output") or "(no output)")
        for line in text.splitlines() or [""]:
            _safe_echo(f"    {line}")
        click.echo()


@main.command("remove")
@click.argument("job_id")
def job_remove(job_id):
    """Remove a job (and its stored results)."""
    if _store().remove(job_id):
        click.echo(f"Removed job {job_id}.")
    else:
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)


@main.command("enable")
@click.argument("job_id")
def job_enable(job_id):
    """Enable a job."""
    _set_enabled(job_id, True)


@main.command("disable")
@click.argument("job_id")
def job_disable(job_id):
    """Disable a job."""
    _set_enabled(job_id, False)


def _set_enabled(job_id: str, on: bool) -> None:
    try:
        _store().update(job_id, enabled=on)
    except KeyError:
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)
    click.echo(f"{'Enabled' if on else 'Disabled'} job {job_id}.")


if __name__ == "__main__":
    main()
