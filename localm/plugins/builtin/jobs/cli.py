# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm job`` CLI: manage scheduled recurring jobs from the terminal.

Subcommands:
  localm job add NAME --prompt "..." [--cron "..." | --every SECONDS]
                                     [--coder --cwd DIR --scope GLOB --allow-shell]
                                     [--rag --collection NAME]
                                     [--model M] [--disabled]
  localm job list
  localm job run JOB_ID              run a job once now (records a result)
  localm job remove JOB_ID
  localm job enable JOB_ID
  localm job disable JOB_ID

The CLI mutates the same on-disk store the GUI/server use, so a running server
picks changes up on its next scheduler tick.
"""

from __future__ import annotations

import sys

import click


@click.group(name="job")
def main() -> None:
    """Manage scheduled recurring jobs (run a chat/coder prompt on a schedule)."""


def _store():
    from localm.plugins.builtin.jobs.store import JobStore
    return JobStore()


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
@click.option("--model", default=None, help="Model to run the job with.")
@click.option("--disabled", is_flag=True, default=False,
              help="Create the job disabled (it will not run until enabled).")
def job_add(name, prompt, cron, every, coder, memory, rag, collection, cwd,
            scope, allow_shell, model, disabled):
    """Add a new scheduled job."""
    from localm.plugins.builtin.jobs.store import Job

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
        sched = (f"every {j.schedule}s" if j.schedule_kind == "interval"
                 else f"cron '{j.schedule}'")
        state = "enabled" if j.enabled else "disabled"
        last = j.last_status or "never run"
        click.echo(f"  {j.id}  {j.name}  [{j.task_kind}, {sched}, {state}]  "
                   f"last: {last}")


@main.command("run")
@click.argument("job_id")
def job_run(job_id):
    """Run a job once now and record its result."""
    from localm.plugins.builtin.jobs.runner import run_job

    store = _store()
    job = store.get(job_id)
    if job is None:
        click.echo(f"No such job: {job_id}", err=True)
        sys.exit(1)
    result = run_job(job, engine=None)
    store.record_result(job_id, result)
    if result.get("status") == "ok":
        click.echo(result.get("output", ""))
    else:
        click.echo(f"Job failed: {result.get('error')}", err=True)
        sys.exit(1)


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
