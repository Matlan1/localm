# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm integrity``: compare localm's files with the installed release, keep the
user's own changes, and repair the differing files (``localm.integrity``)."""

import sys

import click

from ._core import console, main


@main.command("integrity")
@click.argument("paths", nargs=-1)
@click.option("--full", is_flag=True,
              help="Read every file in full, the native runtime included, instead of "
                   "trusting files whose size and time are unchanged.")
@click.option("--keep", "do_keep", is_flag=True,
              help="Keep the differing files (or PATHS) as your own: they are listed "
                   "apart and no longer reported.")
@click.option("--repair", "do_repair", is_flag=True,
              help="Write the differing files (or PATHS) again from the signed release "
                   "of the installed version.")
def integrity_cmd(paths, full: bool, do_keep: bool, do_repair: bool) -> None:
    """Compare localm's files with the list of the installed release.

    Lists every file that differs from the release or is missing, and the files of
    the native runtime whose size or time changed. --keep records files as yours, so
    they are no longer reported while the release keeps them unchanged. --repair
    downloads the installed version's signed build and writes exactly those files
    from it. Exits 1 while files differ."""
    from rich.markup import escape

    from localm import integrity
    from localm.bugreport import LocalmError
    if do_keep and do_repair:
        console.print("[yellow]Use only one of --keep and --repair.[/yellow]")
        sys.exit(2)
    if paths and not (do_keep or do_repair):
        console.print("[yellow]Name files only with --keep or --repair.[/yellow]")
        sys.exit(2)
    result = integrity.run(full=full)
    if not result["available"]:
        console.print(f"[yellow]Not checked:[/yellow] {escape(result['reason'])}")
        sys.exit(1)
    differing = result["differ"] + result["missing"]
    if do_keep or do_repair:
        chosen = list(paths) or differing
        unknown = [p for p in paths if p not in differing and p not in result["kept"]]
        if unknown:
            console.print("[yellow]These files match the release or are not part of it:"
                          f"[/yellow] {escape(', '.join(unknown))}")
            sys.exit(2)
        if not chosen:
            console.print(f"[green]{escape(integrity.summary(result))}.[/green]")
            return
        try:
            if do_keep:
                for rel in integrity.keep(chosen):
                    console.print(f"Kept as yours: {escape(rel)}")
            else:
                res = integrity.repair(chosen, say=lambda m: console.print(
                    f"[dim]{escape(m)}[/dim]"))
                for item in res["failed"]:
                    console.print(f"[red]Not repaired:[/red] {escape(item['path'])} "
                                  f"({escape(item['reason'])})")
                if res["repaired"]:
                    console.print(f"[green]Repaired {len(res['repaired'])} file(s).[/green] "
                                  "Restart localm to use them.")
        except (integrity.IntegrityError, LocalmError) as e:
            what = "Keep" if do_keep else "Repair"
            text = (f"{e.summary} ({e.reason})" if isinstance(e, LocalmError) else str(e))
            console.print(f"[red]{what} failed:[/red] {escape(text)}")
            sys.exit(1)
        result = integrity.run()
    for line in integrity.lines(result):
        console.print(f"  - {escape(line)}")
    for rel in result["kept"]:
        console.print(f"  [dim]kept as yours: {escape(rel)}[/dim]")
    rt = result["runtime"]
    if rt["present"] and not rt["listed"]:
        console.print("  [dim]The native runtime has no list of its files yet (it was "
                      "installed by an older localm); it gets one the next time it is "
                      "installed.[/dim]")
    if result["count"]:
        console.print(f"[yellow]{escape(integrity.summary(result))}.[/yellow] "
                      "'localm integrity --repair' writes them again from the release; "
                      "'localm integrity --keep' keeps them as yours.")
        if rt["differ"] or rt["missing"]:
            console.print("[dim]The native runtime is not part of the release: "
                          "'localm setup-llama' installs it again.[/dim]")
        sys.exit(1)
    console.print(f"[green]{escape(integrity.summary(result))}.[/green]")
