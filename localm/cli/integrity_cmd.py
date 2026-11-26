# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm integrity``: compare localm's files with the installed release, keep the
user's own changes, and repair the differing files (``localm.integrity``)."""

import sys

import click

from localm import integrity

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
                   "of the installed version; without PATHS, also install the native "
                   "runtime again when files of it differ.")
def integrity_cmd(paths, full: bool, do_keep: bool, do_repair: bool) -> None:
    """Compare localm's files with the list of the installed release.

    Lists every file that differs from the release or is missing, and the files of
    the native runtime whose size or time changed. --keep records files as yours, so
    they are no longer reported while the release keeps them unchanged. --repair
    downloads the installed version's signed build and writes exactly those files
    from it; without PATHS it also installs the native runtime again (setup-llama,
    the installed backend and build) when files of it differ. Exits 1 while files
    differ."""
    from rich.markup import escape

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
    rt_differing = result["runtime"]["differ"] + result["runtime"]["missing"]
    runtime_too = do_repair and not paths and bool(rt_differing)
    chosen = []
    if do_keep or do_repair:
        chosen = list(paths) or differing
        unknown = [p for p in paths if p not in differing and p not in result["kept"]]
        if unknown:
            console.print("[yellow]These files match the release or are not part of it:"
                          f"[/yellow] {escape(', '.join(unknown))}")
            sys.exit(2)
        if not chosen and not runtime_too:
            console.print("No file of the release differs, so there is nothing to "
                          f"{'keep' if do_keep else 'repair'}.")
    if chosen:
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
    runtime_failed = False
    if runtime_too:
        from localm.setup_llama.repair import repair_runtime as _repair_runtime
        res = _repair_runtime(say=lambda m: console.print(f"[dim]{escape(m)}[/dim]"))
        if res["repaired"]:
            console.print(f"[green]Installed the {escape(res['backend'] or '')} runtime "
                          f"{escape(res['build'] or '')} again:[/green] "
                          f"{len(res['files'])} file(s) of it repaired.")
            if res["note"]:
                console.print(f"[yellow]{escape(res['note'][0].upper() + res['note'][1:])}."
                              "[/yellow]")
        else:
            runtime_failed = True
            console.print(f"[red]The native runtime was not repaired:[/red] "
                          f"{escape(res['error'])}. To repair it: {escape(res['fix'])}")
    if chosen or runtime_too:
        result = integrity.run()
    for line in integrity.lines(result):
        console.print(f"  - {escape(line)}")
    for rel in result["kept"]:
        console.print(f"  [dim]kept as yours: {escape(rel)}[/dim]")
    rt = result["runtime"]
    if rt["present"] and not rt["listed"] and not rt.get("error"):
        console.print("  [dim]The native runtime has no list of its files yet (it was "
                      "installed by an older localm); it gets one the next time it is "
                      "installed.[/dim]")
    if result["count"]:
        console.print(f"[yellow]{escape(integrity.summary(result))}.[/yellow] "
                      "'localm integrity --repair' writes them again from the release; "
                      "'localm integrity --keep' keeps them as yours.")
        if (rt["differ"] or rt["missing"]) and not runtime_failed:
            console.print("[dim]'localm integrity --repair' also installs the native "
                          "runtime again.[/dim]")
        sys.exit(1)
    console.print(f"[green]{escape(integrity.summary(result))}.[/green]")
