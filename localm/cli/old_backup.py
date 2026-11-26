# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm update --remove-old-backup``: remove the backup an older localm's update
left that is kept because it cannot be proven to be this install's
(``localm.updater_old_clients.remove_kept_backup``)."""

import sys

import click

from ._core import console


def remove_old_backup(*, yes: bool) -> None:
    """Show the backup, ask before removing it unless *yes*, remove it, and exit 1 when
    there is none, the removal is refused or not everything could be removed."""
    from rich.markup import escape

    from localm import updater, updater_old_clients
    from localm.config import home_dir
    root, home = updater.repo_root(), home_dir()
    found = updater_old_clients.kept_backup(root, home)
    if found is None:
        console.print("There is no backup of an earlier update that is kept because it "
                      "cannot be proven to be this install's.")
        sys.exit(1)
    size = max(1, round(found["bytes"] / (1024 * 1024)))
    console.print(f"Backup of the update from localm {escape(found['version'])}, {size} MB, "
                  "kept because it cannot be proven to be this install's.")
    if not yes:
        if not sys.stdin.isatty():
            console.print("[yellow]Not removed:[/yellow] confirm with --yes when no one can "
                          "answer the question.")
            sys.exit(1)
        if not click.confirm("If it belongs to another localm install that uses this data "
                             f"folder, that install can no longer go back to localm "
                             f"{found['version']} with it. Remove it?", default=False):
            console.print("Not removed.")
            sys.exit(1)
    try:
        left = updater_old_clients.remove_kept_backup(root, home)
    except updater_old_clients.Refused as e:
        console.print(f"[yellow]Not removed:[/yellow] {escape(str(e))}.")
        sys.exit(1)
    if left:
        console.print("[red]Not everything could be removed:[/red] "
                      + escape("; ".join(left[:3])))
        sys.exit(1)
    console.print("[green]The backup was removed.[/green]")
