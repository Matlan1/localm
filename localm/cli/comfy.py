# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy` - manage localm's OWN (optional) ComfyUI instance.

STAGE S1 (scaffolding): `status` reports whether a managed instance exists and
which ComfyUI localm targets; `remove` deletes the managed instance under the
localm data dir (reversible, self-contained). `setup` is present but HONEST -
provisioning is not built yet (stages S2/S3), so it changes nothing and says so
(AGENTS.md rule 5: no facade). The user's own ComfyUI is never touched.
"""

import shutil

import click

from ._core import console, main


@main.group("comfy")
def comfy_group() -> None:
    """localm's own managed ComfyUI (optional, off by default).

    localm can run its OWN ComfyUI under the data folder instead of depending on
    your install - so it can pin a known-good version and carry fixes. Off by
    default; your own ComfyUI is never modified. Turn it on in Settings -> Media
    (or: localm config managed_comfy_enabled true) once an instance is installed.
    """


@comfy_group.command("status")
def comfy_status() -> None:
    """Show whether a managed ComfyUI is installed, where, and which ComfyUI
    localm currently targets."""
    from ..config import load_config
    from ..media.managed_comfy import (
        MANAGED_COMFY_API_URL, is_managed_comfy_installed, managed_comfy_paths,
        resolve_comfy_target,
    )

    cfg = load_config()
    paths = managed_comfy_paths()
    installed = is_managed_comfy_installed()

    console.print("[bold]Managed ComfyUI[/bold]")
    console.print(f"  Enabled (setting) : {cfg.get('managed_comfy_enabled', False)}")
    console.print(f"  Preferred target  : {cfg.get('comfy_target', 'own')}")
    if installed:
        console.print(f"  Installed         : yes, at {paths.root}")
        console.print(f"  Managed API URL   : {MANAGED_COMFY_API_URL}")
        console.print(f"  Managed models    : {paths.models_dir}")
    else:
        console.print("  Installed         : no - not set up "
                      "(provisioning arrives in a later stage)")

    target = resolve_comfy_target(cfg)
    which = "localm's managed ComfyUI" if target.managed else "your own ComfyUI"
    console.print(f"  Target now        : {which} ({target.api_url})")


@comfy_group.command("remove")
@click.option("-y", "--yes", is_flag=True,
              help="Do not ask for confirmation.")
@click.option("--models", "with_models", is_flag=True,
              help="Also delete the managed models folder (comfyui-models). Off "
                   "by default: models are expensive to re-download, so they are "
                   "kept unless you ask.")
def comfy_remove(yes: bool, with_models: bool) -> None:
    """Delete localm's managed ComfyUI (under the localm data dir only).

    Removes <LOCALM_HOME>/comfyui. Your own ComfyUI (comfy_workdir) is NEVER
    touched. Add --models to also delete the managed models folder.
    """
    from ..media.managed_comfy import managed_comfy_paths

    paths = managed_comfy_paths()
    targets = []
    if paths.root.exists():
        targets.append(paths.root)
    if with_models and paths.models_dir.exists():
        targets.append(paths.models_dir)

    if not targets:
        console.print("[dim]Nothing to remove - no managed ComfyUI is installed.[/dim]")
        return

    listing = "\n".join(f"  {t}" for t in targets)
    console.print(f"This will delete:\n{listing}")
    if not yes and not click.confirm("Remove it?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return

    failed = []
    for t in targets:
        try:
            shutil.rmtree(t)
        except OSError as e:
            # Do not claim success for a delete that failed (rule 5): report it.
            failed.append(f"{t} ({e})")
    if failed:
        console.print("[red]Could not remove:[/red]\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    console.print("[green]Removed localm's managed ComfyUI.[/green]")


@comfy_group.command("setup")
def comfy_setup() -> None:
    """Set up localm's own ComfyUI. NOT YET IMPLEMENTED (stages S2/S3).

    This is a deliberate honest placeholder: it provisions nothing and changes
    nothing, so `localm comfy` is discoverable without pretending to do work it
    cannot yet do.
    """
    console.print(
        "[yellow]Managed-ComfyUI provisioning is not yet implemented "
        "(stage S2/S3).[/yellow]")
    console.print(
        "localm keeps using your own ComfyUI (comfy_workdir). Nothing was "
        "changed. Track progress in dev-notes/DESIGN-localm-managed-comfyui-*.")
