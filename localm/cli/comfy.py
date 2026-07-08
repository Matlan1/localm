# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy` - manage localm's OWN (optional) ComfyUI instance.

`status` reports whether a managed instance exists and which ComfyUI localm
targets. `remove` deletes the managed instance under the localm data dir
(reversible, self-contained). `setup` provisions one by REPLICATING the user's
existing ComfyUI (stage S2, the COPY path): it clones their ComfyUI at the same
commit, makes a fresh localm venv, installs the same packages, and shares their
models - the user's own ComfyUI is never touched. When the user has NO ComfyUI to
copy, setup honestly reports that the fresh hardware-matched install (stage S3) is
not built yet and changes nothing (AGENTS.md rule 5: no facade).
"""

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
                      "(run 'localm comfy setup' to replicate your ComfyUI)")

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
    from ..media.managed_comfy import (managed_comfy_remove_targets,
                                       remove_managed_comfy)

    targets = managed_comfy_remove_targets(with_models)
    if not targets:
        console.print("[dim]Nothing to remove - no managed ComfyUI is installed.[/dim]")
        return

    listing = "\n".join(f"  {t}" for t in targets)
    console.print(f"This will delete:\n{listing}")
    if not yes and not click.confirm("Remove it?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return

    # remove_managed_comfy is the shared removal (also used by the GUI route); it
    # reports any path it could NOT delete (rule 5) instead of claiming success.
    _, failed = remove_managed_comfy(with_models)
    if failed:
        console.print("[red]Could not remove:[/red]\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    console.print("[green]Removed localm's managed ComfyUI.[/green]")


@comfy_group.command("setup")
@click.option("--copy-custom-nodes/--no-custom-nodes", "copy_custom_nodes",
              default=None,
              help="Copy your ComfyUI custom_nodes into localm's ComfyUI "
                   "(--copy-custom-nodes) or start clean (--no-custom-nodes). If "
                   "omitted, you are asked when custom nodes are present.")
def comfy_setup(copy_custom_nodes) -> None:
    """Set up localm's own ComfyUI by REPLICATING your existing one (stage S2).

    When you already have a working ComfyUI (comfy_workdir set, with a venv under
    it), localm clones it at the same commit into the localm data folder, makes a
    FRESH localm venv, installs the same packages, and shares your models via
    extra_model_paths - your own ComfyUI is never touched. You are asked before any
    custom nodes are copied. A fresh, hardware-matched install for users WITHOUT a
    ComfyUI is a later stage (S3) and is not built yet.
    """
    import sys

    from ..config import load_config
    from ..media import managed_comfy_provision as prov

    cfg = load_config()
    stack = prov.discover_user_comfy(cfg)

    if stack is None:
        # Dispatcher: nothing usable to copy. The fresh hardware-matched install is
        # stage S3 and is NOT built yet - say so honestly and change nothing.
        workdir = cfg.get("comfy_workdir")
        if workdir:
            console.print(
                f"[yellow]Found comfy_workdir ({workdir}) but no usable ComfyUI to "
                "copy - it needs a main.py and a venv (venv/ or .venv/) under "
                "it.[/yellow]")
        else:
            console.print("[yellow]No existing ComfyUI is configured "
                          "(comfy_workdir is unset).[/yellow]")
        console.print(
            "A fresh, hardware-matched ComfyUI install is not yet implemented "
            "(stage S3), so nothing was changed. To replicate an existing ComfyUI, "
            'point localm at it first: localm config comfy_workdir "<path>".')
        return

    n_nodes = prov.count_user_custom_nodes(stack.workdir)
    do_copy = prov.resolve_copy_custom_nodes(
        copy_custom_nodes, n_nodes=n_nodes, interactive=sys.stdin.isatty(),
        confirm=lambda: click.confirm(
            f"Copy your {n_nodes} custom node(s) into localm's ComfyUI?",
            default=False))

    src = f"commit {stack.commit[:12]}" if stack.commit else stack.version_marker
    console.print(f"Replicating your ComfyUI ({src}) into localm's data folder. "
                  "This can take a while (a fresh venv + the same packages)...")
    result = prov.provision_by_copy(
        stack, cfg, copy_custom_nodes=do_copy,
        on_progress=lambda line: console.print(line, style="dim", markup=False))

    if not result.ok:
        console.print(f"[red]{result.message}[/red]")
        raise SystemExit(1)

    console.print(f"[green]{result.message}[/green]")
    console.print(f"  Packages replicated : {result.installed_packages}")
    console.print(f"  Custom nodes copied : {result.custom_nodes_copied}")
    console.print("Turn it on so localm uses it: "
                  "localm config managed_comfy_enabled true "
                  "(comfy_target=own already targets it).")
