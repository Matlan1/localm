# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy` - manage localm's OWN (optional) ComfyUI instance.

`status` reports whether a managed instance exists and which ComfyUI localm
targets. `remove` deletes the managed instance under the localm data dir
(reversible, self-contained). `setup` provisions one: it REPLICATES the user's
existing ComfyUI when they have one (stage S2, the COPY path - clone at the same
commit, a fresh localm venv, the same packages, shared models), or installs a
FRESH, hardware-matched ComfyUI when they do not (stage S3 - a pinned ComfyUI,
the PyTorch build for their GPU, and the custom nodes localm's workflows need).
The user's own ComfyUI is never touched.
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
                   "omitted, you are asked when custom nodes are present. Copy path "
                   "only (a fresh install starts clean).")
def comfy_setup(copy_custom_nodes) -> None:
    """Set up localm's own ComfyUI (stage S2 copy / stage S3 fresh).

    If you already have a working ComfyUI (comfy_workdir set, with a venv under it),
    localm REPLICATES it at the same commit into the localm data folder, makes a FRESH
    localm venv, installs the same packages, and shares your models via
    extra_model_paths (S2). If you do NOT, localm installs a FRESH, hardware-matched
    ComfyUI (S3): it clones a pinned ComfyUI, makes a localm venv, installs the PyTorch
    build for your GPU, and adds the custom nodes its shipped workflows need. Your own
    ComfyUI is never touched. You are asked before any of your custom nodes are copied
    (copy path only).
    """
    import sys

    from ..config import load_config
    from ..media import managed_comfy_fresh as fresh
    from ..media import managed_comfy_provision as prov

    cfg = load_config()
    # Heads-up before a potentially multi-GB operation: which path will run.
    if prov.discover_user_comfy(cfg) is None:
        console.print(
            "No existing ComfyUI to copy - installing a fresh, hardware-matched "
            "ComfyUI under the localm data folder. This downloads several GB "
            "(ComfyUI + PyTorch for your GPU) and can take a while...")
    else:
        console.print("Replicating your existing ComfyUI into the localm data folder. "
                      "This can take a while (a fresh venv + the same packages)...")

    result = fresh.setup_managed_comfy(
        cfg, copy_custom_nodes=copy_custom_nodes, interactive=sys.stdin.isatty(),
        confirm_copy_nodes=lambda n: click.confirm(
            f"Copy your {n} custom node(s) into localm's ComfyUI?", default=False),
        on_progress=lambda line: console.print(line, style="dim", markup=False))

    if not result.ok:
        console.print(f"[red]{result.message}[/red]")
        raise SystemExit(1)

    console.print(f"[green]{result.message}[/green]")
    if result.status == "copied":
        console.print(f"  Packages replicated    : {result.installed_packages}")
        console.print(f"  Custom nodes copied    : {result.custom_nodes_copied}")
    elif result.status == "fresh":
        console.print(f"  Custom nodes installed : {result.custom_nodes_copied}")
    console.print("Turn it on so localm uses it: "
                  "localm config managed_comfy_enabled true "
                  "(comfy_target=own already targets it).")


@comfy_group.command("update")
@click.option("--reinstall-requirements", "reinstall_requirements", is_flag=True,
              default=False,
              help="Also reinstall ComfyUI's requirements into the managed venv "
                   "(use when the new pin changed ComfyUI's dependencies). Off by "
                   "default: a partial dependency upgrade cannot be rolled back "
                   "exactly, so update stays within the safe git rollback unless "
                   "you ask.")
@click.option("--commit", "commit", default=None,
              help="Advanced: update to a specific ComfyUI commit instead of the "
                   "shipped pin (for testing a candidate before it is pinned).")
def comfy_update(reinstall_requirements: bool, commit) -> None:
    """Advance localm's managed ComfyUI to the shipped pinned ComfyUI version and
    re-apply localm's patch set.

    localm pins a known-good ComfyUI commit and carries a small set of its own
    fixes on top (e.g. the ACE-Step __func__ tolerance). This moves the managed
    checkout to the current pin and re-applies those fixes. It is safe: on any
    failure it rolls the managed ComfyUI back to its previous version. Your own
    ComfyUI is never touched.
    """
    from ..config import load_config
    from ..media import managed_comfy as mc
    from ..media import managed_comfy_update as upd
    from ..media.managed_comfy_fresh import (COMFYUI_PINNED_COMMIT,
                                             COMFYUI_PINNED_VERSION)

    cfg = load_config()
    if not mc.is_managed_comfy_installed():
        console.print("No managed ComfyUI is installed - nothing to update. Run "
                      "'localm comfy setup' first.")
        raise SystemExit(1)

    if commit is None:
        console.print(f"Updating to the pinned ComfyUI {COMFYUI_PINNED_VERSION} "
                      f"({COMFYUI_PINNED_COMMIT[:12]}) and re-applying localm's "
                      "patches. This can take a while...")
    else:
        console.print(f"Updating to ComfyUI {commit[:12]} (advanced override) and "
                      "re-applying localm's patches...")

    result = upd.update_managed_comfy(
        cfg, target_commit=commit, reinstall_requirements=reinstall_requirements,
        on_progress=lambda line: console.print(line, style="dim", markup=False))

    if not result.ok:
        console.print(f"[red]{result.message}[/red]")
        raise SystemExit(1)
    console.print(f"[green]{result.message}[/green]")
