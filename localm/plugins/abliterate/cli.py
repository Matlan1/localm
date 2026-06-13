"""``localm abliterate`` — decensor a model with Heretic, then register it.

NOTICE: This command invokes Heretic
(https://github.com/Matlan1/heretic-win-AMD), licensed under AGPL-3.0-or-later,
as a **separate program** via subprocess. localm never imports Heretic and does
not bundle its source; it is fetched into a gitignored folder on your machine on
request. The abliterated model Heretic produces is your own work product and is
not covered by Heretic's license.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from rich.console import Console

from localm.config import HOME_DIR, MODELS_DIR, ensure_dirs, load_config, save_config
from localm.model_manager import add_local

from . import runner

console = Console(highlight=False)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--model",
    required=True,
    help="Hugging Face repo id, or path to a local model / .gguf to abliterate.",
)
@click.option(
    "--gguf-file",
    default=None,
    help="GGUF filename inside the model repo (for GGUF inputs).",
)
@click.option(
    "--export-gguf",
    "export_gguf",
    default=None,
    metavar="TYPE",
    help="Also export the result to GGUF (e.g. q5_k_m, q8_0). Needs llama.cpp in Heretic.",
)
@click.option(
    "--name",
    default=None,
    help="Registry name for the result (default: <model>-heretic).",
)
@click.option(
    "--print-command",
    is_flag=True,
    help="Print the Heretic command that would run, then exit (no launch).",
)
def main(model, gguf_file, export_gguf, name, print_command):
    """
    Abliterate (decensor) a model with Heretic and register it in localm.

    Heretic runs as a separate, interactive program. When it finishes optimizing,
    choose "Save the model to a local folder" and paste the path this command
    prints. After Heretic exits, localm registers the saved model so you can
    `localm run` it.
    """
    cfg = load_config()
    location = runner.find_heretic(cfg.get("heretic_path"), HOME_DIR)

    # --print-command is a non-interactive preview: never prompt to clone, and
    # show the command even if Heretic isn't installed yet.
    if print_command:
        preview = (
            location
            if location.found
            else runner.HereticLocation(repo_dir=None, executable="heretic")
        )
        command = runner.build_command(
            preview,
            model,
            gguf_file=gguf_file,
            export_strategy="merge",
            export_gguf=export_gguf,
        )
        console.print(f"[dim](cwd: {preview.repo_dir or 'PATH'})[/dim]")
        console.print(" ".join(command))
        return

    if not location.found:
        if not _offer_clone(cfg):
            return
        location = runner.find_heretic(cfg.get("heretic_path"), HOME_DIR)
        if not location.found:
            console.print("[red]Heretic still not found after setup.[/red]")
            return

    reg_name = name or runner.derive_name(model)
    target = (MODELS_DIR / reg_name).resolve()

    command = runner.build_command(
        location,
        model,
        gguf_file=gguf_file,
        export_strategy="merge",
        export_gguf=export_gguf,
    )

    ensure_dirs()
    target.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        "[bold]Heretic will open in this terminal.[/bold] Run the abliteration, then:"
    )
    console.print('  • choose [bold]"Save the model to a local folder"[/bold]')
    console.print(
        f"  • paste this path when asked:\n      [bold cyan]{target}[/bold cyan]"
    )
    if export_gguf:
        console.print(
            f"  • a GGUF ([bold]{export_gguf}[/bold]) is written into that folder too"
        )
    console.print("  • then quit Heretic to return here.\n")
    console.print(f"[dim]Running: {' '.join(command)}[/dim]")
    console.print(f"[dim](in {location.repo_dir or 'PATH'})[/dim]\n")

    try:
        subprocess.run(
            command,
            cwd=str(location.repo_dir) if location.repo_dir else None,
        )
    except FileNotFoundError as error:
        console.print(f"[red]Could not launch Heretic:[/red] {error}")
        if location.repo_dir:
            console.print("[yellow]Is `uv` installed and on PATH?[/yellow]")
        return
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return

    _register_result(target, reg_name)


def _offer_clone(cfg: dict) -> bool:
    """Offer to clone the Heretic fork. Returns True if Heretic is now available."""
    dest = runner.default_clone_dir(HOME_DIR)
    console.print("[yellow]Heretic was not found.[/yellow]")
    console.print(
        "It can be cloned (AGPL-3.0, run as a separate program) into:\n"
        f"  [bold]{dest}[/bold]"
    )
    if not click.confirm("Clone the heretic-win-AMD fork now?", default=False):
        console.print(
            "\nManual setup:\n"
            f'  git clone {runner.HERETIC_FORK_URL} "{dest}"\n'
            f'  cd "{dest}" && uv sync\n'
            "Then re-run, or point localm at it:\n"
            "  localm config heretic_path <dir>   (or set LOCALM_HERETIC_PATH=<dir>)"
        )
        return False

    console.print("[dim]Cloning… (Heretic's ROCm setup runs on its first launch)[/dim]")
    if not runner.clone_heretic(dest):
        console.print(
            "[red]Clone failed.[/red] Check git/network, or use the manual steps above."
        )
        return False

    cfg["heretic_path"] = str(dest)
    save_config(cfg)
    console.print(
        f"[green]✓[/green] Heretic ready at {dest} (saved to config as heretic_path)."
    )
    return True


def _register_result(target: Path, reg_name: str) -> None:
    """Register whatever Heretic saved at ``target`` (or ask where it went)."""
    has_config = (target / "config.json").is_file()
    gguf = next(iter(sorted(target.glob("*.gguf"))), None) if target.is_dir() else None

    if gguf is not None:
        # The GGUF is the better runtime fit on limited VRAM — register it as the
        # primary name, and the safetensors directory under a "-hf" suffix.
        add_local(str(gguf), name=reg_name, on_duplicate="register")
        console.print(f"[green]✓[/green] Registered GGUF as [bold]{reg_name}[/bold].")
        if has_config:
            add_local(str(target), name=f"{reg_name}-hf", on_duplicate="register")
            console.print(
                f"[dim]  (safetensors also registered as {reg_name}-hf)[/dim]"
            )
        console.print(f"Run it:  [bold]localm run {reg_name}[/bold]")
        return

    if has_config:
        add_local(str(target), name=reg_name, on_duplicate="register")
        console.print(
            f"[green]✓[/green] Registered [bold]{reg_name}[/bold] (safetensors)."
        )
        console.print(f"Run it:  [bold]localm run {reg_name}[/bold]")
        return

    console.print(f"[yellow]No model found at[/yellow] {target}.")
    where = click.prompt(
        "Where did Heretic save it? (blank to skip)", default="", show_default=False
    )
    where = where.strip().strip('"')
    if not where:
        console.print(
            f"Skipped. Register later with:  [bold]localm add-local <path> --name {reg_name}[/bold]"
        )
        return

    saved = Path(where)
    looks_like_model = (saved.is_dir() and (saved / "config.json").is_file()) or (
        saved.is_file() and saved.suffix.lower() == ".gguf"
    )
    if looks_like_model:
        add_local(str(saved), name=reg_name, on_duplicate="register")
        console.print(
            f"[green]✓[/green] Registered [bold]{reg_name}[/bold]. Run:  localm run {reg_name}"
        )
    else:
        console.print(
            f"[red]{saved} is not a model directory or .gguf.[/red] "
            "Register it manually with `localm add-local`."
        )
