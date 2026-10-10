# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-sdcpp``: install the stable-diffusion.cpp runtime used by the
native image backend."""

from __future__ import annotations

import sys

import click

from . import pins, runtime


@click.command("setup-sdcpp")
@click.option("--backend", "backend", default="auto", show_default=True,
              type=click.Choice(["auto", *runtime.BACKENDS], case_sensitive=False),
              help="Runtime build to install. 'auto' picks the best one for this machine "
                   "and falls back to vulkan, then cpu, if it does not load.")
@click.option("--force", is_flag=True, help="Reinstall even when already installed.")
@click.option("--status", is_flag=True, help="Show what is installed and exit.")
def main(backend: str, force: bool, status: bool) -> None:
    """Install the native image generation runtime (stable-diffusion.cpp)."""
    from rich.console import Console
    from rich.markup import escape
    console = Console()
    if status:
        plat = runtime.platform_key()
        console.print(f"stable-diffusion.cpp {pins.TAG} (commit {pins.COMMIT[:7]})")
        console.print(f"Platform: {plat or 'unsupported'}; recommended backend: "
                      f"{runtime.recommended_backend() if plat else '-'}")
        for b in runtime.available_backends(plat):
            rt = runtime.installed(b)
            failed = runtime.load_test_failed(b)
            if rt is not None:
                devs = ", ".join(f"{n} ({d})" for n, d in rt.devices) or "no devices recorded"
                console.print(f"  {b}: installed at {escape(str(rt.path))} - {escape(devs)}")
            elif failed:
                console.print(f"  {b}: did not load here ({escape(failed)})")
            else:
                console.print(f"  {b}: not installed")
        return
    try:
        rt = runtime.provision(backend, force=force,
                               on_progress=lambda m: console.print(f"[dim]{escape(m)}[/dim]"))
    except runtime.ProvisionError as e:
        console.print(f"[red]Could not install the native image runtime: {escape(str(e))}[/red]")
        sys.exit(1)
    console.print(f"[green]Native image runtime installed ({rt.backend}) at "
                  f"{escape(str(rt.path))}[/green]")
