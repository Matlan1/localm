# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-music``: install the native music runtime (KoboldCpp) and the
default ACE-Step 1.5 models, and check that a short track generates."""

from __future__ import annotations

import sys

import click

from . import pins, runtime


@click.command("setup-music")
@click.option("--backend", "backend", default="auto", show_default=True,
              type=click.Choice(["auto", *runtime.BACKENDS], case_sensitive=False),
              help="Compute backend. 'auto' picks the best one for this machine and falls "
                   "back to vulkan, then cpu, if it does not start.")
@click.option("--no-models", is_flag=True,
              help="Install only the runtime, not the default ACE-Step models.")
@click.option("--no-test", is_flag=True,
              help="Skip generating a short test track after installing.")
@click.option("--force", is_flag=True, help="Reinstall the runtime even when installed.")
@click.option("--status", is_flag=True, help="Show what is installed and exit.")
def main(backend: str, no_models: bool, no_test: bool, force: bool, status: bool) -> None:
    """Install native music generation (ACE-Step 1.5 through KoboldCpp)."""
    from rich.console import Console
    from rich.markup import escape
    console = Console()

    def say(m: str) -> None:
        console.print(f"[dim]{escape(m)}[/dim]")

    if status:
        _status(console)
        return
    from .models import DEFAULT_FILES, DEFAULT_REPO, default_name, resolve_path
    from .music import NativeMusicError, backend_order
    try:
        first = backend_order(backend)[0]
        rt = runtime.ensure_for_backend(first, force=force, on_progress=say)
    except (runtime.ProvisionError, NativeMusicError) as e:
        console.print(f"[red]Could not install the native music runtime: {escape(str(e))}[/red]")
        sys.exit(1)
    console.print(f"[green]Native music runtime installed (KoboldCpp {pins.VERSION}, "
                  f"{rt.build} build) at {escape(str(rt.path))}[/green]")
    if no_models:
        return
    from localm import model_manager as mm
    for comp, fname in DEFAULT_FILES.items():
        if resolve_path(default_name(comp)) is not None:
            continue
        console.print(f"Downloading the default music {comp.replace('_', ' ')} model ({fname})")
        if not mm.pull_model(f"{DEFAULT_REPO}:{fname}"):
            console.print(f"[red]Could not download {escape(fname)}.[/red]")
            sys.exit(1)
    if no_test:
        return
    _test(console, backend, say)


def _status(console) -> None:
    from rich.markup import escape
    from .models import COMPONENTS, default_name, resolve_path
    plat = runtime.platform_key()
    console.print(f"KoboldCpp {pins.TAG} (native music runtime)")
    console.print(f"Platform: {plat or 'unsupported'}; recommended backend: "
                  f"{runtime.recommended_backend() if plat else '-'}")
    for b in runtime.available_builds(plat):
        rt = runtime.installed(b)
        where = f"installed at {escape(str(rt.path))}" if rt else "not installed"
        console.print(f"  {b} build ({', '.join(runtime.BUILD_BACKENDS[b])}): {where}")
    for b in runtime.available_backends(plat):
        failed = runtime.backend_failed(b)
        if failed:
            console.print(f"  {b}: did not work here ({escape(failed)})")
        elif runtime.backend_worked(b):
            console.print(f"  {b}: worked")
    for comp in COMPONENTS:
        p = resolve_path(default_name(comp))
        console.print(f"  default {comp.replace('_', ' ')} model: "
                      f"{escape(str(p)) if p else 'not downloaded'}")


def _test(console, backend: str, say) -> None:
    import tempfile
    from pathlib import Path
    from rich.markup import escape
    from .music import NativeMusicError, ServerError, generate_wav, write_wav
    from .server import stop
    from .models import ModelError
    console.print("Generating a 5 second test track...")
    try:
        data, used = generate_wav({}, backend, {
            "caption": "calm acoustic guitar", "lyrics": "[Instrumental]",
            "instrumental": True, "duration": 5.0, "seed": 1, "stereo": True,
        }, plan=False, on_progress=say)
        with tempfile.TemporaryDirectory() as tmp:
            seconds = write_wav(data, Path(tmp) / "test.wav")
    except (NativeMusicError, ServerError, ModelError, runtime.ProvisionError) as e:
        console.print(f"[red]The test track failed: {escape(str(e))}[/red]")
        sys.exit(1)
    finally:
        stop()
    console.print(f"[green]Native music generation works ({used}): a {seconds:.1f} s test "
                  "track was generated.[/green]")
