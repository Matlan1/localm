"""CLI for the localm GUI plugin: ``localm gui``."""

from __future__ import annotations

import asyncio
import sys
import threading
import webbrowser
from pathlib import Path

import click


def _complete_model(ctx, param, incomplete):
    try:
        from localm.config import load_registry
        return sorted(n for n in load_registry() if n.startswith(incomplete))
    except Exception:
        return []


@click.command("gui")
@click.argument("model", default="", required=False, shell_complete=_complete_model)
@click.option("-H", "--host", default="127.0.0.1", show_default=True,
              help="Bind address. Keep 127.0.0.1 unless you know what you're doing.")
@click.option("-p", "--port", default=None, type=int,
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
@click.option("-c", "--ctx", default=None, type=int, help="Context window size.")
@click.option("-g", "--gpu-layers", default=None, type=int)
@click.option("--no-browser", is_flag=True, help="Don't open the browser automatically.")
def main(model, host, port, ctx, gpu_layers, no_browser):
    """Open the localm web GUI — chat and the coder agent in your browser.

    \b
    MODEL is optional; defaults to the first registered model.
    The GUI runs fully offline against your local models:
      localm gui
      localm gui gemma4-4b
    """
    from rich.console import Console
    console = Console()

    from localm.config import load_registry, pick_port
    from localm.model_manager import get_model_info

    registry = load_registry()
    if not model:
        if not registry:
            console.print("[red]No models registered.[/red] "
                          "Pull one first:  localm pull <name>")
            sys.exit(1)
        model = sorted(registry)[0]

    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)
    model_path, display_hint = info
    display_name = model if model in registry else display_hint

    chosen_port, was_busy = pick_port(port, host="127.0.0.1" if host == "0.0.0.0" else host)
    if was_busy:
        console.print(f"[yellow]Requested port busy — using {chosen_port}.[/yellow]")

    from localm.inference.engine import Engine
    from localm.inference import http_server as hs
    from .web import attach_gui

    def _make_engine(name: str) -> Engine:
        m_info = get_model_info(name)
        if m_info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = m_info
        return Engine(
            str(m_path),
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            display_name=name if name in load_registry() else m_hint,
        )

    engine = _make_engine(model)
    app = hs.create_app(engine)

    state = {"model": model}

    async def switch_model(name: str) -> None:
        """Swap engines under the inference semaphore so no request is mid-flight."""
        loop = asyncio.get_running_loop()
        new_engine = _make_engine(name)
        async with hs._inference_sem:
            old = hs._engine
            if old is not None and old.loaded:
                await loop.run_in_executor(None, old.unload)
            await loop.run_in_executor(None, new_engine.load)
            hs._engine = new_engine
            state["model"] = name

    manager = attach_gui(
        app,
        self_url=f"http://127.0.0.1:{chosen_port}/v1",
        switch_model=switch_model,
        active_model=lambda: state["model"],
    )

    url = f"http://127.0.0.1:{chosen_port}/"
    console.print(f"[bold green]localm GUI[/bold green] → {url}")
    console.print(f"  model: [cyan]{display_name or Path(str(model_path)).stem}[/cyan]")
    console.print("  Ctrl+C to stop")

    if not no_browser:
        # Delay slightly so the server is listening when the tab opens
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    try:
        uvicorn.run(app, host=host, port=chosen_port, log_level="warning")
    finally:
        manager.close_all()
