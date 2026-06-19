# SPDX-License-Identifier: AGPL-3.0-or-later
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


def _gui_bind_warning(host: str):
    """Warning text when the GUI binds past loopback without auth, or None when
    the bind is safe. Builds on the server's check, then escalates for the GUI:
    it also exposes the coder agent (shell + file edits) and has no built-in TLS.
    """
    from localm.cli import _exposed_bind_warning
    base = _exposed_bind_warning(host)
    if base is None:
        return None
    return (
        base
        + "\n  The GUI also exposes the coder agent, which can run shell "
        "commands and edit files here. There is no built-in TLS - put it behind "
        "a reverse proxy (see docs/tls.md) for remote access."
    )


def _lan_ip() -> str:
    """Best-effort primary LAN IPv4 of this machine, or "" if undetermined.
    Opens a UDP socket toward a TEST-NET address to learn the outbound interface;
    no packets are actually sent. Used only to PRINT a reachable phone/LAN URL."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 80))      # TEST-NET-1 (RFC 5737); no traffic
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return ""


def _print_qr(url: str) -> None:
    """[PoC] Print a scannable QR of *url* to the console so a phone can open
    localm without typing the address. Experimental; needs the optional 'qrcode'
    dependency (pip install "localm[qr]"). Best-effort and fully guarded: a
    missing dep or a console that cannot render block glyphs degrades to a hint
    and NEVER breaks GUI startup. If it does not scan, your terminal's colours
    may be inverted - try a light-background terminal."""
    import io
    import sys

    from rich.console import Console
    con = Console()
    try:
        import qrcode
    except ImportError:
        con.print('  [yellow][PoC][/yellow] QR needs an optional dep: '
                  r'[cyan]pip install "localm\[qr]"[/cyan]')
        return
    try:
        q = qrcode.QRCode(border=2)
        q.add_data(url)
        q.make(fit=True)
        buf = io.StringIO()
        q.print_ascii(out=buf, invert=True)   # invert scans better on dark terminals
        con.print("  [yellow][PoC - experimental][/yellow] scan to open localm on your phone:")
        # Write the block glyphs as UTF-8 bytes so a legacy console code page
        # (e.g. Windows cp1252) cannot raise UnicodeEncodeError at startup.
        data = buf.getvalue().encode("utf-8", "replace")
        out = getattr(sys.stdout, "buffer", None)
        if out is not None:
            out.write(data)
            out.flush()
        else:
            sys.stdout.write(buf.getvalue())
    except Exception:
        con.print("  [yellow][PoC][/yellow] [dim](could not render the QR in this "
                  "terminal; just open the URL above on your phone)[/dim]")


@click.command("gui")
@click.argument("model", default="", required=False, shell_complete=_complete_model)
@click.option("-H", "--host", default="127.0.0.1", show_default=True,
              help="Bind address. Keep 127.0.0.1 unless you know what you're doing.")
@click.option("-p", "--port", default=None, type=int,
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
@click.option("-c", "--ctx", default=None, type=int, help="Context window size.")
@click.option("-g", "--gpu-layers", default=None, type=int)
@click.option("--no-browser", is_flag=True, help="Don't open the browser automatically.")
@click.option("--no-model", "no_model", is_flag=True,
              help="Open with no model loaded even when the registry has usable "
                   "models. Pick or switch models on the Models page.")
@click.option("--pull", "pull_spec", default=None, metavar="SPEC",
              help="Open the GUI on the Models page and start downloading SPEC "
                   "(a HuggingFace repo, repo:file.gguf, or https URL). Lets you "
                   "fetch a first model with a progress bar, no model required.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (~/.localm/logs/), capture native llama.cpp "
                   "stderr, log requests, and record raw model output in the log.")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'mode', else privacy]. "
                   "privacy = nothing saved; log = JSONL audit of chat traffic; "
                   "full = log + markdown transcript.")
@click.option("--insecure", is_flag=True,
              help="Allow binding past loopback WITHOUT LOCALM_API_KEY set. This "
                   "exposes the unauthenticated coder agent (shell + file edits) "
                   "to the network - only on a trusted, isolated network.")
@click.option("--qr", "show_qr", is_flag=True,
              help="[PoC] Print a scannable QR of the LAN URL at startup so a "
                   "phone can open localm without typing the address. Needs a "
                   "network bind (-H 0.0.0.0) and the optional 'qrcode' dep "
                   "(pip install \"localm[qr]\"). Experimental.")
def main(model, host, port, ctx, gpu_layers, no_browser, no_model, pull_spec, debug, mode, insecure, show_qr):
    """Open the localm web GUI - chat and the coder agent in your browser.

    \b
    MODEL is optional; defaults to the first registered model. With no model
    registered at all (or with --no-model), the GUI still opens so you can add
    or switch models from the Models page (or pass --pull SPEC to start a
    download immediately):
      localm gui
      localm gui gemma4-4b
      localm gui --no-model
      localm gui --pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf
    """
    from rich.console import Console
    console = Console()

    # A click into this console window must not freeze the server
    # (Windows QuickEdit suspends output, and output blocks inference).
    from localm.winconsole import disable_quickedit
    disable_quickedit()

    if debug:
        from localm.debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")

    import os as _os
    from localm.audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        _os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("server")
    if session_mode != SessionMode.PRIVACY:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in ~/.localm/sessions/)[/dim]")
    elif debug:
        console.print(
            "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log records "
            "requests and raw model output - delete it after analysis if that "
            "matters.")

    from localm.config import load_registry, pick_port
    from localm.model_manager import get_model_info, sync_models_dir

    # Pick up models added to (or gone missing from) the models folder since last run.
    _sync = sync_models_dir()
    if _sync.changed:
        _bits = []
        if _sync.added:
            _bits.append(f"{_sync.added} new")
        if _sync.flagged:
            _bits.append(f"{_sync.flagged} missing")
        if _sync.restored:
            _bits.append(f"{_sync.restored} restored")
        if _sync.pruned:
            _bits.append(f"{_sync.pruned} pruned")
        console.print(f"[dim]Models folder synced: {', '.join(_bits)}.[/dim]")
    if _sync.note:
        console.print(f"[yellow]{_sync.note}[/yellow]")

    registry = load_registry()
    model_less = False
    if no_model:
        # Explicit "open with nothing loaded" even when usable models exist; the
        # user picks or switches on the Models page.
        model_less = True
        model = ""
        console.print("[dim]Opening with no model loaded - "
                      "pick one on the Models page.[/dim]")
    elif not model:
        if not registry:
            # Fresh install: open the GUI anyway so the user can add a model
            # from the Models page (or via --pull). No engine until then.
            model_less = True
            console.print("[yellow]No models registered yet.[/yellow] "
                          "Opening the GUI - add one on the Models page"
                          + (" (download starting)…" if pull_spec else "."))
        else:
            # Pick the first entry that still resolves to a real model file or
            # directory, skipping rows whose file is missing or is not a model,
            # so one bad registry entry never blocks startup.
            model = next((n for n in sorted(registry) if get_model_info(n)), None)
            if model is None:
                model_less = True
                console.print(
                    "[yellow]No loadable models in the registry "
                    "(files missing or not a model).[/yellow] "
                    "Opening the GUI - fix or add one on the Models page.")

    model_path = None
    display_name = ""
    if not model_less:
        info = get_model_info(model)
        if info is None:
            console.print(f"[red]Model not found:[/red] {model}")
            sys.exit(1)
        model_path, display_hint = info
        display_name = model if model in registry else display_hint

    chosen_port, was_busy = pick_port(port, host="127.0.0.1" if host == "0.0.0.0" else host)
    if was_busy:
        console.print(f"[yellow]Requested port busy - using {chosen_port}.[/yellow]")

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

    engine = None
    if not model_less:
        try:
            engine = _make_engine(model)
        except Exception as e:
            # A single bad registry entry must not stop the server from starting;
            # degrade to the model-less path and let the user pick on the Models page.
            console.print(f"[yellow]Could not load model '{model}': {e}[/yellow]")
            console.print("[yellow]Opening the GUI model-less - pick a model on "
                          "the Models page.[/yellow]")
            model_less = True
    app = hs.create_app(engine)

    state = {"model": "" if model_less else model}

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

    base_url = f"http://127.0.0.1:{chosen_port}/"
    # Deep-link the browser to the Models page (and a pending download) when
    # the GUI was opened with --pull or with nothing registered yet.
    open_url = base_url
    if pull_spec:
        from urllib.parse import quote
        open_url = f"{base_url}?view=models&pull={quote(pull_spec, safe='')}"
    elif model_less:
        open_url = f"{base_url}?view=models"

    # Refuse to bind past loopback without auth unless explicitly forced: the
    # GUI exposes not just the chat API but the coder agent, which can run shell
    # commands and edit files on this machine.
    bind_warning = _gui_bind_warning(host)
    if bind_warning and not insecure:
        console.print(f"[bold red]{bind_warning}[/bold red]")
        console.print(
            "[bold red]Refusing to start: binding past loopback without auth. "
            "Set $env:LOCALM_API_KEY first, or pass --insecure to override.[/bold red]")
        sys.exit(2)
    if bind_warning:
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print("[bold yellow]  Proceeding anyway (--insecure set).[/bold yellow]")

    console.print(f"[bold green]localm GUI[/bold green] → {base_url}")
    if model_less:
        console.print("  model: [yellow]none yet - add one on the Models page[/yellow]")
    else:
        console.print(f"  model: [cyan]{display_name or Path(str(model_path)).stem}[/cyan]")
    console.print("  Ctrl+C to stop")

    # Phone / LAN access. The GUI is an installable PWA, so a phone just opens
    # this URL and adds it to the home screen. Bound to loopback, it is only
    # reachable on this machine; bound to the network, print the address a phone
    # on the same Wi-Fi can open. See docs/phone.md (Tailscale for off-LAN use).
    if host in ("127.0.0.1", "localhost", "::1"):
        console.print(
            "  [dim]use from your phone: bind to your network with "
            "[/dim][cyan]localm gui -H 0.0.0.0[/cyan][dim] (set LOCALM_API_KEY "
            "first); then see docs/phone.md[/dim]")
        if show_qr:
            console.print(
                "  [yellow][PoC][/yellow] [dim]--qr needs a network bind to be "
                "scannable: [/dim][cyan]localm gui -H 0.0.0.0 --qr[/cyan]")
    else:
        _ip = _lan_ip()
        if _ip:
            phone_url = f"http://{_ip}:{chosen_port}/"
            console.print(
                f"  [dim]from a phone on this network:[/dim] "
                f"[cyan]{phone_url}[/cyan] "
                "[dim](open it, then Install as app)[/dim]")
            if show_qr:
                _print_qr(phone_url)

    # Preload the model in the background so the first chat reply is fast.
    # Engine.load is lock-protected; a request arriving mid-load waits on it.
    def _preload():
        try:
            engine.load()
        except Exception as e:
            console.print(f"[yellow]Background model load failed: {e}[/yellow]")

    if engine is not None:
        threading.Thread(target=_preload, daemon=True, name="preload").start()

    if not no_browser:
        # Delay slightly so the server is listening when the tab opens
        threading.Timer(1.0, lambda: webbrowser.open(open_url)).start()

    # Record the bind host so the SPA-shell route knows whether every client is
    # loopback (a 127.0.0.1 bind) and can safely seed the API key into the page.
    app.state.bind_host = host

    import uvicorn
    try:
        uvicorn.run(app, host=host, port=chosen_port, log_level="warning")
    finally:
        manager.close_all()
