# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys

import click

from ..config import load_config
from ..model_manager import get_model_info
from ._core import (
    console, main, _complete_model_name,
    _exposed_bind_warning, _setup_tls_or_exit,
)


# ------------------------------------------------------------------ #
#  serve                                                               #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-H", "--host",        default="127.0.0.1", help="Bind address (0.0.0.0 for LAN).")
@click.option("-p", "--port",        default=None,        type=click.IntRange(1, 65535),
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
@click.option("-c", "--ctx",         default=None,        type=int)
@click.option("-g", "--gpu-layers",  default=None,        type=click.IntRange(0, 1000))
@click.option("--mmproj",            default=None)
@click.option("--device",            default=None)
@click.option("--no-tls", is_flag=True,
              help="Serve plain HTTP even on a network bind. Built-in TLS is on "
                   "by default past loopback; this disables it (the API key then "
                   "crosses the network in cleartext - only on a trusted LAN).")
@click.option("--tls-cert", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Use this certificate (PEM) instead of localm's built-in "
                   "local-CA cert. Requires --tls-key.")
@click.option("--tls-key", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Private key (PEM) for --tls-cert.")
@click.option("--insecure", is_flag=True,
              help="Allow binding past loopback WITHOUT LOCALM_API_KEY set. This "
                   "serves the API (and any enabled plugin routes) "
                   "unauthenticated to the whole network - only on a trusted, "
                   "isolated network.")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Project root that keys this instance [default: nearest "
                   ".git/.localcoder above the current directory].")
@click.option("--new", "force_new", is_flag=True,
              help="Start a fresh server even if one is already serving this "
                   "project (by default 'serve' reports the running one).")
@click.option("--isolated", is_flag=True,
              help="Start a private server invisible to discovery (test safety). "
                   "Implies --new.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (~/.localm/logs/), capture native llama.cpp "
                   "stderr, and log requests.")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'mode', else privacy]. "
                   "privacy = nothing saved; log = JSONL audit of chat traffic; "
                   "full = log + markdown transcript.")
def serve(model, host, port, ctx, gpu_layers, mmproj, device, no_tls, tls_cert,
          tls_key, insecure, project, force_new, isolated, debug, mode):
    """Start an OpenAI-compatible inference server.

    \b
    POST http://<host>:<port>/v1/chat/completions
    GET  http://<host>:<port>/v1/models
    GET  http://<host>:<port>/health

    Compatible with any OpenAI client library.
    """
    # A click into this console window must not freeze the server
    # (Windows QuickEdit suspends output, and output blocks inference).
    from ..winconsole import disable_quickedit
    disable_quickedit()

    if debug:
        from ..debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")

    from ..audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("server")
    if session_mode != SessionMode.PRIVACY:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in ~/.localm/sessions/)[/dim]")
    elif debug:
        console.print(
            "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log records "
            "requests and raw model output - delete it after analysis if that "
            "matters.")

    # Refuse a keyless network bind before any other work (fail fast, and match
    # `localm gui`, which already refuses): binding past loopback with no API key
    # serves the OpenAI API - and any enabled plugin routes (e.g. the coder
    # agent's history) - unauthenticated to the whole network. --insecure
    # overrides for a trusted isolated network. (Security review 2026-06-20: this
    # closes a serve-vs-gui fail-open asymmetry.)
    bind_warning = _exposed_bind_warning(host)
    if bind_warning and not insecure:
        console.print(f"[bold red]{bind_warning}[/bold red]")
        console.print(
            "[bold red]Refusing to start: binding past loopback without auth. "
            "Set $env:LOCALM_API_KEY first, or pass --insecure to override.[/bold red]")
        sys.exit(2)
    if bind_warning:
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print("[bold yellow]  Proceeding anyway (--insecure set).[/bold yellow]")

    info = get_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)

    model_path, _display_hint = info

    # Attach-or-spawn (H6 phase 4): if a localm is already serving this project,
    # do not spin a second server that double-loads the model - report it.
    from .. import instances
    from ..config import home_dir
    _root_dir = instances.resolve_root_dir(override=project)
    if not (force_new or isolated):
        _existing = instances.find_attachable(home_dir(), _root_dir)
        if _existing:
            _url = instances.attach_url(_existing)
            console.print(
                f"[bold]localm is already serving[/bold] [cyan]{_root_dir}[/cyan] "
                f"at [bold]{_url}v1[/bold] (pid {_existing.get('pid')}, "
                f"mode {_existing.get('mode')}).")
            console.print(
                "  [dim]Use that endpoint, or pass [/dim][cyan]--new[/cyan]"
                "[dim] to start a separate server.[/dim]")
            return

    from ..inference.engine import Engine
    from ..inference.http_server import serve as http_serve
    from ..model_manager import load_registry as _reg

    display_name = model if model in _reg() else _display_hint

    from ..config import pick_port
    requested = port
    port, was_busy = pick_port(requested, host="127.0.0.1" if host == "0.0.0.0" else host)
    if was_busy:
        console.print(
            f"[yellow]Port {requested if requested is not None else load_config().get('port', 8642)} "
            f"is in use - serving on {port} instead.[/yellow]"
        )

    # Built-in TLS: encrypt the bind out of the box past loopback (NET-1).
    ssl_certfile, ssl_keyfile = _setup_tls_or_exit(
        host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    scheme = "https" if ssl_certfile else "http"

    engine = Engine(
        str(model_path),
        mmproj_path=mmproj,
        n_ctx=ctx,
        n_gpu_layers=gpu_layers,
        device=device,
        display_name=display_name,
    )

    engine.load()

    # Reach-by-name (mDNS): advertise <mdns_name>.local on a network bind so other
    # devices reach this server by name. Started HERE, before printing, so the
    # printed name reflects reality: we only recommend <name>.local when it is
    # actually being advertised (not on a loopback / --isolated bind, not when mDNS
    # is off, and not when the name is already taken - start_advertiser returns None
    # in each case). The handle is closed in the finally below.
    from localm import netname
    mdns_advertiser = None
    if host not in ("127.0.0.1", "localhost", "::1") and not isolated:
        mdns_advertiser = netname.start_advertiser(port, tls=bool(ssl_certfile))
    _adv_name = netname.mdns_fqdn() if mdns_advertiser is not None else None

    # 0.0.0.0 / :: are bind wildcards, not connectable addresses - print the
    # address a client can actually reach (loopback for a wildcard bind, else the
    # bind host itself) instead of a dead 0.0.0.0 URL.
    _primary = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    console.print(
        f"[green]✓[/green] Serving at "
        f"[bold]{scheme}://{_primary}:{port}/v1/chat/completions[/bold]"
    )
    # On a network bind, print the reachable NAMES (localm.local when advertised,
    # the Tailscale MagicDNS name) and IPs so other devices need no address typed
    # by hand.
    if host not in ("127.0.0.1", "localhost", "::1"):
        for _label, _target in netname.network_targets(mdns_name=_adv_name):
            console.print(
                f"[dim]  {_label}:[/dim] "
                f"[bold]{scheme}://{_target}:{port}/v1/chat/completions[/bold]"
            )
        _ts_hint = netname.tailscale_rename_hint()
        if _ts_hint:
            console.print(f"[dim]  {_ts_hint}[/dim]")
    if scheme == "https":
        console.print("[dim]Built-in TLS (self-signed via localm's local CA). "
                      "First connection from a device shows a one-time trust "
                      "warning; install the CA from /localm-ca.crt to remove it.[/dim]")
    console.print("[dim]Ctrl+C to stop[/dim]\n")

    # SRV-4: closing the console window terminates the process WITHOUT running
    # the finally below, so the native model context used to be freed during
    # interpreter teardown - a segfault on exit. Free it (and clear the crash
    # marker so a clean window-close is not reported as a crash) inside the
    # Windows console handler instead. Idempotent with the finally on Ctrl+C.
    from localm import bugreport, winconsole

    def _on_console_close() -> None:
        try:
            engine.unload()
        finally:
            bugreport.disarm_crash_guard()

    winconsole.register_console_handler(_on_console_close)

    try:
        http_serve(engine, host=host, port=port,
                   ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                   project=project, isolated=isolated)
    except KeyboardInterrupt:
        pass
    finally:
        if mdns_advertiser is not None:
            mdns_advertiser.close()
        engine.unload()
