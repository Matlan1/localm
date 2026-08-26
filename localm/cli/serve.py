# SPDX-License-Identifier: AGPL-3.0-or-later

import click

from ._core import (
    main, _complete_model_name,
)


# ------------------------------------------------------------------ #
#  serve                                                               #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-H", "--host",        default=None,
              help="Bind address (0.0.0.0 for LAN) [default: config "
                   "'bind_host' (127.0.0.1)].")
@click.option("-p", "--port",        default=None,        type=click.IntRange(1, 65535),
              help="Port [default: config 'port' (8642), auto-bumps if busy; an "
                   "explicit --port must be free or startup errors].")
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
              help="Write a debug log (<data dir>/logs/), capture native llama.cpp "
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
    from click import get_current_context
    from ..plugins.gui.cli import main as gui_main

    # Invokes the gui plugin's command as 'localm gui --api-mode --no-browser'.
    get_current_context().invoke(
        gui_main,
        model=model,
        host=host,
        port=port,
        ctx=ctx,
        gpu_layers=gpu_layers,
        no_browser=True,
        no_model=False,
        pull_spec=None,
        debug=debug,
        mode=mode,
        insecure=insecure,
        no_tls=no_tls,
        tls_cert=tls_cert,
        tls_key=tls_key,
        show_qr=False,
        project=project,
        force_new=force_new,
        isolated=isolated,
        api_mode=True,
        mmproj=mmproj,
        device=device,
    )
