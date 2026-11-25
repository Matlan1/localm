# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hidden commands the update mechanism runs (``localm.update_mechanism``)."""

import json
import sys
from pathlib import Path

import click

from ._core import main


@main.command("_update-prepare", hidden=True)
@click.option("--root", required=True, type=click.Path(path_type=Path))
@click.option("--id", "update_id", required=True)
@click.option("--uv", required=True)
def update_prepare_cmd(root: Path, update_id: str, uv: str) -> None:
    """Fetch and test what the steps of an update of the install at ROOT to this tree
    need (``update_mechanism.steps_update.prepare``)."""
    from localm.cli import update_prepare as prepared
    sys.exit(prepared.prepare(root, update_id, uv, say=lambda m: print(m, flush=True)))


@main.command("_update-probe", hidden=True)
def update_probe_cmd() -> None:
    """Serve the app with its GUI on a loopback port, without a model and without its
    background services, in a session mode that writes nothing on its own. Prints
    ``{"port": N}`` once it listens and returns when standard input ends
    (``update_mechanism.steps_probe``)."""
    import os

    from localm import audit
    os.environ[audit.MODE_ENV_VAR] = audit.SessionMode.PRIVACY.value
    try:
        _serve_probe()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{type(e).__name__}: {e}", flush=True)
        sys.exit(1)


def _serve_probe() -> None:
    import socket
    import threading

    import uvicorn

    from localm.inference import http_server
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = http_server.create_app(None)
    app.state.bind_host, app.state.instance_port = "127.0.0.1", port
    app.state.instance_scheme = "http"
    http_server.mount_gui_surface(app)
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning",
                                           access_log=False))

    def until_closed() -> None:
        sys.stdin.read()
        server.should_exit = True

    threading.Thread(target=until_closed, daemon=True).start()
    sys.stdout.write(json.dumps({"port": port}) + "\n")
    sys.stdout.flush()
    server.run(sockets=[sock])


@main.command("_update-record", hidden=True)
@click.option("--from", "old", required=True)
@click.option("--to", "new", required=True)
@click.option("--kept", default="[]")
def update_record_cmd(old: str, new: str, kept: str) -> None:
    """Record what an update that is done changed."""
    from localm.cli import update_prepare as prepared
    try:
        problems = prepared.record(old, new, json.loads(kept))
    except (ValueError, RecursionError) as e:
        problems = [f"the kept runtimes: {e}"]
    for problem in problems:
        print(problem, flush=True)
    sys.exit(1 if problems else 0)
