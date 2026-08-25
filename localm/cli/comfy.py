# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy` - manage localm's OWN (optional) ComfyUI instance, and the
lifecycle of whichever ComfyUI localm is targeting.

`status` reports whether a managed instance exists, which ComfyUI localm
targets, and (by default) whether that ComfyUI is actually running. `remove`
deletes the managed instance under the localm data dir (reversible,
self-contained). `setup` provisions one: it REPLICATES the user's existing
ComfyUI when they have one (stage S2, the COPY path - clone at the same commit,
a fresh localm venv, the same packages, shared models), or installs a FRESH,
hardware-matched ComfyUI when they do not (stage S3 - a pinned ComfyUI, the
PyTorch build for their GPU, and the custom nodes localm's workflows need). The
user's own ComfyUI is never touched.

`start` / `stop` / `restart` drive the PROCESS, and they go through the running
localm server rather than acting in-process. That is not indirection for its own
sake: the handle on a ComfyUI localm launched lives in `comfy_client`'s
process-local `_spawned_procs`, so a fresh CLI process calling `stop_comfy()`
directly finds no handle, leaves a server-launched ComfyUI running with its VRAM
held, and reports "localm did not launch this ComfyUI" - a false statement about
the very thing the user asked about. Only the process that spawned it can end
it, so the CLI asks that process. This is the same discover-instance-and-POST
shape `localm stop` and `localm unload` already use.
"""

import time
from pathlib import Path

import click

from ._core import (console, main, no_server_message, report_server_failure,
                    running_server, server_call)
from ..media_workflows import MEDIA_TYPES

# Each client timeout must exceed the SERVER-side budget for the same call, or
# the CLI gives up while the server is still working and reports a failure that
# did not happen. The server bounds status at _COMFY_STATUS_TIMEOUT_S (15s) and
# stop at _COMFY_STOP_TIMEOUT_S (90s) in localm/inference/routes/config.py.
_STATUS_TIMEOUT = 20.0
_STOP_TIMEOUT = 100.0
# Transport slack added on top of a launch budget that is computed, never
# guessed - see _launch_timeout below.
_LAUNCH_SLACK = 30.0

# The image plugin serves its generation routes under /api/imagine, music and
# video under their own names; comfy-launch lives under the generation prefix in
# all three (see the GUI's own GENERATE_PREFIX in static/pages/workflow.js).
_LAUNCH_PREFIX = {"image": "imagine", "music": "music", "video": "video"}


def _launch_timeout(cfg) -> float:
    """How long to wait on a launch, derived from the SAME configured budget
    ensure_comfy will actually honour.

    comfy_launch_timeout exists because a ZLUDA/ROCm cold start compiles GPU
    kernels and can take minutes. An independent client-side guess would drift
    below a user's own setting and abort a launch that was still legitimately
    progressing, so read the one number rather than inventing a second."""
    from ..media.comfy_client import comfy_launch_wait_seconds
    return comfy_launch_wait_seconds(cfg) + _LAUNCH_SLACK * 2


def _restart_timeout(cfg) -> float:
    """The restart route's own budget plus slack: it stops (bounded at the
    server's 90s stop budget) and THEN launches, so a client that only allowed
    for the launch half would abandon a restart mid-flight."""
    return _STOP_TIMEOUT + _launch_timeout(cfg)


def _live_status(server) -> tuple:
    """(state, payload) for GET /v1/comfy/status on the discovered *server*."""
    url, headers = server
    return server_call(url, headers, "GET", "/v1/comfy/status",
                       timeout=_STATUS_TIMEOUT)


@main.group("comfy")
def comfy_group() -> None:
    """localm's own managed ComfyUI (optional, off by default).

    localm can run its OWN ComfyUI under the data folder instead of depending on
    your install - so it can pin a known-good version and carry fixes. Off by
    default (inert until you set one up); your own ComfyUI is never modified.
    Run 'localm comfy setup', then leave comfy_target on its default 'own' (or
    set it in Settings -> Media) to route media to it.
    """


def _print_process_status(api_url: str, ping: bool) -> None:
    """The RUNNING half of `comfy status`: is that ComfyUI up, and can localm
    control it.

    Asks the running localm server, because `launched_by_localm` is only
    knowable inside the process that did the launching (see this module's
    docstring). With no server to ask, fall back to a direct liveness probe -
    which answers "is it up" perfectly well, and answers the control question
    not at all. Those two are printed as different things: "no" would be a
    claim, and the honest word is "unknown".
    """
    console.print()
    console.print("[bold]ComfyUI process[/bold]")
    if not ping:
        console.print("  Running           : [dim]not checked (--no-ping)[/dim]")
        return

    server = running_server()
    if server is not None:
        state, payload = _live_status(server)
        if state == "ok":
            alive = bool((payload or {}).get("alive"))
            ours = bool((payload or {}).get("launched_by_localm"))
            console.print("  Running           : "
                          + ("[green]yes[/green]" if alive else "[yellow]no[/yellow]"))
            if not alive:
                console.print("  [dim]Start it with[/dim] localm comfy start")
                return
            console.print(f"  Launched by localm: {'yes' if ours else 'no'}")
            if ours:
                console.print("  [dim]localm can[/dim] localm comfy stop  [dim]or[/dim]  "
                              "localm comfy restart")
            else:
                console.print("  [dim]You started this one, so[/dim] localm comfy stop "
                              "[dim]aborts its render and frees its VRAM but leaves "
                              "the process running.[/dim]")
            return
        # Could not ask the server. Say so, then still answer what a direct
        # probe CAN answer rather than printing nothing at all.
        report_server_failure(state, payload, "ask the localm server about ComfyUI")

    _print_direct_probe(api_url, had_server=server is not None)


def _print_direct_probe(api_url: str, *, had_server: bool) -> None:
    """Liveness straight from ComfyUI, with the control question left open.

    A direct probe is a real answer to "is it running" - it is the same
    /system_stats call the server's own route makes. It is NOT an answer to
    "did localm launch it": the only process that knows is the one holding the
    subprocess handle, so reporting `no` here would be inventing a negative out
    of not having asked.
    """
    from ..media.comfy_client import _comfy_alive

    alive = _comfy_alive(api_url, timeout=2.0)
    console.print("  Running           : "
                  + ("[green]yes[/green]" if alive else "[yellow]no[/yellow]"))
    reason = ("the localm server could not be asked"
              if had_server else "no localm server is running for this directory")
    console.print(f"  Launched by localm: [dim]unknown - {reason}, and only the "
                  "process that launched ComfyUI knows[/dim]")
    if not had_server:
        console.print("  [dim]Start one with[/dim] localm gui  [dim]to stop or restart "
                      "ComfyUI from here.[/dim]")


@comfy_group.command("status")
@click.option("--ping/--no-ping", "ping", default=True, show_default=True,
              help="Also check whether the targeted ComfyUI is actually running. "
                   "--no-ping keeps this command offline and instant.")
def comfy_status(ping: bool) -> None:
    """Show whether a managed ComfyUI is installed, where, which ComfyUI localm
    currently targets, and whether that ComfyUI is running."""
    from rich.markup import escape

    from ..config import load_config
    from ..media.managed_comfy import (
        MANAGED_COMFY_API_URL, is_managed_comfy_installed, managed_comfy_paths,
        resolve_comfy_target,
    )

    cfg = load_config()
    paths = managed_comfy_paths()
    installed = is_managed_comfy_installed()

    console.print("[bold]Managed ComfyUI[/bold]")
    # comfy_target is a SELECT field (options own/user) when set through the
    # GUI/CLI config setters, but config.json can be hand-edited, so this print
    # does not rely on that validation holding - same defense-in-depth stance
    # as rag.py's already-safe collection names (escape() is a no-op on "own").
    console.print(f"  Preferred target  : {escape(str(cfg.get('comfy_target', 'own')))}")
    if installed:
        # paths.root/models_dir are <LOCALM_HOME>/comfyui(-models) - LOCALM_HOME
        # itself is user-configurable (env var / --home), so these are not
        # provably bracket-free.
        console.print(f"  Installed         : yes, at {escape(str(paths.root))}")
        console.print(f"  Managed API URL   : {MANAGED_COMFY_API_URL}")
        console.print(f"  Managed models    : {escape(str(paths.models_dir))}")
    else:
        console.print("  Installed         : no - not set up "
                      "(run 'localm comfy setup' to replicate your ComfyUI)")

    target = resolve_comfy_target(cfg)
    which = "localm's managed ComfyUI" if target.managed else "your own ComfyUI"
    # target.api_url is comfy_api_url (admin-set free text) when not managed -
    # genuinely user-controlled, unlike MANAGED_COMFY_API_URL above (a fixed
    # loopback constant derived only from MANAGED_COMFY_PORT).
    console.print(f"  Target now        : {which} ({escape(target.api_url)})")

    _print_process_status(target.api_url, ping)


@comfy_group.command("start")
@click.option("--media", "media", default=None,
              type=click.Choice(sorted(_LAUNCH_PREFIX)),
              help="Launch with this media plugin's own ComfyUI settings "
                   "(workdir / launch command). Omit to try image, then music, "
                   "then video, and fall back to the global comfy_launch_cmd.")
def comfy_start(media) -> None:
    """Start ComfyUI without running a generation.

    Uses the same launch path a real generation would (comfy_launch_cmd /
    comfy_workdir, or the managed instance's own), so what starts here is
    exactly what a later `localm image` would have started. A cold ROCm/ZLUDA
    start compiles GPU kernels and can take minutes; this waits for the
    configured comfy_launch_timeout.

    Already running? This says so and changes nothing - it never interrupts a
    render in progress.
    """
    from ..config import load_config

    server = running_server()
    if server is None:
        no_server_message("starting ComfyUI")
        raise SystemExit(1)
    url, headers = server

    # Ask first, and stop here when it is already up. The fallback below
    # (/v1/comfy/restart) would otherwise abort whatever ComfyUI is currently
    # rendering, which is the opposite of what "start" promises.
    state, payload = _live_status(server)
    if state == "ok" and (payload or {}).get("alive"):
        console.print("[green]ComfyUI is already running.[/green]")
        console.print("[dim]Restart it with[/dim] localm comfy restart")
        return
    if state not in ("ok", "unsupported"):
        report_server_failure(state, payload, "ask the localm server about ComfyUI")
        raise SystemExit(1)
    # POSITIVELY established down, as opposed to merely not established up. The
    # per-plugin launch routes are safe either way (ensure_available only brings
    # ComfyUI up), but the kernel fallback below goes through /v1/comfy/restart,
    # whose stop half ABORTS an in-flight render. That one is only allowed on a
    # confirmed-down ComfyUI. An "unsupported" status read is a server too old
    # to have the route; it is not evidence that nothing is rendering.
    confirmed_down = state == "ok" and not (payload or {}).get("alive")

    cfg = load_config()
    timeout = _launch_timeout(cfg)
    console.print("[dim]Starting ComfyUI (a cold start can take minutes)...[/dim]")

    order = [media] if media else ["image", "music", "video"]
    for kind in order:
        st, pl = server_call(url, headers, "POST",
                             f"/api/{_LAUNCH_PREFIX[kind]}/comfy-launch",
                             timeout=timeout)
        if st != "unsupported":
            _report_action(st, pl, "start ComfyUI")
            return
        # 404: that media plugin is not installed on this server. Try the next.

    if media:
        console.print(f"[red]The {media} plugin is not installed on this server[/red], "
                      "so it has no launch route.")
        console.print("[dim]Install it with[/dim] localm plugin install "
                      f"{media}[dim], or omit --media.[/dim]")
        raise SystemExit(1)

    # No media plugin is installed, so no per-plugin launch route exists. The
    # kernel restart route launches from the GLOBAL comfy_launch_cmd, and with
    # ComfyUI confirmed down above its stop half has nothing to abort.
    if not confirmed_down:
        console.print("[yellow]![/yellow]  No media plugin is installed, and this "
                      "server cannot say whether ComfyUI is already running.")
        console.print("[dim]The only remaining way to start it from here also "
                      "aborts any render in progress, so it is not used blind. "
                      "Install a media plugin, or start ComfyUI yourself.[/dim]")
        raise SystemExit(1)
    console.print("[dim]No media plugin is installed - launching from the global "
                  "comfy_launch_cmd instead of a plugin's own.[/dim]")
    st, pl = server_call(url, headers, "POST", "/v1/comfy/restart",
                         timeout=_restart_timeout(cfg))
    _report_action(st, pl, "start ComfyUI")


@comfy_group.command("stop")
def comfy_stop() -> None:
    """Stop ComfyUI: abort the in-flight render, clear the queue, free its VRAM.

    When localm launched it, its process is also ended. When YOU launched it,
    localm only aborts and frees - it never kills a process it did not start,
    and says so rather than implying it stopped something it did not.
    """
    server = running_server()
    if server is None:
        no_server_message("stopping ComfyUI")
        raise SystemExit(1)
    url, headers = server
    console.print("[dim]Stopping ComfyUI...[/dim]")
    st, pl = server_call(url, headers, "POST", "/v1/comfy/stop",
                         timeout=_STOP_TIMEOUT)
    _report_action(st, pl, "stop ComfyUI")


@comfy_group.command("restart")
def comfy_restart() -> None:
    """Restart the ComfyUI localm launched: stop it, then launch a fresh one.

    Only meaningful for a ComfyUI localm started (see `localm comfy status`);
    for one you started yourself, the stop half aborts its render but its
    process is left alone, so the relaunch finds it already up.
    """
    from ..config import load_config

    server = running_server()
    if server is None:
        no_server_message("restarting ComfyUI")
        raise SystemExit(1)
    url, headers = server
    console.print("[dim]Restarting ComfyUI (a cold start can take minutes)...[/dim]")
    st, pl = server_call(url, headers, "POST", "/v1/comfy/restart",
                         timeout=_restart_timeout(load_config()))
    _report_action(st, pl, "restart ComfyUI")


def _report_action(state, payload, what: str) -> None:
    """Print the outcome of a comfy lifecycle POST, and exit 1 on failure.

    The routes answer 200 with ``{"ok": false, "message": ...}`` for a refusal
    they handled cleanly, so a 2xx is NOT by itself success - reading only the
    status code here would report a failed stop as done, which is the exact
    shape of a discarded return value.
    """
    from rich.markup import escape

    if state != "ok":
        report_server_failure(state, payload, what)
        raise SystemExit(1)
    ok = bool((payload or {}).get("ok"))
    message = (payload or {}).get("message") or (
        f"Asked the server to {what}." if ok else f"Could not {what}.")
    colour = "green" if ok else "red"
    console.print(f"[{colour}]{escape(str(message))}[/{colour}]")
    if not ok:
        raise SystemExit(1)


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
    from rich.markup import escape

    from ..media.managed_comfy import (managed_comfy_remove_targets,
                                       remove_managed_comfy)

    targets = managed_comfy_remove_targets(with_models)
    if not targets:
        console.print("[dim]Nothing to remove - no managed ComfyUI is installed.[/dim]")
        return

    # targets are Path objects under LOCALM_HOME, which is user-configurable -
    # same reasoning as comfy_status's paths.root/models_dir above.
    listing = "\n".join(f"  {escape(str(t))}" for t in targets)
    console.print(f"This will delete:\n{listing}")
    if not yes and not click.confirm("Remove it?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return

    # remove_managed_comfy is the shared removal (also used by the GUI route); it
    # reports any path it could NOT delete (rule 5) instead of claiming success.
    # Each failed entry is "<path> (<OSError>)" - both halves can carry the
    # same LOCALM_HOME-derived path text as `targets` above.
    _, failed = remove_managed_comfy(with_models)
    if failed:
        console.print("[red]Could not remove:[/red]\n  "
                      + "\n  ".join(escape(str(f)) for f in failed))
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

    from rich.markup import escape

    from ..config import load_config
    from ..media import managed_comfy_fresh as fresh
    from ..media import managed_comfy_provision as prov
    from ..model_manager import _emit_outcome

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
        _emit_outcome("failed")
        console.print(f"[red]{escape(result.message)}[/red]")
        raise SystemExit(1)

    # Emitted BEFORE any of the prints below: real work (clone/install/venv/
    # marker) is already done by this point, so a crash in one of these purely
    # cosmetic status lines - the exact class of bug pull.py's _report_success
    # exists to guard against - must not un-say a completed install to the GUI
    # job runner, which otherwise infers status from the exit code alone.
    _emit_outcome("done")
    console.print(f"[green]{escape(result.message)}[/green]")
    if result.status == "copied":
        console.print(f"  Packages replicated    : {result.installed_packages}")
        console.print(f"  Custom nodes copied    : {result.custom_nodes_copied}")
    elif result.status == "fresh":
        console.print(f"  Custom nodes installed : {result.custom_nodes_copied}")
    if cfg.get("comfy_target", "own") == "own":
        console.print("localm will now route media to it (comfy_target is "
                      "already 'own', the default).")
    else:
        console.print("Switch to it: localm config comfy_target own "
                      "(currently set to 'user').")


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
    from rich.markup import escape

    from ..config import load_config
    from ..media import managed_comfy as mc
    from ..media import managed_comfy_update as upd
    from ..media.managed_comfy_fresh import (COMFYUI_PINNED_COMMIT,
                                             COMFYUI_PINNED_VERSION)
    from ..model_manager import _emit_outcome

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
        # commit is raw --commit user input (unlike COMFYUI_PINNED_COMMIT above,
        # a hardcoded module constant), so it needs escaping unlike its sibling.
        console.print(f"Updating to ComfyUI {escape(commit[:12])} (advanced override) and "
                      "re-applying localm's patches...")

    result = upd.update_managed_comfy(
        cfg, target_commit=commit, reinstall_requirements=reinstall_requirements,
        on_progress=lambda line: console.print(line, style="dim", markup=False))

    if not result.ok:
        _emit_outcome("failed")
        console.print(f"[red]{escape(result.message)}[/red]")
        raise SystemExit(1)
    # See comfy_setup's identical comment: real work (checkout/patches/rollback)
    # is already done here, so this signal must land before the cosmetic print
    # that follows it, not after.
    _emit_outcome("done")
    console.print(f"[green]{escape(result.message)}[/green]")


@comfy_group.group("workflow")
def workflow_group() -> None:
    """Manage each media plugin's uploaded ComfyUI workflows (image / music /
    video).

    Each media plugin resolves ITS ACTIVE workflow in this order: the file
    selected here, then a legacy override, then the shipped example template -
    so `localm image`/`music`/`video` are governed by whatever is selected
    here, exactly like the GUI's own picker on each media page. Works fully
    offline; no running server needed.
    """


def _fmt_wf_size(n: int) -> str:
    """A short size for a workflow JSON file - these run tens of KB, never the
    GB range `localm list`'s model-file formatting assumes."""
    if n >= 1024 * 1024:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _fmt_wf_age(seconds) -> str:
    """A coarse human duration for the 'Modified' column - mirrors
    localm.cli.keys._fmt_age's granularity (this is "how long ago", not a
    stopwatch)."""
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m ago"
    return f"{s // 86400}d ago"


@workflow_group.command("list")
@click.argument("media", type=click.Choice(MEDIA_TYPES))
def workflow_list(media: str) -> None:
    """List MEDIA's uploaded workflows and show which one is active."""
    from rich.markup import escape

    from ..media_workflows import list_workflows, selected_name

    # media is click.Choice(MEDIA_TYPES)-validated (always "image"/"music"/
    # "video"), so it is safe unescaped throughout this function. active/w["name"]
    # are user-chosen workflow filenames: save_workflow only enforces
    # path-traversal safety (pathsafe.confined_name), never a safe charset, so a
    # name like "flux[draft].json" is a genuinely accepted, real filename.
    active = selected_name(media)
    console.print(f"[bold]{media} workflow[/bold]: "
                 + (f"[green]{escape(active)}[/green]" if active
                    else "[dim](built-in default - none selected)[/dim]"))
    items = list_workflows(media, active=active)
    if not items:
        console.print("[dim]No uploaded workflows. Add one with[/dim] "
                      f"localm comfy workflow add {media} <file.json>")
        return

    from rich.table import Table
    table = Table(header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Active")
    table.add_column("Size")
    table.add_column("Modified")
    now = time.time()
    for w in items:
        # Table cell strings go through the SAME markup parsing as
        # console.print f-strings (confirmed empirically), not just plain text.
        table.add_row(escape(w["name"]), "[green]yes[/green]" if w["is_active"] else "",
                      _fmt_wf_size(w["size_bytes"]), _fmt_wf_age(now - w["mtime"]))
    console.print(table)


@workflow_group.command("add")
@click.argument("media", type=click.Choice(MEDIA_TYPES))
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", "name", default=None,
              help="Stored filename [default: FILE's own name].")
@click.option("--use", "activate", is_flag=True,
              help="Also select it as MEDIA's active workflow.")
def workflow_add(media: str, file: Path, name, activate: bool) -> None:
    """Upload a ComfyUI workflow (in ComfyUI: Save > API format) for MEDIA.

    \b
    Examples:
      localm comfy workflow add image my_flux.json --use
      localm comfy workflow add music alt_ace.json

    Validates FILE is ComfyUI's API-format JSON (a dict of nodes, each with a
    class_type) before storing it, so a bad upload fails now instead of every
    later generation.
    """
    from rich.markup import escape

    from ..media_workflows import save_workflow, select_workflow

    content = file.read_bytes()
    try:
        saved = save_workflow(media, name or file.name, content)
        if activate:
            select_workflow(media, saved)
    except ValueError as e:
        raise click.ClickException(str(e))
    # saved is the stored (user-chosen) filename - see workflow_list's comment.
    console.print(f"[green]Saved[/green] {media} workflow [bold]{escape(saved)}[/bold]"
                 + (" and selected it." if activate else "."))
    if not activate:
        console.print("[dim]Select it with[/dim] "
                      f"localm comfy workflow use {media} {escape(saved)}")


@workflow_group.command("use")
@click.argument("media", type=click.Choice(MEDIA_TYPES))
@click.argument("name", required=False)
@click.option("--clear", is_flag=True,
              help="Clear the selection instead (fall back to the built-in "
                   "default), rather than passing NAME.")
def workflow_use(media: str, name, clear: bool) -> None:
    """Select MEDIA's active workflow (or clear it with --clear).

    Governs `localm image`/`music`/`video` too, not just the GUI's own picker
    on the same page. (The same selection `localm plugin config <media>
    workflow <name>` writes - this is the discoverable, dedicated form of it.)
    """
    from rich.markup import escape

    from ..media_workflows import select_workflow

    if clear:
        if name:
            raise click.UsageError("Pass either NAME or --clear, not both.")
    elif not name:
        raise click.UsageError(
            "Missing argument NAME (or pass --clear to fall back to the "
            "built-in default).")
    try:
        selected = select_workflow(media, None if clear else name)
    except ValueError as e:
        raise click.ClickException(str(e))
    if selected:
        # selected is the user-chosen workflow filename - see workflow_list.
        console.print(f"[green]{media} now uses[/green] [bold]{escape(selected)}[/bold]")
    else:
        console.print(f"[green]{media} cleared[/green] - falling back to the "
                      "built-in default.")


@workflow_group.command("rm")
@click.argument("media", type=click.Choice(MEDIA_TYPES))
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def workflow_rm(media: str, name: str, yes: bool) -> None:
    """Delete an uploaded MEDIA workflow (refuses to delete the active one -
    select another first)."""
    from rich.markup import escape

    from ..media_workflows import delete_workflow

    if not yes:
        click.confirm(f"Delete {media} workflow '{name}'? This can't be undone.",
                      abort=True)
    try:
        delete_workflow(media, name)
    except ValueError as e:
        raise click.ClickException(str(e))
    # name is the user-supplied workflow filename argument - see workflow_list.
    console.print(f"[green]Deleted[/green] {media} workflow [bold]{escape(name)}[/bold].")
