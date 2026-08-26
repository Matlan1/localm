# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI for the localm GUI plugin: ``localm gui``."""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path

import click

from localm.netlisten import is_wildcard_host


def _complete_model(ctx, param, incomplete):
    try:
        from localm.config import load_registry
        return sorted(n for n in load_registry() if n.startswith(incomplete))
    except Exception:
        return []


def _report_preload_failure(console, exc: Exception) -> None:
    """The background model-preload thread's failure handler: notify the console
    AND log it. ``console.print`` alone never reaches the debug log file (it is
    not a logging call), so a preload failure with no other symptom would leave
    no trace a bug report could surface. Logged with the full traceback, so it is
    captured like any other failure."""
    console.print(f"[yellow]Background model load failed: {exc}[/yellow]")
    from localm.debuglog import logger
    logger.exception("background model preload failed")


def _mdns_addresses(host: str):
    """Which addresses mDNS should advertise for a bind on *host*, or None to let
    ``netname.start_advertiser`` pick this machine's LAN IPv4 as before.

    A WILDCARD bind answers on every interface, so the LAN IPv4 that
    ``start_advertiser`` finds for itself is reachable and is the right advert -
    including for ``::``, which localm binds dual-stack (see ``netlisten``), so an
    IPv4 client resolving ``<name>.local`` genuinely connects.

    A SPECIFIC literal answers on exactly one address, and advertising any other
    one publishes a name that does not resolve to a listening socket. So that bind
    advertises ITSELF. This is also what makes ``<name>.local`` usable on a
    specific IPv6 bind, where the LAN IPv4 would be pure fiction."""
    return None if is_wildcard_host(host) else [host]


def _should_auto_open_browser(no_browser: bool) -> bool:
    """Whether THIS process's own startup should auto-open a browser tab.

    False whenever the caller passed --no-browser, and ALSO false when this
    process was re-exec'd by a server restart: http_server._do_restart sets
    LOCALM_RESTART_IN_PROGRESS right before os.execv, because the tab the user
    is already looking at shows a reconnect overlay that polls and reloads
    itself in place once this process is back up - opening a second tab here
    would strand that overlay instead of reusing it. The flag is CONSUMED
    (popped), not merely read, so it can never leak into a later, genuinely
    fresh launch that happens to inherit this process's environment."""
    import os
    is_restart = os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None) is not None
    return (not no_browser) and (not is_restart)


def _tray_callbacks(app, hs):
    """Build the (on_restart, on_stop) callables for the tray control surface
    (appface.start_app_face).

    Returns LAZY closures over *app*, not functools.partial-bound values:
    app.state.instance_id/instance_port are set by instances.advertise()
    inside hs.run_advertised(), which is called just below this function's
    own call site - AFTER the tray is wired. A partial would freeze
    instance_id/port at None (their state at wire time); these closures read
    app.state at CALL time instead, and by the time a user can physically click
    Restart/Stop, run_advertised() has entered advertise()'s context and
    populated both.

    appface invokes on_restart/on_stop with NO arguments
    (``threading.Thread(target=self.on_restart)``, appface.py), and both
    hs._do_restart and hs._do_shutdown are keyword-only with None defaults.
    Wiring the bare functions directly makes a tray Restart/Stop call
    disarm_crash_guard(instance_id=None), which clears the LEGACY unscoped
    marker (bugreport.py's per-instance-scoping fallback) and leaves this
    instance's real server-crash.<instance_id>.marker still armed, so the NEXT
    start reports a crash that never happened. _do_restart losing its port the
    same way makes _restart_argv omit ``-p``, so a re-exec'd server can come
    back on a different port and strand the tray/GUI's own open window on a dead
    one. The HTTP routes (routes/admin.py's restart/stop endpoints) pass the
    real instance_id."""
    def on_restart():
        hs._do_restart(instance_id=getattr(app.state, "instance_id", None),
                       port=getattr(app.state, "instance_port", None))

    def on_stop():
        hs._do_shutdown(instance_id=getattr(app.state, "instance_id", None))

    return on_restart, on_stop


def _gui_bind_warning(host: str):
    """Warning text when the GUI binds past loopback without auth, or None when
    the bind is safe. Builds on the server's check, then escalates for the GUI:
    it also exposes the coder agent (shell + file edits). Traffic itself is
    encrypted by built-in TLS on a network bind; the warning is about the coder
    agent's reach, not about cleartext.
    """
    from localm.cli import _exposed_bind_warning
    base = _exposed_bind_warning(host)
    if base is None:
        return None
    return (
        base
        + "\n  The GUI also exposes the coder agent, which can run shell "
        "commands and edit files here - only expose it on a trusted network."
    )


def _mount_remote_gui(entry: dict) -> bool:
    """Ask a running ``api``-mode instance to mount its GUI surface on demand.
    POSTs to its loopback ``/v1/surfaces/gui`` with the instance's own registry
    attach token (a local same-user secret). Returns True on success, False on
    any failure - an older instance without the endpoint, a missing token, or a
    network error - so the caller can fall back to just opening the address."""
    import requests
    scheme = entry.get("scheme") or "http"
    port = entry.get("port")
    token = entry.get("token")
    if not port or not token:
        return False
    # Dial the loopback address that instance bound.
    from localm.bindhost import self_connect_host, url_host
    url = (f"{scheme}://{url_host(self_connect_host(entry.get('host')))}"
           f":{port}/v1/surfaces/gui")
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except Exception:
        verify = False
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                          timeout=5, verify=verify)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _print_qr(url: str) -> None:
    """[PoC] Print a scannable QR of *url* to the console so a phone can open
    localm without typing the address. Experimental; 'qrcode' is a core
    dependency, so this needs no separate install. Best-effort and fully
    guarded: an unexpectedly missing dep (a broken/partial install) or a
    console that cannot render block glyphs degrades to a hint and NEVER
    breaks GUI startup. If it does not scan, your terminal's colours may be
    inverted - try a light-background terminal."""
    import io
    import sys

    from rich.console import Console
    con = Console()
    try:
        import qrcode
    except ImportError:
        con.print('  [yellow][PoC][/yellow] QR unavailable: the "qrcode" '
                  'package is missing from this install ([cyan]pip install '
                  'qrcode[/cyan] to fix it)')
        return
    try:
        q = qrcode.QRCode(border=2)
        q.add_data(url)
        q.make(fit=True)
        buf = io.StringIO()
        q.print_ascii(out=buf, invert=True)
        con.print("  [yellow][PoC - experimental][/yellow] scan to open localm on your phone:")
        # Write the block glyphs as UTF-8 bytes.
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


# gui options that only shape a FRESH server: passing one explicitly is a
# conflict with attaching to an existing instance. Options not listed here are
# compatible with an attach, or are value-aware (model / host / port) and
# handled below.
_ATTACH_CONFLICT_FLAGS = {
    "ctx": "--ctx", "gpu_layers": "--gpu-layers",
    "pull_spec": "--pull", "mode": "--mode", "insecure": "--insecure",
    "no_tls": "--no-tls", "tls_cert": "--tls-cert", "tls_key": "--tls-key",
    "show_qr": "--qr", "api_mode": "--api-mode", "mmproj": "--mmproj",
    "device": "--device",
}


def _explicit(ctx, name: str) -> bool:
    """True when *name* came from the command line (not its default). Lets us tell
    an explicit `--port 8794` apart from the unset default so we only object to
    flags the user actually typed."""
    from click.core import ParameterSource
    try:
        return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    except Exception:
        return False


def _probe_active_model(existing: dict):
    """The running instance's active model id, or None when it cannot be read
    (unreachable / a chat-scoped attach token that cannot GET /v1/models). Used to
    decide whether an explicitly-named `localm gui MODEL` conflicts with what the
    running server actually serves."""
    try:
        from localm.inference.http_engine import remote_model_status
        scheme = existing.get("scheme") or "http"
        from localm.bindhost import self_connect_host, url_host
        _h = url_host(self_connect_host(existing.get("host")))
        base = f"{scheme}://{_h}:{existing.get('port')}/v1"
        return remote_model_status(base, existing.get("token"))[1]
    except Exception:
        return None


def _attach_conflicts(ctx, existing: dict, model: str) -> list:
    """Command-line options the user passed that an attach to *existing* cannot
    honor - each a reason NOT to silently attach. Returns human-readable strings
    (empty list = attaching is fine). port/host/model are value-aware so re-passing
    what the running server already uses is NOT a conflict."""
    conflicts: list = []
    # port / host: a conflict only when the requested value differs from the
    # running instance's.
    if _explicit(ctx, "port"):
        want, have = ctx.params.get("port"), existing.get("port")
        try:
            same = have is not None and int(want) == int(have)
        except (TypeError, ValueError):
            same = False
        if not same:
            conflicts.append(f"--port {want} (the running server is on {have})")
    if _explicit(ctx, "host"):
        want, have = ctx.params.get("host"), existing.get("host")
        if str(want) != str(have):
            conflicts.append(f"--host {want} (the running server bound {have})")
    # model: probe the running instance; a conflict only when its active model is
    # known and different.
    if model and _explicit(ctx, "model"):
        active = _probe_active_model(existing)
        if active and active != model:
            conflicts.append(
                f"model {model} (the running server serves {active})")
    # everything else: an attach cannot set the running server's ctx / gpu-layers
    # / tls / mode, so an explicit pass is a conflict. --mode is never compared
    # and always conflicts.
    for name, flag in _ATTACH_CONFLICT_FLAGS.items():
        if _explicit(ctx, name):
            conflicts.append(flag)
    return conflicts


@click.command("gui")
@click.argument("model", default="", required=False, shell_complete=_complete_model)
@click.option("-H", "--host", default=None,
              help="Bind address [default: config 'bind_host' (127.0.0.1)]. "
                   "Keep 127.0.0.1 unless you know what you're doing.")
@click.option("-p", "--port", default=None, type=click.IntRange(1, 65535),
              help="Port [default: config 'port' (8642), auto-bumps if busy; an "
                   "explicit --port must be free or startup errors].")
@click.option("-c", "--ctx", default=None, type=int, help="Context window size.")
@click.option("-g", "--gpu-layers", default=None, type=click.IntRange(0, 1000))
@click.option("--no-browser", is_flag=True, help="Don't open the browser automatically.")
@click.option("--no-model", "no_model", is_flag=True,
              help="Open with no model loaded even when the registry has usable "
                   "models. Pick or switch models on the Models page.")
@click.option("--pull", "pull_spec", default=None, metavar="SPEC",
              help="Open the GUI on the Models page and start downloading SPEC "
                   "(a HuggingFace repo, repo:file.gguf, or https URL). Lets you "
                   "fetch a first model with a progress bar, no model required.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (<data dir>/logs/), capture native llama.cpp "
                   "stderr, and log requests. Raw model output is recorded too, "
                   "EXCEPT in privacy mode (chat content is never written there).")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'mode', else privacy]. "
                   "privacy = nothing saved; log = JSONL audit of chat traffic; "
                   "full = log + markdown transcript.")
@click.option("--keep-diagnostics", "keep_diagnostics", is_flag=True,
              help="Keep diagnostics (a hang stack trace, restart breadcrumbs, and "
                   "a debug log) even in privacy mode, so a bug report has "
                   "something to attach. Chat content is never recorded in privacy "
                   "mode. Same as the Settings > Privacy toggle, for this run.")
@click.option("--insecure", is_flag=True,
              help="Allow binding past loopback WITHOUT LOCALM_API_KEY set. This "
                   "exposes the unauthenticated coder agent (shell + file edits) "
                   "to the network - only on a trusted, isolated network.")
@click.option("--no-tls", is_flag=True,
              help="Serve plain HTTP even on a network bind. Built-in TLS is on "
                   "by default past loopback; this disables it (the API key then "
                   "crosses the network in cleartext - only on a trusted LAN).")
@click.option("--tls-cert", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Use this certificate (PEM) instead of localm's built-in "
                   "local-CA cert. Requires --tls-key.")
@click.option("--tls-key", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Private key (PEM) for --tls-cert.")
@click.option("--qr", "show_qr", is_flag=True,
              help="[PoC] Print a scannable QR of the LAN URL at startup so a "
                   "phone can open localm without typing the address. Needs a "
                   "network bind (-H 0.0.0.0). Experimental.")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Project root that keys this instance [default: nearest "
                   ".git/.localcoder above the current directory].")
@click.option("--new", "force_new", is_flag=True,
              help="Start a fresh server even if one is already running for this "
                   "project (by default a second 'localm gui' attaches to it).")
@click.option("--isolated", is_flag=True,
              help="Start a private server that is invisible to discovery - "
                   "nothing attaches to it and it attaches to nothing (test "
                   "safety). Implies --new.")
@click.option("--api-mode", is_flag=True,
              help="Run as an API server only (do not mount the Web GUI).")
@click.option("--mmproj", default=None,
              help="Path to multimodal projector file (for LLaVA).")
@click.option("--device", default=None,
              help="Explicit device (e.g., cuda:0, metal).")
def main(model, host, port, ctx, gpu_layers, no_browser, no_model, pull_spec, debug,
         mode, keep_diagnostics, insecure, no_tls, tls_cert, tls_key, show_qr,
         project, force_new, isolated, api_mode, mmproj, device):
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
    from localm.console import console, show_url

    from localm.winconsole import disable_quickedit, set_console_title
    disable_quickedit()
    # Set the console title now; a richer title carrying the port is set below.
    set_console_title("LocaLM")
    # Set the taskbar grouping id (AppUserModelID) and, best-effort, the console
    # icon. Must run BEFORE the splash/status window is created below.
    from localm.applaunch import apply_window_identity
    apply_window_identity()
    # Print the wordmark line (the M in accent blue).
    console.print("[bold]LocaL[/bold][bold #4f9cf9]M[/bold #4f9cf9]  [dim]local AI, offline[/dim]")

    # --keep-diagnostics is a per-run override of the config toggle; export it so
    # the server's gates resolve it via keep_diagnostics_enabled(). Set BEFORE the
    # debug-log decision below.
    if keep_diagnostics:
        import os as _osd
        _osd.environ["LOCALM_KEEP_DIAGNOSTICS"] = "1"

    if debug:
        from localm.debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")
    else:
        # With keep_diagnostics on, a debug log is written without --debug. Chat
        # content is still never written in privacy mode.
        try:
            from localm.config import keep_diagnostics_enabled
            if keep_diagnostics_enabled():
                from localm.debuglog import enable_debug
                console.print(f"[yellow]debug log (keep_diagnostics):[/yellow] "
                              f"{enable_debug()}")
        except Exception as e:
            # The debug log could not be opened: warn and carry on rather than
            # aborting startup or reporting success.
            console.print(
                f"[yellow]could not enable the keep_diagnostics debug log:[/yellow] "
                f"{e} - bug reports will not include one.")

    import os as _os
    from localm.audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        _os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("server")
    if session_mode != SessionMode.PRIVACY:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in <data dir>/sessions/)[/dim]")
    elif debug:
        console.print(
            "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log records "
            "requests and raw model output - delete it after analysis if that "
            "matters.")

    # If a localm is already running for this project dir, open ITS GUI instead of
    # starting a second server. --new / --isolated force a fresh server.
    from localm.config import home_dir
    from localm import instances
    root_dir = instances.resolve_root_dir(override=project)
    if not (force_new or isolated):
        existing = instances.find_attachable(home_dir(), root_dir)
        if existing:
            # An explicit server-config flag the running instance cannot provide
            # is reported rather than discarded by attaching.
            ctx = click.get_current_context()
            conflicts = _attach_conflicts(ctx, existing, model)
            if conflicts:
                console.print(
                    f"[red]A localm server is already running for [cyan]{root_dir}"
                    f"[/cyan] (pid {existing.get('pid')}, port "
                    f"{existing.get('port')}); it cannot apply:[/red]")
                for c in conflicts:
                    console.print(f"  [red]-[/red] {c}")
                console.print(
                    "[dim]Start a SEPARATE server with your settings using "
                    "[bold]--new[/bold], or drop the option(s) above to attach to "
                    "the running one.[/dim]")
                sys.exit(1)
            url = instances.attach_url(existing)
            console.print(
                f"[bold green]Attaching[/bold green] to the localm already "
                f"running for [cyan]{root_dir}[/cyan] "
                f"(pid {existing.get('pid')}, port {existing.get('port')}).")
            if existing.get("mode") != "full":
                # The running instance is API-only; ask it to mount the GUI
                # surface live using its own attach token.
                if _mount_remote_gui(existing):
                    console.print(
                        "  [green]Mounted the GUI on the running instance.[/green]")
                else:
                    console.print(
                        "  [yellow]Could not mount the GUI on it (an older "
                        "instance?); opening its address anyway.[/yellow]")
            console.print(f"  [dim]Opening[/dim] [cyan]{show_url(url)}[/cyan]")
            if not no_browser:
                # A unique cache-buster forces a fresh navigation instead of
                # focusing an already-open tab or serving a cached shell. That
                # loopback GET / makes the remote re-run its session auto-seed.
                # No launch grant is minted here; the app ignores the stray param.
                import secrets as _secrets
                from localm import appface
                sep = "&" if "?" in url else "?"
                open_url = f"{url}{sep}lm={_secrets.token_hex(3)}"
                # run_native_window BLOCKS this thread until the window closes,
                # and must run on the process's actual main thread - which main()
                # already is here. hide_on_close=False: closing the window ends
                # this attach-only invocation instead of hiding it to a tray.
                if not appface.run_native_window(open_url, hide_on_close=False):
                    webbrowser.open(open_url)
            return

    from localm.bindhost import is_loopback_host, self_connect_host, url_host
    from localm.config import PortInUseError, load_registry, pick_port
    from localm.model_manager import (get_model_info, get_model_mmproj,
                                      is_auto_chat_eligible, sync_models_dir)

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
        if _sync.backfilled:
            _bits.append(f"{_sync.backfilled} metadata backfilled")
        if _sync.mmproj_backfilled:
            _bits.append(f"{_sync.mmproj_backfilled} vision projector backfilled")
        if _bits:
            console.print(f"[dim]Models folder synced: {', '.join(_bits)}.[/dim]")
    if _sync.note:
        console.print(f"[yellow]{_sync.note}[/yellow]")

    registry = load_registry()
    model_less = False
    if no_model:
        # Open with nothing loaded; the user picks or switches on the Models page.
        model_less = True
        model = ""
        console.print("[dim]Opening with no model loaded - "
                      "pick one on the Models page.[/dim]")
    elif not model:
        if not registry:
            # Nothing registered: open the GUI so the user can add a model from
            # the Models page (or via --pull). No engine until then.
            model_less = True
            console.print("[yellow]No models registered yet.[/yellow] "
                          "Opening the GUI - add one on the Models page"
                          + (" (download starting)…" if pull_spec else "."))
        else:
            # Pick the first entry that still resolves to a real model file or
            # directory, skipping rows whose file is missing or is not a model. A
            # type='unknown' model is skipped here but stays runnable by name.
            model = next((n for n in sorted(registry)
                          if get_model_info(n) and is_auto_chat_eligible(registry[n])), None)
            if model is None:
                model_less = True
                console.print(
                    "[yellow]No loadable chat models in the registry "
                    "(files missing, not a model, or type 'unknown').[/yellow] "
                    "Opening the GUI - fix, add, or set a model's type on the Models page.")

    # Effective bind host: an explicit -H wins for this process (and survives an
    # in-place restart, which re-execs the same argv); otherwise the GUI-settable
    # 'bind_host' config key; otherwise loopback. host_from_config marks a bind
    # driven from the GUI, so every failed precondition below degrades LOUDLY to
    # loopback instead of exiting.
    from localm.cli import _resolve_bind_host
    host, host_from_config = _resolve_bind_host(host)
    bind_fallback = None

    # Refuse to bind past loopback without auth unless explicitly forced. Checked
    # before any setup work.
    bind_warning = _gui_bind_warning(host)
    if bind_warning and not insecure and host_from_config:
        # A config-driven network bind without a strong key binds loopback instead
        # of exiting, so the Settings page that caused the bind stays reachable.
        # --insecure has no config form. Surfaced on the console, in the log, and
        # via /api/companion (bind_fallback).
        from localm.auth import any_key_configured
        _why = ("no API key is set" if not any_key_configured()
                else "the API key is too short to be safe")
        bind_fallback = (
            f"The configured bind address ({host}) was not applied: {_why}. "
            f"The server is on 127.0.0.1 (this computer only). Set a strong "
            f"API key (Settings > Security > Owner key, or run: localm key "
            f"generate), then restart the server.")
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print(
            "[bold yellow]  Ignoring the configured bind address and binding "
            "127.0.0.1 (this computer only). Set a strong API key, then "
            "restart.[/bold yellow]")
        from localm.debuglog import logger as _blog
        _blog.warning("config bind_host=%s not applied: %s", host, _why)
        host = "127.0.0.1"
    elif bind_warning and not insecure:
        console.print(f"[bold red]{bind_warning}[/bold red]")
        console.print(
            "[bold red]Refusing to start: binding past loopback without auth. "
            "Set $env:LOCALM_API_KEY first, or pass --insecure to override.[/bold red]")
        sys.exit(2)
    elif bind_warning:
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print("[bold yellow]  Proceeding anyway (--insecure set).[/bold yellow]")

    # A config-sourced address must also be bindable right now, probed with a real
    # throwaway bind, with the same loud loopback fallback. Runs even under
    # --insecure. An explicit -H still fails hard.
    if host_from_config and bind_fallback is None:
        # EVERY config-driven bind is probed, loopback included.
        from localm.cli import _bind_preflight_error
        _bind_err = _bind_preflight_error(host)
        if _bind_err is not None:
            bind_fallback = (
                f"The configured bind address ({host}) was not applied: "
                f"this machine has no usable interface with that address "
                f"right now ({_bind_err}). The server is on 127.0.0.1 "
                f"(this computer only). Fix Settings > Server > Bind "
                f"address (0.0.0.0 = every interface), then restart the "
                f"server.")
            console.print(
                f"[bold yellow]The configured bind address {host} cannot "
                f"be bound on this machine right now ({_bind_err}) - "
                f"ignoring it and binding 127.0.0.1 (this computer "
                f"only).[/bold yellow]")
            from localm.debuglog import logger as _plog
            _plog.warning("config bind_host=%s not applied: %s",
                          host, _bind_err)
            host = "127.0.0.1"

    model_path = None
    display_name = ""
    if not model_less:
        # allow_direct_path: this is the STARTUP model, typed as
        # `localm gui <path>`. The runtime switch path below does not opt in.
        info = get_model_info(model, allow_direct_path=True)
        if info is None:
            console.print(f"[red]Model not found:[/red] {model}")
            sys.exit(1)
        model_path, display_hint = info
        display_name = model if model in registry else display_hint

    # A network bind serves HTTPS. Resolved before attach_gui so the coder/media/
    # RAG self-call URL carries the right scheme, and BEFORE pick_port below so a
    # bind that falls back to loopback here picks its port for the host it will
    # actually bind.
    from localm.cli import _resolve_tls, _setup_tls_or_exit
    if host_from_config:
        # A config-driven bind stays on loopback when built-in TLS cannot be set
        # up, rather than exiting or serving cleartext. An unusable custom cert
        # pair already degrades to the built-in cert inside _resolve_tls. A
        # half-specified CLI --tls-cert/--tls-key propagates as a usage error.
        try:
            ssl_certfile, ssl_keyfile = _resolve_tls(
                host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
        except click.UsageError:
            raise
        except Exception as e:
            bind_fallback = (
                f"The configured bind address ({host}) was not applied: "
                f"built-in TLS could not be set up ({e}). The server is on "
                f"127.0.0.1 (this computer only). Fix TLS (or turn 'Encrypt "
                f"network traffic' off for a trusted network), then restart "
                f"the server.")
            console.print(
                f"[bold yellow]Could not set up built-in TLS: {e} - ignoring "
                f"the configured bind address and binding 127.0.0.1 (this "
                f"computer only) rather than serving the network in "
                f"cleartext.[/bold yellow]")
            from localm.debuglog import logger as _tlog
            _tlog.warning("config bind_host=%s not applied: TLS setup failed: %s",
                          host, e)
            host = "127.0.0.1"
            ssl_certfile = ssl_keyfile = None
    else:
        ssl_certfile, ssl_keyfile = _setup_tls_or_exit(
            host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    scheme = "https" if ssl_certfile else "http"

    try:
        # A wildcard is not itself connectable, so probe the loopback it covers
        # (self_connect_host maps 0.0.0.0 -> 127.0.0.1 and :: -> ::1).
        chosen_port, was_busy = pick_port(port, host=self_connect_host(host))
    except PortInUseError as exc:
        # An explicit --port is honored or refused, never relocated onto another
        # port. Only the default auto-bumps.
        console.print(f"[red]Port {exc.port} is already in use.[/red] "
                      "Free it, or choose another with -p/--port.")
        sys.exit(1)
    if was_busy:
        console.print(f"[yellow]Default port busy - using {chosen_port}.[/yellow]")

    # The authority (host:port) this process uses to reach ITSELF, and the address
    # shown to the user. Derived from the effective bind, not hardcoded to
    # 127.0.0.1. _self_host is the bare address, for socket-level self-connects;
    # _self_authority is the url_host-bracketed form, for anything that goes into
    # a URL.
    _self_host = self_connect_host(host)
    _self_authority = f"{url_host(_self_host)}:{chosen_port}"

    from localm.inference.engine import Engine
    from localm.inference import http_server as hs
    from .web import attach_gui

    def _make_engine(name: str, *, allow_direct_path: bool = False) -> Engine:
        # allow_direct_path defaults to False, so switch_engine - which calls
        # factory(name) positionally with a model name off the wire - cannot opt
        # in. The startup load below passes it explicitly.
        m_info = get_model_info(name, allow_direct_path=allow_direct_path)
        if m_info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = m_info
        # An explicit --mmproj applies only to `model`, the STARTUP model this
        # server was launched with. Every other name falls back to that model's
        # own recorded/sibling projector via get_model_mmproj.
        mmproj_path = (mmproj if name == model else None) or get_model_mmproj(
            name, allow_direct_path=allow_direct_path)
        return Engine(
            str(m_path),
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            mmproj_path=mmproj_path,
            device=device,
            display_name=name if name in load_registry() else m_hint,
        )

    engine = None
    if not model_less:
        try:
            engine = _make_engine(model, allow_direct_path=True)
        except Exception as e:
            # A single bad registry entry degrades to the model-less path instead
            # of stopping the server from starting.
            console.print(f"[yellow]Could not load model '{model}': {e}[/yellow]")
            console.print("[yellow]Opening the GUI model-less - pick a model on "
                          "the Models page.[/yellow]")
            model_less = True
    app = hs.create_app(engine)

    async def switch_model(name: str) -> dict:
        """Swap engines, PREEMPTING any in-flight load so the latest selection
        wins immediately instead of waiting for an abandoned model to finish
        loading (see http_server.switch_engine). Serialised on the inference
        semaphore so no generation is mid-flight."""
        return await hs.switch_engine(name, _make_engine)

    manager = None
    if not api_mode:
        manager = attach_gui(
            app,
            self_url=f"{scheme}://{_self_authority}/v1",
            switch_model=switch_model,
            # Read the authoritative pointer directly rather than shadowing it in
            # a local dict. _active_model_name is updated synchronously by both
            # switch_engine (on load) and unload_all_models/unload_one_model (on
            # unload).
            active_model=lambda: hs._active_model_name or "",
        )

    base_url = f"{scheme}://{_self_authority}/"
    # Deep-link the browser to the Models page (and a pending download) when
    # the GUI was opened with --pull or with nothing registered yet.
    open_url = base_url
    if pull_spec:
        from urllib.parse import quote
        from .web import mint_pull_grant
        # Mint a single-use, spec-bound secret so this deep link auto-starts its
        # own download; a forged `?pull=` link without the secret falls back to an
        # explicit human confirmation.
        pull_token = mint_pull_grant(app, pull_spec)
        open_url = (f"{base_url}?view=models&pull={quote(pull_spec, safe='')}"
                    f"&pull_token={quote(pull_token, safe='')}")
    elif model_less:
        open_url = f"{base_url}?view=models"
    # When auth is on, hand the auto-opened browser a single-use grant in the URL
    # so it lands AUTHENTICATED via a real navigation rather than the implicit
    # GET / cookie auto-seed. Runs on ANY bind, network included; the grant is
    # placed only in the URL opened locally, so a network client never sees it.
    from localm import auth as _auth
    from .web import mint_launch_grant
    if _auth.get_api_key():
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
        _p = urlparse(open_url)
        _q = dict(parse_qsl(_p.query))
        _q["localm_token"] = mint_launch_grant(app)
        open_url = urlunparse(_p._replace(query=urlencode(_q)))

    _wtitle = f"LocaLM  -  localhost:{chosen_port}"
    if not model_less and (display_name or model):
        _wtitle = f"LocaLM  -  {display_name or model}  -  :{chosen_port}"
    set_console_title(_wtitle)
    _srv_name = "localm API server" if api_mode else "localm GUI"
    console.print(f"[bold green]{_srv_name}[/bold green] → {show_url(base_url)}")
    if model_less:
        console.print("  model: [yellow]none yet - add one on the Models page[/yellow]")
    else:
        console.print(f"  model: [cyan]{display_name or Path(str(model_path)).stem}[/cyan]")
    console.print("  Ctrl+C to stop")

    # Reach-by-name (mDNS): advertise <mdns_name>.local on a network bind. Started
    # HERE, before printing, so <name>.local is only recommended when it is
    # actually being advertised. Closed in the finally below.
    from localm import netname
    mdns_advertiser = None
    if not is_loopback_host(host) and not isolated:
        mdns_advertiser = netname.start_advertiser(
            chosen_port, tls=bool(ssl_certfile),
            addresses=_mdns_addresses(host))
    _adv_name = netname.mdns_fqdn() if mdns_advertiser is not None else None

    # Phone / LAN access. The GUI is an installable PWA. Bound to loopback it is
    # reachable only on this machine; bound to the network, print the address a
    # phone on the same Wi-Fi can open.
    if is_loopback_host(host):
        console.print(
            "  [dim]use from your phone: bind to your network with "
            "[/dim][cyan]localm gui -H 0.0.0.0[/cyan][dim] or Settings > Server "
            "> Bind address (set an API key first); see docs/phone.md[/dim]")
        if show_qr:
            console.print(
                "  [yellow][PoC][/yellow] [dim]--qr needs a network bind to be "
                "scannable: [/dim][cyan]localm gui -H 0.0.0.0 --qr[/cyan]")
    else:
        # A network bind: print the reachable NAMES (localm.local when advertised,
        # the Tailscale MagicDNS name) and IPs.
        targets = netname.network_targets(mdns_name=_adv_name, bind_host=host)
        primary_url = None
        qr_url = None
        for _label, _target in targets:
            url = f"{scheme}://{url_host(_target)}:{chosen_port}/"
            suffix = "  [dim](open it, then Install as app)[/dim]" if primary_url is None else ""
            console.print(f"  [dim]{_label}:[/dim] [cyan]{show_url(url)}[/cyan]{suffix}")
            if primary_url is None:
                primary_url = url
            # Prefer an IP for the scannable QR; fall back to the first target
            # below.
            if qr_url is None and "(IP)" in _label:
                qr_url = url
        if primary_url is None:
            # No reachable network address detected: say so instead of printing an
            # unconnectable wildcard URL. The loopback URL above still works on
            # this machine.
            console.print("  [dim]no reachable network address detected - "
                          "this machine only[/dim]")
        if scheme == "https":
            _ca_host = url_host(netname.ca_trust_host(_adv_name)
                                or self_connect_host(host))
            console.print(
                "  [dim]first visit shows a one-time certificate warning; tap "
                "[/dim][cyan]Install certificate[/cyan][dim] on the key screen "
                "(or open [/dim][cyan]"
                + show_url(f"{scheme}://{_ca_host}:{chosen_port}/localm-ca.crt")
                + "[/cyan][dim]) to trust it once - then no warning "
                "and the app installs.[/dim]")
            console.print(
                "  [dim]Firefox has its own certificate store: import the CA in "
                "Firefox (or set about:config security.enterprise_roots.enabled), "
                "not just Windows. The key screen shows the exact steps.[/dim]")
        _ts_hint = netname.tailscale_rename_hint()
        if _ts_hint:
            console.print(f"  [dim]{_ts_hint}[/dim]")
        if show_qr and (qr_url or primary_url):
            _print_qr(qr_url or primary_url)

    # Preload the model in the background. Engine.load is lock-protected; a
    # request arriving mid-load waits on it.
    def _preload():
        try:
            engine.load()
        except Exception as e:
            _report_preload_failure(console, e)

    if engine is not None:
        threading.Thread(target=_preload, daemon=True, name="preload").start()

    # Print the exact local URL, carrying the one-time grant. soft_wrap emits the
    # long URL as ONE line, with no injected newline.
    console.print(f"  [dim]Open the GUI:[/dim] [cyan]{show_url(open_url)}[/cyan]",
                  soft_wrap=True)

    def _open_when_ready(url: str, port: int, timeout: float = 20.0) -> None:
        """Open the browser tab only once the server actually ACCEPTS a
        connection, so a fresh launch never lands the user on the "Can't
        reach the server / reconnecting" overlay because the tab beat the
        listener (the cold-start race). Polls the loopback port; opens
        anyway after *timeout* as a fallback. Only ever used when a native
        window will NOT be used this run (see want_native below) - when it
        will, the equivalent poll-then-open happens inline further down,
        because opening the native window has to block THIS process's own
        main thread, not a background one."""
        import socket
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                with socket.create_connection((_self_host, port), timeout=0.5):
                    break
            except OSError:
                _time.sleep(0.25)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    from localm import appface
    # pywebview's webview.start() must be called from the process's actual main
    # thread, which hs.run_advertised() below otherwise occupies. A native window
    # therefore takes the main thread and the server moves to a background one,
    # in the branch after the tray/status-window setup.
    want_native = _should_auto_open_browser(no_browser) and appface.native_window_available()

    if _should_auto_open_browser(no_browser) and not want_native:
        threading.Thread(target=_open_when_ready, args=(open_url, chosen_port),
                         daemon=True, name="open-browser").start()

    # Record the EFFECTIVE bind host (after any config-bind fallback above), so
    # the SPA-shell route and every trust decision gating on app.state.bind_host
    # match what is actually bound. bind_fallback carries why a configured network
    # bind was not applied, or None; /api/companion surfaces it.
    app.state.bind_host = host
    app.state.bind_fallback = bind_fallback

    # Advertise this server in the instance registry as a "full" surface (API +
    # GUI) so a later launch in the same dir can discover and attach to it.
    # --isolated keeps it invisible to discovery.
    from localm import debuglog
    from localm.config import home_dir as _home_dir
    # Tray control surface (Windows): Open / Copy address / View logs / Restart /
    # Stop. Best-effort and fully guarded; it never blocks the server. Restart and
    # Stop go through _tray_callbacks, not the bare hs._do_restart /
    # hs._do_shutdown. "View logs" dumps the always-on activity buffer (INFO+, no
    # chat content) to a readable file.
    on_restart, on_stop = _tray_callbacks(app, hs)
    app_face = appface.start_app_face(
        name="LocaLM", url=base_url, logfile=_home_dir() / "logs" / "recent.log",
        get_log_lines=debuglog.recent_activity,
        on_restart=on_restart, on_stop=on_stop)
    if app_face is not None:
        # Flip the window from "Starting..." to "Running" (and, on Windows, hide it
        # to the tray) once the port actually accepts connections. Polled in a
        # thread with a raw TCP connect, which works for http and https alike.
        def _mark_ready_when_listening():
            import socket
            import time as _t
            for _ in range(160):   # up to ~40s, then flip anyway
                try:
                    with socket.create_connection((_self_host, chosen_port), 0.5):
                        break
                except OSError:
                    _t.sleep(0.25)
            app_face.set_ready()
            # When the launcher spawned us with our OWN console, hide it now that
            # the server is up and the tray/status window is the surface. A direct
            # `localm gui` in a terminal sets no such flag and is left alone.
            import os as _os2
            if _os2.environ.get("LOCALM_OWN_CONSOLE"):
                from localm.winconsole import hide_console
                hide_console()
        threading.Thread(target=_mark_ready_when_listening,
                         name="localm-ready", daemon=True).start()
        # Route hang-alarm surfacing into the native status window: set_error turns
        # the status red and un-hides the window from the tray, set_ready restores
        # the Running state. Both are queue-based and thread-safe; the alarm calls
        # them from its own daemon thread.
        hs.set_hang_surface(
            lambda text: app_face.set_error(f"Server problem: {text}"),
            app_face.set_ready)

    def _serve():
        # run_advertised performs the shared advertise + run_server sequence. The
        # app object itself is built above, not inside serve().
        try:
            hs.run_advertised(app, host, chosen_port,
                              mode="api" if api_mode else "full",
                              ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                              project=project, isolated=isolated, log_level="warning")
        finally:
            # Runs once the SERVER has actually stopped - on the main thread in the
            # browser-tab case, on the background server thread when want_native is
            # set - never merely once a native window has closed.
            if app_face is not None:
                app_face.close()
            if mdns_advertiser is not None:
                mdns_advertiser.close()
            if manager is not None:
                manager.close_all()
            # No-op when want_native is False. When it is true, this destroys the
            # window that only ever hides itself on its close button, so
            # run_native_window's blocking webview.start() returns and the process
            # can exit.
            appface.close_native_window()

    if want_native:
        # The server thread is non-daemon, so closing the native window does not
        # kill a still-running server; Ctrl+C or the tray Stop button stops it. The
        # process's real main thread goes to the window, the one thread pywebview
        # accepts.
        server_thread = threading.Thread(target=_serve, name="localm-server",
                                        daemon=False)
        server_thread.start()
        import socket as _socket
        import time as _time3
        _deadline = _time3.monotonic() + 20.0
        while _time3.monotonic() < _deadline:
            try:
                with _socket.create_connection((_self_host, chosen_port), 0.5):
                    break
            except OSError:
                _time3.sleep(0.25)
        # on_quit=on_stop is the same callable the tray's Stop button uses, so with
        # "quit when the app window is closed" on, closing the window stops the
        # server.
        if not appface.run_native_window(open_url, on_quit=on_stop):
            webbrowser.open(open_url)
        # MUST join, rather than rely on server_thread being non-daemon:
        # concurrent.futures.thread registers its shutdown via
        # threading._register_atexit(), which fires as soon as this main thread's
        # top-level code finishes, BEFORE non-daemon threads are joined. Joining
        # keeps this thread's top-level code running for exactly as long as the
        # server is, so the shared plugin executor stays up while requests are
        # still being served.
        server_thread.join()
    else:
        _serve()
