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
    not a logging call), so a preload failure with no other symptom - the user
    never explicitly tries to chat - leaves NO trace a bug report could surface.
    The log call carries the full traceback."""
    console.print(f"[yellow]Background model load failed: {exc}[/yellow]")
    from localm.debuglog import logger
    logger.exception("background model preload failed")


def _mdns_addresses(host: str):
    """Which addresses mDNS should advertise for a bind on *host*, or None to let
    ``netname.start_advertiser`` pick this machine's LAN IPv4.

    A WILDCARD bind answers on every interface, so the LAN IPv4 that
    ``start_advertiser`` finds for itself is reachable and is the right advert -
    including for ``::``, which localm binds dual-stack (see ``netlisten``), so an
    IPv4 client resolving ``<name>.local`` genuinely connects.

    A SPECIFIC literal answers on exactly one address, and advertising any other
    one publishes a name that does not resolve to a listening socket. So that bind
    advertises ITSELF. This is also what makes ``<name>.local`` usable on a
    specific IPv6 bind, where the LAN IPv4 would be pure fiction."""
    return None if is_wildcard_host(host) else [host]


def _model_less_hint(api_mode: bool) -> str:
    """The console line shown next to "model:" when nothing is loaded yet."""
    if api_mode:
        return ("  model: [yellow]none yet - "
                "add one with `localm pull <name>`[/yellow]")
    return "  model: [yellow]none yet - add one on the Models page[/yellow]"


def _console_url_line(api_mode: bool, base_url: str, open_url: str) -> tuple:
    """(label, url) for the line naming where to reach the running server.

    api_mode never mounts a GUI, so open_url's GUI-only additions (a
    view=models/pull deep link, a browser auto-login grant) would name a page
    that is not being served; base_url is shown instead.
    """
    if api_mode:
        return "API base", base_url
    return "Open the GUI", open_url


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
    own call site - AFTER the tray is wired, not before. A partial would
    freeze instance_id/port at None (their state at wire time); these
    closures read app.state at CALL time instead - by the time a user can
    physically click Restart/Stop, run_advertised() has long since entered
    advertise()'s context and populated both.

    appface invokes on_restart/on_stop with NO arguments
    (``threading.Thread(target=self.on_restart)``, appface.py), and both
    hs._do_restart and hs._do_shutdown are keyword-only with None defaults, so
    the instance_id has to be supplied here. A tray Restart/Stop that called
    disarm_crash_guard(instance_id=None) would clear the LEGACY unscoped marker
    (bugreport.py's per-instance-scoping fallback) and leave this instance's real
    server-crash.<instance_id>.marker still armed, so the NEXT start would report
    a crash that never happened. The HTTP routes (routes/admin.py's restart/stop
    endpoints) pass the real instance_id the same way. _do_restart without its
    port makes _restart_argv omit ``-p``, so a re-exec'd server can come back on
    a different port, stranding the tray/GUI's own open window on a dead one."""
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
    attach token (a local same-user secret). Returns True on
    success, False on any failure - an older instance without the endpoint, a
    missing token, or a network error - so the caller can fall back to just
    opening the address."""
    import requests
    scheme = entry.get("scheme") or "http"
    port = entry.get("port")
    token = entry.get("token")
    if not port or not token:
        return False
    # Dial the loopback THAT instance bound: an IPv6-bound server does not
    # answer on the IPv4 loopback, and this call is what turns a headless
    # server into a GUI one.
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
        # qrcode is a core dependency (pyproject.toml `dependencies`), so this
        # means the install is broken/partial, not that an extra is missing.
        con.print('  [yellow][PoC][/yellow] QR unavailable: the "qrcode" '
                  'package is missing from this install ([cyan]pip install '
                  'qrcode[/cyan] to fix it)')
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


# gui options that only shape a FRESH server: an attach to an existing instance
# cannot honor them, so passing one explicitly is reported as a conflict rather
# than swallowed. Not listed here = compatible with an attach: no_browser /
# debug / project / force_new / isolated / keep_diagnostics (local or
# attach-control), no_model (only picks a STARTUP model, moot when nothing is
# starting), or value-aware (model / host / port) handled below.
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
    # port / host: conflict only if the requested value DIFFERS from the running
    # instance's - asking for the port it is already on is harmless.
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
    # model: a specific model was named. Probe the running instance; conflict only
    # when its active model is KNOWN and different (unknown -> attach quietly, like
    # `localm run`; same model -> attach). Never silently serve a different model.
    if model and _explicit(ctx, "model"):
        active = _probe_active_model(existing)
        if active and active != model:
            conflicts.append(
                f"model {model} (the running server serves {active})")
    # everything else: an attach cannot retroactively set the running server's
    # ctx / gpu-layers / tls / mode / ..., so an explicit pass is a conflict.
    # (--mode is the session-persistence mode, a different namespace from the
    # entry's SURFACE mode, so it cannot be compared cheaply - any explicit --mode
    # conflicts.)
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

    # A click into this console window must not freeze the server
    # (Windows QuickEdit suspends output, and output blocks inference).
    from localm.winconsole import disable_quickedit, set_console_title
    disable_quickedit()
    # Brand the window right away so it never reads as a python.exe path; a richer
    # title (with the port) is set once the port is chosen below.
    set_console_title("LocaLM")
    # Give this process a real app identity: set the taskbar grouping id
    # (AppUserModelID) now, BEFORE the splash/status window is created below, so its
    # taskbar button groups as LocaLM. Also best-effort sets the console icon (the
    # console is hidden once the server is up; the splash window carries the icon
    # itself). Pairs with the LocaLM.exe launcher so the running app reads as LocaLM.
    from localm.applaunch import apply_window_identity
    apply_window_identity()
    # Light branding: a single wordmark line (the M in accent blue), no noise.
    console.print("[bold]LocaL[/bold][bold #4f9cf9]M[/bold #4f9cf9]  [dim]local AI, offline[/dim]")

    # --keep-diagnostics is a per-run override of the config toggle (the launcher
    # checkbox passes it); export it so the server's gates resolve it via
    # keep_diagnostics_enabled(). Set BEFORE the debug-log decision below.
    if keep_diagnostics:
        import os as _osd
        _osd.environ["LOCALM_KEEP_DIAGNOSTICS"] = "1"

    if debug:
        from localm.debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")
    else:
        # keep_diagnostics: a user who opted into keeping diagnostics for bug
        # reports (even in privacy mode) gets a debug log written too, so a report
        # has request/operation context - without needing to pass --debug. Chat
        # content is still never written in privacy mode (see debug_content_enabled).
        try:
            from localm.config import keep_diagnostics_enabled
            if keep_diagnostics_enabled():
                from localm.debuglog import enable_debug
                console.print(f"[yellow]debug log (keep_diagnostics):[/yellow] "
                              f"{enable_debug()}")
        except Exception as e:
            # The user opted into keep_diagnostics, but the debug log could not be
            # opened (e.g. an unwritable or full LOCALM_HOME). Startup continues,
            # but the failure is warned about so the user knows their bug reports
            # will not include a debug log.
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
            "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log still "
            "records operational lines (requests, timings, errors) - never "
            "raw model output or chat content, even with this flag on. "
            "Delete it after analysis if the operational detail matters to "
            "you.")

    # Attach-or-spawn: if a localm is already running for this project dir, open
    # ITS GUI instead of starting a second server that double-loads the model.
    # --new / --isolated force a fresh server.
    from localm.config import home_dir
    from localm import instances
    root_dir = instances.resolve_root_dir(override=project)
    if not (force_new or isolated):
        existing = instances.find_attachable(home_dir(), root_dir)
        if existing:
            # Do NOT silently discard explicit server-config flags by attaching.
            # If the user asked for something the running instance cannot provide
            # (a different port/host/model, a fresh --mode/--ctx/tls/... ), say so
            # and let them decide: --new starts a separate server with their
            # settings, or they drop the flag to attach.
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
                # On-demand GUI mount: the running instance is API-only; ask it
                # to mount the GUI surface live (no second server, no second
                # model load) using its own attach token.
                if _mount_remote_gui(existing):
                    console.print(
                        "  [green]Mounted the GUI on the running instance.[/green]")
                else:
                    console.print(
                        "  [yellow]Could not mount the GUI on it (an older "
                        "instance?); opening its address anyway.[/yellow]")
            _url_label, _ = _console_url_line(api_mode, url, url)
            console.print(f"  [dim]{_url_label}:[/dim] [cyan]{show_url(url)}[/cyan]",
                          soft_wrap=True)
            if not no_browser:
                # Force a FRESH navigation with a unique cache-buster so the browser
                # actually reloads (instead of silently focusing an already-open tab
                # at the same URL) and cannot serve the shell from a warm service-
                # worker cache. That fresh loopback GET / makes the remote re-run its
                # own session auto-seed, so a relaunch lands authenticated even after
                # a key roll. We do NOT mint a launch grant here: the running instance
                # may be an OLDER localm that has no grant endpoint, whereas the
                # loopback auto-seed exists in every version. The app ignores the
                # stray param.
                import secrets as _secrets
                from localm import appface
                sep = "&" if "?" in url else "?"
                open_url = f"{url}{sep}lm={_secrets.token_hex(3)}"
                # run_native_window BLOCKS this thread until the window closes -
                # correct here: main() is already this process's actual main
                # thread, and this attach-only invocation starts no server of
                # its own to keep the process alive, so blocking here IS what
                # keeps a native window's host process running for as long as
                # the window stays open (see run_native_window's docstring for
                # why it must run on the main thread specifically).
                # hide_on_close=False: this invocation owns no server of its
                # own to keep alive (it only ever attached to one already
                # running elsewhere) - closing this window is this whole
                # process's purpose, so it should just close for real, not
                # hide to a tray this invocation never creates.
                if not appface.run_native_window(open_url, hide_on_close=False):
                    webbrowser.open(open_url)
            return

    from localm.bindhost import is_loopback_host, self_connect_host, url_host
    from localm.config import PortInUseError, load_registry, pick_port
    from localm.model_manager import (get_model_info, get_model_mmproj,
                                      is_auto_chat_eligible, sync_models_dir)

    # Pick up models added to (or gone missing from) the models folder since
    # last run. Local reconciliation only; no network I/O.
    _sync = sync_models_dir(backfill_mmproj=False)
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
        if _bits:
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
            # so one bad registry entry never blocks startup. A type='unknown' model
            # is skipped here (never auto-loaded as chat) but stays runnable by name.
            model = next((n for n in sorted(registry)
                          if get_model_info(n) and is_auto_chat_eligible(registry[n])), None)
            if model is None:
                model_less = True
                console.print(
                    "[yellow]No loadable chat models in the registry "
                    "(files missing, not a model, or type 'unknown').[/yellow] "
                    "Opening the GUI - fix, add, or set a model's type on the Models page.")

    # Effective bind host: an explicit -H wins for this process (and survives
    # an in-place restart, which re-execs the same argv); otherwise the
    # GUI-settable 'bind_host' config key (Applies.RESTART - this read, running
    # in the fresh process, is what makes Settings > Restart server apply it);
    # otherwise loopback. host_from_config marks a bind possibly driven from
    # the GUI by a user with NO terminal: every failed precondition below must
    # then degrade LOUDLY to loopback instead of exiting, or the server dies
    # with no terminal-free way back (the GUI is how that user would fix it).
    from localm.cli import _resolve_bind_host
    host, host_from_config = _resolve_bind_host(host)
    bind_fallback = None

    # Refuse to bind past loopback without auth unless explicitly forced: the GUI
    # exposes not just the chat API but the coder agent, which can run shell
    # commands and edit files on this machine. Checked before any setup work.
    bind_warning = _gui_bind_warning(host)
    if bind_warning and not insecure and host_from_config:
        # A config-driven network bind without a strong key is refused exactly
        # like the exit(2) below - the network is never served unauthenticated,
        # and --insecure has NO config form, so this override can only ever be
        # typed in a terminal - but the refusal here is a loopback
        # bind, not an exit: the server stays reachable on this machine so the
        # Settings page that caused the bind can also fix it. Surfaced on the
        # console, in the log, and via /api/companion (bind_fallback).
        from localm.auth import any_key_configured
        _why = ("no API key is set" if not any_key_configured()
                else "the API key is too short to be safe")
        bind_fallback = (
            f"The configured bind address ({host}) was not applied: {_why}. "
            f"The server is on 127.0.0.1 (this computer only). Set a long, "
            f"random API key (Settings > Security > Owner key, or run: localm "
            f"key generate), then restart the server.")
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print(
            "[bold yellow]  Ignoring the configured bind address and binding "
            "127.0.0.1 (this computer only). Set a long, random API key, then "
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

    # A config-sourced address must also be BINDABLE right now, not merely
    # well-formed: the field's own recommended use (one specific interface IP)
    # goes stale when DHCP reassigns the machine, and handing a stale address
    # to the server kills the process at the socket bind - the locked-out-user
    # failure the auth fallback above exists to prevent, through a different
    # door. Probed with a real throwaway bind (syntax checks cannot see it);
    # same loud loopback fallback. Runs even under --insecure (that flag
    # waives AUTH, not bindability - a dead process helps nobody). An explicit
    # -H keeps failing hard in front of the operator who typed it.
    if host_from_config and bind_fallback is None:
        # EVERY config-driven bind is probed, loopback included: 127.0.0.1 and
        # ::1 bind trivially, but ``::ffff:127.0.0.1`` - which is_loopback_host
        # correctly calls loopback (it IS one) - is refused by Windows (WinError
        # 10049), so skipping the probe for the loopback class would leave the
        # dead-server-with-no-terminal hole open for a value the validator
        # accepts. The probe costs one socket.
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
        # allow_direct_path: this is the STARTUP model, typed by the operator as
        # `localm gui <path>`. The runtime switch path below is a different case
        # and does NOT opt in.
        info = get_model_info(model, allow_direct_path=True)
        if info is None:
            console.print(f"[red]Model not found:[/red] {model}")
            sys.exit(1)
        model_path, display_hint = info
        display_name = model if model in registry else display_hint

    # Built-in TLS: a network bind serves HTTPS out of the box so the API key
    # and all traffic are encrypted. Resolved before attach_gui so the
    # coder/media/RAG self-call URL carries the right scheme - and BEFORE
    # pick_port below, so a config-driven bind that has to fall back to
    # loopback here picks its port for the host it will actually bind.
    from localm.cli import _resolve_tls, _setup_tls_or_exit
    if host_from_config:
        # A config-driven bind must not die over TLS (no terminal to see the
        # exit - see host_from_config above). An unusable CUSTOM cert pair
        # already degrades to the built-in cert inside _resolve_tls; this
        # catches the built-in path itself failing (a broken crypto stack),
        # where the only option that is neither cleartext nor a dead server is
        # staying on loopback. A half-specified CLI --tls-cert/--tls-key is
        # still the operator's usage error and propagates as one.
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
        # An explicit --port is honored or refused, never silently relocated onto
        # another (often the shared default) port. Only the default auto-bumps.
        console.print(f"[red]Port {exc.port} is already in use.[/red] "
                      "Free it, or choose another with -p/--port.")
        sys.exit(1)
    if was_busy:
        console.print(f"[yellow]Default port busy - using {chosen_port}.[/yellow]")

    # The authority (host:port) this process uses to reach ITSELF, and to show the
    # user the address that works on this machine. Derived from the effective bind
    # rather than hardcoded to 127.0.0.1: a server bound only on ::1 (or on one
    # specific interface) has nothing listening on the IPv4 loopback, so every
    # self-call the GUI makes - the coder agent, RAG self-embedding, the chat/media
    # VRAM handover - would dial an address that is not there. url_host brackets an
    # IPv6 literal so the result is a legal URL authority and not https://::1:8642/.
    # The BARE address for socket-level self-connects (create_connection takes
    # an address, never a bracketed URL authority), and the bracketed form for
    # anything that goes into a URL.
    _self_host = self_connect_host(host)
    _self_authority = f"{url_host(_self_host)}:{chosen_port}"

    from localm.inference.engine import Engine
    from localm.inference import http_server as hs
    from .web import attach_gui

    def _make_engine(name: str, *, allow_direct_path: bool = False) -> Engine:
        # This factory has TWO callers with different trust: the startup load just
        # below (operator-typed `localm gui <path>`, opts in) and switch_engine
        # (a model name off the wire from the GUI/API, which must not). Defaulting
        # to False means the wire path gets the safe behaviour by construction -
        # switch_engine calls factory(name) positionally and cannot opt in.
        m_info = get_model_info(name, allow_direct_path=allow_direct_path)
        if m_info is None:
            raise ValueError(f"Model not found: {name}")
        m_path, m_hint = m_info
        # An explicit --mmproj always wins for the model it was given for;
        # otherwise fall back to the model's own recorded/sibling projector
        # (get_model_mmproj), so a pulled vision GGUF with no --mmproj flag
        # keeps image support on every load AND every switch this factory
        # serves.
        #
        # --mmproj is scoped to `model` (the STARTUP model this server was
        # launched with), never to the process. This factory is reused by every
        # later switch_engine call for ANY model name, so an unscoped
        # `mmproj or ...` would apply the startup model's projector to whatever
        # model was switched to next, overriding a DIFFERENT model's own,
        # correctly-recorded projector. Every name other than the startup model
        # falls through to its own registry lookup, exactly as if --mmproj had
        # never been given.
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
            # A single bad registry entry must not stop the server from starting;
            # degrade to the model-less path and let the user pick on the Models page.
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
            # Read the authoritative pointer directly rather than shadowing it
            # in a local dict updated only on load (via on_active), which is
            # never cleared on unload and would keep reporting a model "active"
            # for the rest of the process lifetime. _active_model_name is
            # updated synchronously by both switch_engine (on load) and
            # unload_all_models/unload_one_model (on unload).
            active_model=lambda: hs._active_model_name or "",
        )

    base_url = f"{scheme}://{_self_authority}/"
    # Deep-link the browser to the Models page (and a pending download) when
    # the GUI was opened with --pull or with nothing registered yet.
    open_url = base_url
    if pull_spec:
        from urllib.parse import quote
        from .web import mint_pull_grant
        # Mint a single-use, spec-bound secret so THIS deep link can auto-start
        # its own download with zero clicks, while a forged `?pull=` link
        # elsewhere (which cannot know the secret) falls back to an explicit
        # human confirmation (see init.js / web.py).
        pull_token = mint_pull_grant(app, pull_spec)
        open_url = (f"{base_url}?view=models&pull={quote(pull_spec, safe='')}"
                    f"&pull_token={quote(pull_token, safe='')}")
    elif model_less:
        open_url = f"{base_url}?view=models"
    # One-time launch handoff: when auth is on, hand the auto-opened browser a
    # single-use grant in the URL so it lands AUTHENTICATED via a real navigation,
    # instead of depending on the implicit GET / cookie auto-seed. This runs on ANY
    # bind, including a NETWORK bind - the person launching is on THIS machine (the
    # host) and should never have to type the key, even when the server is exposed to
    # the LAN. The grant is a 256-bit single-use secret only we know and only place in
    # the URL we open locally, so a network client never sees it (see web.py's
    # redemption note).
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
        console.print(_model_less_hint(api_mode))
    else:
        console.print(f"  model: [cyan]{display_name or Path(str(model_path)).stem}[/cyan]")
    console.print("  Ctrl+C to stop")

    # Reach-by-name (mDNS): advertise <mdns_name>.local on a network bind so a
    # phone reaches the GUI by name. Started HERE, before printing, so the printed
    # name reflects reality: we only recommend <name>.local when it is actually
    # being advertised (not on a loopback / --isolated bind, not when mDNS is off,
    # not when the name is taken). Closed in the finally below.
    from localm import netname
    mdns_advertiser = None
    if not is_loopback_host(host) and not isolated:
        mdns_advertiser = netname.start_advertiser(
            chosen_port, tls=bool(ssl_certfile),
            addresses=_mdns_addresses(host))
    _adv_name = netname.mdns_fqdn() if mdns_advertiser is not None else None

    # Phone / LAN access. The GUI is an installable PWA, so a phone just opens
    # this URL and adds it to the home screen. Bound to loopback, it is only
    # reachable on this machine; bound to the network, print the address a phone
    # on the same Wi-Fi can open. See docs/phone.md (Tailscale for off-LAN use).
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
        # the Tailscale MagicDNS name) and IPs so a phone needs no address typed by
        # hand.
        targets = netname.network_targets(mdns_name=_adv_name, bind_host=host)
        primary_url = None
        qr_url = None
        for _label, _target in targets:
            url = f"{scheme}://{url_host(_target)}:{chosen_port}/"
            suffix = "  [dim](open it, then Install as app)[/dim]" if primary_url is None else ""
            console.print(f"  [dim]{_label}:[/dim] [cyan]{show_url(url)}[/cyan]{suffix}")
            if primary_url is None:
                primary_url = url
            # Prefer an IP for the scannable QR (resolves on any phone, even one
            # without mDNS); fall back to the first target below.
            if qr_url is None and "(IP)" in _label:
                qr_url = url
        if primary_url is None:
            # No reachable network address detected (mDNS off, no LAN IPv4, no
            # Tailscale). Do NOT print a dead wildcard URL (https://0.0.0.0/ is not
            # connectable from anywhere) - say so honestly; the loopback URL above
            # still works on this machine.
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

    # Preload the model in the background so the first chat reply is fast.
    # Engine.load is lock-protected; a request arriving mid-load waits on it.
    def _preload():
        try:
            engine.load()
        except Exception as e:
            _report_preload_failure(console, e)

    if engine is not None:
        threading.Thread(target=_preload, daemon=True, name="preload").start()

    # Print the exact local URL (Jupyter-style) so it is copy-pasteable even if the
    # browser does not open. It carries the one-time grant, which is fine: this is the
    # host's own console. soft_wrap so the long URL is emitted as ONE line (a wrapped
    # URL with an injected newline is not copy-pasteable).
    _url_label, _shown_url = _console_url_line(api_mode, base_url, open_url)
    console.print(f"  [dim]{_url_label}:[/dim] [cyan]{show_url(_shown_url)}[/cyan]",
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
    # pywebview's webview.start() has a hard, unconditional requirement to be
    # called from the process's actual main thread. That thread is normally
    # occupied by hs.run_advertised() below (it blocks until Ctrl+C), so
    # deciding to use a native window means giving IT the main thread instead
    # and moving the server to a background one - see the branch after the
    # tray/status-window setup. Decided once, up front, so every "who opens
    # what, on which thread" choice below stays consistent.
    want_native = _should_auto_open_browser(no_browser) and appface.native_window_available()

    if _should_auto_open_browser(no_browser) and not want_native:
        threading.Thread(target=_open_when_ready, args=(open_url, chosen_port),
                         daemon=True, name="open-browser").start()

    # Record the bind host so the SPA-shell route knows whether every client is
    # loopback (a 127.0.0.1 bind) and can safely seed the API key into the page.
    # This is the EFFECTIVE host (after any config-bind fallback above), so every
    # trust decision gating on app.state.bind_host matches what is actually
    # bound. bind_fallback carries WHY a configured network bind was not applied
    # (or None) - /api/companion surfaces it so a browser-only user is told what
    # to fix instead of silently staying unreachable.
    app.state.bind_host = host
    app.state.bind_fallback = bind_fallback

    # Advertise this server in the instance registry as a "full" surface
    # (API + GUI) so a future launch in the same dir can discover and attach to
    # it. --isolated keeps it invisible to discovery.
    from localm import debuglog
    from localm.config import home_dir as _home_dir
    # Tray control surface (Windows): Open / Copy address / View logs / Restart /
    # Stop, so the running server is a real background app, not just a console.
    # Best-effort and fully guarded - it never blocks the server. Restart/Stop are
    # wired to the server's existing hooks via _tray_callbacks, NOT the bare
    # hs._do_restart/hs._do_shutdown (see that function's docstring); "View
    # logs" dumps the always-on activity buffer (INFO+, no chat content) to a
    # readable file. Linux gets a styled Tk control window; see appface.
    on_restart, on_stop = _tray_callbacks(app, hs)
    app_face = appface.start_app_face(
        name="LocaLM", url=base_url, logfile=_home_dir() / "logs" / "recent.log",
        get_log_lines=debuglog.recent_activity,
        on_restart=on_restart, on_stop=on_stop)
    if app_face is not None:
        # Accurate splash: flip the window from "Starting..." to "Running" (and, on
        # Windows, hide it to the tray) once the port is ACTUALLY accepting
        # connections. Polled in a thread so it is independent of the web
        # framework's event API (a raw TCP connect works for http and https alike).
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
            # the server is up and the tray/status window is the surface - so it
            # runs like a background app. A direct `localm gui` in a terminal has no
            # such flag, so that terminal is left alone.
            import os as _os2
            if _os2.environ.get("LOCALM_OWN_CONSOLE"):
                from localm.winconsole import hide_console
                hide_console()
        threading.Thread(target=_mark_ready_when_listening,
                         name="localm-ready", daemon=True).start()
        # Route hang-alarm surfacing into the native status window.
        # set_error turns the status red AND un-hides the window from the tray
        # (see _StatusWindow._poll's "error" branch), so a hung server is
        # unmissable instead of a log line nobody tails; set_ready restores
        # the normal Running state if the condition clears. Both are
        # queue-based and thread-safe - the alarm calls them from its own
        # daemon thread.
        hs.set_hang_surface(
            lambda text: app_face.set_error(f"Server problem: {text}"),
            app_face.set_ready)

    def _serve():
        # The advertise + run_server tail is identical to http_server.serve()'s
        # and is shared via run_advertised. The app object itself is built above
        # rather than inside serve(), because the GUI wires attach_gui and the
        # launch grants onto it first.
        try:
            hs.run_advertised(app, host, chosen_port,
                              mode="api" if api_mode else "full",
                              ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                              project=project, isolated=isolated, log_level="warning")
        finally:
            # Runs once the SERVER has actually stopped - on the main thread
            # in the browser-tab case, on the background server thread in the
            # native-window case (want_native below) - never merely once a
            # native window has closed, which can happen long before the
            # server does (closing the window must not tear down what is
            # still serving requests in the background).
            if app_face is not None:
                app_face.close()
            if mdns_advertiser is not None:
                mdns_advertiser.close()
            if manager is not None:
                manager.close_all()
            # No-op when want_native is False (no native window this run).
            # When it IS true, the window only ever hides on its own close
            # button (appface.run_native_window's _on_closing) - this is
            # what lets it actually be destroyed, so run_native_window's
            # blocking webview.start() call returns and the process can
            # exit, now that the server it was fronting has genuinely
            # stopped.
            appface.close_native_window()

    if want_native:
        # Give this thread to the server (non-daemon: closing the native
        # window must NOT kill a still-running server, matching today's
        # "closing a browser tab doesn't stop the server" behavior - Ctrl+C
        # or the tray Stop button is still how you actually stop it) and
        # hand the process's real main thread to the window instead, since
        # that is the one thread pywebview will accept.
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
        # on_quit=on_stop: the SAME callable the tray's Stop button already
        # uses (_tray_callbacks above) - when the "quit when the app window
        # is closed" setting is on, closing the window stops the server
        # exactly like clicking Stop would, instead of just hiding it.
        if not appface.run_native_window(open_url, on_quit=on_stop):
            webbrowser.open(open_url)
        # MUST join here, not just rely on server_thread being non-daemon:
        # concurrent.futures.thread registers its shutdown via CPython's
        # internal threading._register_atexit(), which fires as soon as THIS
        # (main) thread's top-level code finishes - BEFORE Python waits for
        # non-daemon threads to join. Without this join, main() returning the
        # instant the window closed flips the shared plugin executor's global
        # shutdown flag while the server thread is still alive, and every
        # in-flight request relying on get_plugin_executor() (e.g. GET
        # /api/models) raises "cannot schedule new futures after shutdown" for
        # as long as the server keeps running. Joining keeps this thread's own
        # top-level code running for exactly as long as the server is.
        server_thread.join()
    else:
        _serve()
