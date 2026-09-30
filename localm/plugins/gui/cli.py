# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI for the localm GUI plugin: ``localm gui``."""

from __future__ import annotations

import functools
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
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


def _phone_lan_hint(api_mode: bool) -> str:
    """Console line recommending a network bind so a phone can reach a
    loopback-bound server."""
    alt = "" if api_mode else " or Settings > Server > Bind address (set an API key first)"
    return ("  [dim]use from your phone: bind to your network with "
            "[/dim][cyan]localm gui -H 0.0.0.0[/cyan]"
            f"[dim]{alt}; see docs/phone.md[/dim]")


def _no_model_flag_hint(api_mode: bool) -> str:
    """Console line for an explicit --no-model startup."""
    if api_mode:
        return "[dim]Starting with no model loaded.[/dim]"
    return ("[dim]Opening with no model loaded - "
            "pick one on the Models page.[/dim]")


def _empty_registry_hint(api_mode: bool, pull_spec) -> str:
    """Console line for an empty model registry at startup."""
    if api_mode:
        return ("[yellow]No models registered yet.[/yellow]"
                + (" Download starting…" if pull_spec else ""))
    return ("[yellow]No models registered yet.[/yellow] "
            "Opening the GUI - add one on the Models page"
            + (" (download starting)…" if pull_spec else "."))


def _no_loadable_model_hint(api_mode: bool) -> str:
    """Console line when the registry has entries but none is a loadable chat model."""
    base = ("[yellow]No loadable chat models in the registry "
            "(files missing, not a model, or type 'unknown').[/yellow]")
    if api_mode:
        return base
    return base + (" Opening the GUI - fix, add, or set "
                    "a model's type on the Models page.")


def _engine_load_failed_hint(api_mode: bool) -> str:
    """Console line shown after a named model's engine construction raised."""
    if api_mode:
        return "[yellow]Continuing without a model loaded.[/yellow]"
    return ("[yellow]Opening the GUI model-less - pick a model on "
            "the Models page.[/yellow]")


# How long a self-restart resume (LOCALM_RESTART_IN_PROGRESS) waits for its
# own just-held port to free before giving up - see
# _restart_port_grace_window() below and config.pick_port's
# restart_grace_window.
_RESTART_PORT_GRACE_WINDOW_S = 4.0


def _restart_port_grace_window() -> float:
    """The restart_grace_window to pass to config.pick_port(): 0.0 (no
    grace) for an ordinary launch, _RESTART_PORT_GRACE_WINDOW_S when this
    process was re-exec'd by a server restart (LOCALM_RESTART_IN_PROGRESS).

    Reads the flag non-destructively (os.environ.get, never .pop):
    _should_auto_open_browser below is the flag's consumer and must still
    see it once the server has actually started."""
    import os
    return (_RESTART_PORT_GRACE_WINDOW_S
           if os.environ.get("LOCALM_RESTART_IN_PROGRESS") else 0.0)


def _should_auto_open_browser(no_browser: bool) -> bool:
    """Whether THIS process's own startup should auto-open a browser tab.

    False with --no-browser. Also False when this process was re-exec'd by a
    server restart (LOCALM_RESTART_IN_PROGRESS is set, by
    http_server._set_restart_env) and its value, the previous run's GUI
    surface, is anything other than "window": "browser", "1" (no surface
    recorded), or an unrecognised value. Pops the variable from the
    environment."""
    import os
    restart = os.environ.pop("LOCALM_RESTART_IN_PROGRESS", None)
    tab_reconnects = restart is not None and restart != "window"
    return (not no_browser) and (not tab_reconnects)


def _resolve_gui_launch_mode(no_browser: bool) -> tuple[bool, bool]:
    """Resolve (want_native, should_open_browser) for this process startup.

    want_native: whether to run the native OS app window on the main thread.
    True when no_browser is False and appface.native_window_available() is
    True, on a fresh launch and on a restart alike.

    should_open_browser: whether to spawn the background thread opening a browser
    tab. True only when want_native is False and _should_auto_open_browser is
    True: never with --no-browser, and on a restart only when the previous run
    showed the native app window. Pops LOCALM_RESTART_IN_PROGRESS from the
    environment."""
    from localm import appface
    want_native = (not no_browser) and appface.native_window_available()
    auto_open_browser = _should_auto_open_browser(no_browser)
    return want_native, (not want_native and auto_open_browser)


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


# Upper bound on how long _console_close_cleanup blocks its caller, in
# seconds. Windows allows roughly SPI_GETHUNGAPPTIMEOUT (5s by default) after
# CTRL_CLOSE_EVENT/LOGOFF/SHUTDOWN before killing the process.
_CONSOLE_CLOSE_CLEANUP_BUDGET_S = 3.0


def _console_close_cleanup() -> None:
    """Clear this instance's crash marker, then kill any running coder
    background OS subprocess (see
    localm.plugins.coder.background.JobRegistry.shutdown_all), so
    CTRL_CLOSE_EVENT/logoff/shutdown reads to the crash-recovery watchdog as a
    clean stop, the same as Ctrl+C or the GUI Stop button.

    Only the marker is cleared up front (bugreport.clear_crash_marker): the
    native-fault trace and faulthandler stay attached through the kill and are
    released (bugreport.release_crash_trace) only if the kill finishes within
    the budget. A trace left behind is handled by the next start's
    check_and_report_prior_crash(). The crash guard is left alone when no
    instance has armed one in this process (armed_instance_id() is None).

    Runs the kill on a separate daemon thread and returns within
    _CONSOLE_CLOSE_CLEANUP_BUDGET_S seconds of being called, marker removal
    included, regardless of that thread's state. Model workers
    (GGUF/embedder/STT/HF) are not touched here; see
    localm._mp_spawn.install_parent_death_watchdog.

    Passed to winconsole.register_console_handler, whose contract requires
    the callable to be quick and never raise.
    """
    deadline = time.monotonic() + _CONSOLE_CLOSE_CLEANUP_BUDGET_S
    instance_id = None
    try:
        from localm import bugreport
        instance_id = bugreport.armed_instance_id()
        if instance_id:
            bugreport.clear_crash_marker(instance_id=instance_id)
    except Exception:
        pass

    done = threading.Event()

    def _work() -> None:
        try:
            from localm.plugins.coder.background import get_registry
            get_registry().shutdown_all()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_work, daemon=True,
                     name="localm-console-close-cleanup").start()
    if done.wait(max(0.0, deadline - time.monotonic())) and instance_id:
        try:
            from localm import bugreport
            bugreport.release_crash_trace(instance_id=instance_id)
        except Exception:
            pass


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


def _mount_remote_gui(entry: dict) -> str | None:
    """Ask a running ``api``-mode instance to mount its GUI surface on demand.
    POSTs to its loopback ``/v1/surfaces/gui`` with the instance's own registry
    attach token (a local same-user secret). Returns None when the GUI was
    mounted, otherwise a short reason it was not: an instance entry without a
    port or token, no answer, an older localm without the endpoint (404 or 405),
    a refused token (401 or 403), or the status and ``detail`` of any other
    reply."""
    import requests
    scheme = entry.get("scheme") or "http"
    port = entry.get("port")
    token = entry.get("token")
    if not port:
        return "its instance entry has no port"
    if not token:
        return "its instance entry has no attach token"
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
    except requests.RequestException as e:
        return f"it did not answer ({type(e).__name__})"
    if r.status_code == 200:
        return None
    if r.status_code in (404, 405):
        return "it is an older localm that cannot mount the GUI on demand"
    if r.status_code in (401, 403):
        return f"it refused this process's attach token (HTTP {r.status_code})"
    try:
        body = r.json()
    except ValueError:
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    return f"HTTP {r.status_code}: {detail}" if detail else f"HTTP {r.status_code}"


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
        from localm.model_manager import names_same_model
        if active and not names_same_model(active, model):
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


@dataclass
class _BindPlan:
    """Where this server binds and how this machine reaches it.

    host: the effective bind host, after any loopback fallback.
    from_config: True when host came from the 'bind_host' config key, not -H.
    fallback: why a configured bind address was not applied, or None.
    ssl_certfile, ssl_keyfile: the TLS pair, or None for plain HTTP.
    port: the chosen port; 0 until _pick_gui_port runs.
    self_host: the bare address this process dials to reach itself (never a
        wildcard); "" until _pick_gui_port runs.
    """

    host: str
    from_config: bool
    fallback: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    port: int = 0
    self_host: str = ""

    @property
    def scheme(self) -> str:
        return "https" if self.ssl_certfile else "http"

    @property
    def self_authority(self) -> str:
        """``host:port`` for a URL that reaches this server from this machine;
        an IPv6 address is bracketed."""
        from localm.bindhost import url_host
        return f"{url_host(self.self_host)}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.self_authority}/"


def _prepare_console(console) -> None:
    """Prepare this process's console and app identity, then print the wordmark.

    Disables Windows QuickEdit, registers _console_close_cleanup as the console
    close handler, titles the console "LocaLM" and sets the taskbar app identity
    (applaunch.apply_window_identity), all before any window is created."""
    from localm.winconsole import (disable_quickedit, register_console_handler,
                                    set_console_title)
    disable_quickedit()
    register_console_handler(_console_close_cleanup)
    set_console_title("LocaLM")
    from localm.applaunch import apply_window_identity
    apply_window_identity()
    console.print("[bold]LocaL[/bold][bold #4f9cf9]M[/bold #4f9cf9]  [dim]local AI, offline[/dim]")


def _apply_diagnostics(console, *, debug: bool, keep_diagnostics: bool) -> None:
    """Apply --keep-diagnostics, then --debug.

    --keep-diagnostics exports LOCALM_KEEP_DIAGNOSTICS=1 for this process (read
    by config.keep_diagnostics_enabled). --debug enables the debug log and
    prints its path. Without --debug, the debug log is enabled when
    keep_diagnostics_enabled() is true; a failure there is printed and startup
    continues."""
    if keep_diagnostics:
        import os
        os.environ["LOCALM_KEEP_DIAGNOSTICS"] = "1"

    if debug:
        from localm.debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {enable_debug()}")
        return
    try:
        from localm.config import keep_diagnostics_enabled
        if keep_diagnostics_enabled():
            from localm.debuglog import enable_debug
            console.print(f"[yellow]debug log (keep_diagnostics):[/yellow] "
                          f"{enable_debug()}")
    except Exception as e:
        console.print(
            f"[yellow]could not enable the keep_diagnostics debug log:[/yellow] "
            f"{e} - bug reports will not include one.")


def _apply_session_mode(console, *, mode, debug: bool) -> None:
    """Export --mode (audit.MODE_ENV_VAR) when given, then print the effective
    server session mode when it is not privacy, or the privacy + --debug
    notice."""
    import os
    from localm.audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
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


def _attach_to_running(console, *, model, project, force_new: bool, isolated: bool,
                       api_mode: bool, no_browser: bool) -> bool:
    """Attach to a localm already running for this project instead of starting one.

    Returns False when a new server should start: with --new or --isolated, or
    when instances.find_attachable finds nothing for the project root.
    Otherwise prints the attach, asks an instance whose mode is not "full" to
    mount its GUI (_mount_remote_gui), prints its address, opens it unless
    --no-browser (_open_attached) and returns True. Exits 1 before any of that
    when an option given on the command line conflicts with the running
    instance (_attach_conflicts)."""
    from localm import instances
    from localm.config import home_dir
    from localm.console import show_url
    root_dir = instances.resolve_root_dir(override=project)
    if force_new or isolated:
        return False
    existing = instances.find_attachable(home_dir(), root_dir)
    if not existing:
        return False
    conflicts = _attach_conflicts(click.get_current_context(), existing, model)
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
        reason = _mount_remote_gui(existing)
        if reason is None:
            console.print(
                "  [green]Mounted the GUI on the running instance.[/green]")
        else:
            from rich.markup import escape
            console.print(
                f"  [yellow]Could not mount the GUI on it: {escape(reason)}; "
                "opening its address anyway.[/yellow]")
    _url_label, _ = _console_url_line(api_mode, url, url)
    console.print(f"  [dim]{_url_label}:[/dim] [cyan]{show_url(url)}[/cyan]",
                  soft_wrap=True)
    if not no_browser:
        _open_attached(url)
    return True


def _open_attached(url: str) -> None:
    """Open an attached instance's *url* with a unique ``lm`` cache-busting
    query parameter added.

    Tries the native app window first (appface.run_native_window with
    hide_on_close=False, which blocks until the window closes and must run on
    the process's main thread), then a browser tab."""
    import secrets
    from localm import appface
    sep = "&" if "?" in url else "?"
    open_url = f"{url}{sep}lm={secrets.token_hex(3)}"
    if not appface.run_native_window(open_url, hide_on_close=False):
        webbrowser.open(open_url)


def _sync_models_folder(console) -> None:
    """Reconcile the model registry with the models folder
    (model_manager.sync_models_dir; local only, no network) and print what
    changed, plus any note it returns."""
    from localm.model_manager import sync_models_dir
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


def _select_startup_model(console, model, *, no_model: bool, pull_spec,
                          api_mode: bool):
    """Choose the model this server starts with. Returns
    ``(registry, model, model_less)``.

    --no-model: model "" and model_less True. No MODEL argument and an empty
    registry: model unchanged and model_less True. No MODEL argument otherwise:
    the first registered name (sorted) that get_model_info resolves and
    is_auto_chat_eligible accepts, or None with model_less True when none does.
    A MODEL argument is returned unchanged with model_less False. Each
    model-less outcome prints its console hint."""
    from localm.config import load_registry
    from localm.model_manager import get_model_info, is_auto_chat_eligible
    registry = load_registry()
    model_less = False
    if no_model:
        model_less = True
        model = ""
        console.print(_no_model_flag_hint(api_mode))
    elif not model:
        if not registry:
            model_less = True
            console.print(_empty_registry_hint(api_mode, pull_spec))
        else:
            model = next((n for n in sorted(registry)
                          if get_model_info(n) and is_auto_chat_eligible(registry[n])), None)
            if model is None:
                model_less = True
                console.print(_no_loadable_model_hint(api_mode))
    return registry, model, model_less


def _resolve_gui_bind(console, host, *, insecure: bool) -> _BindPlan:
    """Resolve the effective bind host and apply the network-bind guards.

    The host is an explicit -H, else the 'bind_host' config key, else loopback
    (cli._resolve_bind_host). _guard_unauthenticated_bind runs first; a
    config-sourced host it left in place is then probed by
    _check_config_bind_is_bindable."""
    from localm.cli import _resolve_bind_host
    host, host_from_config = _resolve_bind_host(host)
    plan = _BindPlan(host=host, from_config=host_from_config)
    _guard_unauthenticated_bind(console, plan, insecure=insecure)
    if plan.from_config and plan.fallback is None:
        _check_config_bind_is_bindable(console, plan)
    return plan


def _guard_unauthenticated_bind(console, plan: _BindPlan, *, insecure: bool) -> None:
    """Refuse a bind past loopback without a strong API key (_gui_bind_warning).

    With --insecure (a command-line flag only; it has no config form), prints
    the warning and proceeds. Otherwise a config-sourced host is replaced by
    127.0.0.1, with the warning printed and logged and the reason recorded in
    plan.fallback; an explicit -H prints the refusal and exits 2."""
    bind_warning = _gui_bind_warning(plan.host)
    if bind_warning and not insecure and plan.from_config:
        from localm.auth import any_key_configured
        _why = ("no API key is set" if not any_key_configured()
                else "the API key is too short to be safe")
        plan.fallback = (
            f"The configured bind address ({plan.host}) was not applied: {_why}. "
            f"The server is on 127.0.0.1 (this computer only). Set a long, "
            f"random API key (Settings > Security > Owner key, or run: localm "
            f"key generate), then restart the server.")
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print(
            "[bold yellow]  Ignoring the configured bind address and binding "
            "127.0.0.1 (this computer only). Set a long, random API key, then "
            "restart.[/bold yellow]")
        from localm.debuglog import logger as _blog
        _blog.warning("config bind_host=%s not applied: %s", plan.host, _why)
        plan.host = "127.0.0.1"
    elif bind_warning and not insecure:
        console.print(f"[bold red]{bind_warning}[/bold red]")
        console.print(
            "[bold red]Refusing to start: binding past loopback without auth. "
            "Set $env:LOCALM_API_KEY first, or pass --insecure to override.[/bold red]")
        sys.exit(2)
    elif bind_warning:
        console.print(f"[bold yellow]{bind_warning}[/bold yellow]")
        console.print("[bold yellow]  Proceeding anyway (--insecure set).[/bold yellow]")


def _check_config_bind_is_bindable(console, plan: _BindPlan) -> None:
    """Probe a config-sourced plan.host with a real throwaway bind
    (cli._bind_preflight_error), loopback addresses included. When it cannot be
    bound on this machine right now, print and log it, bind 127.0.0.1 instead
    and record the reason in plan.fallback. Runs under --insecure too."""
    from localm.cli import _bind_preflight_error
    _bind_err = _bind_preflight_error(plan.host)
    if _bind_err is None:
        return
    plan.fallback = (
        f"The configured bind address ({plan.host}) was not applied: "
        f"this machine has no usable interface with that address "
        f"right now ({_bind_err}). The server is on 127.0.0.1 "
        f"(this computer only). Fix Settings > Server > Bind "
        f"address (0.0.0.0 = every interface), then restart the "
        f"server.")
    console.print(
        f"[bold yellow]The configured bind address {plan.host} cannot "
        f"be bound on this machine right now ({_bind_err}) - "
        f"ignoring it and binding 127.0.0.1 (this computer "
        f"only).[/bold yellow]")
    from localm.debuglog import logger as _plog
    _plog.warning("config bind_host=%s not applied: %s",
                  plan.host, _bind_err)
    plan.host = "127.0.0.1"


def _resolve_startup_model_file(console, model, registry, *, model_less: bool):
    """``(info, model_path, display_name)`` for the startup model, or
    ``(None, None, "")`` when model_less.

    *model* may be a registered name or a path on disk
    (registry.get_operator_model_info); display_name is the name when it is
    registered, else the resolver's display hint. Exits 1 when *model*
    resolves to nothing."""
    if model_less:
        return None, None, ""
    from localm.model_manager.registry import get_operator_model_info
    info = get_operator_model_info(model)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)
    model_path, display_hint = info
    return info, model_path, (model if model in registry else display_hint)


def _resolve_gui_tls(console, plan: _BindPlan, *, no_tls: bool, tls_cert,
                     tls_key) -> None:
    """Set plan.ssl_certfile and plan.ssl_keyfile: built-in TLS past loopback
    unless --no-tls, or the --tls-cert/--tls-key pair (cli._resolve_tls).

    For a config-sourced host, a failure other than click.UsageError is printed
    and logged, and the server binds 127.0.0.1 over plain HTTP instead, with the
    reason recorded in plan.fallback. For an explicit -H,
    cli._setup_tls_or_exit exits on a failure."""
    from localm.cli import _resolve_tls, _setup_tls_or_exit
    if not plan.from_config:
        plan.ssl_certfile, plan.ssl_keyfile = _setup_tls_or_exit(
            plan.host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
        return
    try:
        plan.ssl_certfile, plan.ssl_keyfile = _resolve_tls(
            plan.host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    except click.UsageError:
        raise
    except Exception as e:
        plan.fallback = (
            f"The configured bind address ({plan.host}) was not applied: "
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
                      plan.host, e)
        plan.host = "127.0.0.1"
        plan.ssl_certfile = plan.ssl_keyfile = None


def _pick_gui_port(console, plan: _BindPlan, port) -> None:
    """Choose the port and set plan.port and plan.self_host.

    config.pick_port probes the loopback address plan.host covers and, on a
    restart re-exec, waits for the port to free (_restart_port_grace_window).
    A busy explicit --port exits 1 and is never relocated; a busy default is
    bumped and reported."""
    from localm.bindhost import self_connect_host
    from localm.config import PortInUseError, pick_port
    try:
        chosen_port, was_busy = pick_port(
            port, host=self_connect_host(plan.host),
            restart_grace_window=_restart_port_grace_window())
    except PortInUseError as exc:
        console.print(f"[red]Port {exc.port} is already in use.[/red] "
                      "Free it, or choose another with -p/--port.")
        sys.exit(1)
    if was_busy:
        console.print(f"[yellow]Default port busy - using {chosen_port}.[/yellow]")
    plan.port = chosen_port
    plan.self_host = self_connect_host(plan.host)


def _engine_factories(model, *, ctx, gpu_layers, mmproj, device):
    """Build ``(engine_for, make_engine)`` for this run's engine options.

    engine_for(name, m_info, mmproj_path) constructs an unloaded inference
    Engine from a resolved ``(path, display_hint)`` pair, with --ctx,
    --gpu-layers and --device; its display_name is *name* when registered, else
    the hint. make_engine(name) is http_server.switch_engine's factory: it
    resolves registered names only (ValueError otherwise) and applies --mmproj
    only when *name* is the startup *model*; every other model gets its own
    projector (get_model_mmproj). See TestMmprojScopedToStartupModel."""
    from localm.config import load_registry
    from localm.inference.engine import Engine
    from localm.model_manager import get_model_info, get_model_mmproj

    def _engine_for(name: str, m_info, mmproj_path) -> Engine:
        m_path, m_hint = m_info
        return Engine(
            str(m_path),
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            mmproj_path=mmproj_path,
            device=device,
            display_name=name if name in load_registry() else m_hint,
        )

    def _make_engine(name: str) -> Engine:
        m_info = get_model_info(name)
        if m_info is None:
            raise ValueError(f"Model not found: {name}")
        mmproj_path = (mmproj if name == model else None) or get_model_mmproj(name)
        return _engine_for(name, m_info, mmproj_path)

    return _engine_for, _make_engine


def _build_startup_engine(console, engine_for, model, info, *, mmproj,
                          api_mode: bool):
    """Construct the startup model's unloaded Engine with --mmproj, else its
    recorded projector (registry.get_operator_model_mmproj).

    Returns ``(engine, False)``, or ``(None, True)`` after printing the error
    and the model-less hint when construction raises."""
    from localm.model_manager.registry import get_operator_model_mmproj
    try:
        return engine_for(model, info, mmproj or get_operator_model_mmproj(model)), False
    except Exception as e:
        console.print(f"[yellow]Could not load model '{model}': {e}[/yellow]")
        console.print(_engine_load_failed_hint(api_mode))
        return None, True


def _build_app(engine, make_engine, plan: _BindPlan, *, api_mode: bool):
    """Create the server app around *engine* (None: model-less) and, unless
    api_mode, mount the web GUI on it, pointed at this server's own /v1.

    Returns ``(app, manager)``: manager is attach_gui's SessionManager, or None
    in api_mode."""
    from localm.inference import http_server as hs
    from .web import attach_gui
    app = hs.create_app(engine)

    async def switch_model(name: str, *, force: bool = False) -> dict:
        """Swap engines, PREEMPTING any in-flight load so the latest selection
        wins immediately instead of waiting for an abandoned model to finish
        loading (see http_server.switch_engine). Serialised on the inference
        semaphore so no generation is mid-flight."""
        return await hs.switch_engine(name, make_engine, force=force)

    manager = None
    if not api_mode:
        manager = attach_gui(
            app,
            self_url=f"{plan.scheme}://{plan.self_authority}/v1",
            switch_model=switch_model,
            # The live active-model pointer, updated by switch_engine on load
            # and by unload_all_models/unload_one_model on unload.
            active_model=lambda: hs._active_model_name or "",
        )
    return app, manager


def _launch_url(app, base_url: str, *, pull_spec, model_less: bool) -> str:
    """The URL the launching window or browser tab opens.

    base_url, deep-linked to the Models page with a pending download and a
    single-use, spec-bound pull grant (web.mint_pull_grant) for --pull, or to
    the Models page when model_less. When an API key is set, a single-use launch
    grant (web.mint_launch_grant) is added as ``localm_token``, which the GUI
    redeems to sign the opened page in."""
    open_url = base_url
    if pull_spec:
        from urllib.parse import quote
        from .web import mint_pull_grant
        pull_token = mint_pull_grant(app, pull_spec)
        open_url = (f"{base_url}?view=models&pull={quote(pull_spec, safe='')}"
                    f"&pull_token={quote(pull_token, safe='')}")
    elif model_less:
        open_url = f"{base_url}?view=models"
    from localm import auth as _auth
    from .web import mint_launch_grant
    if _auth.get_api_key():
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
        _p = urlparse(open_url)
        _q = dict(parse_qsl(_p.query))
        _q["localm_token"] = mint_launch_grant(app)
        open_url = urlunparse(_p._replace(query=urlencode(_q)))
    return open_url


def _announce_server(console, plan: _BindPlan, *, api_mode: bool, model_less: bool,
                     model, display_name: str, model_path) -> None:
    """Retitle the console with the port (and the model, when one is starting),
    then print the server banner, the model line and the stop hint."""
    from localm.console import show_url
    from localm.winconsole import set_console_title
    _wtitle = f"LocaLM  -  localhost:{plan.port}"
    if not model_less and (display_name or model):
        _wtitle = f"LocaLM  -  {display_name or model}  -  :{plan.port}"
    set_console_title(_wtitle)
    _srv_name = "localm API server" if api_mode else "localm GUI"
    console.print(f"[bold green]{_srv_name}[/bold green] → {show_url(plan.base_url)}")
    if model_less:
        console.print(_model_less_hint(api_mode))
    else:
        console.print(f"  model: [cyan]{display_name or Path(str(model_path)).stem}[/cyan]")
    console.print("  Ctrl+C to stop")


def _start_mdns(plan: _BindPlan, *, isolated: bool):
    """Advertise ``<name>.local`` over mDNS (netname.start_advertiser) for a
    network bind that is not --isolated.

    Returns ``(advertiser, fqdn)``: the advertiser, to close once serving ends,
    and the advertised name; both None when nothing is advertised."""
    from localm import netname
    from localm.bindhost import is_loopback_host
    advertiser = None
    if not is_loopback_host(plan.host) and not isolated:
        advertiser = netname.start_advertiser(
            plan.port, tls=bool(plan.ssl_certfile),
            addresses=_mdns_addresses(plan.host))
    return advertiser, (netname.mdns_fqdn() if advertiser is not None else None)


def _print_reach_hints(console, plan: _BindPlan, adv_name, *, api_mode: bool,
                       show_qr: bool) -> None:
    """Print how to reach this server from a phone or another machine.

    A loopback bind prints the network-bind hint, and with --qr the note that a
    QR needs a network bind. A network bind prints each reachable name and IP
    (netname.network_targets, including *adv_name* when advertised), or that
    none was found; on HTTPS, the one-time certificate-trust steps; the
    Tailscale rename hint when there is one; and with --qr, a QR of the first
    IP address, else of the first address."""
    from localm import netname
    from localm.bindhost import is_loopback_host, self_connect_host, url_host
    from localm.console import show_url
    host, port, scheme = plan.host, plan.port, plan.scheme
    if is_loopback_host(host):
        console.print(_phone_lan_hint(api_mode))
        if show_qr:
            console.print(
                "  [yellow][PoC][/yellow] [dim]--qr needs a network bind to be "
                "scannable: [/dim][cyan]localm gui -H 0.0.0.0 --qr[/cyan]")
        return
    targets = netname.network_targets(mdns_name=adv_name, bind_host=host)
    primary_url = None
    qr_url = None
    for _label, _target in targets:
        url = f"{scheme}://{url_host(_target)}:{port}/"
        suffix = "  [dim](open it, then Install as app)[/dim]" if primary_url is None else ""
        console.print(f"  [dim]{_label}:[/dim] [cyan]{show_url(url)}[/cyan]{suffix}")
        if primary_url is None:
            primary_url = url
        if qr_url is None and "(IP)" in _label:
            qr_url = url
    if primary_url is None:
        console.print("  [dim]no reachable network address detected - "
                      "this machine only[/dim]")
    if scheme == "https":
        _ca_host = url_host(netname.ca_trust_host(adv_name)
                            or self_connect_host(host))
        console.print(
            "  [dim]first visit shows a one-time certificate warning; tap "
            "[/dim][cyan]Install certificate[/cyan][dim] on the key screen "
            "(or open [/dim][cyan]"
            + show_url(f"{scheme}://{_ca_host}:{port}/localm-ca.crt")
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


def _start_preload(console, engine) -> None:
    """Load *engine* on a daemon thread named "preload"; a request that arrives
    mid-load waits on Engine.load's lock. A failure is reported by
    _report_preload_failure. No-op when engine is None."""
    if engine is None:
        return

    def _preload():
        try:
            engine.load()
        except Exception as e:
            _report_preload_failure(console, e)

    threading.Thread(target=_preload, daemon=True, name="preload").start()


def _open_when_ready(url: str, self_host: str, port: int, timeout: float = 20.0) -> None:
    """Open *url* in a browser tab once ``(self_host, port)`` accepts a TCP
    connection, polling every 0.25 s, or after *timeout* seconds regardless. A
    webbrowser.open failure is swallowed."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((self_host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _start_launch_surface(open_url: str, plan: _BindPlan, *, no_browser: bool) -> bool:
    """Choose this run's launch surface and record it for a restart.

    Returns want_native from _resolve_gui_launch_mode, which pops
    LOCALM_RESTART_IN_PROGRESS. When a browser tab is due, starts the
    "open-browser" daemon thread running _open_when_ready. Records "window", or
    "browser" unless --no-browser, with http_server.set_restart_ui."""
    from localm.inference import http_server as hs
    want_native, should_open_browser = _resolve_gui_launch_mode(no_browser)
    if should_open_browser:
        threading.Thread(target=_open_when_ready,
                         args=(open_url, plan.self_host, plan.port),
                         daemon=True, name="open-browser").start()
    if want_native:
        hs.set_restart_ui("window")
    elif not no_browser:
        hs.set_restart_ui("browser")
    return want_native


def _mark_ready_when_listening(app_face, self_host: str, port: int) -> None:
    """Poll ``(self_host, port)`` every 0.25 s for up to 160 attempts, then mark
    *app_face* ready either way. When LOCALM_OWN_CONSOLE is set (a launcher
    started this process with its own console), hide that console."""
    import os
    import socket
    for _ in range(160):
        try:
            with socket.create_connection((self_host, port), 0.5):
                break
        except OSError:
            time.sleep(0.25)
    app_face.set_ready()
    if os.environ.get("LOCALM_OWN_CONSOLE"):
        from localm.winconsole import hide_console
        hide_console()


def _start_app_face(plan: _BindPlan, *, on_restart, on_stop, no_browser: bool):
    """Start the tray / status window (appface.start_app_face) for plan.base_url,
    wired to *on_restart* and *on_stop*; with --no-browser only the tray
    starts, without the status window.

    When one starts, a "localm-ready" daemon thread runs
    _mark_ready_when_listening, and the hang alarm reports into it
    (http_server.set_hang_surface: set_error on a problem, set_ready on
    recovery). Returns the app face, or None."""
    from localm import appface, debuglog
    from localm.config import home_dir
    from localm.inference import http_server as hs
    app_face = appface.start_app_face(
        name="LocaLM", url=plan.base_url, logfile=home_dir() / "logs" / "recent.log",
        get_log_lines=debuglog.recent_activity,
        on_restart=on_restart, on_stop=on_stop, show_window=not no_browser)
    if app_face is not None:
        threading.Thread(target=_mark_ready_when_listening,
                         args=(app_face, plan.self_host, plan.port),
                         name="localm-ready", daemon=True).start()
        hs.set_hang_surface(
            lambda text: app_face.set_error(f"Server problem: {text}"),
            app_face.set_ready)
    return app_face


def _release_after_serving(app_face, mdns_advertiser, manager,
                           server_stopped: threading.Event) -> None:
    """Release what startup opened, once the server has stopped: close the tray /
    status window, the mDNS advertiser and the GUI session manager (each when
    present), set *server_stopped*, then close the native app window
    (appface.close_native_window; a no-op when none is open)."""
    from localm import appface
    if app_face is not None:
        app_face.close()
    if mdns_advertiser is not None:
        mdns_advertiser.close()
    if manager is not None:
        manager.close_all()
    server_stopped.set()
    appface.close_native_window()


def _serve_then_release(app, plan: _BindPlan, *, api_mode: bool, project,
                        isolated: bool, release) -> None:
    """Serve *app* on plan's host and port until the server stops
    (http_server.run_advertised: advertised in the instance registry unless
    isolated), then call release(), also when serving raises."""
    from localm.inference import http_server as hs
    try:
        hs.run_advertised(app, plan.host, plan.port,
                          mode="api" if api_mode else "full",
                          ssl_certfile=plan.ssl_certfile, ssl_keyfile=plan.ssl_keyfile,
                          project=project, isolated=isolated, log_level="warning")
    finally:
        release()


def _serve_beside_native_window(serve, plan: _BindPlan, open_url: str, *, on_quit,
                                server_stopped: threading.Event) -> None:
    """Run serve() on a non-daemon "localm-server" thread and give this thread,
    which must be the process's main thread, to the native app window.

    Stop signals are routed to the server thread while the window holds this
    thread (portmux.route_stop_signals). The window opens once the port accepts
    a connection, after 20 s, or once *server_stopped* is set; its quit action
    is *on_quit*. When the window cannot open while the server is still running,
    a browser tab opens instead and "browser" is recorded for a restart; once
    the server has stopped, no tab opens. Returns once the server thread has
    ended."""
    import socket
    from localm import appface, portmux
    from localm.inference import http_server as hs
    with portmux.route_stop_signals(serving_elsewhere=True):
        server_thread = threading.Thread(target=serve, name="localm-server",
                                         daemon=False)
        server_thread.start()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not server_stopped.is_set():
            try:
                with socket.create_connection((plan.self_host, plan.port), 0.5):
                    break
            except OSError:
                time.sleep(0.25)
        window_loaded = appface.run_native_window(
            open_url, on_quit=on_quit, server_stopped=server_stopped)
        if not window_loaded and not server_stopped.is_set():
            hs.set_restart_ui("browser")
            webbrowser.open(open_url)
        # Returns only once the server thread has ended.
        # See test_native_window_close_waits_for_the_server_to_stop.
        server_thread.join()


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

    # Argument and config resolution.
    _prepare_console(console)
    _apply_diagnostics(console, debug=debug, keep_diagnostics=keep_diagnostics)
    _apply_session_mode(console, mode=mode, debug=debug)

    # Attach decision: open the instance already running for this project, if any.
    if _attach_to_running(console, model=model, project=project, force_new=force_new,
                          isolated=isolated, api_mode=api_mode, no_browser=no_browser):
        return

    _sync_models_folder(console)
    registry, model, model_less = _select_startup_model(
        console, model, no_model=no_model, pull_spec=pull_spec, api_mode=api_mode)
    plan = _resolve_gui_bind(console, host, insecure=insecure)
    info, model_path, display_name = _resolve_startup_model_file(
        console, model, registry, model_less=model_less)
    _resolve_gui_tls(console, plan, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    _pick_gui_port(console, plan, port)

    # Engine creation and app construction.
    engine_for, make_engine = _engine_factories(
        model, ctx=ctx, gpu_layers=gpu_layers, mmproj=mmproj, device=device)
    engine = None
    if not model_less:
        engine, model_less = _build_startup_engine(
            console, engine_for, model, info, mmproj=mmproj, api_mode=api_mode)
    app, manager = _build_app(engine, make_engine, plan, api_mode=api_mode)
    open_url = _launch_url(app, plan.base_url, pull_spec=pull_spec, model_less=model_less)

    _announce_server(console, plan, api_mode=api_mode, model_less=model_less,
                     model=model, display_name=display_name, model_path=model_path)
    mdns_advertiser, adv_name = _start_mdns(plan, isolated=isolated)
    _print_reach_hints(console, plan, adv_name, api_mode=api_mode, show_qr=show_qr)

    # Preload.
    _start_preload(console, engine)
    _url_label, _shown_url = _console_url_line(api_mode, plan.base_url, open_url)
    console.print(f"  [dim]{_url_label}:[/dim] [cyan]{show_url(_shown_url)}[/cyan]",
                  soft_wrap=True)

    # Browser / window / tray.
    want_native = _start_launch_surface(open_url, plan, no_browser=no_browser)
    app.state.bind_host = plan.host
    app.state.bind_fallback = plan.fallback
    from localm.inference import http_server as hs
    on_restart, on_stop = _tray_callbacks(app, hs)
    app_face = _start_app_face(plan, on_restart=on_restart, on_stop=on_stop,
                               no_browser=no_browser)

    # Server start; everything above is released once serving ends.
    server_stopped = threading.Event()
    release = functools.partial(_release_after_serving, app_face, mdns_advertiser,
                                manager, server_stopped)
    serve = functools.partial(_serve_then_release, app, plan, api_mode=api_mode,
                              project=project, isolated=isolated, release=release)
    if want_native:
        _serve_beside_native_window(serve, plan, open_url, on_quit=on_stop,
                                    server_stopped=server_stopped)
    else:
        serve()
