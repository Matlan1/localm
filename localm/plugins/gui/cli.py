# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI for the localm GUI plugin: ``localm gui``."""

from __future__ import annotations

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
    it also exposes the coder agent (shell + file edits). Traffic itself is
    encrypted by built-in TLS on a network bind (NET-1); the warning is about the
    coder agent's reach, not about cleartext.
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
    """Ask a running ``api``-mode instance to mount its GUI surface on demand
    (H6 phase 5). POSTs to its loopback ``/v1/surfaces/gui`` with the instance's
    own registry attach token (a local same-user secret). Returns True on
    success, False on any failure - an older instance without the endpoint, a
    missing token, or a network error - so the caller can fall back to just
    opening the address."""
    import requests
    scheme = entry.get("scheme") or "http"
    port = entry.get("port")
    token = entry.get("token")
    if not port or not token:
        return False
    url = f"{scheme}://127.0.0.1:{port}/v1/surfaces/gui"
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
@click.option("-p", "--port", default=None, type=click.IntRange(1, 65535),
              help="Port [default: config 'port' (8642); auto-bumps when busy].")
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
                   "network bind (-H 0.0.0.0) and the optional 'qrcode' dep "
                   "(pip install \"localm[qr]\"). Experimental.")
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
         mode, insecure, no_tls, tls_cert, tls_key, show_qr, project, force_new,
         isolated, api_mode, mmproj, device):
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

    # Attach-or-spawn (H6 phase 4): if a localm is already running for this
    # project dir, open ITS GUI instead of starting a second server that
    # double-loads the model. --new / --isolated force a fresh server.
    from localm.config import home_dir
    from localm import instances
    root_dir = instances.resolve_root_dir(override=project)
    if not (force_new or isolated):
        existing = instances.find_attachable(home_dir(), root_dir)
        if existing:
            url = instances.attach_url(existing)
            console.print(
                f"[bold green]Attaching[/bold green] to the localm already "
                f"running for [cyan]{root_dir}[/cyan] "
                f"(pid {existing.get('pid')}, port {existing.get('port')}).")
            if existing.get("mode") != "full":
                # On-demand GUI mount (H6 phase 5): the running instance is
                # API-only; ask it to mount the GUI surface live (no second
                # server, no second model load) using its own attach token.
                if _mount_remote_gui(existing):
                    console.print(
                        "  [green]Mounted the GUI on the running instance.[/green]")
                else:
                    console.print(
                        "  [yellow]Could not mount the GUI on it (an older "
                        "instance?); opening its address anyway.[/yellow]")
            console.print(f"  [dim]Opening[/dim] [cyan]{url}[/cyan]")
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
                sep = "&" if "?" in url else "?"
                webbrowser.open(f"{url}{sep}lm={_secrets.token_hex(3)}")
            return

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

    # Refuse to bind past loopback without auth unless explicitly forced: the GUI
    # exposes not just the chat API but the coder agent, which can run shell
    # commands and edit files on this machine. Checked before any setup work.
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

    # Built-in TLS (NET-1): a network bind serves HTTPS out of the box so the
    # API key and all traffic are encrypted. Resolved before attach_gui so the
    # coder/media/RAG self-call URL carries the right scheme.
    from localm.cli import _setup_tls_or_exit
    ssl_certfile, ssl_keyfile = _setup_tls_or_exit(
        host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    scheme = "https" if ssl_certfile else "http"

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
            mmproj_path=mmproj,
            device=device,
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

    async def switch_model(name: str) -> dict:
        """Swap engines, PREEMPTING any in-flight load so the latest selection
        wins immediately instead of waiting for an abandoned model to finish
        loading (see http_server.switch_engine). Serialised on the inference
        semaphore so no generation is mid-flight."""
        return await hs.switch_engine(
            name, _make_engine, on_active=lambda n: state.__setitem__("model", n))

    manager = None
    if not api_mode:
        manager = attach_gui(
            app,
            self_url=f"{scheme}://127.0.0.1:{chosen_port}/v1",
            switch_model=switch_model,
            active_model=lambda: state["model"],
        )

    base_url = f"{scheme}://127.0.0.1:{chosen_port}/"
    # Deep-link the browser to the Models page (and a pending download) when
    # the GUI was opened with --pull or with nothing registered yet.
    open_url = base_url
    if pull_spec:
        from urllib.parse import quote
        open_url = f"{base_url}?view=models&pull={quote(pull_spec, safe='')}"
    elif model_less:
        open_url = f"{base_url}?view=models"
    # One-time launch handoff: when auth is on, hand the auto-opened browser a
    # single-use grant in the URL so it lands AUTHENTICATED via a real navigation,
    # instead of depending on the implicit GET / cookie auto-seed. This runs on ANY
    # bind, including a NETWORK bind - the person launching is on THIS machine (the
    # host) and should never have to type the key, even when the server is exposed to
    # the LAN. The grant is a 256-bit single-use secret only we know and only place in
    # the URL we open locally, so a network client never sees it (see web.py's
    # redemption note). Without this, a network bind treated the host user like a
    # stranger and showed the key gate on their own machine.
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
    console.print(f"[bold green]{_srv_name}[/bold green] → {base_url}")
    if model_less:
        console.print("  model: [yellow]none yet - add one on the Models page[/yellow]")
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
    if host not in ("127.0.0.1", "localhost", "::1") and not isolated:
        mdns_advertiser = netname.start_advertiser(chosen_port, tls=bool(ssl_certfile))
    _adv_name = netname.mdns_fqdn() if mdns_advertiser is not None else None

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
        # A network bind: print the reachable NAMES (localm.local when advertised,
        # the Tailscale MagicDNS name) and IPs so a phone needs no address typed by
        # hand.
        targets = netname.network_targets(mdns_name=_adv_name)
        primary_url = None
        qr_url = None
        for _label, _target in targets:
            url = f"{scheme}://{_target}:{chosen_port}/"
            suffix = "  [dim](open it, then Install as app)[/dim]" if primary_url is None else ""
            console.print(f"  [dim]{_label}:[/dim] [cyan]{url}[/cyan]{suffix}")
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
            _ca_host = netname.ca_trust_host(_adv_name) or (
                "127.0.0.1" if host in ("0.0.0.0", "::") else host)
            console.print(
                "  [dim]first visit shows a one-time certificate warning; tap "
                "[/dim][cyan]Install certificate[/cyan][dim] on the key screen "
                "(or open [/dim][cyan]" + f"{scheme}://{_ca_host}:{chosen_port}"
                "/localm-ca.crt[/cyan][dim]) to trust it once - then no warning "
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
            console.print(f"[yellow]Background model load failed: {e}[/yellow]")

    if engine is not None:
        threading.Thread(target=_preload, daemon=True, name="preload").start()

    # Print the exact local URL (Jupyter-style) so it is copy-pasteable even if the
    # browser does not open. It carries the one-time grant, which is fine: this is the
    # host's own console. soft_wrap so the long URL is emitted as ONE line (a wrapped
    # URL with an injected newline is not copy-pasteable).
    console.print(f"  [dim]Open the GUI:[/dim] [cyan]{open_url}[/cyan]", soft_wrap=True)

    def _open_when_ready(url: str, port: int, timeout: float = 20.0) -> None:
        """Open the browser only once the server actually ACCEPTS a connection, so a
        fresh launch never lands the user on the "Can't reach the server /
        reconnecting" overlay because the tab beat the listener (the cold-start
        race). Polls the loopback port; opens anyway after *timeout* as a fallback."""
        import socket
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                _time.sleep(0.25)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if not no_browser:
        threading.Thread(target=_open_when_ready, args=(open_url, chosen_port),
                         daemon=True, name="open-browser").start()

    # Record the bind host so the SPA-shell route knows whether every client is
    # loopback (a 127.0.0.1 bind) and can safely seed the API key into the page.
    app.state.bind_host = host

    # Advertise this server in the instance registry (H6 phase 3/4) as a "full"
    # surface (API + GUI) so a future launch in the same dir can discover and
    # attach to it. --isolated keeps it invisible to discovery.
    from localm import portmux
    from localm import appface, debuglog
    from localm.config import home_dir as _home_dir
    # Tray control surface (Windows): Open / Copy address / View logs / Restart /
    # Stop, so the running server is a real background app, not just a console.
    # Best-effort and fully guarded - it never blocks the server. Restart/Stop are
    # wired to the server's existing hooks; "View logs" dumps the always-on activity
    # buffer (INFO+, no chat content) to a readable file. (Linux gets a styled Tk
    # control window next; see appface.)
    app_face = appface.start_app_face(
        name="LocaLM", url=base_url, logfile=_home_dir() / "logs" / "recent.log",
        get_log_lines=debuglog.recent_activity,
        on_restart=hs._do_restart, on_stop=hs._do_shutdown)
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
                    with socket.create_connection(("127.0.0.1", chosen_port), 0.5):
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
    try:
        with instances.advertise(app, home_dir(), host=host, port=chosen_port,
                                 mode="full", scheme=scheme, project=project,
                                 isolated=isolated):
            # On a TLS (network) bind, also catch a plain-http request on the same
            # port with an https redirect (issue 8) instead of a bare connection
            # reset; a loopback/plain bind is a direct uvicorn.run.
            portmux.run_server(app, host=host, port=chosen_port, log_level="warning",
                               ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
    finally:
        if app_face is not None:
            app_face.close()
        if mdns_advertiser is not None:
            mdns_advertiser.close()
        if manager is not None:
            manager.close_all()
