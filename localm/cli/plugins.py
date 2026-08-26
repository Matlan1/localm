# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ..console import show_url
from ._core import console, main
from .errors import run_or_die
from ..bindhost import self_connect_host, url_host


# ------------------------------------------------------------------ #
#  Plugin management (external plugins in <data dir>/plugins/)         #
# ------------------------------------------------------------------ #

@main.group()
def plugin() -> None:
    """Manage plugins (status/install/uninstall/enable/disable)."""




# Engine plugins (builtin + external) - install/uninstall + enable/disable +
# status. Two axes: install/uninstall moves a plugin between the available catalog
# (builtin/ + external dirs) and the user's installed set; enable/disable toggles
# an installed plugin active/inactive. A PluginManager with no app is enough to
# flip config (no routes are mounted from the CLI - a running GUI server picks
# HTTP routes up on its next start, while stdio plugins like mcp take effect
# immediately). Nothing is installed by default: only what the user selects.

def _engine_manager():
    from ..plugins.engine import PluginManager
    return PluginManager(None)




def _warn_missing_requires(mgr, name):
    from rich.markup import escape

    missing = mgr.missing_requires(name)
    if missing:
        cmds = "  ".join(f"localm plugin install {escape(m)}" for m in missing)
        console.print(
            f"[yellow]Note:[/yellow] {escape(repr(name))} declares it needs "
            f"{', '.join(escape(m) for m in missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f"Install with:  {cmds}")


# ---- pip-extra auto-install (host-side; the CLI always runs on the host) ---- #

def _auto_deps_default() -> bool:
    """The configured default for auto-installing plugin pip extras."""
    from localm.config import load_config
    return bool(load_config().get("auto_install_plugin_deps", True))


def _set_auto_deps(value: bool) -> None:
    from localm.config import update_config
    update_config(lambda cfg: cfg.__setitem__("auto_install_plugin_deps", bool(value)))


def _console_progress(line: str) -> None:
    # pip output can contain "[" (e.g. localm[voice]); disable rich markup so it
    # is printed verbatim rather than parsed as a style tag.
    console.print(line, markup=False, highlight=False)


def _install_deps(mgr, name) -> bool:
    """Install *name*'s declared pip extras on this host. Returns True on success
    (including the no-op case where the plugin declares none / all are present).
    Surfaces the real installer error on failure (never a hollow success)."""
    from rich.markup import escape

    if not mgr.plugin_missing_deps(name):
        return True                     # nothing declared, or already satisfied
    console.print(f"[dim]Installing dependencies for {escape(name)}...[/dim]")
    res = mgr.install_plugin_deps(name, on_progress=_console_progress)
    if res.ok:
        if res.installed:
            console.print(
                f"[green]Installed dependencies[/green] for "
                f"[bold]{escape(name)}[/bold]: "
                f"{', '.join(escape(i) for i in res.installed)}")
        return True
    console.print(f"[red]Dependency install failed for {escape(name)}:[/red] "
                  f"{escape(res.error)}")
    console.print('[dim]Install manually, e.g.:  '
                  'pip install "localm[<extra>]"[/dim]')
    return False


def _resolve_with_deps(flag) -> bool:
    """The effective auto-install decision: an explicit --with-deps/--no-deps
    wins, else the configured default."""
    return _auto_deps_default() if flag is None else bool(flag)




@plugin.command("install")
@click.argument("target")
@click.option("--force", is_flag=True,
              help="When installing from a directory, overwrite an existing install.")
@click.option("--with-deps/--no-deps", "with_deps", default=None,
              help="Also install the plugin's pip extras on this machine "
                   "(default: the auto_install_plugin_deps setting).")
def plugin_install_engine(target, force, with_deps):
    """Install a plugin and enable it.

    TARGET is either a first-party plugin NAME from the bundled store
    (e.g. ``localm plugin install coder``) or a path to a DIRECTORY containing a
    plugin.toml (a third-party plugin). A first-party plugin's pip extras are
    installed for you unless you pass --no-deps (or turn the setting off).
    """
    from rich.markup import escape

    from localm import cli as _cli
    mgr = _cli._engine_manager()
    src = Path(target)
    if src.is_dir() and (src / "plugin.toml").is_file():
        try:
            spec = mgr.set_installed_from_dir(src, force=force)
        except ValueError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            sys.exit(1)
        console.print(f"[green]Installed[/green] plugin "
                      f"[bold]{escape(spec.name)}[/bold] v{escape(spec.version)}")
        console.print(f"[dim]Granted scope:[/dim] [bold]{escape(spec.scope)}[/bold] "
                      f"(every route/tool it registers is gated on this "
                      f"capability)")
        _warn_missing_requires(mgr, spec.name)
        # A third-party plugin's extras are its own (not localm's); we do not
        # auto-resolve those here. Point the user at its own instructions.
        if spec.requires_extras:
            console.print(
                f"[dim]{escape(spec.name)} declares extra dependencies "
                f"({', '.join(escape(x) for x in spec.requires_extras)}); "
                f"install per its own instructions.[/dim]")
        return
    run_or_die(mgr.set_installed_state, target, True,
              missing_msg=f"No such plugin: {escape(target)}")
    console.print(f"[green]Installed[/green] plugin [bold]{escape(target)}[/bold]")
    _warn_missing_requires(mgr, target)
    if _resolve_with_deps(with_deps):
        _install_deps(mgr, target)
    elif mgr.plugin_missing_deps(target):
        console.print('[dim]Needs pip extras; install them with:  '
                      f'localm plugin install-deps {escape(target)}[/dim]')




# Recommended everyday set for a non-interactive install: the surfaces that work
# without a separate external service. image/music/video need a ComfyUI server,
# voice needs the [voice] extra + a model, mcp is for external clients - all opt-in.
_SETUP_DEFAULTS = ("coder", "rag", "web", "tts")




def _installed_plugin_names(mgr) -> set:
    return {p["name"] for p in mgr.api_state()["plugins"] if p.get("installed")}




def _parse_plugin_selection(raw, available):
    """Parse an interactive selection into a list of plugin names. Accepts a
    comma list of 1-based numbers, names, numeric ranges like ``2-5``, or
    ``all``. Unknown/out-of-range tokens are flagged and skipped; the result is
    de-duplicated while preserving order."""
    from rich.markup import escape

    if not raw.strip():
        return []
    if raw.strip().lower() == "all":
        return [e.name for e in available]
    by_idx = {str(i): e.name for i, e in enumerate(available, 1)}
    names = {e.name for e in available}
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if t in by_idx:
            out.append(by_idx[t])
        elif t in names:
            out.append(t)
        elif t.count("-") == 1 and all(p.strip().isdecimal() for p in t.split("-")):
            # A numeric range like "2-5": expand to the valid indices it covers.
            # isdecimal (not isdigit) so an exotic Unicode digit that int() would
            # reject never slips through and raises here.
            lo, hi = (int(p) for p in t.split("-"))
            matched = [by_idx[str(n)] for n in range(lo, hi + 1) if str(n) in by_idx]
            if matched:
                out.extend(matched)
            else:
                console.print(f"[yellow]Ignoring out-of-range selection: "
                              f"{escape(t)}[/yellow]")
        else:
            console.print(f"[yellow]Ignoring unknown selection: {escape(t)}[/yellow]")
    return list(dict.fromkeys(out))




@plugin.command("setup")
@click.option("--plugins", "plugins_csv", default=None,
              help="Comma-list to install non-interactively (skips the prompt).")
@click.option("--all", "install_all", is_flag=True,
              help="Install every first-party plugin.")
@click.option("--defaults", "install_defaults", is_flag=True,
              help="Install the recommended set non-interactively (coder, rag, web, tts).")
@click.option("--with-deps/--no-deps", "with_deps", default=None,
              help="Install pip extras for the chosen plugins (default: ask "
                   "interactively, else the auto_install_plugin_deps setting).")
def plugin_setup(plugins_csv, install_all, install_defaults, with_deps=None):
    """Choose which first-party plugins to install.

    Out of the box only chat is active; this turns on the features you want
    (coder, image/music/video, rag, web, voice, tts, mcp). Run by the installer
    after dependencies are in place, and any time afterwards. Plugins that need a
    pip extra have it installed for you (you are asked once; --with-deps/--no-deps
    or the auto_install_plugin_deps setting decide non-interactively).
    """
    from rich.markup import escape

    from ..plugins import catalog
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    installed = _installed_plugin_names(mgr)
    available = [e for e in catalog.CATALOG if not e.preinstalled]
    interactive = False

    if install_all:
        chosen = [e.name for e in available]
    elif plugins_csv is not None:
        chosen = _parse_plugin_selection(plugins_csv, available)
        # Non-interactive: a non-empty --plugins that resolved to NOTHING is a
        # typo, not a deliberate skip. Fail loudly so an install/CI script does
        # not read a no-op as success (the per-token "Ignoring unknown selection"
        # notes above already name which tokens were bad). A blank/whitespace
        # value is a deliberate skip and is left to the generic path below.
        if plugins_csv.strip() and not chosen:
            console.print(
                f"[red]No known plugins in --plugins {escape(repr(plugins_csv))}."
                f"[/red] Choose from: "
                f"{', '.join(escape(e.name) for e in available)}.")
            sys.exit(1)
    elif install_defaults:
        chosen = list(_SETUP_DEFAULTS)
    elif not sys.stdin.isatty():
        console.print(
            "[dim]Non-interactive shell - skipping plugin selection. Run "
            "[bold]localm plugin setup[/bold] to choose, or pass "
            "--plugins/--all/--defaults.[/dim]")
        return
    else:
        interactive = True
        console.print("Choose first-party plugins to install (chat is always on):\n")
        for i, e in enumerate(available, 1):
            mark = " [green](installed)[/green]" if e.name in installed else ""
            # localm[<extra>] contains a real, literal "[...]" pip-extra syntax
            # span - escape the WHOLE "localm[...]" fragment (not just e.extra),
            # or Rich parses "[coder]"/"[voice]" etc. as a bogus style tag and
            # silently drops it, printing "(pip extra: localm)" - confirmed
            # empirically against this venv's rich before this fix.
            extra = (f" [dim](pip extra: {escape(f'localm[{e.extra}]')})[/dim]"
                     if e.extra else "")
            console.print(f"  [bold]{i:>2}.[/bold] {escape(e.name):<7} "
                          f"{escape(e.description)}{mark}{extra}")
        console.print(
            "\nEnter numbers or names (comma-separated), a range like 1-5, "
            "'all', or blank to skip.")
        console.print(
            "[dim]This ADDS the features you pick; anything already installed "
            "stays. Remove one later with: localm plugin uninstall <name>.[/dim]")
        # Re-ask on an entry that matched nothing (e.g. junk like "ewew"), so we
        # never silently leave zero plugins after the user clearly tried to pick
        # something. A blank entry is a deliberate "skip" and breaks the loop.
        while True:
            raw = click.prompt("Install", default="", show_default=False)
            chosen = _parse_plugin_selection(raw, available)
            if raw.strip() and not chosen:
                console.print("[yellow]Nothing recognised in that entry. Use the "
                              "numbers/names listed above, a range like 1-5, or "
                              "leave blank to skip.[/yellow]")
                continue
            break

    if not chosen:
        console.print("[dim]No plugins selected.[/dim]")
        return
    for name in chosen:
        if name in installed:
            console.print(f"[dim]{escape(name)} already installed[/dim]")
            continue
        try:
            mgr.set_installed_state(name, True)
        except (KeyError, ValueError) as e:
            console.print(f"[yellow]Skipped {escape(name)}: {escape(str(e))}[/yellow]")
            continue
        console.print(f"[green]Installed[/green] [bold]{escape(name)}[/bold]")
        _warn_missing_requires(mgr, name)

    # Install pip extras for the chosen plugins. Interactively, ask once and
    # remember the answer as the auto_install_plugin_deps setting; otherwise the
    # --with-deps/--no-deps flag or the existing setting decides.
    want = with_deps
    if want is None and interactive:
        need_any = any(mgr.plugin_missing_deps(n) for n in chosen)
        if need_any:
            want = click.confirm(
                "Install the extra Python packages these plugins need?",
                default=_auto_deps_default())
            _set_auto_deps(want)
    if want is None:
        want = _auto_deps_default()
    pending = [n for n in chosen if mgr.plugin_missing_deps(n)]
    if pending and want:
        console.print("\n[bold]Installing plugin dependencies...[/bold]")
        for n in pending:
            _install_deps(mgr, n)
    elif pending:
        console.print("\n[dim]Skipped dependencies for: "
                      f"{', '.join(escape(n) for n in pending)}. Install later "
                      "with:  localm plugin install-deps --all[/dim]")




@plugin.command("install-deps")
@click.argument("name", required=False)
@click.option("--all", "do_all", is_flag=True,
              help="Install missing pip extras for every enabled plugin.")
def plugin_install_deps(name, do_all):
    """Install missing pip extras for plugins on this machine (host repair).

    With a NAME, installs that plugin's declared extras. With --all, scans every
    enabled plugin and installs anything missing. With neither, just lists what
    is missing. This runs pip locally, so it is a host-only operation.
    """
    from rich.markup import escape

    from ..plugins import catalog
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    if name and do_all:
        console.print("[red]Give a NAME or --all, not both.[/red]")
        sys.exit(1)
    if name:
        known = set(catalog.names()) | _installed_plugin_names(mgr)
        if name not in known:
            console.print(f"[red]No such plugin: {escape(name)}[/red]")
            sys.exit(1)
        if not mgr.plugin_missing_deps(name):
            console.print(f"[green]{escape(name)} has its dependencies "
                          "(or declares none).[/green]")
            return
        sys.exit(0 if _install_deps(mgr, name) else 1)
    missing = mgr.all_missing_deps(enabled_only=True)
    if not missing:
        console.print("[green]All enabled plugins have their dependencies.[/green]")
        return
    if not do_all:
        console.print("[bold]Missing plugin dependencies:[/bold]")
        for n, reqs in missing.items():
            console.print(f"  [bold]{escape(n)}[/bold]: "
                          f"{', '.join(escape(r) for r in reqs)}")
        console.print("\nInstall them with:  [bold]localm plugin install-deps --all[/bold]")
        return
    ok = True
    for n in missing:
        ok = _install_deps(mgr, n) and ok
    sys.exit(0 if ok else 1)




@plugin.command("uninstall")
@click.argument("name")
@click.option("--delete-data", is_flag=True,
              help="Also delete this plugin's stored data (default: keep it).")
def plugin_uninstall_engine(name, delete_data):
    """Uninstall (deselect) an engine plugin. Keeps its data unless --delete-data."""
    from rich.markup import escape

    from localm import cli as _cli
    mgr = _cli._engine_manager()
    was = run_or_die(mgr.uninstall, name, delete_data=delete_data,
                     missing_msg=f"No such plugin: {escape(name)}")
    if was:
        console.print(f"[yellow]Uninstalled[/yellow] plugin [bold]{escape(name)}[/bold]")
    else:
        console.print(f"[dim]Plugin {escape(repr(name))} was not installed.[/dim]")




@plugin.command("enable")
@click.argument("name")
@click.option("--with-deps/--no-deps", "with_deps", default=None,
              help="Also install the plugin's pip extras on this machine "
                   "(default: the auto_install_plugin_deps setting).")
def plugin_enable(name, with_deps):
    """Enable an installed engine plugin (must be installed first)."""
    from rich.markup import escape

    from localm import cli as _cli
    mgr = _cli._engine_manager()
    run_or_die(mgr.set_enabled_state, name, True,
              missing_msg=f"No such plugin: {escape(name)}")
    console.print(f"[green]Enabled[/green] plugin [bold]{escape(name)}[/bold]")
    _warn_missing_requires(mgr, name)
    if _resolve_with_deps(with_deps):
        _install_deps(mgr, name)
    elif mgr.plugin_missing_deps(name):
        console.print('[dim]Needs pip extras; install them with:  '
                      f'localm plugin install-deps {escape(name)}[/dim]')




@plugin.command("disable")
@click.argument("name")
def plugin_disable(name):
    """Disable an installed engine plugin (keeps it installed)."""
    from rich.markup import escape

    from localm import cli as _cli
    mgr = _cli._engine_manager()
    run_or_die(mgr.set_enabled_state, name, False,
              missing_msg=f"No such plugin: {escape(name)}")
    console.print(f"[yellow]Disabled[/yellow] plugin [bold]{escape(name)}[/bold]")




@plugin.command("refresh")
@click.argument("name", required=False)
def plugin_refresh(name):
    """Re-sync installed first-party plugins with the bundled store.

    A localm upgrade ships newer plugin code, but an already-installed copy in
    your data dir keeps shadowing it until refreshed - so you silently run stale
    plugin code (including missing fixes). With no NAME, refreshes every
    installed first-party plugin whose code changed; with a NAME, just that one.
    A running GUI server picks the new code up on its next start.
    """
    from rich.markup import escape

    from localm import cli as _cli
    mgr = _cli._engine_manager()
    try:
        refreshed = mgr.refresh(name)
    except KeyError:
        console.print(f"[red]No such installed first-party plugin: "
                      f"{escape(str(name))}[/red]")
        sys.exit(1)
    if refreshed:
        for n in refreshed:
            console.print(f"[green]Refreshed[/green] plugin [bold]{escape(n)}[/bold]")
    else:
        console.print("[dim]All first-party plugins are up to date.[/dim]")




@plugin.command("status")
def plugin_status():
    """Show engine plugins: installed/available and their enabled state."""
    from rich.markup import escape

    from localm import cli as _cli
    state = _cli._engine_manager().api_state()
    plugins = state.get("plugins", [])
    if not plugins:
        console.print("[dim]No engine plugins discovered.[/dim]")
        return
    console.print("[bold]Installed[/bold]")
    any_installed = False
    for p in plugins:
        if not p.get("installed"):
            continue
        any_installed = True
        mark = "[green]on [/green]" if p.get("active") else "[yellow]off[/yellow]"
        # An installed plugin's own name/description/scope is read from ITS OWN
        # plugin.toml (a third-party plugin can supply anything for description
        # - unlike name, it is not restricted to isidentifier()), so both are
        # escaped defensively - never trust manifest text is display-safe.
        desc = f" - {escape(p['description'])}" if p.get("description") else ""
        console.print(f"  {mark} [bold]{escape(p['name'])}[/bold]{desc}")
    if not any_installed:
        console.print("  [dim](none - only chat is active by default)[/dim]")
    console.print("[bold]Available[/bold] (not installed)")
    any_available = False
    for p in plugins:
        if p.get("installed"):
            continue
        any_available = True
        desc = f" - {escape(p['description'])}" if p.get("description") else ""
        console.print(f"  [dim]+[/dim]  [bold]{escape(p['name'])}[/bold]{desc}")
    if not any_available:
        console.print("  [dim](none)[/dim]")


# ------------------------------------------------------------------ #
#  Plugin-scoped settings (`localm plugin config`)                     #
#                                                                      #
#  The terminal counterpart to the GUI's three plugin settings routes. #
#  See settings_schema.plugin_config_kind for WHY this has two paths:  #
#  the media and tts blocks are static schemas this process can read   #
#  offline, while a host.add_settings() block only exists inside a     #
#  process that has LOADED that plugin - which the CLI deliberately    #
#  never is (set_enabled_state and friends exist to keep it that way,  #
#  and _load would fire the on_first_use hook for what is only a read).#
#  So the generic case asks a RUNNING localm, and when there is none   #
#  it says exactly that rather than reporting an empty field list.     #
# ------------------------------------------------------------------ #

def _plugin_install_state(name):
    """(installed, active) for *name*, without loading a single plugin."""
    from localm import cli as _cli
    for p in _cli._engine_manager().api_state().get("plugins", []):
        if p.get("name") == name:
            return bool(p.get("installed")), bool(p.get("active"))
    return False, False


def _attached_server():
    """``(url, headers, None)`` for a running localm serving this directory, or
    ``(None, None, reason)``. LOCALM_URL targets a different instance, in which
    case there is no registry entry to read an attach token from and only an
    owner key (LOCALM_API_KEY / the persisted one) authenticates - the same
    trade `localm unload` documents."""
    import os

    from .. import instances
    from ..auth import resolve_bearer_headers
    from ..config import home_dir

    override = os.environ.get("LOCALM_URL", "").rstrip("/")
    if override:
        return override, resolve_bearer_headers(None), None
    entry = instances.find_attachable(home_dir(), instances.resolve_root_dir())
    if entry is None:
        return None, None, "no running localm server was found for this directory"
    scheme = entry.get("scheme", "http")
    url = f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{entry.get('port')}"
    return url, resolve_bearer_headers(entry.get("token")), None


def _server_error(resp) -> str:
    """The server's own reason for refusing, or a bare status when it gave
    none. Returns text already safe to interpolate into a markup f-string
    (like ``show_url``) - *detail* is server-supplied and not restricted to a
    safe character class, so callers must not need to remember to escape it."""
    from rich.markup import escape

    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = None
    return escape(str(detail)) if detail else f"HTTP {resp.status_code}"


def _fmt_field_value(f) -> str:
    """One field's resolved value for the listing. Returns text already safe
    to interpolate into a markup f-string (like ``show_url``) - *val* is a
    plugin setting's stored value and can be anything the caller wrote via
    ``localm plugin config`` or the GUI, so callers must not need to remember
    to escape it."""
    from rich.markup import escape

    if "value" not in f:
        # A SECRET field's value never round-trips out of the server in
        # plaintext (plugin_settings_schema_json omits it deliberately), so
        # report whether it is configured, never what it is.
        return "[dim](set)[/dim]" if f.get("is_override") else "[dim](not set)[/dim]"
    val = f.get("value")
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None or val == "":
        return "[dim](none)[/dim]"
    if isinstance(val, list):
        return (", ".join(escape(str(x)) for x in val) if val
                else "[dim](none)[/dim]")
    return escape(str(val))


def _print_fields(name, fields, *, note=None):
    from rich.markup import escape

    if note:
        console.print(note)
    if not fields:
        console.print(f"[dim]{escape(name)} declares no settings.[/dim]")
        return
    width = max(len(f["key"]) for f in fields)
    for f in fields:
        origin = "" if f.get("is_override") else "  [dim](default)[/dim]"
        # f["key"] is a settings key name: fixed for the static media/tts
        # schemas, but server-reported (and third-party-plugin-controllable)
        # for a runtime host.add_settings() block - escape either way.
        console.print(f"  [bold]{escape(f['key']):<{width}}[/bold]  "
                      f"{_fmt_field_value(f)}{origin}")
    console.print(f"[dim]Set one with:  localm plugin config {escape(name)} "
                  f"<key> <value>   (a blank value clears it)[/dim]")


def _runtime_fields(name):
    """One plugin's add_settings() fields from a RUNNING server, or exit with a
    message that says which of the several "nothing to show" states this is."""
    import os

    import requests
    from rich.markup import escape

    from .. import tls

    def _explain_from_local_state():
        """Why this plugin has no fields, read from THIS machine's install set.
        Only sound when the server we would ask is this home's own - see the
        remote note below."""
        installed, active = _plugin_install_state(name)
        if not installed:
            console.print(f"[red]No such plugin:[/red] {escape(name)}")
            console.print("[dim]See[/dim] localm plugin status [dim]for the "
                          "installed ones.[/dim]")
            sys.exit(1)
        if not active:
            console.print(f"[yellow]{escape(name)} is installed but not "
                          f"enabled[/yellow], so it has declared no settings.")
            console.print(f"[dim]Enable it with:[/dim]  "
                          f"localm plugin enable {escape(name)}")
            sys.exit(1)

    url, headers, why = _attached_server()
    # LOCALM_URL points at an instance that is NOT this home, so the local
    # installed/enabled set says nothing about what IT has loaded. Asking the
    # wrong machine's plugin list would produce a confident "No such plugin"
    # about a plugin the target is running perfectly well.
    remote = bool(os.environ.get("LOCALM_URL", "").strip())
    if url is None:
        _explain_from_local_state()
        # AGENTS.md rule 5: this is "could not ask", NOT "there is nothing
        # there". A plugin declares its settings while it LOADS, so only a
        # running localm knows this one's field list - reporting an empty
        # section here would be a different, and false, answer.
        console.print(f"[yellow]{escape(name)}'s settings are declared when "
                      f"the plugin loads[/yellow], so a running localm is "
                      f"needed to list them, and {escape(str(why))}.")
        console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]  "
                      "localm serve <model>[dim], or set LOCALM_URL.[/dim]")
        sys.exit(1)
    try:
        resp = requests.get(f"{url}/v1/plugins/settings", headers=headers,
                            timeout=15, verify=tls.requests_verify(url))
    except requests.RequestException as e:
        # show_url() already returns text safe to interpolate here (it escapes
        # internally, same convention as _server_error/_fmt_field_value below).
        console.print(f"[red]Could not reach the localm server at "
                      f"{show_url(url)}:[/red] {escape(str(e))}")
        sys.exit(1)
    if resp.status_code != 200:
        console.print(f"[red]The server refused the request:[/red] "
                      f"{_server_error(resp)}")
        sys.exit(1)
    for section in resp.json().get("plugins", []):
        if section.get("plugin") == name:
            return url, headers, section.get("fields") or []
    # The server DID answer and has no section for this plugin. Locally that is
    # worth explaining (not installed / not enabled); against a remote instance
    # this machine's install set cannot say, so report only what was observed.
    if not remote:
        _explain_from_local_state()
    else:
        console.print(f"[dim]The localm at {show_url(url)} reports no settings "
                      f"for {escape(name)}.[/dim]")
        sys.exit(1)
    return url, headers, []


@plugin.command("config")
@click.argument("name")
@click.argument("key", required=False)
@click.argument("value", required=False)
def plugin_config(name, key, value):
    """Show or set one plugin's own settings.

    These are the per-plugin blocks the GUI edits under Settings, which
    `localm config` cannot reach: it writes top-level keys, while these live
    under a plugin of their own.

    \b
    Examples:
      localm plugin config image                      # list, with resolved values
      localm plugin config image workdir              # show one
      localm plugin config image workdir /srv/comfy   # set it for image only
      localm plugin config image workdir ""           # clear, back to the shared default
      localm plugin config video float_type fp16
      localm plugin config music use_config_from image
      localm plugin config tts voice af_heart

    image, music, video and tts are readable and settable offline. Any other
    plugin declares its settings as it loads, so listing or setting those needs
    a running localm (this command finds it the way `localm status` does).
    """
    from rich.markup import escape

    from ..config import load_config
    from ..settings_schema import (apply_local_plugin_config,
                                   local_plugin_config_fields,
                                   plugin_config_kind)

    if plugin_config_kind(name) == "runtime":
        _runtime_plugin_config(name, key, value)
        return

    installed, active = _plugin_install_state(name)
    note = None
    if not active:
        # Not a refusal: the settings routes deliberately accept a write for an
        # inactive plugin so it can be configured BEFORE being enabled (see
        # _tts_payload's `active` note). Say so instead of failing.
        state = "installed but not enabled" if installed else "not installed"
        note = (f"[dim]{escape(name)} is {state}; these settings are stored "
                f"either way and apply once it runs.[/dim]")

    if key is None:
        _print_fields(name, local_plugin_config_fields(name, load_config()),
                      note=note)
        return

    if value is None:
        fields = {f["key"]: f for f in local_plugin_config_fields(name, load_config())}
        f = fields.get(key)
        if f is None:
            from ..settings_schema import local_plugin_config_keys
            console.print(f"[red]Unknown setting for {escape(name)}:[/red] "
                          f"{escape(key)}")
            console.print(
                "[dim]Settable keys: "
                f"{', '.join(escape(k) for k in local_plugin_config_keys(name))}"
                f"[/dim]")
            sys.exit(1)
        console.print(_fmt_field_value(f))
        return

    try:
        _, stored = apply_local_plugin_config(name, key, value)
    except ValueError as e:
        raise click.ClickException(str(e))
    _report_set(name, key, stored)


def _report_set(name, key, stored):
    from rich.markup import escape

    if stored is None:
        console.print(f"[green]OK[/green] {escape(name)}.{escape(key)} cleared "
                      f"[dim](back to the default)[/dim]")
    else:
        console.print(f"[green]OK[/green] {escape(name)}.{escape(key)} = "
                      f"{escape(str(stored))}")


def _runtime_plugin_config(name, key, value):
    """`plugin config` for a host.add_settings() block, over a running server."""
    import requests
    from rich.markup import escape

    from .. import tls
    url, headers, fields = _runtime_fields(name)

    if key is None:
        _print_fields(name, fields)
        return
    # These key names are server-reported, from a third-party plugin's own
    # host.add_settings() call - not restricted to a safe character class.
    known = [f["key"] for f in fields]
    if key not in known:
        console.print(f"[red]Unknown setting for {escape(name)}:[/red] "
                      f"{escape(key)}")
        if known:
            console.print("[dim]Settable keys: "
                          f"{', '.join(escape(k) for k in known)}[/dim]")
        sys.exit(1)
    if value is None:
        console.print(_fmt_field_value(
            next(f for f in fields if f["key"] == key)))
        return

    try:
        resp = requests.post(f"{url}/v1/plugins/{name}/settings", json={key: value},
                             headers=headers, timeout=30,
                             verify=tls.requests_verify(url))
    except requests.RequestException as e:
        console.print(f"[red]Could not reach the localm server at "
                      f"{show_url(url)}:[/red] {escape(str(e))}")
        sys.exit(1)
    if resp.status_code != 200:
        console.print(f"[red]The server refused the change:[/red] "
                      f"{_server_error(resp)}")
        sys.exit(1)
    saved = {f["key"]: f for f in resp.json().get("fields", [])}.get(key, {})
    if not saved.get("is_override"):
        _report_set(name, key, None)
    elif "value" not in saved:
        # A SECRET field's value is deliberately never echoed back
        # (plugin_settings_schema_json omits it), so confirm the write without
        # inventing a value - and, the reason this branch exists, without
        # reading that absence as "cleared", which is what a plain
        # saved.get("value") did: it reported a successful SET as a CLEAR.
        console.print(f"[green]OK[/green] {escape(name)}.{escape(key)} set "
                      f"[dim](not shown)[/dim]")
    else:
        _report_set(name, key, saved.get("value"))
