# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ._core import console, main


# ------------------------------------------------------------------ #
#  Plugin management (external plugins in ~/.localm/plugins/)          #
# ------------------------------------------------------------------ #

@main.group()
def plugin() -> None:
    """Manage external plugins (installed in ~/.localm/plugins/)."""




@plugin.command("list")
def plugin_list():
    """List installed external plugins."""
    from ..plugins.loader import (discover_errors, discover_plugins,
                                  discover_warnings, plugins_dir)

    manifests = discover_plugins()
    errors = discover_errors()
    warnings = discover_warnings()
    if not manifests and not errors:
        console.print(f"[dim]No external plugins installed ({plugins_dir()})[/dim]")
        return
    for m in manifests:
        desc = f" - {m.description}" if m.description else ""
        console.print(f"  [bold]{m.name}[/bold] v{m.version}{desc}")
        if m.tool_exports:
            console.print(f"    [dim]tools: {', '.join(m.tool_exports)}[/dim]")
    for err in errors:
        console.print(f"  [yellow]invalid:[/yellow] {err}")
    for warn in warnings:
        console.print(f"  [yellow]warning:[/yellow] {warn}")




@plugin.command("remove")
@click.argument("name")
def plugin_remove(name):
    """Remove an installed plugin by name."""
    from ..plugins.loader import PluginError, remove_plugin

    try:
        existed = remove_plugin(name)
    except PluginError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if existed:
        console.print(f"[green]Removed[/green] plugin [bold]{name}[/bold]")
    else:
        console.print(f"[yellow]Plugin {name!r} is not installed[/yellow]")




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
    missing = mgr.missing_requires(name)
    if missing:
        cmds = "  ".join(f"localm plugin install {m}" for m in missing)
        console.print(
            f"[yellow]Note:[/yellow] {name!r} declares it needs "
            f"{', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} "
            f"not installed. Install with:  {cmds}")


# ---- pip-extra auto-install (host-side; the CLI always runs on the host) ---- #

def _auto_deps_default() -> bool:
    """The configured default for auto-installing plugin pip extras."""
    from localm.config import load_config
    return bool(load_config().get("auto_install_plugin_deps", True))


def _set_auto_deps(value: bool) -> None:
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["auto_install_plugin_deps"] = bool(value)
    save_config(cfg)


def _console_progress(line: str) -> None:
    # pip output can contain "[" (e.g. localm[voice]); disable rich markup so it
    # is printed verbatim rather than parsed as a style tag.
    console.print(line, markup=False, highlight=False)


def _install_deps(mgr, name) -> bool:
    """Install *name*'s declared pip extras on this host. Returns True on success
    (including the no-op case where the plugin declares none / all are present).
    Surfaces the real installer error on failure (never a hollow success)."""
    if not mgr.plugin_missing_deps(name):
        return True                     # nothing declared, or already satisfied
    console.print(f"[dim]Installing dependencies for {name}...[/dim]")
    res = mgr.install_plugin_deps(name, on_progress=_console_progress)
    if res.ok:
        if res.installed:
            console.print(f"[green]Installed dependencies[/green] for "
                          f"[bold]{name}[/bold]: {', '.join(res.installed)}")
        return True
    console.print(f"[red]Dependency install failed for {name}:[/red] {res.error}")
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
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    src = Path(target)
    if src.is_dir() and (src / "plugin.toml").is_file():
        try:
            spec = mgr.set_installed_from_dir(src, force=force)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        console.print(f"[green]Installed[/green] plugin [bold]{spec.name}[/bold] "
                      f"v{spec.version}")
        _warn_missing_requires(mgr, spec.name)
        # A third-party plugin's extras are its own (not localm's); we do not
        # auto-resolve those here. Point the user at its own instructions.
        if spec.requires_extras:
            console.print(f"[dim]{spec.name} declares extra dependencies "
                          f"({', '.join(spec.requires_extras)}); install per its "
                          f"own instructions.[/dim]")
        return
    try:
        mgr.set_installed_state(target, True)
    except KeyError:
        console.print(f"[red]No such plugin: {target}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Installed[/green] plugin [bold]{target}[/bold]")
    _warn_missing_requires(mgr, target)
    if _resolve_with_deps(with_deps):
        _install_deps(mgr, target)
    elif mgr.plugin_missing_deps(target):
        console.print('[dim]Needs pip extras; install them with:  '
                      f'localm plugin install-deps {target}[/dim]')




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
                console.print(f"[yellow]Ignoring out-of-range selection: {t}[/yellow]")
        else:
            console.print(f"[yellow]Ignoring unknown selection: {t}[/yellow]")
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
                f"[red]No known plugins in --plugins {plugins_csv!r}.[/red] "
                f"Choose from: {', '.join(e.name for e in available)}.")
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
            extra = f" [dim](pip extra: localm[{e.extra}])[/dim]" if e.extra else ""
            console.print(f"  [bold]{i:>2}.[/bold] {e.name:<7} {e.description}{mark}{extra}")
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
            console.print(f"[dim]{name} already installed[/dim]")
            continue
        try:
            mgr.set_installed_state(name, True)
        except (KeyError, ValueError) as e:
            console.print(f"[yellow]Skipped {name}: {e}[/yellow]")
            continue
        console.print(f"[green]Installed[/green] [bold]{name}[/bold]")
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
                      f"{', '.join(pending)}. Install later with:  "
                      "localm plugin install-deps --all[/dim]")




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
    from ..plugins import catalog
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    if name and do_all:
        console.print("[red]Give a NAME or --all, not both.[/red]")
        sys.exit(1)
    if name:
        known = set(catalog.names()) | _installed_plugin_names(mgr)
        if name not in known:
            console.print(f"[red]No such plugin: {name}[/red]")
            sys.exit(1)
        if not mgr.plugin_missing_deps(name):
            console.print(f"[green]{name} has its dependencies "
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
            console.print(f"  [bold]{n}[/bold]: {', '.join(reqs)}")
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
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    try:
        was = mgr.uninstall(name, delete_data=delete_data)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if was:
        console.print(f"[yellow]Uninstalled[/yellow] plugin [bold]{name}[/bold]")
    else:
        console.print(f"[dim]Plugin {name!r} was not installed.[/dim]")




@plugin.command("enable")
@click.argument("name")
@click.option("--with-deps/--no-deps", "with_deps", default=None,
              help="Also install the plugin's pip extras on this machine "
                   "(default: the auto_install_plugin_deps setting).")
def plugin_enable(name, with_deps):
    """Enable an installed engine plugin (must be installed first)."""
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    try:
        mgr.set_enabled_state(name, True)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]Enabled[/green] plugin [bold]{name}[/bold]")
    _warn_missing_requires(mgr, name)
    if _resolve_with_deps(with_deps):
        _install_deps(mgr, name)
    elif mgr.plugin_missing_deps(name):
        console.print('[dim]Needs pip extras; install them with:  '
                      f'localm plugin install-deps {name}[/dim]')




@plugin.command("disable")
@click.argument("name")
def plugin_disable(name):
    """Disable an installed engine plugin (keeps it installed)."""
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    try:
        mgr.set_enabled_state(name, False)
    except KeyError:
        console.print(f"[red]No such plugin: {name}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[yellow]Disabled[/yellow] plugin [bold]{name}[/bold]")




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
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    try:
        refreshed = mgr.refresh(name)
    except KeyError:
        console.print(f"[red]No such installed first-party plugin: {name}[/red]")
        sys.exit(1)
    if refreshed:
        for n in refreshed:
            console.print(f"[green]Refreshed[/green] plugin [bold]{n}[/bold]")
    else:
        console.print("[dim]All first-party plugins are up to date.[/dim]")




@plugin.command("status")
def plugin_status():
    """Show engine plugins: installed/available and their enabled state."""
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
        desc = f" - {p['description']}" if p.get("description") else ""
        console.print(f"  {mark} [bold]{p['name']}[/bold]{desc}")
    if not any_installed:
        console.print("  [dim](none - only chat is active by default)[/dim]")
    console.print("[bold]Available[/bold] (not installed)")
    any_available = False
    for p in plugins:
        if p.get("installed"):
            continue
        any_available = True
        desc = f" - {p['description']}" if p.get("description") else ""
        console.print(f"  [dim]+[/dim]  [bold]{p['name']}[/bold]{desc}")
    if not any_available:
        console.print("  [dim](none)[/dim]")
