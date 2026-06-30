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
    from ..plugins.loader import discover_errors, discover_plugins, plugins_dir

    manifests = discover_plugins()
    errors = discover_errors()
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




@plugin.command("install")
@click.argument("target")
@click.option("--force", is_flag=True,
              help="When installing from a directory, overwrite an existing install.")
def plugin_install_engine(target, force):
    """Install a plugin and enable it.

    TARGET is either a first-party plugin NAME from the bundled store
    (e.g. ``localm plugin install coder``) or a path to a DIRECTORY containing a
    plugin.toml (a third-party plugin). For plugins with extra dependencies, also
    run the matching pip extra, e.g. pip install "localm[coder]".
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
def plugin_setup(plugins_csv, install_all, install_defaults):
    """Choose which first-party plugins to install.

    Out of the box only chat is active; this turns on the features you want
    (coder, image/music/video, rag, web, voice, tts, mcp). Run by the installer
    after dependencies are in place, and any time afterwards. Some plugins also
    need a pip extra (e.g. pip install "localm[coder]").
    """
    from ..plugins import catalog
    from localm import cli as _cli
    mgr = _cli._engine_manager()
    installed = _installed_plugin_names(mgr)
    available = [e for e in catalog.CATALOG if not e.preinstalled]

    if install_all:
        chosen = [e.name for e in available]
    elif plugins_csv is not None:
        chosen = _parse_plugin_selection(plugins_csv, available)
    elif install_defaults:
        chosen = list(_SETUP_DEFAULTS)
    elif not sys.stdin.isatty():
        console.print(
            "[dim]Non-interactive shell - skipping plugin selection. Run "
            "[bold]localm plugin setup[/bold] to choose, or pass "
            "--plugins/--all/--defaults.[/dim]")
        return
    else:
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
def plugin_enable(name):
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
