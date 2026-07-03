# SPDX-License-Identifier: AGPL-3.0-or-later
import sys

import click

from ._core import console, main


# ------------------------------------------------------------------ #
#  Plugin: coder (optional extra)                                      #
# ------------------------------------------------------------------ #

# Register ``localm coder`` when the coder plugin is installed.
# The plugin is gated behind ``pip install "localm[coder]"`` so the import
# is wrapped in a try/except - the base localm install keeps working fine
# if the extra was never requested.
# MCP server plugin - expose localm to MCP clients (Claude Desktop, etc.)
try:
    from ..plugins.mcpserver.cli import main as _mcp_main
    main.add_command(_mcp_main, name="mcp")
except ImportError:
    pass



# GUI plugin - browser interface for chat and the coder agent
try:
    from ..plugins.gui.cli import main as _gui_main
    main.add_command(_gui_main, name="gui")
except ImportError:
    pass



# Jobs plugin - scheduled recurring tasks (localm job add/list/run/...)
try:
    from ..plugins.builtin.jobs.cli import main as _jobs_main
    main.add_command(_jobs_main, name="job")
except ImportError:
    pass



try:
    from ..plugins.coder.cli import main as _coder_main
    main.add_command(_coder_main, name="coder")
except ImportError:
    @main.command("coder", context_settings={"ignore_unknown_options": True})
    def _coder_stub(**_):
        """Offline AI coding agent (run: pip install "localm[coder]" to enable)."""
        console.print(
            '[yellow]The coder plugin could not be loaded.[/yellow]\n'
            'Install its dependencies with:  [bold]pip install "localm[coder]"[/bold]\n'
            '  or (editable):  [bold]pip install -e ".[coder]"[/bold]\n'
            'then activate it with:  [bold]localm plugin install coder[/bold]'
        )



# Provision the native llama.cpp binaries into localm's own venv (self-contained).
from ..setup_llama import main as _setup_llama_main


main.add_command(_setup_llama_main, name="setup-llama")


@main.command("setup-embeddings")
@click.option("--model", "model", default=None,
              help="Embedding model to install (a known key, a registered model "
                   "name, or a path to a GGUF). Persisted as the embedding_model "
                   "config. Default: the current embedding_model config.")
def setup_embeddings(model):
    """Install the on-device embedding model for semantic search (memory + RAG).

    localm's chat models make poor embeddings, so semantic retrieval uses a small
    dedicated model (bge-small, ~25 MB). This downloads it into
    <home>/models/embeddings/ so memory and RAG retrieval become semantic instead
    of lexical. Respects net_mode=off (a hard kill switch)."""
    from ..config import load_config, update_config
    from ..inference.embedder import (DEFAULT_EMBEDDING_MODEL,
                                      KNOWN_EMBEDDING_MODELS,
                                      resolve_embedding_model_path)
    if model:
        update_config(lambda c: c.update({"embedding_model": model}))
    name = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    console.print(f"Installing embedding model: [bold cyan]{name}[/bold cyan]")
    path = resolve_embedding_model_path(allow_download=True)
    if not path:
        console.print(
            "[red]Could not install the embedding model.[/red] It must be a known "
            f"key {tuple(KNOWN_EMBEDDING_MODELS)}, a registered model, or a GGUF "
            "path, and network must be enabled (net_mode is not 'off').")
        sys.exit(1)
    console.print(f"[green]Embedding model ready:[/green] {path}\n"
                  "Memory and RAG will now use semantic search.")


@main.command("bug-report")
@click.option("-m", "--message", default="", help="One-line summary of the problem.")
def bug_report_cmd(message: str) -> None:
    """Generate an editable bug report and offer to send it to the maintainer.

    Collects a useful, safe diagnostic snapshot (OS, GPU, driver, backend, the
    loaded model, an allowlisted config subset, key dependency versions, and the
    in-memory recent-activity log - never your API key, config secrets, or chat
    content), saves an editable markdown file, and offers to email it, open a
    GitHub issue, or hand it off yourself."""
    from localm import bugreport
    bugreport.report_failure(
        summary=message or "user-reported issue",
        context={"operation": "bug-report"},
        as_failure=False,
        interactive=bool(getattr(sys.stdin, "isatty", lambda: False)()))




@main.command("update")
@click.option("--check", "check_only", is_flag=True,
              help="Only report whether an update is available; do not apply.")
@click.option("-y", "--yes", is_flag=True, help="Apply without the confirmation prompt.")
@click.option("--rollback", "do_rollback", is_flag=True,
              help="Restore the previous build from the last update backup.")
def update_cmd(check_only: bool, yes: bool, do_rollback: bool) -> None:
    """Check for a newer localm build and apply it. You always initiate it - localm
    never updates itself automatically.

    Most updates apply with just a restart (the install is editable); localm tells you
    what an update entails before applying. Requires the update channel to be
    configured (the bug-report proxy with an update token); see tools/bugreport-proxy/."""
    from localm import updater
    from localm.bugreport import LocalmError

    if do_rollback:
        try:
            updater.rollback_last()
        except LocalmError as e:
            console.print(f"[yellow]{e.summary}[/yellow] ({e.reason}).")
            return
        console.print("[green]Rolled back.[/green] Restart localm to load the previous build.")
        return

    if not updater.available():
        console.print("[yellow]The updater is not configured.[/yellow] Ask the maintainer "
                      "for an updated build, or set the update endpoint in config.")
        return
    try:
        info = updater.check()
    except LocalmError as e:
        console.print(f"[yellow]Could not check for updates:[/yellow] {e.summary} ({e.reason}).")
        return

    cur, latest = info["current"], info["latest"]
    if not latest:
        console.print(f"[dim]No releases published yet. You are on {cur}.[/dim]")
        return
    if not info["newer"]:
        console.print(f"[green]localm is up to date[/green] (running {cur}; latest {latest}).")
        return

    console.print(f"[bold]Update available:[/bold] {latest}  [dim](you have {cur})[/dim]")
    if info.get("notes"):
        console.print(f"[dim]{str(info['notes'])[:500]}[/dim]")
    if check_only:
        console.print("[dim]Run `localm update` to apply it.[/dim]")
        return

    asset = info.get("asset") or {}
    if not asset.get("id"):
        console.print("[yellow]This release has no downloadable build attached.[/yellow] "
                      "Ask the maintainer for the build.")
        return
    if not yes and not click.confirm(f"  Download and apply {latest}?", default=True):
        console.print("[dim]Not applied.[/dim]")
        return

    console.print(f"[dim]Downloading and applying {latest} ...[/dim]")
    try:
        res = updater.apply(asset["id"], signature=info.get("signature"))
    except LocalmError as e:
        # apply() already rolled back; surface honestly, never a false success.
        console.print(f"[red]Update failed:[/red] {e.summary} ({e.reason}).")
        return
    console.print(f"[green]Updated to {res['version']}[/green] "
                  f"({updater.class_summary(res['klass'])}).")
    if res["klass"] == "setup":
        console.print("[yellow]This update needs setup.bat re-run (a Python/venv change) "
                      "before it will start.[/yellow]")
    else:
        console.print("[bold]Restart localm to load the new version.[/bold]  "
                      "[dim](undo: localm update --rollback)[/dim]")




@main.command("issues")
@click.argument("number", required=False, type=int)
@click.option("--state", type=click.Choice(["open", "closed", "all"]), default="all",
              show_default=True, help="Which issues to list.")
def issues_cmd(number, state) -> None:
    """List the project's issues (open/closed), or show one by NUMBER.

    Read-only, through the bug-report proxy - no GitHub account needed. Lets a tester
    see whether a bug they filed is acknowledged or fixed. Requires the proxy to be
    configured."""
    from localm import issue_tracker
    from localm.bugreport import LocalmError

    if not issue_tracker.available():
        console.print("[yellow]The issues tracker is not configured.[/yellow]")
        return
    try:
        if number is not None:
            it = issue_tracker.get_issue(number)
            if not it:
                console.print(f"[yellow]Issue #{number} not found.[/yellow]")
                return
            st = it.get("state", "?")
            console.print(f"[bold]#{it.get('number')}[/bold] {st}: {it.get('title', '')}")
            if it.get("html_url"):
                console.print(f"[dim]{it['html_url']}[/dim]")
            return
        issues = issue_tracker.list_issues(state)
    except LocalmError as e:
        console.print(f"[yellow]Could not load issues:[/yellow] {e.summary} ({e.reason}).")
        return

    if not issues:
        console.print("[dim]No issues.[/dim]")
        return
    for it in issues:
        st = it.get("state", "?")
        color = "green" if st == "closed" else "yellow"
        # No literal square brackets around dynamic text - rich would parse them as
        # markup tags and drop them.
        console.print(f"  [{color}]#{it.get('number')} {st}[/{color}]  {it.get('title', '')}")
    console.print(f"[dim]{len(issues)} issue(s). Detail: localm issues <number>.[/dim]")
