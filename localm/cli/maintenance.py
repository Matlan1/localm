# SPDX-License-Identifier: AGPL-3.0-or-later
import logging
import sys

import click

from localm.debuglog import defer_log

from ._core import console, main


# ------------------------------------------------------------------ #
#  Plugin CLI wiring (first-party plugins that ship a Click group)      #
# ------------------------------------------------------------------ #

def _wire_plugin_cli_entries() -> None:
    """Wire each first-party plugin's CLI Click group from its manifest's
    ``cli`` entry point (PluginManager.cli_entries()), so shipping a new
    first-party plugin with a CLI surface needs a manifest line, not a new
    block here. A plugin's optional pip extras gate it via ImportError. It is
    NOT gated on `plugin install`/enabled state, so e.g. `localm coder` stays
    reachable regardless of that toggle.

    The Click command/group's OWN declared name (e.g. jobs/cli.py's
    ``@click.group(name="job")``) is used as-is, not the plugin's catalog
    name: jobs' catalog name is "jobs" but its CLI verb is "job"."""
    import importlib

    from ..plugins.engine import PluginManager
    mgr = PluginManager(None)
    wired = set()
    for name, entry in mgr.cli_entries():
        mod_name, _, attr = entry.partition(":")
        attr = attr or "main"
        try:
            mod = importlib.import_module(mod_name)
            main.add_command(getattr(mod, attr))
            wired.add(name)
        except ImportError as e:
            # An ImportError here is usually an optional pip extra that is not
            # installed (the verb is simply unavailable), but it also catches a
            # broken first-party plugin module. Recorded rather than failing
            # startup, so the two stay distinguishable.
            #
            # defer_log, NOT logger.debug: this runs at module-import time,
            # before Click invokes main() to install any handler, so a direct
            # logger.debug() is dropped at the call.
            defer_log(logging.DEBUG, "plugin CLI %r not wired (import failed): %s",
                      name, e)
    if "coder" not in wired:
        @main.command("coder", context_settings={"ignore_unknown_options": True})
        def _coder_stub(**_):
            """Offline AI coding agent (run: pip install "localm[coder]" to enable)."""
            console.print(
                '[yellow]The coder plugin could not be loaded.[/yellow]\n'
                'Install its dependencies with:  [bold]pip install "localm[coder]"[/bold]\n'
                '  or (editable):  [bold]pip install -e ".[coder]"[/bold]\n'
                'then activate it with:  [bold]localm plugin install coder[/bold]'
            )


_wire_plugin_cli_entries()


def _wire_gui_cli() -> None:
    """Wire the GUI (the WebUI) directly rather than through cli_entries():
    it is core kernel surface, not a PluginManager-tracked plugin (no
    plugin.toml - it is not an installable feature)."""
    try:
        from ..plugins.gui.cli import main as _gui_main
        main.add_command(_gui_main, name="gui")
    except ImportError as e:
        # defer_log, NOT logger.debug: this runs at module-import time, before
        # Click invokes main() to install any handler.
        defer_log(logging.DEBUG, "gui CLI not wired (import failed): %s", e)

        @main.command("gui", context_settings={"ignore_unknown_options": True})
        def _gui_stub(**_):
            """The web UI (currently unavailable - see the error below)."""
            console.print(
                '[yellow]The GUI could not be loaded.[/yellow]\n'
                'It is core to localm, not an optional extra, so this usually '
                'means a broken or partial install. Run with '
                '[bold]LOCALM_DEBUG=1[/bold] to see the import error, then '
                'try:  [bold]pip install -e .[/bold]'
            )


_wire_gui_cli()



# Provision the native llama.cpp binaries into localm's own venv (self-contained).
from ..setup_llama import main as _setup_llama_main


main.add_command(_setup_llama_main, name="setup-llama")


@main.command("setup-embeddings")
@click.option("--model", "model", default=None,
              help="Embedding model to install (a known key, a registered model "
                   "name, or a path to a GGUF). Persisted as the embedding_model "
                   "config. Default: the current embedding_model config.")
@click.option("-y", "--yes", "yes", is_flag=True,
              help="Skip the pre-switch collection-impact confirmation.")
def setup_embeddings(model, yes=False):
    """Install the on-device embedding model for semantic search (memory + RAG).

    Semantic retrieval uses a small dedicated model (bge-small, ~25 MB) rather
    than the chat model: the bundled GGUF runtime cannot embed a chat model (the
    ctypes binding exposes no create_embedding). This downloads it into
    <home>/models/embeddings/ so memory and RAG retrieval become semantic instead
    of lexical. Respects net_mode=off (a hard kill switch). A freshly downloaded
    known model is also synced into the Model Manager registry (type "embedding")
    so it shows up in `localm list` / the GUI Models page; this sync is best-effort
    and never touches an already-registered or user-pointed model."""
    from pathlib import Path

    from rich.markup import escape

    from ..config import load_config, update_config
    from ..inference.embedder import (DEFAULT_EMBEDDING_MODEL,
                                      KNOWN_EMBEDDING_MODELS,
                                      resolve_embedding_model_path)
    if model:
        current = str(load_config().get("embedding_model") or "")
        if model != current:
            # The third writer of embedding_model, alongside the RAG picker
            # (POST /api/rag/embedding) and PATCH /v1/config. Like those two, it
            # warns what the switch is about to invalidate BEFORE it happens.
            from ..rag import collection_provenance_note, collection_provenance_report
            affected = collection_provenance_report()
            if affected:
                console.print(
                    f"[yellow]{escape(collection_provenance_note(model, affected))}[/yellow]")
                for c in affected:
                    built = (f" (built with {escape(c['built_with'])})"
                             if c.get("built_with") else "")
                    chunks = f" - {c['n_chunks']} chunks" if c.get("n_chunks") is not None else ""
                    console.print(f"  - {escape(c['name'])}{built}{chunks}")
                if not yes:
                    # NOT abort=True: a script or non-interactive run with
                    # nobody to answer PROCEEDS with the switch. A collection's
                    # chunk text and existing vectors stay on disk either way,
                    # falling back to lexical search until re-embedded.
                    try:
                        proceed = click.confirm("Continue with the switch?")
                    except click.Abort:
                        console.print(
                            "[yellow]Not an interactive terminal, so nothing can "
                            "answer that. Proceeding with the switch - pass --yes "
                            "to silence this prompt next time.[/yellow]")
                        proceed = True
                    if not proceed:
                        raise click.Abort()
        update_config(lambda c: c.update({"embedding_model": model}))
    name = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    console.print(f"Installing embedding model: [bold cyan]{escape(name)}[/bold cyan]")
    path = resolve_embedding_model_path(allow_download=True)
    if not path:
        console.print(
            "[red]Could not install the embedding model.[/red] It must be a known "
            f"key {tuple(KNOWN_EMBEDDING_MODELS)}, a registered model, or a GGUF "
            "path, and network must be enabled (net_mode is not 'off').")
        sys.exit(1)

    synced_note = ""
    try:
        p = Path(path).resolve()
        # Only register a KNOWN-key download (lives directly under the dedicated
        # embeddings dir) - never a user-pointed external GGUF or an already
        # registered model, which keep whatever registration/type they already
        # have (never silently override an existing choice).
        from ..inference.embedder import _embeddings_dir
        if p.parent == _embeddings_dir().resolve():
            from ..config import load_registry
            from ..model_manager import find_aliases_by_path, _register, _sanitize_name
            if not find_aliases_by_path(p, load_registry()):
                reg_name = _sanitize_name(f"embedding-{name}")
                _register(reg_name, p, source="setup-embeddings", model_type="embedding")
                synced_note = (f"\nRegistered as [bold]{escape(reg_name)}[/bold] "
                               "(type 'embedding') - visible in `localm list` / the GUI.")
    except Exception as e:
        # Best-effort visibility sync only - the embedding model itself is
        # already installed and fully functional regardless of whether this
        # optional Model-Manager registration succeeds. Surfaced at debug
        # level rather than silenced.
        from ..debuglog import logger as _logger
        _logger.debug("setup-embeddings: could not sync into the model registry (%s)", e)

    # Memory records written before an embedder existed carry NO vector, and
    # nothing else fills them in: backfill_vectors' only other caller is the
    # consolidation pass, which is optional and may never run. Below
    # VEC_COVERAGE the semantic gate is unusable and recall falls back to
    # promoting profile facts by IMPORTANCE, so backfill them here and report
    # what actually happened in the ready message below.
    mem_note = ""
    try:
        from ..inference.embedder import embed_texts
        from ..memory.backfill import backfill_all, vectorless_scan
        from ..memory.store import _memory_root as _mem_root

        _root = _mem_root(None)
        pending, unreadable_ns = vectorless_scan(_root)
        if pending or unreadable_ns:
            if pending:
                console.print(f"  Embedding {pending} stored memory item(s) ...")
            res = backfill_all(_root, embed_texts)
            if res["remaining"] or res["unreadable"]:
                bits = []
                if res["embedded"]:
                    bits.append(f"embedded {res['embedded']} item(s)")
                if res["remaining"]:
                    bits.append(f"{res['remaining']} could not be embedded")
                if res["unreadable"]:
                    bits.append(
                        f"{res['unreadable']} namespace(s) could not be read")
                mem_note = ("\nMemory: " + "; ".join(bits) +
                            " - stay lexical, re-run to retry.")
            elif res["embedded"]:
                mem_note = (f"\nMemory: {res['embedded']} stored item(s) embedded, "
                            "so recall is semantic for those too.")
    except Exception as e:
        # Never fail the install over the backfill, and never claim it happened
        # either.
        from ..debuglog import logger as _logger
        _logger.debug("setup-embeddings: memory vector backfill skipped (%s)", e)
        mem_note = ("\nMemory: stored items could not be embedded just now, so "
                    "recall stays lexical for them until this succeeds.")

    console.print(
        f"[green]Embedding model ready:[/green] {escape(str(path))}{synced_note}{mem_note}\n"
        "New memory items are embedded as they are written. Existing RAG "
        "collections stay lexical (BM25) until re-embedded: run `localm rag reembed "
        "<name>` for each (works from the chunk text already stored, no original "
        "files needed), or click 're-embed' on the Knowledge page.")


@main.command("make-launcher")
@click.option("--force", is_flag=True,
              help="Rebuild the launcher even if it already exists (use after a "
                   "Python upgrade to refresh the copied interpreter).")
@click.option("--quiet", is_flag=True,
              help="Only report problems. For callers that print their own next "
                   "steps (setup), so this step does not hand out a competing "
                   "way to start localm.")
def make_launcher_cmd(force: bool, quiet: bool) -> None:
    """Build the native app launcher so the server runs as LocaLM, not python.

    Windows: creates <venv>\\localm-app\\LocaLM.exe (a branded copy of the venv
    interpreter, with the LocaLM icon) - launch it as `LocaLM.exe -m localm gui`
    and Task Manager shows LocaLM.exe. Linux: creates <venv>/bin/LocaLM and a
    LocaLM.desktop launcher. It is a branded copy of the interpreter, not a
    compiled binary; it stays inside this clone's venv. Re-runnable; setup runs it
    for you."""
    from localm import applaunch
    from rich.markup import escape
    res = applaunch.make_launcher(force=force)
    for note in res.notes:
        console.print(f"  [dim]-[/dim] {escape(note)}")
    if not res.ok:
        console.print("[yellow]Could not build the native launcher.[/yellow] "
                      "`localm gui` still works.")
        sys.exit(1)
    if quiet:
        return
    if res.path:
        console.print(f"[green]App executable ready:[/green] {escape(str(res.path))}")
    if res.desktop_file:
        console.print(f"[dim]Desktop entry:[/dim] {escape(str(res.desktop_file))}")
    if sys.platform == "win32" and res.path:
        console.print(f'[dim]Launch it:[/dim] "{escape(str(res.path))}" -m localm gui')


@main.command("bug-report")
@click.option("-m", "--message", default="", help="What were you doing?")
@click.option("-e", "--expected", default="",
              help="What did you expect to happen? (optional)")
@click.option("-w", "--happened", default="", help="What actually happened?")
@click.option("--no-log", "no_log", is_flag=True,
              help="Do not attach this run's debug log tail even if one exists. "
                   "Attached by default when present (debug mode only - a normal "
                   "run has none to attach - and always scrubbed, never chat "
                   "content, #961).")
@click.option("-s", "--send", is_flag=True,
              help="Send it to the maintainer immediately via the hosted proxy - "
                   "no GitHub account needed. Works from any shell, interactive or "
                   "not (a script, a non-tty launcher, an SSH session).")
def bug_report_cmd(message: str, expected: str, happened: str,
                   no_log: bool, send: bool) -> None:
    """Generate an editable bug report and offer to send it to the maintainer.

    Three DISTINCT questions - what you were doing, what you expected, what
    actually happened - the same three the GUI's "Report a bug" form asks, built
    from the same template. Answer inline with -m/-e/-w for a scripted or
    non-interactive run; run with no flags in a real terminal and you are
    prompted for each (Enter to skip). At least one of "what were you doing" /
    "what actually happened" is required.

    Collects a useful, safe diagnostic snapshot (OS, GPU, driver, backend, the
    loaded model, an allowlisted config subset, key dependency versions, and the
    in-memory recent-activity log - never your API key, config secrets, or chat
    content), saves an editable markdown file, and offers to send it via the
    account-less hosted channel, email it, or hand it off yourself. Pass --send
    to skip the menu and send it immediately. No GitHub account is ever
    needed."""
    from localm import bugreport
    from rich.markup import escape
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    description, what_expected, what_happened = message, expected, happened
    if interactive:
        console.print("[bold]Filing a bug report.[/bold] Press Enter to skip a question.")
        if not description:
            description = click.prompt("What were you doing?",
                                       default="", show_default=False)
        if not what_expected:
            what_expected = click.prompt("What did you expect to happen? (optional)",
                                         default="", show_default=False)
        if not what_happened:
            what_happened = click.prompt("What actually happened?",
                                         default="", show_default=False)
    description = description.strip()
    what_expected = what_expected.strip()
    what_happened = what_happened.strip()
    if not description and not what_happened:
        console.print("[yellow]Describe the problem first (-m/-w, or answer the "
                      "prompt) - an empty report helps no one.[/yellow]")
        return

    summary = bugreport.report_title("", what_happened, description)
    console.print(f"[bold]Filing a bug report:[/bold] {escape(summary)}")
    # The reporter's server may have hung in a DIFFERENT process (this CLI is not
    # it), so its captured freeze trace is found via the live instance registry,
    # not this process's pid.
    hang = bugreport.live_server_hang_trace()
    path = bugreport.save_user_report(
        description, what_i_expected=what_expected, what_happened=what_happened,
        include_log=not no_log, extra_hang_trace=hang)
    if path is not None:
        console.print(
            f"[dim]A bug report was saved (edit it before sending):[/dim] {escape(str(path))}")
    else:
        console.print("[yellow]Could not save a report file; you can still copy the "
                      "details above.[/yellow]")
    text = path.read_text(encoding="utf-8") if path is not None else ""
    bugreport.offer_to_send(summary, path, text, interactive=interactive,
                            auto_send=send)




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
    from rich.markup import escape

    if do_rollback:
        try:
            updater.rollback_last()
        except LocalmError as e:
            console.print(f"[yellow]{escape(e.summary)}[/yellow] ({escape(e.reason)}).")
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
        console.print(
            f"[yellow]Could not check for updates:[/yellow] {escape(e.summary)} "
            f"({escape(e.reason)}).")
        return

    cur, latest = info["current"], info["latest"]
    if not latest:
        console.print(f"[dim]No releases published yet. You are on {escape(cur)}.[/dim]")
        return
    if not info["newer"]:
        if not info.get("comparable", True):
            console.print(
                f"[yellow]Could not tell whether {escape(latest)} is newer than your "
                f"version {escape(cur)}[/yellow] (unrecognized version format) - "
                "check the release notes yourself before assuming you are up to date.")
        else:
            console.print(f"[green]localm is up to date[/green] "
                          f"(running {escape(cur)}; latest {escape(latest)}).")
        return

    console.print(f"[bold]Update available:[/bold] {escape(latest)}  "
                  f"[dim](you have {escape(cur)})[/dim]")
    if info.get("notes"):
        console.print(f"[dim]{escape(str(info['notes'])[:500])}[/dim]")
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

    console.print(f"[dim]Downloading and applying {escape(latest)} ...[/dim]")
    try:
        res = updater.apply(asset["id"], signature=info.get("signature"))
    except LocalmError as e:
        # apply() has already rolled back by this point.
        console.print(f"[red]Update failed:[/red] {escape(e.summary)} ({escape(e.reason)}).")
        return
    console.print(f"[green]Updated to {escape(res['version'])}[/green] "
                  f"({updater.class_summary(res['klass'])}).")
    if res["klass"] == "setup":
        console.print("[yellow]This update needs setup.bat re-run (a Python/venv change) "
                      "before it will start.[/yellow]")
    else:
        console.print("[bold]Restart localm to load the new version.[/bold]  "
                      "[dim](undo: localm update --rollback)[/dim]")
        console.print("[dim]If the new version will not start, run the rollback script "
                      "in the install folder (rollback.bat / rollback.sh) to restore "
                      "this one.[/dim]")




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
    from rich.markup import escape

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
            # number/state/title come straight from the GitHub API via the
            # proxy, so they are untrusted content and are escaped: a bracketed
            # issue title must not be parsed as markup.
            console.print(
                f"[bold]#{escape(str(it.get('number')))}[/bold] {escape(str(st))}: "
                f"{escape(str(it.get('title', '')))}")
            if it.get("html_url"):
                console.print(f"[dim]{escape(str(it['html_url']))}[/dim]")
            return
        issues = issue_tracker.list_issues(state)
    except LocalmError as e:
        console.print(
            f"[yellow]Could not load issues:[/yellow] {escape(e.summary)} ({escape(e.reason)}).")
        return

    if not issues:
        console.print("[dim]No issues.[/dim]")
        return
    for it in issues:
        st = it.get("state", "?")
        color = "green" if st == "closed" else "yellow"
        # No literal square brackets around dynamic text - rich would parse them as
        # markup tags and drop them. The interpolated data (number/state/title) is
        # externally sourced (the GitHub API via the proxy) and is escaped so it
        # cannot be parsed as markup either, regardless of what it contains.
        console.print(
            f"  [{color}]#{escape(str(it.get('number')))} {escape(str(st))}[/{color}]  "
            f"{escape(str(it.get('title', '')))}")
    console.print(f"[dim]{len(issues)} issue(s). Detail: localm issues <number>.[/dim]")
