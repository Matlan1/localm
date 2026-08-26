# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import sys

import click

from ._core import main


@contextlib.contextmanager
def _refuse_if_locked(console):
    """Turn a cross-process write-lock refusal into a clear message and exit 1.

    A `rag` write shares its collection with a running server (its scheduler, or
    an add from the GUI). The store waits a bounded time for that other process
    and then refuses rather than interleaving with it; the refusal names who
    holds the collection, and that message is printed instead of a
    traceback."""
    from rich.markup import escape

    from ..rag import CollectionLockedError
    try:
        yield
    except CollectionLockedError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)


@main.group("rag")
def rag_group():
    """Knowledge collections - chat with your documents (offline RAG).

    Index files or folders into named collections, then ground chat replies
    in them: pick the collection in the GUI's parameters drawer, or query
    from the terminal. Retrieval is BM25 (always available offline);
    embeddings are blended in when indexed via the GUI with a model that
    supports them. PDF needs the [rag] extra:  pip install "localm[rag]"
    """




@rag_group.command("list")
def rag_list():
    """List collections with document and chunk counts."""
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection, collection_names
    console = Console()
    names = collection_names()
    if not names:
        console.print("[dim]No collections yet. "
                      "Create one:  localm rag add <name> <path>[/dim]")
        return
    for n in names:
        s = Collection(n).stats()
        retrieval = "hybrid" if s["has_vectors"] else "BM25"
        marker = ("  [yellow](corrupt index - run 'localm rag repair')[/yellow]"
                  if s.get("corrupt") else "")
        console.print(f"[cyan]{escape(n)}[/cyan]  {s['n_docs']} docs · "
                      f"{s['n_chunks']} chunks · {retrieval}{marker}")




@rag_group.command("add")
@click.argument("collection")
@click.argument("paths", nargs=-1, required=True)
@click.option("--force", is_flag=True,
              help="Re-index even unchanged files (rebuild stale entries).")
@click.option("--embed", is_flag=True,
              help="Also compute embeddings at index time via a running localm "
                   "server, for hybrid (vector+lexical) retrieval - matching the "
                   "GUI. Degrades to lexical-only if no server is reachable.")
@click.option("--url", default=None,
              help="Server base URL for --embed (default: auto-discover a running "
                   "instance, else the configured port on localhost).")
def rag_add(collection, paths, force, embed, url):
    """Index files/folders into COLLECTION (created if missing).

    Folders are indexed recursively (txt/md/pdf/docx/html/code). Unchanged
    files (same content) are skipped; changed ones are re-indexed. Use --force
    to re-index regardless. By default CLI indexing is lexical-only; pass --embed
    to add embeddings via a running server, matching the GUI.

    \b
    Examples:
      localm rag add manuals path/to/printer-manual.pdf
      localm rag add project path/to/myapp --embed
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    # The configured embedding model's name, so a fresh collection ends up with
    # embedding_model() on record. None (never embedded) is fine: the store only
    # records the name the first time embedding actually succeeds.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    with _refuse_if_locked(console):
        coll.create()             # a write too: takes the same lock
        result = coll.add_paths(
            list(paths), force=force, embed_fn=embed_fn, model_name=model_name,
            on_progress=lambda t: console.print(f"  [dim]{escape(t)}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{escape(collection)}'")
    _report_add_paths_result(result)




@rag_group.command("repair")
@click.argument("collection")
@click.option("--embed", is_flag=True,
              help="Also compute embeddings while repairing, via a running localm "
                   "server - matching 'rag add --embed' / the GUI. Without this, "
                   "repairing a collection that already has embeddings REMOVES them "
                   "for every re-indexed document (you will be asked to confirm).")
@click.option("--url", default=None,
              help="Server base URL for --embed (default: auto-discover a running "
                   "instance, else the configured port on localhost).")
@click.option("-y", "--yes", is_flag=True,
              help="Skip the embeddings-loss confirmation.")
def rag_repair(collection, embed, url, yes):
    """Re-index every file in COLLECTION, rebuilding stale entries.

    Use this when an index may be out of date (e.g. files edited in place
    without a size change). Re-reads and re-chunks every indexed document.

    Repair is a full re-index, not an in-place patch, so without --embed every
    re-indexed chunk gets no vector and a collection that had semantic search
    drops to lexical-only (BM25). On a collection that HAS embeddings you are
    asked before that happens. Run non-interactively (cron, CI, a script) there
    is nothing to answer, so repair keeps the embeddings rather than dropping
    them silently or refusing to run; pass --yes to accept the drop, or --embed
    to recompute them explicitly.
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    paths = coll.documents()
    if not paths:
        if coll.corrupt:
            # Corrupt is not the same as empty: say so instead of "no
            # indexed documents".
            console.print(f"[yellow]'{escape(collection)}' index is corrupt and no "
                          "document sources survived to rebuild from.[/yellow]")
        else:
            console.print(f"[yellow]'{escape(collection)}' has no indexed documents.[/yellow]")
        return
    # An `upload:<name>` doc has no server-side source - the uploaded bytes are
    # never retained - so add_paths silently drops it
    # (Path('upload:x').is_file() is always False) and repair touches nothing
    # for it. A collection built ENTIRELY from uploads is refused; a MIXED one
    # proceeds on what it can rebuild and says what it cannot.
    repairable = [p for p in paths if not p.startswith("upload:")]
    upload_only = len(paths) - len(repairable)
    if not repairable:
        if coll.corrupt:
            console.print(
                f"[yellow]'{escape(collection)}' index is corrupt, but every document "
                "here was added via upload and has no server-side source to "
                "rebuild from - repair cannot fix it. Re-upload the affected "
                "file(s) instead.[/yellow]")
        else:
            console.print(
                f"[yellow]'{escape(collection)}' has no server-side document sources "
                "to repair (every document here was added via upload).[/yellow]")
        return
    paths = repairable
    if coll.corrupt:
        console.print(f"[yellow]'{escape(collection)}' index was corrupt; rebuilding "
                      f"from {len(paths)} source(s).[/yellow]")
    if upload_only:
        console.print(
            f"[yellow]{upload_only} uploaded document(s) in '{escape(collection)}' have "
            "no server-side source and will be left as-is.[/yellow]")
    # A repair is force=True on every doc: without --embed, every re-indexed
    # chunk gets NO vector, dropping a collection that currently has semantic
    # search back to BM25-only. Surface that and require an explicit yes.
    if not embed and coll.stats().get("has_vectors") and not yes:
        console.print(
            f"[yellow]'{escape(collection)}' currently has semantic (hybrid) search. "
            "Repairing without --embed will REMOVE the existing embeddings for "
            "every re-indexed document (it goes back to BM25/lexical-only).[/yellow]")
        # NOT `abort=True`: that collapses "the user answered no" and "there
        # was no user to answer" into the same Abort. Without it, an explicit
        # "no" RETURNS False while EOF still raises Abort, so the two stay
        # separable.
        try:
            proceed = click.confirm("Continue and drop the embeddings?")
        except click.Abort:
            # Nobody was there to answer: keep the embeddings the collection
            # already has, say so, and let the repair run (exit 0).
            console.print(
                "[yellow]Not an interactive terminal, so nothing can answer that. "
                "Repairing WITH embeddings so they are not lost - pass --yes to "
                "repair lexical-only instead, or --embed to silence this.[/yellow]")
            embed = True
        else:
            if not proceed:
                raise click.Abort()
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    # See `rag add`: record the model this repair actually re-embeds with.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    with _refuse_if_locked(console):
        result = coll.add_paths(
            paths, force=True, embed_fn=embed_fn, model_name=model_name,
            on_progress=lambda t: console.print(f"  [dim]{escape(t)}[/dim]"))
    console.print(f"[green]repaired: {result['updated']} re-indexed, "
                  f"{result['added']} added[/green] - "
                  f"{result['chunks']} chunks in '{escape(collection)}'")
    _report_add_paths_result(result)




@rag_group.command("resync")
@click.argument("collection")
@click.option("--embed", is_flag=True,
              help="Also compute embeddings for newly indexed documents, via a "
                   "running localm server - matching 'rag add --embed' / the GUI. "
                   "Without it, new documents are indexed lexical-only.")
@click.option("--url", default=None,
              help="Server base URL for --embed (default: auto-discover a running "
                   "instance, else the configured port on localhost).")
@click.option("--prune-missing", is_flag=True,
              help="Also REMOVE index entries whose source file is gone. Off by "
                   "default: a vanished file is flagged, not deleted, so an "
                   "unplugged drive or a mid-sync folder cannot destroy the index.")
def rag_resync(collection, embed, url, prune_missing):
    """Re-sync COLLECTION with the folders it was indexed from.

    Re-walks each indexed folder, so a file ADDED to it since the last index is
    picked up, a changed file is re-indexed, and an unchanged file is skipped by
    content hash (the same incremental path `rag add` uses). Individually
    indexed files are re-checked too.

    A document whose file has VANISHED is flagged, not removed: its chunks stay
    searchable and the flag clears by itself if the file comes back, so a moved
    file or an unplugged drive is never silently forgotten. Pass --prune-missing
    to actually delete those entries. A folder that is not currently reachable is
    reported and skipped WHOLE - nothing under it is touched.

    Schedule this instead of running it by hand:
    `localm job add sync-docs --rag --collection COLLECTION --cron "0 3 * * *"`.
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {escape(collection)}")
        sys.exit(1)
    if not coll.roots():
        # Not an error: with no folder to re-walk, a re-sync can only refresh
        # the files it already knows, so say that rather than reporting a bare
        # "0 added".
        console.print(
            f"[yellow]'{escape(collection)}' has no indexed folders recorded, so this "
            f"only re-checks its {len(coll.documents())} known document(s). "
            f"Index a folder (localm rag add {escape(collection)} <folder>) to have new "
            f"files in it picked up.[/yellow]")
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    # See `rag add`: record the model any newly-embedded document used.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    # policy=None, like `rag add` from the CLI: a local operator stays
    # unconfined (the credential-dir hard floor still applies inside
    # confine_index_path). The SCHEDULED job path passes indexing_policy()
    # instead.
    with _refuse_if_locked(console):
        result = coll.resync(
            embed_fn=embed_fn, policy=None, prune_missing=prune_missing,
            model_name=model_name,
            on_progress=lambda t: console.print(f"  [dim]{escape(t)}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{escape(collection)}' over "
                  f"{len(result['roots'])} folder(s)")
    if result["restored"]:
        console.print(f"[green]{len(result['restored'])} previously missing "
                      f"document(s) are back.[/green]")
    if result["missing"]:
        console.print(
            f"[yellow]{len(result['missing'])} document(s) are no longer on "
            f"disk. They are flagged, NOT removed - re-run with --prune-missing "
            f"to delete them from the index:[/yellow]")
        for p in result["missing"][:10]:
            console.print(f"  [yellow]missing:[/yellow] {escape(p)}")
    if result["pruned"]:
        console.print(f"[yellow]pruned {len(result['pruned'])} entr"
                      f"{'y' if len(result['pruned']) == 1 else 'ies'} whose file "
                      f"is gone.[/yellow]")
    for r in result["unavailable_roots"] + result["blocked_roots"]:
        console.print(f"[yellow]skipped folder {escape(r['root'])}: {escape(r['reason'])} - "
                      f"nothing under it was indexed, flagged, or removed."
                      f"[/yellow]")
    if result.get("vector_degrade_reason"):
        # Also logged by the store.
        console.print(
            f"[yellow]Semantic search is degraded: "
            f"{escape(result['vector_degrade_reason'])}. The stored vector index was "
            f"left in place, not deleted - rebuild it with "
            f"'localm rag repair {escape(collection)} --embed'.[/yellow]")
    _report_add_paths_result(result)




@rag_group.command("reembed")
@click.argument("collection")
@click.option("--url", default=None,
              help="Base URL of a running localm server (auto-detected otherwise).")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rag_reembed(collection, url, yes):
    """Recompute a collection's vectors with the CURRENT embedding model.

    Use this after changing the embedding model. It re-embeds from the chunk text
    already stored in the collection, so the ORIGINAL SOURCE FILES are not needed
    and nothing is deleted or re-chunked - which is the difference from
    `rag repair --embed`, that re-indexes from the source files and cannot help
    when they have moved or the documents arrived as uploads.
    """
    from rich.console import Console
    from rich.markup import escape
    from localm.config import load_config
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    from localm.rag.store import Collection

    console = Console()
    coll = Collection(collection)
    if not coll.exists():
        console.print(f"[red]No collection named '{escape(collection)}'.[/red]")
        raise SystemExit(1)

    n = len(coll._chunks)
    if not n:
        console.print(f"[yellow]'{escape(collection)}' has no chunks to re-embed.[/yellow]")
        return
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
    was = coll.embedding_model()
    console.print(
        f"Re-embedding [bold]{escape(collection)}[/bold]: {n} chunks"
        + (f", built with {escape(was)}" if was else "")
        + f" -> {escape(model)}.")
    if not yes and not click.confirm("  Proceed?", default=True):
        console.print("[dim]Cancelled - nothing changed.[/dim]")
        return

    with _refuse_if_locked(console):
        try:
            res = coll.reembed(embed_fn=_cli_rag_embed_fn(url), model_name=model,
                               on_progress=lambda m, **_: console.print(f"[dim]{escape(m)}[/dim]"))
        except Exception as e:
            # reembed only swaps the new index in after the whole set is
            # computed and validated, so the previous one is still intact.
            console.print(f"[red]Re-embed failed: {escape(str(e))}[/red]")
            console.print("[dim]The previous index was left untouched.[/dim]")
            raise SystemExit(1)
    console.print(
        f"[green]Done.[/green] {res['chunks']} chunks re-embedded at "
        f"{res['dim']} dimensions with {escape(model)}.")


def _cli_rag_embed_fn(url):
    """Build a query embedder that calls a running localm server's
    /v1/embeddings (for `rag query --embed`).

    Returns a lazy callable: it only connects when the query is actually
    embedded, so a missing server degrades to lexical-only inside
    Collection.query rather than failing the command. *url* overrides the
    auto-discovered base URL."""
    base = url
    if not base:
        try:
            from localm import instances
            from localm.config import home_dir
            entry = instances.attach_target(home_dir(), instances.resolve_root_dir())
            base = instances.attach_url(entry) if entry else None
        except Exception:
            base = None
    if not base:
        from localm.config import load_config
        base = f"http://127.0.0.1:{load_config().get('port', 8642)}"
    base = base.rstrip("/")
    embeddings_url = base + ("/embeddings" if base.endswith("/v1") else "/v1/embeddings")

    # Send the CONFIGURED embedding model name, as the GUI's server-side
    # self-embed does: /v1/embeddings routes an embed request to the dedicated
    # embedder ONLY when the model name matches the registered embedder OR the
    # configured `embedding_model`, and otherwise falls through to the chat
    # path.
    from localm.config import load_config as _lc
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    _emb_model = str(_lc().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip() or "localm"

    def _embed(texts: list) -> list:
        import requests
        from localm import tls as _tls
        from localm.auth import get_api_key
        headers = {}
        # get_api_key() reads the env var, else the persisted <home>/auth.key,
        # where a `localm key generate` / launcher-keyed server keeps it.
        key = get_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(embeddings_url, json={"input": texts, "model": _emb_model},
                          headers=headers, timeout=120,
                          verify=_tls.requests_verify(embeddings_url))
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    return _embed




@rag_group.command("query")
@click.argument("collection")
@click.argument("text")
@click.option("-k", default=4, show_default=True, help="Number of results.")
@click.option("--embed", is_flag=True,
              help="Embed the query via a running localm server for hybrid "
                   "(vector+lexical) retrieval, matching the GUI. Degrades to "
                   "lexical-only if no server is reachable.")
@click.option("--url", default=None,
              help="Server base URL for --embed (default: auto-discover a running "
                   "instance, else the configured port on localhost).")
def rag_query(collection, text, k, embed, url):
    """Show the top-K chunks COLLECTION returns for TEXT.

    By default the CLI scores lexically (BM25). Pass --embed to also embed the
    query against a running localm server, matching the GUI's hybrid ranking.
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    console = Console()
    coll = Collection(collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {escape(collection)}")
        sys.exit(1)
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    hits = coll.query(text, k=k, embed_fn=embed_fn)
    if not hits:
        console.print("[dim](no matches)[/dim]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"[cyan][{i}][/cyan] [bold]{escape(h['source'])}[/bold]:{h['pos']} "
                      f"[dim](score {h['score']})[/dim]")
        excerpt = h["text"][:300].replace("\n", " ")
        console.print(f"    {escape(excerpt)}\n")




@rag_group.command("docs")
@click.argument("collection")
def rag_docs(collection):
    """List COLLECTION's indexed documents: path, chunk count, and status.

    A document flagged (missing) was indexed but its source file is no longer
    on disk (see 'rag resync'); its chunks stay searchable until removed. One
    flagged (uploaded) was added via the GUI/API upload path rather than from
    a file path, so 'rag repair'/'rag resync' cannot re-read it from disk.
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {escape(collection)}")
        sys.exit(1)
    docs = coll.docs()
    if not docs:
        console.print(f"[dim]'{escape(collection)}' has no indexed documents.[/dim]")
        return
    for d in docs:
        n = d.get("chunks", 0)
        tags = ""
        if d.get("missing"):
            tags += "  [yellow](missing)[/yellow]"
        if d.get("uploaded"):
            tags += "  [dim](uploaded)[/dim]"
        console.print(f"{escape(d['path'])}  [dim]{n} chunk{'s' if n != 1 else ''}[/dim]{tags}")


@rag_group.command("rm-doc")
@click.argument("collection")
@click.argument("path")
def rag_rm_doc(collection, path):
    """Remove one document from COLLECTION (see 'rag docs' for PATH values).

    Drops only this document's chunks (and vectors) from the index; the
    original file on disk is untouched. Use 'rag rm' to delete the whole
    collection instead.
    """
    from rich.console import Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {escape(collection)}")
        sys.exit(1)
    with _refuse_if_locked(console):
        if coll.remove_doc(path):
            console.print(f"[green]Removed '{escape(path)}' from '{escape(collection)}'.[/green]")
        else:
            console.print(f"[red]Not in this collection:[/red] {escape(path)}")
            sys.exit(1)


@rag_group.command("rm")
@click.argument("collection")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rag_rm(collection, yes):
    """Delete a collection (the index only - original files are untouched)."""
    from rich.console import Console
    from rich.markup import escape

    from ..rag import delete_collection
    console = Console()
    if not yes:
        click.confirm(f"Delete collection '{collection}'? Original files are "
                      "kept; only the index is removed.", abort=True)
    with _refuse_if_locked(console):
        try:
            # on_wait: a delete that queues behind a running index reports
            # what it is waiting for.
            if delete_collection(
                    collection,
                    on_wait=lambda t: console.print(f"  [dim]{escape(t)}[/dim]")):
                console.print(f"[green]Deleted '{escape(collection)}'.[/green]")
            else:
                console.print(f"[red]No such collection:[/red] {escape(collection)}")
                sys.exit(1)
        except ValueError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
            sys.exit(1)
