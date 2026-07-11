# SPDX-License-Identifier: AGPL-3.0-or-later
import sys

import click

from ._core import main


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
        console.print(f"[cyan]{n}[/cyan]  {s['n_docs']} docs · "
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
    to add embeddings via a running server, matching the GUI (REC-RAG-EMBED-PARITY).

    \b
    Examples:
      localm rag add manuals path/to/printer-manual.pdf
      localm rag add project path/to/myapp --embed
    """
    from rich.console import Console
    from ..rag import Collection
    console = Console()
    try:
        coll = Collection(collection)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    coll.create()
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    result = coll.add_paths(list(paths), force=force, embed_fn=embed_fn,
                            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{collection}'")
    for f in result["failed"]:
        console.print(f"  [yellow]failed:[/yellow] {f['path']}: {f['error']}")
    if result["failed"]:
        sys.exit(1)




@rag_group.command("repair")
@click.argument("collection")
def rag_repair(collection):
    """Re-index every file in COLLECTION, rebuilding stale entries.

    Use this when an index may be out of date (e.g. files edited in place
    without a size change). Re-reads and re-chunks every indexed document.
    """
    from rich.console import Console
    from ..rag import Collection
    console = Console()
    try:
        coll = Collection(collection)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    paths = coll.documents()
    if not paths:
        if coll.corrupt:
            # Corrupt is not the same as empty: say so instead of the misleading
            # "no indexed documents" (AGENTS rule 5 - do not hide the problem).
            console.print(f"[yellow]'{collection}' index is corrupt and no "
                          "document sources survived to rebuild from.[/yellow]")
        else:
            console.print(f"[yellow]'{collection}' has no indexed documents.[/yellow]")
        return
    if coll.corrupt:
        console.print(f"[yellow]'{collection}' index was corrupt; rebuilding "
                      f"from {len(paths)} source(s).[/yellow]")
    result = coll.add_paths(paths, force=True,
                            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]repaired: {result['updated']} re-indexed, "
                  f"{result['added']} added[/green] - "
                  f"{result['chunks']} chunks in '{collection}'")
    for f in result["failed"]:
        console.print(f"  [yellow]failed:[/yellow] {f['path']}: {f['error']}")
    if result["failed"]:
        sys.exit(1)




def _cli_rag_embed_fn(url):
    """Build a query embedder that calls a running localm server's
    /v1/embeddings (for `rag query --embed`), so the CLI gets the same hybrid
    retrieval the GUI does. Returns a lazy callable - it only connects when the
    query is actually embedded, so a missing server degrades to lexical-only
    inside Collection.query rather than failing the command. *url* overrides the
    auto-discovered base URL."""
    import os

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

    def _embed(texts: list) -> list:
        import requests
        from localm import tls as _tls
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(embeddings_url, json={"input": texts, "model": "localm"},
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
    from ..rag import Collection
    console = Console()
    coll = Collection(collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {collection}")
        sys.exit(1)
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    hits = coll.query(text, k=k, embed_fn=embed_fn)
    if not hits:
        console.print("[dim](no matches)[/dim]")
        return
    for i, h in enumerate(hits, 1):
        console.print(f"[cyan][{i}][/cyan] [bold]{h['source']}[/bold]:{h['pos']} "
                      f"[dim](score {h['score']})[/dim]")
        excerpt = h["text"][:300].replace("\n", " ")
        console.print(f"    {excerpt}\n")




@rag_group.command("rm")
@click.argument("collection")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rag_rm(collection, yes):
    """Delete a collection (the index only - original files are untouched)."""
    from rich.console import Console
    from ..rag import delete_collection
    console = Console()
    if not yes:
        click.confirm(f"Delete collection '{collection}'? Original files are "
                      "kept; only the index is removed.", abort=True)
    try:
        if delete_collection(collection):
            console.print(f"[green]Deleted '{collection}'.[/green]")
        else:
            console.print(f"[red]No such collection:[/red] {collection}")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
