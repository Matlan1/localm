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
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    coll.create()
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    result = coll.add_paths(list(paths), force=force, embed_fn=embed_fn,
                            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{collection}'")
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
    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
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
    # A repair is force=True on every doc: without --embed, every re-indexed
    # chunk gets NO vector, silently dropping a collection that currently has
    # semantic search back to BM25-only (this is exactly the bug behind
    # REC-RAG-REPAIR-EMBED - a plain "repaired" success line gave no hint that
    # embeddings had just been removed). Surface it and require an explicit
    # yes rather than hiding a real capability loss (AGENTS rule 5).
    if not embed and coll.stats().get("has_vectors") and not yes:
        console.print(
            f"[yellow]'{collection}' currently has semantic (hybrid) search. "
            "Repairing without --embed will REMOVE the existing embeddings for "
            "every re-indexed document (it goes back to BM25/lexical-only).[/yellow]")
        # Deliberately NOT `abort=True`: it collapses "the user answered no" and
        # "there was no user to answer" into the same Abort. That is REG-589 - run
        # from cron/CI/a script with no stdin, click.confirm hits EOF, and repair
        # exits non-zero having done NOTHING, on exactly the stale/corrupt hybrid
        # indexes it exists to rebuild. Without abort=True, an explicit "no"
        # RETURNS False while EOF still raises Abort, so the two are separable.
        try:
            proceed = click.confirm("Continue and drop the embeddings?")
        except click.Abort:
            # Nobody was there to answer. Both alternatives are bad: aborting
            # stops repairing (REG-589), and proceeding would silently delete this
            # collection's semantic search (REC-RAG-REPAIR-EMBED, the reason this
            # prompt exists). So take the branch that loses NOTHING - preserve the
            # embeddings the collection already has - and SAY so rather than
            # choosing silently (rule 5). The repair still happens, exit 0.
            console.print(
                "[yellow]Not an interactive terminal, so nothing can answer that. "
                "Repairing WITH embeddings so they are not lost - pass --yes to "
                "repair lexical-only instead, or --embed to silence this.[/yellow]")
            embed = True
        else:
            if not proceed:
                raise click.Abort()
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    result = coll.add_paths(paths, force=True, embed_fn=embed_fn,
                            on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]repaired: {result['updated']} re-indexed, "
                  f"{result['added']} added[/green] - "
                  f"{result['chunks']} chunks in '{collection}'")
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
    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {collection}")
        sys.exit(1)
    if not coll.roots():
        # Not an error: the collection works, it just has no folder to re-walk,
        # so a re-sync can only refresh the files it already knows. Say which it
        # is rather than reporting a bare "0 added" that looks like a no-op.
        console.print(
            f"[yellow]'{collection}' has no indexed folders recorded, so this "
            f"only re-checks its {len(coll.documents())} known document(s). "
            f"Index a folder (localm rag add {collection} <folder>) to have new "
            f"files in it picked up.[/yellow]")
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    # policy=None, exactly like `rag add` from the CLI: the local operator can
    # already read their own files, so they stay unconfined (the credential-dir
    # hard floor still applies inside confine_index_path). The SCHEDULED job path
    # is different on purpose and passes indexing_policy() - it is not a local
    # operator and can be created through the API by a scoped key.
    result = coll.resync(embed_fn=embed_fn, policy=None,
                         prune_missing=prune_missing,
                         on_progress=lambda t: console.print(f"  [dim]{t}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{result['chunks']} chunks in '{collection}' over "
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
            console.print(f"  [yellow]missing:[/yellow] {p}")
    if result["pruned"]:
        console.print(f"[yellow]pruned {len(result['pruned'])} entr"
                      f"{'y' if len(result['pruned']) == 1 else 'ies'} whose file "
                      f"is gone.[/yellow]")
    for r in result["unavailable_roots"] + result["blocked_roots"]:
        console.print(f"[yellow]skipped folder {r['root']}: {r['reason']} - "
                      f"nothing under it was indexed, flagged, or removed."
                      f"[/yellow]")
    if result.get("vector_degrade_reason"):
        # The store logs this, but a CLI user does not read the debug log.
        console.print(
            f"[yellow]Semantic search is degraded: "
            f"{result['vector_degrade_reason']}. The stored vector index was "
            f"left in place, not deleted - rebuild it with "
            f"'localm rag repair {collection} --embed'.[/yellow]")
    _report_add_paths_result(result)




def _cli_rag_embed_fn(url):
    """Build a query embedder that calls a running localm server's
    /v1/embeddings (for `rag query --embed`), so the CLI gets the same hybrid
    retrieval the GUI does. Returns a lazy callable - it only connects when the
    query is actually embedded, so a missing server degrades to lexical-only
    inside Collection.query rather than failing the command. *url* overrides the
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

    # Send the CONFIGURED embedding model name, exactly as the GUI's server-side
    # self-embed does (plugins/builtin/rag/plug.py _make_self_embed). The old
    # hardcoded "model": "localm" was the bug: /v1/embeddings routes an embed
    # request to the dedicated embedder ONLY when the model name matches the
    # registered embedder OR the configured `embedding_model` (see
    # inference/routes/chat.py). "localm" matches neither, so with a dedicated
    # embedder and no chat model loaded it fell through to the chat path and
    # 503'd - making EVERY `rag add/query --embed` from the CLI silently index
    # lexical-only, while the GUI (which sends the config name) got real
    # embeddings. This is the REC-RAG-EMBED-PARITY the docstring promised but the
    # name broke.
    from localm.config import load_config as _lc
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    _emb_model = str(_lc().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip() or "localm"

    def _embed(texts: list) -> list:
        import requests
        from localm import tls as _tls
        from localm.auth import get_api_key
        headers = {}
        # env var, else the persisted <home>/auth.key: a `localm key generate` /
        # launcher-keyed server keeps the key in auth.key, not the env, so reading
        # env-only 401'd every --embed call and silently indexed lexical-only
        # (memory-audit cluster 19).
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
