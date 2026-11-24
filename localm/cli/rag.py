# SPDX-License-Identifier: AGPL-3.0-or-later
import contextlib
import sys
from pathlib import Path

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


@contextlib.contextmanager
def _refuse_bad_path(console):
    """Turn a ValueError raised while indexing (a refused path, a dimension
    mismatch, a re-index that could not keep its embeddings) into a red message
    and exit 1 instead of a traceback."""
    from rich.markup import escape

    from ..rag import EmbeddingLossRefused
    try:
        yield
    except EmbeddingLossRefused as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        console.print(
            "[yellow]Start a localm server that has an embedding model "
            "('localm setup-embeddings') and run this again, or pass --yes "
            "without --embed to re-index lexical-only and drop the "
            "embeddings.[/yellow]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)


def _require_collection(console, coll, name):
    """Print "No such collection" and exit 1 when *coll* does not exist."""
    from rich.markup import escape
    if not coll.exists():
        console.print(f"[red]No such collection:[/red] {escape(name)}")
        sys.exit(1)


def _guard_vector_loss(console, coll, name, *, embed, yes, doing, doing_short):
    """Warn and ask before a re-index drops the embeddings *name* has.

    Returns ``(embed, allow_loss)``: the ``embed`` flag to use, and whether the
    person chose to lose the embeddings (``--yes`` without ``--embed``, or an
    answered yes). With no vectors, ``--embed`` or ``--yes`` nothing is asked.
    Answering no aborts. With no one to answer (no terminal) the embeddings are
    kept by re-embedding, and if that cannot be done the re-index is refused
    rather than dropping them. *doing* names the action in the warning,
    *doing_short* in the non-interactive note."""
    from rich.markup import escape
    if embed:
        return True, False
    if not coll.stats().get("has_vectors"):
        return False, True
    if yes:
        return False, True
    console.print(
        f"[yellow]'{escape(name)}' currently has semantic (hybrid) search. "
        f"{doing} without --embed will REMOVE the existing embeddings for "
        "every re-indexed document (it goes back to BM25/lexical-only).[/yellow]")
    # NOT `abort=True`: that collapses "the user answered no" and "there
    # was no user to answer" into the same Abort. Without it, an explicit
    # "no" RETURNS False while EOF still raises Abort, so the two stay
    # separable.
    try:
        proceed = click.confirm("Continue and drop the embeddings?")
    except click.Abort:
        console.print(
            "[yellow]Not an interactive terminal, so nothing can answer that. "
            f"{doing_short} WITH embeddings so they are not lost, and refuses if "
            "they cannot be computed - pass --yes to continue lexical-only "
            "instead, or --embed to silence this.[/yellow]")
        return True, False
    if not proceed:
        raise click.Abort()
    return False, True


def _note_vector_gap(console, name, *, had_vectors, embedded, result):
    """Say so when documents were indexed without embeddings into a collection
    that has them: semantic search then covers fewer of its chunks."""
    from rich.markup import escape
    n = result["added"] + result["updated"]
    if embedded or not had_vectors or not n:
        return
    console.print(
        f"[yellow]{n} document(s) were indexed without embeddings into "
        f"'{escape(name)}', which has them, so semantic search covers fewer of its "
        f"chunks. Close the gap with 'localm rag repair {escape(name)} --embed'.[/yellow]")


def _report_not_found(console, name, result):
    """Say which named paths no longer exist, and what to do about the
    documents indexed from them."""
    from rich.markup import escape
    gone = result.get("not_found") or []
    if not gone:
        return
    console.print(
        f"[yellow]{len(gone)} named path(s) no longer exist on disk and were not "
        f"re-indexed; anything already indexed from them is still in "
        f"'{escape(name)}'.[/yellow]")
    for p in gone[:10]:
        console.print(f"  [yellow]gone:[/yellow] {escape(p)}")
    if len(gone) > 10:
        console.print(f"  [yellow]... and {len(gone) - 10} more[/yellow]")
    console.print(
        f"[yellow]'localm rag resync {escape(name)}' flags them as missing, "
        f"'localm rag rm-doc {escape(name)} <path>' removes one.[/yellow]")


def _discard_if_empty(console, name):
    """Remove collection *name* when it holds no documents, after a refused
    first add left it behind. Says so when it cannot."""
    from rich.markup import escape

    from ..rag import Collection, CollectionLockedError, delete_collection
    try:
        if not Collection(name).documents():
            delete_collection(name)
    except (ValueError, OSError, CollectionLockedError) as e:
        console.print(f"[yellow]Could not remove the empty collection "
                      f"'{escape(name)}': {escape(str(e))}[/yellow]")


@main.group("rag")
def rag_group():
    """Knowledge collections - chat with your documents (offline RAG).

    Index files or folders into named collections, then ground chat replies
    in them: pick the collection in the GUI's parameters drawer, or query
    from the terminal. Retrieval is BM25 (always available offline);
    pass --embed, or index from the GUI, to blend in embeddings. PDF parsing is built in (pypdf).
    """




@rag_group.command("list")
def rag_list():
    """List collections with document and chunk counts."""
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection, collection_names
    from ..rag._store.indexing import count_noun
    console = Console()
    names = collection_names()
    if not names:
        console.print("[dim]No collections yet. "
                      "Create one:  localm rag add <name> <path>[/dim]")
        return
    for n in names:
        s = Collection(n, cache=False).stats()
        retrieval = "hybrid" if s["has_vectors"] else "BM25"
        marker = ("  [yellow](corrupt index - run 'localm rag repair')[/yellow]"
                  if s.get("corrupt") else "")
        console.print(f"[cyan]{escape(n)}[/cyan]  {count_noun(s['n_docs'], 'doc')} · "
                      f"{count_noun(s['n_chunks'], 'chunk')} · {retrieval}{marker}")




@rag_group.command("add")
@click.argument("collection")
@click.argument("paths", nargs=-1, required=True)
@click.option("--force", is_flag=True,
              help="Re-index even unchanged files (rebuild stale entries). On a "
                   "collection that has embeddings, without --embed this REMOVES "
                   "them for every re-indexed document (you will be asked to "
                   "confirm).")
@click.option("--embed", is_flag=True,
              help="Also compute embeddings at index time via a running localm "
                   "server, for hybrid (vector+lexical) retrieval - matching the "
                   "GUI. Degrades to lexical-only if no server is reachable.")
@click.option("--url", default=None,
              help="Server base URL for --embed (default: auto-discover a running "
                   "instance, else the configured port on localhost).")
@click.option("-y", "--yes", is_flag=True,
              help="Skip the embeddings-loss confirmation of --force.")
def rag_add(collection, paths, force, embed, url, yes):
    """Index files/folders into COLLECTION (created if missing).

    Folders are indexed recursively (txt/md/pdf/docx/html/code). Unchanged
    files (same content) are skipped, whatever their modified time; changed
    ones are re-indexed. Use --force to re-index regardless. By default CLI
    indexing is lexical-only; pass --embed to add embeddings via a running
    server, matching the GUI.

    --force on a collection that HAS embeddings, without --embed, would drop
    them for every re-indexed document, so you are asked first. Run
    non-interactively there is nothing to answer, so the embeddings are kept by
    re-embedding, and the command stops with exit 1, changing nothing, when no
    embedder can be reached; pass --yes to accept the drop. --embed with an
    unreachable embedder stops the same way.

    \b
    Examples:
      localm rag add manuals path/to/printer-manual.pdf
      localm rag add project path/to/myapp --embed
    """
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..pathsafe import is_unc_or_device_path
    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    console = Console()
    missing = [p for p in paths
               if not is_unc_or_device_path(str(Path(p).expanduser()))
               and not Path(p).expanduser().exists()]
    if missing:
        for p in missing:
            console.print(f"[red]Path does not exist: {escape(p)}[/red]")
        console.print("[red]Nothing was added.[/red]")
        sys.exit(1)
    from ..rag._store.indexing import count_noun
    coll = run_or_die(Collection, collection)
    had_vectors = bool(coll.exists() and coll.stats().get("has_vectors"))
    allow_loss = False
    if force:
        embed, allow_loss = _guard_vector_loss(
            console, coll, collection, embed=embed, yes=yes,
            doing="Re-indexing with --force", doing_short="Re-indexing")
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
    created_here = not coll.exists()
    with _refuse_if_locked(console), _refuse_bad_path(console):
        coll.create()             # a write too: takes the same lock
        try:
            result = coll.add_paths(
                list(paths), force=force, embed_fn=embed_fn, model_name=model_name,
                allow_vector_loss=allow_loss,
                on_progress=lambda t, **_: console.print(f"  [dim]{escape(t)}[/dim]"))
        except ValueError:
            if created_here:
                _discard_if_empty(console, collection)
            raise
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{count_noun(result['chunks'], 'chunk')} in '{escape(collection)}'")
    _report_not_found(console, collection, result)
    if not force:
        _note_vector_gap(console, collection, had_vectors=had_vectors,
                         embedded=embed, result=result)
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
    is nothing to answer, so repair keeps the embeddings by re-embedding, and
    stops with exit 1, changing nothing, when no embedder can be reached; pass
    --yes to accept the drop, or --embed to recompute them explicitly.
    """
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    from ..rag._store.indexing import count_noun
    console = Console()
    coll = run_or_die(Collection, collection)
    _require_collection(console, coll, collection)
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
    from localm.rag.store import _network_drives_allowed_fresh, floor_refusal
    allow_net = _network_drives_allowed_fresh()
    paths = []
    for p in repairable:
        reason = floor_refusal(p, allow_net)
        if reason:
            console.print(f"[yellow]skipping {escape(p)}: {escape(reason)}[/yellow]")
        else:
            paths.append(p)
    if not paths:
        console.print(f"[yellow]'{escape(collection)}' has no document source that "
                      "may be re-indexed.[/yellow]")
        return
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
    embed, allow_loss = _guard_vector_loss(
        console, coll, collection, embed=embed, yes=yes,
        doing="Repairing", doing_short="Repairing")
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    # See `rag add`: record the model this repair actually re-embeds with.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    with _refuse_if_locked(console), _refuse_bad_path(console):
        result = coll.add_paths(
            paths, force=True, embed_fn=embed_fn, model_name=model_name,
            allow_vector_loss=allow_loss,
            on_progress=lambda t, **_: console.print(f"  [dim]{escape(t)}[/dim]"))
    console.print(f"[green]repaired: {result['updated']} re-indexed, "
                  f"{result['added']} added[/green] - "
                  f"{count_noun(result['chunks'], 'chunk')} in '{escape(collection)}'")
    _report_not_found(console, collection, result)
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
    reported and skipped WHOLE - nothing under it is indexed or flagged. A
    document under a credential folder (.ssh, .aws, ...) is always removed.

    Schedule this instead of running it by hand:
    `localm job add sync-docs --rag --collection COLLECTION --cron "0 3 * * *"`.
    """
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import _report_add_paths_result, run_or_die
    from ..rag._store.indexing import count_noun
    console = Console()
    coll = run_or_die(Collection, collection)
    _require_collection(console, coll, collection)
    had_vectors = bool(coll.stats().get("has_vectors"))
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
    with _refuse_if_locked(console), _refuse_bad_path(console):
        result = coll.resync(
            embed_fn=embed_fn, policy=None, prune_missing=prune_missing,
            model_name=model_name,
            on_progress=lambda t, **_: console.print(f"  [dim]{escape(t)}[/dim]"))
    console.print(f"[green]{result['added']} added, {result['updated']} updated, "
                  f"{result['skipped']} unchanged[/green] - "
                  f"{count_noun(result['chunks'], 'chunk')} in '{escape(collection)}' over "
                  f"{len(result['roots'])} folder(s)")
    if result["restored"]:
        console.print(f"[green]{len(result['restored'])} previously missing "
                      f"document(s) are back.[/green]")
    still_gone = sorted(set(result["missing"]) | set(result["still_missing"]))
    if still_gone:
        console.print(
            f"[yellow]{len(still_gone)} document(s) are no longer on "
            f"disk. They are flagged, NOT removed - re-run with --prune-missing "
            f"to delete them from the index:[/yellow]")
        for p in still_gone[:10]:
            console.print(f"  [yellow]missing:[/yellow] {escape(p)}")
        if len(still_gone) > 10:
            console.print(f"  [yellow]... and {len(still_gone) - 10} more "
                          f"(see 'localm rag docs {escape(collection)}')[/yellow]")
    if result["pruned"]:
        console.print(f"[yellow]pruned {len(result['pruned'])} entr"
                      f"{'y' if len(result['pruned']) == 1 else 'ies'} (file gone "
                      f"or under a credential folder).[/yellow]")
    for r in result["unavailable_roots"] + result["blocked_roots"]:
        console.print(f"[yellow]skipped folder {escape(r['root'])}: {escape(r['reason'])} - "
                      f"nothing under it was indexed or flagged."
                      f"[/yellow]")
    if result.get("vector_degrade_reason"):
        # Also logged by the store.
        console.print(
            f"[yellow]Semantic search is degraded: "
            f"{escape(result['vector_degrade_reason'])}. The stored vector index was "
            f"left in place, not deleted - rebuild it with "
            f"'localm rag repair {escape(collection)} --embed'.[/yellow]")
    _note_vector_gap(console, collection, had_vectors=had_vectors,
                     embedded=embed, result=result)
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
    from localm.console import SafeConsole as Console
    from rich.markup import escape
    from localm.config import load_config
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    from localm.rag.store import Collection

    from localm.rag._store.indexing import count_noun

    console = Console()
    coll = Collection(collection)
    _require_collection(console, coll, collection)

    n = len(coll._chunks)
    if not n:
        console.print(f"[yellow]'{escape(collection)}' has no chunks to re-embed.[/yellow]")
        return
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
    was = coll.embedding_model()
    console.print(
        f"Re-embedding [bold]{escape(collection)}[/bold]: {count_noun(n, 'chunk')}"
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
            raise SystemExit(1) from e
    console.print(
        f"[green]Done.[/green] {count_noun(res['chunks'], 'chunk')} re-embedded at "
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
        from localm.jsonreply import response_json
        return [d["embedding"] for d in response_json(r)["data"]]
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
@click.option("--relevant-only", is_flag=True,
              help="Drop hits below the relevance floor the GUI chat applies "
                   "before injecting excerpts.")
@click.option("--rerank/--no-rerank", default=None,
              help="Re-score the best hits with an installed reranker model. "
                   "Default: the rag_rerank setting (on, and used only while a "
                   "reranker is installed).")
@click.option("--context", default=None,
              help="The previous chat message. With --relevant-only, a TEXT that "
                   "refers back to the conversation is searched together with it.")
def rag_query(collection, text, k, embed, url, relevant_only, rerank, context):
    """Show the top-K chunks COLLECTION returns for TEXT.

    By default the CLI scores lexically (BM25). Pass --embed to also embed the
    query against a running localm server, matching the GUI's hybrid ranking.
    Every top-K hit is shown however weak it is, unless --relevant-only drops
    those below the relevance floor.
    """
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection
    console = Console()
    coll = Collection(collection)
    _require_collection(console, coll, collection)
    embed_fn = _cli_rag_embed_fn(url) if embed else None
    from ..rag.rerank import rerank_plan
    plan = rerank_plan(enabled=rerank)
    if plan.note:
        console.print(f"[yellow]Not reranking:[/yellow] {escape(plan.note)}")
    elif rerank and plan.fn is None:
        console.print("[yellow]Not reranking:[/yellow] no reranker model is "
                      "installed (see 'localm pull').")
    from ..debuglog import native_console_paused
    with native_console_paused():
        hits = coll.query(text, k=k, embed_fn=embed_fn,
                          relevant_only=relevant_only, **plan.query_kwargs(),
                          context=context)
    if coll.rerank_degrade_reason:
        console.print(f"[yellow]Not reranked:[/yellow] "
                      f"{escape(coll.rerank_degrade_reason)}")
    if not hits:
        console.print("[dim](no relevant matches)[/dim]" if relevant_only
                      else "[dim](no matches)[/dim]")
        return
    for i, h in enumerate(hits, 1):
        scored = f"score {h['score']}"
        if "rerank_score" in h:
            scored += f", rerank {h['rerank_score']}"
        console.print(f"[cyan][{i}][/cyan] [bold]{escape(h['source'])}[/bold]:{h['pos']} "
                      f"[dim]({scored})[/dim]")
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
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    _require_collection(console, coll, collection)
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
    from localm.console import SafeConsole as Console
    from rich.markup import escape

    from ..rag import Collection
    from .errors import run_or_die
    console = Console()
    coll = run_or_die(Collection, collection)
    _require_collection(console, coll, collection)
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
    from localm.console import SafeConsole as Console
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
