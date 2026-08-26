# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm memory`` CLI: read and manage durable chat memory from the terminal.

Subcommands:
  localm memory list [--all] [--json]     what localm has remembered about you
  localm memory show ID                   one fact in full, with its provenance
  localm memory add TEXT [--kind K]       save a fact yourself
  localm memory forget ID [--yes]         delete one fact (NOT recoverable)
  localm memory forgotten                 facts localm archived on its own
  localm memory restore ID                bring an archived fact back
  localm memory corrections               proposals consolidation left for review
  localm memory accept ID / reject ID     resolve one proposal
  localm memory clear [--yes]             erase everything (NOT recoverable)

FORGET IS NOT ARCHIVED. ``MemoryStore.delete`` is a hard delete that never writes
the forgotten sidecar. Only prune eviction and an accepted correction archive a
record, so ``forgotten`` lists what localm dropped ON ITS OWN and ``restore`` can
only reach those.

NAMESPACE. Every command opens the SAME store the chat routes use -
``open_store(principal=None, agent="chat", scope_key="", root=<home>/memory)`` via
:func:`_store`. A CLI that opened a different agent, scope_key or root would show
an empty list and let the user conclude localm has learned nothing about them, so
that one helper is the only place any of those four values is written, and
``test_memory_cli.py`` pins CLI and route against each other.

PRINCIPAL. ``None``, which ``principal_of`` maps to the shared ``"owner"``
namespace, matching every write path on the route side, which collapses an
ADMIN-scoped caller to owner. A terminal user standing at the machine IS the
owner; there is no bearer to hash.
"""

from __future__ import annotations

import contextlib
import json as _json
import sys
import time

import click


@contextlib.contextmanager
def _refuse_if_locked():
    """Turn a cross-process write-lock refusal into a clear message and exit 1.

    A memory write shares its namespace with a running server (its consolidation
    pass, or an edit from the GUI). The store waits a bounded time for that other
    process and then refuses rather than interleaving with it, so this command
    reports WHO holds it - which the error already names - instead of showing
    "localm hit an unexpected error" and saving a bug report for a normal,
    recoverable situation. Same shape as ``localm rag``'s handler for the
    identical error.

    Nothing was written when this fires: the store refuses rather than proceeding
    unprotected, so re-running once the other process finishes is always safe."""
    from localm.rag.collection_lock import CollectionLockedError
    try:
        yield
    except CollectionLockedError as e:
        _fail(str(e))


class _MemoryGroup(click.Group):
    """Applies _refuse_if_locked to every subcommand, present and future.

    Wrapped at the group rather than in each command body, so a verb added later
    does not silently get the traceback-and-bug-report behaviour back."""

    def invoke(self, ctx):
        with _refuse_if_locked():
            return super().invoke(ctx)


@click.group(name="memory", cls=_MemoryGroup)
def main() -> None:
    """Read and manage what localm remembers about you across conversations."""


def _store():
    """The chat memory store, opened exactly as the routes open it.

    The four values here are load-bearing and are not parameterised: see this
    module's NAMESPACE note.
    """
    from localm import memory as _mem
    from localm.config import home_dir
    return _mem.open_store(None, "chat", "", root=home_dir() / "memory")


def _embed_fn():
    """The embedding callable, or None when no embedder is available.

    None is a NORMAL outcome, not a failure: recall and consolidation fall back to
    lexical BM25, and the store's own writers all accept ``embed_fn=None``. No
    command here requires an embedder to run; a fact saved without a vector is
    picked up by the next backfill.
    """
    try:
        from localm.inference.embedder import get_embedder
        emb = get_embedder()
        return emb.embed if emb is not None else None
    except Exception:                                          # noqa: BLE001
        # Best effort. Reported rather than swallowed, so a vectorless save is
        # visible and can be correlated with a broken embedder later.
        click.echo("Note: no embedding model available, so this runs without "
                   "semantic vectors (lexical fallback).", err=True)
        return None


def _fail(msg: str) -> None:
    """Print to stderr and exit non-zero.

    These commands are scriptable, and a "no such id" that exited 0 would be
    indistinguishable from a successful forget to anything reading the exit code.
    """
    click.echo(msg, err=True)
    sys.exit(1)


def _age(ts: float) -> str:
    if not ts:
        return "?"
    secs = max(0.0, time.time() - float(ts))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{int(secs // size)}{unit} ago"
    return "just now"


def _short(text: str, limit: int = 72) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


# ------------------------------------------------------------------ #
#  Reading                                                            #
# ------------------------------------------------------------------ #

@main.command("list")
@click.option("--all", "show_all", is_flag=True,
              help="Show every field, not just id/kind/text.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def memory_list(show_all: bool, as_json: bool) -> None:
    """What localm has remembered about you."""
    records = _store().all()
    if as_json:
        click.echo(_json.dumps([r.to_dict() for r in records], indent=2))
        return
    if not records:
        click.echo("Nothing remembered yet. localm saves facts as you chat "
                   "(outside privacy mode), or add one with `localm memory add`.")
        return
    click.echo(f"{len(records)} remembered fact(s):")
    for r in records:
        click.echo(f"  {r.id}  [{r.kind}] {_short(r.text)}")
        if show_all:
            click.echo(f"      importance {r.importance} - used {r.uses}x - "
                       f"created {_age(r.created)} - source {r.source or '?'}")


@main.command("show")
@click.argument("mem_id")
def memory_show(mem_id: str) -> None:
    """One fact in full, with its provenance."""
    rec = _store().get(mem_id)
    if rec is None:
        _fail(f"No memory with id {mem_id}.")
    click.echo(rec.text)
    click.echo("")
    click.echo(f"id         {rec.id}")
    click.echo(f"kind       {rec.kind}")
    click.echo(f"importance {rec.importance}")
    click.echo(f"uses       {rec.uses}")
    click.echo(f"created    {_age(rec.created)}")
    click.echo(f"updated    {_age(rec.updated)}")
    click.echo(f"last used  {_age(rec.last_used)}")
    click.echo(f"source     {rec.source or '?'}")


@main.command("forgotten")
def memory_forgotten() -> None:
    """Facts localm dropped on its own, which can still be brought back.

    NOT the things you forgot by hand: ``memory forget`` is a hard delete (see
    that command). This archive is filled by exactly two paths - prune evicting a
    record at the cap, and an accepted correction archiving the record it replaced.
    """
    rows = _store().forgotten()
    if not rows:
        click.echo("Nothing has been archived for this store. localm files a "
                   "fact here when it evicts one at the cap, or when you accept "
                   "a correction that replaces one.")
        return
    click.echo(f"{len(rows)} archived fact(s) - restore one with "
               "`localm memory restore <id>`:")
    for row in rows:
        # Rows are the record's own fields plus `forgotten_at`. There is no
        # `reason` key here - that belongs to the coder's episode archive, which
        # is a different store; printing one would render "?" forever.
        click.echo(f"  {row.get('id', '?')}  {_age(row.get('forgotten_at', 0))}"
                   f"  {_short(row.get('text', ''))}")


# ------------------------------------------------------------------ #
#  Writing                                                            #
# ------------------------------------------------------------------ #

@main.command("add")
@click.argument("text")
@click.option("--kind", type=click.Choice(["semantic", "episodic"]),
              default="semantic", show_default=True,
              help="semantic = a standing fact; episodic = something that happened.")
@click.option("--importance", default=0.8, show_default=True, type=float,
              help="Weighting used when recall ranks and prune evicts.")
def memory_add(text: str, kind: str, importance: float) -> None:
    """Save a fact yourself, without going through a chat turn.

    Produces the SAME record `POST /api/memory/append` does - kind semantic,
    source user, importance 0.8, and the same cap refusal.
    `MemoryRecord.__post_init__` silently coerces an unknown kind to "semantic"
    and an unknown source to "synth", so different values here would file the
    user's own assertion as machine-synthesised at a lower weight and prune's
    user-fact eviction reporting would stop seeing it. Both coercions are silent,
    so the choices here are constrained rather than free text.
    """
    if not text.strip():
        _fail("Refusing to save an empty memory.")
    from localm.memory.store import N_MAX, MemoryRecord
    store = _store()
    # Refuse past the cap rather than accept a fact the next prune would silently
    # evict - the same guard, and the same wording, as the append route.
    if len(store.all()) >= N_MAX:
        _fail(f"Memory is at its {N_MAX}-record cap; forget a fact before "
              "adding another.")
    rec = store.add(MemoryRecord(text=text.strip(), kind=kind, source="user",
                                 importance=importance), embed_fn=_embed_fn())
    click.echo(f"Remembered {rec.id}: {_short(rec.text)}")


@main.command("forget")
@click.argument("mem_id")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation.")
def memory_forget(mem_id: str, yes: bool) -> None:
    """Delete one fact. NOT recoverable.

    ``store.delete`` is a hard delete that does not write the forgotten archive
    at all. Only prune eviction and an accepted correction archive a record, so
    `memory restore` cannot reach anything deleted here.
    """
    store = _store()
    rec = store.get(mem_id)
    if rec is None:
        _fail(f"No memory with id {mem_id}.")
    if not yes:
        click.echo(f"Forget: {_short(rec.text)}")
        click.confirm("This deletes it outright and cannot be undone. Continue?",
                      abort=True)
    if not store.delete(mem_id):
        # get() found it a moment ago, so a False here is a real failure to write,
        # not a missing id - do not report it as a successful forget.
        _fail(f"Could not forget {mem_id}: the store refused the delete. "
              "Nothing was changed.")
    click.echo(f"Deleted {mem_id}.")


@main.command("restore")
@click.argument("mem_id")
def memory_restore(mem_id: str) -> None:
    """Bring a forgotten fact back into recall."""
    rec = _store().restore_forgotten(mem_id, embed_fn=_embed_fn())
    if rec is None:
        _fail(f"No forgotten memory with id {mem_id}. "
              "`localm memory forgotten` lists what can be restored.")
    click.echo(f"Restored {rec.id}: {_short(rec.text)}")


@main.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation.")
def memory_clear(yes: bool) -> None:
    """Erase everything localm remembers about you. NOT recoverable."""
    store = _store()
    live = len(store.all())
    gone = len(store.forgotten())
    if not store.remnants():
        click.echo("Nothing to clear.")
        return
    if not yes:
        click.echo(f"This erases {live} remembered fact(s) and {gone} archived "
                   "one(s), including the copies you could otherwise restore.")
        click.confirm("This cannot be undone. Continue?", abort=True)
    # include_forgotten=True: a plain clear() leaves every archived record
    # readable in the sidecar, so reporting the memory cleared without it would
    # be untrue.
    store.clear(include_forgotten=True)
    # Read back rather than trusting clear(). This command has no undo, so
    # "erased" is only claimed once the read-back confirms it; a partial erase
    # reported as success would leave remembered text on disk.
    after = _store()
    remnants = after.remnants()
    if remnants:
        _fail(f"Erase did not fully complete: {', '.join(remnants)} still on "
              "disk. This is NOT reported as cleared.")
    click.echo(f"Erased {live} remembered and {gone} forgotten fact(s).")


# ------------------------------------------------------------------ #
#  Corrections - the half the terminal could produce but never read   #
# ------------------------------------------------------------------ #

@main.command("corrections")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def memory_corrections(as_json: bool) -> None:
    """Proposals consolidation left for you to review.

    `localm job add --memory` schedules the consolidation that CREATES these.
    """
    rows = _store().corrections()
    if as_json:
        click.echo(_json.dumps([c.to_dict() for c in rows], indent=2))
        return
    if not rows:
        click.echo("No pending corrections.")
        return
    click.echo(f"{len(rows)} pending correction(s) - resolve with "
               "`localm memory accept|reject <id>`:")
    for c in rows:
        click.echo(f"  {c.id}  [{c.action}] confidence {c.confidence}")
        if c.target_text:
            click.echo(f"      was:  {_short(c.target_text)}")
        if c.proposed_text:
            click.echo(f"      now:  {_short(c.proposed_text)}")


def _resolve(correction_id: str, accept: bool) -> None:
    if not _store().resolve_correction(correction_id, accept,
                                       embed_fn=_embed_fn()):
        # `resolve_correction` returns None for TWO different reasons: the id is
        # genuinely unknown, OR the corrections sidecar could not be READ (a
        # transient lock), which it logs a warning for and treats as
        # non-destructive. `corrections()` cannot disambiguate either - it returns
        # [] on the same unreadable sidecar - so the message names both cases and
        # the exit stays non-zero, since the resolve did not happen either way.
        _fail(f"Did not resolve {correction_id}: either there is no pending "
              "correction with that id, or the corrections file could not be "
              "read just now. Nothing was changed. `localm memory corrections` "
              "lists what is pending; run with LOCALM_DEBUG=1 to see which it "
              "was.")
    click.echo(("Accepted " if accept else "Rejected ") + correction_id + ".")


@main.command("accept")
@click.argument("correction_id")
def memory_accept(correction_id: str) -> None:
    """Apply one proposed correction."""
    _resolve(correction_id, True)


@main.command("reject")
@click.argument("correction_id")
def memory_reject(correction_id: str) -> None:
    """Discard one proposed correction, leaving the memory as it was."""
    _resolve(correction_id, False)
