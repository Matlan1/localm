# SPDX-License-Identifier: AGPL-3.0-or-later
"""The checks every CLI writer of the ``embedding_model`` setting runs before it
writes: ``localm setup-embeddings --model`` and ``localm config
embedding_model``."""

from __future__ import annotations

import sys

import click

from ._core import console


def check_embedding_switch(model: str, *, yes: bool) -> bool:
    """Refuse *model* (exit 1) when it cannot be the embedding model, warn when
    it names nothing yet, and when it differs from the current setting list the
    collections the switch invalidates and ask before going on (``--yes``
    skips the question; with nobody to answer, the switch goes on and says so).

    Returns True when, after the switch, collections that are lexical-only (or
    built with another model) still need ``localm rag reembed``."""
    from rich.markup import escape

    from localm.config import load_config
    from localm.inference.embedder import embedding_model_refusal, embedding_model_unresolved
    why = embedding_model_refusal(model)
    if why:
        console.print(f"[red]Not switching the embedding model:[/red] {escape(why)}")
        sys.exit(1)
    unresolved = embedding_model_unresolved(model)
    if unresolved:
        console.print(f"[yellow]{escape(unresolved)}[/yellow]")
    current = str(load_config().get("embedding_model") or "")
    if model == current:
        return True
    from localm.rag import (collection_provenance_note, collection_provenance_report,
                            lexical_only_collection_names)
    affected = collection_provenance_report(candidate_model=model)
    if not affected:
        return bool(lexical_only_collection_names())
    console.print(f"[yellow]{escape(collection_provenance_note(model, affected))}[/yellow]")
    for c in affected:
        built = f" (built with {escape(c['built_with'])})" if c.get("built_with") else ""
        if c.get("mixed"):
            built += " (mixed: vectors from more than one model)"
        chunks = f" - {c['n_chunks']} chunks" if c.get("n_chunks") is not None else ""
        console.print(f"  - {escape(c['name'])}{built}{chunks}")
    if not yes:
        try:
            proceed = click.confirm("Continue with the switch?")
        except click.Abort:
            console.print(
                "[yellow]Not an interactive terminal, so nothing can answer that. "
                "Proceeding with the switch - pass --yes to silence this prompt "
                "next time.[/yellow]")
            proceed = True
        if not proceed:
            raise click.Abort()
    return True
