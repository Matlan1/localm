# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm rerank``: rank documents against a query with a reranker model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from ._core import _complete_model_name, console, main

_EXCERPT_CHARS = 90


def _read_documents(documents, docs_file) -> list:
    """The documents named as arguments plus the non-blank lines of
    *docs_file*, in that order."""
    texts = list(documents)
    if docs_file:
        try:
            raw = Path(docs_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise click.ClickException(f"cannot read {docs_file}: {e}")
        texts.extend(line for line in raw.splitlines() if line.strip())
    return texts


@main.command("rerank")
@click.argument("query")
@click.argument("documents", nargs=-1)
@click.option("--model", "model", default=None, shell_complete=_complete_model_name,
              help="Registered reranker model. Default: the only registered "
                   "reranker.")
@click.option("--file", "docs_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Read more documents from a text file, one per line.")
@click.option("--top-n", "top_n", type=click.IntRange(min=1), default=None,
              help="Show only the best N documents.")
@click.option("--json", "as_json", is_flag=True,
              help="Print the ranking as JSON (the /v1/rerank result shape).")
def rerank_cmd(query, documents, model, docs_file, top_n, as_json):
    """Rank DOCUMENTS by relevance to QUERY with a reranker model.

    Runs the model in this process (a bge-reranker or Qwen3-Reranker GGUF added
    with `localm add` or `localm pull`). The server's POST /v1/rerank takes the
    same inputs. A higher score is more relevant.

    \b
    Example:
      localm rerank "what is a cat" "a small pet" "a car engine" --model bge-reranker
    """
    from rich.markup import escape

    from ..inference import reranker
    from ..inference.backends.base import (
        PretokenizerUnsafeInputError, RerankerHeadMissingError, RerankInputError)

    texts = _read_documents(documents, docs_file)
    if not texts:
        raise click.UsageError("give at least one document, as an argument or "
                               "with --file")
    if not query.strip():
        raise click.UsageError("the query must not be empty")
    try:
        name, path = reranker.resolve_reranker(model)
        outcome = reranker.rerank(path, query, texts)
    except reranker.RerankerModelError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)
    except (RerankInputError, PretokenizerUnsafeInputError,
            RerankerHeadMissingError, RuntimeError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)
    results = reranker.rank_results(outcome.scored, top_n, outcome.labels)
    for row in results:
        row["document"] = {"text": texts[row["index"]]}
    if as_json:
        click.echo(json.dumps({"model": name, "results": results}, indent=2))
        return
    for rank, row in enumerate(results, 1):
        text = " ".join(row["document"]["text"].split())
        if len(text) > _EXCERPT_CHARS:
            text = text[:_EXCERPT_CHARS - 3] + "..."
        marker = " (document cut to fit the model window)" if row.get("truncated") else ""
        console.print(f"{rank:>3}. [bold]{row['relevance_score']:.4f}[/bold]  "
                      f"[dim]#{row['index']}[/dim]  {escape(text)}{marker}")
