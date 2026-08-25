# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG tools: rag_list_collections and rag_search."""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult


def tool_rag_list_collections(cwd: Path) -> ToolResult:
    """List every indexed RAG collection with its stats."""
    from localm.rag.store import Collection, collection_names

    names = collection_names()
    if not names:
        return ToolResult.success(
            "No RAG collections have been indexed yet. Create one with: "
            "localm rag add <name> <path>",
            summary="0 collections",
        )
    lines = []
    for name in names:
        stats = Collection(name).stats()
        mode = "hybrid" if stats["has_vectors"] else "BM25"
        missing = f", {stats['n_missing']} missing" if stats["n_missing"] else ""
        lines.append(
            f"- {name}: {stats['n_docs']} docs, {stats['n_chunks']} chunks, "
            f"{mode}{missing}"
        )
    return ToolResult.success(
        "\n".join(lines),
        summary=f"{len(names)} collection(s)",
    )


def tool_rag_search(cwd: Path, collection: str, query: str, k: int = 4) -> ToolResult:
    """Query a RAG collection and return the top-k matching excerpts."""
    from localm.rag.store import Collection, collection_names
    from localm.textguard import neutralise

    if not query.strip():
        return ToolResult.error("query must not be empty.")

    coll = Collection(collection)
    if not coll.exists():
        available = collection_names()
        hint = (f" Available: {', '.join(available)}." if available
                else " No collections have been indexed yet.")
        return ToolResult.error(f"No RAG collection named '{collection}'.{hint}")

    hits = coll.query(query, k=max(1, min(int(k), 20)), embed_fn=None)
    if not hits:
        return ToolResult.success(
            f"No matches for '{query}' in collection '{collection}'.",
            summary=f"rag_search '{collection}': 0 hits",
        )

    lines = []
    for i, hit in enumerate(hits, 1):
        text = neutralise(str(hit.get("text", "")))
        source = hit.get("source", "?")
        pos = hit.get("pos")
        loc = f"{source}:{pos}" if pos is not None else str(source)
        lines.append(f"[{i}] {loc} (score {hit.get('score', 0):.3f})\n{text}")

    return ToolResult.success(
        "\n\n".join(lines),
        summary=f"rag_search '{collection}': {len(hits)} hit(s)",
    )
