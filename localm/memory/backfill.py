# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fill in missing memory vectors across every namespace, to completion.

WHY THIS EXISTS. ``MemoryStore.backfill_vectors`` is bounded (64 per call) and
had exactly ONE caller: the consolidation pass. Its own docstring said "a regular
background pass calls this so coverage climbs" - there was no such pass. So on an
install where auto-consolidation never ran, records written before an embedder
existed kept no vector indefinitely, `_vector_status` stayed `low_coverage`, and
recall silently fell back to lexical-only.

That was not a cosmetic gap. Below `VEC_COVERAGE` the semantic relevance gate is
unusable, which opens the trusted-fact fallback - and that fallback promotes
profile facts by IMPORTANCE, not relevance. Reported live 2026-08-14: a request
to greet a friend answered by recalling an unrelated person from days earlier,
with the log reading

    memory recall: injected 2 record(s), degrade=low_coverage

Meanwhile `setup-embeddings` told the user "Memory now retrieves semantically",
which was false for every record already stored - while correctly warning that
RAG collections stay lexical until re-embedded. Memory got the confident claim
and RAG got the honest one, which is backwards.

WHAT THIS DOES. Walks every namespace under the memory root and drives the
existing bounded backfill until nothing is left to embed, so the promise
`setup-embeddings` makes is true when it returns. Bounded per call is still the
right shape for the store (a single 20k-record save should not stall); the loop
belongs here, at the point where a caller genuinely wants completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

# A namespace can be large, and each pass re-saves the whole store. This caps the
# total work one invocation will do so a pathological store cannot hang a CLI
# command forever; the remainder is picked up by the next run. Deliberately far
# above any realistic memory store - it is a stop, not a budget.
_MAX_PASSES_PER_NS = 400


def _namespaces(root: Path):
    """Every namespace store file under *root*.

    Layout is ``<root>/<agent>/<ns_hash>.jsonl`` (see store.namespace_file). The
    sidecars that live beside a store - the episodic watermark, the forgotten
    archive, the pending corrections - are JSON, not JSONL, so matching the
    store extension excludes them by construction rather than by an
    ever-growing deny list.
    """
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.jsonl"))


def backfill_all(root: Path, embed_fn: Optional[Callable], *,
                 on_progress: Optional[Callable] = None) -> dict:
    """Embed every vectorless record in every namespace under *root*.

    Returns ``{"namespaces": n, "embedded": n, "remaining": n, "unreadable": n}``.
    ``remaining`` is non-zero only when a namespace hit ``_MAX_PASSES_PER_NS`` or
    an embed call kept failing - it is reported rather than hidden, so a caller
    can say so instead of claiming completion it did not reach (AGENTS.md rule
    5). ``unreadable`` counts namespaces that could not even be opened (corrupt
    JSONL, a locked file) - kept separate from ``remaining`` because such a
    namespace's true vectorless count is unknown, not zero, so folding it into
    "remaining" (or dropping it) would read as a clean pass.
    """
    if embed_fn is None:
        return {"namespaces": 0, "embedded": 0, "remaining": 0, "unreadable": 0}

    from localm.memory.store import MemoryStore

    embedded = 0
    remaining = 0
    unreadable = 0
    seen = 0
    for path in _namespaces(root):
        try:
            store = MemoryStore.open_file(path)
        except Exception as e:
            unreadable += 1
            from localm.debuglog import logger as _dbg
            _dbg.warning("memory backfill: could not open namespace %s (%s)",
                         path, e)
            continue
        seen += 1
        for _ in range(_MAX_PASSES_PER_NS):
            try:
                filled = store.backfill_vectors(embed_fn)
            except Exception:
                # One namespace failing must not abandon the rest; the shortfall
                # shows up in "remaining" rather than as a silent success.
                break
            if not filled:
                break
            embedded += filled
            if on_progress:
                on_progress(embedded)
        remaining += store.vectorless_count()
    return {"namespaces": seen, "embedded": embedded, "remaining": remaining,
            "unreadable": unreadable}


def vectorless_scan(root: Path) -> tuple[int, int]:
    """Count vectorless records across every namespace under *root*, and count
    namespaces that could not even be opened (corrupt JSONL, a locked or
    unreadable file). Returns ``(total, unreadable)``.

    An unreadable namespace previously contributed 0 to ``total`` - exactly
    what a fully embedded namespace also contributes - so a root where every
    vectorless namespace happened to be unreadable looked identical to one
    with nothing left to do. Reporting the two counts separately makes that
    shortfall visible to callers instead of silently absorbing it.
    """
    from localm.memory.store import MemoryStore

    total = 0
    unreadable = 0
    for path in _namespaces(root):
        try:
            total += MemoryStore.open_file(path).vectorless_count()
        except Exception as e:
            unreadable += 1
            from localm.debuglog import logger as _dbg
            _dbg.warning("memory backfill: could not open namespace %s (%s)",
                         path, e)
    return total, unreadable


def vectorless_total(root: Path) -> int:
    """How many records across every namespace still lack a vector. Used to
    decide whether a backfill is worth announcing, and to report honestly."""
    total, _unreadable = vectorless_scan(root)
    return total
