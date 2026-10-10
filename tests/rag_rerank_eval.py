# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retrieval-quality harness for the Knowledge rerank stage.

Indexes the fixture corpus under ``tests/fixtures/rag_rerank_eval/corpus`` into a
throwaway ``Collection``, runs every labelled query in ``queries.json`` and
scores the returned ranking. A chunk of a query's document is relevant when its
whitespace-collapsed text contains one of the query's gold phrases.

Run it for a before/after table (the embedder and reranker are the ones the
running configuration resolves; set ``LOCALM_HOME`` to a throwaway home)::

    python -m tests.rag_rerank_eval --embed --reranker NAME --candidates 10,20,40
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rag_rerank_eval"
CORPUS_DIR = FIXTURE_DIR / "corpus"
QUERIES_FILE = FIXTURE_DIR / "queries.json"
OFFTOPIC_FILE = FIXTURE_DIR / "offtopic.json"
COLLECTION_NAME = "rerank-eval"

_WS = re.compile(r"\s+")


def load_queries() -> list[dict]:
    """The labelled queries: ``{id, doc, query, gold}`` dicts."""
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))["queries"]


def load_offtopic() -> list[dict]:
    """Questions the corpus does not answer: ``{id, category, query}`` dicts."""
    return json.loads(OFFTOPIC_FILE.read_text(encoding="utf-8"))["queries"]


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def is_relevant(hit: dict, query: dict) -> bool:
    """True when *hit* is a chunk of the query's document holding a gold phrase."""
    if Path(hit["source"]).name != query["doc"]:
        return False
    body = _collapse(hit["text"])
    return any(_collapse(g) in body for g in query["gold"])


def relevance_vector(hits: list[dict], query: dict) -> list[int]:
    """1/0 relevance of each hit, in rank order."""
    return [1 if is_relevant(h, query) else 0 for h in hits]


def hit_at(rel: list[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel: list[int]) -> float:
    for i, r in enumerate(rel, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at(rel: list[int], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG@k; *n_relevant* is how many relevant chunks exist."""
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:k], 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, n_relevant) + 1))
    return dcg / ideal if ideal else 0.0


def build_collection(base: Path, embed_fn: Optional[Callable] = None,
                     model_name: Optional[str] = None,
                     corpus_dir: Optional[Path] = None):
    """Index the markdown files of *corpus_dir* (default: the fixture corpus)
    into a fresh collection under *base*."""
    from localm.rag import Collection
    coll = Collection(COLLECTION_NAME, base=base).create()
    files = sorted((corpus_dir or CORPUS_DIR).glob("*.md"))
    result = coll.add_paths([str(p) for p in files],
                            embed_fn=embed_fn, model_name=model_name)
    if result["failed"]:
        raise RuntimeError(f"corpus files failed to index: {result['failed']}")
    return coll


def relevant_chunk_counts(coll, queries: list[dict]) -> dict[str, int]:
    """How many chunks of the collection are relevant to each query id."""
    return {q["id"]: sum(1 for c in coll._chunks if is_relevant(c, q))
            for q in queries}


def evaluate(coll, queries: list[dict], *, embed_fn: Optional[Callable] = None,
             rerank_fn: Optional[Callable] = None, k: int = 4,
             candidates: int = 20, relevant_only: bool = False) -> dict:
    """Metrics for one configuration over *queries*, with per-query detail."""
    counts = relevant_chunk_counts(coll, queries)
    rows = []
    for q in queries:
        started = time.perf_counter()
        hits = coll.query(q["query"], k=k, embed_fn=embed_fn,
                          relevant_only=relevant_only, rerank_fn=rerank_fn,
                          rerank_candidates=candidates)
        elapsed = time.perf_counter() - started
        rel = relevance_vector(hits, q)
        rows.append({
            "id": q["id"], "first_rank": next((i for i, r in enumerate(rel, 1) if r), None),
            "hit1": hit_at(rel, 1), "hitk": hit_at(rel, k),
            "mrr": reciprocal_rank(rel), "ndcg": ndcg_at(rel, k, counts[q["id"]]),
            "degrade": getattr(coll, "rerank_degrade_reason", None),
            "ms": elapsed * 1000,
        })
    n = len(rows)
    return {
        "n": n, "k": k,
        "hit@1": sum(r["hit1"] for r in rows) / n,
        f"hit@{k}": sum(r["hitk"] for r in rows) / n,
        f"mrr@{k}": sum(r["mrr"] for r in rows) / n,
        f"ndcg@{k}": sum(r["ndcg"] for r in rows) / n,
        "ms/query": sum(r["ms"] for r in rows) / n,
        "rerank_degraded": sum(1 for r in rows if r["degrade"]),
        "rows": rows,
    }


def pool_ceiling(coll, queries: list[dict], *, embed_fn: Optional[Callable] = None,
                 sizes: tuple = (4, 10, 20, 40)) -> dict[int, float]:
    """For each pool size N, the share of queries with a relevant chunk in the
    unreranked top N: the best a reranker over N candidates could reach."""
    deepest = max(sizes)
    ranks = []
    for q in queries:
        hits = coll.query(q["query"], k=deepest, embed_fn=embed_fn)
        rel = relevance_vector(hits, q)
        ranks.append(next((i for i, r in enumerate(rel, 1) if r), None))
    return {n: sum(1 for r in ranks if r is not None and r <= n) / len(ranks)
            for n in sizes}


def evaluate_gate(coll, queries: list[dict], offtopic: list[dict], *,
                  embed_fn: Optional[Callable] = None,
                  rerank_fn: Optional[Callable] = None,
                  min_score: Optional[float] = None, k: int = 4,
                  candidates: int = 20) -> dict:
    """How a ``relevant_only`` configuration treats on-topic and off-topic questions.

    ``recall`` is the share of *queries* with a relevant chunk among the returned
    hits and ``answered`` the share returning any hit. ``precision`` is the share
    of *offtopic* questions that return nothing, overall and per category."""
    def ask(text: str) -> list[dict]:
        return coll.query(text, k=k, embed_fn=embed_fn, relevant_only=True,
                          rerank_fn=rerank_fn, rerank_candidates=candidates,
                          rerank_min_score=min_score)

    on = [(q, ask(q["query"])) for q in queries]
    off = [(q, ask(q["query"])) for q in offtopic]
    categories: dict[str, list[bool]] = {}
    for q, hits in off:
        categories.setdefault(q["category"], []).append(not hits)
    return {
        "recall": sum(any(relevance_vector(h, q)) for q, h in on) / len(on),
        "answered": sum(1 for _, h in on if h) / len(on),
        "precision": sum(1 for _, h in off if not h) / len(off),
        "categories": {c: sum(v) / len(v) for c, v in sorted(categories.items())},
        "leaks": [q["id"] for q, h in off if h],
    }


def format_gate_table(results: dict[str, dict]) -> str:
    """A fixed-width comparison of named gates from :func:`evaluate_gate`."""
    cats = sorted({c for r in results.values() for c in r["categories"]})
    out = [f"{'gate':<40}{'recall':>8}{'answered':>10}{'precision':>11}"
           + "".join(f"{c:>14}" for c in cats)]
    for name, res in results.items():
        out.append(f"{name:<40}{res['recall']:>8.3f}{res['answered']:>10.3f}"
                   f"{res['precision']:>11.3f}"
                   + "".join(f"{res['categories'].get(c, float('nan')):>14.2f}"
                             for c in cats))
    return chr(10).join(out)


def format_table(results: dict[str, dict]) -> str:
    """A fixed-width comparison of named configurations."""
    any_res = next(iter(results.values()))
    k = any_res["k"]
    cols = ["hit@1", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}"]
    out = [f"{'config':<34}" + "".join(f"{c:>10}" for c in cols)
           + f"{'ms/query':>10}{'degraded':>10}"]
    for name, res in results.items():
        out.append(f"{name:<34}" + "".join(f"{res[c]:>10.3f}" for c in cols)
                   + f"{res['ms/query']:>10.1f}{res['rerank_degraded']:>10d}")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embed", action="store_true",
                    help="embed with the configured embedding model (hybrid retrieval)")
    ap.add_argument("--reranker", action="append", default=[],
                    help="registered reranker model name to compare against "
                         "(repeatable)")
    ap.add_argument("--candidates", default="20",
                    help="comma-separated candidate counts to try (default 20)")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--corpus-dir", default=None,
                    help="index this folder of .md files instead of the fixture "
                         "corpus (the labelled documents must be in it)")
    ap.add_argument("--gate", action="store_true",
                    help="compare relevant_only gates on the on-topic questions "
                         "(recall) and the off-topic ones (precision)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="reranker score a hit needs under --gate (default: the "
                         "calibrated value of each --reranker, if any)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the full per-query results to this file")
    args = ap.parse_args(argv)

    embed_fn = None
    model_name = None
    if args.embed:
        from localm.inference.embedder import embed_texts
        from localm.config import load_config

        def embed_fn(texts):
            out = embed_texts(texts)
            if out is None:
                raise RuntimeError("the embedder is unavailable")
            return out
        model_name = str(load_config().get("embedding_model") or "")

    rerankers = {}
    if args.reranker:
        from localm.rag.rerank import make_rerank_fn
        rerankers = {name: make_rerank_fn(name)[1] for name in args.reranker}
    gates: dict[str, dict] = {}

    queries = load_queries()
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="rerank-eval-") as tmp:
        coll = build_collection(Path(tmp), embed_fn, model_name,
                                Path(args.corpus_dir) if args.corpus_dir else None)
        results["baseline"] = evaluate(coll, queries, embed_fn=embed_fn, k=args.k)
        for name, rerank_fn in rerankers.items():
            for c in [int(x) for x in args.candidates.split(",") if x.strip()]:
                results[f"{name} candidates={c}"] = evaluate(
                    coll, queries, embed_fn=embed_fn, rerank_fn=rerank_fn,
                    k=args.k, candidates=c)
        ceiling = pool_ceiling(coll, queries, embed_fn=embed_fn)
        if args.gate:
            from localm.rag.rerank import calibrated_min_score
            offtopic = load_offtopic()
            gates["floor only"] = evaluate_gate(
                coll, queries, offtopic, embed_fn=embed_fn, k=args.k)
            for name, rerank_fn in rerankers.items():
                c = int(args.candidates.split(",")[0])
                min_score = (args.min_score if args.min_score is not None
                             else calibrated_min_score(name))
                gates[f"{name}: floor, then rerank"] = evaluate_gate(
                    coll, queries, offtopic, embed_fn=embed_fn,
                    rerank_fn=rerank_fn, k=args.k, candidates=c)
                if min_score is not None:
                    gates[f"{name}: score >= {min_score:g}"] = evaluate_gate(
                        coll, queries, offtopic, embed_fn=embed_fn,
                        rerank_fn=rerank_fn, min_score=min_score, k=args.k,
                        candidates=c)
    print(format_table(results))
    print("unreranked recall within top N: "
          + ", ".join(f"N={n}: {v:.3f}" for n, v in ceiling.items()))
    if gates:
        print()
        print(format_gate_table(gates))
        for name, res in gates.items():
            if res["leaks"]:
                print(f"leaks [{name}]: {', '.join(res['leaks'])}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({**results, **({"gates": gates} if gates else {})}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
